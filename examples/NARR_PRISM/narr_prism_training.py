"""Training utilities for the NARR-to-PRISM downscaling model.

Closely mirrors the CORDEX_ML training logic but loads data through
the NARR_PRISM dataset and uses the NARR_PRISM YAML configuration.

Usage (via narr_prism_finetune.py):
    python narr_prism_finetune.py --config NARR_PRISM_subdomain.yaml
"""

from __future__ import annotations

import math
import os
import socket
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler as CudaGradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from narr_prism_dataset import NarrPrismDataset

try:
    from granitewxc.models.loss import build_loss_fn, rmse_loss
except ModuleNotFoundError:
    def rmse_loss(y_hat: torch.Tensor, y: dict[str, torch.Tensor]) -> torch.Tensor:
        target = y["y"]
        valid = torch.isfinite(target)
        if not bool(valid.any()):
            return y_hat.sum() * 0.0
        diff = (y_hat - torch.where(valid, target, y_hat.detach())) ** 2
        valid_f = valid.to(y_hat.dtype)
        return torch.sqrt((diff * valid_f).sum() / valid_f.sum().clamp(min=1.0) + 1e-12)

    def build_loss_fn(config, output_vars):
        del config, output_vars
        return rmse_loss

from granitewxc.models.model import get_finetune_model_UNET
from granitewxc.utils.config import ExperimentConfig
from granitewxc.utils.normalization import (
    apply_scalar_paths,
    assert_scalars_available,
    log_case_context,
    log_scalar_summary,
)
from granitewxc.utils.predictands import build_predictand_specs
from granitewxc.utils.distributed import init_ddp
from granitewxc.utils.trainer import train_model
from narr_prism_utils import case_output_dir, get_case_name

try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import (
        size_based_auto_wrap_policy,
        transformer_auto_wrap_policy,
    )
    _FSDP_AVAILABLE = True
except ImportError:
    _FSDP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_path(raw: str) -> str:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return str(p)
    return str((REPO_ROOT / p).resolve())


def _should_use_gpu(config: ExperimentConfig) -> bool:
    target = getattr(config, "device_target", None)
    if target:
        norm = str(target).lower()
        if norm == "cpu":
            return False
        if norm in {"cuda", "gpu"}:
            if not torch.cuda.is_available():
                raise RuntimeError("device_target=cuda but no CUDA device")
            return True
    return torch.cuda.is_available()


def _pick_free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return str(s.getsockname()[1])


def _fsdp_enabled(config: ExperimentConfig) -> bool:
    strategy = getattr(config, "distributed_strategy", None)
    if isinstance(strategy, str) and strategy.lower() == "fsdp":
        return True
    return False


def _build_fsdp_precision(config: ExperimentConfig):
    if not _FSDP_AVAILABLE:
        return None
    precision = getattr(config, "fsdp_precision", None)
    if not precision:
        precision = "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None
    if not precision:
        return None
    precision = precision.lower()
    if precision in {"bf16", "bfloat16"} and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif precision in {"fp16", "float16"}:
        dtype = torch.float16
    else:
        return None
    return MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)


def _wrap_fsdp(model: torch.nn.Module, local_rank: int, config: ExperimentConfig):
    min_params = getattr(config, "fsdp_min_num_params", None)
    policy = None
    attention_scope = str(
        getattr(getattr(config, "model", None), "backbone_attention_scope", "")
    ).strip().lower()
    if attention_scope == "windowed_local":
        # ``ClimateDownscaleFinetuneUNETModel._windowed_local_backbone`` calls
        # the Prithvi transformer blocks directly instead of entering through
        # LocalGlobalLocalBlock.forward().  A size-based policy can leave the
        # small LayerNorm parameters owned by an enclosing FSDP wrapper while
        # wrapping only the large MLP child.  The enclosing wrapper is then
        # bypassed, so non-owner ranks see zero-length LayerNorm parameters.
        # Make every directly called transformer an atomic FSDP unit instead.
        transformer_types = {
            type(module)
            for module in model.modules()
            if module.__class__.__name__ == "Transformer"
            and module.__class__.__module__.startswith("PrithviWxC")
        }
        if not transformer_types:
            raise RuntimeError(
                "windowed_local FSDP training requires Prithvi transformer blocks"
            )
        policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_types,
        )
    elif min_params:
        policy = partial(size_based_auto_wrap_policy, min_num_params=int(min_params))
    mp_ = _build_fsdp_precision(config)
    return FSDP(
        model,
        device_id=local_rank,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_,
        auto_wrap_policy=policy,
        limit_all_gathers=True,
        use_orig_params=True,
    )


# ---------------------------------------------------------------------------
# Dataset / DataLoader
# ---------------------------------------------------------------------------

class _WrappedDataset(torch.utils.data.Dataset):
    """Adapt NarrPrismDataset output dict to the model's expected format.

    Pads the input ``x`` spatial dimensions to the nearest multiple of
    ``mask_unit_size × patch_size`` (default 32) so the Prithvi backbone
    assertion ``n_lats % mask_unit_size_px == 0`` is satisfied.
    The model auto-interpolates its output to ``batch['y'].shape[-2:]``,
    so no padding is needed on ``y``.

    If the base dataset has a static elevation channel appended to ``x``,
    this wrapper can split it out into ``static_x`` / ``static_y`` when
    ``num_static_channels > 0``.
    """

    def __init__(
        self,
        base: NarrPrismDataset,
        num_static_channels: int = 0,
        pad_multiple: int = 32,
        phase1_cache_reader: Any | None = None,
        cache_split: str | None = None,
    ) -> None:
        self.base = base
        self.num_static_channels = num_static_channels
        self.pad_multiple = pad_multiple
        self.phase1_cache_reader = phase1_cache_reader
        self.cache_split = cache_split
        if self.phase1_cache_reader is not None and not self.cache_split:
            raise ValueError("cache_split is required with a Phase-1 cache reader")
        self._static_y: Optional[torch.Tensor] = None
        if num_static_channels > 0 and base._elevation is not None:
            elev = torch.from_numpy(base._elevation.astype("float32"))
            self._static_y = elev.unsqueeze(0)  # (1, H_target, W_target)

    def __len__(self) -> int:
        return len(self.base)

    def _pad_to_multiple(self, x: torch.Tensor) -> torch.Tensor:
        """Pad spatial dims (H, W) to the nearest multiple of pad_multiple."""
        m = self.pad_multiple
        _, H, W = x.shape[-3], x.shape[-2], x.shape[-1]
        pad_h = (m - H % m) % m
        pad_w = (m - W % m) % m
        if pad_h == 0 and pad_w == 0:
            return x
        # F.pad pads from last dim inward: (left, right, top, bottom)
        return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    def _attach_phase1_cache(
        self, wrapped: Dict[str, Any], source: Dict[str, Any]
    ) -> Dict[str, Any]:
        if self.phase1_cache_reader is None:
            return wrapped
        if "date" not in source:
            raise KeyError("Cached Phase-2 training requires a sample date")
        offset = source.get("__output_scaler_offset", source.get("__scaler_offset"))
        if offset is None or not torch.is_tensor(offset) or offset.numel() != 2:
            raise ValueError(
                "Cached Phase-2 training requires a two-value output scaler offset"
            )
        y0, x0 = (int(value) for value in offset.reshape(-1).tolist())
        height, width = (int(value) for value in source["y"].shape[-2:])
        cached = self.phase1_cache_reader.load_crop(
            self.cache_split,
            source["date"],
            y0,
            x0,
            height,
            width,
        )
        required = (
            "__phase1_normalized",
            "__residual_target_normalized",
            "__residual_valid_mask",
        )
        missing = [key for key in required if key not in cached]
        if missing:
            raise RuntimeError(
                f"Phase-1 residual cache reader did not return required keys: {missing}"
            )
        expected_shape = tuple(source["y"].shape)
        for key in required:
            value = cached[key]
            if not torch.is_tensor(value) or tuple(value.shape) != expected_shape:
                observed = tuple(value.shape) if torch.is_tensor(value) else None
                raise ValueError(
                    f"Cached field {key} has shape {observed}; expected {expected_shape}"
                )
        wrapped.update(cached)
        return wrapped

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.base[idx]
        x_full = sample["x"]  # (C_dyn [+ num_static], H_in, W_in)
        tile_metadata = {
            key: sample[key]
            for key in (
                "date",
                "tile_index",
                "__scaler_offset",
                "__input_scaler_offset",
                "__output_scaler_offset",
                "__output_crop",
            )
            if key in sample
        }

        if self.num_static_channels > 0 and self._static_y is not None:
            x_dyn = x_full[: -self.num_static_channels]
            static_x = x_full[-self.num_static_channels:]
            wrapped = {
                "x": self._pad_to_multiple(x_dyn),
                "y": sample["y"],
                "static_x": self._pad_to_multiple(static_x),
                "static_y": self._static_y,
            }
            wrapped.update(tile_metadata)
            return self._attach_phase1_cache(wrapped, sample)

        wrapped = {
            "x": self._pad_to_multiple(x_full),
            "y": sample["y"],
        }
        wrapped.update(tile_metadata)
        return self._attach_phase1_cache(wrapped, sample)


class _DateGroupedTileSampler(torch.utils.data.Sampler[int]):
    """Shuffle dates reproducibly while keeping every date's tiles together.

    Daily cache arrays can then remain in a tiny per-worker LRU for all tiles
    of one date instead of reopening a 1024 x 1024 NetCDF after a global random
    shuffle. Tile order is independently shuffled within each date.
    """

    def __init__(
        self,
        *,
        num_dates: int,
        tiles_per_date: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.num_dates = int(num_dates)
        self.tiles_per_date = int(tiles_per_date)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0
        if self.num_dates < 1 or self.tiles_per_date < 1:
            raise ValueError("Date-grouped sampling requires dates and tiles")
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError("Invalid date-grouped sampler rank/world size")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        dates = torch.randperm(self.num_dates, generator=generator).tolist()
        # Match DistributedSampler's equal-length contract so synchronized
        # ranks cannot hang when the number of dates is not divisible by the
        # world size. Padding duplicates complete date groups, never individual
        # tiles, which preserves the daily-file locality guarantee.
        dates_per_rank = math.ceil(self.num_dates / self.world_size)
        total_dates = dates_per_rank * self.world_size
        if total_dates > len(dates):
            padding = total_dates - len(dates)
            dates.extend((dates * math.ceil(padding / len(dates)))[:padding])
        dates = dates[self.rank : total_dates : self.world_size]
        for date_index in dates:
            tiles = torch.randperm(
                self.tiles_per_date, generator=generator
            ).tolist()
            base_index = date_index * self.tiles_per_date
            for tile_index in tiles:
                yield base_index + tile_index

    def __len__(self) -> int:
        return math.ceil(self.num_dates / self.world_size) * self.tiles_per_date


def _build_dataloader(
    config_path: str,
    config: ExperimentConfig,
    mode: str,
    *,
    shuffle: bool,
    distributed: bool,
    rank: int,
    world_size: int,
    phase1_cache_reader: Any | None = None,
) -> DataLoader:
    base = NarrPrismDataset(
        config_path,
        mode=mode,
        cache_predictor_days=phase1_cache_reader is not None,
        cache_target_days=phase1_cache_reader is not None,
    )
    num_static = int(getattr(getattr(config, "model", object()), "num_static_channels", 0))
    # Compute required padding multiple: mask_unit_size × patch_size
    mask_unit = getattr(config, "mask_unit_size", [16, 16])
    patch_sz = getattr(getattr(config, "model", object()), "downscaling_patch_size", [2, 2])
    pad_multiple = (mask_unit[0] if isinstance(mask_unit, list) else mask_unit) * (
        patch_sz[0] if isinstance(patch_sz, list) else patch_sz
    )
    dataset = _WrappedDataset(
        base,
        num_static_channels=num_static,
        pad_multiple=pad_multiple,
        phase1_cache_reader=phase1_cache_reader,
        cache_split=mode if phase1_cache_reader is not None else None,
    )

    sampler = None
    if phase1_cache_reader is not None and shuffle:
        tile_slices = getattr(base, "_tile_slices", None)
        if not tile_slices:
            raise ValueError(
                "Phase-1 cache training requires deterministic tiled spatial sampling"
            )
        refinement_cfg = getattr(getattr(config, "model", object()), "refinement", {})
        seed = (
            refinement_cfg.get("seed", 1234)
            if isinstance(refinement_cfg, dict)
            else getattr(refinement_cfg, "seed", 1234)
        )
        sampler = _DateGroupedTileSampler(
            num_dates=len(base.dates),
            tiles_per_date=len(tile_slices),
            seed=int(seed),
            rank=rank,
            world_size=world_size,
        )
        shuffle = False
    elif distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        shuffle = False

    batch_size = int(getattr(config, "batch_size", 1))
    num_workers = int(getattr(config, "dl_num_workers", 0))
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
    )


def get_dataloaders(
    config_path: str,
    config: ExperimentConfig,
    rank: int = 0,
    world_size: int = 1,
    phase1_cache_reader: Any | None = None,
) -> Tuple[DataLoader, DataLoader]:
    distributed = world_size > 1
    dates_cfg = getattr(config, "dates", {}) or {}
    validation_cfg = (
        dates_cfg.get("validation", {})
        if isinstance(dates_cfg, dict)
        else getattr(dates_cfg, "validation", {})
    ) or {}
    validation_mode = (
        "validation"
        if (
            (validation_cfg.get("start") if isinstance(validation_cfg, dict) else getattr(validation_cfg, "start", None))
            and (validation_cfg.get("end") if isinstance(validation_cfg, dict) else getattr(validation_cfg, "end", None))
        )
        else "training"
    )
    if validation_mode == "training":
        print(
            "[training] WARNING: dates.validation is absent; using the legacy "
            "in-sample validation fallback. Add a held-out validation split "
            "before comparing model checkpoints."
        )
    train_loader = _build_dataloader(
        config_path, config, "training",
        shuffle=True, distributed=distributed, rank=rank, world_size=world_size,
        phase1_cache_reader=phase1_cache_reader,
    )
    val_loader = _build_dataloader(
        config_path, config, validation_mode,
        shuffle=False, distributed=distributed, rank=rank, world_size=world_size,
        phase1_cache_reader=phase1_cache_reader,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------

def load_pretrained_weights(model: torch.nn.Module, weights_path: str) -> Tuple[int, int]:
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        weights = checkpoint["model"]
    else:
        weights = checkpoint
    if not isinstance(weights, dict):
        weights = weights.state_dict() if hasattr(weights, "state_dict") else dict(weights)

    model_state = model.state_dict()
    scaler_parts = ("input_scalers_", "output_scalers_", "static_input_scalers_", "static_output_scalers_")
    compatible = {}
    skipped = 0
    for key, value in weights.items():
        if any(part in key for part in scaler_parts):
            skipped += 1
            continue
        target = model_state.get(key)
        if target is None or target.shape != value.shape:
            skipped += 1
            continue
        compatible[key] = value

    model_state.update(compatible)
    model.load_state_dict(model_state, strict=False)
    return len(compatible), skipped


def create_finetune_model(config: ExperimentConfig, verbose: bool = True) -> torch.nn.Module:
    # Safety guards for fields that must exist before calling get_finetune_model_UNET.
    # These are normally provided by NARR_PRISM.yaml but we guard here for robustness.
    for attr in ("input_static_surface_vars", "input_surface_vars", "other",
                 "vertical_level1_vars", "input_level1",
                 "vertical_level2_vars", "input_level2"):
        if not hasattr(config.data, attr):
            setattr(config.data, attr, [])

    output_vars = list(getattr(config.data, "output_vars",
                               getattr(config.data, "target_variables", [])))
    if not output_vars:
        output_vars = list(getattr(config.data, "target_variables", []))
    if not hasattr(config.data, "output_vars") or not config.data.output_vars:
        config.data.output_vars = output_vars

    if not hasattr(config.data, "input_vars") or not config.data.input_vars:
        pred_vars = getattr(config.data, "predictor_variables", {})
        config.data.input_vars = list(pred_vars.keys()) if isinstance(pred_vars, dict) else []
    if not hasattr(config.data, "input_levels") or not config.data.input_levels:
        pred_vars = getattr(config.data, "predictor_variables", {})
        all_levels: set = set()
        if isinstance(pred_vars, dict):
            for levels in pred_vars.values():
                if isinstance(levels, list):
                    all_levels.update(levels)
        config.data.input_levels = sorted(all_levels) if all_levels else [0]
    if not hasattr(config.data, "vertical_pres_vars") or not config.data.vertical_pres_vars:
        config.data.vertical_pres_vars = list(config.data.input_vars)
    if not hasattr(config.data, "input_level_pres") or not config.data.input_level_pres:
        config.data.input_level_pres = list(config.data.input_levels)
    predictand_specs = build_predictand_specs(config, output_vars=output_vars)
    if verbose:
        print("[model] predictand configuration:")
        for spec in predictand_specs:
            print(f"  - {spec.name}: scaling={spec.scaling.method}")

    model = get_finetune_model_UNET(config)
    weights_path = _resolve_path(str(config.path_model_weights))
    loaded, skipped = load_pretrained_weights(model, weights_path)
    if verbose:
        print(f"[model] loaded {loaded} tensors, skipped {skipped}")
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _build_grad_scaler(enabled: bool):
    amp_mod = getattr(torch, "amp", None)
    if amp_mod:
        cls = getattr(amp_mod, "GradScaler", None)
        if cls:
            try:
                return cls("cuda", enabled=enabled)
            except TypeError:
                return cls(enabled=enabled)
    return CudaGradScaler(enabled=enabled)


def build_optimizer_scheduler(
    config: ExperimentConfig, model: torch.nn.Module, train_steps: int, use_gpu: bool
):
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scaler = _build_grad_scaler(enabled=use_gpu and torch.cuda.is_available())
    accum = max(1, int(getattr(config, "gradient_accumulation_steps", 1)))
    steps_per_epoch = max(1, min(train_steps, config.limit_steps_train))
    opt_steps = (steps_per_epoch + accum - 1) // accum
    total = config.num_epochs * max(1, opt_steps)
    scheduler = CosineAnnealingLR(optimizer, T_max=total, eta_min=config.min_lr)
    return optimizer, scaler, scheduler


def _single_gpu_worker(
    rank: int,
    config: ExperimentConfig,
    config_path: str,
    save_every: int,
    return_dict: Dict,
) -> None:
    if rank != 0:
        return
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    train_loader, val_loader = get_dataloaders(config_path, config, rank=0, world_size=1)
    model = create_finetune_model(config).to(device)
    optimizer, scaler, scheduler = build_optimizer_scheduler(
        config, model, len(train_loader), use_gpu=True
    )
    output_vars = list(getattr(config.data, "output_vars", getattr(config.data, "target_variables", [])))
    loss_fn = build_loss_fn(config, output_vars)

    train_losses, val_losses = train_model(
        config, model, train_loader, val_loader,
        optimizer, scheduler, scaler, 0, True, save_every, loss_fn,
    )
    return_dict["train_losses"] = train_losses
    return_dict["val_losses"] = val_losses


def _distributed_worker(
    rank: int,
    world_size: int,
    config: ExperimentConfig,
    config_path: str,
    save_every: int,
    return_dict: Dict,
) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    local_rank, global_rank = init_ddp(use_gpu=True)

    try:
        device = torch.device(f"cuda:{local_rank}")
        train_loader, val_loader = get_dataloaders(
            config_path, config, rank=global_rank, world_size=world_size,
        )
        model = create_finetune_model(config, verbose=(global_rank == 0)).to(device)
        if _fsdp_enabled(config) and _FSDP_AVAILABLE:
            model = _wrap_fsdp(model, local_rank, config)
        else:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank,
            )
        optimizer, scaler, scheduler = build_optimizer_scheduler(
            config, model, len(train_loader), use_gpu=True,
        )
        output_vars = list(getattr(config.data, "output_vars", getattr(config.data, "target_variables", [])))
        loss_fn = build_loss_fn(config, output_vars)
        train_losses, val_losses = train_model(
            config, model, train_loader, val_loader,
            optimizer, scheduler, scaler, local_rank, True, save_every, loss_fn,
        )
        if global_rank == 0:
            return_dict["train_losses"] = train_losses
            return_dict["val_losses"] = val_losses
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def run_training(
    config: ExperimentConfig,
    config_path: str,
    num_gpus: Optional[int] = None,
    save_every: int = 5,
) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """Launch training (single- or multi-GPU) and return (train_losses, val_losses)."""
    case_name = get_case_name(config)
    checkpoint_root = getattr(config, "checkpoint_dir", None)
    if checkpoint_root:
        config.checkpoint_dir = str(case_output_dir(checkpoint_root, case_name))
    elif getattr(config, "path_experiment", None):
        config.checkpoint_dir = str(
            case_output_dir(Path(config.path_experiment) / "weights", case_name)
        )
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    print(f"[training] case_name={case_name}")
    print(f"[training] checkpoint_dir={config.checkpoint_dir}")

    # Wire the model + dataset to the per-case, per-channel scalers and fail
    # fast if they are missing. Centralized here so BOTH the CLI entry point and
    # the notebook (which calls run_training directly) consume the SAME
    # case-scoped scalers instead of the YAML's default (flat) scaler paths.
    log_case_context(config, "finetune")
    assert_scalars_available(config, role="finetune")
    apply_scalar_paths(config)
    log_scalar_summary(config, "training")

    use_gpu = _should_use_gpu(config)

    if not use_gpu:
        print("[training] running on CPU")
        train_loader, val_loader = get_dataloaders(config_path, config)
        model = create_finetune_model(config)
        optimizer, scaler, scheduler = build_optimizer_scheduler(
            config, model, len(train_loader), use_gpu=False,
        )
        output_vars = list(getattr(config.data, "output_vars", getattr(config.data, "target_variables", [])))
        loss_fn = build_loss_fn(config, output_vars)
        return train_model(
            config, model, train_loader, val_loader,
            optimizer, scheduler, scaler, 0, False, save_every, loss_fn,
        )

    available_gpus = torch.cuda.device_count()
    if num_gpus is None:
        training_cfg = getattr(config, "training", None)
        if training_cfg and hasattr(training_cfg, "num_gpus"):
            num_gpus = int(training_cfg.num_gpus)
        else:
            num_gpus = 1
    num_gpus = min(num_gpus, available_gpus)

    if num_gpus <= 1:
        print("[training] single-GPU mode")
        # Default to GPU 0, but honor an explicit external CUDA_VISIBLE_DEVICES
        # (e.g. to target a less-contended GPU on a shared machine).
        _cvd = os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        print(f"[training] CUDA_VISIBLE_DEVICES={_cvd}")
        # Run in-process for single-GPU so notebook outputs keep one live tqdm
        # bar per epoch (mp.spawn child-process stdout tends to fragment bars).
        return_dict: Dict[str, Any] = {}
        _single_gpu_worker(0, config, config_path, save_every, return_dict)
    else:
        manager = mp.Manager()
        return_dict = manager.dict()
        print(f"[training] distributed mode ({num_gpus} GPUs)")
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = _pick_free_port()
        mp.spawn(
            _distributed_worker,
            args=(num_gpus, config, config_path, save_every, return_dict),
            nprocs=num_gpus,
        )

    return return_dict.get("train_losses"), return_dict.get("val_losses")
