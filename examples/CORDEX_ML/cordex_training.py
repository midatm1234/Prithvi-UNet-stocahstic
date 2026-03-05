"""Utilities to train the CORDEX finetuning model in single- or multi-GPU mode."""

from __future__ import annotations

import os
import sys
import subprocess
import socket
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from functools import partial
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from cordex_dataset import CordexDownscaleDataset

# Notebook workflows execute from examples/CORDEX_ML, so ensure local package
# imports resolve to this repository instead of an older site-packages install.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from granitewxc.models.loss import rmse_loss
except ModuleNotFoundError:
    # Fallback for environments where the installed granitewxc package
    # does not expose granitewxc.models.loss.
    def rmse_loss(y_hat: torch.Tensor, y: dict[torch.Tensor]) -> torch.Tensor:
        return torch.sqrt(torch.mean((y_hat - y["y"]) ** 2))
from granitewxc.models.model import get_finetune_model_UNET
from granitewxc.utils.config import ExperimentConfig
from granitewxc.utils.predictands import build_predictand_specs
from granitewxc.utils.distributed import init_ddp
from granitewxc.utils.trainer import train_model
from torch.multiprocessing.spawn import ProcessRaisedException, ProcessExitedException


def _ensure_expandable_cuda_segments() -> None:
    """Request expandable CUDA allocator segments to reduce fragmentation OOMs."""
    desired = "expandable_segments:True"
    current = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").strip()
    if not current:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = desired
        return

    entries = [item.strip() for item in current.split(",") if item.strip()]
    keys = {item.split(":", 1)[0].strip().lower() for item in entries if ":" in item}
    if "expandable_segments" not in keys:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"{current},{desired}"


def _should_use_gpu(config: ExperimentConfig) -> bool:
    target = getattr(config, "device_target", None)
    if target:
        normalized = str(target).lower()
        if normalized == "cpu":
            return False
        if normalized in {"cuda", "gpu"}:
            if not torch.cuda.is_available():
                raise RuntimeError("device_target set to 'cuda' but no CUDA device is available.")
            return True
    return torch.cuda.is_available()


def _query_gpu_stats() -> list[dict[str, int]]:
    """Return GPU stats from nvidia-smi (indices are original physical IDs)."""
    query = "index,memory.total,memory.used,utilization.gpu"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            text=True,
        )
    except Exception:
        return []

    stats: list[dict[str, int]] = []
    for raw in out.strip().splitlines():
        if not raw.strip():
            continue
        idx_str, total_str, used_str, util_str = [part.strip() for part in raw.split(",")]
        util_digits = "".join(ch for ch in util_str if ch.isdigit())
        total_mb = int(total_str)
        used_mb = int(used_str)
        util = int(util_digits) if util_digits else 100
        stats.append(
            {
                "index": int(idx_str),
                "total_mb": total_mb,
                "used_mb": used_mb,
                "free_mb": max(total_mb - used_mb, 0),
                "util": util,
            }
        )
    return stats


def _single_gpu_retry_order() -> list[int]:
    """
    Return physical GPU IDs in retry order: current visible GPU first, then most idle.
    """
    stats = _query_gpu_stats()
    if not stats:
        return [0]

    current_ids = []
    current_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if current_visible:
        current_ids = [int(part.strip()) for part in current_visible.split(",") if part.strip()]
    current_id = current_ids[0] if current_ids else stats[0]["index"]

    ranked = sorted(stats, key=lambda g: (-g["free_mb"], g["util"], g["used_mb"], g["index"]))
    others = [g["index"] for g in ranked if g["index"] != current_id]
    return [current_id, *others]


def _pick_free_master_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _is_addr_in_use_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "eaddrinuse" in text or "address already in use" in text


def _resolve_paths(paths: Iterable[str]) -> list[str]:
    resolved: list[str] = []
    for path in paths:
        path_obj = Path(path)
        if path_obj.is_absolute():
            resolved.append(str(path_obj))
        else:
            resolved.append(str((REPO_ROOT / path_obj).resolve()))
    return resolved


def _resolve_path(path: str) -> str:
    return _resolve_paths([path])[0]


def _level_suffix(level) -> str:
    value = str(level)
    return value[:-2] if value.endswith(".0") else value


def build_predictor_names(config: ExperimentConfig) -> list[str]:
    return [
        f"{var}_{_level_suffix(level)}"
        for var in config.data.input_vars
        for level in config.data.input_levels
    ]


def _build_base_dataset(
    config: ExperimentConfig,
    predictor_paths: Sequence[str],
    target_paths: Sequence[str],
    *,
    crop_size: Tuple[int, int],
    random_crop: bool,
):
    predictor_vars = build_predictor_names(config)
    target_variables = list(config.data.output_vars)
    use_static = bool(getattr(config.data, "use_static", getattr(config, "finetune_w_static", True)))
    orography_path = None
    if use_static:
        orography_path = _resolve_path(config.data.static_path)
    return CordexDownscaleDataset(
        predictor_files=predictor_paths,
        target_files=target_paths,
        orography_file=orography_path,
        predictor_variables=predictor_vars,
        target_variables=target_variables,
        crop_size=crop_size,
        random_crop=random_crop,
        use_static=use_static,
    )


def _fsdp_enabled(config: ExperimentConfig) -> bool:
    strategy = getattr(config, "distributed_strategy", None)
    if isinstance(strategy, str) and strategy.lower() == "fsdp":
        return True
    return bool(getattr(config, "use_fsdp", False))


def _build_fsdp_precision(config: ExperimentConfig) -> MixedPrecision | None:
    precision = getattr(config, "fsdp_precision", None)
    if not precision:
        precision = "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None

    if not precision:
        return None

    precision = precision.lower()
    if precision in {"bf16", "bfloat16"} and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif precision in {"fp16", "float16", "half"}:
        dtype = torch.float16
    else:
        return None

    return MixedPrecision(
        param_dtype=dtype,
        reduce_dtype=dtype,
        buffer_dtype=dtype,
    )


def _wrap_model_with_fsdp(
    model: torch.nn.Module, local_rank: int, config: ExperimentConfig
) -> torch.nn.Module:
    min_params = getattr(config, "fsdp_min_num_params", None)
    auto_wrap_policy = None
    if min_params:
        auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=int(min_params))

    sharding = getattr(config, "fsdp_sharding_strategy", "full_shard")
    sharding = sharding.lower() if isinstance(sharding, str) else "full_shard"
    if sharding == "shard_grad_op":
        strategy = ShardingStrategy.SHARD_GRAD_OP
    else:
        strategy = ShardingStrategy.FULL_SHARD

    mixed_precision = _build_fsdp_precision(config)

    return FSDP(
        model,
        device_id=local_rank,
        sharding_strategy=strategy,
        mixed_precision=mixed_precision,
        auto_wrap_policy=auto_wrap_policy,
        limit_all_gathers=True,
        use_orig_params=True,
    )


class CordexWrappedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset: CordexDownscaleDataset):
        self.base = base_dataset

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        sample = self.base[idx]
        x = sample["x"]
        if not getattr(self.base, "use_static", True):
            return {"x": x, "y": sample["y"]}
        dynamic = x[:-1]
        static = x[-1:].clone()
        return {"x": dynamic, "y": sample["y"], "static_x": static, "static_y": static}


def build_dataloader(
    config: ExperimentConfig,
    predictor_paths: Sequence[str],
    target_paths: Sequence[str],
    *,
    shuffle: bool,
    use_gpu: bool,
    distributed: bool,
    rank: int,
    world_size: int,
    crop_size: Tuple[int, int],
    random_crop: bool,
) -> DataLoader:
    predictor_paths = _resolve_paths(predictor_paths)
    target_paths = _resolve_paths(target_paths)
    base_dataset = _build_base_dataset(
        config,
        predictor_paths,
        target_paths,
        crop_size=crop_size,
        random_crop=random_crop,
    )
    dataset = CordexWrappedDataset(base_dataset)

    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        shuffle = False

    num_workers = int(getattr(config, "dl_num_workers", 0))
    # Pinning can trigger CUDA allocator pressure for large CORDEX tensors when
    # using worker processes. Keep it opt-in for multi-worker loaders.
    default_pin_memory = bool(use_gpu and num_workers == 0)
    pin_memory = bool(
        getattr(
            config,
            "dl_pin_memory",
            getattr(config, "pin_memory", default_pin_memory),
        )
    )

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        prefetch_factor = int(getattr(config, "dl_prefetch_size", 0) or 0)
        if prefetch_factor > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(**loader_kwargs)


def get_dataloaders(
    config: ExperimentConfig, use_gpu: bool, rank: int = 0, world_size: int = 1
) -> Tuple[DataLoader, DataLoader]:
    distributed = world_size > 1
    crop_size = (
        int(config.data.target_size_lat),
        int(config.data.target_size_lon),
    )
    train_loader = build_dataloader(
        config,
        config.data.training_predictor_paths,
        config.data.training_target_paths,
        shuffle=True,
        use_gpu=use_gpu,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        crop_size=crop_size,
        random_crop=True,
    )
    val_loader = build_dataloader(
        config,
        config.data.validation_predictor_paths,
        config.data.validation_target_paths,
        shuffle=False,
        use_gpu=use_gpu,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        crop_size=crop_size,
        random_crop=False,
    )
    return train_loader, val_loader


def load_pretrained_weights(model: torch.nn.Module, weights_path: str) -> Tuple[int, int]:
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        weights = checkpoint["model"]
    else:
        weights = checkpoint

    if not isinstance(weights, dict):
        weights = weights.state_dict() if hasattr(weights, "state_dict") else dict(weights)

    model_state = model.state_dict()
    weights_have_module_prefix = all(key.startswith("module.") for key in weights.keys())
    model_expects_module_prefix = all(key.startswith("module.") for key in model_state.keys())

    if model_expects_module_prefix and not weights_have_module_prefix:
        weights = weights.__class__((f"module.{key}", value) for key, value in weights.items())
    elif weights_have_module_prefix and not model_expects_module_prefix:
        prefix_len = len("module.")
        weights = weights.__class__((key[prefix_len:], value) for key, value in weights.items())

    compatible = {}
    skipped = 0
    scaler_key_parts = (
        "input_scalers_",
        "output_scalers_",
        "static_input_scalers_",
        "static_output_scalers_",
    )
    for key, value in weights.items():
        # Keep run-specific normalization tensors from the current config/scaler files.
        if any(part in key for part in scaler_key_parts):
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
    if not hasattr(config.data, "input_static_surface_vars"):
        config.data.input_static_surface_vars = []
    predictand_specs = build_predictand_specs(
        config, output_vars=list(getattr(config.data, "output_vars", []))
    )
    if verbose:
        print("[predictands] resolved training output configuration:")
        for spec in predictand_specs:
            print(
                f"  - {spec.name}: allow_negative_value={spec.allow_negative_value}, "
                f"nonnegativity=({spec.nonnegativity.enabled}, {spec.nonnegativity.method}), "
                f"scaling=({spec.scaling.method}, {spec.scaling.scale_stat})"
            )
    target = str(getattr(config, "device_target", "") or "").lower()
    if target != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = get_finetune_model_UNET(config)
    loaded, skipped = load_pretrained_weights(model, _resolve_path(config.path_model_weights))
    if verbose:
        print(
            f"Loaded {loaded} tensors from {config.path_model_weights}. "
            f"Skipped {skipped} mismatched entries."
        )
    return model


def build_optimizer_scheduler(
    config: ExperimentConfig, model: torch.nn.Module, train_loader_length: int, use_gpu: bool
):
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scaler = GradScaler(enabled=use_gpu and torch.cuda.is_available())
    total_steps = config.num_epochs * max(1, min(train_loader_length, config.limit_steps_train))
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=config.min_lr,
    )
    return optimizer, scaler, scheduler


def _distributed_worker(rank: int, world_size: int, config: ExperimentConfig, save_every: int, return_dict):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    local_rank, global_rank = init_ddp(use_gpu=True)
    device = torch.device(f"cuda:{local_rank}")

    train_loader, val_loader = get_dataloaders(
        config, use_gpu=True, rank=global_rank, world_size=world_size
    )

    model = create_finetune_model(config, verbose=(global_rank == 0))
    model = model.to(device)
    _configure_skip_offload(model, config)
    if _fsdp_enabled(config):
        model = _wrap_model_with_fsdp(model, local_rank, config)
    else:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    optimizer, scaler, scheduler = build_optimizer_scheduler(
        config, model, len(train_loader), use_gpu=True
    )

    train_losses, val_losses = train_model(
        config,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        scaler,
        local_rank,
        True,
        save_every,
        rmse_loss,
    )

    if global_rank == 0:
        return_dict["train_losses"] = train_losses
        return_dict["val_losses"] = val_losses

    dist.destroy_process_group()


def run_training(config: ExperimentConfig, num_gpus: int | None = None, save_every: int = 5):
    _ensure_expandable_cuda_segments()
    use_gpu = _should_use_gpu(config)

    if num_gpus is None:
        num_gpus = torch.cuda.device_count() if use_gpu else 0
    else:
        num_gpus = min(num_gpus, torch.cuda.device_count() if use_gpu else 0)
    num_gpus = max(1, num_gpus)

    if not use_gpu:
        device = torch.device("cpu")
        train_loader, val_loader = get_dataloaders(config, use_gpu=False, rank=0, world_size=1)
        model = create_finetune_model(config).to(device)
        _configure_skip_offload(model, config)
        optimizer, scaler, scheduler = build_optimizer_scheduler(
            config, model, len(train_loader), use_gpu=False
        )
        train_losses, val_losses = train_model(
            config,
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            scaler,
            local_rank=0,
            use_gpu=False,
            save_every=save_every,
            loss_func=rmse_loss,
        )
        return train_losses, val_losses

    if num_gpus == 1:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        mp.set_start_method("spawn", force=True)

        tried: list[int] = []
        oom_errors: list[str] = []
        for gpu_id in _single_gpu_retry_order():
            if gpu_id in tried:
                continue
            tried.append(gpu_id)
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            print(f"Single-GPU attempt on physical GPU {gpu_id}.")

            manager = mp.Manager()
            return_dict = manager.dict()
            torch.cuda.empty_cache()
            try:
                for attempt in range(5):
                    os.environ["MASTER_PORT"] = _pick_free_master_port()
                    try:
                        mp.spawn(
                            _distributed_worker,
                            args=(1, config, save_every, return_dict),
                            nprocs=1,
                            join=True,
                        )
                        break
                    except ProcessRaisedException as exc:
                        if attempt < 4 and _is_addr_in_use_error(exc):
                            print(
                                "DDP rendezvous port collision detected; "
                                f"retrying with a new port (attempt {attempt + 2}/5)."
                            )
                            continue
                        raise
                return return_dict.get("train_losses"), return_dict.get("val_losses")
            except ProcessRaisedException as exc:
                message = str(exc).lower()
                if "out of memory" in message or "cuda oom" in message:
                    oom_errors.append(str(exc))
                    print(
                        f"CUDA OOM on physical GPU {gpu_id}; "
                        "retrying on a different idle GPU if available."
                    )
                    continue
                raise
            except ProcessExitedException as exc:
                print(
                    f"Single-GPU run on physical GPU {gpu_id} exited unexpectedly ({exc}); "
                    "trying another idle GPU if available."
                )
                continue

        if oom_errors:
            raise RuntimeError(
                "Single-GPU training encountered CUDA OOM on all available GPU candidates."
            )
        raise RuntimeError("Single-GPU training failed on all available GPU candidates.")

    os.environ.setdefault("MASTER_ADDR", "localhost")

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    return_dict = manager.dict()

    torch.cuda.empty_cache()

    try:
        for attempt in range(5):
            os.environ["MASTER_PORT"] = _pick_free_master_port()
            try:
                mp.spawn(
                    _distributed_worker,
                    args=(num_gpus, config, save_every, return_dict),
                    nprocs=num_gpus,
                    join=True,
                )
                break
            except ProcessRaisedException as exc:
                if attempt < 4 and _is_addr_in_use_error(exc):
                    print(
                        "DDP rendezvous port collision detected; "
                        f"retrying with a new port (attempt {attempt + 2}/5)."
                    )
                    continue
                raise
    except ProcessRaisedException as exc:
        message = str(exc).lower()
        if "out of memory" in message and num_gpus > 1:
            if not _fsdp_enabled(config):
                print("Encountered CUDA OOM with DDP; enabling FSDP retry.")
                setattr(config, "distributed_strategy", "fsdp")
                if not getattr(config, "fsdp_precision", None):
                    setattr(
                        config,
                        "fsdp_precision",
                        "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp16",
                    )
                return run_training(config, num_gpus=num_gpus, save_every=save_every)

            reduced = max(1, num_gpus // 2)
            if reduced == num_gpus:
                reduced = num_gpus - 1
            print(
                f"Encountered CUDA OOM during multi-GPU setup; retrying with {reduced} GPU(s)."
            )
            return run_training(config, num_gpus=reduced, save_every=save_every)
        raise
    except ProcessExitedException as exc:
        print(f"Distributed training failed with signal/exit ({exc}); retrying on a single GPU.")
        return run_training(config, num_gpus=1, save_every=save_every)

    return return_dict.get("train_losses"), return_dict.get("val_losses")
def _configure_skip_offload(model: torch.nn.Module, config: ExperimentConfig):
    if not getattr(config, "skip_activation_offload", False):
        return
    if hasattr(model, "set_skip_activation_devices"):
        model.set_skip_activation_devices([torch.device("cpu")])
