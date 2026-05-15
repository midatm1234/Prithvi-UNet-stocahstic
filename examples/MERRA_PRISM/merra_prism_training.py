"""Training utilities for the MERRA2-to-PRISM downscaling model.

Closely mirrors the CORDEX_ML training logic but loads data through
the MERRA_PRISM dataset and uses the MERRA_PRISM YAML configuration.

Usage (via merra_prism_finetune.py):
    python merra_prism_finetune.py --config MERRA_PRISM.yaml
"""

from __future__ import annotations

import math
import os
import socket
import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

from merra_prism_dataset import MerraPrismDataset

try:
    from granitewxc.models.loss import build_loss_fn, rmse_loss
except ModuleNotFoundError:
    def rmse_loss(y_hat: torch.Tensor, y: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.sqrt(torch.mean((y_hat - y["y"]) ** 2))

    def build_loss_fn(config, output_vars):
        del config, output_vars
        return rmse_loss

from granitewxc.models.model import get_finetune_model_UNET
from granitewxc.utils.config import ExperimentConfig, get_config
from granitewxc.utils.predictands import build_predictand_specs
from granitewxc.utils.distributed import init_ddp
from granitewxc.utils.trainer import train_model

try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
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
    if min_params:
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
    """Adapt MerraPrismDataset output dict to the model's expected format."""

    def __init__(self, base: MerraPrismDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.base[idx]
        return {"x": sample["x"], "y": sample["y"]}


def _build_dataloader(
    config_path: str,
    config: ExperimentConfig,
    mode: str,
    *,
    shuffle: bool,
    distributed: bool,
    rank: int,
    world_size: int,
) -> DataLoader:
    base = MerraPrismDataset(config_path, mode=mode)
    dataset = _WrappedDataset(base)

    sampler = None
    if distributed:
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
) -> Tuple[DataLoader, DataLoader]:
    distributed = world_size > 1
    train_loader = _build_dataloader(
        config_path, config, "training",
        shuffle=True, distributed=distributed, rank=rank, world_size=world_size,
    )
    val_loader = _build_dataloader(
        config_path, config, "training",
        shuffle=False, distributed=distributed, rank=rank, world_size=world_size,
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
    if not hasattr(config.data, "input_static_surface_vars"):
        config.data.input_static_surface_vars = []
    output_vars = list(getattr(config.data, "output_vars", getattr(config.data, "target_variables", [])))
    if not output_vars:
        output_vars = list(getattr(config.data, "target_variables", []))
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

    manager = mp.Manager()
    return_dict = manager.dict()

    if num_gpus <= 1:
        print("[training] single-GPU mode")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        mp.spawn(_single_gpu_worker, args=(config, config_path, save_every, return_dict), nprocs=1)
    else:
        print(f"[training] distributed mode ({num_gpus} GPUs)")
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = _pick_free_port()
        mp.spawn(
            _distributed_worker,
            args=(num_gpus, config, config_path, save_every, return_dict),
            nprocs=num_gpus,
        )

    return return_dict.get("train_losses"), return_dict.get("val_losses")
