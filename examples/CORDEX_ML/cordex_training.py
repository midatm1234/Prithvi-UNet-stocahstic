"""Utilities to train the CORDEX finetuning model in single- or multi-GPU mode."""

from __future__ import annotations

import os
import sys
import subprocess
import socket
import math
import json
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler as CudaGradScaler
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
    from granitewxc.models.loss import build_loss_fn, rmse_loss
except ModuleNotFoundError:
    # Fallback for environments where the installed granitewxc package
    # does not expose the updated loss module.
    def rmse_loss(y_hat: torch.Tensor, y: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.sqrt(torch.mean((y_hat - y["y"]) ** 2))
    def build_loss_fn(config, output_vars):  # type: ignore[override]
        del config, output_vars
        return rmse_loss
from granitewxc.models.model import get_finetune_model_UNET
from granitewxc.utils.config import ExperimentConfig
from granitewxc.utils.predictands import build_predictand_specs
from granitewxc.utils.distributed import init_ddp
from granitewxc.utils.trainer import train_model
from torch.multiprocessing.spawn import ProcessRaisedException, ProcessExitedException


def _ensure_expandable_cuda_segments() -> None:
    """Request CUDA allocator settings that reduce fragmentation OOMs."""
    desired_entries = [
        ("expandable_segments", "True"),
        ("garbage_collection_threshold", "0.8"),
        ("max_split_size_mb", "128"),
    ]
    preferred_key = "PYTORCH_ALLOC_CONF"
    deprecated_key = "PYTORCH_CUDA_ALLOC_CONF"

    # Prefer the new env var; if only the deprecated one is set, migrate it.
    current = os.environ.get(preferred_key, "").strip()
    deprecated_current = os.environ.get(deprecated_key, "").strip()
    if not current:
        current = deprecated_current

    if not current:
        os.environ[preferred_key] = ",".join(f"{k}:{v}" for k, v in desired_entries)
    else:
        entries = [item.strip() for item in current.split(",") if item.strip()]
        keys = {item.split(":", 1)[0].strip().lower() for item in entries if ":" in item}
        missing = [f"{k}:{v}" for k, v in desired_entries if k.lower() not in keys]
        if missing:
            current = f"{current},{','.join(missing)}"
        os.environ[preferred_key] = current

    # Remove deprecated key so torch does not emit deprecation warnings.
    os.environ.pop(deprecated_key, None)


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

    process_counts = _query_gpu_process_counts_by_index()
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
                "proc_count": process_counts.get(int(idx_str), 0),
            }
        )
    return stats


def _query_gpu_process_counts_by_index() -> dict[int, int]:
    """
    Return the number of active compute processes per physical GPU index.
    Uses nvidia-smi UUID mapping so the result is robust to CUDA_VISIBLE_DEVICES remapping.
    """
    try:
        gpu_rows = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            text=True,
        )
    except Exception:
        return {}

    uuid_to_index: dict[str, int] = {}
    counts: dict[int, int] = {}
    for raw in gpu_rows.strip().splitlines():
        row = raw.strip()
        if not row:
            continue
        parts = [part.strip() for part in row.split(",", 1)]
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        idx = int(parts[0])
        uuid = parts[1]
        uuid_to_index[uuid] = idx
        counts[idx] = 0

    if not uuid_to_index:
        return counts

    try:
        proc_rows = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
            text=True,
        )
    except Exception:
        return counts

    for raw in proc_rows.strip().splitlines():
        row = raw.strip()
        if not row:
            continue
        parts = [part.strip() for part in row.split(",", 1)]
        if len(parts) != 2:
            continue
        gpu_uuid, pid = parts
        if not pid.isdigit():
            continue
        idx = uuid_to_index.get(gpu_uuid)
        if idx is None:
            continue
        counts[idx] = counts.get(idx, 0) + 1
    return counts


def _parse_visible_device_ids(value: str) -> list[int] | None:
    """
    Parse CUDA_VISIBLE_DEVICES when it is an index list (e.g. "0,2,3").
    Returns None for non-index tokens (UUID/MIG identifiers).
    """
    if not value:
        return []
    parsed: list[int] = []
    seen: set[int] = set()
    for token in [item.strip() for item in value.split(",") if item.strip()]:
        if not token.isdigit():
            return None
        gpu_id = int(token)
        if gpu_id not in seen:
            parsed.append(gpu_id)
            seen.add(gpu_id)
    return parsed


def _count_visible_devices(value: str) -> int | None:
    """Return number of CUDA-visible devices from env syntax, if specified."""
    value = (value or "").strip()
    if not value:
        return None
    tokens = [item.strip() for item in value.split(",") if item.strip()]
    if not tokens:
        return None
    return len(tokens)


def _set_cuda_visible_devices(value: str | None) -> None:
    if value is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(value)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _single_gpu_retry_order(allowed_gpu_ids: Sequence[int] | None = None) -> list[int]:
    """
    Return physical GPU IDs in retry order.
    Idle GPUs are always preferred. By default, busy GPUs are skipped when at
    least one idle candidate exists to avoid colliding with other active jobs.
    By default, "idle" means no active compute process on that GPU.
    Set CORDEX_INCLUDE_BUSY_GPU_CANDIDATES=1 to re-enable busy-GPU fallback.
    """
    allowed: list[int] | None = None
    if allowed_gpu_ids is not None:
        seen: set[int] = set()
        allowed = []
        for gpu_id in allowed_gpu_ids:
            idx = int(gpu_id)
            if idx not in seen:
                allowed.append(idx)
                seen.add(idx)

    stats = _query_gpu_stats()
    if allowed is not None and stats:
        allowed_set = set(allowed)
        stats = [gpu for gpu in stats if gpu["index"] in allowed_set]
    if not stats:
        if allowed:
            return allowed
        return [0]

    # By default, treat "idle" as no running compute process.
    # Optional fallback heuristic: low memory use + low utilization.
    require_no_compute_process = _env_flag("CORDEX_IDLE_REQUIRE_NO_COMPUTE_PROCESS", True)
    try:
        idle_max_used_mb = int(os.environ.get("CORDEX_IDLE_MAX_USED_MB", "1024"))
    except Exception:
        idle_max_used_mb = 1024
    try:
        idle_max_util = int(os.environ.get("CORDEX_IDLE_MAX_UTIL", "10"))
    except Exception:
        idle_max_util = 10
    include_busy_candidates = _env_flag("CORDEX_INCLUDE_BUSY_GPU_CANDIDATES", False)

    def _is_idle(gpu: dict[str, int]) -> bool:
        if require_no_compute_process:
            return int(gpu.get("proc_count", 0)) == 0
        return gpu["used_mb"] <= idle_max_used_mb and gpu["util"] <= idle_max_util

    idle_gpus = [gpu for gpu in stats if _is_idle(gpu)]
    busy_gpus = [gpu for gpu in stats if not _is_idle(gpu)]

    idle_ranked = sorted(idle_gpus, key=lambda g: (-g["free_mb"], g["used_mb"], g["util"], g["index"]))
    busy_ranked = sorted(busy_gpus, key=lambda g: (-g["free_mb"], g["util"], g["used_mb"], g["index"]))
    if idle_ranked:
        if include_busy_candidates:
            ordered = [gpu["index"] for gpu in [*idle_ranked, *busy_ranked]]
        else:
            ordered = [gpu["index"] for gpu in idle_ranked]
    else:
        ordered = [gpu["index"] for gpu in busy_ranked]

    if allowed:
        seen_stats = {gpu["index"] for gpu in stats}
        # Preserve allowed IDs that are not present in current nvidia-smi output.
        missing = [gpu_id for gpu_id in allowed if gpu_id not in seen_stats]
        ordered.extend(missing)
    return ordered


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


def _coerce_mapping(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {}


def _set_training_distributed_mode(config: ExperimentConfig, mode: str) -> None:
    """Synchronize distributed mode at both top-level and nested training config."""
    mode = str(mode).lower()
    if mode not in {"ddp", "fsdp", "none"}:
        return

    if mode == "fsdp":
        setattr(config, "distributed_strategy", "fsdp")
    elif mode == "ddp":
        setattr(config, "distributed_strategy", "ddp")

    training_cfg = getattr(config, "training", None)
    if training_cfg is None:
        return
    if isinstance(training_cfg, dict):
        training_cfg["distributed"] = mode
        return
    try:
        setattr(training_cfg, "distributed", mode)
    except Exception:
        pass


def _set_training_runtime_param(config: ExperimentConfig, key: str, value: int | str) -> None:
    """Synchronize a runtime knob at both top-level and nested training config."""
    setattr(config, key, value)
    training_cfg = getattr(config, "training", None)
    if training_cfg is None:
        return
    if isinstance(training_cfg, dict):
        training_cfg[key] = value
        return
    try:
        setattr(training_cfg, key, value)
    except Exception:
        pass


def _resolve_checkpoint_dir(config: ExperimentConfig) -> str | None:
    checkpoint_dir = getattr(config, "checkpoint_dir", None)
    if checkpoint_dir:
        return str(checkpoint_dir)
    path_experiment = getattr(config, "path_experiment", None)
    if path_experiment:
        return os.path.join(str(path_experiment), "weights")
    return None


def _find_latest_retry_checkpoint(config: ExperimentConfig) -> str | None:
    checkpoint_dir = _resolve_checkpoint_dir(config)
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return None

    latest_epoch: tuple[int, str] | None = None
    for name in os.listdir(checkpoint_dir):
        if not (name.startswith("epoch_") and name.endswith(".ckpt")):
            continue
        token = name[len("epoch_") : -len(".ckpt")]
        if not token.isdigit():
            continue
        candidate = (int(token), os.path.join(checkpoint_dir, name))
        if latest_epoch is None or candidate[0] > latest_epoch[0]:
            latest_epoch = candidate

    if latest_epoch is not None:
        return latest_epoch[1]

    for leaf in ("last.ckpt", "best.ckpt"):
        candidate = os.path.join(checkpoint_dir, leaf)
        if os.path.exists(candidate):
            return candidate
    return None


def _enable_resume_for_retry(config: ExperimentConfig, *, reason: str) -> bool:
    checkpoint = _find_latest_retry_checkpoint(config)
    if checkpoint is None:
        return False
    setattr(config, "resume_training", True)
    setattr(config, "resume_from_last_checkpoint", False)
    setattr(config, "resume_checkpoint_path", checkpoint)
    print(f"[resume] {reason}; restarting from {checkpoint}")
    return True


def _resolve_retry_limit(
    config: ExperimentConfig, *, config_key: str, env_key: str, default_value: int
) -> int:
    training_cfg = _coerce_mapping(getattr(config, "training", None))
    value = training_cfg.get(config_key, getattr(config, config_key, None))
    if value is None:
        value = os.environ.get(env_key, str(default_value))
    try:
        parsed = int(value)
    except Exception:
        parsed = default_value
    return max(0, parsed)


def _infer_completed_epoch_from_checkpoint(path: str) -> int:
    name = os.path.basename(path)
    if name.startswith("epoch_") and name.endswith(".ckpt"):
        token = name[len("epoch_") : -len(".ckpt")]
        if token.isdigit():
            return max(0, int(token))

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        epoch_idx = int(payload.get("epoch", -1)) if isinstance(payload, dict) else -1
        return max(0, epoch_idx + 1)
    except Exception:
        return 0


def _latest_completed_epoch(config: ExperimentConfig) -> int:
    checkpoint = _find_latest_retry_checkpoint(config)
    if checkpoint is None:
        return 0
    return _infer_completed_epoch_from_checkpoint(checkpoint)


def _record_auto_restart(config: ExperimentConfig, *, reason: str) -> None:
    total = int(getattr(config, "_auto_restart_total", 0)) + 1
    setattr(config, "_auto_restart_total", total)

    last_progress_epoch = int(getattr(config, "_auto_restart_last_progress_epoch", 0))
    latest_epoch = _latest_completed_epoch(config)
    if latest_epoch > last_progress_epoch:
        last_progress_epoch = latest_epoch
        no_progress = 0
    else:
        no_progress = int(getattr(config, "_auto_restart_no_progress", 0)) + 1
    setattr(config, "_auto_restart_last_progress_epoch", last_progress_epoch)
    setattr(config, "_auto_restart_no_progress", no_progress)

    max_total = _resolve_retry_limit(
        config,
        config_key="max_auto_restarts",
        env_key="CORDEX_MAX_AUTO_RESTARTS",
        default_value=12,
    )
    max_no_progress = _resolve_retry_limit(
        config,
        config_key="max_no_progress_restarts",
        env_key="CORDEX_MAX_NO_PROGRESS_RESTARTS",
        default_value=4,
    )

    print(
        f"[retry] {reason}; auto_restart={total}/{max_total}, "
        f"no_progress={no_progress}/{max_no_progress}, "
        f"latest_completed_epoch={latest_epoch}."
    )

    if max_total > 0 and total > max_total:
        raise RuntimeError(
            "Aborting auto-retries: maximum restart budget exceeded "
            f"({total} > {max_total})."
        )
    if max_no_progress > 0 and no_progress > max_no_progress:
        raise RuntimeError(
            "Aborting auto-retries: no completed epoch progress across retries "
            f"({no_progress} > {max_no_progress})."
        )


def _prepare_retry(config: ExperimentConfig, *, reason: str) -> None:
    _enable_resume_for_retry(config, reason=reason)
    _record_auto_restart(config, reason=reason)


def _resolve_training_runtime(config: ExperimentConfig) -> dict[str, int | str | None]:
    training_cfg = _coerce_mapping(getattr(config, "training", None))
    per_device_batch = int(training_cfg.get("per_device_batch_size", getattr(config, "batch_size", 1)))
    grad_accum = int(
        training_cfg.get(
            "gradient_accumulation_steps",
            getattr(config, "gradient_accumulation_steps", 1),
        )
    )
    num_gpus_cfg = training_cfg.get("num_gpus", getattr(config, "num_gpus", None))
    if num_gpus_cfg is not None:
        num_gpus_cfg = int(num_gpus_cfg)
    distributed_mode = str(
        training_cfg.get(
            "distributed",
            getattr(config, "distributed_strategy", "ddp"),
        )
    ).lower()
    if distributed_mode not in {"ddp", "fsdp", "none"}:
        distributed_mode = "ddp"

    setattr(config, "batch_size", max(1, per_device_batch))
    setattr(config, "per_device_batch_size", max(1, per_device_batch))
    setattr(config, "gradient_accumulation_steps", max(1, grad_accum))
    if distributed_mode == "fsdp":
        setattr(config, "distributed_strategy", "fsdp")
    elif distributed_mode == "ddp":
        setattr(config, "distributed_strategy", "ddp")

    return {
        "per_device_batch_size": max(1, per_device_batch),
        "gradient_accumulation_steps": max(1, grad_accum),
        "num_gpus": num_gpus_cfg,
        "distributed": distributed_mode,
    }


def _allow_single_gpu_multi_fallback(
    config: ExperimentConfig, *, available_gpus: int | None = None
) -> bool:
    """
    Control whether single-GPU OOM should auto-escalate to multi-GPU.
    Default is disabled unless explicitly enabled via config/env.
    """
    def _parse_bool(value) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return None

    training_cfg = _coerce_mapping(getattr(config, "training", None))
    flag = training_cfg.get("allow_multi_gpu_fallback", getattr(config, "allow_multi_gpu_fallback", None))
    parsed = _parse_bool(flag)
    if parsed is not None:
        return parsed

    env = os.environ.get("CORDEX_ALLOW_MULTI_GPU_FALLBACK", "")
    parsed_env = _parse_bool(env)
    if parsed_env is not None:
        return parsed_env

    # Keep single-GPU as the stable default behavior.
    return False


def _allow_distributed_to_single_gpu_fallback(config: ExperimentConfig) -> bool:
    """
    Control whether distributed (>1 GPU) failures may auto-fallback to single-GPU.
    Default is enabled for backwards compatibility unless explicitly disabled.
    """

    def _parse_bool(value) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return None

    training_cfg = _coerce_mapping(getattr(config, "training", None))
    flag = training_cfg.get(
        "allow_distributed_single_gpu_fallback",
        getattr(config, "allow_distributed_single_gpu_fallback", None),
    )
    parsed = _parse_bool(flag)
    if parsed is not None:
        return parsed

    disable_env = _parse_bool(os.environ.get("CORDEX_DISABLE_DISTRIBUTED_TO_SINGLE_FALLBACK", ""))
    if disable_env is not None:
        return not disable_env

    allow_env = _parse_bool(os.environ.get("CORDEX_ALLOW_DISTRIBUTED_SINGLE_GPU_FALLBACK", ""))
    if allow_env is not None:
        return allow_env

    return True


def _allow_oom_crop_shrink(config: ExperimentConfig) -> bool:
    """
    Control whether OOM backoff may alter train/target crop geometry.
    Default is disabled to keep training results comparable.
    """

    def _parse_bool(value) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return None

    training_cfg = _coerce_mapping(getattr(config, "training", None))
    flag = training_cfg.get("allow_oom_crop_shrink", getattr(config, "allow_oom_crop_shrink", None))
    parsed = _parse_bool(flag)
    if parsed is not None:
        return parsed

    env = os.environ.get("CORDEX_ALLOW_OOM_CROP_SHRINK", "")
    parsed_env = _parse_bool(env)
    if parsed_env is not None:
        return parsed_env

    return False


def _resolve_single_gpu_oom_retry_limit(config: ExperimentConfig) -> int:
    training_cfg = _coerce_mapping(getattr(config, "training", None))
    value = training_cfg.get(
        "max_single_gpu_oom_retries",
        getattr(config, "max_single_gpu_oom_retries", None),
    )
    if value is None:
        value = os.environ.get("CORDEX_MAX_SINGLE_GPU_OOM_RETRIES", "5")
    try:
        parsed = int(value)
    except Exception:
        parsed = 5
    return max(0, parsed)


def _choose_multi_gpu_oom_fallback(
    *,
    available_gpus: int,
    per_device_batch_size: int,
    target_effective_batch_size: int,
) -> tuple[int, int] | None:
    """
    Pick a multi-GPU fallback (num_gpus, grad_accum) that best preserves
    effective batch size when single-GPU attempts OOM.
    """
    max_gpus = max(1, min(4, int(available_gpus)))
    if max_gpus <= 1:
        return None

    # Prefer counts that preserve target effective batch size exactly.
    candidates = list(range(max_gpus, 1, -1))
    for num_gpus in candidates:
        denom = per_device_batch_size * num_gpus
        if denom > 0 and target_effective_batch_size % denom == 0:
            return num_gpus, max(1, target_effective_batch_size // denom)

    # Fall back to the largest candidate and nearest integer accumulation.
    num_gpus = candidates[0]
    denom = max(1, per_device_batch_size * num_gpus)
    grad_accum = max(1, round(target_effective_batch_size / denom))
    return num_gpus, grad_accum


def _align_up(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def _align_down(value: int, multiple: int, floor_value: int) -> int:
    if multiple <= 1:
        return max(floor_value, value)
    aligned = (value // multiple) * multiple
    return max(floor_value, aligned)


def _resolve_crop_alignment(config: ExperimentConfig) -> int:
    align = 1
    downsample = int(getattr(config.data, "downsample_factor", 1) or 1)
    if downsample > 1:
        align = math.lcm(align, downsample)

    mask_size = getattr(config, "mask_unit_size", None)
    if isinstance(mask_size, (list, tuple)):
        for item in mask_size:
            item = int(item)
            if item > 1:
                align = math.lcm(align, item)

    patch_size = getattr(getattr(config, "model", None), "downscaling_patch_size", None)
    if isinstance(patch_size, (list, tuple)):
        for item in patch_size:
            item = int(item)
            if item > 1:
                align = math.lcm(align, item)

    return max(1, align)


def _reduce_train_crop_size_for_oom(
    config: ExperimentConfig,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    target_lat = int(getattr(config.data, "target_size_lat", 0) or 0)
    target_lon = int(getattr(config.data, "target_size_lon", 0) or 0)
    if target_lat <= 0 or target_lon <= 0:
        return None

    current_lat = int(getattr(config.data, "train_crop_size_lat", target_lat) or target_lat)
    current_lon = int(getattr(config.data, "train_crop_size_lon", target_lon) or target_lon)
    if current_lat <= 0 or current_lon <= 0:
        current_lat, current_lon = target_lat, target_lon

    min_crop = max(16, int(os.environ.get("CORDEX_OOM_MIN_TRAIN_CROP_SIZE", "64")))
    align = _resolve_crop_alignment(config)
    min_lat = _align_up(min_crop, align)
    min_lon = _align_up(min_crop, align)

    if current_lat <= min_lat and current_lon <= min_lon:
        return None

    next_lat = _align_down(max(min_lat, current_lat // 2), align, min_lat)
    next_lon = _align_down(max(min_lon, current_lon // 2), align, min_lon)

    if next_lat >= current_lat and current_lat > min_lat:
        next_lat = _align_down(current_lat - align, align, min_lat)
    if next_lon >= current_lon and current_lon > min_lon:
        next_lon = _align_down(current_lon - align, align, min_lon)
    if next_lat >= current_lat and next_lon >= current_lon:
        return None

    next_lat = min(next_lat, target_lat)
    next_lon = min(next_lon, target_lon)

    setattr(config.data, "train_crop_size_lat", int(next_lat))
    setattr(config.data, "train_crop_size_lon", int(next_lon))
    return (current_lat, current_lon), (next_lat, next_lon)


def _reduce_target_crop_size_for_oom(config: ExperimentConfig) -> tuple[tuple[int, int], tuple[int, int]] | None:
    current_lat = int(getattr(config.data, "target_size_lat", 0) or 0)
    current_lon = int(getattr(config.data, "target_size_lon", 0) or 0)
    if current_lat <= 0 or current_lon <= 0:
        return None

    min_crop = max(16, int(os.environ.get("CORDEX_OOM_MIN_TARGET_SIZE", "64")))
    align = _resolve_crop_alignment(config)
    min_lat = _align_up(min_crop, align)
    min_lon = _align_up(min_crop, align)
    if current_lat <= min_lat and current_lon <= min_lon:
        return None

    next_lat = _align_down(max(min_lat, current_lat // 2), align, min_lat)
    next_lon = _align_down(max(min_lon, current_lon // 2), align, min_lon)

    if next_lat >= current_lat and current_lat > min_lat:
        next_lat = _align_down(current_lat - align, align, min_lat)
    if next_lon >= current_lon and current_lon > min_lon:
        next_lon = _align_down(current_lon - align, align, min_lon)
    if next_lat >= current_lat and next_lon >= current_lon:
        return None

    setattr(config.data, "target_size_lat", int(next_lat))
    setattr(config.data, "target_size_lon", int(next_lon))

    downsample = max(1, int(getattr(config.data, "downsample_factor", 1) or 1))
    if hasattr(config.data, "input_size_lat"):
        setattr(config.data, "input_size_lat", max(1, next_lat // downsample))
    if hasattr(config.data, "input_size_lon"):
        setattr(config.data, "input_size_lon", max(1, next_lon // downsample))

    return (current_lat, current_lon), (next_lat, next_lon)


def _apply_single_gpu_oom_backoff(
    config: ExperimentConfig,
    *,
    num_gpus: int,
    target_effective_batch_size: int,
) -> str | None:
    updates: list[str] = []
    num_workers = int(getattr(config, "dl_num_workers", 0))
    prefetch_size = int(getattr(config, "dl_prefetch_size", 0) or 0)
    pin_memory = bool(
        getattr(
            config,
            "dl_pin_memory",
            getattr(config, "pin_memory", False),
        )
    )
    if num_workers > 0:
        setattr(config, "dl_num_workers", 0)
        updates.append("dl_num_workers=0")
    if prefetch_size > 0:
        setattr(config, "dl_prefetch_size", 0)
        updates.append("dl_prefetch_size=0")
    if pin_memory:
        setattr(config, "dl_pin_memory", False)
        setattr(config, "pin_memory", False)
        updates.append("pin_memory=False")

    if not bool(getattr(config, "backbone_gradient_checkpointing", False)):
        setattr(config, "backbone_gradient_checkpointing", True)
        updates.append("backbone_gradient_checkpointing=True")

    if not bool(getattr(config, "skip_activation_offload", False)):
        setattr(config, "skip_activation_offload", True)
        updates.append("skip_activation_offload=True")

    if updates:
        return "applied emergency memory profile (" + ", ".join(updates) + ")"

    per_device_batch_size = int(getattr(config, "per_device_batch_size", getattr(config, "batch_size", 1)))
    if per_device_batch_size > 1:
        new_batch_size = max(1, per_device_batch_size // 2)
        _set_training_runtime_param(config, "batch_size", new_batch_size)
        _set_training_runtime_param(config, "per_device_batch_size", new_batch_size)
        denom = max(1, num_gpus * new_batch_size)
        new_grad_accum = max(1, round(target_effective_batch_size / denom))
        _set_training_runtime_param(config, "gradient_accumulation_steps", new_grad_accum)
        return (
            "reduced per-device batch size "
            f"{per_device_batch_size}->{new_batch_size} and adjusted "
            f"gradient_accumulation_steps={new_grad_accum}"
        )

    if _allow_oom_crop_shrink(config):
        train_crop_sizes = _reduce_train_crop_size_for_oom(config)
        if train_crop_sizes is not None:
            old_size, new_size = train_crop_sizes
            return (
                "reduced train_crop_size "
                f"{old_size[0]}x{old_size[1]} -> {new_size[0]}x{new_size[1]}"
            )

        crop_sizes = _reduce_target_crop_size_for_oom(config)
        if crop_sizes is not None:
            old_size, new_size = crop_sizes
            return (
                "reduced target_size "
                f"{old_size[0]}x{old_size[1]} -> {new_size[0]}x{new_size[1]}"
            )

    return None


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
    random_crop_offset: Tuple[int, int] = (0, 0),
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
        random_crop_offset=random_crop_offset,
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


class CordexIndexSubset(torch.utils.data.Dataset):
    """Index subset that preserves the wrapped dataset's ``base`` interface."""

    def __init__(self, dataset: CordexWrappedDataset, indices: range):
        self.dataset = dataset
        self.indices = indices
        self.base = dataset.base

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.dataset[self.indices[idx]]


def _contiguous_holdout_indices(total: int, fraction: float, role: str) -> range:
    """Return disjoint leading-training or trailing-validation indices."""

    total = int(total)
    fraction = float(fraction)
    if total < 2:
        raise ValueError("A train/validation holdout requires at least two samples.")
    if not 0.0 < fraction < 1.0:
        raise ValueError(
            f"data.validation_holdout_fraction must lie in (0, 1), got {fraction}."
        )
    validation_count = max(1, int(round(total * fraction)))
    validation_count = min(validation_count, total - 1)
    split = total - validation_count
    if role == "train":
        return range(0, split)
    if role == "validation":
        return range(split, total)
    raise ValueError(f"Unknown holdout role {role!r}.")


def _validate_train_validation_source_paths(
    train_predictors: Sequence[str],
    train_targets: Sequence[str],
    validation_predictors: Sequence[str],
    validation_targets: Sequence[str],
) -> bool:
    """Return exact paired-source identity and reject any partial overlap."""

    def path_identities(paths: Sequence[str]) -> list[str]:
        return [os.path.normcase(os.path.normpath(os.fspath(path))) for path in paths]

    train_predictor_ids = path_identities(train_predictors)
    train_target_ids = path_identities(train_targets)
    validation_predictor_ids = path_identities(validation_predictors)
    validation_target_ids = path_identities(validation_targets)
    sources_identical = (
        train_predictor_ids == validation_predictor_ids
        and train_target_ids == validation_target_ids
    )
    predictor_overlap = sorted(
        set(train_predictor_ids).intersection(validation_predictor_ids)
    )
    target_overlap = sorted(set(train_target_ids).intersection(validation_target_ids))
    if not sources_identical and (predictor_overlap or target_overlap):
        raise ValueError(
            "Training and validation source lists partially overlap. Source lists "
            "must be either exactly identical paired lists (with a configured "
            "contiguous-tail holdout) or fully disjoint. "
            f"Shared predictor paths: {predictor_overlap or 'none'}; "
            f"shared target paths: {target_overlap or 'none'}."
        )
    return sources_identical


def _validate_scalar_holdout_metadata(
    config: ExperimentConfig,
    *,
    source_sample_count: int,
    training_sample_count: int,
    holdout_fraction: float,
    holdout_strategy: str,
) -> dict:
    """Fail closed unless scalar metadata proves training-only estimation."""

    model_cfg = getattr(config, "model", None)
    scalar_path = getattr(model_cfg, "target_sigma", None)
    if scalar_path in (None, ""):
        scalers = getattr(getattr(config, "data", None), "scalers", None)
        if isinstance(scalers, dict):
            scalar_path = scalers.get("targets_std")
        else:
            scalar_path = getattr(scalers, "targets_std", None)
    if scalar_path in (None, ""):
        raise ValueError(
            "A validation holdout is configured, but no target scalar path is "
            "available to verify training-only scalar fitting."
        )

    metadata_path = Path(_resolve_path(str(scalar_path))).parent / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError(
            "Validation-holdout training requires scalar selection metadata at "
            f"{metadata_path}. Recompute scalars with compute_scalars_cordex.py."
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Scalar selection metadata is unreadable at {metadata_path}: {exc}."
        ) from exc

    selection = metadata.get("sample_selection")
    if not isinstance(selection, dict):
        raise ValueError(
            "Scalar metadata predates training-only holdout selection (missing "
            f"sample_selection in {metadata_path}). Recompute scalars; old "
            f"metadata reports num_samples={metadata.get('num_samples')!r}, but "
            f"the training partition contains {training_sample_count}."
        )

    expected = {
        "policy": "leading_training_partition_excluding_contiguous_validation_tail",
        "source_sample_count": int(source_sample_count),
        "used_sample_count": int(training_sample_count),
        "used_index_start": 0,
        "used_index_stop_exclusive": int(training_sample_count),
        "excluded_index_start": int(training_sample_count),
        "excluded_index_stop_exclusive": int(source_sample_count),
        "config_training_validation_sources_identical": True,
        "validation_holdout_strategy": str(holdout_strategy),
    }
    mismatches = [
        f"{key}: metadata={selection.get(key)!r}, expected={value!r}"
        for key, value in expected.items()
        if selection.get(key) != value
    ]
    try:
        saved_fraction = float(selection.get("validation_holdout_fraction"))
    except (TypeError, ValueError):
        saved_fraction = float("nan")
    if not math.isclose(
        saved_fraction, float(holdout_fraction), rel_tol=0.0, abs_tol=1e-12
    ):
        mismatches.append(
            "validation_holdout_fraction: "
            f"metadata={selection.get('validation_holdout_fraction')!r}, "
            f"expected={holdout_fraction!r}"
        )
    if metadata.get("num_samples") != int(training_sample_count):
        mismatches.append(
            f"num_samples: metadata={metadata.get('num_samples')!r}, "
            f"expected={training_sample_count!r}"
        )
    if mismatches:
        raise ValueError(
            "Scalar metadata is incompatible with the configured validation "
            "holdout; refusing leaked normalization statistics:\n  - "
            + "\n  - ".join(mismatches)
            + "\nRecompute scalars with compute_scalars_cordex.py."
        )
    return metadata


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
    random_crop_offset: Tuple[int, int] = (0, 0),
    holdout_fraction: float | None = None,
    holdout_role: str | None = None,
) -> DataLoader:
    predictor_paths = _resolve_paths(predictor_paths)
    target_paths = _resolve_paths(target_paths)
    base_dataset = _build_base_dataset(
        config,
        predictor_paths,
        target_paths,
        crop_size=crop_size,
        random_crop=random_crop,
        random_crop_offset=random_crop_offset,
    )
    if random_crop and not bool(getattr(base_dataset, "random_crop", False)):
        fine_h, fine_w = getattr(base_dataset, "fine_shape", (None, None))
        crop_h, crop_w = getattr(base_dataset, "crop_size", (None, None))
        if rank == 0:
            print(
                "[data] random_crop requested but disabled because "
                f"crop_size={crop_h}x{crop_w} covers full grid {fine_h}x{fine_w}. "
                "Set data.train_crop_size_lat/lon smaller than the target grid to enable crop augmentation."
            )
    dataset = CordexWrappedDataset(base_dataset)
    if holdout_role is not None:
        if holdout_fraction is None:
            raise ValueError("holdout_role requires holdout_fraction.")
        indices = _contiguous_holdout_indices(
            len(dataset), holdout_fraction, holdout_role
        )
        dataset = CordexIndexSubset(dataset, indices)
        if rank == 0:
            print(
                f"[data] {holdout_role} uses disjoint contiguous indices "
                f"[{indices.start}, {indices.stop}) of {len(base_dataset)} samples."
            )

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
) -> Tuple[DataLoader, DataLoader | None]:
    distributed = world_size > 1
    target_crop_size = (
        int(config.data.target_size_lat),
        int(config.data.target_size_lon),
    )
    train_crop_size = (
        int(getattr(config.data, "train_crop_size_lat", target_crop_size[0])),
        int(getattr(config.data, "train_crop_size_lon", target_crop_size[1])),
    )
    offset_cfg = getattr(config.data, "train_random_crop_offset", (0, 0))
    if isinstance(offset_cfg, (int, float)):
        train_random_crop_offset = (max(0, int(offset_cfg)), max(0, int(offset_cfg)))
    elif isinstance(offset_cfg, (list, tuple)) and len(offset_cfg) == 2:
        train_random_crop_offset = (max(0, int(offset_cfg[0])), max(0, int(offset_cfg[1])))
    else:
        train_random_crop_offset = (0, 0)

    validation_enabled = bool(getattr(config, "validation_enabled", True))
    if not validation_enabled:
        if rank == 0:
            print(
                "[data] validation disabled; using 100% of the configured "
                "training sources with no validation loader or holdout."
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
            crop_size=train_crop_size,
            random_crop=True,
            random_crop_offset=train_random_crop_offset,
            holdout_fraction=None,
            holdout_role=None,
        )
        return train_loader, None

    val_crop_size = (
        int(getattr(config.data, "val_crop_size_lat", target_crop_size[0])),
        int(getattr(config.data, "val_crop_size_lon", target_crop_size[1])),
    )

    train_predictors = _resolve_paths(config.data.training_predictor_paths)
    train_targets = _resolve_paths(config.data.training_target_paths)
    validation_predictors = _resolve_paths(config.data.validation_predictor_paths)
    validation_targets = _resolve_paths(config.data.validation_target_paths)
    sources_overlap_exactly = _validate_train_validation_source_paths(
        train_predictors,
        train_targets,
        validation_predictors,
        validation_targets,
    )
    holdout_fraction = float(
        getattr(config.data, "validation_holdout_fraction", 0.0) or 0.0
    )
    holdout_strategy = str(
        getattr(config.data, "validation_holdout_strategy", "contiguous_tail")
    ).strip().lower()
    if holdout_strategy != "contiguous_tail":
        raise ValueError(
            "Only data.validation_holdout_strategy='contiguous_tail' is "
            f"implemented, got {holdout_strategy!r}."
        )
    if sources_overlap_exactly and not 0.0 < holdout_fraction < 1.0:
        raise ValueError(
            "Training and validation paths are identical, but no disjoint holdout "
            "is configured. Set data.validation_holdout_fraction in (0, 1), or "
            "provide separate validation files."
        )
    if not sources_overlap_exactly and holdout_fraction != 0.0 and rank == 0:
        print(
            "[data] separate validation files are configured; "
            "data.validation_holdout_fraction is ignored."
        )
    shared_holdout = holdout_fraction if sources_overlap_exactly else None

    train_loader = build_dataloader(
        config,
        config.data.training_predictor_paths,
        config.data.training_target_paths,
        shuffle=True,
        use_gpu=use_gpu,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        crop_size=train_crop_size,
        random_crop=True,
        random_crop_offset=train_random_crop_offset,
        holdout_fraction=shared_holdout,
        holdout_role="train" if shared_holdout is not None else None,
    )
    if shared_holdout is not None:
        _validate_scalar_holdout_metadata(
            config,
            source_sample_count=len(train_loader.dataset.base),
            training_sample_count=len(train_loader.dataset),
            holdout_fraction=shared_holdout,
            holdout_strategy=holdout_strategy,
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
        crop_size=val_crop_size,
        random_crop=False,
        random_crop_offset=(0, 0),
        holdout_fraction=shared_holdout,
        holdout_role="validation" if shared_holdout is not None else None,
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


def _build_grad_scaler(enabled: bool):
    # torch.amp.GradScaler is the non-deprecated API in recent torch releases.
    amp_module = getattr(torch, "amp", None)
    if amp_module is not None:
        grad_scaler_cls = getattr(amp_module, "GradScaler", None)
        if grad_scaler_cls is not None:
            try:
                return grad_scaler_cls("cuda", enabled=enabled)
            except TypeError:
                # Older torch variants may not accept device-type as first arg.
                return grad_scaler_cls(enabled=enabled)
    return CudaGradScaler(enabled=enabled)


def build_optimizer_scheduler(
    config: ExperimentConfig, model: torch.nn.Module, train_loader_length: int, use_gpu: bool
):
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("No trainable parameters are available for the optimizer.")
    optimizer = AdamW(trainable_parameters, lr=config.learning_rate)
    scaler = _build_grad_scaler(enabled=use_gpu and torch.cuda.is_available())
    accumulation_steps = max(1, int(getattr(config, "gradient_accumulation_steps", 1)))
    steps_per_epoch = max(1, min(train_loader_length, config.limit_steps_train))
    optimizer_steps_per_epoch = (steps_per_epoch + accumulation_steps - 1) // accumulation_steps
    total_steps = config.num_epochs * max(1, optimizer_steps_per_epoch)
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
    try:
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
        loss_fn = build_loss_fn(config, list(config.data.output_vars))
        if global_rank == 0 and hasattr(loss_fn, "describe"):
            print(f"[loss] active: {loss_fn.describe()}")

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
            loss_fn,
        )

        if global_rank == 0:
            return_dict["train_losses"] = train_losses
            return_dict["val_losses"] = val_losses
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _single_gpu_worker(rank: int, config: ExperimentConfig, save_every: int, return_dict):
    """
    Run single-GPU training without DDP/FSDP wrapping to reduce memory overhead.
    This is executed in a spawned subprocess so CUDA_VISIBLE_DEVICES retries remain isolated.
    """
    if rank != 0:
        return

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    train_loader, val_loader = get_dataloaders(config, use_gpu=True, rank=0, world_size=1)

    model = create_finetune_model(config, verbose=True).to(device)
    _configure_skip_offload(model, config)

    optimizer, scaler, scheduler = build_optimizer_scheduler(
        config, model, len(train_loader), use_gpu=True
    )
    loss_fn = build_loss_fn(config, list(config.data.output_vars))
    if hasattr(loss_fn, "describe"):
        print(f"[loss] active: {loss_fn.describe()}")

    train_losses, val_losses = train_model(
        config,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        scaler,
        local_rank=0,
        use_gpu=True,
        save_every=save_every,
        loss_func=loss_fn,
    )
    return_dict["train_losses"] = train_losses
    return_dict["val_losses"] = val_losses


def _run_distributed_from_env(config: ExperimentConfig, save_every: int):
    local_rank, global_rank = init_ddp(use_gpu=True)
    try:
        world_size = dist.get_world_size()
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
        loss_fn = build_loss_fn(config, list(config.data.output_vars))
        if global_rank == 0 and hasattr(loss_fn, "describe"):
            print(f"[loss] active: {loss_fn.describe()}")

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
            loss_fn,
        )
        if global_rank == 0:
            return train_losses, val_losses
        return None, None
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def run_training(config: ExperimentConfig, num_gpus: int | None = None, save_every: int = 5):
    _ensure_expandable_cuda_segments()
    use_gpu = _should_use_gpu(config)
    runtime = _resolve_training_runtime(config)

    launched_with_torchrun = bool(os.environ.get("WORLD_SIZE")) and bool(os.environ.get("RANK"))
    if launched_with_torchrun:
        if not use_gpu:
            raise RuntimeError("torchrun distributed launch requires CUDA for this training path.")
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        per_device_batch_size = int(getattr(config, "per_device_batch_size", config.batch_size))
        grad_accum = int(getattr(config, "gradient_accumulation_steps", 1))
        effective_batch_size = per_device_batch_size * world_size * grad_accum
        setattr(config, "effective_batch_size", effective_batch_size)
        print(
            "[training] external launcher detected (torchrun). "
            f"per_device_batch_size={per_device_batch_size}, world_size={world_size}, "
            f"gradient_accumulation_steps={grad_accum}, effective_batch_size={effective_batch_size}"
        )
        return _run_distributed_from_env(config, save_every=save_every)

    requested_num_gpus = num_gpus if num_gpus is not None else runtime["num_gpus"]
    available_gpus = torch.cuda.device_count() if use_gpu else 0
    if requested_num_gpus is None:
        # Default to single-GPU unless config/runtime explicitly asks for more.
        # This keeps notebook/script launches predictable and avoids accidental
        # all-visible-GPU fan-out.
        num_gpus = 1 if available_gpus > 0 else 0
    else:
        num_gpus = min(int(requested_num_gpus), available_gpus)
    num_gpus = max(1, min(4, num_gpus))
    if runtime["distributed"] == "none":
        if requested_num_gpus is not None and int(requested_num_gpus) > 1:
            raise RuntimeError(
                "Multi-GPU was explicitly requested but training.distributed='none'. "
                "Set training.distributed to 'ddp' or 'fsdp'."
            )
        num_gpus = 1
    visible_env_preflight = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    visible_count = _count_visible_devices(visible_env_preflight)
    if visible_count is not None and visible_count > 0 and num_gpus > visible_count:
        print(
            f"[training] requested num_gpus={num_gpus} but only {visible_count} device(s) are "
            f"visible via CUDA_VISIBLE_DEVICES={visible_env_preflight}; capping to {visible_count}."
        )
        num_gpus = visible_count
    if requested_num_gpus is not None and int(requested_num_gpus) > 1 and num_gpus < int(requested_num_gpus):
        raise RuntimeError(
            f"Requested num_gpus={int(requested_num_gpus)} but only {num_gpus} usable GPU(s) were resolved. "
            "Ensure enough idle GPUs are exposed in CUDA_VISIBLE_DEVICES."
        )

    per_device_batch_size = int(getattr(config, "per_device_batch_size", config.batch_size))
    grad_accum = int(getattr(config, "gradient_accumulation_steps", 1))
    effective_batch_size = per_device_batch_size * num_gpus * grad_accum
    effective_distributed = runtime["distributed"] if num_gpus > 1 else "none"
    setattr(config, "effective_batch_size", effective_batch_size)
    print(
        "[training] per_device_batch_size="
        f"{per_device_batch_size}, num_gpus={num_gpus}, "
        f"gradient_accumulation_steps={grad_accum}, "
        f"effective_batch_size={effective_batch_size}, "
        f"distributed={effective_distributed}"
    )
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        print(f"[training] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    allow_dist_single_fallback = _allow_distributed_to_single_gpu_fallback(config)
    if num_gpus > 1 and not allow_dist_single_fallback:
        print(
            "[training] strict multi-GPU mode enabled: distributed failures will not auto-fallback "
            "to single-GPU."
        )

    if not use_gpu:
        device = torch.device("cpu")
        train_loader, val_loader = get_dataloaders(config, use_gpu=False, rank=0, world_size=1)
        model = create_finetune_model(config).to(device)
        _configure_skip_offload(model, config)
        optimizer, scaler, scheduler = build_optimizer_scheduler(
            config, model, len(train_loader), use_gpu=False
        )
        loss_fn = build_loss_fn(config, list(config.data.output_vars))
        if hasattr(loss_fn, "describe"):
            print(f"[loss] active: {loss_fn.describe()}")
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
            loss_func=loss_fn,
        )
        return train_losses, val_losses

    if num_gpus == 1:
        mp.set_start_method("spawn", force=True)
        single_gpu_entry_visible = os.environ.get("CUDA_VISIBLE_DEVICES")

        visible_env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        visible_device_ids = _parse_visible_device_ids(visible_env) if visible_env else []
        tried: list[int | None] = []
        oom_errors: list[str] = []
        strict_visible_env = os.environ.get("CORDEX_STRICT_VISIBLE_DEVICES", "").strip().lower()
        if strict_visible_env in {"1", "true", "yes", "on"}:
            strict_visible = True
        elif strict_visible_env in {"0", "false", "no", "off"}:
            strict_visible = False
        else:
            # Respect explicit CUDA_VISIBLE_DEVICES pinning by default.
            strict_visible = bool(visible_env)
        if visible_env:
            if strict_visible:
                if visible_device_ids and len(visible_device_ids) > 0:
                    pinned_gpu = int(visible_device_ids[0])
                    gpu_candidates = [pinned_gpu]
                    print(
                        "Single-GPU retries are pinned to the first CUDA_VISIBLE_DEVICES entry "
                        f"(physical GPU {pinned_gpu}). Set CORDEX_STRICT_VISIBLE_DEVICES=0 "
                        "to allow retries across the visible list."
                    )
                else:
                    gpu_candidates = [None]
                    print(
                        "Single-GPU retries are pinned to the current CUDA_VISIBLE_DEVICES selection "
                        "(non-index identifier list)."
                    )
            else:
                allowed_gpu_ids = visible_device_ids if isinstance(visible_device_ids, list) else None
                gpu_candidates = [*(_single_gpu_retry_order(allowed_gpu_ids=allowed_gpu_ids))]
                if isinstance(visible_device_ids, list) and visible_device_ids:
                    print(
                        "CUDA_VISIBLE_DEVICES is set; single-GPU mode will retry only within that list "
                        "and start from the most-idle GPU. Set CORDEX_STRICT_VISIBLE_DEVICES=1 to pin "
                        "to the first visible entry."
                    )
                else:
                    print(
                        "CUDA_VISIBLE_DEVICES uses non-index identifiers; single-GPU mode will retry "
                        "by global GPU idleness ordering."
                    )
        else:
            gpu_candidates = [*(_single_gpu_retry_order())]
        visible_candidates = [candidate for candidate in gpu_candidates if candidate is not None]
        if visible_candidates:
            print(f"Single-GPU candidate order (physical IDs): {visible_candidates}")
        for gpu_id in gpu_candidates:
            if gpu_id in tried:
                continue
            tried.append(gpu_id)
            gpu_label = (
                str(gpu_id) if gpu_id is not None else "current CUDA_VISIBLE_DEVICES selection"
            )
            if gpu_id is not None:
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                print(
                    f"Single-GPU attempt on physical GPU {gpu_id} "
                    "(mapped to local cuda:0 inside the worker process)."
                )
            else:
                print("Single-GPU attempt using existing CUDA_VISIBLE_DEVICES selection.")

            manager = mp.Manager()
            try:
                return_dict = manager.dict()
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.ipc_collect()
                mp.spawn(
                    _single_gpu_worker,
                    args=(config, save_every, return_dict),
                    nprocs=1,
                    join=True,
                )
                result = return_dict.get("train_losses"), return_dict.get("val_losses")
                _set_cuda_visible_devices(single_gpu_entry_visible)
                return result
            except ProcessRaisedException as exc:
                message = str(exc).lower()
                if "out of memory" in message or "cuda oom" in message:
                    oom_errors.append(str(exc))
                    remaining = len([candidate for candidate in gpu_candidates if candidate not in tried])
                    if remaining > 0:
                        print(
                            f"CUDA OOM on GPU candidate {gpu_label}; "
                            "retrying on a different idle GPU if available."
                        )
                    else:
                        print(
                            f"CUDA OOM on GPU candidate {gpu_label}; "
                            "no unused GPU candidates remain for this backoff stage."
                        )
                    continue
                _set_cuda_visible_devices(single_gpu_entry_visible)
                raise
            except ProcessExitedException as exc:
                print(
                    f"Single-GPU run on GPU candidate {gpu_label} exited unexpectedly ({exc}); "
                    "trying another idle GPU if available."
                )
                continue
            finally:
                try:
                    manager.shutdown()
                except Exception:
                    pass

        if oom_errors:
            allow_multi_fallback = _allow_single_gpu_multi_fallback(
                config, available_gpus=available_gpus
            )
            if (
                allow_multi_fallback
                and not bool(getattr(config, "_single_gpu_oom_multi_retry_done", False))
            ):
                fallback = _choose_multi_gpu_oom_fallback(
                    available_gpus=available_gpus,
                    per_device_batch_size=per_device_batch_size,
                    target_effective_batch_size=max(1, effective_batch_size),
                )
                if fallback is not None:
                    fallback_num_gpus, fallback_grad_accum = fallback
                    _set_training_runtime_param(config, "num_gpus", fallback_num_gpus)
                    _set_training_runtime_param(config, "gradient_accumulation_steps", fallback_grad_accum)
                    setattr(config, "_single_gpu_oom_multi_retry_done", True)
                    adjusted_effective = per_device_batch_size * fallback_num_gpus * fallback_grad_accum
                    print(
                        "Single-GPU OOM across candidates; retrying with multi-GPU training: "
                        f"num_gpus={fallback_num_gpus}, per_device_batch_size={per_device_batch_size}, "
                        f"gradient_accumulation_steps={fallback_grad_accum}, "
                        f"effective_batch_size={adjusted_effective}."
                    )
                    _prepare_retry(
                        config, reason="retrying after single-GPU OOM escalation to multi-GPU"
                    )
                    _set_cuda_visible_devices(single_gpu_entry_visible)
                    return run_training(config, num_gpus=fallback_num_gpus, save_every=save_every)

            backoff_update = _apply_single_gpu_oom_backoff(
                config,
                num_gpus=1,
                target_effective_batch_size=max(1, effective_batch_size),
            )
            if backoff_update is not None:
                retries = int(getattr(config, "_single_gpu_oom_backoff_retries", 0))
                retry_limit = _resolve_single_gpu_oom_retry_limit(config)
                if retries >= retry_limit:
                    print(
                        "Single-GPU OOM across all candidates; "
                        f"memory backoff retry limit reached ({retry_limit})."
                    )
                    _set_cuda_visible_devices(single_gpu_entry_visible)
                    raise RuntimeError(
                        "Single-GPU training exhausted CUDA OOM retry budget. "
                        "Lower training crop size or effective batch size in the config."
                    )
                retries += 1
                setattr(config, "_single_gpu_oom_backoff_retries", retries)
                print(
                    "Single-GPU OOM across all candidates; applying memory backoff "
                    f"({retries}): {backoff_update}."
                )
                _prepare_retry(
                    config, reason="retrying after single-GPU OOM memory backoff"
                )
                _set_cuda_visible_devices(single_gpu_entry_visible)
                return run_training(config, num_gpus=1, save_every=save_every)

            if not allow_multi_fallback:
                print(
                    "Single-GPU OOM across all candidates. "
                    "Auto multi-GPU fallback is disabled "
                    "(set training.allow_multi_gpu_fallback=true or "
                    "CORDEX_ALLOW_MULTI_GPU_FALLBACK=1 to enable)."
                )
            _set_cuda_visible_devices(single_gpu_entry_visible)
            raise RuntimeError(
                "Single-GPU training encountered CUDA OOM on all available GPU candidates."
            )
        _set_cuda_visible_devices(single_gpu_entry_visible)
        raise RuntimeError("Single-GPU training failed on all available GPU candidates.")

    os.environ.setdefault("MASTER_ADDR", "localhost")

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    try:
        return_dict = manager.dict()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()

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
                    _set_training_distributed_mode(config, "fsdp")
                    if not getattr(config, "fsdp_precision", None):
                        setattr(
                            config,
                            "fsdp_precision",
                            "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp16",
                        )
                    _prepare_retry(
                        config, reason="retrying after multi-GPU DDP OOM with FSDP enabled"
                    )
                    return run_training(config, num_gpus=num_gpus, save_every=save_every)

                if not allow_dist_single_fallback:
                    raise RuntimeError(
                        "Encountered CUDA OOM during multi-GPU training and strict multi-GPU mode is enabled; "
                        "refusing automatic reduction to fewer GPUs."
                    ) from exc
                reduced = max(1, num_gpus // 2)
                if reduced == num_gpus:
                    reduced = num_gpus - 1
                print(
                    f"Encountered CUDA OOM during multi-GPU setup; retrying with {reduced} GPU(s)."
                )
                _prepare_retry(
                    config, reason=f"retrying after multi-GPU OOM with reduced GPU count ({reduced})"
                )
                return run_training(config, num_gpus=reduced, save_every=save_every)
            if num_gpus > 1:
                if not allow_dist_single_fallback:
                    raise RuntimeError(
                        "Distributed worker raised an exception and strict multi-GPU mode is enabled; "
                        "refusing automatic single-GPU fallback."
                    ) from exc
                print(
                    "Distributed worker raised an exception; retrying on a single GPU "
                    "from the latest checkpoint if available."
                )
                _prepare_retry(
                    config, reason="retrying after distributed worker exception on single GPU"
                )
                return run_training(config, num_gpus=1, save_every=save_every)
            raise
        except ProcessExitedException as exc:
            if num_gpus > 1 and not allow_dist_single_fallback:
                raise RuntimeError(
                    "Distributed training exited unexpectedly and strict multi-GPU mode is enabled; "
                    "refusing automatic single-GPU fallback."
                ) from exc
            print(f"Distributed training failed with signal/exit ({exc}); retrying on a single GPU.")
            _prepare_retry(
                config, reason="retrying after distributed process exit on single GPU"
            )
            return run_training(config, num_gpus=1, save_every=save_every)

        return return_dict.get("train_losses"), return_dict.get("val_losses")
    finally:
        try:
            manager.shutdown()
        except Exception:
            pass


def _configure_skip_offload(model: torch.nn.Module, config: ExperimentConfig):
    if not getattr(config, "skip_activation_offload", False):
        return
    if hasattr(model, "set_skip_activation_devices"):
        model.set_skip_activation_devices([torch.device("cpu")])
