"""Build a reusable deterministic Phase-1 and residual cache for NARR--PRISM.

The expensive deterministic Prithvi--UNet is frozen during stochastic Phase-2
training.  Re-running it for every tile, every epoch, and every refinement head
is unnecessary.  This command runs the exact Phase-1 ``last.ckpt`` once over
the training and validation dates, writes one atomic NetCDF file per day, and
fits training-only residual statistics that every compatible refinement head
can reuse.

Residuals use the repository's canonical definition::

    residual_target_normalized = encode(PRISM_target_physical)
                                 - encode(Phase1_prediction_physical)

The full-day Phase-1 field is produced with the same halo tiles and overlap
blend as deterministic inference.  The final blended physical field is encoded
once, ensuring ``decode(deterministic_normalized) == deterministic_physical``.

Examples
--------
Build on four GPUs (one independent date shard per process)::

    mamba run -n Prithvi python examples/NARR_PRISM/narr_prism_phase1_cache.py build \
      --config examples/NARR_PRISM/NARR_PRISM_diffusion_transformer.yaml \
      --checkpoint examples/NARR_PRISM/experiments/checkpoints/narr_prism_California/last.ckpt \
      --output-root examples/NARR_PRISM/experiments/phase1_residual_cache \
      --parallel-gpus 0,1,2,3

Interrupted runs are resumable: already complete, contract-compatible daily
files are validated and skipped.  A manifest is marked ``complete`` only after
the exact inventory and training-only residual statistics have been verified.
"""

from __future__ import annotations

import argparse
import atexit
import fcntl
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch

try:
    from netCDF4 import Dataset as NetCDFDataset
except ImportError:  # pragma: no cover - required by the Prithvi environment
    NetCDFDataset = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from narr_prism_dataset import NarrPrismDataset
from narr_prism_inference import (
    TARGET_VALID_MASK_CONTENT_SHA256_ATTR,
    TARGET_VALID_MASK_CRITERION_ATTR,
    TARGET_VALID_MASK_GRID_FINGERPRINT_ATTR,
    TARGET_VALID_MASK_SHA256_ATTR,
    TARGET_VALID_MASK_SOURCE_SPLIT_ATTR,
    TARGET_VALID_MASK_TRAINING_SOURCE_SIGNATURE_ATTR,
    _target_valid_mask_content_sha256,
    VAR_UNITS,
    _cap_tile_batch_for_cuda_indexing,
    _clear_cuda_memory,
    _load_model,
    _pad_multiple_from_config,
    _pad_to_multiple,
    _sanity_check_outputs,
    _target_valid_mask_provenance,
)
from narr_prism_utils import (
    date_range,
    get_case_name,
    load_yaml,
    parse_date_range_from_config,
    resolve_path,
)

from granitewxc.refinement.checkpoint import (
    extract_model_state,
    phase1_state_fingerprint,
)
from granitewxc.refinement.target_space import NormalizedTargetSpace
from granitewxc.utils.config import get_config
from granitewxc.utils.normalization import (
    apply_scalar_paths,
    assert_scalars_available,
    load_target_valid_mask,
    sha256_file,
)
from granitewxc.utils.prism_checkpoint import build_prism_checkpoint_contract
from granitewxc.utils.prism_grid import validate_prism_grid
from granitewxc.utils.prism_tiling import (
    TilePlan,
    WeightedTileStitcher,
    blend_window,
    extract_halo_context,
)

CACHE_SCHEMA = "narr_prism_phase1_residual_cache"
CACHE_SCHEMA_VERSION = 1
CACHE_STATE_INCOMPLETE = "incomplete"
CACHE_STATE_COMPLETE = "complete"
MANIFEST_NAME = "manifest.json"
DAILY_PREFIX = "narr_prism_phase1_residual_"
BUILD_LOCK_NAME = ".narr_prism_phase1_cache.lock"
WORKER_LEASE_NAME = ".narr_prism_phase1_cache.workers.lock"
DAILY_SUFFIX = ".nc"
REQUIRED_DAILY_VARIABLES = (
    "deterministic_physical",
    "deterministic_normalized",
    "residual_target_normalized",
    "residual_valid_mask",
    "prism_valid_mask",
)
DEFAULT_SPLITS = ("training", "validation")
TARGET_SPACE_NAME = "phase1_normalized_target_space"
RESIDUAL_DEFINITION = (
    "encode(PRISM_target_physical)-encode(Phase1_deterministic_physical)"
)
PHASE1_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def configure_phase1_cache_numerics() -> dict[str, Any]:
    """Apply the exact FP32 CUDA policy used to create cached Phase-1 fields.

    PyTorch's CUDA defaults are process-dependent.  In particular, cuDNN may
    enable TF32 even when the YAML requests FP32.  A live parity check must use
    the same backend policy as cache generation; otherwise identical weights
    and inputs can differ solely because a different convolution kernel was
    selected.
    """
    existing_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing_workspace not in (None, PHASE1_CUBLAS_WORKSPACE_CONFIG):
        raise RuntimeError(
            "The Phase-1 cache requires CUBLAS_WORKSPACE_CONFIG="
            f"{PHASE1_CUBLAS_WORKSPACE_CONFIG}, but this process has "
            f"{existing_workspace!r}. Start a fresh process with the cache "
            "workspace policy so live Phase-1 values remain reproducible."
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = PHASE1_CUBLAS_WORKSPACE_CONFIG
    torch.set_float32_matmul_precision("highest")
    if torch.backends.cuda.is_built():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    return {
        "dtype": "float32",
        "autocast": False,
        "allow_tf32": False,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _plain(value: Any) -> Any:
    """Convert config objects and NumPy values into canonical JSON values."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "__dict__"):
        return {
            str(key): _plain(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return value


class CacheBuildLockError(RuntimeError):
    """Raised when another parent process owns the cache build lock."""


class CacheBuildNotActiveError(RuntimeError):
    """Raised when an incomplete cache has no active parent or worker lease."""


class CacheBuildLock:
    """Parent-only advisory lock for one cache output root.

    The kernel releases flock automatically if the owner exits or crashes.
    Worker shards never instantiate this class; one parent holds it through
    manifest creation, daily generation, and finalization.
    """

    def __init__(
        self,
        output_root: str | os.PathLike,
        *,
        command: Sequence[str] | None = None,
    ) -> None:
        self.output_root = resolve_path(output_root)
        self.path = self.output_root / BUILD_LOCK_NAME
        self.command = list(command if command is not None else sys.argv)
        self._handle: Any | None = None
        self.owner: dict[str, Any] | None = None

    @staticmethod
    def _read_owner(handle: Any) -> dict[str, Any]:
        for _ in range(10):
            handle.seek(0)
            raw = handle.read().strip()
            if raw:
                try:
                    owner = json.loads(raw)
                except json.JSONDecodeError:
                    owner = None
                if isinstance(owner, dict) and owner.get("pid"):
                    return owner
            time.sleep(0.02)
        return {
            "pid": "unknown",
            "host": "unknown",
            "started_utc": "unknown",
        }

    @staticmethod
    def _write_owner(handle: Any, owner: Mapping[str, Any]) -> None:
        handle.seek(0)
        handle.truncate()
        json.dump(_plain(owner), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())

    def acquire(self) -> "CacheBuildLock":
        if self._handle is not None:
            raise RuntimeError(
                "Cache build lock is already held: {}".format(self.path)
            )
        self.output_root.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            owner = self._read_owner(handle)
            handle.close()
            command = owner.get("command", "unknown")
            raise CacheBuildLockError(
                "Another parent is already building this Phase-1 cache: "
                "lock={}, pid={}, host={}, started_utc={}, command={!r}. "
                "Wait for that build to finish or stop that PID cleanly, then "
                "rerun; completed daily files are resumable.".format(
                    self.path,
                    owner.get("pid", "unknown"),
                    owner.get("host", "unknown"),
                    owner.get("started_utc", "unknown"),
                    command,
                )
            ) from exc
        except BaseException:
            handle.close()
            raise
        self.owner = {
            "state": "active",
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_utc": _utc_now(),
            "command": self.command,
            "cwd": os.getcwd(),
        }
        self._write_owner(handle, self.owner)
        self._handle = handle
        return self

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        try:
            released = dict(self.owner or {})
            released["state"] = "released"
            released["released_utc"] = _utc_now()
            self._write_owner(handle, released)
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._handle = None

    def __enter__(self) -> "CacheBuildLock":
        return self.acquire()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release()


class CacheWorkerLeaseError(RuntimeError):
    """Raised while orphaned parallel cache workers still own their lease."""


class CacheWorkerLease:
    """Shared lease inherited by every worker child across exec.

    A new parent first requests an exclusive lease. Existing orphan workers
    retain a shared lock after a parent SIGKILL, so this exclusive probe fails.
    The successful parent then downgrades to shared and passes the descriptor to
    every child; the kernel releases it only after the parent and all children
    close their inherited descriptors.
    """

    def __init__(
        self,
        cache_dir: str | os.PathLike,
        *,
        command: Sequence[str] | None = None,
    ) -> None:
        self.cache_dir = resolve_path(cache_dir)
        self.path = self.cache_dir / WORKER_LEASE_NAME
        self.command = list(command if command is not None else sys.argv)
        self._handle: Any | None = None

    def acquire(self) -> "CacheWorkerLease":
        if self._handle is not None:
            raise RuntimeError(
                "Cache worker lease is already held: {}".format(self.path)
            )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            owner = CacheBuildLock._read_owner(handle)
            handle.close()
            raise CacheWorkerLeaseError(
                "Orphaned or active Phase-1 cache workers still own "
                "digest lease {} (parent_pid={}, host={}, started_utc={}, "
                "command={!r}). Wait for those workers to exit or stop them "
                "cleanly before resuming; this prevents duplicate daily writers.".format(
                    self.path,
                    owner.get("pid", "unknown"),
                    owner.get("host", "unknown"),
                    owner.get("started_utc", "unknown"),
                    owner.get("command", "unknown"),
                )
            ) from exc
        except BaseException:
            handle.close()
            raise
        owner = {
            "state": "active",
            "parent_pid": os.getpid(),
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_utc": _utc_now(),
            "command": self.command,
            "cwd": os.getcwd(),
        }
        CacheBuildLock._write_owner(handle, owner)
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        self._handle = handle
        return self

    def fileno(self) -> int:
        if self._handle is None:
            raise RuntimeError("Cache worker lease has not been acquired")
        return int(self._handle.fileno())

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        handle.close()
        self._handle = None

    def __enter__(self) -> "CacheWorkerLease":
        return self.acquire()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release()


def _probe_advisory_lock(path: Path) -> tuple[bool, dict[str, Any]]:
    """Return whether an existing advisory lock is held without changing it."""
    if not path.is_file():
        return False, {}
    with path.open("r", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True, CacheBuildLock._read_owner(handle)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return False, CacheBuildLock._read_owner(handle)


def cache_build_activity(
    output_root: str | os.PathLike,
    *,
    cache_dir: str | os.PathLike | None = None,
) -> dict[str, Any]:
    """Inspect the authoritative parent lock and optional worker lease.

    Lock ownership is determined with ``flock``. JSON lock-file metadata is
    informative only and is never treated as proof that a process is active.
    This function is read-only and does not create either lock file.
    """
    root = resolve_path(output_root)
    parent_active, parent_owner = _probe_advisory_lock(root / BUILD_LOCK_NAME)
    workers_active = False
    worker_owner: dict[str, Any] = {}
    if cache_dir is not None:
        directory = resolve_path(cache_dir)
        workers_active, worker_owner = _probe_advisory_lock(
            directory / WORKER_LEASE_NAME
        )
    return {
        "output_root": str(root),
        "parent_active": parent_active,
        "parent_owner": parent_owner,
        "workers_active": workers_active,
        "worker_owner": worker_owner,
        "active": parent_active or workers_active,
    }


def contract_digest(contract: Mapping[str, Any]) -> str:
    """Return the canonical SHA-256 identity for a cache contract."""
    encoded = json.dumps(
        _plain(contract),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_identity(checkpoint: str | os.PathLike) -> dict[str, Any]:
    path = resolve_path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(
            f"Phase-1 checkpoint is not a regular file: {path}"
        )
    digest = sha256_file(path)
    return {
        "file_name": path.name,
        "size_bytes": int(path.stat().st_size),
        "sha256": digest,
    }


def checkpoint_state_fingerprint(checkpoint: str | os.PathLike) -> str:
    """Return a canonical fingerprint of the Phase-1 tensor state.

    This complements the exact checkpoint-file SHA-256: the state fingerprint
    remains stable across harmless envelope moves while the file SHA binds the
    exact metadata-bearing ``last.ckpt`` artifact.
    """
    path = resolve_path(checkpoint)
    payload = torch.load(
        path, map_location="cpu", mmap=True, weights_only=False
    )
    return phase1_state_fingerprint(extract_model_state(payload))


def _configured_dates(cfg: Mapping[str, Any], split: str) -> list[str]:
    start, end = parse_date_range_from_config(dict(cfg), split)
    return [value.isoformat() for value in date_range(start, end)]


def _pair_from_config(
    data: Mapping[str, Any], prefix: str, default: int
) -> list[int]:
    return [
        int(data.get(f"{prefix}_lat", default)),
        int(data.get(f"{prefix}_lon", default)),
    ]


def _cache_geometry(
    cfg: Mapping[str, Any], grid_shape: Sequence[int]
) -> dict[str, Any]:
    data = cfg.get("data", {}) or {}
    inference = cfg.get("inference", {}) or {}
    core = _pair_from_config(data, "train_crop_size", 256)
    stride = _pair_from_config(data, "training_tile_stride", core[0])
    halo = _pair_from_config(data, "training_halo", 0)
    if (
        str(data.get("training_spatial_sampling", "random")).strip().lower()
        != "tiled"
    ):
        raise ValueError(
            "The reusable daily Phase-1 cache requires "
            "data.training_spatial_sampling='tiled'; random crops do not have a "
            "stable cache/inventory contract."
        )
    if any(step <= 0 for step in stride) or any(size <= 0 for size in core):
        raise ValueError(
            f"Invalid training core/stride: core={core}, stride={stride}"
        )
    overlap = [core[index] - stride[index] for index in range(2)]
    if any(value < 0 for value in overlap):
        raise ValueError(
            f"Training stride {stride} exceeds core {core}; the daily cache would "
            "leave gaps."
        )
    blend_mode = (
        str(
            inference.get(
                "inference_blend_window",
                inference.get("blend_window", "hann"),
            )
        )
        .strip()
        .lower()
    )
    plan = TilePlan.build(tuple(grid_shape), core, overlap=overlap, halo=halo)
    return {
        "domain_shape": [int(value) for value in grid_shape],
        "core_shape": list(plan.core_shape),
        "stride": stride,
        "overlap": list(plan.overlap),
        "halo": list(plan.halo),
        "blend_mode": blend_mode,
        "skip_empty_target_tiles": bool(
            data.get("skip_empty_target_tiles", True)
        ),
        "min_valid_target_fraction": float(
            data.get("min_valid_target_fraction", 1.0e-4)
        ),
    }


def cache_contract(
    cfg: Mapping[str, Any],
    config: Any,
    phase1_checkpoint: str | os.PathLike,
    *,
    grid_shape: Sequence[int] | None = None,
    phase1_fingerprint: str | None = None,
    splits: Sequence[str] = DEFAULT_SPLITS,
) -> dict[str, Any]:
    """Build the immutable scientific contract that keys a cache directory."""
    checkpoint = _checkpoint_identity(phase1_checkpoint)
    if phase1_fingerprint is None:
        phase1_fingerprint = checkpoint_state_fingerprint(phase1_checkpoint)
    checkpoint["phase1_fingerprint"] = str(phase1_fingerprint)
    checkpoint["fingerprint_algorithm"] = "phase1_state_sha256"
    pipeline = build_prism_checkpoint_contract(config)
    if pipeline is None:
        raise ValueError(
            "The selected config is not a PRISM pipeline configuration."
        )
    grid_entry = (pipeline.get("coordinates_and_artifacts") or {}).get(
        "grid"
    ) or {}
    if grid_shape is None:
        grid_shape = grid_entry.get("shape")
    if not grid_shape or len(grid_shape) != 2:
        raise ValueError("Cannot determine the canonical PRISM grid shape.")
    target_variables = list(
        (pipeline.get("channels") or {}).get("target_variables") or []
    )
    if not target_variables:
        raise ValueError("The config has no ordered target variables.")
    normalized_splits = tuple(str(value).strip().lower() for value in splits)
    unknown = sorted(set(normalized_splits) - set(DEFAULT_SPLITS))
    if unknown:
        raise ValueError(f"Unsupported cache split(s): {unknown}")
    if len(set(normalized_splits)) != len(normalized_splits):
        raise ValueError("Cache splits must not contain duplicates.")
    refinement = ((cfg.get("model") or {}).get("refinement") or {})
    residual_normalization = refinement.get("residual_normalization") or {}
    if not bool(residual_normalization.get("enabled", False)):
        raise ValueError(
            "The reusable residual cache requires enabled residual normalization."
        )
    residual_epsilon = float(residual_normalization.get("epsilon", 0.0))
    if not np.isfinite(residual_epsilon) or residual_epsilon <= 0.0:
        raise ValueError(
            "model.refinement.residual_normalization.epsilon must be finite "
            "and positive."
        )

    return {
        "schema": CACHE_SCHEMA,
        "schema_version": CACHE_SCHEMA_VERSION,
        "case_name": get_case_name(cfg),
        "phase1_checkpoint": checkpoint,
        "prism_pipeline_contract": _plain(pipeline),
        "target_variables": target_variables,
        "split_dates": {
            split: {
                "start": _configured_dates(cfg, split)[0],
                "end": _configured_dates(cfg, split)[-1],
                "count": len(_configured_dates(cfg, split)),
            }
            for split in normalized_splits
        },
        "cache_geometry": _cache_geometry(cfg, grid_shape),
        "precision": {
            "dtype": "float32",
            "autocast": False,
            "allow_tf32": False,
        },
        "baseline_semantics": (
            "overlap_weighted_Phase1_physical_then_encoded_once"
        ),
        "target_space": TARGET_SPACE_NAME,
        "residual_definition": RESIDUAL_DEFINITION,
        "invalid_residual_fill": 0.0,
        "statistics": {
            "fit_split": "training",
            "weighting": "training_tile_overlap_multiplicity",
            "variance": "population",
            "residual_normalization_enabled": True,
            "residual_normalization_epsilon": residual_epsilon,
        },
    }


def cache_directory(cache_root: str | os.PathLike, digest: str) -> Path:
    """Return the digest-scoped cache directory under ``cache_root``."""
    return resolve_path(cache_root) / str(digest)[:32]


def daily_cache_path(
    cache_dir: str | os.PathLike, split: str, sample_date: Any
) -> Path:
    token = str(sample_date)[:10].replace("-", "")
    if len(token) != 8 or not token.isdigit():
        raise ValueError(f"Invalid cache date: {sample_date!r}")
    normalized_split = str(split).strip().lower()
    if normalized_split not in DEFAULT_SPLITS:
        raise ValueError(f"Unsupported cache split: {split!r}")
    return (
        Path(cache_dir)
        / normalized_split
        / f"{DAILY_PREFIX}{token}{DAILY_SUFFIX}"
    )


def _manifest_splits(
    cfg: Mapping[str, Any], splits: Sequence[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in splits:
        dates = _configured_dates(cfg, split)
        result[split] = {
            "start": dates[0],
            "end": dates[-1],
            "expected_count": len(dates),
            "expected_dates": dates,
            "directory": split,
            "file_pattern": f"{DAILY_PREFIX}YYYYMMDD{DAILY_SUFFIX}",
            "completed_count": 0,
            "inventory_digest_sha256": None,
        }
    return result


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(_plain(payload), handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _new_manifest(
    cfg: Mapping[str, Any], contract: Mapping[str, Any], splits: Sequence[str]
) -> dict[str, Any]:
    now = _utc_now()
    digest = contract_digest(contract)
    return {
        "schema": CACHE_SCHEMA,
        "schema_version": CACHE_SCHEMA_VERSION,
        "state": CACHE_STATE_INCOMPLETE,
        "contract_digest": digest,
        "contract": _plain(contract),
        "created_utc": now,
        "updated_utc": now,
        "splits": _manifest_splits(cfg, splits),
        "residual_normalization": None,
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _assert_manifest_schema(manifest: Mapping[str, Any], path: Path) -> None:
    if manifest.get("schema") != CACHE_SCHEMA:
        raise RuntimeError(
            f"Unsupported cache schema in {path}: {manifest.get('schema')!r}"
        )
    if int(manifest.get("schema_version", -1)) != CACHE_SCHEMA_VERSION:
        raise RuntimeError(
            f"Cache schema version in {path} is {manifest.get('schema_version')!r}; "
            f"expected {CACHE_SCHEMA_VERSION}. Rebuild the cache."
        )
    contract = manifest.get("contract")
    if not isinstance(contract, Mapping):
        raise RuntimeError(f"Cache manifest {path} has no contract mapping.")
    computed = contract_digest(contract)
    if manifest.get("contract_digest") != computed:
        raise RuntimeError(
            f"Cache manifest {path} contract digest is invalid: "
            f"stored={manifest.get('contract_digest')}, computed={computed}."
        )


def _resolve_manifest_path(
    root_or_path: str | os.PathLike,
    expected_digest: str | None = None,
) -> Path:
    path = resolve_path(root_or_path)
    if path.is_file():
        if path.name != MANIFEST_NAME:
            raise ValueError(f"Expected {MANIFEST_NAME}, got {path}")
        return path
    direct = path / MANIFEST_NAME
    if direct.is_file():
        return direct
    if expected_digest is not None:
        expected = cache_directory(path, expected_digest) / MANIFEST_NAME
        if expected.is_file():
            return expected
        raise FileNotFoundError(f"No compatible cache manifest at {expected}")
    candidates = (
        sorted(path.glob(f"*/{MANIFEST_NAME}")) if path.is_dir() else []
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No {MANIFEST_NAME} found under {path}")
    raise RuntimeError(
        f"Multiple cache manifests exist under {path}; provide the digest directory "
        "or config/checkpoint needed to resolve one exactly."
    )


def _daily_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _hash_array(digest: Any, name: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(name.encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.view(np.uint8).tobytes())


def daily_content_digest(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash all scientific arrays in a daily cache artifact."""
    digest = hashlib.sha256()
    for name in ("lat", "lon", *REQUIRED_DAILY_VARIABLES):
        if name not in arrays:
            raise KeyError(f"Daily content is missing {name!r}")
        _hash_array(digest, name, np.asarray(arrays[name]))
    return digest.hexdigest()


def validate_daily_cache(
    path: str | os.PathLike,
    expected_manifest: Mapping[str, Any],
    *,
    expected_split: str | None = None,
    expected_date: Any | None = None,
    verify_content: bool = False,
) -> dict[str, Any]:
    """Validate one daily NetCDF against an already validated manifest."""
    if NetCDFDataset is None:
        raise ImportError(
            "netCDF4 is required to validate Phase-1 cache files"
        )
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Missing daily Phase-1 cache file: {file_path}"
        )
    contract = expected_manifest["contract"]
    variables = list(contract["target_variables"])
    shape = tuple(
        int(value) for value in contract["cache_geometry"]["domain_shape"]
    )
    with NetCDFDataset(file_path, "r") as dataset:
        attrs = {
            name: _daily_attr(dataset.getncattr(name))
            for name in dataset.ncattrs()
        }
        if attrs.get("cache_schema") != CACHE_SCHEMA:
            raise RuntimeError(
                f"{file_path} has wrong cache schema {attrs.get('cache_schema')!r}"
            )
        if int(attrs.get("cache_schema_version", -1)) != CACHE_SCHEMA_VERSION:
            raise RuntimeError(
                f"{file_path} has an incompatible cache schema version"
            )
        if (
            attrs.get("cache_contract_digest")
            != expected_manifest["contract_digest"]
        ):
            raise RuntimeError(
                f"{file_path} was built for a different cache contract"
            )
        if expected_split is not None and attrs.get("split") != str(
            expected_split
        ):
            raise RuntimeError(
                f"{file_path} split={attrs.get('split')!r}, expected {expected_split!r}"
            )
        normalized_date = (
            str(expected_date)[:10] if expected_date is not None else None
        )
        if (
            normalized_date is not None
            and attrs.get("date") != normalized_date
        ):
            raise RuntimeError(
                f"{file_path} date={attrs.get('date')!r}, expected {normalized_date!r}"
            )
        try:
            stored_variables = json.loads(str(attrs["target_variables_json"]))
        except (KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"{file_path} has invalid target-variable metadata"
            ) from exc
        if stored_variables != variables:
            raise RuntimeError(
                f"{file_path} variable order {stored_variables} != {variables}"
            )
        try:
            stored_units = json.loads(str(attrs["target_units_json"]))
        except (KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"{file_path} has invalid target-unit metadata"
            ) from exc
        expected_units = [VAR_UNITS.get(name, "") for name in variables]
        if stored_units != expected_units:
            raise RuntimeError(
                f"{file_path} target units {stored_units} != {expected_units}"
            )
        semantic_attrs = {
            "target_space": contract["target_space"],
            "residual_definition": contract["residual_definition"],
            "baseline_semantics": contract["baseline_semantics"],
        }
        for attr_name, expected_value in semantic_attrs.items():
            if attrs.get(attr_name) != expected_value:
                raise RuntimeError(
                    f"{file_path} {attr_name}={attrs.get(attr_name)!r}, "
                    f"expected {expected_value!r}"
                )
        expected_channel_dims = ("channel", "lat", "lon")
        for name in REQUIRED_DAILY_VARIABLES[:4]:
            observed_dims = tuple(dataset.variables[name].dimensions)
            if observed_dims != expected_channel_dims:
                raise RuntimeError(
                    f"{file_path}:{name} dimensions {observed_dims} != "
                    f"{expected_channel_dims}"
                )
        static_dims = tuple(dataset.variables["prism_valid_mask"].dimensions)
        if static_dims != ("lat", "lon"):
            raise RuntimeError(
                f"{file_path}:prism_valid_mask dimensions {static_dims} are invalid"
            )
        for axis_name in ("lat", "lon"):
            axis_dims = tuple(dataset.variables[axis_name].dimensions)
            if axis_dims != (axis_name,):
                raise RuntimeError(
                    f"{file_path}:{axis_name} dimensions {axis_dims} are invalid"
                )
        channel_variable = dataset.variables.get("channel")
        if channel_variable is None:
            raise RuntimeError(f"{file_path} lacks the channel coordinate")
        if tuple(channel_variable.dimensions) != ("channel",):
            raise RuntimeError(
                f"{file_path}:channel has invalid dimensions "
                f"{channel_variable.dimensions}"
            )
        channel_values = np.asarray(channel_variable[:])
        expected_channels = np.arange(len(variables), dtype=np.int32)
        if not np.array_equal(channel_values, expected_channels):
            raise RuntimeError(
                f"{file_path}:channel values {channel_values.tolist()} != "
                f"{expected_channels.tolist()}"
            )
        missing = [
            name
            for name in REQUIRED_DAILY_VARIABLES
            if name not in dataset.variables
        ]
        if missing:
            raise RuntimeError(f"{file_path} is missing variables: {missing}")
        lat_values = np.asarray(dataset.variables["lat"][:], dtype=np.float64)
        lon_values = np.asarray(dataset.variables["lon"][:], dtype=np.float64)
        observed_grid = validate_prism_grid(
            lat_values,
            lon_values,
            context=f"{file_path} daily cache grid",
        ).manifest_entry()
        expected_grid = (
            (contract.get("prism_pipeline_contract") or {})
            .get("coordinates_and_artifacts", {})
            .get("grid")
        )
        if expected_grid is not None and observed_grid != expected_grid:
            raise RuntimeError(
                f"{file_path} grid contract {observed_grid} != {expected_grid}"
            )

        static_mask = np.asarray(
            dataset.variables["prism_valid_mask"][:],
            dtype=np.uint8,
        )
        residual_mask = np.asarray(
            dataset.variables["residual_valid_mask"][:],
            dtype=np.uint8,
        )
        if not np.isin(static_mask, (0, 1)).all():
            raise RuntimeError(f"{file_path}:prism_valid_mask is not boolean")
        if not np.isin(residual_mask, (0, 1)).all():
            raise RuntimeError(f"{file_path}:residual_valid_mask is not boolean")
        if np.any(residual_mask.astype(bool) & ~static_mask.astype(bool)[None]):
            raise RuntimeError(
                f"{file_path}:residual_valid_mask is valid outside static support"
            )

        mask_entry = (
            (contract.get("prism_pipeline_contract") or {})
            .get("coordinates_and_artifacts", {})
            .get("target_valid_mask")
        ) or {}
        training_signature = mask_entry.get(
            "training_source_artifact_split_signature"
        )
        expected_mask_attrs = {
            TARGET_VALID_MASK_SHA256_ATTR: mask_entry.get("sha256"),
            TARGET_VALID_MASK_CRITERION_ATTR: mask_entry.get("criterion"),
            TARGET_VALID_MASK_SOURCE_SPLIT_ATTR: mask_entry.get("source_split"),
            TARGET_VALID_MASK_GRID_FINGERPRINT_ATTR: mask_entry.get(
                "grid_fingerprint"
            ),
            TARGET_VALID_MASK_TRAINING_SOURCE_SIGNATURE_ATTR: (
                str(training_signature)
                if training_signature is not None
                else "not-applicable:raw-training-source-mode"
            ),
        }
        if mask_entry:
            for attr_name, expected_value in expected_mask_attrs.items():
                if attrs.get(attr_name) != expected_value:
                    raise RuntimeError(
                        f"{file_path} {attr_name}={attrs.get(attr_name)!r}, "
                        f"expected {expected_value!r}"
                    )
        expected_mask_content = _target_valid_mask_content_sha256(static_mask)
        if attrs.get(TARGET_VALID_MASK_CONTENT_SHA256_ATTR) != expected_mask_content:
            raise RuntimeError(
                f"{file_path} target-valid-mask content hash is invalid"
            )
        for name in REQUIRED_DAILY_VARIABLES[:4]:
            observed = tuple(dataset.variables[name].shape)
            expected = (len(variables), *shape)
            if observed != expected:
                raise RuntimeError(
                    f"{file_path}:{name} shape {observed} != {expected}"
                )
        if tuple(dataset.variables["prism_valid_mask"].shape) != shape:
            raise RuntimeError(
                f"{file_path}:prism_valid_mask has the wrong shape"
            )
        if tuple(dataset.variables["lat"].shape) != (shape[0],):
            raise RuntimeError(f"{file_path}:lat has the wrong shape")
        if tuple(dataset.variables["lon"].shape) != (shape[1],):
            raise RuntimeError(f"{file_path}:lon has the wrong shape")
        stored_content = str(attrs.get("content_sha256", ""))
        if len(stored_content) != 64:
            raise RuntimeError(
                f"{file_path} lacks a valid content_sha256 attribute"
            )
        if verify_content:
            arrays = {
                name: np.asarray(dataset.variables[name][:])
                for name in ("lat", "lon", *REQUIRED_DAILY_VARIABLES)
            }
            observed_digest = daily_content_digest(arrays)
            if stored_content != observed_digest:
                raise RuntimeError(
                    f"{file_path} content digest mismatch: stored={stored_content}, "
                    f"computed={observed_digest}"
                )
    return {
        "path": str(file_path),
        "split": attrs.get("split"),
        "date": attrs.get("date"),
        "content_sha256": stored_content,
    }


def _inventory(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    *,
    verify_content: bool = False,
    show_progress: bool = False,
) -> dict[str, Any]:
    cache_dir = manifest_path.parent
    result: dict[str, Any] = {}
    for split, record in manifest["splits"].items():
        digest = hashlib.sha256()
        count = 0
        dates: Any = record["expected_dates"]
        if show_progress and tqdm is not None:
            dates = tqdm(
                dates,
                desc=f"Validate Phase1 cache ({split})",
                unit="day",
            )
        for sample_date in dates:
            path = daily_cache_path(cache_dir, split, sample_date)
            metadata = validate_daily_cache(
                path,
                manifest,
                expected_split=split,
                expected_date=sample_date,
                verify_content=verify_content,
            )
            digest.update(sample_date.encode("ascii"))
            digest.update(metadata["content_sha256"].encode("ascii"))
            count += 1
        result[split] = {
            "completed_count": count,
            "inventory_digest_sha256": digest.hexdigest(),
        }
    return result


def load_and_validate_manifest(
    root_or_configured_path: str | os.PathLike,
    *,
    cfg: Mapping[str, Any] | None = None,
    config: Any | None = None,
    phase1_checkpoint: str | os.PathLike | None = None,
    phase1_fingerprint: str | None = None,
    require_complete: bool = True,
    validate_inventory: bool = True,
    verify_daily_content: bool = False,
    show_inventory_progress: bool = False,
) -> dict[str, Any]:
    """Load and fail-closed validate a cache manifest.

    ``root_or_configured_path`` may be a cache root, digest directory, or direct
    manifest path.  Passing ``cfg``, ``config`` and ``phase1_checkpoint`` rebuilds
    the expected immutable contract and selects the exact digest directory.
    Expensive checkpoint hashing occurs once here, never in DataLoader workers.
    """
    expected_contract = None
    expected_digest = None
    if any(value is not None for value in (cfg, config, phase1_checkpoint)):
        if cfg is None or config is None or phase1_checkpoint is None:
            raise ValueError(
                "cfg, config, and phase1_checkpoint must be supplied together for "
                "contract validation."
            )
        # Discover requested split names from an existing direct manifest when
        # possible; otherwise use the production training+validation default.
        splits = DEFAULT_SPLITS
        direct = resolve_path(root_or_configured_path)
        direct_manifest = (
            direct if direct.name == MANIFEST_NAME else direct / MANIFEST_NAME
        )
        if direct_manifest.is_file():
            raw = _read_json(direct_manifest)
            if isinstance(raw.get("splits"), Mapping):
                splits = tuple(raw["splits"])
        expected_contract = cache_contract(
            cfg,
            config,
            phase1_checkpoint,
            phase1_fingerprint=phase1_fingerprint,
            splits=splits,
        )
        expected_digest = contract_digest(expected_contract)
    manifest_path = _resolve_manifest_path(
        root_or_configured_path, expected_digest
    )
    manifest = _read_json(manifest_path)
    _assert_manifest_schema(manifest, manifest_path)
    if expected_contract is not None and manifest["contract"] != _plain(
        expected_contract
    ):
        raise RuntimeError(
            f"Cache manifest {manifest_path} does not match the active Phase-1/config contract."
        )
    if phase1_fingerprint is not None:
        observed = manifest["contract"]["phase1_checkpoint"].get(
            "phase1_fingerprint"
        )
        if observed != str(phase1_fingerprint):
            raise RuntimeError(
                f"Cache Phase-1 fingerprint {observed!r} != {phase1_fingerprint!r}"
            )
    if require_complete and manifest.get("state") != CACHE_STATE_COMPLETE:
        raise RuntimeError(
            f"Cache manifest {manifest_path} is {manifest.get('state')!r}, not complete. "
            "Resume the build and finalize it before training."
        )
    if validate_inventory:
        observed_inventory = _inventory(
            manifest,
            manifest_path,
            verify_content=verify_daily_content,
            show_progress=show_inventory_progress,
        )
        if manifest.get("state") == CACHE_STATE_COMPLETE:
            for split, observed in observed_inventory.items():
                recorded = manifest["splits"][split]
                if (
                    int(recorded.get("completed_count", -1))
                    != observed["completed_count"]
                ):
                    raise RuntimeError(
                        f"Cache {split} completed-count metadata is invalid"
                    )
                if (
                    recorded.get("inventory_digest_sha256")
                    != observed["inventory_digest_sha256"]
                ):
                    raise RuntimeError(
                        f"Cache {split} inventory digest is invalid"
                    )
    # Non-serialized convenience field used by the picklable reader.
    manifest = dict(manifest)
    manifest["_manifest_path"] = str(manifest_path)
    return manifest


def load_daily_cache(
    path: str | os.PathLike,
    expected_manifest: Mapping[str, Any],
    *,
    validate: bool = True,
    variables: Sequence[str] = (
        "deterministic_normalized",
        "residual_target_normalized",
        "residual_valid_mask",
    ),
) -> dict[str, np.ndarray]:
    """Load selected arrays without rehashing the checkpoint or full inventory."""
    if validate:
        validate_daily_cache(path, expected_manifest)
    requested = tuple(str(name) for name in variables)
    unknown = sorted(set(requested) - set(REQUIRED_DAILY_VARIABLES))
    if unknown:
        raise ValueError(f"Unknown daily cache variables requested: {unknown}")
    with NetCDFDataset(path, "r") as dataset:
        return {
            name: np.asarray(dataset.variables[name][:]) for name in requested
        }


@dataclass
class Phase1ResidualCacheReader:
    """Picklable, worker-safe reader for cropped daily Phase-1 cache arrays."""

    manifest: dict[str, Any]
    validate_daily: bool = True
    _cached_path: str | None = field(default=None, init=False, repr=False)
    _cached_arrays: dict[str, np.ndarray] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        manifest_path = self.manifest.get("_manifest_path")
        if not manifest_path:
            raise ValueError(
                "Reader manifest must contain the resolved _manifest_path"
            )
        self.cache_dir = str(Path(manifest_path).parent)

    @classmethod
    def from_path(
        cls,
        root_or_manifest: str | os.PathLike,
        *,
        validate_inventory: bool = True,
        **validation_kwargs: Any,
    ) -> "Phase1ResidualCacheReader":
        manifest = load_and_validate_manifest(
            root_or_manifest,
            validate_inventory=validate_inventory,
            **validation_kwargs,
        )
        return cls(manifest)

    def path_for(self, split: str, sample_date: Any) -> Path:
        return daily_cache_path(self.cache_dir, split, sample_date)

    def _load_day(
        self,
        split: str,
        sample_date: Any,
        *,
        include_physical: bool = False,
    ) -> dict[str, np.ndarray]:
        """Open/decompress one daily file once per DataLoader worker.

        Tiled training enumerates every crop for one date consecutively. The
        one-entry LRU therefore turns same-day tile reads into one NetCDF open
        without allowing an unbounded per-worker cache.
        """
        path = self.path_for(split, sample_date)
        path_text = str(path)
        cache_hit = self._cached_path == path_text
        if cache_hit and (
            not include_physical
            or "deterministic_physical" in self._cached_arrays
        ):
            return self._cached_arrays
        with NetCDFDataset(path, "r") as dataset:
            if self.validate_daily:
                attrs = {
                    name: _daily_attr(dataset.getncattr(name))
                    for name in dataset.ncattrs()
                }
                if (
                    attrs.get("cache_contract_digest")
                    != self.manifest["contract_digest"]
                ):
                    raise RuntimeError(
                        f"{path} was built for a different cache contract"
                    )
                if (
                    attrs.get("split") != str(split)
                    or attrs.get("date") != str(sample_date)[:10]
                ):
                    raise RuntimeError(
                        f"{path} split/date metadata is incompatible"
                    )
            names = (
                ("deterministic_physical",)
                if cache_hit
                else (
                    "deterministic_normalized",
                    "residual_target_normalized",
                    "residual_valid_mask",
                    *(("deterministic_physical",) if include_physical else ()),
                )
            )
            arrays = {
                name: np.asarray(dataset.variables[name][:]).copy()
                for name in names
            }
        if cache_hit:
            self._cached_arrays.update(arrays)
            return self._cached_arrays
        self._cached_path = path_text
        self._cached_arrays = arrays
        return arrays

    def load_crop(
        self,
        split: str,
        sample_date: Any,
        y0: int,
        x0: int,
        height: int,
        width: int,
        *,
        include_physical: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Load one spatial crop as CPU tensors for DataLoader collation."""
        shape = tuple(
            int(value)
            for value in self.manifest["contract"]["cache_geometry"][
                "domain_shape"
            ]
        )
        y0, x0, height, width = map(int, (y0, x0, height, width))
        if min(y0, x0, height, width) < 0 or height == 0 or width == 0:
            raise ValueError(
                "Cache crop origin/shape must be non-negative and non-empty"
            )
        if y0 + height > shape[0] or x0 + width > shape[1]:
            raise ValueError(
                f"Cache crop {(y0, x0, height, width)} exceeds domain {shape}"
            )
        arrays = self._load_day(
            split, sample_date, include_physical=include_physical
        )
        result = {
            "__phase1_normalized": torch.from_numpy(
                arrays["deterministic_normalized"][
                    :, y0 : y0 + height, x0 : x0 + width
                ].copy()
            ),
            "__residual_target_normalized": torch.from_numpy(
                arrays["residual_target_normalized"][
                    :, y0 : y0 + height, x0 : x0 + width
                ].copy()
            ),
            "__residual_valid_mask": torch.from_numpy(
                arrays["residual_valid_mask"][
                    :, y0 : y0 + height, x0 : x0 + width
                ].copy()
            ).to(torch.bool),
        }
        if include_physical:
            result["__phase1_physical"] = torch.from_numpy(
                arrays["deterministic_physical"][
                    :, y0 : y0 + height, x0 : x0 + width
                ].copy()
            )
        return result


def _write_daily_cache(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    split: str,
    sample_date: Any,
    lat: np.ndarray,
    lon: np.ndarray,
    deterministic_physical: np.ndarray,
    deterministic_normalized: np.ndarray,
    residual_target_normalized: np.ndarray,
    residual_valid_mask: np.ndarray,
    prism_valid_mask: np.ndarray,
    mask_provenance: Mapping[str, Any],
) -> str:
    if NetCDFDataset is None:
        raise ImportError("netCDF4 is required to write Phase-1 cache files")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    arrays = {
        "lat": np.asarray(lat, dtype=np.float64),
        "lon": np.asarray(lon, dtype=np.float64),
        "deterministic_physical": np.asarray(
            deterministic_physical, dtype=np.float32
        ),
        "deterministic_normalized": np.asarray(
            deterministic_normalized, dtype=np.float32
        ),
        "residual_target_normalized": np.asarray(
            residual_target_normalized, dtype=np.float32
        ),
        "residual_valid_mask": np.asarray(residual_valid_mask, dtype=np.uint8),
        "prism_valid_mask": np.asarray(prism_valid_mask, dtype=np.uint8),
    }
    content_sha = daily_content_digest(arrays)
    variables = list(manifest["contract"]["target_variables"])
    units = [VAR_UNITS.get(name, "") for name in variables]
    try:
        with NetCDFDataset(temporary, "w", format="NETCDF4") as dataset:
            dataset.createDimension("channel", len(variables))
            dataset.createDimension("lat", len(lat))
            dataset.createDimension("lon", len(lon))
            channel = dataset.createVariable("channel", "i4", ("channel",))
            channel[:] = np.arange(len(variables), dtype=np.int32)
            lat_var = dataset.createVariable("lat", "f8", ("lat",))
            lon_var = dataset.createVariable("lon", "f8", ("lon",))
            lat_var[:] = arrays["lat"]
            lon_var[:] = arrays["lon"]
            for name in REQUIRED_DAILY_VARIABLES[:3]:
                variable = dataset.createVariable(
                    name,
                    "f4",
                    ("channel", "lat", "lon"),
                    zlib=True,
                    complevel=4,
                    shuffle=True,
                    fletcher32=True,
                )
                variable[:] = arrays[name]
            residual_mask = dataset.createVariable(
                "residual_valid_mask",
                "u1",
                ("channel", "lat", "lon"),
                zlib=True,
                complevel=4,
                shuffle=True,
                fletcher32=True,
            )
            static_mask = dataset.createVariable(
                "prism_valid_mask",
                "u1",
                ("lat", "lon"),
                zlib=True,
                complevel=4,
                shuffle=True,
                fletcher32=True,
            )
            residual_mask[:] = arrays["residual_valid_mask"]
            static_mask[:] = arrays["prism_valid_mask"]
            dataset.cache_schema = CACHE_SCHEMA
            dataset.cache_schema_version = CACHE_SCHEMA_VERSION
            dataset.cache_contract_digest = manifest["contract_digest"]
            dataset.content_sha256 = content_sha
            dataset.split = str(split)
            dataset.date = str(sample_date)[:10]
            dataset.case_name = manifest["contract"]["case_name"]
            dataset.phase1_checkpoint_sha256 = manifest["contract"][
                "phase1_checkpoint"
            ]["sha256"]
            dataset.phase1_fingerprint = manifest["contract"][
                "phase1_checkpoint"
            ]["phase1_fingerprint"]
            dataset.target_variables_json = json.dumps(variables)
            dataset.target_units_json = json.dumps(units)
            dataset.target_space = TARGET_SPACE_NAME
            dataset.residual_definition = RESIDUAL_DEFINITION
            dataset.baseline_semantics = manifest["contract"][
                "baseline_semantics"
            ]
            for name, value in mask_provenance.items():
                dataset.setncattr(str(name), str(value))
            dataset.sync()
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return content_sha


def _retained_tile_positions(
    plan: TilePlan,
    static_valid_mask: np.ndarray,
    geometry: Mapping[str, Any],
) -> list[tuple[int, int]]:
    positions = list(plan.positions)
    if (
        geometry["skip_empty_target_tiles"]
        and geometry["min_valid_target_fraction"] > 0
    ):
        core_h, core_w = plan.core_shape
        threshold = float(geometry["min_valid_target_fraction"])
        positions = [
            (y0, x0)
            for y0, x0 in positions
            if float(
                static_valid_mask[y0 : y0 + core_h, x0 : x0 + core_w].mean()
            )
            >= threshold
        ]
    if not positions:
        raise RuntimeError("No cache tiles remain after target-mask filtering")
    return positions


def _coverage_multiplicity(
    shape: Sequence[int],
    positions: Iterable[tuple[int, int]],
    core: Sequence[int],
) -> np.ndarray:
    coverage = np.zeros(tuple(int(value) for value in shape), dtype=np.int32)
    height, width = map(int, core)
    for y0, x0 in positions:
        coverage[y0 : y0 + height, x0 : x0 + width] += 1
    return coverage


def _is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "CUDA out of memory" in str(exc)
    )


@torch.inference_mode()
def _build_day(
    *,
    model: torch.nn.Module,
    target_space: NormalizedTargetSpace,
    dataset: NarrPrismDataset,
    sample_date: date,
    split: str,
    manifest: Mapping[str, Any],
    output_path: Path,
    plan: TilePlan,
    tile_positions: Sequence[tuple[int, int]],
    static_valid_mask: np.ndarray,
    mask_provenance: Mapping[str, Any],
    config: Any,
    device: torch.device,
    batch_size: int,
) -> None:
    target_variables = list(manifest["contract"]["target_variables"])
    n_channels = len(target_variables)
    core_h, core_w = plan.core_shape
    halo_h, halo_w = plan.halo
    pad_multiple = _pad_multiple_from_config(config)
    day_predictors = dataset._load_predictor_day(sample_date)
    target_physical_cpu = dataset._load_targets(
        sample_date, slice(None), slice(None)
    )
    stitcher = WeightedTileStitcher(n_channels, dataset.fine_shape)
    window = blend_window(
        plan.core_shape,
        plan.overlap,
        mode=manifest["contract"]["cache_geometry"]["blend_mode"],
    )
    start = 0
    current_batch = max(1, int(batch_size))
    while start < len(tile_positions):
        chunk = list(tile_positions[start : start + current_batch])
        try:
            xs = [
                _pad_to_multiple(
                    extract_halo_context(
                        day_predictors,
                        (y0, x0),
                        plan.core_shape,
                        plan.halo,
                    ).unsqueeze(0),
                    pad_multiple,
                )
                for y0, x0 in chunk
            ]
            x = torch.cat(xs, dim=0).to(
                device, dtype=torch.float32, non_blocking=True
            )
            offsets = torch.tensor(chunk, dtype=torch.long, device=device)
            input_offsets = offsets - torch.tensor(
                [halo_h, halo_w], dtype=torch.long, device=device
            )
            output_crop = torch.tensor(
                [plan.output_crop] * len(chunk),
                dtype=torch.long,
                device=device,
            )
            target_stub = torch.zeros(
                (len(chunk), n_channels, core_h, core_w),
                dtype=torch.float32,
                device=device,
            )
            prediction = model(
                {
                    "x": x,
                    "y": target_stub,
                    "__scaler_offset": offsets,
                    "__input_scaler_offset": input_offsets,
                    "__output_scaler_offset": offsets,
                    "__output_crop": output_crop,
                }
            )
            if isinstance(prediction, dict):
                prediction = prediction.get(
                    "y_hat",
                    prediction.get("output", next(iter(prediction.values()))),
                )
            prediction = prediction[..., :core_h, :core_w]
            values = prediction.detach().to("cpu", dtype=torch.float32).numpy()
            for index, origin in enumerate(chunk):
                stitcher.add(values[index], origin, window)
            start += len(chunk)
        except BaseException as exc:
            if not _is_cuda_oom(exc) or len(chunk) <= 1:
                raise
            current_batch = max(1, len(chunk) // 2)
            print(
                f"[phase1-cache] {split} {sample_date}: CUDA OOM at tile "
                f"{start + 1}/{len(tile_positions)}; retrying batch={current_batch}",
                flush=True,
            )
            _clear_cuda_memory(device)

    if np.any(static_valid_mask & (stitcher.weight <= 0)):
        raise RuntimeError(
            f"{split} {sample_date}: valid PRISM cells were not covered"
        )
    deterministic_physical = stitcher.finalize(
        require_full_coverage=False
    ).astype(np.float32)
    # Retain finite baseline values everywhere reached by a training tile. Cells
    # that can never enter a retained training/validation tile remain NaN.
    finite_baseline = np.isfinite(deterministic_physical).all(axis=0)
    _sanity_check_outputs(
        deterministic_physical[np.newaxis], target_variables, str(sample_date)
    )
    deterministic_filled = np.where(
        finite_baseline[np.newaxis], deterministic_physical, 0.0
    ).astype(np.float32)
    deterministic_tensor = (
        torch.from_numpy(deterministic_filled).unsqueeze(0).to(device)
    )
    offset = torch.zeros((1, 2), dtype=torch.long, device=device)
    deterministic_normalized = (
        target_space.encode(deterministic_tensor, scaler_offset=offset)[0]
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    deterministic_normalized[:, ~finite_baseline] = np.nan

    target_physical = target_physical_cpu.unsqueeze(0).to(
        device, dtype=torch.float32
    )
    baseline_for_residual = (
        torch.from_numpy(np.nan_to_num(deterministic_normalized, nan=0.0))
        .unsqueeze(0)
        .to(device)
    )
    residual, residual_valid = target_space.residual_target(
        target_physical, baseline_for_residual, scaler_offset=offset
    )
    # A target cannot be used where no deterministic tile was retained.
    covered = torch.from_numpy(finite_baseline).to(
        device=device, dtype=torch.bool
    )
    residual_valid = residual_valid & covered.view(1, 1, *covered.shape)
    residual = torch.where(
        residual_valid, residual, torch.zeros_like(residual)
    )
    residual_np = residual[0].cpu().numpy().astype(np.float32)
    valid_np = residual_valid[0].cpu().numpy().astype(np.uint8)
    if not np.isfinite(residual_np).all():
        raise RuntimeError(
            f"{split} {sample_date}: residual cache contains non-finite values"
        )

    _write_daily_cache(
        output_path,
        manifest=manifest,
        split=split,
        sample_date=sample_date,
        lat=dataset.fine_lat,
        lon=dataset.fine_lon,
        deterministic_physical=deterministic_physical,
        deterministic_normalized=deterministic_normalized,
        residual_target_normalized=residual_np,
        residual_valid_mask=valid_np,
        prism_valid_mask=static_valid_mask,
        mask_provenance=mask_provenance,
    )
    validate_daily_cache(
        output_path,
        manifest,
        expected_split=split,
        expected_date=sample_date,
        verify_content=True,
    )


def _prepare_manifest(
    cfg: Mapping[str, Any],
    config: Any,
    checkpoint: str,
    output_root: str,
    splits: Sequence[str],
    phase1_fingerprint: str,
    *,
    create: bool,
) -> tuple[dict[str, Any], Path]:
    assert_scalars_available(config, role="phase1 residual cache")
    apply_scalar_paths(config)
    # Constructing one split dataset validates grid/dates/preprocessed artifacts.
    probe_dataset = (
        NarrPrismDataset(str(config._config_path), mode=splits[0])
        if hasattr(config, "_config_path")
        else None
    )
    grid_shape = (
        probe_dataset.fine_shape if probe_dataset is not None else None
    )
    contract = cache_contract(
        cfg,
        config,
        checkpoint,
        grid_shape=grid_shape,
        phase1_fingerprint=phase1_fingerprint,
        splits=splits,
    )
    digest = contract_digest(contract)
    directory = cache_directory(output_root, digest)
    manifest_path = directory / MANIFEST_NAME
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        _assert_manifest_schema(manifest, manifest_path)
        if manifest["contract"] != contract:
            raise RuntimeError(
                f"Existing cache manifest {manifest_path} has a different contract."
            )
    elif create:
        manifest = _new_manifest(cfg, contract, splits)
        _atomic_json(manifest_path, manifest)
    else:
        raise FileNotFoundError(
            f"Cache manifest does not exist: {manifest_path}"
        )
    manifest = dict(manifest)
    manifest["_manifest_path"] = str(manifest_path)
    return manifest, manifest_path


def _resolve_output_root(cfg: Mapping[str, Any], explicit: str | None) -> str:
    if explicit:
        return str(resolve_path(explicit))
    performance = cfg.get("performance", {}) or {}
    phase1_cache = performance.get("phase1_cache", {}) or {}
    configured = phase1_cache.get("path")
    if configured:
        return str(resolve_path(configured))
    experiment = resolve_path(
        cfg.get("path_experiment", "./examples/NARR_PRISM/experiments")
    )
    return str(experiment / "phase1_residual_cache" / get_case_name(cfg))


def _parse_splits(value: str | Sequence[str]) -> tuple[str, ...]:
    items = value.split(",") if isinstance(value, str) else list(value)
    splits = tuple(
        str(item).strip().lower() for item in items if str(item).strip()
    )
    if not splits:
        raise ValueError("At least one cache split is required")
    unknown = sorted(set(splits) - set(DEFAULT_SPLITS))
    if unknown:
        raise ValueError(f"Unsupported cache split(s): {unknown}")
    return splits


def _build_tasks(
    cfg: Mapping[str, Any], splits: Sequence[str]
) -> list[tuple[str, date]]:
    tasks: list[tuple[str, date]] = []
    for split in splits:
        start, end = parse_date_range_from_config(dict(cfg), split)
        tasks.extend((split, value) for value in date_range(start, end))
    return tasks


def _build_worker(args: argparse.Namespace) -> int:
    cfg = load_yaml(args.config)
    config = get_config(str(resolve_path(args.config)))
    # Retain the exact source path for dataset construction in _prepare_manifest.
    config._config_path = str(resolve_path(args.config))
    checkpoint = str(resolve_path(args.checkpoint))
    state_fingerprint = getattr(args, "phase1_fingerprint", None)
    if state_fingerprint is None:
        state_fingerprint = checkpoint_state_fingerprint(checkpoint)
    elif len(state_fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in state_fingerprint
    ):
        raise ValueError("--phase1-fingerprint must be a lowercase SHA-256")
    args._phase1_fingerprint = state_fingerprint
    splits = _parse_splits(args.splits)
    output_root = _resolve_output_root(cfg, args.output_root)
    manifest, manifest_path = _prepare_manifest(
        cfg,
        config,
        checkpoint,
        output_root,
        splits,
        state_fingerprint,
        create=not args.worker,
    )
    if manifest.get("state") == CACHE_STATE_COMPLETE:
        load_and_validate_manifest(manifest_path, validate_inventory=True)
        print(f"[phase1-cache] already complete: {manifest_path}")
        return 0

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    numeric_policy = configure_phase1_cache_numerics()
    print(
        "[phase1-cache] numerical policy: "
        + json.dumps(numeric_policy, sort_keys=True),
        flush=True,
    )
    model = _load_model(config, checkpoint, device, data_parallel=False)
    if any(
        parameter.dtype != torch.float32 for parameter in model.parameters()
    ):
        raise RuntimeError("Phase-1 cache generation requires an FP32 model")
    target_space = NormalizedTargetSpace(model)

    geometry = manifest["contract"]["cache_geometry"]
    static_valid_mask = load_target_valid_mask(
        config,
        role="NARR Phase-1 residual cache",
        expected_shape=tuple(geometry["domain_shape"]),
    )
    mask_provenance = _target_valid_mask_provenance(config, static_valid_mask)
    plan = TilePlan.build(
        tuple(geometry["domain_shape"]),
        geometry["core_shape"],
        overlap=geometry["overlap"],
        halo=geometry["halo"],
    )
    tile_positions = _retained_tile_positions(
        plan, static_valid_mask, geometry
    )
    batch_size = _cap_tile_batch_for_cuda_indexing(
        batch_size=max(1, int(args.batch_size)),
        config=config,
        tile_h=plan.context_shape[0],
        tile_w=plan.context_shape[1],
        device=device,
        inf_cfg=cfg.get("inference", {}) or {},
    )
    tasks = _build_tasks(cfg, splits)
    tasks = [
        task
        for index, task in enumerate(tasks)
        if index % int(args.date_shard_count) == int(args.date_shard_index)
    ]
    if not tasks:
        raise RuntimeError(
            f"Date shard {args.date_shard_index}/{args.date_shard_count} is empty"
        )
    print(
        f"[phase1-cache] worker shard={args.date_shard_index}/{args.date_shard_count} "
        f"device={device} days={len(tasks)} tiles/day={len(tile_positions)} "
        f"batch={batch_size} cache={manifest_path.parent}",
        flush=True,
    )
    datasets = {
        split: NarrPrismDataset(str(resolve_path(args.config)), mode=split)
        for split in splits
    }
    iterator: Any = tasks
    if tqdm is not None and not args.no_progress:
        iterator = tqdm(
            tasks,
            desc=f"Phase1 cache shard {args.date_shard_index}",
            unit="day",
        )
    for split, sample_date in iterator:
        output_path = daily_cache_path(
            manifest_path.parent, split, sample_date
        )
        if output_path.exists():
            try:
                validate_daily_cache(
                    output_path,
                    manifest,
                    expected_split=split,
                    expected_date=sample_date,
                    verify_content=bool(args.repair_invalid),
                )
            except Exception:
                if not args.repair_invalid:
                    raise
            else:
                continue
        _build_day(
            model=model,
            target_space=target_space,
            dataset=datasets[split],
            sample_date=sample_date,
            split=split,
            manifest=manifest,
            output_path=output_path,
            plan=plan,
            tile_positions=tile_positions,
            static_valid_mask=static_valid_mask,
            mask_provenance=mask_provenance,
            config=config,
            device=device,
            batch_size=batch_size,
        )
    return 0


def _inventory_digest_for_split(
    manifest: Mapping[str, Any], manifest_path: Path, split: str
) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    dates = manifest["splits"][split]["expected_dates"]
    iterator: Any = dates
    if tqdm is not None:
        iterator = tqdm(
            dates,
            desc=f"Finalize cache inventory ({split})",
            unit="day",
        )
    for sample_date in iterator:
        metadata = validate_daily_cache(
            daily_cache_path(manifest_path.parent, split, sample_date),
            manifest,
            expected_split=split,
            expected_date=sample_date,
            verify_content=True,
        )
        digest.update(sample_date.encode("ascii"))
        digest.update(metadata["content_sha256"].encode("ascii"))
        count += 1
    return count, digest.hexdigest()


def _fit_training_residual_stats(
    manifest: Mapping[str, Any], manifest_path: Path
) -> dict[str, Any]:
    if "training" not in manifest["splits"]:
        raise RuntimeError(
            "Residual normalization requires the training split"
        )
    geometry = manifest["contract"]["cache_geometry"]
    first_date = manifest["splits"]["training"]["expected_dates"][0]
    first_path = daily_cache_path(manifest_path.parent, "training", first_date)
    with NetCDFDataset(first_path, "r") as dataset:
        static_valid_mask = np.asarray(
            dataset.variables["prism_valid_mask"][:], dtype=bool
        )
    plan = TilePlan.build(
        tuple(geometry["domain_shape"]),
        geometry["core_shape"],
        overlap=geometry["overlap"],
        halo=geometry["halo"],
    )
    positions = _retained_tile_positions(plan, static_valid_mask, geometry)
    multiplicity = _coverage_multiplicity(
        geometry["domain_shape"], positions, geometry["core_shape"]
    ).astype(np.float64)
    n_channels = len(manifest["contract"]["target_variables"])
    sums = np.zeros(n_channels, dtype=np.float64)
    squares = np.zeros(n_channels, dtype=np.float64)
    counts = np.zeros(n_channels, dtype=np.int64)
    dates = manifest["splits"]["training"]["expected_dates"]
    iterator: Any = dates
    if tqdm is not None:
        iterator = tqdm(dates, desc="Finalize residual statistics", unit="day")
    for sample_date in iterator:
        path = daily_cache_path(manifest_path.parent, "training", sample_date)
        with NetCDFDataset(path, "r") as dataset:
            residual = np.asarray(
                dataset.variables["residual_target_normalized"][:],
                dtype=np.float64,
            )
            valid = np.asarray(
                dataset.variables["residual_valid_mask"][:], dtype=bool
            )
        weights = valid.astype(np.float64) * multiplicity[np.newaxis]
        sums += (residual * weights).sum(axis=(1, 2), dtype=np.float64)
        squares += (np.square(residual) * weights).sum(
            axis=(1, 2), dtype=np.float64
        )
        counts += weights.sum(axis=(1, 2), dtype=np.float64).astype(np.int64)
    if np.any(counts <= 0):
        raise RuntimeError(
            "At least one residual channel has no valid training values"
        )
    means = sums / counts
    variance = np.maximum(squares / counts - np.square(means), 0.0)
    std = np.sqrt(variance)
    statistics_contract = manifest["contract"]["statistics"]
    if not statistics_contract.get("residual_normalization_enabled"):
        raise RuntimeError("Cache contract disables residual normalization")
    epsilon = float(statistics_contract["residual_normalization_epsilon"])
    std = np.maximum(std, epsilon)
    return {
        "enabled": True,
        "fitted": True,
        "mean": means.tolist(),
        "std": std.tolist(),
        "count": counts.tolist(),
        "epsilon": epsilon,
        "target_variables": list(manifest["contract"]["target_variables"]),
        "space": TARGET_SPACE_NAME,
        "residual_definition": RESIDUAL_DEFINITION,
        "fit_split": "training",
        "fit_start": dates[0],
        "fit_end": dates[-1],
        "fit_days": len(dates),
        "weighting": "training_tile_overlap_multiplicity",
        "training_tiles_per_day": len(positions),
        "variance": "population",
    }


def finalize_cache(manifest_path: str | os.PathLike) -> dict[str, Any]:
    """Verify exact inventory, fit training stats, and atomically mark complete."""
    path = _resolve_manifest_path(manifest_path)
    manifest = _read_json(path)
    _assert_manifest_schema(manifest, path)
    if manifest.get("state") == CACHE_STATE_COMPLETE:
        return load_and_validate_manifest(path, validate_inventory=True)
    for split in manifest["splits"]:
        count, digest = _inventory_digest_for_split(manifest, path, split)
        manifest["splits"][split]["completed_count"] = count
        manifest["splits"][split]["inventory_digest_sha256"] = digest
    manifest["residual_normalization"] = _fit_training_residual_stats(
        manifest, path
    )
    manifest["state"] = CACHE_STATE_COMPLETE
    manifest["updated_utc"] = _utc_now()
    manifest["completed_utc"] = manifest["updated_utc"]
    _atomic_json(path, manifest)
    return load_and_validate_manifest(path, validate_inventory=True)


def _terminate_processes(
    processes: Sequence[tuple[str, subprocess.Popen[Any]]],
    *,
    timeout: float = 10.0,
) -> None:
    """Terminate, then kill and reap all unfinished cache workers."""
    for _, process in processes:
        if process.poll() is None:
            process.terminate()
    for _, process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
        process.wait()


def _install_worker_signal_handlers(
    processes: Sequence[tuple[str, subprocess.Popen[Any]]],
    *,
    release_worker_lease: Any,
) -> dict[signal.Signals, Any]:
    """Handle SIGINT/SIGTERM by stopping children before lock release."""
    previous = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    handling = False

    def handle(signum: int, frame: Any) -> None:
        del frame
        nonlocal handling
        if handling:
            return
        handling = True
        for managed in previous:
            signal.signal(managed, signal.SIG_IGN)
        try:
            _terminate_processes(processes)
        finally:
            release_worker_lease()
        raise SystemExit(128 + int(signum))

    for signum in previous:
        signal.signal(signum, handle)
    return previous


def _restore_signal_handlers(previous: Mapping[signal.Signals, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def _cleanup_parallel_parent(
    processes: Sequence[tuple[str, subprocess.Popen[Any]]],
    previous_handlers: Mapping[signal.Signals, Any],
    worker_lease: CacheWorkerLease,
    *,
    terminate: bool,
) -> None:
    if terminate:
        _terminate_processes(processes)
    _restore_signal_handlers(previous_handlers)
    atexit.unregister(_terminate_processes)
    atexit.unregister(CacheWorkerLease.release)
    worker_lease.release()


def _wait_for_processes(
    processes: Sequence[tuple[str, subprocess.Popen[Any]]],
) -> None:
    """Poll all workers and fail immediately when any shard exits nonzero."""
    while True:
        failures = [
            (gpu_id, int(code))
            for gpu_id, process in processes
            if (code := process.poll()) not in (None, 0)
        ]
        if failures:
            raise RuntimeError(f"Phase-1 cache worker failures: {failures}")
        if all(process.poll() == 0 for _, process in processes):
            return
        time.sleep(0.2)


def _parallel_build(args: argparse.Namespace) -> int:
    gpu_ids = [
        value.strip()
        for value in args.parallel_gpus.split(",")
        if value.strip()
    ]
    if not gpu_ids:
        raise ValueError("--parallel-gpus did not contain a GPU id")
    cfg = load_yaml(args.config)
    config = get_config(str(resolve_path(args.config)))
    config._config_path = str(resolve_path(args.config))
    splits = _parse_splits(args.splits)
    checkpoint = str(resolve_path(args.checkpoint))
    state_fingerprint = checkpoint_state_fingerprint(checkpoint)

    output_root = _resolve_output_root(cfg, args.output_root)
    _, manifest_path = _prepare_manifest(
        cfg,
        config,
        checkpoint,
        output_root,
        splits,
        state_fingerprint,
        create=True,
    )
    processes: list[tuple[str, subprocess.Popen[Any]]] = []
    worker_lease = CacheWorkerLease(manifest_path.parent).acquire()
    previous_handlers = _install_worker_signal_handlers(
        processes,
        release_worker_lease=worker_lease.release,
    )
    atexit.register(CacheWorkerLease.release, worker_lease)
    atexit.register(_terminate_processes, processes)
    for shard, gpu_id in enumerate(gpu_ids):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "build",
            "--config",
            str(resolve_path(args.config)),
            "--checkpoint",
            str(resolve_path(args.checkpoint)),
            "--output-root",
            output_root,
            "--splits",
            ",".join(splits),
            "--device",
            "cuda:0",
            "--batch-size",
            str(args.batch_size),
            "--date-shard-index",
            str(shard),
            "--date-shard-count",
            str(len(gpu_ids)),
            "--worker",
            "--phase1-fingerprint",
            state_fingerprint,
        ]
        if args.no_progress:
            command.append("--no-progress")
        if args.repair_invalid:
            command.append("--repair-invalid")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu_id
        environment.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
        )
        print(
            f"[phase1-cache] launch shard={shard} physical_gpu={gpu_id}",
            flush=True,
        )
        try:
            process = subprocess.Popen(
                command,
                env=environment,
                pass_fds=(worker_lease.fileno(),),
            )
        except BaseException:
            _cleanup_parallel_parent(
                processes,
                previous_handlers,
                worker_lease,
                terminate=True,
            )
            raise
        processes.append((gpu_id, process))
    try:
        _wait_for_processes(processes)
    except BaseException:
        _cleanup_parallel_parent(
            processes,
            previous_handlers,
            worker_lease,
            terminate=True,
        )
        raise
    _cleanup_parallel_parent(
        processes,
        previous_handlers,
        worker_lease,
        terminate=False,
    )
    finalized = finalize_cache(manifest_path)
    print(
        f"[phase1-cache] complete: {finalized['_manifest_path']} "
        f"stats={json.dumps(finalized['residual_normalization'], sort_keys=True)}"
    )
    return 0


def _run_parent_build(args: argparse.Namespace) -> int:
    if args.parallel_gpus and not args.worker:
        return _parallel_build(args)
    result = _build_worker(args)
    if (
        result == 0
        and not args.worker
        and int(args.date_shard_count) == 1
        and int(args.date_shard_index) == 0
    ):
        cfg = load_yaml(args.config)
        config = get_config(str(resolve_path(args.config)))
        config._config_path = str(resolve_path(args.config))
        state_fingerprint = getattr(
            args, "_phase1_fingerprint", None
        ) or checkpoint_state_fingerprint(args.checkpoint)
        manifest, manifest_path = _prepare_manifest(
            cfg,
            config,
            str(resolve_path(args.checkpoint)),
            _resolve_output_root(cfg, args.output_root),
            _parse_splits(args.splits),
            state_fingerprint,
            create=False,
        )
        if manifest.get("state") != CACHE_STATE_COMPLETE:
            finalize_cache(manifest_path)
            print(f"[phase1-cache] complete: {manifest_path}")
    return result


def cmd_build(args: argparse.Namespace) -> int:
    if args.worker:
        return _build_worker(args)
    cfg = load_yaml(args.config)
    output_root = _resolve_output_root(cfg, args.output_root)
    with CacheBuildLock(output_root):
        return _run_parent_build(args)


def _cmd_finalize_unlocked(args: argparse.Namespace) -> int:
    manifest = finalize_cache(args.cache)
    print(
        json.dumps(
            {
                key: value
                for key, value in manifest.items()
                if key != "contract"
            },
            indent=2,
        )
    )
    return 0


def cmd_finalize(args: argparse.Namespace) -> int:
    manifest_path = _resolve_manifest_path(args.cache)
    cache_root = manifest_path.parent.parent
    with CacheBuildLock(cache_root):
        return _cmd_finalize_unlocked(args)


def cmd_validate(args: argparse.Namespace) -> int:
    manifest = load_and_validate_manifest(
        args.cache,
        require_complete=not args.allow_incomplete,
        validate_inventory=True,
        verify_daily_content=args.deep,
    )
    print(
        json.dumps(
            {
                "manifest": manifest["_manifest_path"],
                "state": manifest["state"],
                "contract_digest": manifest["contract_digest"],
                "splits": manifest["splits"],
                "residual_normalization": manifest.get(
                    "residual_normalization"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def describe_cache(
    cache: str | os.PathLike,
    *,
    validate_present: bool = False,
) -> dict[str, Any]:
    """Describe recorded completion and observed resumable file progress."""
    manifest = load_and_validate_manifest(
        cache,
        require_complete=False,
        validate_inventory=False,
    )
    cache_dir = Path(manifest["_manifest_path"]).parent
    split_summaries: dict[str, Any] = {}
    for name, record in manifest["splits"].items():
        base = {
            key: record.get(key)
            for key in (
                "start",
                "end",
                "expected_count",
                "completed_count",
                "inventory_digest_sha256",
            )
        }
        present = 0
        invalid = 0
        invalid_examples: list[dict[str, str]] = []
        for sample_date in record["expected_dates"]:
            path = daily_cache_path(cache_dir, name, sample_date)
            if not path.is_file():
                continue
            present += 1
            if validate_present:
                try:
                    validate_daily_cache(
                        path,
                        manifest,
                        expected_split=name,
                        expected_date=sample_date,
                    )
                except Exception as exc:
                    invalid += 1
                    if len(invalid_examples) < 5:
                        invalid_examples.append(
                            {
                                "date": sample_date,
                                "path": str(path),
                                "error": str(exc),
                            }
                        )
        expected = int(record["expected_count"])
        base.update(
            {
                "observed_present_count": present,
                "observed_missing_count": expected - present,
                "observed_valid_count": (
                    present - invalid if validate_present else None
                ),
                "observed_invalid_count": invalid
                if validate_present
                else None,
                "observed_validation": (
                    "shallow_manifest_contract"
                    if validate_present
                    else "not_scanned"
                ),
                "observed_invalid_examples": (
                    invalid_examples if validate_present else []
                ),
            }
        )
        split_summaries[name] = base
    return {
        "manifest": manifest["_manifest_path"],
        "state": manifest["state"],
        "contract_digest": manifest["contract_digest"],
        "case_name": manifest["contract"]["case_name"],
        "phase1_checkpoint": manifest["contract"]["phase1_checkpoint"],
        "target_variables": manifest["contract"]["target_variables"],
        "geometry": manifest["contract"]["cache_geometry"],
        "splits": split_summaries,
        "residual_normalization": manifest.get("residual_normalization"),
    }


def wait_for_cache_completion(
    output_root: str | os.PathLike,
    *,
    cfg: Mapping[str, Any],
    config: Any,
    phase1_checkpoint: str | os.PathLike,
    phase1_fingerprint: str | None = None,
    initial_validated_manifest: Mapping[str, Any] | None = None,
    poll_seconds: float = 60.0,
    timeout_seconds: float | None = None,
    status_callback: Callable[[dict[str, Any]], None] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Wait for an already-active compatible cache build to finalize.

    This never starts or stops a process. It validates the immutable contract
    whenever a manifest is visible, uses kernel advisory locks as the source of
    truth for activity, and performs full inventory validation before returning.
    If no parent or worker owns the cache, ``CacheBuildNotActiveError`` tells the
    caller it is safe to launch/resume a builder.
    """
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive when provided")
    started = time.monotonic()
    root = resolve_path(output_root)
    manifest_path: Path | None = None
    validated_contract: Any | None = None
    if initial_validated_manifest is not None:
        if "_manifest_path" not in initial_validated_manifest:
            raise ValueError(
                "initial_validated_manifest must come from "
                "load_and_validate_manifest"
            )
        manifest_path = Path(str(initial_validated_manifest["_manifest_path"]))
        validated_contract = initial_validated_manifest["contract"]
    while True:
        manifest: dict[str, Any] | None = None
        summary: dict[str, Any] | None = None
        cache_dir: Path | None = None
        try:
            if manifest_path is None:
                manifest = load_and_validate_manifest(
                    root,
                    cfg=cfg,
                    config=config,
                    phase1_checkpoint=phase1_checkpoint,
                    phase1_fingerprint=phase1_fingerprint,
                    require_complete=False,
                    validate_inventory=False,
                )
                manifest_path = Path(manifest["_manifest_path"])
                validated_contract = manifest["contract"]
            else:
                # The checkpoint and immutable contract were authenticated on
                # the first successful read. Poll the direct manifest path so
                # a multi-hour wait does not rehash the 2.9-GB checkpoint every
                # minute, while still rejecting any mid-build contract change.
                manifest = load_and_validate_manifest(
                    manifest_path,
                    require_complete=False,
                    validate_inventory=False,
                )
                if manifest["contract"] != validated_contract:
                    raise RuntimeError(
                        f"Cache contract changed while waiting: {manifest_path}"
                    )
        except FileNotFoundError:
            # A competing parent can own the root lock just before it creates
            # the first incomplete manifest. Other validation failures remain
            # fatal and are intentionally not caught.
            pass
        if manifest is not None:
            cache_dir = Path(manifest["_manifest_path"]).parent
            if manifest.get("state") == CACHE_STATE_COMPLETE:
                return load_and_validate_manifest(
                    root,
                    cfg=cfg,
                    config=config,
                    phase1_checkpoint=phase1_checkpoint,
                    phase1_fingerprint=phase1_fingerprint,
                    require_complete=True,
                    validate_inventory=True,
                )
            summary = describe_cache(
                manifest["_manifest_path"], validate_present=False
            )
        activity = cache_build_activity(root, cache_dir=cache_dir)
        elapsed = time.monotonic() - started
        status = {
            "elapsed_seconds": elapsed,
            "activity": activity,
            "summary": summary,
        }
        if status_callback is not None:
            status_callback(status)
        if not activity["active"]:
            # Close the small race where finalization completed between the
            # manifest read and the lock probe.
            try:
                return load_and_validate_manifest(
                    root,
                    cfg=cfg,
                    config=config,
                    phase1_checkpoint=phase1_checkpoint,
                    phase1_fingerprint=phase1_fingerprint,
                    require_complete=True,
                    validate_inventory=True,
                )
            except FileNotFoundError:
                pass
            except RuntimeError as exc:
                if "not complete" not in str(exc):
                    raise
            raise CacheBuildNotActiveError(
                f"Phase-1 cache at {root} is incomplete, but no parent or "
                "worker owns its advisory lock; launch/resume the cache builder."
            )
        if timeout_seconds is not None and elapsed >= timeout_seconds:
            raise TimeoutError(
                f"Timed out after {elapsed:.1f}s waiting for Phase-1 cache {root}"
            )
        sleep_fn(poll_seconds)


def cmd_describe(args: argparse.Namespace) -> int:
    summary = describe_cache(
        args.cache,
        validate_present=args.validate_files,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build/resume the daily cache")
    build.add_argument("--config", required=True)
    build.add_argument("--checkpoint", required=True)
    build.add_argument("--output-root", default=None)
    build.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    build.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    build.add_argument("--batch-size", type=int, default=4)
    build.add_argument(
        "--phase1-fingerprint",
        default=None,
        help=argparse.SUPPRESS,
    )
    build.add_argument("--parallel-gpus", default=None)
    build.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    build.add_argument("--date-shard-index", type=int, default=0)
    build.add_argument("--date-shard-count", type=int, default=1)
    build.add_argument("--repair-invalid", action="store_true")
    build.add_argument("--no-progress", action="store_true")
    build.set_defaults(func=cmd_build)

    finalize = subparsers.add_parser(
        "finalize", help="verify a manually sharded build and mark it complete"
    )
    finalize.add_argument("--cache", required=True)
    finalize.set_defaults(func=cmd_finalize)

    validate = subparsers.add_parser(
        "validate", help="validate manifest and exact inventory"
    )
    validate.add_argument("--cache", required=True)
    validate.add_argument("--allow-incomplete", action="store_true")
    validate.add_argument(
        "--deep", action="store_true", help="rehash every daily array"
    )
    validate.set_defaults(func=cmd_validate)

    describe = subparsers.add_parser(
        "describe", help="print cache contract and status"
    )
    describe.add_argument("--cache", required=True)
    describe.add_argument(
        "--validate-files",
        action="store_true",
        help="shallow-validate every observed daily file",
    )
    describe.set_defaults(func=cmd_describe)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(args, "date_shard_count"):
        if args.date_shard_count < 1:
            raise ValueError("--date-shard-count must be >= 1")
        if not 0 <= args.date_shard_index < args.date_shard_count:
            raise ValueError(
                "--date-shard-index must be within the shard count"
            )
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
