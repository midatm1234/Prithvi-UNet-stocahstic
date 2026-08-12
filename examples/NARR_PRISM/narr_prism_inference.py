"""Inference for the NARR-to-PRISM downscaling model.

Loads a trained checkpoint, runs prediction over the YAML-defined inference
date range, denormalizes outputs, and writes NetCDF files.

Usage:
    python narr_prism_inference.py --config NARR_PRISM_subdomain.yaml [--checkpoint path/to/last.ckpt]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import date, timedelta
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import torch

try:
    import xarray as xr
except ImportError:
    xr = None

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from granitewxc.utils.config import get_config
from granitewxc.utils.normalization import (
    TARGET_VALID_MASK_CRITERION,
    TARGET_VALID_MASK_MANIFEST_KEY,
    TRAINING_SOURCE_ARTIFACT_SPLIT_SIGNATURE_KEY,
    apply_scalar_paths,
    assert_scalars_available,
    assert_target_grid_matches,
    load_manifest,
    load_target_valid_mask,
    log_case_context,
    log_scalar_summary,
    resolve_scalar_dir,
    targets_are_spatial,
)
from granitewxc.utils.prism_tiling import (
    TilePlan,
    WeightedTileStitcher,
    blend_window,
    boundary_gradient_ratio,
    extract_halo_context,
    overlap_crossovers,
    overlap_disagreement,
    tile_origins as shared_tile_origins,
)
from granitewxc.utils.prism_grid import validate_prism_grid

from narr_prism_dataset import NarrPrismDataset
from narr_prism_utils import (
    case_output_dir,
    get_case_name,
    load_yaml,
    narr_source_var,
    parse_date_range_from_config,
    resolve_path,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TARGET_VALID_MASK_CONTENT_SHA256_ATTR = "target_valid_mask_content_sha256"
TARGET_VALID_MASK_SHA256_ATTR = "target_valid_mask_sha256"
TARGET_VALID_MASK_CRITERION_ATTR = "target_valid_mask_criterion"
TARGET_VALID_MASK_SOURCE_SPLIT_ATTR = "target_valid_mask_source_split"
TARGET_VALID_MASK_TRAINING_SOURCE_SIGNATURE_ATTR = (
    "target_valid_mask_training_source_artifact_split_signature"
)
TARGET_VALID_MASK_GRID_FINGERPRINT_ATTR = (
    "target_valid_mask_grid_fingerprint"
)
PREDICTION_SPLITS = ("validation", "inference")


def _validate_prediction_split(split: str) -> str:
    """Return a supported prediction split or fail before opening data."""
    normalized = str(split).strip().lower()
    if normalized not in PREDICTION_SPLITS:
        raise ValueError(
            f"split must be one of {PREDICTION_SPLITS}, got {split!r}"
        )
    return normalized


def _split_output_root(output_root: str | Path, split: str) -> Path:
    """Keep validation products in a directory distinct from final inference.

    The historical inference directory is unchanged. Validation uses a
    ``validation`` child unless the caller has already supplied a path scoped
    to that split (optionally followed by ``case_name``).
    """
    split = _validate_prediction_split(split)
    path = resolve_path(output_root)
    if split == "inference":
        return path
    if path.name == split or path.parent.name == split:
        return path
    return path / split


def _expected_split_dates(
    cfg: Mapping[str, Any], split: str = "inference"
) -> List[str]:
    """Return every configured split date, inclusive, in canonical order."""
    split = _validate_prediction_split(split)
    start, end = parse_date_range_from_config(dict(cfg), split)
    return [
        str(start + timedelta(days=index))
        for index in range((end - start).days + 1)
    ]


def _validate_dataset_dates(
    cfg: Mapping[str, Any],
    dates: Sequence[Any],
    split: str = "inference",
) -> List[Any]:
    """Require exact ordered coverage of the inclusive configured split."""
    split = _validate_prediction_split(split)
    observed = [str(value)[:10] for value in dates]
    expected = _expected_split_dates(cfg, split)
    if observed != expected:
        first_difference = next(
            (
                index
                for index, pair in enumerate(
                    zip(observed, expected, strict=False)
                )
                if pair[0] != pair[1]
            ),
            min(len(observed), len(expected)),
        )
        raise ValueError(
            f"NarrPrismDataset(mode={split!r}) dates do not exactly match the "
            f"inclusive YAML dates.{split} range: "
            f"observed_count={len(observed)}, expected_count={len(expected)}, "
            f"first_difference={first_difference}"
        )
    return list(dates)


def _target_valid_mask_content_sha256(mask: np.ndarray) -> str:
    """Hash canonical mask cells independently of their file container."""
    values = np.ascontiguousarray(np.asarray(mask, dtype=np.uint8))
    digest = hashlib.sha256()
    digest.update(b"granitewxc-target-valid-mask-cells-v1\0")
    digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _target_valid_mask_provenance(
    config: Any, mask: np.ndarray
) -> Dict[str, str]:
    """Return NetCDF-safe provenance for an authenticated support mask."""
    scalar_dir = resolve_scalar_dir(config, for_writing=False)
    manifest = load_manifest(scalar_dir)
    entry = (
        manifest.get(TARGET_VALID_MASK_MANIFEST_KEY)
        if isinstance(manifest, Mapping)
        else None
    )
    if not isinstance(entry, Mapping):
        raise ValueError(
            f"{scalar_dir}: normalization manifest does not bind the training "
            "target-valid-mask artifact"
        )
    required = {
        "sha256": entry.get("sha256"),
        "criterion": entry.get("criterion"),
        "source_split": entry.get("source_split"),
        "grid_fingerprint": entry.get("grid_fingerprint"),
    }
    missing = [
        name
        for name, value in required.items()
        if not str(value or "").strip()
    ]
    if missing:
        raise ValueError(
            f"{scalar_dir}: target-valid-mask manifest entry lacks {missing}"
        )
    if required["criterion"] != TARGET_VALID_MASK_CRITERION:
        raise ValueError(
            f"{scalar_dir}: unsupported target-valid-mask criterion "
            f"{required['criterion']!r}"
        )
    training_source_signature = entry.get(
        TRAINING_SOURCE_ARTIFACT_SPLIT_SIGNATURE_KEY
    )
    # Raw-source scalar runs have no preprocessed split signature. Preserve
    # that fact rather than writing an empty/ambiguous NetCDF attribute.
    training_source_text = (
        str(training_source_signature)
        if training_source_signature is not None
        else "not-applicable:raw-training-source-mode"
    )
    return {
        TARGET_VALID_MASK_SHA256_ATTR: str(required["sha256"]),
        TARGET_VALID_MASK_CONTENT_SHA256_ATTR: (
            _target_valid_mask_content_sha256(mask)
        ),
        TARGET_VALID_MASK_CRITERION_ATTR: str(required["criterion"]),
        TARGET_VALID_MASK_SOURCE_SPLIT_ATTR: str(required["source_split"]),
        TARGET_VALID_MASK_TRAINING_SOURCE_SIGNATURE_ATTR: training_source_text,
        TARGET_VALID_MASK_GRID_FINGERPRINT_ATTR: str(
            required["grid_fingerprint"]
        ),
    }


def _find_checkpoint(cfg: Dict[str, Any], explicit: Optional[str]) -> str:
    """Locate the model checkpoint to use for inference."""
    candidates = ("last.ckpt", "best.ckpt")
    if explicit:
        p = resolve_path(explicit)
        if p.exists():
            return str(p)
        raise FileNotFoundError(f"Explicit checkpoint not found: {p}")
    case_name = get_case_name(cfg)
    inf_cfg = cfg.get("inference", {})
    ckpt = inf_cfg.get("checkpoint_path")
    if ckpt:
        p = resolve_path(ckpt)
        if p.exists():
            return str(p)
        raise FileNotFoundError(f"inference.checkpoint_path not found: {p}")
    resume_ckpt = cfg.get("resume_checkpoint_path")
    if resume_ckpt:
        p = resolve_path(resume_ckpt)
        if p.exists():
            return str(p)
        print(f"[inference] resume_checkpoint_path not found (skipping): {p}", flush=True)

    checkpoint_dir = cfg.get("checkpoint_dir")
    if checkpoint_dir:
        case_checkpoint_dir = case_output_dir(resolve_path(checkpoint_dir), case_name)
        for candidate in candidates:
            p = case_checkpoint_dir / candidate
            if p.exists():
                return str(p)

    run_dir = cfg.get("run_dir")
    if run_dir:
        case_run_dir = case_output_dir(resolve_path(run_dir), case_name)
        for candidate in candidates:
            p = case_run_dir / candidate
            if p.exists():
                return str(p)
        for candidate in candidates:
            p = case_run_dir / "checkpoints" / candidate
            if p.exists():
                return str(p)

    exp = resolve_path(cfg.get("path_experiment", "."))
    for candidate in candidates:
        p = case_output_dir(resolve_path(exp) / "checkpoints", case_name) / candidate
        if p.exists():
            return str(p)

    raise FileNotFoundError(
        "Cannot locate a trained checkpoint. "
        f"Searched only case-scoped directories for '{case_name}'. Pass "
        "--checkpoint explicitly if the checkpoint lives elsewhere; checkpoints "
        "from sibling cases are never selected automatically."
    )


def _load_model(
    config: Any,
    checkpoint_path: str,
    device: torch.device,
    data_parallel: bool = False,
) -> torch.nn.Module:
    """Re-create the model architecture and load trained weights.

    When ``data_parallel`` is True and more than one CUDA device is visible, the
    model is wrapped in ``torch.nn.DataParallel`` so a batch of tiles is split
    across all visible GPUs.
    """
    from granitewxc.models.model import get_finetune_model_UNET

    if not hasattr(config.data, "input_static_surface_vars"):
        config.data.input_static_surface_vars = []

    model = get_finetune_model_UNET(config)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    from granitewxc.utils.prism_checkpoint import validate_prism_checkpoint_contract

    # A serialized module/object has no auditable pipeline envelope.  Corrected
    # PRISM configurations therefore reject it just like any other checkpoint
    # that predates the contract.
    checkpoint_envelope = ckpt if isinstance(ckpt, dict) else {}
    validate_prism_checkpoint_contract(
        config, checkpoint_envelope, role="NARR inference"
    )
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        state = ckpt.state_dict() if hasattr(ckpt, "state_dict") else dict(ckpt)

    # Strip DDP/FSDP module prefix if needed
    cleaned = {}
    for k, v in state.items():
        key = k
        if key.startswith("module."):
            key = key[len("module."):]
        if torch.is_tensor(v) and not torch.isfinite(v).all():
            bad = int((~torch.isfinite(v)).sum().item())
            raise ValueError(
                f"Checkpoint {checkpoint_path} tensor '{k}' contains "
                f"{bad} non-finite value(s); refusing to run inference."
            )
        cleaned[key] = v

    model_state = model.state_dict()
    scaler_state_keys = {
        "input_scalers_mu",
        "input_scalers_sigma",
        "output_scalers_mu",
        "output_scalers_sigma",
        "static_input_scalers_mu",
        "static_input_scalers_sigma",
        "static_output_scalers_mu",
        "static_output_scalers_sigma",
    }
    compatible = {}
    skipped_scalers = []
    for key, value in cleaned.items():
        # Scalers are coordinate-bearing, case-scoped config artifacts. Never
        # let an explicitly supplied or stale checkpoint override them, even
        # when the tensor shape happens to match.
        if key in scaler_state_keys:
            skipped_scalers.append(key)
            continue
        expected = model_state.get(key)
        if (
            expected is not None
            and torch.is_tensor(value)
            and tuple(value.shape) != tuple(expected.shape)
        ):
            raise ValueError(
                f"Checkpoint tensor '{key}' shape {tuple(value.shape)} does not "
                f"match configured model shape {tuple(expected.shape)}"
            )
        compatible[key] = value

    if skipped_scalers:
        print(
            "[inference] keeping case-scoped config scalers authoritative; "
            f"ignored {len(skipped_scalers)} scaler tensor(s) stored in checkpoint"
        )

    load_result = model.load_state_dict(compatible, strict=False)
    disallowed_missing = [
        key for key in load_result.missing_keys if key not in scaler_state_keys
    ]
    if disallowed_missing or load_result.unexpected_keys:
        raise ValueError(
            f"Checkpoint {checkpoint_path} is not architecture-complete for the "
            "configured NARR model: "
            f"missing={disallowed_missing}, "
            f"unexpected={list(load_result.unexpected_keys)}. Only case-scoped "
            "scaler buffers may be absent or replaced."
        )
    model.to(device)
    model.eval()

    if data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        n_gpu = torch.cuda.device_count()
        print(f"[inference] wrapping model in DataParallel across {n_gpu} GPUs")
        model = torch.nn.DataParallel(model, device_ids=list(range(n_gpu)))
    return model


# ---------------------------------------------------------------------------
# Output metadata and physical-range sanity checks
# ---------------------------------------------------------------------------
#
# IMPORTANT (root-cause note): the downscaling model used here
# (``ClimateDownscaleFinetuneUNETModel``) applies the target output scalers
# *internally* inside ``_decode_outputs`` (``decoded = normalized * sigma + mu``)
# and ``forward`` returns the result in **physical units** already
# (degC for tmax/tmin, mm/day for ppt). The inference loop must therefore NOT
# denormalize a second time. A previous version multiplied the already-physical
# output by the gridpoint std and re-added the gridpoint mean, which produced
# impossible temperatures (e.g. -97 degC .. +92 degC). See ``run_inference``.

# Physical units written into the NetCDF metadata for each target variable.
VAR_UNITS: Dict[str, str] = {
    "ppt": "mm/day",
    "tmax": "degC",
    "tmin": "degC",
}

# Physically plausible bounds (Celsius) for the temperature predictands. Values
# outside this band indicate a normalization/denormalization or unit error.
TEMP_PHYSICAL_MIN_C = -90.0
TEMP_PHYSICAL_MAX_C = 70.0
TEMP_VARS = ("tmax", "tmin")


def _channel_stats(prediction: np.ndarray, channel: int) -> Tuple[float, float, float, float]:
    """Return (min, max, mean, std) over finite values of one channel."""
    a = prediction[:, channel, ...]
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return (float("nan"),) * 4
    return (
        float(finite.min()),
        float(finite.max()),
        float(finite.mean()),
        float(finite.std()),
    )


def _print_channel_stats(
    prediction: np.ndarray,
    target_variables: Sequence[str],
    header: str,
) -> None:
    """Print per-channel min/max/mean/std for a (B, C, H, W) prediction batch."""
    print(header)
    for ch_idx, var in enumerate(target_variables):
        vmin, vmax, vmean, vstd = _channel_stats(prediction, ch_idx)
        unit = VAR_UNITS.get(var, "")
        print(
            f"    {var:>5} [{unit}]: min={vmin:8.3f} max={vmax:8.3f} "
            f"mean={vmean:8.3f} std={vstd:8.3f}"
        )


def _sanity_check_outputs(
    prediction: np.ndarray,
    target_variables: Sequence[str],
    date_string: str,
) -> None:
    """Validate that the (already physical-unit) model output is reasonable.

    Raises ``ValueError`` if any temperature channel falls outside the physically
    plausible Celsius band. This is a guard against denormalization/unit bugs --
    it does NOT clip the output (clipping would hide the real problem).
    Cells where ``tmax < tmin`` are reported but not treated as fatal, since an
    under-trained checkpoint can produce a small number of such crossings.
    """
    var_index = {v: i for i, v in enumerate(target_variables)}

    for var in TEMP_VARS:
        if var not in var_index:
            continue
        ch = var_index[var]
        vmin, vmax, _, _ = _channel_stats(prediction, ch)
        if vmin < TEMP_PHYSICAL_MIN_C or vmax > TEMP_PHYSICAL_MAX_C:
            raise ValueError(
                f"[{date_string}] '{var}' is outside the physical range "
                f"[{TEMP_PHYSICAL_MIN_C}, {TEMP_PHYSICAL_MAX_C}] degC "
                f"(min={vmin:.2f}, max={vmax:.2f}). This signals a "
                f"normalization/denormalization or unit error -- the model "
                f"output is already in physical units and must not be "
                f"denormalized again."
            )

    if "tmax" in var_index and "tmin" in var_index:
        tmax = prediction[:, var_index["tmax"], ...]
        tmin = prediction[:, var_index["tmin"], ...]
        valid = np.isfinite(tmax) & np.isfinite(tmin)
        crossings = int(np.sum((tmax < tmin) & valid))
        total = int(np.sum(valid))
        if crossings:
            pct = 100.0 * crossings / max(total, 1)
            print(
                f"[inference] {date_string}: tmax < tmin at {crossings}/{total} "
                f"cells ({pct:.3f}%)"
            )


def _validate_output(
    prediction: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    target_variables: Sequence[str],
    inference_dates: Sequence[date],
) -> None:
    """Run basic sanity checks on the inference output."""
    expected_time = len(inference_dates)
    if prediction.shape[0] != expected_time:
        raise ValueError(
            f"Inference output time dimension ({prediction.shape[0]}) does not match "
            f"requested dates ({expected_time})"
        )
    n_vars = len(target_variables)
    if prediction.shape[1] != n_vars:
        raise ValueError(
            f"Inference output has {prediction.shape[1]} variables, expected {n_vars}"
        )
    if np.any(~np.isfinite(prediction)):
        n_bad = int(np.sum(~np.isfinite(prediction)))
        raise ValueError(f"{n_bad} non-finite values detected in output")


def _validate_prediction_batch(
    prediction: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    target_variables: Sequence[str],
    valid_mask: Optional[np.ndarray] = None,
) -> None:
    """Run lightweight checks before streaming a batch to disk."""
    if prediction.ndim != 4:
        raise ValueError(f"Expected prediction shape (B, C, H, W), got {prediction.shape}")
    n_vars = len(target_variables)
    if prediction.shape[1] != n_vars:
        raise ValueError(
            f"Inference output has {prediction.shape[1]} variables, expected {n_vars}"
        )
    expected_hw = (len(target_lat), len(target_lon))
    if prediction.shape[-2:] != expected_hw:
        raise ValueError(
            f"Inference output grid {prediction.shape[-2:]} does not match "
            f"target grid {expected_hw}"
        )
    if valid_mask is None:
        if np.any(~np.isfinite(prediction)):
            n_bad = int(np.sum(~np.isfinite(prediction)))
            raise ValueError(f"{n_bad} non-finite values detected in output batch")
    else:
        finite_required = valid_mask[np.newaxis, np.newaxis, :, :]
        bad = ~np.isfinite(prediction) & finite_required
        if np.any(bad):
            n_bad = int(np.sum(bad))
            raise ValueError(
                f"{n_bad} non-finite values detected over valid target cells"
            )


def _pad_to_multiple(x: torch.Tensor, multiple: int) -> torch.Tensor:
    """Pad spatial dims (H, W) to the nearest multiple of ``multiple``."""
    pad_h = (multiple - x.shape[-2] % multiple) % multiple
    pad_w = (multiple - x.shape[-1] % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x
    return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")


def _pad_multiple_from_config(config: Any) -> int:
    mask_unit = getattr(config, "mask_unit_size", [16, 16])
    patch_sz = getattr(getattr(config, "model", object()), "downscaling_patch_size", [2, 2])
    mask_lat = mask_unit[0] if isinstance(mask_unit, list) else mask_unit
    patch_lat = patch_sz[0] if isinstance(patch_sz, list) else patch_sz
    return int(mask_lat) * int(patch_lat)


def _tile_origins(total: int, tile: int, stride: int) -> List[int]:
    """Return tile start indices covering [0, total) with the given stride.

    The final tile is shifted so it ends exactly at ``total`` (no out-of-bounds,
    full coverage of the domain edge).
    """
    return shared_tile_origins(total, tile, stride)


def _hann_window_2d(h: int, w: int) -> np.ndarray:
    """2-D separable Hann blend window with a small floor (avoids zero weight at
    domain corners where only one tile contributes)."""
    wy = np.hanning(h + 2)[1:-1] if h > 1 else np.ones(1)
    wx = np.hanning(w + 2)[1:-1] if w > 1 else np.ones(1)
    win = np.outer(wy, wx).astype(np.float32)
    return np.maximum(win, 1e-3)


class _ShapeOnlyTarget:
    """Minimal target placeholder for models that only inspect batch["y"].shape."""

    def __init__(self, batch_size: int, n_targets: int, target_shape: Tuple[int, int]) -> None:
        self.shape = torch.Size((batch_size, n_targets, *target_shape))


def _as_date_strings(batch_dates: Any) -> List[str]:
    if isinstance(batch_dates, (list, tuple)):
        return [str(d) for d in batch_dates]
    if hasattr(batch_dates, "tolist"):
        values = batch_dates.tolist()
        if isinstance(values, list):
            return [str(d) for d in values]
    return [str(batch_dates)]


def _date_strings_to_epoch_days(date_strings: Iterable[str]) -> np.ndarray:
    dates = np.array([np.datetime64(str(d), "D") for d in date_strings])
    epoch = np.datetime64("1970-01-01", "D")
    return (dates - epoch).astype("timedelta64[D]").astype(np.int32)


def _select_inference_dates(
    available_dates: Sequence[date],
    selected_dates: Optional[Sequence[Any]],
) -> List[date]:
    """Return requested dates in caller order after exact availability checks.

    ``None`` preserves the normal full-range behavior.  Explicit selection is
    primarily useful for small CPU smoke tests: the model is loaded once and
    only the requested, non-contiguous days are evaluated.
    """
    available = list(available_dates)
    if selected_dates is None:
        return available
    if len(selected_dates) == 0:
        raise ValueError("selected_dates must contain at least one date")

    available_by_date = {date.fromisoformat(str(value)[:10]): value for value in available}
    requested: List[date] = []
    for value in selected_dates:
        try:
            normalized = date.fromisoformat(str(value)[:10])
        except ValueError as exc:
            raise ValueError(
                f"selected_dates contains an invalid ISO date: {value!r}"
            ) from exc
        requested.append(normalized)

    if len(set(requested)) != len(requested):
        raise ValueError("selected_dates must not contain duplicates")
    missing = [value.isoformat() for value in requested if value not in available_by_date]
    if missing:
        raise ValueError(
            "selected_dates are outside the configured/available inference dates: "
            + ", ".join(missing)
        )
    return [available_by_date[value] for value in requested]


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        for idx in range(torch.cuda.device_count()):
            torch.cuda.synchronize(idx)


def _tensor_to_float32_numpy(value: torch.Tensor) -> np.ndarray:
    """Detach inference output and cross the NumPy boundary as float32.

    NumPy cannot consume ``torch.bfloat16`` tensors directly. Raw diagnostic
    predictions retain the autocast dtype, so cast explicitly while copying to
    CPU instead of relying on ``numpy().astype(...)`` after the transfer.
    """
    return value.detach().to(device="cpu", dtype=torch.float32).numpy()


def _add_time(times: Dict[str, float], key: str, seconds: float) -> None:
    times[key] = times.get(key, 0.0) + seconds


def _format_timing(times: Dict[str, float]) -> str:
    total = sum(times.values())
    parts = [f"{name}={value:.2f}s" for name, value in times.items()]
    parts.append(f"total={total:.2f}s")
    return " ".join(parts)


def _tail_text_file(path: Path, max_lines: int = 80) -> str:
    """Return the last lines of a text log for concise worker failure reports."""
    if not path.exists():
        return "<log file was not created>"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError as exc:
        return f"<could not read log: {exc}>"
    tail = "".join(lines[-max_lines:]).rstrip()
    return tail or "<log file is empty>"


def _cap_tile_batch_for_cuda_indexing(
    batch_size: int,
    config: Any,
    tile_h: int,
    tile_w: int,
    device: torch.device,
    inf_cfg: Dict[str, Any],
) -> int:
    """Keep inference batches below CUDA kernels that require 32-bit indexing."""
    requested = max(1, int(batch_size))
    explicit_cap = inf_cfg.get("max_tile_batch_size")
    if explicit_cap is not None:
        capped = min(requested, max(1, int(explicit_cap)))
    else:
        capped = requested

    if device.type == "cuda":
        model_cfg = getattr(config, "model", None)
        downscaling_embed_dim = int(getattr(model_cfg, "downscaling_embed_dim", 512))
        conv_input_channels = 2 * downscaling_embed_dim
        # Leave headroom below 2**31 because padding/convolution kernels may
        # create temporary views and indexed intermediates around this tensor.
        max_indexed_elements = int((2**31 - 1) * 0.50)
        max_by_indexing = max(
            1,
            max_indexed_elements // max(1, conv_input_channels * tile_h * tile_w),
        )
        capped = min(capped, max_by_indexing)

    if capped < requested:
        print(
            f"[inference] reducing tile batch size from {requested} to {capped} "
            "to stay below CUDA 32-bit indexing limits",
            flush=True,
        )
    return capped


def _is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "CUDA out of memory" in str(exc)
    )


def _clear_cuda_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _load_native_predictor_diagnostic(
    dataset: NarrPrismDataset, sample_date: Any
) -> Dict[str, np.ndarray]:
    """Load the first configured NARR channel before PRISM regridding."""
    variable, level = dataset.predictor_vars[0]
    if dataset._predictor_map:
        path = dataset._predictor_map[sample_date][variable]
    else:
        product = dataset._preprocessed_map[sample_date]
        with xr.open_dataset(str(product)) as product_ds:
            sources = str(product_ds.attrs.get("source_narr", ""))
        source_map = {
            item.split("=", 1)[0]: item.split("=", 1)[1]
            for item in sources.split(";")
            if "=" in item
        }
        if variable not in source_map:
            raise ValueError(
                f"Preprocessed product {product} does not identify native source for {variable}"
            )
        path = Path(source_map[variable])
    with xr.open_dataset(str(path)) as ds:
        da = ds[narr_source_var(variable)]
        if "time" in da.dims:
            da = da.sel(time=np.datetime64(sample_date), drop=True)
        level_name = "level" if "level" in da.dims else "lev" if "lev" in da.dims else None
        if level_name is not None:
            da = da.sel({level_name: level}, drop=True)
        return {
            "native_predictor": np.asarray(da.values, dtype=np.float32),
            "native_lat": np.asarray(ds["lat"].values, dtype=np.float64),
            "native_lon": np.asarray(ds["lon"].values, dtype=np.float64),
            "native_predictor_name": np.asarray(f"{variable}_{level}"),
        }


def _open_streaming_output(  # noqa: C901
    out_path: Path,
    target_variables: Sequence[str],
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    checkpoint_path: str,
    inference_dates: Sequence[date],
    case_name: str,
    valid_mask: np.ndarray,
    mask_provenance: Mapping[str, str],
    split: str = "inference",
    configured_split_dates: Optional[Sequence[Any]] = None,
) -> Any:
    """Create a NetCDF3 file that can be filled along time without buffering."""
    try:
        from scipy.io import netcdf_file
    except ImportError as exc:
        raise ImportError("scipy is required for streaming inference output") from exc

    split = _validate_prediction_split(split)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    expected_mask_shape = (len(target_lat), len(target_lon))
    if valid_mask.shape != expected_mask_shape:
        raise ValueError(
            f"target valid mask {valid_mask.shape} != output grid "
            f"{expected_mask_shape}"
        )
    required_mask_attrs = (
        TARGET_VALID_MASK_SHA256_ATTR,
        TARGET_VALID_MASK_CONTENT_SHA256_ATTR,
        TARGET_VALID_MASK_CRITERION_ATTR,
        TARGET_VALID_MASK_SOURCE_SPLIT_ATTR,
        TARGET_VALID_MASK_TRAINING_SOURCE_SIGNATURE_ATTR,
        TARGET_VALID_MASK_GRID_FINGERPRINT_ATTR,
    )
    missing_mask_attrs = [
        name for name in required_mask_attrs if name not in mask_provenance
    ]
    if missing_mask_attrs:
        raise ValueError(
            "deterministic daily output lacks target-valid-mask provenance "
            f"attributes: {missing_mask_attrs}"
        )
    computed_mask_digest = _target_valid_mask_content_sha256(valid_mask)
    if (
        str(mask_provenance[TARGET_VALID_MASK_CONTENT_SHA256_ATTR])
        != computed_mask_digest
    ):
        raise ValueError(
            "deterministic daily output target-valid-mask content digest does "
            "not match the supplied mask cells"
        )
    if (
        str(mask_provenance[TARGET_VALID_MASK_CRITERION_ATTR])
        != TARGET_VALID_MASK_CRITERION
    ):
        raise ValueError("unsupported deterministic output mask criterion")
    if str(mask_provenance[TARGET_VALID_MASK_SOURCE_SPLIT_ATTR]) != "training":
        raise ValueError("deterministic output mask must come from training")
    output_grid_fingerprint = validate_prism_grid(
        target_lat, target_lon, context="NARR inference NetCDF"
    ).fingerprint
    if (
        str(mask_provenance[TARGET_VALID_MASK_GRID_FINGERPRINT_ATTR])
        != output_grid_fingerprint
    ):
        raise ValueError(
            "deterministic output target-valid-mask grid fingerprint does not "
            "match its output coordinates"
        )

    if out_path.exists():
        out_path.unlink()

    nc = netcdf_file(str(out_path), "w", version=2)
    nc.createDimension("time", None)
    nc.createDimension("lat", len(target_lat))
    nc.createDimension("lon", len(target_lon))

    time_var = nc.createVariable("time", "i", ("time",))
    time_var.units = "days since 1970-01-01"
    time_var.calendar = "proleptic_gregorian"

    # Preserve the exact canonical PRISM coordinates.  Float32 down-casting can
    # move an endpoint enough for a later inclusive slice to lose one grid cell.
    lat_var = nc.createVariable("lat", "d", ("lat",))
    lon_var = nc.createVariable("lon", "d", ("lon",))
    lat_var[:] = target_lat.astype(np.float64)
    lon_var[:] = target_lon.astype(np.float64)

    mask_var = nc.createVariable("prism_valid_mask", "b", ("lat", "lon"))
    mask_var[:] = valid_mask.astype(np.int8)
    mask_var.long_name = "canonical PRISM valid-cell mask"
    mask_var.flag_values = np.asarray([0, 1], dtype=np.int8)
    mask_var.flag_meanings = "invalid valid"

    for var in target_variables:
        out_var = nc.createVariable(var, "f", ("time", "lat", "lon"))
        out_var.long_name = var
        # tmax/tmin are degC, ppt is mm/day; default to "" for unknown vars.
        out_var.units = VAR_UNITS.get(var, "")

    nc.description = "NARR-to-PRISM downscaling inference output"
    nc.case_name = case_name
    nc.prism_grid_fingerprint = output_grid_fingerprint
    nc.checkpoint = checkpoint_path
    configured_dates = list(configured_split_dates or inference_dates)
    if not configured_dates:
        raise ValueError("configured_split_dates must not be empty")
    nc.dataset_split = split
    nc.split_start = str(configured_dates[0])
    nc.split_end = str(configured_dates[-1])
    # Retain historical attribute names for consumers of default inference
    # products. ``dataset_split`` is authoritative for validation products.
    nc.inference_start = str(inference_dates[0])
    nc.inference_end = str(inference_dates[-1])
    for name, value in mask_provenance.items():
        setattr(nc, str(name), str(value))
    nc.flush()
    return nc


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_inference(
    config_path: str,
    cfg: Dict[str, Any],
    config: Any,
    checkpoint_path: str,
    output_dir: str,
    device: torch.device,
    batch_size: int = 1,
    date_shard_index: int = 0,
    date_shard_count: int = 1,
    show_progress: bool = True,
    data_parallel: bool = True,
    selected_dates: Optional[Sequence[Any]] = None,
    split: str = "inference",
    include_observed_target_diagnostics: bool = False,
) -> Path:
    """Execute prediction over one exact YAML split and write daily NetCDF outputs.

    The PRISM grid (~3105x7025) is far too large for a single forward pass, so
    each day is predicted tile-by-tile and stitched with a Hann blend window.
    Pass ``selected_dates`` to evaluate a small, possibly non-contiguous subset
    while loading the model only once.

    Key correctness points (the previous version was broken on all three):
      * Predictors are regridded onto each PRISM tile (co-registered with the
        target) by the dataset's ``_load_predictor`` -- NOT fed as a raw,
        whole-domain coarse array that the model would merely stretch.
      * The dataset returns RAW physical inputs; the model normalizes them
        internally. We must NOT normalize the input here.
      * The model returns predictions in physical units (its ``_decode_outputs``
        applies the output scalers). We must NOT denormalize the output here.
    """
    if xr is None:
        raise ImportError("xarray is required for inference")

    split = _validate_prediction_split(split)
    data_cfg = cfg.get("data", {})
    inf_cfg = cfg.get("inference", {})
    if include_observed_target_diagnostics and split != "validation":
        raise ValueError(
            "observed-target stage diagnostics are allowed only for the "
            "validation split; final inference remains target-free"
        )
    if include_observed_target_diagnostics and not bool(
        (inf_cfg.get("diagnostics", {}) or {}).get("enabled", False)
    ):
        raise ValueError(
            "include_observed_target_diagnostics requires "
            "inference.diagnostics.enabled=true"
        )
    case_name = get_case_name(cfg)
    target_variables: List[str] = list(data_cfg.get("target_variables", []))

    # Log the active case + case-scoped directories, then require the per-case
    # scalers to exist. Inference must denormalize with the SAME per-case
    # scalers used at training; never silently borrow another case's files.
    log_case_context(cfg, "inference")
    assert_scalars_available(cfg, role="inference")

    # Point the model at the per-case scalers. The notebook calls run_inference
    # directly with a fresh config, so we must not rely on the CLI having wired
    # them; log shapes + sha256 to prove train/infer used identical files.
    apply_scalar_paths(config)
    log_scalar_summary(config, "inference")

    # Co-registered, raw-physical dataset (same class used for training). It
    # regrids NARR onto the PRISM grid on the fly and exposes the fine grid.
    dataset = NarrPrismDataset(
        config_path, mode=split, load_observed_targets=False
    )
    all_inference_dates = _validate_dataset_dates(cfg, dataset.dates, split)
    dates_to_process = _select_inference_dates(all_inference_dates, selected_dates)
    if date_shard_count < 1:
        raise ValueError("date_shard_count must be >= 1")
    if date_shard_index < 0 or date_shard_index >= date_shard_count:
        raise ValueError(
            f"date_shard_index must be in [0, {date_shard_count}), got {date_shard_index}"
        )
    inference_dates = [
        sample_date
        for idx, sample_date in enumerate(dates_to_process)
        if idx % date_shard_count == date_shard_index
    ]
    if not inference_dates:
        raise ValueError(
            f"Date shard {date_shard_index}/{date_shard_count} has no dates to process"
        )

    # Load the multi-gigabyte model only after cheap date/config validation.
    print(f"[inference] loading checkpoint: {checkpoint_path}")
    model = _load_model(config, checkpoint_path, device, data_parallel=data_parallel)
    n_gpu = torch.cuda.device_count() if device.type == "cuda" else 1
    if n_gpu > 1:
        # Feed each GPU several tiles per forward pass. A microbatch of one
        # tile/GPU makes DataParallel overhead dominate and leaves A100 memory
        # mostly idle.
        tiles_per_gpu = max(1, int(inf_cfg.get("tiles_per_gpu", 4)))
        batch_size = max(batch_size, n_gpu * tiles_per_gpu)
        print(
            f"[inference] using {n_gpu} GPUs; effective tile batch size = {batch_size} "
            f"({tiles_per_gpu} tiles/GPU)"
        )
    batch_size = max(batch_size, int(inf_cfg.get("batch_size", batch_size)))
    mixed_precision = bool(inf_cfg.get("mixed_precision", True)) and device.type == "cuda"
    amp_dtype_name = str(inf_cfg.get("mixed_precision_dtype", "bfloat16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name in {"bf16", "bfloat16"} else torch.float16
    amp_dtype_label = "bfloat16" if amp_dtype is torch.bfloat16 else "float16"
    print(
        f"[inference] device={device} cuda_devices={n_gpu} "
        f"mixed_precision={'on' if mixed_precision else 'off'}"
        f"{f' dtype={amp_dtype_label}' if mixed_precision else ''}"
    )

    target_lat = dataset.fine_lat
    target_lon = dataset.fine_lon
    fine_h, fine_w = dataset.fine_shape
    n_vars = len(target_variables)
    print(
        f"[inference] split={split} "
        f"{len(inference_dates)}/{len(all_inference_dates)} configured dates; "
        f"selected={len(dates_to_process)}; "
        f"date_shard={date_shard_index}/{date_shard_count}; PRISM grid {fine_h}x{fine_w}"
    )

    # Tile geometry (fine-grid pixels). Defaults mirror the training crop.
    tile_cfg = inf_cfg.get("inference_tile_size") or inf_cfg.get(
        "boundary_mitigation", {}
    ).get("tile_size") or [256, 256]
    overlap_cfg = inf_cfg.get("inference_overlap") or inf_cfg.get(
        "boundary_mitigation", {}
    ).get("overlap") or [128, 128]
    halo_cfg = inf_cfg.get("inference_halo") or inf_cfg.get(
        "boundary_mitigation", {}
    ).get("halo") or [0, 0]
    force_full_frame = bool(
        inf_cfg.get(
            "force_full_frame",
            inf_cfg.get("boundary_mitigation", {}).get("force_full_frame", False),
        )
    )
    if force_full_frame:
        tile_cfg = [fine_h, fine_w]
        overlap_cfg = [0, 0]
        # Keep the configured exterior halo so full-frame and tiled runs use
        # identical boundary context before retaining the same domain core.
    plan = TilePlan.build(
        (fine_h, fine_w), tile_cfg, overlap=overlap_cfg, halo=halo_cfg
    )
    if str(getattr(config.model, "backbone_attention_scope", "legacy_global")).lower() == "windowed_local":
        plan.assert_globally_aligned(getattr(config, "mask_unit_size", [16, 16]))
    tile_h, tile_w = plan.core_shape
    ov_h, ov_w = plan.overlap
    halo_h, halo_w = plan.halo

    pad_multiple = _pad_multiple_from_config(config)
    # Per-gridpoint (spatial) target scalers are cropped per tile by the model
    # using each tile's origin. Padding a tile would push edge-tile offsets past
    # the scaler grid (silent wrong bilinear fallback), so require pad-free tiles
    # and a scaler grid that matches this inference domain. Channel-only scalers
    # are grid-agnostic and skip all of this.
    scalar_dir = resolve_scalar_dir(config, for_writing=False)
    spatial_targets = targets_are_spatial(scalar_dir)
    if spatial_targets:
        _tmean = np.load(os.path.join(str(scalar_dir), "targets_mean.npy"), mmap_mode="r")
        assert_target_grid_matches("targets_mean", _tmean, fine_h, fine_w)
        print(
            f"[inference] per-gridpoint target scalers active: grid "
            f"{tuple(_tmean.shape[-2:])} == domain ({fine_h},{fine_w}); "
            f"passing per-tile __scaler_offset"
        )
    lat_origins = list(plan.lat_origins)
    lon_origins = list(plan.lon_origins)
    candidate_tile_positions = plan.positions
    skip_empty_target_tiles = bool(data_cfg.get("skip_empty_target_tiles", True))
    min_valid_target_fraction = float(data_cfg.get("min_valid_target_fraction", 1.0e-4))
    # Production inference is predictor-only. Its static support is a separately
    # signed artifact derived from the configured training targets; finite
    # neutral-filled target scalers and inference-period PRISM observations are
    # both invalid substitutes.
    target_valid_mask = load_target_valid_mask(
        config,
        role="NARR deterministic inference",
        expected_shape=(fine_h, fine_w),
    )
    target_valid_mask_provenance = _target_valid_mask_provenance(
        config, target_valid_mask
    )
    if skip_empty_target_tiles and min_valid_target_fraction > 0.0:
        tile_positions = [
            (lat0, lon0)
            for lat0, lon0 in candidate_tile_positions
            if float(target_valid_mask[lat0 : lat0 + tile_h, lon0 : lon0 + tile_w].mean())
            >= min_valid_target_fraction
        ]
    else:
        tile_positions = candidate_tile_positions
    if not tile_positions:
        raise ValueError("No inference tiles remain after target-mask filtering")
    n_tiles = len(tile_positions)
    batch_size = _cap_tile_batch_for_cuda_indexing(
        batch_size=batch_size,
        config=config,
        tile_h=plan.context_shape[0],
        tile_w=plan.context_shape[1],
        device=device,
        inf_cfg=inf_cfg,
    )
    print(
        f"[inference] tiling: core=({tile_h},{tile_w}) halo=({halo_h},{halo_w}) "
        f"context={plan.context_shape} overlap=({ov_h},{ov_w}) "
        f"-> {len(lat_origins)}x{len(lon_origins)} = "
        f"{len(candidate_tile_positions)} candidate, {n_tiles} kept/day; "
        f"batch={batch_size}"
    )
    cache_predictors = bool(inf_cfg.get("cache_regridded_predictors", True))
    if (halo_h or halo_w) and not cache_predictors:
        print(
            "[inference] enabling full-day predictor cache because halo extraction "
            "requires context outside each output core"
        )
        cache_predictors = True
    print(
        f"[inference] predictor cache={'on' if cache_predictors else 'off'}; "
        "timing stages: preprocessing transfer forward postprocess write"
    )

    output_path = case_output_dir(_split_output_root(output_dir, split), case_name)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"[inference] case_name={case_name}")
    print(f"[inference] output_dir={output_path}")
    diagnostics_cfg = inf_cfg.get("diagnostics", {}) or {}
    diagnostics_enabled = bool(diagnostics_cfg.get("enabled", False))
    diagnostics_max_days = max(0, int(diagnostics_cfg.get("max_days", 1)))
    diagnostics_dir = output_path / str(diagnostics_cfg.get("directory", "diagnostics"))
    if diagnostics_enabled:
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[inference] intermediate diagnostics enabled for first "
            f"{diagnostics_max_days} shard day(s): {diagnostics_dir}"
        )
    diagnostic_target_dataset: Optional[NarrPrismDataset] = None
    if include_observed_target_diagnostics:
        diagnostic_target_dataset = NarrPrismDataset(
            config_path,
            mode=split,
            load_observed_targets=True,
        )
        _validate_dataset_dates(cfg, diagnostic_target_dataset.dates, split)
        print(
            "[inference] validation observations are isolated to stage "
            "diagnostics and are not exposed to model prediction batches"
        )

    date_iter = enumerate(inference_dates)
    if tqdm is not None and show_progress:
        date_iter = tqdm(
            list(enumerate(inference_dates)),
            total=len(inference_dates),
            desc="NARR-PRISM inference (days)",
            unit="day",
        )

    write_count = 0
    run_times: Dict[str, float] = {}
    with torch.inference_mode():
        for _di, sample_date in date_iter:
            date_string = str(sample_date)
            diagnostic_active = diagnostics_enabled and _di < diagnostics_max_days
            day_times: Dict[str, float] = {
                "preprocessing": 0.0,
                "transfer": 0.0,
                "forward": 0.0,
                "postprocess": 0.0,
                "write": 0.0,
            }

            # Float64 accumulation avoids order-dependent roundoff across many
            # overlaps. Only output cores are accumulated; halo pixels are never
            # retained.
            stitcher = WeightedTileStitcher(n_vars, (fine_h, fine_w))
            blend_mode = str(
                inf_cfg.get(
                    "inference_blend_window",
                    inf_cfg.get("blend_window", inf_cfg.get("boundary_mitigation", {}).get("blend_mode", "hann")),
                )
            ).lower()
            win = blend_window(plan.core_shape, plan.overlap, mode=blend_mode)
            tile_predictions: Dict[Tuple[int, int], np.ndarray] = {}
            raw_tile_predictions: Dict[Tuple[int, int], np.ndarray] = {}

            day_predictors: Optional[torch.Tensor] = None
            if cache_predictors:
                t0 = time.perf_counter()
                day_predictors = dataset._load_predictor_day(sample_date)
                _add_time(day_times, "preprocessing", time.perf_counter() - t0)

            start = 0
            current_batch_size = batch_size
            while start < len(tile_positions):
                chunk = tile_positions[start : start + current_batch_size]

                xb_cpu = xb = yb = pred = raw_pred = pred_np = None
                xs: List[torch.Tensor] = []
                try:
                    # Raw-physical, co-registered predictor tiles (model normalizes
                    # internally -> do NOT normalize here). All tiles share the same
                    # (tile_h, tile_w) shape, so they stack into one batch.
                    t0 = time.perf_counter()
                    for lat0, lon0 in chunk:
                        if day_predictors is None:
                            day_predictors = dataset._load_predictor_day(sample_date)
                        x = extract_halo_context(
                            day_predictors,
                            (lat0, lon0),
                            plan.core_shape,
                            plan.halo,
                        )
                        xs.append(_pad_to_multiple(x.unsqueeze(0), pad_multiple))
                    xb_cpu = torch.cat(xs, dim=0)
                    scaler_offsets_cpu = torch.tensor(chunk, dtype=torch.long)
                    _add_time(day_times, "preprocessing", time.perf_counter() - t0)

                    t0 = time.perf_counter()
                    xb = xb_cpu.to(device, non_blocking=True)
                    scaler_offsets = scaler_offsets_cpu.to(device, non_blocking=True)

                    # Real tensor (not a shape-only stub) so DataParallel scatters it
                    # along the batch dim consistently with x.
                    yb = torch.zeros(
                        (xb.shape[0], n_vars, tile_h, tile_w),
                        dtype=xb.dtype,
                        device=device,
                    )
                    _sync_if_cuda(device)
                    _add_time(day_times, "transfer", time.perf_counter() - t0)

                    t0 = time.perf_counter()
                    amp_context = (
                        torch.autocast(device_type="cuda", dtype=amp_dtype)
                        if mixed_precision
                        else nullcontext()
                    )
                    with amp_context:
                        input_offsets = scaler_offsets - torch.tensor(
                            [halo_h, halo_w], dtype=torch.long, device=device
                        )
                        output_crop = torch.tensor(
                            [plan.output_crop] * xb.shape[0],
                            dtype=torch.long,
                            device=device,
                        )
                        model_batch = {
                                "x": xb,
                                "y": yb,
                                # Legacy key remains for old checkpoints/models.
                                "__scaler_offset": scaler_offsets,
                                "__input_scaler_offset": input_offsets,
                                "__output_scaler_offset": scaler_offsets,
                                "__output_crop": output_crop,
                        }
                        model_result = model(
                            model_batch, return_raw_output=diagnostic_active
                        )
                        if diagnostic_active:
                            pred, raw_pred = model_result
                        else:
                            pred = model_result
                    if isinstance(pred, dict):
                        pred = pred.get(
                            "y_hat", pred.get("output", next(iter(pred.values())))
                        )
                    # New models crop the normalized prediction before spatial
                    # decoding. This fallback preserves compatibility with legacy
                    # models when halo is zero.
                    pred = pred[..., :tile_h, :tile_w]
                    _sync_if_cuda(device)
                    _add_time(day_times, "forward", time.perf_counter() - t0)

                    t0 = time.perf_counter()
                    # Model output is already physical units -> write directly.
                    pred_np = _tensor_to_float32_numpy(pred)
                    raw_pred_np = (
                        _tensor_to_float32_numpy(raw_pred)
                        if raw_pred is not None
                        else None
                    )

                    for i, (lat0, lon0) in enumerate(chunk):
                        tile_predictions[(lat0, lon0)] = pred_np[i]
                        if raw_pred_np is not None:
                            raw_tile_predictions[(lat0, lon0)] = raw_pred_np[i]
                        stitcher.add(pred_np[i], (lat0, lon0), win)
                    _add_time(day_times, "postprocess", time.perf_counter() - t0)
                    start += len(chunk)
                except BaseException as exc:
                    if not _is_cuda_oom(exc) or len(chunk) <= 1:
                        raise
                    reduced_batch_size = max(1, len(chunk) // 2)
                    print(
                        f"[inference] CUDA OOM at {date_string} tile "
                        f"{start + 1}/{len(tile_positions)} with batch={len(chunk)}; "
                        f"retrying with batch={reduced_batch_size}",
                        flush=True,
                    )
                    current_batch_size = reduced_batch_size
                    _clear_cuda_memory(device)
                    continue
                finally:
                    del xb_cpu, xb, yb, pred, raw_pred, pred_np, xs

            t0 = time.perf_counter()
            # Blend: divide the weighted sum by accumulated weights. Ocean-only
            # tiles may be skipped, but every valid target cell must be covered.
            if target_valid_mask is None:
                if np.any(stitcher.weight <= 0.0):
                    raise ValueError("Tiled inference left domain cells uncovered")
            elif np.any(target_valid_mask & (stitcher.weight <= 0.0)):
                raise ValueError("Tiled inference left valid PRISM cells uncovered")
            prediction = stitcher.finalize(require_full_coverage=False).astype(np.float32)
            prediction = prediction[np.newaxis]  # (1, C, H, W) for the helpers
            if target_valid_mask is not None:
                prediction[:, :, ~target_valid_mask] = np.nan

            _validate_prediction_batch(
                prediction,
                target_lat,
                target_lon,
                target_variables,
                valid_mask=target_valid_mask,
            )
            _print_channel_stats(
                prediction,
                target_variables,
                f"[inference] {date_string} stitched output statistics (physical units):",
            )
            # Physical-range guard (raises on impossible temperatures); reports
            # tmax<tmin crossings without failing.
            _sanity_check_outputs(prediction, target_variables, date_string)

            # Measure the actual equal-weight crossover bands (not merely tile
            # origins) and disagreement between unblended neighboring cores.
            lat_crossovers = overlap_crossovers(lat_origins, tile_h)
            lon_crossovers = overlap_crossovers(lon_origins, tile_w)
            for _ci, _vname in enumerate(target_variables):
                _field = prediction[0, _ci]
                _r_lat = boundary_gradient_ratio(_field, lat_crossovers, axis=0)
                _r_lon = boundary_gradient_ratio(_field, lon_crossovers, axis=1)
                _flag = (
                    "  <-- possible block artifact"
                    if (np.isfinite(_r_lat) and _r_lat > 1.5)
                    or (np.isfinite(_r_lon) and _r_lon > 1.5)
                    else ""
                )
                print(
                    f"[inference] {date_string} overlap-crossover gradient ratio {_vname}: "
                    f"lat={_r_lat:.2f} lon={_r_lon:.2f} (1.0=background){_flag}"
                )
            pair_metrics: List[Dict[str, float]] = []
            for lat0 in lat_origins:
                for left, right in zip(
                    lon_origins[:-1], lon_origins[1:], strict=True
                ):
                    if (lat0, left) in tile_predictions and (lat0, right) in tile_predictions:
                        pair_metrics.append(
                            overlap_disagreement(
                                tile_predictions[(lat0, left)], (lat0, left),
                                tile_predictions[(lat0, right)], (lat0, right),
                            )
                        )
            for lon0 in lon_origins:
                for top, bottom in zip(
                    lat_origins[:-1], lat_origins[1:], strict=True
                ):
                    if (top, lon0) in tile_predictions and (bottom, lon0) in tile_predictions:
                        pair_metrics.append(
                            overlap_disagreement(
                                tile_predictions[(top, lon0)], (top, lon0),
                                tile_predictions[(bottom, lon0)], (bottom, lon0),
                            )
                        )
            finite_rmse = [m["rmse"] for m in pair_metrics if np.isfinite(m["rmse"])]
            if finite_rmse:
                print(
                    f"[inference] {date_string} preblend overlap disagreement: "
                    f"mean_rmse={np.mean(finite_rmse):.4f} degC/mm "
                    f"max_rmse={np.max(finite_rmse):.4f} pairs={len(finite_rmse)}"
                )

            if diagnostic_active:
                channel_indices = [
                    int(idx) for idx in diagnostics_cfg.get("predictor_channels", [0])
                ]
                if day_predictors is None:
                    raise RuntimeError("diagnostic predictor cache was not populated")
                n_predictor_channels = int(day_predictors.shape[0])
                channel_indices = [
                    idx for idx in channel_indices if 0 <= idx < n_predictor_channels
                ]
                if not channel_indices:
                    channel_indices = [0]
                input_mean = np.load(Path(config.model.input_mu), mmap_mode="r")
                input_std = np.load(Path(config.model.input_sigma), mmap_mode="r")
                regridded = day_predictors[channel_indices].cpu().numpy().astype(np.float32)
                normalized = (
                    regridded
                    - np.asarray(input_mean[channel_indices], dtype=np.float32)[:, None, None]
                ) / (
                    np.asarray(input_std[channel_indices], dtype=np.float32)[:, None, None]
                    + float(getattr(config, "input_scalers_epsilon", 1.0e-6))
                )
                tile_order = list(tile_predictions)
                diagnostic_payload: Dict[str, Any] = {
                    "date": np.asarray(date_string),
                    "lat": np.asarray(target_lat, dtype=np.float64),
                    "lon": np.asarray(target_lon, dtype=np.float64),
                    "predictor_channel_indices": np.asarray(channel_indices, dtype=np.int32),
                    "regridded_predictors": regridded,
                    "normalized_predictors": normalized.astype(np.float32),
                    "tile_origins": np.asarray(tile_order, dtype=np.int32),
                    "individual_tile_predictions": np.stack(
                        [tile_predictions[pos] for pos in tile_order]
                    ),
                    "tile_weight_sum": stitcher.weight.astype(np.float32),
                    "stitched_denormalized_output": prediction[0],
                    "overlap_crossovers_lat": np.asarray(lat_crossovers, dtype=np.int32),
                    "overlap_crossovers_lon": np.asarray(lon_crossovers, dtype=np.int32),
                }
                if diagnostic_target_dataset is not None:
                    truth = diagnostic_target_dataset._load_targets(
                        sample_date, slice(None), slice(None)
                    ).cpu().numpy().astype(np.float32)
                    diagnostic_payload["prism_target"] = truth
                    diagnostic_payload["inference_minus_prism"] = (
                        prediction[0] - truth
                    )
                else:
                    diagnostic_payload["target_free_inference"] = np.asarray(
                        True
                    )
                if raw_tile_predictions:
                    diagnostic_payload["normalized_tile_predictions"] = np.stack(
                        [raw_tile_predictions[pos] for pos in tile_order]
                    )
                diagnostic_payload.update(
                    _load_native_predictor_diagnostic(dataset, sample_date)
                )
                diagnostic_path = diagnostics_dir / f"stages_{date_string}.npz"
                np.savez_compressed(diagnostic_path, **diagnostic_payload)
                print(f"[inference] wrote intermediate stage diagnostics: {diagnostic_path}")
            _add_time(day_times, "postprocess", time.perf_counter() - t0)

            date_token = np.datetime64(date_string, "D").astype(object).strftime("%Y%m%d")
            out_path = output_path / f"{case_name}_inference_{date_token}.nc"
            temporary_path = out_path.with_name(
                f".{out_path.name}.{os.getpid()}.tmp"
            )
            t0 = time.perf_counter()
            out_nc = _open_streaming_output(
                temporary_path,
                target_variables,
                target_lat,
                target_lon,
                checkpoint_path,
                [date_string],
                case_name,
                target_valid_mask,
                target_valid_mask_provenance,
                split,
                all_inference_dates,
            )
            try:
                out_nc.variables["time"][:] = _date_strings_to_epoch_days([date_string])
                for ch_idx, var in enumerate(target_variables):
                    out_nc.variables[var][0:1, :, :] = prediction[:, ch_idx, :, :]
                out_nc.flush()
            finally:
                out_nc.close()
            os.replace(temporary_path, out_path)
            _add_time(day_times, "write", time.perf_counter() - t0)
            write_count += 1
            for key, value in day_times.items():
                _add_time(run_times, key, value)
            print(f"[timing] {date_string} {_format_timing(day_times)}")
            del day_predictors, stitcher, tile_predictions, raw_tile_predictions, prediction

    if write_count != len(inference_dates):
        raise RuntimeError(
            f"Wrote {write_count} daily files, expected {len(inference_dates)}"
        )

    print(f"[timing] aggregate {_format_timing(run_times)} days={write_count}")
    print(f"[inference] daily outputs saved -> {output_path}")

    return output_path


def run_parallel_inference(
    config_path: str,
    checkpoint_path: str,
    output_dir: str,
    gpu_ids: Sequence[str],
    batch_size: int,
    device: str = "cuda",
    split: str = "inference",
    include_observed_target_diagnostics: bool = False,
) -> Path:
    """Run independent date shards in one subprocess per GPU.

    This is faster than ``torch.nn.DataParallel`` for this workload because each
    daily output is independent and tile batches are relatively small. Each
    worker sees one CUDA device, avoids scatter/gather, and writes a disjoint
    set of daily NetCDF files into the same case output directory.
    """
    split = _validate_prediction_split(split)
    if include_observed_target_diagnostics and split != "validation":
        raise ValueError(
            "observed-target stage diagnostics are allowed only for validation"
        )
    cfg = load_yaml(config_path)
    case_name = get_case_name(cfg)
    # Fail fast with a clear, case-specific message before spawning workers if
    # the per-case scalers are missing (this is where the stale shared-scalar
    # crash used to surface).
    log_case_context(cfg, "inference")
    assert_scalars_available(cfg, role="inference")
    split_output_dir = _split_output_root(output_dir, split)
    output_path = case_output_dir(split_output_dir, case_name)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset = NarrPrismDataset(
        config_path, mode=split, load_observed_targets=False
    )
    total_dates = len(_validate_dataset_dates(cfg, dataset.dates, split))
    del dataset

    script_path = Path(__file__).resolve()
    n_workers = len(gpu_ids)
    if n_workers < 1:
        raise ValueError("At least one GPU id is required for parallel inference")
    if total_dates < n_workers:
        raise ValueError(
            f"Parallel inference has {total_dates} date(s) but {n_workers} worker(s). "
            f"Use fewer GPUs or expand dates.{split}."
        )

    print(
        f"[parallel] launching {n_workers} inference workers over GPUs "
        f"{','.join(gpu_ids)} for split={split} {total_dates} dates -> "
        f"{output_path}",
        flush=True,
    )
    procs: List[subprocess.Popen] = []
    log_handles: List[Any] = []
    worker_logs: List[Path] = []
    run_started_ns = time.time_ns()
    try:
        for shard_idx, gpu_id in enumerate(gpu_ids):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            log_path = output_path / f"parallel_worker_{shard_idx}_gpu{gpu_id}.log"
            cmd = [
                sys.executable,
                str(script_path),
                "--config",
                str(config_path),
                "--checkpoint",
                str(checkpoint_path),
                "--output-dir",
                str(split_output_dir),
                "--batch-size",
                str(batch_size),
                "--device",
                device,
                "--date-shard-index",
                str(shard_idx),
                "--date-shard-count",
                str(n_workers),
                "--no-data-parallel",
                "--no-progress",
                "--split",
                split,
            ]
            if include_observed_target_diagnostics:
                cmd.append("--include-observed-target-diagnostics")
            print(
                f"[parallel] worker {shard_idx}: GPU {gpu_id}, "
                f"log: {log_path}, command: {' '.join(cmd)}",
                flush=True,
            )
            log_fh = open(log_path, "w", encoding="utf-8")
            log_handles.append(log_fh)
            worker_logs.append(log_path)
            procs.append(
                subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )

        failures: List[Tuple[int, int]] = []
        progress = (
            tqdm(total=total_dates, desc="NARR-PRISM inference (days)", unit="day")
            if tqdm is not None
            else None
        )
        last_done = 0
        try:
            while True:
                # Count only files written by this invocation. This keeps a resumed
                # or repeated run from starting the bar with stale outputs included.
                done = sum(
                    path.stat().st_mtime_ns >= run_started_ns
                    for path in output_path.glob(f"{case_name}_inference_*.nc")
                )
                done = min(done, total_dates)
                live = sum(1 for proc in procs if proc.poll() is None)
                if progress is not None:
                    progress.n = done
                    progress.set_postfix(live_workers=live, refresh=True)
                elif done != last_done:
                    print(f"[parallel] completed {done}/{total_dates} daily files", flush=True)
                last_done = done

                failures = [
                    (worker_idx, proc.returncode)
                    for worker_idx, proc in enumerate(procs)
                    if proc.poll() not in (None, 0)
                ]
                if failures or all(proc.poll() is not None for proc in procs):
                    break
                time.sleep(2.0)
        finally:
            if progress is not None:
                progress.close()

        if failures:
            for log_fh in log_handles:
                log_fh.flush()
            raise RuntimeError(
                "Parallel inference worker failure(s): "
                + ", ".join(
                    f"worker {idx} rc={rc} log={worker_logs[idx]}"
                    for idx, rc in failures
                )
                + "\n\n"
                + "\n\n".join(
                    f"--- tail worker {idx} ({worker_logs[idx]}) ---\n"
                    + _tail_text_file(worker_logs[idx])
                    for idx, _rc in failures
                )
            )
    except BaseException:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
        raise
    finally:
        for log_fh in log_handles:
            log_fh.close()

    print(f"[parallel] all workers finished -> {output_path}", flush=True)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NARR-PRISM inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the same YAML used for training (e.g., NARR_PRISM_subdomain.yaml)",
    )
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument(
        "--split",
        choices=PREDICTION_SPLITS,
        default="inference",
        help=(
            "YAML date split to predict; validation outputs are isolated in a "
            "validation subdirectory"
        ),
    )
    parser.add_argument(
        "--include-observed-target-diagnostics",
        action="store_true",
        help=(
            "validation only: include PRISM truth in explicitly enabled stage "
            "diagnostics; prediction batches remain target-free"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument(
        "--parallel-gpus",
        default=None,
        help="Comma-separated physical GPU ids for process-level date sharding, e.g. 0,1",
    )
    parser.add_argument("--date-shard-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--date-shard-count", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--no-progress", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--no-data-parallel",
        action="store_true",
        help="Disable torch.nn.DataParallel even if multiple CUDA devices are visible",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    config = get_config(str(Path(args.config).resolve()))

    # Scaler resolution + wiring + logging happen inside run_inference /
    # run_parallel_inference so the notebook (which calls those directly)
    # behaves identically to this CLI entry point.
    checkpoint = _find_checkpoint(cfg, args.checkpoint)
    output_root = args.output_dir or cfg.get("inference", {}).get(
        "output_dir", "./examples/NARR_PRISM/experiments/inference_output"
    )
    output_dir = str(_split_output_root(output_root, args.split))

    if args.parallel_gpus and args.date_shard_count == 1:
        gpu_ids = [gpu.strip() for gpu in args.parallel_gpus.split(",") if gpu.strip()]
        run_parallel_inference(
            config_path=str(Path(args.config).resolve()),
            checkpoint_path=checkpoint,
            output_dir=output_dir,
            gpu_ids=gpu_ids,
            batch_size=args.batch_size,
            device=args.device,
            split=args.split,
            include_observed_target_diagnostics=(
                args.include_observed_target_diagnostics
            ),
        )
        return

    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[inference] CUDA not available, falling back to CPU")
        device_str = "cpu"
    device = torch.device(device_str)

    run_inference(
        config_path=str(Path(args.config).resolve()),
        cfg=cfg,
        config=config,
        checkpoint_path=checkpoint,
        output_dir=output_dir,
        device=device,
        batch_size=args.batch_size,
        date_shard_index=args.date_shard_index,
        date_shard_count=args.date_shard_count,
        show_progress=not args.no_progress,
        data_parallel=not args.no_data_parallel,
        split=args.split,
        include_observed_target_diagnostics=(
            args.include_observed_target_diagnostics
        ),
    )


if __name__ == "__main__":
    main()
