"""Inference for the MERRA2-to-PRISM downscaling model.

Loads a trained checkpoint, runs prediction over the YAML-defined inference
date range, denormalizes outputs, and writes NetCDF files.

Usage:
    python merra_prism_inference.py --config MERRA_PRISM.yaml [--checkpoint path/to/best.ckpt]
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
from granitewxc.utils.predictands import build_predictand_specs

from merra_prism_dataset import MerraPrismDataset
from merra_prism_utils import (
    case_output_dir,
    expand_predictor_variables,
    get_case_name,
    load_yaml,
    parse_date_range_from_config,
    resolve_path,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_checkpoint(cfg: Dict[str, Any], explicit: Optional[str]) -> str:
    """Locate the model checkpoint to use for inference."""
    if explicit:
        return str(resolve_path(explicit))
    case_name = get_case_name(cfg)
    inf_cfg = cfg.get("inference", {})
    ckpt = inf_cfg.get("checkpoint_path")
    if ckpt:
        return str(resolve_path(ckpt))

    checkpoint_dir = cfg.get("checkpoint_dir")
    if checkpoint_dir:
        case_checkpoint_dir = case_output_dir(checkpoint_dir, case_name)
        for candidate in ("best.ckpt", "last.ckpt"):
            p = case_checkpoint_dir / candidate
            if p.exists():
                return str(p)

    run_dir = cfg.get("run_dir")
    if run_dir:
        case_run_dir = case_output_dir(run_dir, case_name)
        for candidate in ("best.ckpt", "last.ckpt"):
            p = case_run_dir / candidate
            if p.exists():
                return str(p)
        for candidate in ("best.ckpt", "last.ckpt"):
            p = case_run_dir / "checkpoints" / candidate
            if p.exists():
                return str(p)

    exp = cfg.get("path_experiment", ".")
    for candidate in ("best.ckpt", "last.ckpt"):
        p = case_output_dir(resolve_path(exp) / "checkpoints", case_name) / candidate
        if p.exists():
            return str(p)
    raise FileNotFoundError("Cannot locate a trained checkpoint. Pass --checkpoint explicitly.")


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

    model.load_state_dict(cleaned, strict=False)
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
    if tile >= total:
        return [0]
    origins = list(range(0, total - tile + 1, stride))
    if origins[-1] != total - tile:
        origins.append(total - tile)
    return origins


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


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        for idx in range(torch.cuda.device_count()):
            torch.cuda.synchronize(idx)


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


def _open_streaming_output(
    out_path: Path,
    target_variables: Sequence[str],
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    checkpoint_path: str,
    inference_dates: Sequence[date],
    case_name: str,
) -> Any:
    """Create a NetCDF3 file that can be filled along time without buffering."""
    try:
        from scipy.io import netcdf_file
    except ImportError as exc:
        raise ImportError("scipy is required for streaming inference output") from exc

    if out_path.exists():
        out_path.unlink()

    nc = netcdf_file(str(out_path), "w", version=2)
    nc.createDimension("time", None)
    nc.createDimension("lat", len(target_lat))
    nc.createDimension("lon", len(target_lon))

    time_var = nc.createVariable("time", "i", ("time",))
    time_var.units = "days since 1970-01-01"
    time_var.calendar = "proleptic_gregorian"

    lat_var = nc.createVariable("lat", "f", ("lat",))
    lon_var = nc.createVariable("lon", "f", ("lon",))
    lat_var[:] = target_lat.astype(np.float32)
    lon_var[:] = target_lon.astype(np.float32)

    for var in target_variables:
        out_var = nc.createVariable(var, "f", ("time", "lat", "lon"))
        out_var.long_name = var
        # tmax/tmin are degC, ppt is mm/day; default to "" for unknown vars.
        out_var.units = VAR_UNITS.get(var, "")

    nc.description = "MERRA2-to-PRISM downscaling inference output"
    nc.case_name = case_name
    nc.checkpoint = checkpoint_path
    nc.inference_start = str(inference_dates[0])
    nc.inference_end = str(inference_dates[-1])
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
    data_parallel: bool = True,
) -> Path:
    """Execute inference over the YAML-defined date range and write daily NetCDF outputs.

    The PRISM grid (~3105x7025) is far too large for a single forward pass, so
    each day is predicted tile-by-tile and stitched with a Hann blend window.

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

    data_cfg = cfg.get("data", {})
    inf_cfg = cfg.get("inference", {})
    case_name = get_case_name(cfg)
    target_variables: List[str] = list(data_cfg.get("target_variables", []))

    # Load model
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

    # Co-registered, raw-physical dataset (same class used for training). It
    # regrids MERRA2 onto the PRISM grid on the fly and exposes the fine grid.
    dataset = MerraPrismDataset(config_path, mode="inference")
    all_inference_dates = dataset.dates
    if date_shard_count < 1:
        raise ValueError("date_shard_count must be >= 1")
    if date_shard_index < 0 or date_shard_index >= date_shard_count:
        raise ValueError(
            f"date_shard_index must be in [0, {date_shard_count}), got {date_shard_index}"
        )
    inference_dates = [
        sample_date
        for idx, sample_date in enumerate(all_inference_dates)
        if idx % date_shard_count == date_shard_index
    ]
    if not inference_dates:
        raise ValueError(
            f"Date shard {date_shard_index}/{date_shard_count} has no dates to process"
        )
    target_lat = dataset.fine_lat
    target_lon = dataset.fine_lon
    fine_h, fine_w = dataset.fine_shape
    n_vars = len(target_variables)
    print(
        f"[inference] {len(inference_dates)}/{len(all_inference_dates)} dates; "
        f"date_shard={date_shard_index}/{date_shard_count}; PRISM grid {fine_h}x{fine_w}"
    )

    # Tile geometry (fine-grid pixels). Defaults mirror the training crop.
    tile_cfg = inf_cfg.get("inference_tile_size") or inf_cfg.get(
        "boundary_mitigation", {}
    ).get("tile_size") or [256, 256]
    overlap_cfg = inf_cfg.get("inference_overlap") or inf_cfg.get(
        "boundary_mitigation", {}
    ).get("overlap") or [64, 64]
    tile_h = min(int(tile_cfg[0]), fine_h)
    tile_w = min(int(tile_cfg[1]), fine_w)
    ov_h = int(overlap_cfg[0])
    ov_w = int(overlap_cfg[1])
    stride_h = max(1, tile_h - ov_h)
    stride_w = max(1, tile_w - ov_w)

    pad_multiple = _pad_multiple_from_config(config)
    lat_origins = _tile_origins(fine_h, tile_h, stride_h)
    lon_origins = _tile_origins(fine_w, tile_w, stride_w)
    candidate_tile_positions = [
        (lat0, lon0) for lat0 in lat_origins for lon0 in lon_origins
    ]
    skip_empty_target_tiles = bool(data_cfg.get("skip_empty_target_tiles", True))
    min_valid_target_fraction = float(data_cfg.get("min_valid_target_fraction", 1.0e-4))
    if skip_empty_target_tiles and min_valid_target_fraction > 0.0:
        target_valid_mask = dataset._load_target_valid_mask()
        tile_positions = [
            (lat0, lon0)
            for lat0, lon0 in candidate_tile_positions
            if float(target_valid_mask[lat0 : lat0 + tile_h, lon0 : lon0 + tile_w].mean())
            >= min_valid_target_fraction
        ]
    else:
        target_valid_mask = None
        tile_positions = candidate_tile_positions
    if not tile_positions:
        raise ValueError("No inference tiles remain after target-mask filtering")
    n_tiles = len(tile_positions)
    batch_size = _cap_tile_batch_for_cuda_indexing(
        batch_size=batch_size,
        config=config,
        tile_h=tile_h,
        tile_w=tile_w,
        device=device,
        inf_cfg=inf_cfg,
    )
    print(
        f"[inference] tiling: tile=({tile_h},{tile_w}) overlap=({ov_h},{ov_w}) "
        f"-> {len(lat_origins)}x{len(lon_origins)} = "
        f"{len(candidate_tile_positions)} candidate, {n_tiles} kept/day; "
        f"batch={batch_size}"
    )
    cache_predictors = bool(inf_cfg.get("cache_regridded_predictors", True))
    print(
        f"[inference] predictor cache={'on' if cache_predictors else 'off'}; "
        "timing stages: preprocessing transfer forward postprocess write"
    )

    output_path = case_output_dir(output_dir, case_name)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"[inference] case_name={case_name}")
    print(f"[inference] output_dir={output_path}")

    date_iter = enumerate(inference_dates)
    if tqdm is not None:
        date_iter = tqdm(
            list(enumerate(inference_dates)),
            total=len(inference_dates),
            desc="MERRA-PRISM inference (days)",
            unit="day",
        )

    write_count = 0
    run_times: Dict[str, float] = {}
    with torch.inference_mode():
        for _di, sample_date in date_iter:
            date_string = str(sample_date)
            day_times: Dict[str, float] = {
                "preprocessing": 0.0,
                "transfer": 0.0,
                "forward": 0.0,
                "postprocess": 0.0,
                "write": 0.0,
            }

            # Accumulators for the stitched full-grid prediction.
            accum = np.zeros((n_vars, fine_h, fine_w), dtype=np.float32)
            weight = np.zeros((fine_h, fine_w), dtype=np.float32)

            win = _hann_window_2d(tile_h, tile_w)

            day_predictors: Optional[torch.Tensor] = None
            if cache_predictors:
                t0 = time.perf_counter()
                day_predictors = dataset._load_predictor_day(sample_date)
                _add_time(day_times, "preprocessing", time.perf_counter() - t0)

            start = 0
            current_batch_size = batch_size
            while start < len(tile_positions):
                chunk = tile_positions[start : start + current_batch_size]

                xb_cpu = xb = yb = pred = pred_np = None
                xs: List[torch.Tensor] = []
                try:
                    # Raw-physical, co-registered predictor tiles (model normalizes
                    # internally -> do NOT normalize here). All tiles share the same
                    # (tile_h, tile_w) shape, so they stack into one batch.
                    t0 = time.perf_counter()
                    for lat0, lon0 in chunk:
                        lat_slice = slice(lat0, lat0 + tile_h)
                        lon_slice = slice(lon0, lon0 + tile_w)
                        if day_predictors is None:
                            x = dataset._load_predictor(sample_date, lat_slice, lon_slice)
                        else:
                            x = day_predictors[:, lat_slice, lon_slice]
                        xs.append(_pad_to_multiple(x.unsqueeze(0), pad_multiple))
                    xb_cpu = torch.cat(xs, dim=0)
                    _add_time(day_times, "preprocessing", time.perf_counter() - t0)

                    t0 = time.perf_counter()
                    xb = xb_cpu.to(device, non_blocking=True)

                    # Real tensor (not a shape-only stub) so DataParallel scatters it
                    # along the batch dim consistently with x.
                    yb = torch.zeros(
                        (xb.shape[0], n_vars, xb.shape[-2], xb.shape[-1]),
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
                        pred = model({"x": xb, "y": yb})
                    if isinstance(pred, dict):
                        pred = pred.get(
                            "y_hat", pred.get("output", next(iter(pred.values())))
                        )
                    # Crop away the reflect padding -> back to tile size.
                    pred = pred[..., :tile_h, :tile_w]
                    _sync_if_cuda(device)
                    _add_time(day_times, "forward", time.perf_counter() - t0)

                    t0 = time.perf_counter()
                    # Model output is already physical units -> write directly.
                    pred_np = pred.detach().cpu().numpy().astype(np.float32)

                    for i, (lat0, lon0) in enumerate(chunk):
                        lat_slice = slice(lat0, lat0 + tile_h)
                        lon_slice = slice(lon0, lon0 + tile_w)
                        accum[:, lat_slice, lon_slice] += pred_np[i] * win[np.newaxis]
                        weight[lat_slice, lon_slice] += win
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
                    del xb_cpu, xb, yb, pred, pred_np, xs

            t0 = time.perf_counter()
            # Blend: normalize by accumulated Hann weights -> (C, H, W).
            prediction = np.full_like(accum, np.nan)
            valid_weight = weight > 0.0
            prediction[:, valid_weight] = accum[:, valid_weight] / weight[valid_weight]
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
            _add_time(day_times, "postprocess", time.perf_counter() - t0)

            date_token = np.datetime64(date_string, "D").astype(object).strftime("%Y%m%d")
            out_path = output_path / f"{case_name}_inference_{date_token}.nc"
            t0 = time.perf_counter()
            out_nc = _open_streaming_output(
                out_path,
                target_variables,
                target_lat,
                target_lon,
                checkpoint_path,
                [date_string],
                case_name,
            )
            try:
                out_nc.variables["time"][:] = _date_strings_to_epoch_days([date_string])
                for ch_idx, var in enumerate(target_variables):
                    out_nc.variables[var][0:1, :, :] = prediction[:, ch_idx, :, :]
                out_nc.flush()
            finally:
                out_nc.close()
            _add_time(day_times, "write", time.perf_counter() - t0)
            write_count += 1
            for key, value in day_times.items():
                _add_time(run_times, key, value)
            print(f"[timing] {date_string} {_format_timing(day_times)}")
            del day_predictors, accum, weight, prediction

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
) -> Path:
    """Run independent date shards in one subprocess per GPU.

    This is faster than ``torch.nn.DataParallel`` for this workload because each
    daily output is independent and tile batches are relatively small. Each
    worker sees one CUDA device, avoids scatter/gather, and writes a disjoint
    set of daily NetCDF files into the same case output directory.
    """
    cfg = load_yaml(config_path)
    case_name = get_case_name(cfg)
    output_path = case_output_dir(output_dir, case_name)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset = MerraPrismDataset(config_path, mode="inference")
    total_dates = len(dataset.dates)
    del dataset

    script_path = Path(__file__).resolve()
    n_workers = len(gpu_ids)
    if n_workers < 1:
        raise ValueError("At least one GPU id is required for parallel inference")
    if total_dates < n_workers:
        raise ValueError(
            f"Parallel inference has {total_dates} date(s) but {n_workers} worker(s). "
            "Use fewer GPUs or expand dates.inference."
        )

    print(
        f"[parallel] launching {n_workers} inference workers over GPUs "
        f"{','.join(gpu_ids)} for {total_dates} dates -> {output_path}",
        flush=True,
    )
    procs: List[subprocess.Popen] = []
    log_handles: List[Any] = []
    worker_logs: List[Path] = []
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
                str(output_dir),
                "--batch-size",
                str(batch_size),
                "--device",
                device,
                "--date-shard-index",
                str(shard_idx),
                "--date-shard-count",
                str(n_workers),
                "--no-data-parallel",
            ]
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
        last_done = -1
        last_report = 0.0
        while True:
            done = len(list(output_path.glob(f"{case_name}_inference_*.nc")))
            now = time.monotonic()
            if done != last_done or now - last_report >= 60.0:
                pct = 100.0 * done / max(total_dates, 1)
                live = sum(1 for proc in procs if proc.poll() is None)
                print(
                    f"[parallel] completed {done}/{total_dates} daily files "
                    f"({pct:.1f}%); live_workers={live}",
                    flush=True,
                )
                last_done = done
                last_report = now

            failures = [
                (worker_idx, proc.returncode)
                for worker_idx, proc in enumerate(procs)
                if proc.poll() not in (None, 0)
            ]
            if failures:
                break
            if all(proc.poll() is not None for proc in procs):
                break
            time.sleep(15.0)

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
        description="Run MERRA-PRISM inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to MERRA_PRISM.yaml")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument(
        "--parallel-gpus",
        default=None,
        help="Comma-separated physical GPU ids for process-level date sharding, e.g. 0,1",
    )
    parser.add_argument("--date-shard-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--date-shard-count", type=int, default=1, help=argparse.SUPPRESS)
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
    case_name = get_case_name(cfg)

    checkpoint = _find_checkpoint(cfg, args.checkpoint)
    output_root = args.output_dir or cfg.get("inference", {}).get(
        "output_dir", "./examples/MERRA_PRISM/experiments/inference_output"
    )
    output_dir = str(case_output_dir(output_root, case_name))

    if args.parallel_gpus and args.date_shard_count == 1:
        gpu_ids = [gpu.strip() for gpu in args.parallel_gpus.split(",") if gpu.strip()]
        run_parallel_inference(
            config_path=str(Path(args.config).resolve()),
            checkpoint_path=checkpoint,
            output_dir=output_dir,
            gpu_ids=gpu_ids,
            batch_size=args.batch_size,
            device=args.device,
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
        data_parallel=not args.no_data_parallel,
    )


if __name__ == "__main__":
    main()
