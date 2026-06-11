"""Inference for the MERRA2-to-PRISM downscaling model.

Loads a trained checkpoint, runs prediction over the YAML-defined inference
date range, denormalizes outputs, and writes NetCDF files.

Usage:
    python merra_prism_inference.py --config MERRA_PRISM.yaml [--checkpoint path/to/best.ckpt]
"""

from __future__ import annotations

import argparse
import os
import sys
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


def _load_model(config: Any, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Re-create the model architecture and load trained weights."""
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
        cleaned[key] = v

    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()
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
        print(f"[inference] WARNING: {n_bad} non-finite values detected in output")


def _validate_prediction_batch(
    prediction: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    target_variables: Sequence[str],
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
    if np.any(~np.isfinite(prediction)):
        n_bad = int(np.sum(~np.isfinite(prediction)))
        print(f"[inference] WARNING: {n_bad} non-finite values detected in output batch")


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
    case_name = get_case_name(cfg)
    target_variables: List[str] = list(data_cfg.get("target_variables", []))

    # Load model
    print(f"[inference] loading checkpoint: {checkpoint_path}")
    model = _load_model(config, checkpoint_path, device)

    # Co-registered, raw-physical dataset (same class used for training). It
    # regrids MERRA2 onto the PRISM grid on the fly and exposes the fine grid.
    dataset = MerraPrismDataset(config_path, mode="inference")
    inference_dates = dataset.dates
    target_lat = dataset.fine_lat
    target_lon = dataset.fine_lon
    fine_h, fine_w = dataset.fine_shape
    n_vars = len(target_variables)
    print(f"[inference] {len(inference_dates)} dates; PRISM grid {fine_h}x{fine_w}")

    # Tile geometry (fine-grid pixels). Defaults mirror the training crop.
    inf_cfg = cfg.get("inference", {})
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
    n_tiles = len(lat_origins) * len(lon_origins)
    print(
        f"[inference] tiling: tile=({tile_h},{tile_w}) overlap=({ov_h},{ov_w}) "
        f"-> {len(lat_origins)}x{len(lon_origins)} = {n_tiles} tiles/day"
    )

    output_path = Path(output_dir)
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
    with torch.inference_mode():
        for _di, sample_date in date_iter:
            date_string = str(sample_date)

            # Accumulators for the stitched full-grid prediction.
            accum = np.zeros((n_vars, fine_h, fine_w), dtype=np.float32)
            weight = np.zeros((fine_h, fine_w), dtype=np.float32)

            for lat0 in lat_origins:
                for lon0 in lon_origins:
                    lat_slice = slice(lat0, lat0 + tile_h)
                    lon_slice = slice(lon0, lon0 + tile_w)

                    # Raw-physical, co-registered predictor tile (model
                    # normalizes internally -> do NOT normalize here).
                    x = dataset._load_predictor(sample_date, lat_slice, lon_slice)
                    x = _pad_to_multiple(x.unsqueeze(0), pad_multiple).to(
                        device, non_blocking=True
                    )

                    y_shape = _ShapeOnlyTarget(1, n_vars, x.shape[-2:])
                    pred = model({"x": x, "y": y_shape})
                    if isinstance(pred, dict):
                        pred = pred.get(
                            "y_hat", pred.get("output", next(iter(pred.values())))
                        )
                    # Crop away the reflect padding -> back to tile size.
                    pred = pred[..., :tile_h, :tile_w]
                    # Model output is already physical units -> write directly.
                    pred_np = pred.detach().cpu().numpy().astype(np.float32)[0]

                    win = _hann_window_2d(tile_h, tile_w)
                    accum[:, lat_slice, lon_slice] += pred_np * win[np.newaxis]
                    weight[lat_slice, lon_slice] += win

                    del x, pred, pred_np
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

            # Blend: normalize by accumulated Hann weights -> (C, H, W).
            prediction = accum / np.maximum(weight, 1e-6)[np.newaxis]
            prediction = prediction[np.newaxis]  # (1, C, H, W) for the helpers

            _validate_prediction_batch(prediction, target_lat, target_lon, target_variables)
            _print_channel_stats(
                prediction,
                target_variables,
                f"[inference] {date_string} stitched output statistics (physical units):",
            )
            # Physical-range guard (raises on impossible temperatures); reports
            # tmax<tmin crossings without failing.
            _sanity_check_outputs(prediction, target_variables, date_string)

            date_token = np.datetime64(date_string, "D").astype(object).strftime("%Y%m%d")
            out_path = output_path / f"{case_name}_inference_{date_token}.nc"
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
            write_count += 1

    if write_count != len(inference_dates):
        raise RuntimeError(
            f"Wrote {write_count} daily files, expected {len(inference_dates)}"
        )

    print(f"[inference] daily outputs saved -> {output_path}")

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
    )


if __name__ == "__main__":
    main()
