"""Compute normalisation scalars for the MERRA2-to-PRISM downscaling workflow.

Statistics are computed **only** over the YAML-defined training period so that
the same scalars can be reused consistently during training and inference.

Usage:
    python compute_scalars_merra_prism.py --config MERRA_PRISM.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    import xarray as xr
except ImportError:
    xr = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from merra_prism_utils import (
    align_dates,
    discover_all_prism_targets,
    discover_merra2_files,
    expand_predictor_variables,
    load_elevation,
    load_yaml,
    parse_date_range_from_config,
    resolve_path,
    validate_target_variables,
)

try:
    from granitewxc.utils.config import get_config
    from granitewxc.utils.predictands import PredictandSpec, build_predictand_specs
except ImportError:
    get_config = None
    PredictandSpec = None
    build_predictand_specs = None

EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-channel scalars for MERRA-PRISM training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to MERRA_PRISM.yaml")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override scalar output directory (default: from YAML data.scalar_dir)",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=50,
        help="Print progress every N samples",
    )
    return parser.parse_args()


def _load_predictor_arrays(
    path: Path,
    variables: Sequence[Tuple[str, float]],
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> np.ndarray:
    """Load predictors from a MERRA2 file, regridded onto (target_lat, target_lon).

    Predictors are bilinearly interpolated onto the (strided) PRISM grid so the
    input scalars are computed from the SAME co-registered fields the dataset
    feeds the model at train/inference time.
    """
    arrays: List[np.ndarray] = []
    with xr.open_dataset(str(path)) as ds:
        lat_name = next((n for n in ("lat", "latitude", "y") if n in ds.coords or n in ds.dims), "lat")
        lon_name = next((n for n in ("lon", "longitude", "x") if n in ds.coords or n in ds.dims), "lon")
        for var, level in variables:
            if var not in ds.data_vars:
                raise ValueError(f"Variable '{var}' not found in {path}")
            da = ds[var]
            if "time" in da.dims:
                da = da.isel(time=0, drop=True)
            if "lev" in da.dims:
                da = da.sel(lev=level, drop=True)
            da = da.interp({lat_name: target_lat, lon_name: target_lon}, method="linear")
            arr = np.asarray(da.values, dtype=np.float32)
            # Preserve NaN (below-surface / missing). Only scrub +/-inf.
            arr[np.isinf(arr)] = np.nan
            arrays.append(arr)
    return np.stack(arrays, axis=0)


def _find_numeric_datavar(ds: Any, path: Any) -> str:
    """Return the first data variable with a numeric dtype, skipping metadata vars like 'crs'."""
    for v in ds.data_vars:
        if np.issubdtype(ds[v].dtype, np.number):
            return v
    raise ValueError(f"No numeric data variable found in {path}")


def _load_target_arrays(
    target_maps: Dict[str, Dict[Any, Path]],
    target_variables: Sequence[str],
    sample_date: Any,
) -> np.ndarray:
    """Load all PRISM target variables for a single date → (C, H, W)."""
    arrays: List[np.ndarray] = []
    for var in target_variables:
        path = target_maps[var][sample_date]
        with xr.open_dataset(str(path)) as ds:
            dvar = _find_numeric_datavar(ds, path)
            da = ds[dvar]
            if "time" in da.dims:
                da = da.isel(time=0, drop=True)
            arr = da.values.astype(np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            arrays.append(arr)
    return np.stack(arrays, axis=0)


def compute_scalars(
    cfg: Dict[str, Any],
    progress_interval: int = 50,
) -> Dict[str, np.ndarray]:
    """Compute channel-wise mean/std for predictors and targets over the training period."""
    if xr is None:
        raise ImportError("xarray is required")

    data_cfg = cfg.get("data", {})
    predictor_dir = resolve_path(data_cfg["predictor_dir"])
    target_dir = resolve_path(data_cfg["target_dir"])
    predictor_variables = expand_predictor_variables(data_cfg.get("predictor_variables", {}))
    target_variables: List[str] = list(data_cfg.get("target_variables", []))

    validate_target_variables(target_dir, target_variables)

    # Always use training dates for scalar computation
    start, end = parse_date_range_from_config(cfg, "training")
    print(f"[scalars] computing over training period: {start} → {end}")

    # Discover files
    merra_files = discover_merra2_files(predictor_dir, start, end)
    prism_files = discover_all_prism_targets(target_dir, target_variables, start, end)

    if not merra_files:
        raise RuntimeError(f"No MERRA2 files found in {predictor_dir} for {start}–{end}")

    # Align dates
    predictor_dates = [d for d, _ in merra_files]
    target_date_maps_dates = {
        var: [d for d, _ in fl] for var, fl in prism_files.items()
    }
    aligned_dates = align_dates(predictor_dates, target_date_maps_dates)
    print(f"[scalars] {len(aligned_dates)} aligned dates")

    if not aligned_dates:
        raise RuntimeError("No aligned dates — cannot compute scalars")

    pred_map = {d: p for d, p in merra_files}
    target_maps = {var: {d: p for d, p in fl} for var, fl in prism_files.items()}

    # PRISM (fine) grid, strided for fast-but-representative scalar estimation.
    # Predictors are regridded onto these coords (co-registered with targets) so
    # input scalars match what the dataset feeds the model.
    stride = int(data_cfg.get("scalar_stride", 8))
    first_tpath = target_maps[target_variables[0]][aligned_dates[0]]
    with xr.open_dataset(str(first_tpath)) as _tds:
        _latn = next((n for n in ("lat", "latitude", "y") if n in _tds.coords), "lat")
        _lonn = next((n for n in ("lon", "longitude", "x") if n in _tds.coords), "lon")
        fine_lat = _tds[_latn].values.astype(np.float64)
        fine_lon = _tds[_lonn].values.astype(np.float64)
    sub_lat = fine_lat[::stride]
    sub_lon = fine_lon[::stride]
    print(f"[scalars] estimating on strided PRISM grid {sub_lat.size}x{sub_lon.size} (stride={stride})")

    # Load static elevation regridded onto the strided PRISM grid.
    elev_file = data_cfg.get("static_elevation_file", None)
    elev_var = data_cfg.get("static_elevation_var", None)
    elevation_arr: Optional[np.ndarray] = None
    if elev_file:
        elev_path = resolve_path(elev_file)
        if elev_path.exists():
            elevation_arr = load_elevation(
                elev_path, var_name=elev_var or None,
                target_lat=sub_lat, target_lon=sub_lon,
            )
            print(f"[scalars] using elevation channel {elevation_arr.shape} from {elev_path.name}")
        else:
            print(f"[scalars] WARN: elevation file not found: {elev_path}; skipping elevation channel")

    # Accumulators — n_pred includes the elevation channel if present
    n_pred_dynamic = len(predictor_variables)
    n_pred = n_pred_dynamic + (1 if elevation_arr is not None else 0)
    n_tgt = len(target_variables)

    x_sum = np.zeros(n_pred, dtype=np.float64)
    x_sumsq = np.zeros(n_pred, dtype=np.float64)
    x_count = np.zeros(n_pred, dtype=np.float64)  # per-channel valid (non-NaN) count

    y_sum = np.zeros(n_tgt, dtype=np.float64)
    y_sumsq = np.zeros(n_tgt, dtype=np.float64)
    y_count = np.zeros(n_tgt, dtype=np.float64)  # per-channel valid (non-NaN) count

    # Per-gridpoint accumulators for targets (allocated lazily)
    y_grid_sum: Optional[np.ndarray] = None
    y_grid_sumsq: Optional[np.ndarray] = None
    y_grid_count = 0

    for idx, sample_date in enumerate(aligned_dates):
        # Predictors regridded onto the strided PRISM grid (co-registered).
        x = _load_predictor_arrays(pred_map[sample_date], predictor_variables, sub_lat, sub_lon)

        # Append elevation as the last channel (already on the strided grid).
        if elevation_arr is not None:
            x = np.concatenate([x, elevation_arr.astype(np.float32)[np.newaxis]], axis=0)

        # NaN-aware accumulation: below-surface (high-terrain) pressure-level
        # cells are NaN and MUST NOT bias the mean/std. Count valid cells per
        # channel separately (different levels have different NaN coverage).
        x_flat = x.reshape(n_pred, -1).astype(np.float64)
        x_valid = np.isfinite(x_flat)
        x_sum += np.nansum(np.where(x_valid, x_flat, 0.0), axis=1)
        x_sumsq += np.nansum(np.where(x_valid, x_flat ** 2, 0.0), axis=1)
        x_count += x_valid.sum(axis=1)

        # Targets subsampled on the same strided grid (PRISM is NaN over ocean).
        y = _load_target_arrays(target_maps, target_variables, sample_date)
        y = y[:, ::stride, ::stride]
        y_flat = y.reshape(n_tgt, -1).astype(np.float64)
        y_valid = np.isfinite(y_flat)
        y_sum += np.nansum(np.where(y_valid, y_flat, 0.0), axis=1)
        y_sumsq += np.nansum(np.where(y_valid, y_flat ** 2, 0.0), axis=1)
        y_count += y_valid.sum(axis=1)

        # Gridpoint accumulators (only used if a predictand requests gridpoint).
        y0 = np.nan_to_num(y.astype(np.float64), nan=0.0)
        if y_grid_sum is None:
            y_grid_sum = np.zeros_like(y0, dtype=np.float64)
            y_grid_sumsq = np.zeros_like(y0, dtype=np.float64)
        y_grid_sum += y0
        y_grid_sumsq += y0 ** 2
        y_grid_count += 1

        if progress_interval > 0 and (idx + 1) % progress_interval == 0:
            print(f"[scalars] processed {idx + 1}/{len(aligned_dates)} dates")

    # Finalize
    if np.any(x_count == 0):
        raise RuntimeError("Some predictor channel has no valid (non-NaN) pixels")
    if np.any(y_count == 0):
        raise RuntimeError("Some target channel has no valid (non-NaN) pixels")

    # Report per-channel valid coverage so missing-data handling is auditable.
    for ci in range(n_pred):
        frac = 100.0 * x_count[ci] / (len(aligned_dates) * (x.shape[-1] * x.shape[-2]))
        print(f"[scalars] input ch{ci}: valid {x_count[ci]:.0f} px ({frac:.1f}% of grid·dates)")

    inputs_mean = (x_sum / x_count).astype(np.float32)
    x_var = np.maximum(x_sumsq / x_count - (x_sum / x_count) ** 2, 0.0)
    inputs_std = np.sqrt(x_var).astype(np.float32)

    targets_mean = (y_sum / y_count).astype(np.float32)
    y_var = np.maximum(y_sumsq / y_count - (y_sum / y_count) ** 2, 0.0)
    targets_std = np.sqrt(y_var).astype(np.float32)

    # Gridpoint scalars
    targets_grid_mean = (y_grid_sum / y_grid_count).astype(np.float32)
    targets_grid_var = np.maximum(
        y_grid_sumsq / y_grid_count - (y_grid_sum / y_grid_count) ** 2, 0.0
    )
    targets_grid_std = np.sqrt(targets_grid_var).astype(np.float32)

    # Apply predictand-aware target scaling. This MUST run regardless of whether
    # any variable uses gridpoint normalization: e.g. ppt uses divide_only/global
    # (mean forced to 0, std = mean) and tmax/tmin use zscore/global. Previously
    # this whole block was gated behind `use_gridpoint`, so in the all-global
    # configuration ppt kept its raw mean (~1.2) instead of 0, breaking the
    # model's divide_only output scaler.
    predictands_cfg = cfg.get("predictands", {})
    use_gridpoint = any(
        predictands_cfg.get(var, {}).get("normalization", {}).get("mode") == "gridpoint"
        for var in target_variables
    )

    # Start from global per-channel stats; per-variable rules below override.
    final_targets_mean = targets_mean.copy()
    final_targets_std = targets_std.copy()

    for ch_idx, var in enumerate(target_variables):
        var_cfg = predictands_cfg.get(var, {})
        scaling_cfg = var_cfg.get("scaling", {})
        norm_cfg = var_cfg.get("normalization", {})
        # The active method comes from scaling.method (divide_only|zscore); fall
        # back to the normalization.method for older configs.
        method = scaling_cfg.get("method", norm_cfg.get("method", "zscore"))
        if method == "standardize":
            method = "zscore"
        mode = norm_cfg.get("mode", "global")
        eps = max(float(norm_cfg.get("eps_std", EPS)), EPS)

        if method == "divide_only":
            # Divide-only channels (ppt): centre at 0, scale by the mean so the
            # model's get_scalers divide_only guard (target_mu == 0) holds.
            final_targets_mean[ch_idx] = 0.0
            if mode == "gridpoint":
                final_targets_std[ch_idx] = np.maximum(targets_grid_mean[ch_idx], eps)
            else:
                final_targets_std[ch_idx] = max(float(targets_mean[ch_idx]), eps)
        else:  # zscore / standardize
            if mode == "gridpoint":
                final_targets_mean[ch_idx] = targets_grid_mean[ch_idx]
                final_targets_std[ch_idx] = np.maximum(targets_grid_std[ch_idx], eps)
            else:
                final_targets_mean[ch_idx] = targets_mean[ch_idx]
                final_targets_std[ch_idx] = max(float(targets_std[ch_idx]), eps)

    # If any variable is gridpoint, the saved arrays must be gridpoint-shaped so
    # the model can broadcast per-pixel scalers; otherwise they stay (C,).
    if use_gridpoint:
        gp_mean = targets_grid_mean.copy()
        gp_std = np.maximum(targets_grid_std.copy(), EPS)
        for ch_idx, var in enumerate(target_variables):
            # Broadcast the per-channel scalar resolved above across the grid for
            # any channel that is NOT itself gridpoint (keeps shapes consistent).
            mode = predictands_cfg.get(var, {}).get("normalization", {}).get("mode", "global")
            if mode != "gridpoint":
                gp_mean[ch_idx] = final_targets_mean[ch_idx]
                gp_std[ch_idx] = final_targets_std[ch_idx]
        final_targets_mean = gp_mean
        final_targets_std = gp_std

    # ------------------------------------------------------------------
    # Append validity-mask channel scalers. The dataset adds one binary mask
    # channel per predictor channel (1 valid / 0 missing). Masks must pass
    # through the model's internal z-score UNCHANGED, so their scalers are
    # mu=0, sigma=1. Order matches the dataset: [data_0..N-1, mask_0..N-1].
    mask_mean = np.zeros(n_pred, dtype=np.float32)
    mask_std = np.ones(n_pred, dtype=np.float32)
    inputs_mean = np.concatenate([inputs_mean, mask_mean], axis=0)
    inputs_std = np.concatenate([inputs_std, mask_std], axis=0)
    print(
        f"[scalars] input scalers: {n_pred} data + {n_pred} mask channels "
        f"= {inputs_mean.shape[0]} total"
    )

    return {
        "inputs_mean": inputs_mean,
        "inputs_std": inputs_std,
        "targets_mean": final_targets_mean,
        "targets_std": final_targets_std,
        "targets_mean_raw": targets_mean,
        "targets_std_raw": targets_std,
        "targets_grid_mean": targets_grid_mean,
        "targets_grid_std": targets_grid_std,
        "input_pixel_count": x_count.tolist(),
        "target_pixel_count": y_count.tolist(),
        "target_grid_sample_count": y_grid_count,
    }


def _compact_summary(arr: np.ndarray) -> dict:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"shape": list(arr.shape), "finite_count": 0}
    return {
        "shape": list(arr.shape),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
    }


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    data_cfg = cfg.get("data", {})

    output_dir = args.output_dir or str(data_cfg.get("scalar_dir", "./examples/MERRA_PRISM/scalars"))
    output_dir = str(resolve_path(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    stats = compute_scalars(cfg, progress_interval=args.progress_interval)

    # Save .npy files
    for key in ("inputs_mean", "inputs_std", "targets_mean", "targets_std"):
        path = os.path.join(output_dir, f"{key}.npy")
        np.save(path, stats[key])
        print(f"[scalars] saved {path}  shape={stats[key].shape}")

    # Save metadata
    metadata = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "config": args.config,
        "num_aligned_dates": stats["target_grid_sample_count"],
        "input_channels": int(stats["inputs_mean"].shape[0]),
        "target_channels": int(stats["targets_mean"].shape[0]) if stats["targets_mean"].ndim >= 1 else 0,
        "input_pixel_count": stats["input_pixel_count"],
        "target_pixel_count": stats["target_pixel_count"],
        "summary": {
            "inputs_mean": _compact_summary(stats["inputs_mean"]),
            "inputs_std": _compact_summary(stats["inputs_std"]),
            "targets_mean": _compact_summary(stats["targets_mean"]),
            "targets_std": _compact_summary(stats["targets_std"]),
        },
    }
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"[scalars] metadata → {meta_path}")
    print("[scalars] done.")


if __name__ == "__main__":
    main()
