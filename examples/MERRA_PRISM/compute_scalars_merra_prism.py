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
    path: Path, variables: Sequence[str]
) -> np.ndarray:
    """Load all predictor variables from a single MERRA2 file → (C, H, W)."""
    arrays: List[np.ndarray] = []
    with xr.open_dataset(str(path)) as ds:
        for var in variables:
            if var not in ds.data_vars:
                raise ValueError(f"Variable '{var}' not found in {path}")
            da = ds[var]
            if "time" in da.dims:
                da = da.isel(time=0, drop=True)
            arr = da.values.astype(np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            arrays.append(arr)
    return np.stack(arrays, axis=0)


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
            dvar = list(ds.data_vars)[0]
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
    predictor_variables: List[str] = list(data_cfg.get("predictor_variables", []))
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

    # Accumulators
    n_pred = len(predictor_variables)
    n_tgt = len(target_variables)

    x_sum = np.zeros(n_pred, dtype=np.float64)
    x_sumsq = np.zeros(n_pred, dtype=np.float64)
    x_count = 0

    y_sum = np.zeros(n_tgt, dtype=np.float64)
    y_sumsq = np.zeros(n_tgt, dtype=np.float64)
    y_count = 0

    # Per-gridpoint accumulators for targets (allocated lazily)
    y_grid_sum: Optional[np.ndarray] = None
    y_grid_sumsq: Optional[np.ndarray] = None
    y_grid_count = 0

    for idx, sample_date in enumerate(aligned_dates):
        # Predictors
        x = _load_predictor_arrays(pred_map[sample_date], predictor_variables)
        x_flat = x.reshape(n_pred, -1).astype(np.float64)
        x_sum += x_flat.sum(axis=1)
        x_sumsq += (x_flat ** 2).sum(axis=1)
        x_count += x_flat.shape[1]

        # Targets
        y = _load_target_arrays(target_maps, target_variables, sample_date)
        y_flat = y.reshape(n_tgt, -1).astype(np.float64)
        y_sum += y_flat.sum(axis=1)
        y_sumsq += (y_flat ** 2).sum(axis=1)
        y_count += y_flat.shape[1]

        # Gridpoint accumulators
        if y_grid_sum is None:
            y_grid_sum = np.zeros_like(y, dtype=np.float64)
            y_grid_sumsq = np.zeros_like(y, dtype=np.float64)
        y_grid_sum += y.astype(np.float64)
        y_grid_sumsq += (y.astype(np.float64)) ** 2
        y_grid_count += 1

        if progress_interval > 0 and (idx + 1) % progress_interval == 0:
            print(f"[scalars] processed {idx + 1}/{len(aligned_dates)} dates")

    # Finalize
    if x_count == 0:
        raise RuntimeError("No predictor pixels observed")
    if y_count == 0:
        raise RuntimeError("No target pixels observed")

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

    # Apply predictand-aware target scaling if config has predictands block
    final_targets_mean = targets_mean
    final_targets_std = targets_std

    predictands_cfg = cfg.get("predictands", {})
    use_gridpoint = any(
        predictands_cfg.get(var, {}).get("normalization", {}).get("mode") == "gridpoint"
        for var in target_variables
    )

    if use_gridpoint:
        final_targets_mean = targets_grid_mean.copy()
        final_targets_std = targets_grid_std.copy()

        for ch_idx, var in enumerate(target_variables):
            var_cfg = predictands_cfg.get(var, {})
            scaling_cfg = var_cfg.get("scaling", {})
            norm_cfg = var_cfg.get("normalization", {})
            method = scaling_cfg.get("method", "zscore")
            mode = norm_cfg.get("mode", "global")
            eps = max(float(norm_cfg.get("eps_std", EPS)), EPS)

            if method == "divide_only":
                final_targets_mean[ch_idx] = 0.0
                if mode == "gridpoint":
                    final_targets_std[ch_idx] = np.maximum(
                        targets_grid_mean[ch_idx], eps
                    )
                else:
                    final_targets_std[ch_idx] = max(float(targets_mean[ch_idx]), eps)
            elif method == "zscore":
                if mode == "gridpoint":
                    final_targets_mean[ch_idx] = targets_grid_mean[ch_idx]
                    final_targets_std[ch_idx] = np.maximum(targets_grid_std[ch_idx], eps)
                else:
                    final_targets_mean[ch_idx] = targets_mean[ch_idx]
                    final_targets_std[ch_idx] = max(float(targets_std[ch_idx]), eps)

    return {
        "inputs_mean": inputs_mean,
        "inputs_std": inputs_std,
        "targets_mean": final_targets_mean,
        "targets_std": final_targets_std,
        "targets_mean_raw": targets_mean,
        "targets_std_raw": targets_std,
        "targets_grid_mean": targets_grid_mean,
        "targets_grid_std": targets_grid_std,
        "input_pixel_count": x_count,
        "target_pixel_count": y_count,
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
