"""Compute normalisation scalars for the NARR-to-PRISM downscaling workflow.

Statistics are computed **only** over the YAML-defined training period so that
the same scalars can be reused consistently during training and inference.

Usage:
    python compute_scalars_narr_prism.py --config NARR_PRISM_subdomain.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import xarray as xr
except ImportError:
    xr = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from narr_prism_utils import (
    align_dates,
    discover_all_prism_targets,
    discover_narr_files,
    expand_predictor_variables,
    interpolate_narr_to_grid,
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

from granitewxc.utils import normalization as norm
from granitewxc.utils import prism_grid as prism_grid_contract
from granitewxc.utils import prism_preprocessed
from granitewxc.utils.streaming_quantile import (
    StreamingHistogramQuantile,
    percentile_probability,
    resolve_divide_only_scale,
)

EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-channel scalars for NARR-PRISM training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to NARR_PRISM_subdomain.yaml")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override scalar output directory (default: <preprocessed_dir>/<case_name>/scalars)",
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
    sample_date: Any,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> np.ndarray:
    """Load predictors from a NARR file, regridded onto (target_lat, target_lon).

    Predictors are bilinearly interpolated onto the (strided) PRISM grid so the
    input scalars are computed from the SAME co-registered fields the dataset
    feeds the model at train/inference time.
    """
    arrays: List[np.ndarray] = []
    for var, level in variables:
        arrays.append(
            interpolate_narr_to_grid(path, var, level, sample_date, target_lat, target_lon)
        )
    return np.stack(arrays, axis=0)


def _find_numeric_datavar(ds: Any, path: Any) -> str:
    """Return the first data variable with a numeric dtype, skipping metadata vars like 'crs'."""
    for v in ds.data_vars:
        if np.issubdtype(ds[v].dtype, np.number):
            return v
    raise ValueError(f"No numeric data variable found in {path}")


def _coord_subset_slice(
    coords: np.ndarray,
    lower: Optional[float],
    upper: Optional[float],
    axis_name: str,
) -> slice:
    """Return a contiguous index slice for inclusive coordinate bounds."""
    if lower is None and upper is None:
        return slice(None)

    lo = float(np.nanmin(coords) if lower is None else lower)
    hi = float(np.nanmax(coords) if upper is None else upper)
    if hi < lo:
        lo, hi = hi, lo

    coord_min = float(np.nanmin(coords))
    coord_max = float(np.nanmax(coords))
    if hi < coord_min or lo > coord_max:
        raise ValueError(
            f"spatial_subset {axis_name} bounds [{lo}, {hi}] do not overlap "
            f"grid range [{coord_min}, {coord_max}]"
        )

    diffs = np.diff(np.asarray(coords, dtype=np.float64))
    finite_diffs = np.abs(diffs[np.isfinite(diffs) & (diffs != 0)])
    tol = 0.0 if finite_diffs.size == 0 else float(np.nanmedian(finite_diffs)) * 0.51

    mask = (coords >= lo - tol) & (coords <= hi + tol)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError(
            f"spatial_subset {axis_name} bounds [{lo}, {hi}] selected no grid cells"
        )
    return slice(int(idx[0]), int(idx[-1]) + 1)


def _spatial_subset_slices(
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    data_cfg: Dict[str, Any],
) -> Tuple[slice, slice]:
    spatial_subset = data_cfg.get("spatial_subset", {}) or {}
    if not bool(spatial_subset.get("enabled", False)):
        return slice(None), slice(None)

    lat_slice = _coord_subset_slice(
        lat_vals,
        spatial_subset.get("lat_min"),
        spatial_subset.get("lat_max"),
        "latitude",
    )
    lon_slice = _coord_subset_slice(
        lon_vals,
        spatial_subset.get("lon_min"),
        spatial_subset.get("lon_max"),
        "longitude",
    )
    return lat_slice, lon_slice


def _load_target_arrays(
    target_maps: Dict[str, Dict[Any, Path]],
    target_variables: Sequence[str],
    sample_date: Any,
    canonical_grid: prism_grid_contract.CanonicalPrismGrid,
    lat_slice: slice = slice(None),
    lon_slice: slice = slice(None),
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
            lat_name = next(
                (n for n in ("lat", "latitude", "y") if n in ds.coords),
                None,
            )
            lon_name = next(
                (n for n in ("lon", "longitude", "x") if n in ds.coords),
                None,
            )
            if lat_name is None or lon_name is None:
                raise ValueError(f"Cannot find PRISM coordinates in {path}")
            observed_lat = np.asarray(ds[lat_name].values, dtype=np.float64)[lat_slice]
            observed_lon = np.asarray(ds[lon_name].values, dtype=np.float64)[lon_slice]
            prism_grid_contract.assert_grid_matches(
                canonical_grid,
                observed_lat,
                observed_lon,
                context=f"PRISM target {path}",
            )
            if lat_name not in da.dims or lon_name not in da.dims:
                raise ValueError(
                    f"PRISM target {path} variable {dvar!r} must use coordinate "
                    f"dimensions ({lat_name!r}, {lon_name!r}); got {da.dims}"
                )
            da = da.transpose(lat_name, lon_name)
            arr = da.values[lat_slice, lon_slice].astype(np.float32)
            # Preserve PRISM missing cells as NaN so ocean/outside-CONUS pixels
            # do not bias target mean/std toward zero. Only scrub +/-inf.
            arr[np.isinf(arr)] = np.nan
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
    normalization_cfg = cfg.get("normalization", {}) or {}
    predictor_mode = str(normalization_cfg.get("predictor_mode", "global")).lower()
    if predictor_mode not in {"global", "gridpoint"}:
        raise ValueError(
            "normalization.predictor_mode must be either 'global' or 'gridpoint'"
        )
    use_input_gridpoint = predictor_mode == "gridpoint"
    use_preprocessed = bool(data_cfg.get("use_preprocessed", False))
    predictor_variables = expand_predictor_variables(data_cfg.get("predictor_variables", {}))
    target_variables: List[str] = list(data_cfg.get("target_variables", []))
    preprocessed_predictor_names = [
        prism_preprocessed.predictor_name(var, level)
        for var, level in predictor_variables
    ]
    preprocessed_target_names = [
        prism_preprocessed.target_name(var) for var in target_variables
    ]
    include_elevation = bool(data_cfg.get("static_elevation_file"))
    preprocessed_required_variables = [
        *preprocessed_predictor_names,
        *preprocessed_target_names,
    ]
    if include_elevation:
        preprocessed_required_variables.append("static_elevation")

    # Always use training dates for scalar computation
    start, end = parse_date_range_from_config(cfg, "training")
    print(f"[scalars] computing over training period: {start} → {end}")

    preprocessed_map: Dict[Any, Path] = {}
    pred_map: Dict[Any, Any] = {}
    target_maps: Dict[str, Dict[Any, Path]] = {}
    if use_preprocessed:
        aligned_dates = prism_preprocessed.inclusive_daily_dates(start, end)
        preprocessed_dir = norm.case_preprocess_dir(cfg) / "training"
        preprocessed_map = prism_preprocessed.require_daily_products(
            preprocessed_dir, "narr_prism", aligned_dates
        )
        canonical_grid = prism_grid_contract.load_canonical_grid(
            norm.case_preprocess_dir(cfg), required=True
        )
        assert canonical_grid is not None
        fine_lat, fine_lon = canonical_grid.lat, canonical_grid.lon
        lat_slice, lon_slice = slice(None), slice(None)
        print(
            f"[scalars] strict preprocessed source: {preprocessed_dir} "
            f"({len(aligned_dates)} required daily products; raw fallback disabled)"
        )
    else:
        predictor_dir = resolve_path(data_cfg["predictor_dir"])
        target_dir = resolve_path(data_cfg["target_dir"])
        validate_target_variables(target_dir, target_variables)
        narr_files = discover_narr_files(
            predictor_dir, start, end, predictor_variables
        )
        prism_files = discover_all_prism_targets(
            target_dir, target_variables, start, end
        )
        if not narr_files:
            raise RuntimeError(
                f"No NARR files found in {predictor_dir} for {start}–{end}"
            )
        predictor_dates = [d for d, _ in narr_files]
        target_date_maps_dates = {
            var: [d for d, _ in fl] for var, fl in prism_files.items()
        }
        aligned_dates = align_dates(predictor_dates, target_date_maps_dates)
        if not aligned_dates:
            raise RuntimeError("No aligned dates — cannot compute scalars")
        pred_map = {d: p for d, p in narr_files}
        target_maps = {
            var: {d: p for d, p in fl} for var, fl in prism_files.items()
        }
    print(f"[scalars] {len(aligned_dates)} aligned dates")

    # PRISM (fine) grid, strided for fast-but-representative scalar estimation.
    # Predictors are regridded onto these coords (co-registered with targets) so
    # input scalars match what the dataset feeds the model.
    stride = int(data_cfg.get("scalar_stride", 8))
    predictands_cfg = cfg.get("predictands", {})
    histogram_maximum = float(
        normalization_cfg.get("quantile_histogram_maximum", 512.0)
    )
    histogram_bins = int(normalization_cfg.get("quantile_histogram_bins", 65536))
    quantile_accumulators: Dict[int, StreamingHistogramQuantile] = {}
    for ch_idx, var in enumerate(target_variables):
        var_cfg = predictands_cfg.get(var, {}) or {}
        scaling_cfg = var_cfg.get("scaling", {}) or {}
        norm_cfg = var_cfg.get("normalization", {}) or {}
        method = str(
            scaling_cfg.get("method", norm_cfg.get("method", "zscore"))
        ).lower()
        default_scale_stat = "p95" if method == "divide_only" else "mean"
        scale_stat = str(
            norm_cfg.get(
                "scale_stat", scaling_cfg.get("scale_stat", default_scale_stat)
            )
        ).lower()
        if method == "divide_only" and scale_stat.startswith("p"):
            percentile_probability(scale_stat)  # validate before expensive I/O
            quantile_accumulators[ch_idx] = StreamingHistogramQuantile(
                maximum=histogram_maximum,
                bins=histogram_bins,
            )
    use_target_gridpoint = any(
        ((predictands_cfg.get(var, {}) or {}).get("normalization", {}) or {}).get("mode")
        == "gridpoint"
        for var in target_variables
    )
    if quantile_accumulators and stride != 1:
        print(
            "[scalars] percentile scaling requires every training-period pixel; "
            f"overriding scalar_stride={stride} to 1"
        )
        stride = 1
    if use_input_gridpoint and stride != 1:
        print(
            "[scalars] normalization.predictor_mode=gridpoint requires full-resolution "
            f"input scalers; overriding scalar_stride={stride} to 1"
        )
        stride = 1
    if use_target_gridpoint and stride != 1:
        # Per-gridpoint TARGET scalers are a native-resolution climatology; a
        # strided (coarse) estimate would reintroduce block artifacts.
        print(
            "[scalars] a predictand uses normalization.mode: gridpoint -> forcing "
            f"scalar_stride={stride} to 1 for a full-resolution target climatology"
        )
        stride = 1
    if not use_preprocessed:
        first_tpath = target_maps[target_variables[0]][aligned_dates[0]]
        with xr.open_dataset(str(first_tpath)) as _tds:
            _latn = next(
                (n for n in ("lat", "latitude", "y") if n in _tds.coords),
                "lat",
            )
            _lonn = next(
                (n for n in ("lon", "longitude", "x") if n in _tds.coords),
                "lon",
            )
            fine_lat = _tds[_latn].values.astype(np.float64)
            fine_lon = _tds[_lonn].values.astype(np.float64)
        lat_slice, lon_slice = _spatial_subset_slices(
            fine_lat, fine_lon, data_cfg
        )
        fine_lat = fine_lat[lat_slice]
        fine_lon = fine_lon[lon_slice]
        canonical_grid = prism_grid_contract.ensure_canonical_grid(
            norm.case_preprocess_dir(cfg),
            fine_lat,
            fine_lon,
            source=str(first_tpath),
            context="NARR scalar target grid",
        )
        fine_lat, fine_lon = canonical_grid.lat, canonical_grid.lon
    sub_lat = fine_lat[::stride]
    sub_lon = fine_lon[::stride]
    print(f"[scalars] estimating on strided PRISM grid {sub_lat.size}x{sub_lon.size} (stride={stride})")
    print(
        f"[scalars] canonical PRISM grid fingerprint="
        f"{canonical_grid.fingerprint[:12]}"
    )
    expected_preprocessing_fields: Dict[str, Any] = {}
    split_preprocessing_signature: Optional[str] = None
    predictor_preprocessing_signature: Optional[str] = None
    training_source_artifact_signatures: Dict[str, str] = {}
    if use_preprocessed:
        expected_preprocessing_fields = (
            prism_preprocessed.preprocessing_config_fields(
                data_type="narr_prism",
                predictor_variables=predictor_variables,
                target_variables=target_variables,
                include_targets=True,
                regrid_method=str(data_cfg.get("regrid_method", "bilinear")),
                canonical_grid_fingerprint=canonical_grid.fingerprint,
                static_elevation_required=include_elevation,
                static_elevation_variable=data_cfg.get("static_elevation_var"),
            )
        )

    # Load static elevation regridded onto the strided PRISM grid.
    elev_file = data_cfg.get("static_elevation_file", None)
    elev_var = data_cfg.get("static_elevation_var", None)
    elevation_arr: Optional[np.ndarray] = None
    if use_preprocessed:
        first_date = aligned_dates[0]
        first_product = preprocessed_map[first_date]
        with xr.open_dataset(str(first_product)) as ds:
            lat_name, lon_name = prism_preprocessed.validate_daily_product(
                ds,
                first_product,
                canonical_grid,
                mode="training",
                sample_date=first_date,
                required_variables=preprocessed_required_variables,
                expected_preprocessing_fields=expected_preprocessing_fields,
            )
            split_preprocessing_signature = str(
                ds.attrs[prism_preprocessed.PREPROCESSING_SIGNATURE_ATTR]
            )
            predictor_preprocessing_signature = str(
                ds.attrs[
                    prism_preprocessed.PREDICTOR_PREPROCESSING_SIGNATURE_ATTR
                ]
            )
            if include_elevation:
                elevation_arr = prism_preprocessed.read_spatial_variable(
                    ds,
                    "static_elevation",
                    lat_name,
                    lon_name,
                    lat_slice=slice(None, None, stride),
                    lon_slice=slice(None, None, stride),
                )
    elif elev_file:
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

    # Optional per-gridpoint input accumulators. For time-varying predictors this
    # gives one normalization scalar per channel and PRISM grid cell, computed
    # over training time only. Static elevation is handled later with broadcast
    # global scalers because its temporal std is zero at every grid point.
    x_grid_sum: Optional[np.ndarray] = None
    x_grid_sumsq: Optional[np.ndarray] = None
    x_grid_count: Optional[np.ndarray] = None

    y_sum = np.zeros(n_tgt, dtype=np.float64)
    y_sumsq = np.zeros(n_tgt, dtype=np.float64)
    y_count = np.zeros(n_tgt, dtype=np.float64)  # per-channel valid (non-NaN) count

    # Per-gridpoint accumulators for targets (allocated lazily)
    y_grid_sum: Optional[np.ndarray] = None
    y_grid_sumsq: Optional[np.ndarray] = None
    y_grid_count: Optional[np.ndarray] = None

    for idx, sample_date in enumerate(aligned_dates):
        if use_preprocessed:
            product = preprocessed_map[sample_date]
            with xr.open_dataset(str(product)) as ds:
                lat_name, lon_name = prism_preprocessed.validate_daily_product(
                    ds,
                    product,
                    canonical_grid,
                    mode="training",
                    sample_date=sample_date,
                    required_variables=preprocessed_required_variables,
                    expected_preprocessing_fields=expected_preprocessing_fields,
                    expected_preprocessing_signature=split_preprocessing_signature,
                )
                training_source_artifact_signatures[str(sample_date)] = str(
                    ds.attrs[prism_preprocessed.SOURCE_ARTIFACT_SIGNATURE_ATTR]
                )
                grid_slice = slice(None, None, stride)
                x_arrays = [
                    prism_preprocessed.read_spatial_variable(
                        ds,
                        name,
                        lat_name,
                        lon_name,
                        lat_slice=grid_slice,
                        lon_slice=grid_slice,
                    )
                    for name in preprocessed_predictor_names
                ]
                if include_elevation:
                    x_arrays.append(
                        prism_preprocessed.read_spatial_variable(
                            ds,
                            "static_elevation",
                            lat_name,
                            lon_name,
                            lat_slice=grid_slice,
                            lon_slice=grid_slice,
                        )
                    )
                x = np.stack(x_arrays, axis=0)
                y = np.stack(
                    [
                        prism_preprocessed.read_spatial_variable(
                            ds,
                            name,
                            lat_name,
                            lon_name,
                            lat_slice=grid_slice,
                            lon_slice=grid_slice,
                        )
                        for name in preprocessed_target_names
                    ],
                    axis=0,
                )
        else:
            x = _load_predictor_arrays(
                pred_map[sample_date],
                predictor_variables,
                sample_date,
                sub_lat,
                sub_lon,
            )
            y = _load_target_arrays(
                target_maps,
                target_variables,
                sample_date,
                canonical_grid,
                lat_slice=lat_slice,
                lon_slice=lon_slice,
            )
            y = y[:, ::stride, ::stride]

        # Append raw-source elevation; preprocessed products already supplied it.
        if elevation_arr is not None and not use_preprocessed:
            x = np.concatenate([x, elevation_arr.astype(np.float32)[np.newaxis]], axis=0)

        # NaN-aware accumulation: below-surface (high-terrain) pressure-level
        # cells are NaN and MUST NOT bias the mean/std. Count valid cells per
        # channel separately (different levels have different NaN coverage).
        x_flat = x.reshape(n_pred, -1).astype(np.float64)
        x_valid = np.isfinite(x_flat)
        x_sum += np.nansum(np.where(x_valid, x_flat, 0.0), axis=1)
        x_sumsq += np.nansum(np.where(x_valid, x_flat ** 2, 0.0), axis=1)
        x_count += x_valid.sum(axis=1)

        if use_input_gridpoint:
            x0 = np.where(np.isfinite(x), x.astype(np.float64), 0.0)
            x_valid_grid = np.isfinite(x)
            if x_grid_sum is None:
                x_grid_sum = np.zeros_like(x0, dtype=np.float64)
                x_grid_sumsq = np.zeros_like(x0, dtype=np.float64)
                x_grid_count = np.zeros_like(x0, dtype=np.float64)
            x_grid_sum += x0
            x_grid_sumsq += x0 ** 2
            x_grid_count += x_valid_grid.astype(np.float64)

        # Targets use the same strided native PRISM grid as predictors.
        for ch_idx, accumulator in quantile_accumulators.items():
            accumulator.update(y[ch_idx])
        y_flat = y.reshape(n_tgt, -1).astype(np.float64)
        y_valid = np.isfinite(y_flat)
        y_sum += np.nansum(np.where(y_valid, y_flat, 0.0), axis=1)
        y_sumsq += np.nansum(np.where(y_valid, y_flat ** 2, 0.0), axis=1)
        y_count += y_valid.sum(axis=1)

        # Gridpoint accumulators (only used if a predictand requests gridpoint).
        y0 = np.where(np.isfinite(y), y.astype(np.float64), 0.0)
        y_valid_grid = np.isfinite(y)
        if y_grid_sum is None:
            y_grid_sum = np.zeros_like(y0, dtype=np.float64)
            y_grid_sumsq = np.zeros_like(y0, dtype=np.float64)
            y_grid_count = np.zeros_like(y0, dtype=np.float64)
        y_grid_sum += y0
        y_grid_sumsq += y0 ** 2
        y_grid_count += y_valid_grid.astype(np.float64)

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

    if use_input_gridpoint:
        assert x_grid_sum is not None
        assert x_grid_sumsq is not None
        assert x_grid_count is not None
        if np.any(x_grid_count == 0):
            print(
                "[scalars] WARN: some input grid cells have no finite predictor values; "
                "their gridpoint scalers will be set to neutral values."
            )
        safe_x_grid_count = np.maximum(x_grid_count, 1.0)
        inputs_grid_mean = (x_grid_sum / safe_x_grid_count).astype(np.float32)
        inputs_grid_var = np.maximum(
            x_grid_sumsq / safe_x_grid_count
            - (x_grid_sum / safe_x_grid_count) ** 2,
            0.0,
        )
        inputs_grid_std = np.sqrt(inputs_grid_var).astype(np.float32)
        missing_x_grid = x_grid_count == 0
        inputs_grid_mean[missing_x_grid] = 0.0
        inputs_grid_std[missing_x_grid] = 1.0

        # Static elevation is included as the final data channel when present.
        # Its temporal std at a fixed grid point is exactly zero, so keep the
        # previous global channel normalization and broadcast it over the grid.
        if elevation_arr is not None:
            elev_idx = n_pred - 1
            inputs_grid_mean[elev_idx] = inputs_mean[elev_idx]
            inputs_grid_std[elev_idx] = max(float(inputs_std[elev_idx]), EPS)

        inputs_mean = inputs_grid_mean
        inputs_std = np.maximum(inputs_grid_std, EPS)

    targets_mean = (y_sum / y_count).astype(np.float32)
    y_var = np.maximum(y_sumsq / y_count - (y_sum / y_count) ** 2, 0.0)
    targets_std = np.sqrt(y_var).astype(np.float32)

    # Gridpoint scalars
    assert y_grid_sum is not None
    assert y_grid_sumsq is not None
    assert y_grid_count is not None
    if np.any(y_grid_count == 0):
        print(
            "[scalars] WARN: some target grid cells have no finite PRISM values; "
            "their gridpoint scalers will be set to neutral values."
        )
    safe_grid_count = np.maximum(y_grid_count, 1.0)
    targets_grid_mean = (y_grid_sum / safe_grid_count).astype(np.float32)
    targets_grid_var = np.maximum(
        y_grid_sumsq / safe_grid_count - (y_grid_sum / safe_grid_count) ** 2, 0.0
    )
    targets_grid_std = np.sqrt(targets_grid_var).astype(np.float32)
    missing_grid = y_grid_count == 0
    targets_grid_mean[missing_grid] = 0.0
    targets_grid_std[missing_grid] = 1.0

    # Apply predictand-aware target scaling. This MUST run regardless of whether
    # any variable uses gridpoint normalization: e.g. ppt uses divide_only/global
    # (mean forced to 0, scale = configured p95) and tmax/tmin use zscore. Previously
    # this whole block was gated behind `use_gridpoint`, so in the all-global
    # configuration ppt kept its raw mean (~1.2) instead of 0, breaking the
    # model's divide_only output scaler.
    # Phase A resolves a per-CHANNEL SCALAR for every target (used directly for
    # global channels and as the broadcast fill for gridpoint channels). Spatial
    # (per-gridpoint) maps are only assembled in Phase B, so we never store a 2-D
    # field inside the 1-D (C,) array (which would raise).
    final_targets_mean = targets_mean.copy()
    final_targets_std = targets_std.copy()
    gridpoint_channels: List[int] = []
    scaling_statistics: Dict[str, Any] = {}

    for ch_idx, var in enumerate(target_variables):
        var_cfg = predictands_cfg.get(var, {})
        scaling_cfg = var_cfg.get("scaling", {})
        norm_cfg = var_cfg.get("normalization", {}) or {}
        # The active method comes from scaling.method (divide_only|zscore); fall
        # back to the normalization.method for older configs.
        method = str(
            scaling_cfg.get("method", norm_cfg.get("method", "zscore"))
        ).lower()
        if method == "standardize":
            method = "zscore"
        mode = norm_cfg.get("mode", "global")
        default_scale_stat = "p95" if method == "divide_only" else "mean"
        scale_stat = str(
            norm_cfg.get(
                "scale_stat", scaling_cfg.get("scale_stat", default_scale_stat)
            )
        ).lower()
        eps = max(float(norm_cfg.get("eps_std", EPS)), EPS)

        if mode == "gridpoint" and method == "divide_only":
            raise ValueError(
                f"Target '{var}' requests normalization.mode: gridpoint with "
                f"scaling.method: divide_only, which is not supported. Keep "
                f"divide_only variables (e.g. ppt) at normalization.mode: global, "
                f"or switch to zscore for per-gridpoint targets."
            )

        if method == "divide_only":
            fixed_scale = scaling_cfg.get(
                "fixed_scale", norm_cfg.get("fixed_scale")
            )
            if fixed_scale is not None:
                fixed_scale = float(fixed_scale)
            resolved_scale, scale_provenance = resolve_divide_only_scale(
                scale_stat,
                training_mean=float(targets_mean[ch_idx]),
                finite_count=int(y_count[ch_idx]),
                epsilon=eps,
                fixed_scale=fixed_scale,
                quantile_accumulator=quantile_accumulators.get(ch_idx),
            )
            # Divide-only channels are centred at zero and scaled by their
            # configured training statistic (p95 for hurdle precipitation).
            final_targets_mean[ch_idx] = 0.0
            final_targets_std[ch_idx] = resolved_scale
            scaling_statistics[var] = scale_provenance
        else:  # zscore / standardize -> per-channel global scalar
            final_targets_mean[ch_idx] = targets_mean[ch_idx]
            final_targets_std[ch_idx] = max(float(targets_std[ch_idx]), eps)
            scaling_statistics[var] = {
                "scale_stat": "mean_std",
                "mean": float(final_targets_mean[ch_idx]),
                "std": float(final_targets_std[ch_idx]),
                "finite_count": int(y_count[ch_idx]),
            }

        if mode == "gridpoint":
            gridpoint_channels.append(ch_idx)

    # If any target is per-gridpoint, promote the scalers to (C, H, W): broadcast
    # every channel's resolved scalar across the grid, then overwrite the
    # gridpoint channels with their native-resolution PRISM climatology. The
    # climatology grid matches the (unstrided) PRISM domain because gridpoint
    # targets forced scalar_stride == 1 above.
    use_gridpoint = len(gridpoint_channels) > 0
    if use_gridpoint:
        H, W = int(targets_grid_mean.shape[1]), int(targets_grid_mean.shape[2])
        c = final_targets_mean.shape[0]
        gp_mean = np.broadcast_to(final_targets_mean.reshape(c, 1, 1), (c, H, W)).copy()
        gp_std = np.broadcast_to(final_targets_std.reshape(c, 1, 1), (c, H, W)).copy()
        for ch_idx in gridpoint_channels:
            eps = max(
                float(
                    ((predictands_cfg.get(target_variables[ch_idx], {}) or {})
                     .get("normalization", {}) or {}).get("eps_std", EPS)
                ),
                EPS,
            )
            gp_mean[ch_idx] = targets_grid_mean[ch_idx]
            gp_std[ch_idx] = np.maximum(targets_grid_std[ch_idx], eps)
        final_targets_mean = gp_mean.astype(np.float32)
        final_targets_std = gp_std.astype(np.float32)
        print(
            f"[scalars] per-gridpoint target scalers active for channels "
            f"{gridpoint_channels} -> targets_mean/std shape {final_targets_mean.shape}"
        )

    # ------------------------------------------------------------------
    # Append validity-mask channel scalers. The dataset adds one binary mask
    # channel per predictor channel (1 valid / 0 missing). Masks must pass
    # through the model's internal z-score UNCHANGED, so their scalers are
    # mu=0, sigma=1. Order matches the dataset: [data_0..N-1, mask_0..N-1].
    if inputs_mean.ndim == 3:
        mask_shape = inputs_mean.shape
        mask_mean = np.zeros(mask_shape, dtype=np.float32)
        mask_std = np.ones(mask_shape, dtype=np.float32)
    else:
        mask_mean = np.zeros(n_pred, dtype=np.float32)
        mask_std = np.ones(n_pred, dtype=np.float32)
    inputs_mean = np.concatenate([inputs_mean, mask_mean], axis=0)
    inputs_std = np.concatenate([inputs_std, mask_std], axis=0)
    print(
        f"[scalars] input scalers: {n_pred} data + {n_pred} mask channels "
        f"= {inputs_mean.shape[0]} total"
    )

    training_source_artifact_split_signature = (
        prism_preprocessed.split_source_artifact_signature(
            training_source_artifact_signatures
        )
        if use_preprocessed
        else None
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
        "target_grid_sample_count": len(aligned_dates),
        "scaling_statistics": scaling_statistics,
        "predictor_preprocessing_signature": predictor_preprocessing_signature,
        norm.TRAINING_SOURCE_ARTIFACT_SIGNATURES_KEY: (
            training_source_artifact_signatures
        ),
        norm.TRAINING_SOURCE_ARTIFACT_SPLIT_SIGNATURE_KEY: (
            training_source_artifact_split_signature
        ),
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

    if args.output_dir:
        output_dir = str(resolve_path(args.output_dir))
    else:
        # Save per-channel scalers under the owning case_name so training and
        # inference load exactly these files (single source of truth):
        # <preprocessed_dir>/<case_name>/scalars.
        output_dir = str(norm.resolve_scalar_dir(cfg, for_writing=True))
    os.makedirs(output_dir, exist_ok=True)

    case_name = norm.get_case_name(cfg)
    print(f"[scalars] Using case_name: {case_name}")
    print(f"[scalars] Using preprocessing directory: {norm.case_preprocess_dir(cfg)}")
    print(f"[scalars] Writing scalars to: {output_dir}")

    stats = compute_scalars(cfg, progress_interval=args.progress_interval)

    for key in ("inputs_mean", "inputs_std"):
        norm.assert_valid_predictor_scaler(key, stats[key], cfg)
    for key in ("targets_mean", "targets_std"):
        norm.assert_valid_target_scaler_for_config(key, stats[key], cfg)
    for key in ("inputs_mean", "inputs_std", "targets_mean", "targets_std"):
        path = os.path.join(output_dir, f"{key}.npy")
        np.save(path, stats[key])
        print(
            f"[scalars] saved {path}  shape={stats[key].shape}  "
            f"({norm.scaler_kind(stats[key])})"
        )

    # Save metadata
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config": args.config,
        "num_aligned_dates": stats["target_grid_sample_count"],
        "input_channels": int(stats["inputs_mean"].shape[0]),
        "target_channels": int(stats["targets_mean"].shape[0]) if stats["targets_mean"].ndim >= 1 else 0,
        "input_pixel_count": stats["input_pixel_count"],
        "target_pixel_count": stats["target_pixel_count"],
        "scaling_statistics": stats["scaling_statistics"],
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

    # Provenance manifest (sha256 + shapes + training date range) so training
    # and inference can assert they consumed byte-for-byte identical scalers.
    try:
        train_start, train_end = parse_date_range_from_config(cfg, "training")
        train_range = [str(train_start), str(train_end)]
    except Exception:
        train_range = None
    manifest_path = norm.write_manifest(
        output_dir,
        case_name=norm.get_case_name(cfg),
        predictor_mode=str((cfg.get("normalization", {}) or {}).get("predictor_mode", "global")),
        cfg=cfg,
        train_date_range=train_range,
        extra={
            "scaling_statistics": stats["scaling_statistics"],
            "predictor_preprocessing_signature": stats[
                "predictor_preprocessing_signature"
            ],
            norm.TRAINING_SOURCE_ARTIFACT_SIGNATURES_KEY: stats[
                norm.TRAINING_SOURCE_ARTIFACT_SIGNATURES_KEY
            ],
            norm.TRAINING_SOURCE_ARTIFACT_SPLIT_SIGNATURE_KEY: stats[
                norm.TRAINING_SOURCE_ARTIFACT_SPLIT_SIGNATURE_KEY
            ],
        },
    )
    print(f"[scalars] normalization manifest → {manifest_path}")
    print("[scalars] done.")


if __name__ == "__main__":
    main()
