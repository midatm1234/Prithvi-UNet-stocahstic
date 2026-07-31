#!/usr/bin/env python3
"""Evaluate NARR-PRISM or MERRA-PRISM daily inference without regridding.

The evaluator streams one prediction/truth day at a time, requires the case's
persisted canonical PRISM grid, and writes a flat CSV, a provenance-rich JSON,
and a four-panel climatology/RMSE figure.  It is intentionally shared by both
example pipelines and has no model or dataset dependency.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import sys
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from granitewxc.utils.normalization import case_preprocess_dir  # noqa: E402
from granitewxc.utils.prism_evaluation import (  # noqa: E402
    StreamingMetricMeans,
    StreamingVariableMetrics,
    boundary_gradient_error,
    exact_axis_slice,
    map_native_coordinate_boundaries,
    median_boundary_period,
    prefix_metrics,
    spatial_power_metrics,
    validate_evaluation_coordinates,
)
from granitewxc.utils.prism_grid import (  # noqa: E402
    CanonicalPrismGrid,
    load_canonical_grid,
    validate_prism_grid,
)
from granitewxc.utils.prism_tiling import (  # noqa: E402
    TilePlan,
    overlap_crossovers,
)


DATE_RE = re.compile(r"(\d{8})")
LAT_NAMES = ("lat", "latitude", "y")
LON_NAMES = ("lon", "longitude", "x")
UNITS = {"ppt": "mm day$^{-1}$", "tmax": r"$^\circ$C", "tmin": r"$^\circ$C"}


def _resolve_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict) or not cfg.get("case_name"):
        raise ValueError(f"{config_path}: expected a mapping with case_name")
    if not isinstance(cfg.get("data"), dict):
        raise ValueError(f"{config_path}: data section is required")
    return cfg


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _daily_range(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError(f"end date {end} precedes start date {start}")
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def _pair(value: Any, *, name: str, default: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, (int, np.integer)):
        return int(value), int(value)
    values = tuple(int(v) for v in value)
    if len(values) != 2:
        raise ValueError(f"{name} must be an integer or [lat,lon], got {value!r}")
    return values


def _coord_name(ds: Any, candidates: Sequence[str], *, context: str) -> str:
    lowered = {str(name).lower(): str(name) for name in ds.variables}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    raise ValueError(f"{context}: cannot find coordinate among {tuple(candidates)}")


def _spatial_data_variable(
    ds: Any,
    variable: str,
    lat_name: str,
    lon_name: str,
    *,
    context: str,
    allow_generic_fallback: bool,
) -> str:
    # Inference must identify the requested predictand by name.  Falling back
    # to the first numeric field could silently score tmax using ppt merely
    # because ppt happens to be first in the NetCDF file.
    named_matches = [
        str(name)
        for name, data_array in ds.data_vars.items()
        if str(name).casefold() == str(variable).casefold()
        and lat_name in data_array.dims
        and lon_name in data_array.dims
    ]
    if len(named_matches) == 1:
        return named_matches[0]
    if len(named_matches) > 1:
        raise ValueError(
            f"{context}: multiple case-insensitive variables match {variable!r}: "
            f"{named_matches}"
        )
    if not allow_generic_fallback:
        raise ValueError(
            f"{context}: requested inference variable {variable!r} is absent; "
            f"available data variables are {list(map(str, ds.data_vars))}"
        )

    # Daily PRISM rasters are commonly single-band files named Band1.  This
    # generic path is deliberately available to truth only.
    for preferred in ("Band1", "band1"):
        if preferred in ds.data_vars:
            dims = set(ds[preferred].dims)
            if lat_name in dims and lon_name in dims:
                return str(preferred)
    for name, data_array in ds.data_vars.items():
        dims = set(data_array.dims)
        if (
            lat_name in dims
            and lon_name in dims
            and np.issubdtype(data_array.dtype, np.number)
        ):
            return str(name)
    raise ValueError(
        f"{context}: no numeric {lat_name}/{lon_name} data variable found for {variable}"
    )


def _attribute_text(ds: Any, name: str, *, context: str) -> str:
    if name not in ds.attrs:
        raise ValueError(f"{context}: required global attribute {name!r} is absent")
    raw = ds.attrs[name]
    value = (
        raw.decode("utf-8", errors="strict") if isinstance(raw, bytes) else str(raw)
    ).strip()
    if not value:
        raise ValueError(f"{context}: required global attribute {name!r} is empty")
    return value


def _dated_files(directory: Path, pattern: str = "*.nc*") -> dict[date, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    result: dict[date, Path] = {}
    for path in sorted(directory.glob(pattern)):
        if not path.is_file():
            continue
        match = DATE_RE.search(path.name)
        if match is None:
            continue
        try:
            file_date = datetime.strptime(match.group(1), "%Y%m%d").date()
        except ValueError:
            continue
        if file_date in result:
            raise ValueError(
                f"Duplicate daily files for {file_date}: {result[file_date]} and {path}"
            )
        result[file_date] = path
    return result


def _inference_directory(cfg: Mapping[str, Any], override: str | None) -> Path:
    case_name = str(cfg["case_name"])
    raw = override or (cfg.get("inference", {}) or {}).get(
        "output_dir", "./experiments/inference_output"
    )
    root = _resolve_path(str(raw))
    # An explicit direct case directory is accepted; otherwise retain the same
    # case-scoping convention as the two inference entry points.
    if root.name == case_name or any(root.glob("*inference_*.nc")):
        return root
    return root / case_name


def _truth_files(target_root: Path, variable: str, dates: Iterable[date]) -> dict[date, Path]:
    needed = set(dates)
    found: dict[date, Path] = {}
    for year in sorted({d.year for d in needed}):
        year_dir = target_root / variable / str(year)
        if not year_dir.is_dir():
            continue
        for day, path in _dated_files(year_dir).items():
            if day in needed:
                found[day] = path
    return found


def _read_inference_day(
    path: Path,
    variable: str,
    grid: CanonicalPrismGrid,
    *,
    case_name: str,
    expected_date: date,
    allow_legacy_float32: bool,
) -> tuple[np.ndarray, bool, dict[str, str]]:
    import xarray as xr

    with xr.open_dataset(path, decode_times=False, mask_and_scale=True) as ds:
        stored_case = _attribute_text(ds, "case_name", context=str(path))
        if stored_case != case_name:
            raise ValueError(
                f"{path}: inference case_name={stored_case!r} does not match "
                f"configured case {case_name!r}"
            )
        checkpoint = _attribute_text(ds, "checkpoint", context=str(path))
        for attr_name in ("inference_start", "inference_end"):
            stored_date = _attribute_text(ds, attr_name, context=str(path))
            if stored_date != expected_date.isoformat():
                raise ValueError(
                    f"{path}: {attr_name}={stored_date!r} does not match file date "
                    f"{expected_date.isoformat()!r}"
                )

        if "time" not in ds.variables:
            raise ValueError(f"{path}: daily inference file has no time coordinate")
        time_values = np.asarray(ds["time"].values).reshape(-1)
        if time_values.size != 1:
            raise ValueError(
                f"{path}: daily inference file must contain exactly one time value; "
                f"got {time_values.size}"
            )
        raw_units = ds["time"].attrs.get("units", "")
        time_units = (
            raw_units.decode("utf-8", errors="strict")
            if isinstance(raw_units, bytes)
            else str(raw_units)
        ).strip()
        if not time_units.startswith("days since 1970-01-01"):
            raise ValueError(
                f"{path}: unsupported time units {time_units!r}; expected days since "
                "1970-01-01"
            )
        expected_epoch_day = (expected_date - date(1970, 1, 1)).days
        observed_epoch_day = float(time_values[0])
        if not np.isfinite(observed_epoch_day) or observed_epoch_day != expected_epoch_day:
            raise ValueError(
                f"{path}: time={observed_epoch_day!r} does not encode expected date "
                f"{expected_date.isoformat()} (epoch day {expected_epoch_day})"
            )

        legacy_metadata = False
        stored_fingerprint = ds.attrs.get("prism_grid_fingerprint")
        if stored_fingerprint is None:
            if not allow_legacy_float32:
                raise ValueError(
                    f"{path}: required global attribute 'prism_grid_fingerprint' is "
                    "absent. Only an explicit --legacy-float32-coordinates audit may "
                    "read pre-contract output."
                )
            legacy_metadata = True
        else:
            stored_fingerprint = (
                stored_fingerprint.decode("utf-8", errors="strict")
                if isinstance(stored_fingerprint, bytes)
                else str(stored_fingerprint)
            ).strip()
            if stored_fingerprint != grid.fingerprint:
                raise ValueError(
                    f"{path}: prism_grid_fingerprint={stored_fingerprint!r} does not "
                    f"match canonical fingerprint {grid.fingerprint!r}"
                )

        lat_name = _coord_name(ds, LAT_NAMES, context=str(path))
        lon_name = _coord_name(ds, LON_NAMES, context=str(path))
        legacy_lat = validate_evaluation_coordinates(
            grid.lat,
            ds[lat_name].values,
            name="latitude",
            context=str(path),
            allow_legacy_float32=allow_legacy_float32,
        )
        legacy_lon = validate_evaluation_coordinates(
            grid.lon,
            ds[lon_name].values,
            name="longitude",
            context=str(path),
            allow_legacy_float32=allow_legacy_float32,
        )
        data_name = _spatial_data_variable(
            ds,
            variable,
            lat_name,
            lon_name,
            context=str(path),
            allow_generic_fallback=False,
        )
        da = ds[data_name]
        extra_dims = [dim for dim in da.dims if dim not in (lat_name, lon_name)]
        for dim in extra_dims:
            if int(da.sizes[dim]) != 1:
                raise ValueError(
                    f"{path}: daily inference variable {data_name!r} has "
                    f"non-singleton extra dimension {dim}={da.sizes[dim]}"
                )
            da = da.isel({dim: 0}, drop=True)
        values = np.asarray(da.transpose(lat_name, lon_name).values, dtype=np.float64)
    if values.shape != grid.shape:
        raise ValueError(
            f"{path}: inference array shape {values.shape} != canonical {grid.shape}"
        )
    values[~np.isfinite(values)] = np.nan
    return values, bool(legacy_lat or legacy_lon or legacy_metadata), {
        "checkpoint": checkpoint,
        "case_name": stored_case,
        "date": expected_date.isoformat(),
    }


def _read_truth_day(path: Path, variable: str, grid: CanonicalPrismGrid) -> np.ndarray:
    import xarray as xr

    with xr.open_dataset(path, decode_times=False, mask_and_scale=True) as ds:
        lat_name = _coord_name(ds, LAT_NAMES, context=str(path))
        lon_name = _coord_name(ds, LON_NAMES, context=str(path))
        lat_slice = exact_axis_slice(
            ds[lat_name].values, grid.lat, name="latitude", context=str(path)
        )
        lon_slice = exact_axis_slice(
            ds[lon_name].values, grid.lon, name="longitude", context=str(path)
        )
        data_name = _spatial_data_variable(
            ds,
            variable,
            lat_name,
            lon_name,
            context=str(path),
            allow_generic_fallback=True,
        )
        da = ds[data_name]
        extra_dims = [dim for dim in da.dims if dim not in (lat_name, lon_name)]
        for dim in extra_dims:
            if int(da.sizes[dim]) != 1:
                raise ValueError(
                    f"{path}: PRISM variable {data_name!r} has non-singleton "
                    f"extra dimension {dim}={da.sizes[dim]}"
                )
            da = da.isel({dim: 0}, drop=True)
        values = np.asarray(
            da.transpose(lat_name, lon_name)
            .isel({lat_name: lat_slice, lon_name: lon_slice})
            .values,
            dtype=np.float64,
        )
    if values.shape != grid.shape:
        raise ValueError(f"{path}: PRISM array shape {values.shape} != {grid.shape}")
    values[~np.isfinite(values)] = np.nan
    return values


def _legacy_subset_axis(
    axis: np.ndarray, lower: Any, upper: Any, *, name: str
) -> np.ndarray:
    coords = np.asarray(axis, dtype=np.float64)
    if lower is None and upper is None:
        return coords
    lo = float(np.nanmin(coords) if lower is None else lower)
    hi = float(np.nanmax(coords) if upper is None else upper)
    if hi < lo:
        lo, hi = hi, lo
    steps = np.abs(np.diff(coords))
    steps = steps[np.isfinite(steps) & (steps > 0)]
    tolerance = float(np.median(steps)) * 0.51 if steps.size else 0.0
    indices = np.flatnonzero((coords >= lo - tolerance) & (coords <= hi + tolerance))
    if not indices.size:
        raise ValueError(f"Legacy {name} subset [{lo}, {hi}] selected no PRISM cells")
    return coords[int(indices[0]) : int(indices[-1]) + 1]


def _legacy_grid_from_truth(
    cfg: Mapping[str, Any], truth_path: Path
) -> CanonicalPrismGrid:
    """Reconstruct a read-only pre-contract grid for explicit before audits."""

    import xarray as xr

    subset = (cfg.get("data", {}) or {}).get("spatial_subset", {}) or {}
    with xr.open_dataset(truth_path, decode_times=False) as ds:
        lat_name = _coord_name(ds, LAT_NAMES, context=str(truth_path))
        lon_name = _coord_name(ds, LON_NAMES, context=str(truth_path))
        lat = np.asarray(ds[lat_name].values, dtype=np.float64)
        lon = np.asarray(ds[lon_name].values, dtype=np.float64)
    if bool(subset.get("enabled", False)):
        lat = _legacy_subset_axis(
            lat, subset.get("lat_min"), subset.get("lat_max"), name="latitude"
        )
        lon = _legacy_subset_axis(
            lon, subset.get("lon_min"), subset.get("lon_max"), name="longitude"
        )
    return validate_prism_grid(lat, lon, context="legacy config-derived PRISM grid")


def _tile_geometry(
    cfg: Mapping[str, Any], shape: tuple[int, int]
) -> tuple[list[int], list[int], dict[str, Any]]:
    inference = cfg.get("inference", {}) or {}
    mitigation = inference.get("boundary_mitigation", {}) or {}
    force_full = bool(
        inference.get("force_full_frame", mitigation.get("force_full_frame", False))
    )
    if force_full:
        return [], [], {
            "force_full_frame": True,
            "tile_size": list(shape),
            "overlap": [0, 0],
            "lat_origins": [0],
            "lon_origins": [0],
            "lat_crossovers": [],
            "lon_crossovers": [],
        }
    tile = _pair(
        inference.get("inference_tile_size", mitigation.get("tile_size")),
        name="inference tile size",
        default=(256, 256),
    )
    overlap = _pair(
        inference.get("inference_overlap", mitigation.get("overlap")),
        name="inference overlap",
        default=(64, 64),
    )
    plan = TilePlan.build(shape, tile, overlap=overlap)
    y_crossovers = overlap_crossovers(plan.lat_origins, plan.core_shape[0])
    x_crossovers = overlap_crossovers(plan.lon_origins, plan.core_shape[1])
    return y_crossovers, x_crossovers, {
        "force_full_frame": False,
        "tile_size": list(plan.core_shape),
        "overlap": list(plan.overlap),
        "lat_origins": list(plan.lat_origins),
        "lon_origins": list(plan.lon_origins),
        "lat_crossovers": y_crossovers,
        "lon_crossovers": x_crossovers,
    }


def _first_predictor_file(cfg: Mapping[str, Any], override: str | None) -> Path | None:
    if override:
        path = _resolve_path(override)
        if not path.is_file():
            raise FileNotFoundError(f"Native predictor grid file not found: {path}")
        return path
    predictor_dir = _resolve_path(str((cfg.get("data", {}) or {}).get("predictor_dir", "")))
    if not predictor_dir.is_dir():
        return None
    for pattern in ("*.nc", "*.nc4", "*.cdf"):
        candidates = sorted(predictor_dir.rglob(pattern))
        if candidates:
            return candidates[0]
    return None


def _fallback_native_geometry(
    cfg: Mapping[str, Any], grid: CanonicalPrismGrid
) -> tuple[list[int], list[int], tuple[float, float], str]:
    data_type = str((cfg.get("data", {}) or {}).get("type", "")).lower()
    target_dlat = float(np.median(np.abs(np.diff(grid.lat))))
    target_dlon = float(np.median(np.abs(np.diff(grid.lon))))
    if "merra" in data_type:
        periods = (0.5 / target_dlat, 0.625 / target_dlon)
    else:
        mid_lat = float(np.mean(grid.lat))
        target_dy_km = 111.32 * target_dlat
        target_dx_km = 111.32 * np.cos(np.deg2rad(mid_lat)) * target_dlon
        periods = (32.463 / target_dy_km, 32.463 / target_dx_km)
    y_period, x_period = (max(2, int(round(v))) for v in periods)
    y_positions = list(range(y_period // 2, grid.shape[0], y_period))
    x_positions = list(range(x_period // 2, grid.shape[1], x_period))
    return y_positions, x_positions, periods, f"nominal-{data_type or 'unknown'}"


def _native_geometry(
    cfg: Mapping[str, Any],
    grid: CanonicalPrismGrid,
    *,
    predictor_override: str | None,
    period_override: tuple[float, float] | None,
) -> tuple[list[int], list[int], tuple[float, float], dict[str, Any]]:
    if period_override is not None:
        py, px = period_override
        if py <= 1 or px <= 1:
            raise ValueError("--native-period-pixels values must both exceed one")
        y_positions = list(range(max(1, int(round(py / 2))), grid.shape[0], int(round(py))))
        x_positions = list(range(max(1, int(round(px / 2))), grid.shape[1], int(round(px))))
        return y_positions, x_positions, (py, px), {
            "method": "explicit-period",
            "source_file": None,
            "lat_boundaries": y_positions,
            "lon_boundaries": x_positions,
        }

    predictor_path = _first_predictor_file(cfg, predictor_override)
    if predictor_path is not None:
        import xarray as xr

        try:
            with xr.open_dataset(predictor_path, decode_times=False) as ds:
                lat_name = _coord_name(ds, ("lat", "latitude"), context=str(predictor_path))
                lon_name = _coord_name(ds, ("lon", "longitude"), context=str(predictor_path))
                native_lat = np.asarray(ds[lat_name].values, dtype=np.float64)
                native_lon = np.asarray(ds[lon_name].values, dtype=np.float64)
            if native_lat.ndim == 1 and native_lon.ndim == 1:
                lat_transect, lon_transect = native_lat, native_lon
                method = "native-1d-cell-boundaries"
            elif native_lat.ndim == 2 and native_lon.ndim == 2:
                # A curvilinear NARR cell edge is not axis-aligned on PRISM.
                # Median row/column transects are a reproducible approximation.
                lat_transect = np.nanmedian(native_lat, axis=1)
                lon_transect = np.nanmedian(native_lon, axis=0)
                method = "curvilinear-median-transect-approximation"
            else:
                raise ValueError(
                    f"unsupported native coordinate shapes {native_lat.shape}/{native_lon.shape}"
                )
            y_positions = map_native_coordinate_boundaries(grid.lat, lat_transect)
            x_positions = map_native_coordinate_boundaries(grid.lon, lon_transect)
            periods = (
                median_boundary_period(y_positions),
                median_boundary_period(x_positions),
            )
            if y_positions and x_positions and all(np.isfinite(periods)):
                return y_positions, x_positions, periods, {
                    "method": method,
                    "source_file": str(predictor_path),
                    "lat_boundaries": y_positions,
                    "lon_boundaries": x_positions,
                }
        except (OSError, ValueError) as exc:
            warnings.warn(
                f"Could not derive approximate native boundaries from {predictor_path}: {exc}; "
                "using nominal grid spacing",
                RuntimeWarning,
            )

    y_positions, x_positions, periods, method = _fallback_native_geometry(cfg, grid)
    return y_positions, x_positions, periods, {
        "method": method,
        "source_file": str(predictor_path) if predictor_path else None,
        "lat_boundaries": y_positions,
        "lon_boundaries": x_positions,
    }


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _finite_or_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_or_none(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = sorted({str(key) for row in rows for key in row})
    leading = [name for name in ("variable", "days", "mean_bias", "climatology_rmse") if name in fieldnames]
    fieldnames = leading + [name for name in fieldnames if name not in leading]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_four_panel(
    path: Path,
    *,
    case_name: str,
    variable: str,
    start: date,
    end: date,
    lat: np.ndarray,
    lon: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    bias: np.ndarray,
    rmse: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    common = np.concatenate(
        [prediction[np.isfinite(prediction)], target[np.isfinite(target)]]
    )
    if not common.size:
        raise ValueError(f"Cannot plot {variable}: climatologies contain no finite values")
    mean_vmin, mean_vmax = np.quantile(common, [0.01, 0.99])
    finite_bias = np.abs(bias[np.isfinite(bias)])
    bias_limit = float(np.quantile(finite_bias, 0.99)) if finite_bias.size else 1.0
    bias_limit = max(bias_limit, np.finfo(np.float32).eps)
    finite_rmse = rmse[np.isfinite(rmse)]
    rmse_max = float(np.quantile(finite_rmse, 0.99)) if finite_rmse.size else 1.0
    rmse_max = max(rmse_max, np.finfo(np.float32).eps)
    extent = [float(lon[0]), float(lon[-1]), float(lat[0]), float(lat[-1])]
    origin = "lower" if lat[-1] > lat[0] else "upper"
    unit = UNITS.get(variable, "")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    panels = (
        (axes[0, 0], prediction, "Inference climatology", "magma", mean_vmin, mean_vmax),
        (axes[0, 1], target, "PRISM climatology", "magma", mean_vmin, mean_vmax),
        (axes[1, 0], bias, "Inference - PRISM", "RdBu_r", -bias_limit, bias_limit),
        (axes[1, 1], rmse, "Temporal RMSE", "inferno", 0.0, rmse_max),
    )
    for axis, values, title, cmap, vmin, vmax in panels:
        image = axis.imshow(
            values,
            origin=origin,
            extent=extent,
            cmap=cmap,
            vmin=float(vmin),
            vmax=float(vmax),
            interpolation="nearest",
            aspect="auto",
        )
        axis.set_title(title)
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
        fig.colorbar(image, ax=axis, shrink=0.82, label=unit)
    fig.suptitle(
        f"{variable} evaluation: {case_name} ({start.isoformat()} to {end.isoformat()})",
        fontsize=15,
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _parse_period(value: str | None) -> tuple[float, float] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("native period must be LAT_PIXELS,LON_PIXELS")
    return float(parts[0]), float(parts[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream exact-grid NARR/MERRA PRISM inference diagnostics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="NARR_PRISM or MERRA_PRISM YAML")
    parser.add_argument("--inference-dir", default=None, help="Inference root or direct case directory")
    parser.add_argument("--output-dir", default=None, help="Directory for CSV/JSON/PNG")
    parser.add_argument("--canonical-grid-dir", default=None, help="Override case directory containing prism_grid.npz/json")
    parser.add_argument("--start", default=None, help="First evaluation date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="Last evaluation date (YYYY-MM-DD)")
    parser.add_argument("--variables", nargs="+", default=None, help="Target variables to evaluate")
    parser.add_argument("--figure-variable", default="tmax", help="Variable for the four-panel figure")
    parser.add_argument("--run-label", default="evaluation", help="Filename label such as before/after")
    parser.add_argument("--block-period", type=float, default=8.0, help="Pixel period used for decoder-block spectral power")
    parser.add_argument("--boundary-half-width", type=int, default=2, help="Half-width around boundary-gradient lines")
    parser.add_argument("--native-grid-file", default=None, help="Optional representative native predictor NetCDF")
    parser.add_argument("--native-period-pixels", default=None, help="Override approximate native spacing as LAT,LON pixels")
    parser.add_argument("--allow-missing-dates", action="store_true", help="Evaluate only dates with both inference and truth")
    parser.add_argument(
        "--legacy-float32-coordinates",
        action="store_true",
        help="Audit old float32-coordinate outputs within 5e-6 degrees; never interpolates",
    )
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N days (0 disables)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config)
    case_name = str(cfg["case_name"])
    data_cfg = cfg["data"]
    variables = list(args.variables or data_cfg.get("target_variables", []))
    if not variables:
        raise ValueError("No variables supplied and data.target_variables is empty")
    if args.figure_variable not in variables:
        raise ValueError(
            f"--figure-variable {args.figure_variable!r} must be in {variables}"
        )

    configured_dates = cfg.get("dates", {}).get("inference", {})
    start = _parse_date(args.start or str(configured_dates.get("start")))
    end = _parse_date(args.end or str(configured_dates.get("end")))
    requested_dates = _daily_range(start, end)

    inference_dir = _inference_directory(cfg, args.inference_dir)
    inference_map = _dated_files(inference_dir, "*inference_*.nc")
    target_root = _resolve_path(str(data_cfg["target_dir"]))

    grid_dir = (
        _resolve_path(args.canonical_grid_dir)
        if args.canonical_grid_dir
        else case_preprocess_dir(cfg)
    )
    grid = load_canonical_grid(grid_dir, required=False)
    grid_source = "persisted-contract"
    if grid is None:
        if not args.legacy_float32_coordinates:
            raise FileNotFoundError(
                f"No canonical PRISM grid contract under {grid_dir}. Run preprocessing "
                "for this exact config/case before evaluation. For a read-only audit of "
                "old float32 files only, pass --legacy-float32-coordinates."
            )
        first_var = variables[0]
        seed_truth = _truth_files(target_root, first_var, requested_dates)
        if not seed_truth:
            raise FileNotFoundError(f"No PRISM truth files found for {first_var}")
        grid = _legacy_grid_from_truth(cfg, seed_truth[min(seed_truth)])
        grid_source = "legacy-read-only-config-bounds"
        warnings.warn(
            "Canonical contract is absent; reconstructing the exact float64 PRISM subset "
            "from config bounds for this explicit legacy before-audit. No grid is written.",
            RuntimeWarning,
        )
    assert grid is not None

    missing_inference = [day for day in requested_dates if day not in inference_map]
    if missing_inference and not args.allow_missing_dates:
        preview = ", ".join(str(day) for day in missing_inference[:5])
        raise FileNotFoundError(
            f"Missing {len(missing_inference)} inference days under {inference_dir}: "
            f"{preview}{' ...' if len(missing_inference) > 5 else ''}"
        )
    evaluation_dates = [day for day in requested_dates if day in inference_map]
    if not evaluation_dates:
        raise ValueError("No inference dates remain for evaluation")

    y_crossovers, x_crossovers, tile_metadata = _tile_geometry(cfg, grid.shape)
    native_period_override = _parse_period(args.native_period_pixels)
    y_native, x_native, native_period, native_metadata = _native_geometry(
        cfg,
        grid,
        predictor_override=args.native_grid_file,
        period_override=native_period_override,
    )

    output_dir = (
        _resolve_path(args.output_dir)
        if args.output_dir
        else _resolve_path(str(cfg.get("path_experiment", "./experiments")))
        / "comparison_plots"
        / case_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(args.run_label)).strip("_")
    stem = f"{case_name}_{safe_label}_{start:%Y%m%d}_{end:%Y%m%d}"
    csv_path = output_dir / f"{stem}_metrics.csv"
    json_path = output_dir / f"{stem}_metrics.json"
    figure_path = output_dir / f"{stem}_{args.figure_variable}_comparison.png"

    rows: list[dict[str, Any]] = []
    variable_json: dict[str, Any] = {}
    figure_fields: dict[str, np.ndarray] | None = None
    legacy_paths: set[Path] = set()
    checkpoint_paths: set[str] = set()
    actual_date_sets: dict[str, list[date]] = {}
    for variable in variables:
        truth_map = _truth_files(target_root, variable, evaluation_dates)
        missing_truth = [day for day in evaluation_dates if day not in truth_map]
        if missing_truth and not args.allow_missing_dates:
            preview = ", ".join(str(day) for day in missing_truth[:5])
            raise FileNotFoundError(
                f"Missing {len(missing_truth)} PRISM {variable} days: {preview}"
            )
        variable_dates = [day for day in evaluation_dates if day in truth_map]
        if not variable_dates:
            raise ValueError(f"No paired inference/PRISM dates for {variable}")
        actual_date_sets[variable] = variable_dates
        accumulator = StreamingVariableMetrics(grid.shape)
        daily_crossover_accumulator = StreamingMetricMeans()
        daily_native_accumulator = StreamingMetricMeans()
        daily_power_accumulator = StreamingMetricMeans()
        for day_index, day in enumerate(variable_dates, start=1):
            prediction, used_legacy, inference_provenance = _read_inference_day(
                inference_map[day],
                variable,
                grid,
                case_name=case_name,
                expected_date=day,
                allow_legacy_float32=args.legacy_float32_coordinates,
            )
            truth = _read_truth_day(truth_map[day], variable, grid)
            accumulator.update(prediction, truth)
            daily_crossover_accumulator.update(
                boundary_gradient_error(
                    prediction,
                    truth,
                    y_positions=y_crossovers,
                    x_positions=x_crossovers,
                    half_width=args.boundary_half_width,
                )
            )
            daily_native_accumulator.update(
                boundary_gradient_error(
                    prediction,
                    truth,
                    y_positions=y_native,
                    x_positions=x_native,
                    half_width=args.boundary_half_width,
                )
            )
            daily_power_accumulator.update(
                spatial_power_metrics(
                    prediction,
                    truth,
                    block_period=args.block_period,
                    native_period=native_period,
                )
            )
            checkpoint_paths.add(inference_provenance["checkpoint"])
            if used_legacy:
                legacy_paths.add(inference_map[day])
            if args.progress_every > 0 and (
                day_index % args.progress_every == 0 or day_index == len(variable_dates)
            ):
                print(
                    f"[evaluation] {variable}: {day_index}/{len(variable_dates)} "
                    f"({day})",
                    flush=True,
                )

        finalized = accumulator.finalize()
        crossover_metrics = boundary_gradient_error(
            finalized.prediction_climatology,
            finalized.target_climatology,
            y_positions=y_crossovers,
            x_positions=x_crossovers,
            half_width=args.boundary_half_width,
        )
        native_metrics = boundary_gradient_error(
            finalized.prediction_climatology,
            finalized.target_climatology,
            y_positions=y_native,
            x_positions=x_native,
            half_width=args.boundary_half_width,
        )
        power_metrics = spatial_power_metrics(
            finalized.prediction_climatology,
            finalized.target_climatology,
            block_period=args.block_period,
            native_period=native_period,
        )
        daily_crossover_metrics = daily_crossover_accumulator.finalize()
        daily_native_metrics = daily_native_accumulator.finalize()
        daily_power_metrics = daily_power_accumulator.finalize()
        row = {"variable": variable, **finalized.scalars}
        row.update(prefix_metrics("overlap_crossover", crossover_metrics))
        row.update(prefix_metrics("native_boundary", native_metrics))
        row.update(prefix_metrics("power", power_metrics))
        row.update(
            prefix_metrics("daily_mean_overlap_crossover", daily_crossover_metrics)
        )
        row.update(prefix_metrics("daily_mean_native_boundary", daily_native_metrics))
        row.update(prefix_metrics("daily_mean_power", daily_power_metrics))
        rows.append(row)
        variable_json[variable] = {
            "metrics": finalized.scalars,
            "overlap_crossover_gradient_error": crossover_metrics,
            "approximate_native_boundary_gradient_error": native_metrics,
            "spatial_power": power_metrics,
            "daily_mean_overlap_crossover_gradient_error": daily_crossover_metrics,
            "daily_mean_approximate_native_boundary_gradient_error": daily_native_metrics,
            "daily_mean_spatial_power": daily_power_metrics,
        }
        if variable == args.figure_variable:
            figure_fields = {
                "prediction": finalized.prediction_climatology.copy(),
                "target": finalized.target_climatology.copy(),
                "bias": finalized.bias_climatology.copy(),
                "rmse": finalized.temporal_rmse.copy(),
            }
        del (
            accumulator,
            daily_crossover_accumulator,
            daily_native_accumulator,
            daily_power_accumulator,
            finalized,
        )
        gc.collect()

    if len(checkpoint_paths) != 1:
        raise ValueError(
            "Evaluation input mixes checkpoint provenance across daily files: "
            f"{sorted(checkpoint_paths)}. Evaluate each checkpoint separately."
        )
    checkpoint_path = next(iter(checkpoint_paths))

    metadata: dict[str, Any] = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "case_name": case_name,
        "config": str(Path(args.config).expanduser().resolve()),
        "inference_directory": str(inference_dir),
        "inference_checkpoint": checkpoint_path,
        "target_directory": str(target_root),
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "requested_day_count": len(requested_dates),
        "evaluated_days_by_variable": {
            variable: {
                "count": len(days),
                "start": min(days).isoformat(),
                "end": max(days).isoformat(),
            }
            for variable, days in actual_date_sets.items()
        },
        "canonical_grid": {
            "source": grid_source,
            "directory": str(grid_dir),
            **grid.manifest_entry(),
            "legacy_float32_coordinate_files": len(legacy_paths),
        },
        "truth_alignment": "exact contiguous canonical coordinates; no interpolation",
        "tile_geometry": tile_metadata,
        "approximate_native_grid": {
            **native_metadata,
            "period_lat_pixels": native_period[0],
            "period_lon_pixels": native_period[1],
        },
        "metric_definitions": {
            "climatology_rmse": "spatial RMSE between paired-day inference and PRISM climatologies",
            "spatial_correlation": "Pearson correlation of paired-day climatologies over valid cells",
            "boundary_gradient_error": "abs(diff(inference climatology) - diff(PRISM climatology)) in boundary bands",
            "daily_mean_boundary_gradient_error": "arithmetic mean of paired-day boundary-gradient diagnostics; cannot cancel between days",
            "block_power": f"normalized spectral band energy around wavelength {args.block_period:g} pixels",
            "daily_mean_block_power": "arithmetic mean of paired-day normalized spectral diagnostics",
        },
        "variables": variable_json,
    }
    _write_csv(csv_path, rows)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(_finite_or_none(metadata), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")

    if figure_fields is None:
        raise AssertionError("Figure variable was not accumulated")
    figure_dates = actual_date_sets[args.figure_variable]
    _plot_four_panel(
        figure_path,
        case_name=case_name,
        variable=args.figure_variable,
        start=min(figure_dates),
        end=max(figure_dates),
        lat=grid.lat,
        lon=grid.lon,
        **figure_fields,
    )
    print(f"[evaluation] CSV: {csv_path}")
    print(f"[evaluation] JSON: {json_path}")
    print(f"[evaluation] figure: {figure_path}")


if __name__ == "__main__":
    main()
