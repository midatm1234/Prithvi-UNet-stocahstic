"""Preprocess MERRA2 predictors and PRISM targets for the downscaling workflow.

Steps performed:
  1. Read daily MERRA2 files and PRISM target files for the requested period.
  2. Align predictor and target dates.
  3. Regrid / interpolate MERRA2 predictors to the PRISM target grid.
  4. Write preprocessed NetCDF files (one per date) to the output directory.

Usage:
    python preproc_merra_prism.py --config MERRA_PRISM.yaml [--mode training|inference]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import xarray as xr
except ImportError:
    xr = None

try:
    import xesmf as xe
except Exception:
    xe = None

from merra_prism_utils import (
    align_dates,
    case_output_dir,
    discover_all_prism_targets,
    discover_merra2_files,
    expand_predictor_variables,
    get_case_name,
    load_elevation,
    load_yaml,
    parse_date_range_from_config,
    resolve_path,
    validate_dates_exist,
    validate_target_variables,
)


LAT_CANDIDATES = ("lat", "latitude", "y")
LON_CANDIDATES = ("lon", "longitude", "x")


def _find_numeric_datavar(ds: Any, path: Any) -> str:
    """Return the first data variable with a numeric dtype, skipping metadata vars like 'crs'."""
    import numpy as np
    for v in ds.data_vars:
        if np.issubdtype(ds[v].dtype, np.number):
            return v
    raise ValueError(f"No numeric data variable found in {path}")


def _infer_coord_name(ds: Any, candidates: Sequence[str]) -> str:
    for name in candidates:
        if name in ds.coords or name in ds.data_vars:
            return name
    raise ValueError(f"Cannot find coordinate among {candidates}")


# Output NetCDF fill value for float fields. NaN is preserved on disk via this
# _FillValue / missing_value so that below-surface (high-terrain) pressure-level
# cells and ocean/outside-CONUS target cells remain MISSING rather than becoming
# real zeros.
FILL_VALUE = np.float32(np.nan)


def _diagnose_field(name: str, level: Optional[float], arr: np.ndarray) -> int:
    """Print NaN / zero / skipna-range diagnostics for one regridded field.

    Returns the number of NaN cells so callers can sanity-check that masked
    (below-surface) regions stayed NaN instead of collapsing to zeros.
    """
    total = arr.size
    nan_mask = ~np.isfinite(arr)
    n_nan = int(nan_mask.sum())
    n_zero = int((arr == 0).sum())
    finite = arr[~nan_mask]
    if finite.size:
        vmin, vmax, vmean = float(finite.min()), float(finite.max()), float(finite.mean())
    else:
        vmin = vmax = vmean = float("nan")
    lvl = "" if level is None else f"@{int(level)}hPa"
    print(
        f"    [diag] {name}{lvl}: NaN={n_nan}/{total} ({100.0 * n_nan / total:.2f}%) "
        f"zeros={n_zero} min={vmin:.3f} max={vmax:.3f} mean={vmean:.3f} (skipna)"
    )
    return n_nan


def _build_grid(ds: Any, lat_name: str, lon_name: str) -> Any:
    """Build a lightweight xr.Dataset with just lat/lon for regridding."""
    lat = ds[lat_name]
    lon = ds[lon_name]
    if lat.name != "lat":
        lat = lat.rename("lat")
    if lon.name != "lon":
        lon = lon.rename("lon")
    return xr.Dataset({"lat": lat, "lon": lon})


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


def _rename_lat_lon(da: Any, lat_name: str, lon_name: str) -> Any:
    rename_map = {}
    if lat_name in da.dims and lat_name != "lat":
        rename_map[lat_name] = "lat"
    if lon_name in da.dims and lon_name != "lon":
        rename_map[lon_name] = "lon"
    if rename_map:
        da = da.rename(rename_map)
    coord_map = {}
    if lat_name in da.coords and lat_name != "lat":
        coord_map[lat_name] = "lat"
    if lon_name in da.coords and lon_name != "lon":
        coord_map[lon_name] = "lon"
    if coord_map:
        da = da.rename(coord_map)
    return da


def _build_regridder(
    coarse_grid: Any,
    fine_grid: Any,
    method: str = "bilinear",
) -> Any:
    """Build an xESMF regridder, or fall back to xarray interp."""
    if xe is not None:
        try:
            return xe.Regridder(coarse_grid, fine_grid, method=method, periodic=False)
        except Exception as exc:
            print(f"[preproc] xESMF init failed ({exc}); falling back to xarray interp")
    return None


def _get_target_grid(
    prism_files: Dict[str, List[Tuple[Any, Path]]],
    data_cfg: Dict[str, Any],
) -> Tuple[Any, Any, Tuple[slice, slice]]:
    """Open the first available PRISM file and extract its lat/lon grid."""
    for var, flist in prism_files.items():
        if flist:
            _, first_path = flist[0]
            with xr.open_dataset(str(first_path)) as ds:
                lat_name = _infer_coord_name(ds, LAT_CANDIDATES)
                lon_name = _infer_coord_name(ds, LON_CANDIDATES)
                lat_vals = ds[lat_name].values
                lon_vals = ds[lon_name].values
                lat_slice, lon_slice = _spatial_subset_slices(lat_vals, lon_vals, data_cfg)
                lat_vals = lat_vals[lat_slice]
                lon_vals = lon_vals[lon_slice]
                grid = xr.Dataset({"lat": ("lat", lat_vals), "lon": ("lon", lon_vals)})
                return grid, (lat_vals, lon_vals), (lat_slice, lon_slice)
    raise RuntimeError("No PRISM target files found to extract grid")


def preprocess(
    cfg: Dict[str, Any],
    mode: str = "training",
    overwrite: bool = False,
) -> Path:
    """Run the full preprocessing pipeline for *mode* (training | inference)."""
    if xr is None:
        raise ImportError("xarray is required for preprocessing")

    data_cfg = cfg.get("data", {})
    predictor_dir = resolve_path(data_cfg["predictor_dir"])
    target_dir = resolve_path(data_cfg["target_dir"])
    target_variables: List[str] = list(data_cfg.get("target_variables", []))
    predictor_variables = expand_predictor_variables(data_cfg.get("predictor_variables", {}))
    regrid_method: str = data_cfg.get("regrid_method", "bilinear")

    case_name = get_case_name(cfg)
    output_dir = case_output_dir(
        resolve_path(data_cfg.get("preprocessed_dir", "./preprocessed")) / mode,
        case_name,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[preproc] case_name={case_name}")
    print(f"[preproc] output_dir={output_dir}")

    validate_target_variables(target_dir, target_variables)

    # Date range
    start, end = parse_date_range_from_config(cfg, mode)
    print(f"[preproc] {mode} period: {start} → {end}")

    # Discover files
    merra_files = discover_merra2_files(predictor_dir, start, end)
    prism_files = discover_all_prism_targets(target_dir, target_variables, start, end)

    if not merra_files:
        raise RuntimeError(f"No MERRA2 files found in {predictor_dir} for {start}–{end}")
    for var in target_variables:
        if not prism_files.get(var):
            raise RuntimeError(f"No PRISM files found for variable '{var}' in {start}–{end}")

    # Align dates
    predictor_dates = [d for d, _ in merra_files]
    target_date_maps = {var: [d for d, _ in fl] for var, fl in prism_files.items()}
    aligned_dates = align_dates(predictor_dates, target_date_maps)
    print(f"[preproc] aligned {len(aligned_dates)} dates")

    validate_dates_exist(aligned_dates, predictor_dates, "predictor")

    # Build predictor lookup
    pred_map = {d: p for d, p in merra_files}
    target_maps = {var: {d: p for d, p in fl} for var, fl in prism_files.items()}

    # Get target grid from first PRISM file
    target_grid, (target_lat, target_lon), target_slices = _get_target_grid(prism_files, data_cfg)
    print(f"[preproc] PRISM target grid {len(target_lat)}x{len(target_lon)}")

    # Load static elevation and regrid to the PRISM target grid
    elev_file = data_cfg.get("static_elevation_file", None)
    elev_var = data_cfg.get("static_elevation_var", None)
    elevation_arr: Optional[np.ndarray] = None
    if elev_file:
        elev_path = resolve_path(elev_file)
        if elev_path.exists():
            elevation_arr = load_elevation(
                elev_path, var_name=elev_var or None,
                target_lat=target_lat, target_lon=target_lon,
            )
            print(f"[preproc] loaded elevation {elevation_arr.shape} from {elev_path.name}")
        else:
            print(f"[preproc] WARN: elevation file not found: {elev_path}")

    # Build regridder from first MERRA2 file
    regridder = None
    first_merra_path = pred_map[aligned_dates[0]]
    with xr.open_dataset(str(first_merra_path)) as ds_pred:
        pred_lat_name = _infer_coord_name(ds_pred, LAT_CANDIDATES)
        pred_lon_name = _infer_coord_name(ds_pred, LON_CANDIDATES)
        pred_grid = _build_grid(ds_pred, pred_lat_name, pred_lon_name)
        regridder = _build_regridder(pred_grid, target_grid, method=regrid_method)

    # Process each date
    for i, sample_date in enumerate(aligned_dates):
        out_file = output_dir / f"merra_prism_{sample_date:%Y%m%d}.nc"
        if out_file.exists() and not overwrite:
            continue

        # Read MERRA2 predictors and regrid. NaNs (e.g. pressure levels that lie
        # BELOW the surface over high terrain — MERRA stores these as missing)
        # are PRESERVED, not converted to zero. Bilinear interp/regrid propagates
        # NaN, which conservatively grows the below-surface mask by one stencil.
        merra_path = pred_map[sample_date]
        pred_arrays: Dict[str, np.ndarray] = {}
        pred_levels: Dict[str, Optional[float]] = {}
        with xr.open_dataset(str(merra_path)) as ds_pred:
            for var, level in predictor_variables:
                channel_name = f"{var}_{int(level)}"
                pred_levels[channel_name] = level
                if var not in ds_pred.data_vars:
                    # Entire variable missing: write NaN (missing), NOT zeros, so
                    # downstream masking treats it as invalid rather than real 0.
                    print(f"[preproc] WARN: variable '{var}' missing in {merra_path}, filling with NaN")
                    pred_arrays[channel_name] = np.full(
                        (len(target_lat), len(target_lon)), np.nan, dtype=np.float32
                    )
                    continue
                da = ds_pred[var]
                if "time" in da.dims:
                    da = da.isel(time=0, drop=True)
                if "lev" in da.dims:
                    da = da.sel(lev=level, drop=True)
                da = _rename_lat_lon(da, pred_lat_name, pred_lon_name)
                if regridder is not None:
                    regridded = regridder(da)
                    arr = regridded.values.astype(np.float32)
                else:
                    arr = da.interp(lat=target_lat, lon=target_lon, method="linear").values.astype(np.float32)
                # Keep +/-inf out (shouldn't occur) but DO NOT touch NaN.
                arr[np.isinf(arr)] = np.nan
                pred_arrays[channel_name] = arr

        # Read PRISM targets. PRISM is NaN over ocean / outside CONUS (~44%);
        # preserve that so the loss can ignore those pixels. ppt keeps its real
        # zeros (dry) which are distinct from NaN (missing).
        tgt_arrays: Dict[str, np.ndarray] = {}
        for var in target_variables:
            tgt_path = target_maps[var][sample_date]
            with xr.open_dataset(str(tgt_path)) as ds_tgt:
                dvar = _find_numeric_datavar(ds_tgt, tgt_path)
                da = ds_tgt[dvar]
                if "time" in da.dims:
                    da = da.isel(time=0, drop=True)
                arr = da.values.astype(np.float32)
                lat_slice, lon_slice = target_slices
                if arr.ndim >= 2:
                    arr = arr[lat_slice, lon_slice]
                arr[np.isinf(arr)] = np.nan
                tgt_arrays[var] = arr

        # Diagnostics: confirm below-surface predictor cells stayed NaN and warn
        # if a pressure-level field collapsed to suspiciously many zeros (a sign
        # NaNs were silently converted to 0 somewhere upstream).
        if (i + 1) % 100 == 0 or i == 0:
            print(f"[preproc] {sample_date:%Y-%m-%d} field diagnostics:")
            for channel_name, arr in pred_arrays.items():
                n_nan = _diagnose_field(channel_name, pred_levels.get(channel_name), arr)
                lvl = pred_levels.get(channel_name)
                # Upper-air 700/850 hPa fields are expected to be NaN over high
                # terrain; a near-total absence of NaN combined with many exact
                # zeros usually means the missing-data mask was lost.
                n_zero = int((arr == 0).sum())
                if lvl is not None and lvl >= 700 and n_nan == 0 and n_zero > 0:
                    print(
                        f"    [preproc] WARNING: {channel_name} has 0 NaN but {n_zero} "
                        f"exact zeros — below-surface cells may have been zero-filled."
                    )
            for var, arr in tgt_arrays.items():
                _diagnose_field(f"target_{var}", None, arr)

        # Build output dataset
        out_ds_vars: Dict[str, Any] = {}
        for var, arr in pred_arrays.items():
            out_ds_vars[f"predictor_{var}"] = (("lat", "lon"), arr)
        for var, arr in tgt_arrays.items():
            out_ds_vars[f"target_{var}"] = (("lat", "lon"), arr)
        # Embed static elevation so each preprocessed file is self-contained
        if elevation_arr is not None:
            out_ds_vars["static_elevation"] = (("lat", "lon"), elevation_arr)

        out_ds = xr.Dataset(
            out_ds_vars,
            coords={"lat": target_lat, "lon": target_lon},
            attrs={
                "date": str(sample_date),
                "source_merra2": str(merra_path),
                "has_elevation": str(elevation_arr is not None),
                "mode": mode,
                "missing_value_note": (
                    "NaN marks missing data: pressure-level predictors below the "
                    "surface over high terrain, and PRISM targets over ocean / "
                    "outside CONUS. These are NOT physical zeros."
                ),
            },
        )
        # Preserve NaN explicitly via _FillValue / missing_value on every float
        # field so the missing-data mask survives a NetCDF round-trip.
        encoding = {
            name: {"_FillValue": FILL_VALUE, "dtype": "float32"}
            for name in out_ds_vars
        }
        out_ds.to_netcdf(str(out_file), encoding=encoding)

        if (i + 1) % 100 == 0 or i == 0:
            print(f"[preproc] processed {i + 1}/{len(aligned_dates)} dates")

    print(f"[preproc] {mode} preprocessing complete → {output_dir}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess MERRA2 predictors and PRISM targets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to MERRA_PRISM.yaml")
    parser.add_argument(
        "--mode",
        choices=["training", "inference", "both"],
        default="both",
        help="Which date range to preprocess",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    modes = ["training", "inference"] if args.mode == "both" else [args.mode]
    for mode in modes:
        preprocess(cfg, mode=mode, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
