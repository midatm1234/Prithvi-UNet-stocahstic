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
    discover_all_prism_targets,
    discover_merra2_files,
    load_yaml,
    parse_date_range_from_config,
    resolve_path,
    validate_dates_exist,
    validate_target_variables,
)


LAT_CANDIDATES = ("lat", "latitude", "y")
LON_CANDIDATES = ("lon", "longitude", "x")


def _infer_coord_name(ds: Any, candidates: Sequence[str]) -> str:
    for name in candidates:
        if name in ds.coords or name in ds.data_vars:
            return name
    raise ValueError(f"Cannot find coordinate among {candidates}")


def _build_grid(ds: Any, lat_name: str, lon_name: str) -> Any:
    """Build a lightweight xr.Dataset with just lat/lon for regridding."""
    lat = ds[lat_name]
    lon = ds[lon_name]
    if lat.name != "lat":
        lat = lat.rename("lat")
    if lon.name != "lon":
        lon = lon.rename("lon")
    return xr.Dataset({"lat": lat, "lon": lon})


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


def _get_target_grid(prism_files: Dict[str, List[Tuple[Any, Path]]]) -> Tuple[Any, Any]:
    """Open the first available PRISM file and extract its lat/lon grid."""
    for var, flist in prism_files.items():
        if flist:
            _, first_path = flist[0]
            with xr.open_dataset(str(first_path)) as ds:
                lat_name = _infer_coord_name(ds, LAT_CANDIDATES)
                lon_name = _infer_coord_name(ds, LON_CANDIDATES)
                grid = _build_grid(ds, lat_name, lon_name)
                lat_vals = ds[lat_name].values
                lon_vals = ds[lon_name].values
                return grid, (lat_vals, lon_vals)
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
    predictor_variables: List[str] = list(data_cfg.get("predictor_variables", []))
    regrid_method: str = data_cfg.get("regrid_method", "bilinear")

    output_dir = resolve_path(data_cfg.get("preprocessed_dir", "./preprocessed"))
    output_dir = output_dir / mode
    output_dir.mkdir(parents=True, exist_ok=True)

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
    target_grid, (target_lat, target_lon) = _get_target_grid(prism_files)

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

        # Read MERRA2 predictors and regrid
        merra_path = pred_map[sample_date]
        pred_arrays: Dict[str, np.ndarray] = {}
        with xr.open_dataset(str(merra_path)) as ds_pred:
            for var in predictor_variables:
                if var not in ds_pred.data_vars:
                    print(f"[preproc] WARN: variable '{var}' missing in {merra_path}, filling with zeros")
                    pred_arrays[var] = np.zeros((len(target_lat), len(target_lon)), dtype=np.float32)
                    continue
                da = ds_pred[var]
                if "time" in da.dims:
                    da = da.isel(time=0, drop=True)
                da = _rename_lat_lon(da, pred_lat_name, pred_lon_name)
                if regridder is not None:
                    regridded = regridder(da)
                    arr = regridded.values.astype(np.float32)
                else:
                    arr = da.interp(lat=target_lat, lon=target_lon, method="linear").values.astype(np.float32)
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                pred_arrays[var] = arr

        # Read PRISM targets
        tgt_arrays: Dict[str, np.ndarray] = {}
        for var in target_variables:
            tgt_path = target_maps[var][sample_date]
            with xr.open_dataset(str(tgt_path)) as ds_tgt:
                dvar = list(ds_tgt.data_vars)[0]
                da = ds_tgt[dvar]
                if "time" in da.dims:
                    da = da.isel(time=0, drop=True)
                arr = da.values.astype(np.float32)
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                tgt_arrays[var] = arr

        # Build output dataset
        out_ds_vars: Dict[str, Any] = {}
        for var, arr in pred_arrays.items():
            out_ds_vars[f"predictor_{var}"] = (("lat", "lon"), arr)
        for var, arr in tgt_arrays.items():
            out_ds_vars[f"target_{var}"] = (("lat", "lon"), arr)

        out_ds = xr.Dataset(
            out_ds_vars,
            coords={"lat": target_lat, "lon": target_lon},
            attrs={
                "date": str(sample_date),
                "source_merra2": str(merra_path),
                "mode": mode,
            },
        )
        out_ds.to_netcdf(str(out_file))

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
