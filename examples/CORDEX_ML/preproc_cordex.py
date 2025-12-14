"""Regrid CORDEX coarse predictors onto target grid for CORDEX-ML-Bench."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Sequence

import xarray as xr
import xesmf as xe


LAT_CANDIDATES = ("lat", "latitude", "rlat", "y")
LON_CANDIDATES = ("lon", "longitude", "rlon", "x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regrid coarse CORDEX predictors to the target grid",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--predictor-files",
        nargs="+",
        required=True,
        help="Paths to coarse predictor NetCDF files",
    )
    parser.add_argument(
        "--target-sample",
        required=True,
        help="High-resolution NetCDF file that defines the target grid",
    )
    parser.add_argument(
        "--orography-file",
        required=True,
        help="Static orography NetCDF file",
    )
    parser.add_argument(
        "--orography-var",
        default="orog",
        help="Name of the orography variable inside the static file",
    )
    parser.add_argument(
        "--input-vars",
        nargs="+",
        default=["u", "v", "q", "t", "z"],
        help="Base variable names to interpolate",
    )
    parser.add_argument(
        "--input-levels",
        nargs="+",
        default=["850", "700", "500"],
        help="Pressure levels (strings) appended to variable names",
    )
    parser.add_argument(
        "--regrid-method",
        default="bilinear",
        help="xESMF regridding method",
    )
    parser.add_argument(
        "--output-dir",
        default="./preprocessed",
        help="Directory to write the regridded NetCDF files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files if they already exist",
    )
    return parser.parse_args()


def infer_coord_name(ds: xr.Dataset, candidates: Iterable[str]) -> str:
    for name in candidates:
        if name in ds.coords or name in ds.data_vars:
            return name
    raise ValueError(f"Unable to find coordinate among {candidates}")


def rename_array(array: xr.DataArray, old_name: str, new_name: str) -> xr.DataArray:
    result = array
    if old_name != new_name and old_name in result.dims:
        result = result.rename({old_name: new_name})
    if result.name != new_name:
        result = result.rename(new_name)
    return result


def rename_lat_lon(data: xr.DataArray, lat_name: str, lon_name: str) -> xr.DataArray:
    rename_map = {}
    if lat_name in data.dims:
        rename_map[lat_name] = "lat"
    if lon_name in data.dims:
        rename_map[lon_name] = "lon"
    if rename_map:
        data = data.rename(rename_map)

    coord_map = {}
    if lat_name in data.coords:
        coord_map[lat_name] = "lat"
    if lon_name in data.coords:
        coord_map[lon_name] = "lon"
    if coord_map:
        data = data.rename(coord_map)

    return data


def build_grid(ds: xr.Dataset, lat_name: str, lon_name: str) -> xr.Dataset:
    lat = rename_array(ds[lat_name], lat_name, "lat")
    lon = rename_array(ds[lon_name], lon_name, "lon")
    return xr.Dataset({"lat": lat, "lon": lon})


def load_regridded_orography(
    orog_path: str,
    fine_grid: xr.Dataset,
    regrid_method: str,
    var_name: str,
) -> xr.DataArray:
    with xr.open_dataset(orog_path) as ds:
        if var_name not in ds.data_vars:
            raise ValueError(f"Orography variable '{var_name}' missing in {orog_path}")
        lat_name = infer_coord_name(ds, LAT_CANDIDATES)
        lon_name = infer_coord_name(ds, LON_CANDIDATES)

        grid_in = build_grid(ds, lat_name, lon_name)
        regridder = xe.Regridder(grid_in, fine_grid, method=regrid_method, periodic=False)
        data = rename_lat_lon(ds[var_name], lat_name, lon_name)
        regridded = regridder(data)
        return regridded.rename(var_name)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with xr.open_dataset(args.target_sample) as target_ds:
        fine_lat_name = infer_coord_name(target_ds, LAT_CANDIDATES)
        fine_lon_name = infer_coord_name(target_ds, LON_CANDIDATES)
        fine_grid = build_grid(target_ds, fine_lat_name, fine_lon_name)

    orography = load_regridded_orography(
        args.orography_file, fine_grid, args.regrid_method, args.orography_var
    )

    var_names = [f"{var}_{level}" for var in args.input_vars for level in args.input_levels]

    for predictor_path in args.predictor_files:
        predictor_path = os.fspath(predictor_path)
        output_path = Path(args.output_dir) / f"{Path(predictor_path).stem}_regridded.nc"
        if output_path.exists() and not args.overwrite:
            print(f"[Skip] {output_path} already exists")
            continue

        with xr.open_dataset(predictor_path) as ds:
            lat_name = infer_coord_name(ds, LAT_CANDIDATES)
            lon_name = infer_coord_name(ds, LON_CANDIDATES)
            grid_in = build_grid(ds, lat_name, lon_name)
            regridder = xe.Regridder(grid_in, fine_grid, method=args.regrid_method, periodic=False)

            out_vars: List[xr.DataArray] = []
            out_names: List[str] = []

            for var_name in var_names:
                if var_name not in ds.data_vars:
                    print(f"[Warn] {var_name} missing in {predictor_path}, skipping")
                    continue

                data = rename_lat_lon(ds[var_name], lat_name, lon_name)
                regridded = regridder(data)
                out_vars.append(regridded.rename(var_name))
                out_names.append(var_name)

        output_ds = xr.Dataset({name: var for name, var in zip(out_names, out_vars)})
        output_ds[args.orography_var] = orography
        output_ds.to_netcdf(output_path)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()

