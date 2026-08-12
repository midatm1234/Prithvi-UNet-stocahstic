"""Preprocess NARR predictors and optional PRISM targets for downscaling.

Steps performed:
  1. Read daily NARR files and, for supervised splits, PRISM target files.
  2. Align predictor and target dates when observed targets are requested.
  3. Regrid / interpolate NARR predictors to the PRISM target grid.
  4. Write preprocessed NetCDF files (one per date) to the output directory.

Usage:
    python preproc_narr_prism.py --config NARR_PRISM_subdomain.yaml [--mode training|validation|inference]
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

from narr_prism_utils import (
    align_dates,
    build_narr_regridder,
    discover_all_prism_targets,
    discover_narr_files,
    expand_predictor_variables,
    get_case_name,
    interpolate_narr_to_grid,
    load_elevation,
    load_yaml,
    narr_regrid_cache_path,
    parse_date_range_from_config,
    resolve_path,
    validate_dates_exist,
    validate_target_variables,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from granitewxc.utils import normalization as norm
from granitewxc.utils import prism_grid as prism_grid_contract
from granitewxc.utils.prism_preprocessed import (
    build_preprocessing_contract,
    inclusive_daily_dates,
    preprocessing_attrs,
    source_artifact_signature,
    validate_daily_product,
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


def _mode_saves_targets(cfg: Dict[str, Any], mode: str) -> bool:
    """Return whether a split embeds observed PRISM fields.

    Training and validation require targets. Independent inference is
    predictor-only by default; evaluation reads PRISM separately by exact date.
    """
    pp_cfg = cfg.get("preprocess", {}) or {}
    if mode == "training":
        return bool(pp_cfg.get("save_train_targets", True))
    if mode == "validation":
        return bool(pp_cfg.get("save_val_targets", True))
    if mode == "inference":
        return bool(
            pp_cfg.get("save_inference_targets", False)
            or pp_cfg.get(
                "inference_include_observed_targets_for_eval", False
            )
        )
    raise ValueError(f"Unsupported preprocessing mode: {mode}")


def preprocess(
    cfg: Dict[str, Any],
    mode: str = "training",
    overwrite: bool = False,
    date_shard_index: int = 0,
    date_shard_count: int = 1,
) -> Path:
    """Run the full preprocessing pipeline for training/validation/inference."""
    if xr is None:
        raise ImportError("xarray is required for preprocessing")

    data_cfg = cfg.get("data", {})
    predictor_dir = resolve_path(data_cfg["predictor_dir"])
    target_dir_value = data_cfg.get("target_dir")
    target_dir = resolve_path(target_dir_value) if target_dir_value else None
    target_variables: List[str] = list(data_cfg.get("target_variables", []))
    predictor_variables = expand_predictor_variables(data_cfg.get("predictor_variables", {}))
    include_targets = _mode_saves_targets(cfg, mode)
    regrid_method = str(data_cfg.get("regrid_method", "bilinear")).lower()
    if regrid_method not in {"linear", "bilinear"}:
        raise ValueError(
            "NARR continuous predictors use piecewise-linear barycentric "
            f"regridding; data.regrid_method={regrid_method!r} is unsupported"
        )
    if regrid_method == "bilinear":
        print(
            "[preproc] NOTE: NARR regrid_method='bilinear' is a legacy alias "
            "for curvilinear Delaunay barycentric linear interpolation; prefer "
            "regrid_method='linear' for unambiguous provenance"
        )

    case_name = get_case_name(cfg)
    # Case-scoped layout: <preprocessed_dir>/<case_name>/<mode>. Isolating by
    # case ensures reruns for a different YAML/case never overwrite another
    # case's normalized predictors/targets.
    output_dir = norm.case_preprocess_dir(cfg) / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[preproc] Using case_name: {case_name}")
    print(f"[preproc] Using preprocessing directory: {norm.case_preprocess_dir(cfg)}")
    print(f"[preproc] Writing {mode} outputs to: {output_dir}")
    print(
        f"[preproc] include_targets={include_targets}; independent inference "
        "is predictor-only unless explicitly enabled for evaluation"
    )

    if include_targets:
        if target_dir is None:
            raise ValueError(
                "data.target_dir is required when preprocessing observed "
                "PRISM targets"
            )
        validate_target_variables(target_dir, target_variables)

    # Date range
    start, end = parse_date_range_from_config(cfg, mode)
    print(f"[preproc] {mode} period: {start} → {end}")

    # Discover files
    narr_files = discover_narr_files(predictor_dir, start, end, predictor_variables)
    prism_files = (
        discover_all_prism_targets(
            target_dir, target_variables, start, end
        )
        if include_targets
        else {variable: [] for variable in target_variables}
    )

    if not narr_files:
        raise RuntimeError(f"No NARR files found in {predictor_dir} for {start}–{end}")
    if include_targets:
        for var in target_variables:
            if not prism_files.get(var):
                raise RuntimeError(
                    f"No PRISM files found for variable '{var}' in {start}–{end}"
                )

    # Align dates
    predictor_dates = [d for d, _ in narr_files]
    target_date_maps = {var: [d for d, _ in fl] for var, fl in prism_files.items()}
    requested_dates = inclusive_daily_dates(start, end)
    validate_dates_exist(requested_dates, predictor_dates, "predictor")
    if include_targets:
        for var, available_dates in target_date_maps.items():
            validate_dates_exist(
                requested_dates, available_dates, f"target ({var})"
            )
        aligned_dates = align_dates(predictor_dates, target_date_maps)
        if aligned_dates != requested_dates:
            raise RuntimeError(
                "Aligned NARR/PRISM dates do not exactly cover the configured split"
            )
    else:
        aligned_dates = requested_dates
    if date_shard_count < 1 or not 0 <= date_shard_index < date_shard_count:
        raise ValueError(
            f"invalid date shard {date_shard_index}/{date_shard_count}"
        )
    total_aligned = len(aligned_dates)
    aligned_dates = [
        value
        for idx, value in enumerate(aligned_dates)
        if idx % date_shard_count == date_shard_index
    ]
    print(
        f"[preproc] aligned {total_aligned} dates; shard "
        f"{date_shard_index}/{date_shard_count} processes {len(aligned_dates)}"
    )

    # Build predictor lookup
    pred_map = {d: p for d, p in narr_files}
    target_maps = {var: {d: p for d, p in fl} for var, fl in prism_files.items()}

    # Target-bearing training/validation establishes the canonical PRISM grid.
    # Predictor-only inference reuses that persisted geometry and never opens
    # held-out PRISM merely to rediscover coordinates.
    if include_targets:
        _, (target_lat, target_lon), target_slices = _get_target_grid(
            prism_files, data_cfg
        )
        first_target_path = next(
            str(path)
            for file_list in prism_files.values()
            for _, path in file_list[:1]
        )
        canonical_grid = prism_grid_contract.ensure_canonical_grid(
            norm.case_preprocess_dir(cfg),
            target_lat,
            target_lon,
            source=first_target_path,
            context="NARR preprocessing target grid",
        )
    else:
        canonical_grid = prism_grid_contract.load_canonical_grid(
            norm.case_preprocess_dir(cfg), required=True
        )
        assert canonical_grid is not None
        target_slices = (slice(None), slice(None))
    target_lat, target_lon = canonical_grid.lat, canonical_grid.lon
    print(f"[preproc] PRISM target grid {len(target_lat)}x{len(target_lon)}")
    print(
        f"[preproc] canonical PRISM grid fingerprint="
        f"{canonical_grid.fingerprint[:12]}"
    )

    # NARR coordinates are fixed across variables and dates.  Build the
    # Delaunay simplex/weight geometry once on the complete canonical PRISM
    # grid, cache it under this case, and reuse it for every daily channel.
    first_predictor_var = predictor_variables[0][0]
    regrid_cache = narr_regrid_cache_path(norm.case_preprocess_dir(cfg))
    narr_regridder = build_narr_regridder(
        narr_files[0][1],
        first_predictor_var,
        target_lat,
        target_lon,
        cache_path=regrid_cache,
    )
    cache_state = "loaded" if narr_regridder.loaded_from_cache else "built"
    print(
        f"[preproc] {cache_state} NARR barycentric geometry: "
        f"{narr_regridder.valid_target_count}/{int(np.prod(narr_regridder.target_shape))} "
        f"target cells inside source hull; cache={regrid_cache}"
    )

    # Load static elevation and regrid to the PRISM target grid
    elev_file = data_cfg.get("static_elevation_file", None)
    elev_var = data_cfg.get("static_elevation_var", None)
    elevation_arr: Optional[np.ndarray] = None
    elevation_sha256: Optional[str] = None
    if elev_file:
        elev_path = resolve_path(elev_file)
        if elev_path.exists():
            elevation_arr = load_elevation(
                elev_path, var_name=elev_var or None,
                target_lat=target_lat, target_lon=target_lon,
            )
            elevation_sha256 = norm.sha256_file(elev_path)
            print(f"[preproc] loaded elevation {elevation_arr.shape} from {elev_path.name}")
        else:
            raise FileNotFoundError(
                f"Configured static_elevation_file was not found: {elev_path}"
            )

    required_product_variables = [
        f"predictor_{var}_{int(level)}" for var, level in predictor_variables
    ]
    if include_targets:
        required_product_variables.extend(
            f"target_{var}" for var in target_variables
        )
    if elevation_arr is not None:
        required_product_variables.append("static_elevation")

    preprocessing_contract = build_preprocessing_contract(
        data_type="narr_prism",
        predictor_variables=predictor_variables,
        target_variables=target_variables,
        include_targets=include_targets,
        regrid_method=regrid_method,
        canonical_grid_fingerprint=canonical_grid.fingerprint,
        static_elevation_sha256=elevation_sha256,
        static_elevation_variable=elev_var,
        source_grid_fingerprint=narr_regridder.source_fingerprint,
        algorithm="scipy-delaunay-barycentric-linear-v1",
    )

    # Process each date
    for i, sample_date in enumerate(aligned_dates):
        out_file = output_dir / f"narr_prism_{sample_date:%Y%m%d}.nc"
        narr_file_map = pred_map[sample_date]
        daily_sources = {
            f"predictor:{variable}": path
            for variable, path in narr_file_map.items()
        }
        if include_targets:
            daily_sources.update(
                {
                    f"target:{variable}": target_maps[variable][sample_date]
                    for variable in target_variables
                }
            )
        daily_source_signature = source_artifact_signature(daily_sources)
        if out_file.exists() and not overwrite:
            try:
                with xr.open_dataset(str(out_file)) as existing:
                    validate_daily_product(
                        existing,
                        out_file,
                        canonical_grid,
                        mode=mode,
                        sample_date=sample_date,
                        required_variables=required_product_variables,
                        expected_preprocessing_contract=preprocessing_contract,
                        expected_source_artifact_signature=daily_source_signature,
                    )
            except Exception as exc:
                print(
                    f"[preproc] regenerating stale/invalid {out_file.name}: {exc}"
                )
            else:
                continue

        # Read NARR predictors and regrid. NaNs (e.g. pressure levels that lie
        # BELOW the surface over high terrain — NARR stores these as missing)
        # are PRESERVED, not converted to zero. Bilinear interp/regrid propagates
        # NaN, which conservatively grows the below-surface mask by one stencil.
        pred_arrays: Dict[str, np.ndarray] = {}
        pred_levels: Dict[str, Optional[float]] = {}
        for var, level in predictor_variables:
            channel_name = f"{var}_{int(level)}"
            pred_levels[channel_name] = level
            pred_arrays[channel_name] = interpolate_narr_to_grid(
                narr_file_map,
                var,
                level,
                sample_date,
                target_lat,
                target_lon,
                regridder=narr_regridder,
            )

        # Read PRISM targets. PRISM is NaN over ocean / outside CONUS (~44%);
        # preserve that so the loss can ignore those pixels. ppt keeps its real
        # zeros (dry) which are distinct from NaN (missing).
        tgt_arrays: Dict[str, np.ndarray] = {}
        for var in target_variables if include_targets else ():
            tgt_path = target_maps[var][sample_date]
            with xr.open_dataset(str(tgt_path)) as ds_tgt:
                dvar = _find_numeric_datavar(ds_tgt, tgt_path)
                da = ds_tgt[dvar]
                if "time" in da.dims:
                    da = da.isel(time=0, drop=True)
                lat_slice, lon_slice = target_slices
                lat_name = _infer_coord_name(ds_tgt, LAT_CANDIDATES)
                lon_name = _infer_coord_name(ds_tgt, LON_CANDIDATES)
                observed_lat = np.asarray(
                    ds_tgt[lat_name].values, dtype=np.float64
                )[lat_slice]
                observed_lon = np.asarray(
                    ds_tgt[lon_name].values, dtype=np.float64
                )[lon_slice]
                prism_grid_contract.assert_grid_matches(
                    canonical_grid,
                    observed_lat,
                    observed_lon,
                    context=f"PRISM target {tgt_path}",
                )
                if lat_name not in da.dims or lon_name not in da.dims:
                    raise ValueError(
                        f"PRISM target {tgt_path} variable {dvar!r} must use "
                        f"coordinate dimensions ({lat_name!r}, {lon_name!r}); "
                        f"got {da.dims}"
                    )
                da = da.transpose(lat_name, lon_name)
                arr = da.values[lat_slice, lon_slice].astype(np.float32)
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
            if tgt_arrays:
                for var, arr in tgt_arrays.items():
                    _diagnose_field(f"target_{var}", None, arr)
            else:
                print(
                    "    [diag] targets omitted from predictor-only "
                    "inference preprocessing"
                )

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
                "source_narr": ";".join(
                    f"{var}={path}" for var, path in sorted(narr_file_map.items())
                ),
                "has_elevation": str(elevation_arr is not None),
                "mode": mode,
                "prism_grid_fingerprint": canonical_grid.fingerprint,
                "source_artifact_signature": daily_source_signature,
                **preprocessing_attrs(preprocessing_contract),
                "contains_targets": str(bool(tgt_arrays)),
                "targets_are_optional_evaluation_data": str(
                    mode == "inference" and bool(tgt_arrays)
                ),
                "missing_value_note": (
                    "NaN marks missing data: pressure-level predictors below the "
                    "surface over high terrain. When target_* variables are "
                    "present, target NaNs mark ocean / outside CONUS. These "
                    "are NOT physical zeros."
                ),
            },
        )
        # Preserve NaN explicitly via _FillValue / missing_value on every float
        # field so the missing-data mask survives a NetCDF round-trip.
        encoding = {
            name: {"_FillValue": FILL_VALUE, "dtype": "float32"}
            for name in out_ds_vars
        }
        # A killed date shard must not leave a partial file that a later run
        # mistakes for a complete preprocessing product.
        temporary = out_file.with_name(f".{out_file.name}.{os.getpid()}.tmp")
        try:
            out_ds.to_netcdf(str(temporary), encoding=encoding)
            os.replace(temporary, out_file)
        finally:
            temporary.unlink(missing_ok=True)

        if (i + 1) % 100 == 0 or i == 0:
            print(f"[preproc] processed {i + 1}/{len(aligned_dates)} dates")

    print(f"[preproc] {mode} preprocessing complete → {output_dir}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess NARR predictors and PRISM targets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to NARR_PRISM_subdomain.yaml")
    parser.add_argument(
        "--mode",
        choices=["training", "validation", "inference", "both"],
        default="both",
        help="Which date range to preprocess",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--date-shard-index", type=int, default=0)
    parser.add_argument("--date-shard-count", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    if args.mode == "both":
        modes = ["training"]
        validation = (cfg.get("dates") or {}).get("validation") or {}
        if validation.get("start") and validation.get("end"):
            modes.append("validation")
        modes.append("inference")
    else:
        modes = [args.mode]
    for mode in modes:
        preprocess(
            cfg,
            mode=mode,
            overwrite=args.overwrite,
            date_shard_index=args.date_shard_index,
            date_shard_count=args.date_shard_count,
        )


if __name__ == "__main__":
    main()
