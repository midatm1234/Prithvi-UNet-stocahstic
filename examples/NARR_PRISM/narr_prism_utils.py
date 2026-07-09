"""Shared utilities for the NARR-to-PRISM downscaling workflow."""

from __future__ import annotations

import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file and return the contents as a dict."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"YAML config not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at the top level of {path}")
    get_case_name(data)
    return data


def resolve_path(raw: str | Path) -> Path:
    """Return an absolute path; resolve relative paths against REPO_ROOT."""
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def get_case_name(cfg: Any) -> str:
    """Return the configured case name from a dict-like or object config."""
    if isinstance(cfg, dict):
        case_name = cfg.get("case_name")
    else:
        case_name = getattr(cfg, "case_name", None)
    if not case_name:
        raise ValueError("case_name must be set in the YAML config")
    return str(case_name)


def case_output_dir(base_dir: str | Path, case_name: str) -> Path:
    """Resolve an output root and append case_name unless it is already present."""
    path = resolve_path(base_dir)
    if path.name == case_name:
        return path
    return path / case_name


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

_DATE_FMT = "%Y-%m-%d"


def parse_date(value: str) -> date:
    """Parse an ISO-style date string (YYYY-MM-DD)."""
    return datetime.strptime(value, _DATE_FMT).date()


def date_range(start: date, end: date) -> List[date]:
    """Return a list of dates from *start* to *end* inclusive."""
    if end < start:
        raise ValueError(f"end date {end} is before start date {start}")
    days = (end - start).days + 1
    return [start + timedelta(days=i) for i in range(days)]


def parse_date_range_from_config(cfg: Dict[str, Any], section: str) -> Tuple[date, date]:
    """Extract start/end dates from ``cfg['dates'][section]``."""
    dates_block = cfg.get("dates", {})
    section_block = dates_block.get(section, {})
    start_str = section_block.get("start")
    end_str = section_block.get("end")
    if not start_str or not end_str:
        raise ValueError(
            f"dates.{section}.start and dates.{section}.end must be set in the YAML config"
        )
    return parse_date(str(start_str)), parse_date(str(end_str))


# ---------------------------------------------------------------------------
# NARR file discovery
# ---------------------------------------------------------------------------

_NARR_MONTH_RE = re.compile(r"(\d{6})")

NARR_VARIABLE_ALIASES = {
    "QV": "shum",
    "Q": "shum",
    "SHUM": "shum",
    "U": "uwnd",
    "UWND": "uwnd",
    "V": "vwnd",
    "VWND": "vwnd",
    "T": "air",
    "AIR": "air",
    "H": "hgt",
    "HGT": "hgt",
}


def narr_source_var(var_name: str) -> str:
    """Return the NARR variable/file stem for a configured predictor name."""
    return NARR_VARIABLE_ALIASES.get(var_name.upper(), var_name.lower())


def _month_start(value: date) -> date:
    return date(value.year, value.month, 1)


def _next_month(value: date) -> date:
    if value.month == 12:
        return date(value.year + 1, 1, 1)
    return date(value.year, value.month + 1, 1)


def _iter_months(start: date, end: date) -> Sequence[date]:
    months: List[date] = []
    cur = _month_start(start)
    while cur <= end:
        months.append(cur)
        cur = _next_month(cur)
    return months


def discover_narr_files(
    predictor_dir: str | Path,
    start: date,
    end: date,
    predictor_variables: Optional[Sequence[Tuple[str, float]]] = None,
) -> List[Tuple[date, Dict[str, Path]]]:
    """Find NARR monthly files covering daily dates in [start, end].

    NARR files are stored as ``<predictor_dir>/<variable>/<variable>.YYYYMM.nc``.
    The returned path is a mapping from configured predictor name to that
    predictor's monthly NetCDF file for the date.
    """
    predictor_dir = Path(predictor_dir)
    if not predictor_dir.is_dir():
        raise FileNotFoundError(f"NARR predictor directory not found: {predictor_dir}")

    configured_vars = sorted(
        {var for var, _ in predictor_variables}
        if predictor_variables
        else {p.name for p in predictor_dir.iterdir() if p.is_dir()}
    )
    if not configured_vars:
        raise ValueError("No NARR predictor variables were specified or discovered")

    monthly_files: Dict[str, Dict[date, Path]] = {}
    for var in configured_vars:
        source_var = narr_source_var(var)
        var_dir = predictor_dir / source_var
        if not var_dir.is_dir():
            raise FileNotFoundError(f"NARR variable directory not found: {var_dir}")

        by_month: Dict[date, Path] = {}
        for entry in sorted(var_dir.glob("*.nc")):
            match = _NARR_MONTH_RE.search(entry.name)
            if match is None:
                continue
            try:
                month = datetime.strptime(match.group(1), "%Y%m").date()
            except ValueError:
                continue
            by_month[_month_start(month)] = entry
        monthly_files[var] = by_month

    results: List[Tuple[date, Dict[str, Path]]] = []
    for month in _iter_months(start, end):
        month_end = _next_month(month) - timedelta(days=1)
        first_day = max(start, month)
        last_day = min(end, month_end)
        file_map: Dict[str, Path] = {}
        missing: List[str] = []
        for var in configured_vars:
            path = monthly_files[var].get(month)
            if path is None:
                missing.append(f"{var} ({narr_source_var(var)})")
            else:
                file_map[var] = path
        if missing:
            raise FileNotFoundError(
                f"Missing NARR monthly files for {month:%Y-%m}: {', '.join(missing)}"
            )
        for day in date_range(first_day, last_day):
            results.append((day, dict(file_map)))

    results.sort(key=lambda t: t[0])
    return results


def _select_narr_day_level(da: Any, sample_date: date, level: float) -> Any:
    if "time" in da.dims:
        da = da.sel(time=np.datetime64(sample_date), drop=True)
    if "level" in da.dims:
        da = da.sel(level=level, drop=True)
    elif "lev" in da.dims:
        da = da.sel(lev=level, drop=True)
    return da


def interpolate_narr_to_grid(
    file_map: Mapping[str, Path] | Path,
    var: str,
    level: float,
    sample_date: date,
    target_lat: Any,
    target_lon: Any,
) -> "np.ndarray":
    """Load one NARR channel and interpolate it onto a 1D PRISM lat/lon grid."""
    try:
        import numpy as np
        import xarray as xr
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    except ImportError as exc:
        raise ImportError("numpy, xarray, and scipy are required for NARR interpolation") from exc

    if isinstance(file_map, Mapping):
        path = Path(file_map[var])
    else:
        path = Path(file_map)
    source_var = narr_source_var(var)

    with xr.open_dataset(str(path)) as ds:
        if source_var not in ds.data_vars:
            raise ValueError(f"NARR file {path} does not contain variable '{source_var}'")
        da = _select_narr_day_level(ds[source_var], sample_date, level)
        values = np.asarray(da.values, dtype=np.float32)
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
        lon = np.asarray(ds["lon"].values, dtype=np.float64)

    points = np.column_stack([lon.ravel(), lat.ravel()])
    flat_values = values.ravel()
    finite = np.isfinite(flat_values) & np.isfinite(points).all(axis=1)
    target_lon2d, target_lat2d = np.meshgrid(
        np.asarray(target_lon, dtype=np.float64),
        np.asarray(target_lat, dtype=np.float64),
    )
    target_points = np.column_stack([target_lon2d.ravel(), target_lat2d.ravel()])

    out = np.full(target_points.shape[0], np.nan, dtype=np.float32)
    if finite.any():
        linear = LinearNDInterpolator(points[finite], flat_values[finite], fill_value=np.nan)
        out = np.asarray(linear(target_points), dtype=np.float32)
        missing = ~np.isfinite(out)
        if missing.any():
            nearest = NearestNDInterpolator(points[finite], flat_values[finite])
            nearest_values = np.asarray(nearest(target_points[missing]), dtype=np.float32)
            out[missing] = nearest_values
    out[np.isinf(out)] = np.nan
    return out.reshape(target_lat2d.shape).astype(np.float32)


# ---------------------------------------------------------------------------
# PRISM file discovery
# ---------------------------------------------------------------------------

_PRISM_DATE_RE = re.compile(r"(\d{8})")


def discover_prism_files(
    target_dir: str | Path,
    variable: str,
    start: date,
    end: date,
) -> List[Tuple[date, Path]]:
    """Find PRISM daily NetCDF files for *variable* in [start, end].

    Expected layout: ``<target_dir>/<variable>/<YYYY>/*.nc``
    """
    var_root = Path(target_dir) / variable
    if not var_root.is_dir():
        raise FileNotFoundError(
            f"PRISM variable directory not found: {var_root}"
        )

    results: List[Tuple[date, Path]] = []
    start_year = start.year
    end_year = end.year
    for year in range(start_year, end_year + 1):
        year_dir = var_root / str(year)
        if not year_dir.is_dir():
            continue
        for entry in sorted(year_dir.iterdir()):
            if not entry.is_file() or not entry.suffix == ".nc":
                continue
            match = _PRISM_DATE_RE.search(entry.name)
            if match is None:
                continue
            try:
                file_date = datetime.strptime(match.group(1), "%Y%m%d").date()
            except ValueError:
                continue
            if start <= file_date <= end:
                results.append((file_date, entry))

    results.sort(key=lambda t: t[0])
    return results


def discover_all_prism_targets(
    target_dir: str | Path,
    target_variables: Sequence[str],
    start: date,
    end: date,
) -> Dict[str, List[Tuple[date, Path]]]:
    """Discover PRISM files for each target variable."""
    result: Dict[str, List[Tuple[date, Path]]] = {}
    for var in target_variables:
        files = discover_prism_files(target_dir, var, start, end)
        result[var] = files
    return result


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def align_dates(
    predictor_dates: Sequence[date],
    target_date_maps: Dict[str, Sequence[date]],
) -> List[date]:
    """Return the sorted intersection of predictor dates and all target variable dates."""
    common = set(predictor_dates)
    for var, dates in target_date_maps.items():
        target_set = set(dates)
        common = common & target_set
        if not common:
            raise ValueError(
                f"No overlapping dates between predictors and target variable '{var}'"
            )
    aligned = sorted(common)
    if not aligned:
        raise ValueError("Date alignment produced an empty set")
    return aligned


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_dates_exist(
    requested: Sequence[date],
    available: Sequence[date],
    label: str,
) -> None:
    """Check that all *requested* dates are present in *available*."""
    available_set = set(available)
    missing = [d for d in requested if d not in available_set]
    if missing:
        sample = missing[:5]
        raise ValueError(
            f"{len(missing)} requested {label} dates are missing from input files. "
            f"First missing: {sample}"
        )


def validate_target_variables(
    target_dir: str | Path,
    target_variables: Sequence[str],
) -> None:
    """Confirm that each target variable subfolder exists under *target_dir*."""
    target_dir = Path(target_dir)
    for var in target_variables:
        var_dir = target_dir / var
        if not var_dir.is_dir():
            raise FileNotFoundError(
                f"Target variable directory not found: {var_dir}. "
                f"Expected layout: {target_dir}/<variable>/<YYYY>/*.nc"
            )


# ---------------------------------------------------------------------------
# Predictor variable expansion
# ---------------------------------------------------------------------------

def expand_predictor_variables(
    predictor_cfg: Dict[str, Any],
) -> List[Tuple[str, float]]:
    """Expand a predictor_variables mapping into a list of (var_name, level) tuples.

    The YAML format is::

        predictor_variables:
          QV: [500, 700, 850]
          U:  [500, 700, 850]

    Returns a flat list such as ``[("QV", 500.0), ("QV", 700.0), ..., ("U", 500.0), ...]``.
    """
    if not isinstance(predictor_cfg, dict):
        raise TypeError(
            "predictor_variables must be a mapping of variable names to pressure-level "
            "lists, e.g.  QV: [500, 700, 850].  Got a flat list — update the YAML."
        )
    specs: List[Tuple[str, float]] = []
    for var, levels in predictor_cfg.items():
        for lev in levels:
            specs.append((str(var), float(lev)))
    return specs


# ---------------------------------------------------------------------------
# Static elevation helpers
# ---------------------------------------------------------------------------

def load_elevation(
    elevation_file: str | Path,
    var_name: Optional[str] = None,
    target_lat: Optional[Any] = None,
    target_lon: Optional[Any] = None,
) -> "np.ndarray":
    """Load the static elevation field and optionally regrid to *target_lat/lon*.

    Parameters
    ----------
    elevation_file : path
        Path to ``prism_elevation.nc`` (or similar).
    var_name : str | None
        Variable name inside the file.  Auto-detected when None.
    target_lat, target_lon : array-like | None
        If provided, the elevation field is interpolated to this grid.

    Returns
    -------
    np.ndarray
        2-D float32 array of shape (nlat, nlon).
    """
    try:
        import xarray as xr
        import numpy as np
    except ImportError:
        raise ImportError("xarray and numpy are required to load elevation data")

    elevation_file = resolve_path(elevation_file)
    if not elevation_file.exists():
        raise FileNotFoundError(f"Elevation file not found: {elevation_file}")

    with xr.open_dataset(str(elevation_file)) as ds:
        # Auto-detect variable
        if var_name and var_name in ds.data_vars:
            da = ds[var_name]
        else:
            candidates = [v for v in ds.data_vars if "elev" in v.lower() or "dem" in v.lower() or "topo" in v.lower()]
            da = ds[candidates[0]] if candidates else ds[list(ds.data_vars)[0]]

        # Drop time if present
        if "time" in da.dims:
            da = da.isel(time=0, drop=True)

        # Regrid if target grid provided
        if target_lat is not None and target_lon is not None:
            lat_candidates = ("lat", "latitude", "y")
            lon_candidates = ("lon", "longitude", "x")
            lat_name = next((n for n in lat_candidates if n in da.dims or n in da.coords), None)
            lon_name = next((n for n in lon_candidates if n in da.dims or n in da.coords), None)
            if lat_name and lon_name:
                try:
                    da = da.interp(
                        {lat_name: target_lat, lon_name: target_lon},
                        method="linear",
                    )
                except Exception:
                    pass  # Keep original resolution if regrid fails

        arr = da.values.astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.ndim == 1:
            raise ValueError("Elevation field is 1-D; expected a 2-D lat/lon grid")
        return arr
