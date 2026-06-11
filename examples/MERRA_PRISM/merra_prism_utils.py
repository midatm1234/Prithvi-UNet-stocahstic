"""Shared utilities for the MERRA2-to-PRISM downscaling workflow."""

from __future__ import annotations

import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    """Resolve an archive root and append case_name unless it is already present."""
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
# MERRA2 file discovery
# ---------------------------------------------------------------------------

_MERRA2_DATE_RE = re.compile(r"(\d{8})")


def discover_merra2_files(
    predictor_dir: str | Path,
    start: date,
    end: date,
) -> List[Tuple[date, Path]]:
    """Find MERRA2 daily files in *predictor_dir* within [start, end].

    The function expects filenames to contain a ``YYYYMMDD`` date token.
    Returns a sorted list of (date, path) tuples.
    """
    predictor_dir = Path(predictor_dir)
    if not predictor_dir.is_dir():
        raise FileNotFoundError(f"MERRA2 predictor directory not found: {predictor_dir}")

    results: List[Tuple[date, Path]] = []
    for entry in sorted(predictor_dir.iterdir()):
        if not entry.is_file():
            continue
        match = _MERRA2_DATE_RE.search(entry.name)
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
