"""Shared utilities for the NARR-to-PRISM downscaling workflow."""

from __future__ import annotations

import os
import re
import hashlib
import warnings
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


_NARR_REGRID_CACHE_VERSION = 1
_NARR_REGRID_CACHE_NAME = "narr_barycentric_regrid_weights.npz"


def _coordinate_fingerprint(*arrays: Any) -> str:
    """Return a stable fingerprint for ordered floating-point coordinates."""
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def narr_regrid_cache_path(case_preprocess_dir: str | Path) -> Path:
    """Return the case-scoped cache path for NARR-to-PRISM geometry."""
    return Path(case_preprocess_dir) / _NARR_REGRID_CACHE_NAME


class NARRBarycentricRegridder:
    """Reusable linear regridder for a curvilinear NARR source grid.

    Delaunay simplex lookup and barycentric weights depend only on the source
    and target coordinates, not on a day's predictor values.  They are built
    once and can be cached per case.  Applying the weights is then a small
    gather/multiply operation, including for a rectangular target crop.

    Missing values are deliberately conservative: a target is finite only if
    it is inside the source convex hull *and* all three values at its source
    simplex vertices are finite.  No nearest-neighbour fallback is performed.
    """

    def __init__(
        self,
        *,
        source_shape: Sequence[int],
        target_shape: Sequence[int],
        vertices: np.ndarray,
        weights: np.ndarray,
        source_fingerprint: str,
        target_fingerprint: str,
        loaded_from_cache: bool = False,
    ) -> None:
        self.source_shape = tuple(int(value) for value in source_shape)
        self.target_shape = tuple(int(value) for value in target_shape)
        if len(self.source_shape) != 2 or len(self.target_shape) != 2:
            raise ValueError("NARR source and target grids must both be two-dimensional")

        self.vertices = np.ascontiguousarray(vertices, dtype=np.int32)
        self.weights = np.ascontiguousarray(weights, dtype=np.float64)
        expected = (int(np.prod(self.target_shape)), 3)
        if self.vertices.shape != expected or self.weights.shape != expected:
            raise ValueError(
                "Invalid barycentric cache arrays: expected "
                f"{expected}, got vertices={self.vertices.shape}, "
                f"weights={self.weights.shape}"
            )
        self.source_fingerprint = str(source_fingerprint)
        self.target_fingerprint = str(target_fingerprint)
        self.loaded_from_cache = bool(loaded_from_cache)
        self._sparse_operator = None

    @property
    def valid_target_count(self) -> int:
        """Number of target cells inside the source coordinate convex hull."""
        return int(np.count_nonzero(self.vertices[:, 0] >= 0))

    @classmethod
    def from_grids(
        cls,
        source_lat: Any,
        source_lon: Any,
        target_lat: Any,
        target_lon: Any,
        *,
        cache_path: Optional[str | Path] = None,
    ) -> "NARRBarycentricRegridder":
        """Build or load geometry for an ordered source and target grid."""
        source_lat_array = np.asarray(source_lat, dtype=np.float64)
        source_lon_array = np.asarray(source_lon, dtype=np.float64)
        target_lat_array = np.asarray(target_lat, dtype=np.float64)
        target_lon_array = np.asarray(target_lon, dtype=np.float64)

        if source_lat_array.ndim != 2 or source_lon_array.ndim != 2:
            raise ValueError("NARR latitude/longitude coordinates must be two-dimensional")
        if source_lat_array.shape != source_lon_array.shape:
            raise ValueError(
                "NARR latitude/longitude shapes differ: "
                f"{source_lat_array.shape} versus {source_lon_array.shape}"
            )
        if target_lat_array.ndim != 1 or target_lon_array.ndim != 1:
            raise ValueError("PRISM target latitude/longitude must be one-dimensional")
        if target_lat_array.size == 0 or target_lon_array.size == 0:
            raise ValueError("PRISM target latitude/longitude must be non-empty")
        if not np.isfinite(target_lat_array).all() or not np.isfinite(target_lon_array).all():
            raise ValueError("PRISM target coordinates contain non-finite values")

        source_shape = source_lat_array.shape
        target_shape = (target_lat_array.size, target_lon_array.size)
        source_fingerprint = _coordinate_fingerprint(
            source_lat_array, source_lon_array
        )
        target_fingerprint = _coordinate_fingerprint(
            target_lat_array, target_lon_array
        )

        resolved_cache = Path(cache_path) if cache_path is not None else None
        if resolved_cache is not None and resolved_cache.is_file():
            try:
                cached = cls._load_cache(
                    resolved_cache,
                    source_shape=source_shape,
                    target_shape=target_shape,
                    source_fingerprint=source_fingerprint,
                    target_fingerprint=target_fingerprint,
                )
            except (KeyError, OSError, ValueError, EOFError) as exc:
                warnings.warn(
                    f"Ignoring invalid NARR regrid cache {resolved_cache}: {exc}",
                    RuntimeWarning,
                )
            else:
                if cached is not None:
                    return cached

        built = cls._build(
            source_lat_array,
            source_lon_array,
            target_lat_array,
            target_lon_array,
            source_fingerprint=source_fingerprint,
            target_fingerprint=target_fingerprint,
        )
        if resolved_cache is not None:
            built._write_cache(resolved_cache)
        return built

    @classmethod
    def _build(
        cls,
        source_lat: np.ndarray,
        source_lon: np.ndarray,
        target_lat: np.ndarray,
        target_lon: np.ndarray,
        *,
        source_fingerprint: str,
        target_fingerprint: str,
    ) -> "NARRBarycentricRegridder":
        try:
            from scipy.spatial import Delaunay
        except ImportError as exc:
            raise ImportError("scipy is required for NARR interpolation") from exc

        source_points_all = np.column_stack(
            [source_lon.ravel(), source_lat.ravel()]
        )
        finite_source = np.isfinite(source_points_all).all(axis=1)
        source_indices = np.flatnonzero(finite_source)
        if source_indices.size < 3:
            raise ValueError("NARR grid has fewer than three finite coordinate points")
        source_points = source_points_all[finite_source]
        if np.unique(source_points, axis=0).shape[0] < 3:
            raise ValueError("NARR grid has fewer than three unique coordinate points")

        triangulation = Delaunay(source_points)
        target_lon_2d, target_lat_2d = np.meshgrid(target_lon, target_lat)
        target_points = np.column_stack(
            [target_lon_2d.ravel(), target_lat_2d.ravel()]
        )
        simplex = triangulation.find_simplex(target_points)
        inside = simplex >= 0

        vertices = np.full((target_points.shape[0], 3), -1, dtype=np.int32)
        weights = np.full((target_points.shape[0], 3), np.nan, dtype=np.float64)
        if inside.any():
            inside_simplex = simplex[inside]
            transform = triangulation.transform[inside_simplex]
            delta = target_points[inside] - transform[:, 2, :]
            first_weights = np.einsum(
                "nij,nj->ni", transform[:, :2, :], delta
            )
            inside_weights = np.column_stack(
                [first_weights, 1.0 - first_weights.sum(axis=1)]
            )
            inside_vertices = source_indices[
                triangulation.simplices[inside_simplex]
            ]
            vertices[inside] = inside_vertices.astype(np.int32, copy=False)
            weights[inside] = inside_weights

        return cls(
            source_shape=source_lat.shape,
            target_shape=(target_lat.size, target_lon.size),
            vertices=vertices,
            weights=weights,
            source_fingerprint=source_fingerprint,
            target_fingerprint=target_fingerprint,
        )

    @classmethod
    def _load_cache(
        cls,
        path: Path,
        *,
        source_shape: Sequence[int],
        target_shape: Sequence[int],
        source_fingerprint: str,
        target_fingerprint: str,
    ) -> Optional["NARRBarycentricRegridder"]:
        with np.load(path, allow_pickle=False) as data:
            version = int(np.asarray(data["format_version"]).item())
            cached_source_shape = tuple(
                int(value) for value in np.asarray(data["source_shape"]).tolist()
            )
            cached_target_shape = tuple(
                int(value) for value in np.asarray(data["target_shape"]).tolist()
            )
            cached_source_fingerprint = str(
                np.asarray(data["source_fingerprint"]).item()
            )
            cached_target_fingerprint = str(
                np.asarray(data["target_fingerprint"]).item()
            )
            if (
                version != _NARR_REGRID_CACHE_VERSION
                or cached_source_shape != tuple(source_shape)
                or cached_target_shape != tuple(target_shape)
                or cached_source_fingerprint != source_fingerprint
                or cached_target_fingerprint != target_fingerprint
            ):
                return None
            return cls(
                source_shape=cached_source_shape,
                target_shape=cached_target_shape,
                vertices=np.asarray(data["vertices"], dtype=np.int32),
                weights=np.asarray(data["weights"], dtype=np.float64),
                source_fingerprint=cached_source_fingerprint,
                target_fingerprint=cached_target_fingerprint,
                loaded_from_cache=True,
            )

    def _write_cache(self, path: Path) -> None:
        """Atomically persist this geometry so date shards can safely share it."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            with open(temporary, "wb") as handle:
                np.savez_compressed(
                    handle,
                    format_version=np.asarray(_NARR_REGRID_CACHE_VERSION, dtype=np.int64),
                    source_shape=np.asarray(self.source_shape, dtype=np.int64),
                    target_shape=np.asarray(self.target_shape, dtype=np.int64),
                    source_fingerprint=np.asarray(self.source_fingerprint),
                    target_fingerprint=np.asarray(self.target_fingerprint),
                    vertices=self.vertices,
                    weights=self.weights,
                )
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def assert_grids_match(
        self,
        source_lat: Any,
        source_lon: Any,
        target_lat: Any,
        target_lon: Any,
    ) -> None:
        """Reject applying cached weights to differently ordered coordinates."""
        observed_source = _coordinate_fingerprint(source_lat, source_lon)
        observed_target = _coordinate_fingerprint(target_lat, target_lon)
        if observed_source != self.source_fingerprint:
            raise ValueError(
                "NARR source grid differs from the grid used to build regrid weights"
            )
        if observed_target != self.target_fingerprint:
            raise ValueError(
                "PRISM target grid differs from the grid used to build regrid weights"
            )

    def _operator(self) -> Any:
        """Return a lazily built CSR operator with three entries per simplex."""
        if self._sparse_operator is None:
            try:
                from scipy.sparse import csr_matrix
            except ImportError as exc:
                raise ImportError("scipy is required for NARR interpolation") from exc

            valid = self.vertices[:, 0] >= 0
            counts = np.where(valid, 3, 0).astype(np.int64, copy=False)
            indptr = np.empty(self.vertices.shape[0] + 1, dtype=np.int64)
            indptr[0] = 0
            np.cumsum(counts, out=indptr[1:])
            self._sparse_operator = csr_matrix(
                (
                    self.weights[valid].ravel(),
                    self.vertices[valid].ravel(),
                    indptr,
                ),
                shape=(self.vertices.shape[0], int(np.prod(self.source_shape))),
            )
        return self._sparse_operator

    def apply(
        self,
        values: Any,
        *,
        target_slices: Optional[Tuple[slice, slice]] = None,
        target_chunk_size: Optional[int] = None,
    ) -> np.ndarray:
        """Apply cached weights to one or more ``[..., y, x]`` source fields."""
        source = np.asarray(values)
        if source.ndim < 2 or tuple(source.shape[-2:]) != self.source_shape:
            raise ValueError(
                f"NARR values end in {source.shape[-2:]}, expected {self.source_shape}"
            )
        if target_chunk_size is not None and target_chunk_size < 1:
            raise ValueError("target_chunk_size must be positive")

        if target_slices is None:
            target_indices = np.arange(self.vertices.shape[0], dtype=np.int64)
            output_shape = self.target_shape
        else:
            if len(target_slices) != 2:
                raise ValueError("target_slices must contain latitude and longitude slices")
            rows = np.arange(self.target_shape[0], dtype=np.int64)[target_slices[0]]
            columns = np.arange(self.target_shape[1], dtype=np.int64)[target_slices[1]]
            target_indices = (
                rows[:, np.newaxis] * self.target_shape[1] + columns[np.newaxis, :]
            ).ravel()
            output_shape = (rows.size, columns.size)

        flat_source = source.reshape((-1, int(np.prod(self.source_shape))))
        operator = self._operator()
        if target_slices is None and target_chunk_size is None:
            flat_output = np.asarray(operator.dot(flat_source.T)).T
            flat_output[:, self.vertices[:, 0] < 0] = np.nan
            result = flat_output.reshape(source.shape[:-2] + self.target_shape)
            result[~np.isfinite(result)] = np.nan
            return result.astype(np.float32, copy=False)

        flat_output = np.full(
            (flat_source.shape[0], target_indices.size), np.nan, dtype=np.float64
        )
        chunk_size = (
            max(1, target_indices.size)
            if target_chunk_size is None
            else target_chunk_size
        )
        for start in range(0, target_indices.size, chunk_size):
            stop = min(start + chunk_size, target_indices.size)
            selected = target_indices[start:stop]
            valid_geometry = self.vertices[selected, 0] >= 0
            if not valid_geometry.any():
                continue
            interpolated = np.asarray(operator[selected].dot(flat_source.T)).T
            interpolated[:, ~valid_geometry] = np.nan
            flat_output[:, start:stop] = interpolated

        result = flat_output.reshape(source.shape[:-2] + tuple(output_shape))
        result[~np.isfinite(result)] = np.nan
        return result.astype(np.float32, copy=False)


def load_narr_grid(
    file_map: Mapping[str, Path] | Path,
    var: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load the two-dimensional NARR latitude and longitude coordinates."""
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError("xarray is required for NARR interpolation") from exc

    path = Path(file_map[var]) if isinstance(file_map, Mapping) else Path(file_map)
    with xr.open_dataset(str(path)) as dataset:
        lat = np.asarray(dataset["lat"].values, dtype=np.float64)
        lon = np.asarray(dataset["lon"].values, dtype=np.float64)
    return lat, lon


def build_narr_regridder(
    file_map: Mapping[str, Path] | Path,
    var: str,
    target_lat: Any,
    target_lon: Any,
    *,
    cache_path: Optional[str | Path] = None,
) -> NARRBarycentricRegridder:
    """Build/load a reusable regridder using coordinates from one NARR file."""
    source_lat, source_lon = load_narr_grid(file_map, var)
    return NARRBarycentricRegridder.from_grids(
        source_lat,
        source_lon,
        target_lat,
        target_lon,
        cache_path=cache_path,
    )


def interpolate_narr_to_grid(
    file_map: Mapping[str, Path] | Path,
    var: str,
    level: float,
    sample_date: date,
    target_lat: Any,
    target_lon: Any,
    *,
    regridder: Optional[NARRBarycentricRegridder] = None,
    target_slices: Optional[Tuple[slice, slice]] = None,
) -> "np.ndarray":
    """Load one NARR channel and linearly interpolate it to a PRISM grid.

    Continuous predictors are only defined where linear interpolation is
    supported by finite source points.  In particular, values outside the
    finite-point convex hull remain NaN; filling those cells by nearest
    neighbour would create artificial constant blocks at the domain edge.
    """
    try:
        import numpy as np
        import xarray as xr
    except ImportError as exc:
        raise ImportError("numpy and xarray are required for NARR interpolation") from exc

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

    if regridder is None:
        regridder = NARRBarycentricRegridder.from_grids(
            lat, lon, target_lat, target_lon
        )
    else:
        regridder.assert_grids_match(lat, lon, target_lat, target_lon)
    return regridder.apply(values, target_slices=target_slices)


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

    if (target_lat is None) != (target_lon is None):
        raise ValueError("target_lat and target_lon must be provided together")

    with xr.open_dataset(str(elevation_file)) as ds:
        # Auto-detect variable
        if var_name is not None:
            if var_name not in ds.data_vars:
                raise ValueError(
                    f"Elevation file {elevation_file} does not contain variable "
                    f"'{var_name}'"
                )
            da = ds[var_name]
        else:
            candidates = [v for v in ds.data_vars if "elev" in v.lower() or "dem" in v.lower() or "topo" in v.lower()]
            da = ds[candidates[0]] if candidates else ds[list(ds.data_vars)[0]]

        # Drop time if present
        if "time" in da.dims:
            da = da.isel(time=0, drop=True)

        # Regrid if target grid provided
        if target_lat is not None and target_lon is not None:
            target_lat_array = np.asarray(target_lat, dtype=np.float64)
            target_lon_array = np.asarray(target_lon, dtype=np.float64)
            if target_lat_array.ndim != 1 or target_lon_array.ndim != 1:
                raise ValueError("target_lat and target_lon must be one-dimensional")
            if target_lat_array.size == 0 or target_lon_array.size == 0:
                raise ValueError("target_lat and target_lon must be non-empty")

            lat_candidates = ("lat", "latitude", "y")
            lon_candidates = ("lon", "longitude", "x")
            lat_name = next((n for n in lat_candidates if n in da.dims or n in da.coords), None)
            lon_name = next((n for n in lon_candidates if n in da.dims or n in da.coords), None)
            if lat_name is None or lon_name is None:
                raise ValueError(
                    f"Cannot align elevation variable '{da.name}' from "
                    f"{elevation_file}: no latitude/longitude coordinates"
                )
            if lat_name not in da.coords or lon_name not in da.coords:
                raise ValueError(
                    f"Cannot align elevation variable '{da.name}' from "
                    f"{elevation_file}: latitude/longitude coordinates are required"
                )
            if da.coords[lat_name].ndim != 1 or da.coords[lon_name].ndim != 1:
                raise ValueError(
                    f"Cannot align elevation variable '{da.name}' from "
                    f"{elevation_file}: latitude/longitude coordinates must be "
                    "one-dimensional"
                )

            try:
                da = da.interp(
                    {lat_name: target_lat_array, lon_name: target_lon_array},
                    method="linear",
                )
                da = da.transpose(lat_name, lon_name)
            except Exception as exc:
                raise ValueError(
                    f"Failed to interpolate elevation variable '{da.name}' from "
                    f"{elevation_file} to target grid "
                    f"({target_lat_array.size}, {target_lon_array.size})"
                ) from exc

        arr = da.values.astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.ndim != 2:
            raise ValueError(
                f"Elevation field has shape {arr.shape}; expected a 2-D lat/lon grid"
            )
        if target_lat is not None and target_lon is not None:
            expected_shape = (target_lat_array.size, target_lon_array.size)
            if arr.shape != expected_shape:
                raise ValueError(
                    f"Interpolated elevation shape {arr.shape} does not match "
                    f"target grid {expected_shape}"
                )
        return arr
