"""Repair invalid predictor values with nearest-valid replacement."""

from __future__ import annotations

from collections import deque
import warnings
from typing import Iterable

import numpy as np
import xarray as xr

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - optional dependency
    cKDTree = None


def _flatten_fill_values(raw) -> list[float]:
    if raw is None:
        return []
    arr = np.asarray(raw).reshape(-1)
    out: list[float] = []
    for value in arr:
        try:
            out.append(float(value))
        except Exception:
            continue
    return out


def _collect_fill_values(da: xr.DataArray) -> list[float]:
    values: list[float] = []
    for source in (da.attrs, da.encoding):
        if not isinstance(source, dict):
            continue
        for key in ("_FillValue", "missing_value", "fill_value"):
            values.extend(_flatten_fill_values(source.get(key)))
    unique: list[float] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def _invalid_mask(values: np.ndarray, fill_values: Iterable[float]) -> np.ndarray:
    mask = ~np.isfinite(values)
    for fill_value in fill_values:
        if np.isnan(fill_value):
            continue
        mask |= values == fill_value
    return mask


def _infer_coord_name(
    da: xr.DataArray, candidates: tuple[str, ...], fallback_key: str
) -> str | None:
    for name in candidates:
        if name in da.dims or name in da.coords:
            return name
    for dim in da.dims:
        if fallback_key in dim.lower():
            return dim
    for coord in da.coords:
        if fallback_key in coord.lower():
            return coord
    return None


def _coords_for_distance(
    da: xr.DataArray, lat_name: str, lon_name: str
) -> tuple[np.ndarray, np.ndarray]:
    lat_data = da[lat_name].values if lat_name in da.coords else None
    lon_data = da[lon_name].values if lon_name in da.coords else None
    if lat_data is None or lon_data is None:
        raise KeyError("Latitude/longitude coordinates are missing")

    if lat_data.ndim == 1 and lon_data.ndim == 1:
        lat2d, lon2d = np.meshgrid(lat_data, lon_data, indexing="ij")
        return lat2d, lon2d

    if lat_data.ndim == 2 and lon_data.ndim == 2:
        if lat_data.shape != lon_data.shape:
            raise ValueError("2D latitude/longitude shapes do not match")
        return lat_data, lon_data

    raise ValueError("Unsupported latitude/longitude dimensionality")


def _normalize_longitude(lon: np.ndarray) -> np.ndarray:
    lon = np.asarray(lon, dtype=np.float64)
    return ((lon + 180.0) % 360.0) - 180.0


def _fill_with_bfs(values: np.ndarray, invalid: np.ndarray) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError("BFS fallback expects 2D inputs")

    filled = values.copy()
    visited = ~invalid
    queue: deque[tuple[int, int]] = deque()

    valid_positions = np.argwhere(visited)
    for row, col in valid_positions:
        queue.append((int(row), int(col)))

    while queue:
        row, col = queue.popleft()
        for drow, dcol in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nrow = row + drow
            ncol = col + dcol
            if nrow < 0 or ncol < 0 or nrow >= values.shape[0] or ncol >= values.shape[1]:
                continue
            if visited[nrow, ncol]:
                continue
            filled[nrow, ncol] = filled[row, col]
            visited[nrow, ncol] = True
            queue.append((nrow, ncol))

    return filled


def _time_level_indices(other_dims: list[str], idx: tuple[int, ...]) -> tuple[int | None, int | None]:
    time_idx = None
    level_idx = None
    for i, dim in enumerate(other_dims):
        dim_lower = dim.lower()
        if time_idx is None and "time" in dim_lower:
            time_idx = idx[i]
        if level_idx is None and any(key in dim_lower for key in ("lev", "level", "plev", "isobaric")):
            level_idx = idx[i]
    return time_idx, level_idx


def summarize_invalid_counts_xr(
    ds: xr.Dataset,
    *,
    var_names: Iterable[str] | None = None,
) -> dict[str, int]:
    """Count invalid values (NaN/inf/fill) per variable."""
    names = list(var_names) if var_names is not None else list(ds.data_vars)
    counts: dict[str, int] = {}
    for name in names:
        if name not in ds.data_vars:
            continue
        da = ds[name]
        if not np.issubdtype(da.dtype, np.number):
            continue
        values = np.asarray(da.values)
        fill_values = _collect_fill_values(da)
        counts[name] = int(_invalid_mask(values, fill_values).sum())
    return counts


def repair_invalid_by_nearest_xr(
    ds: xr.Dataset,
    *,
    var_names: Iterable[str] | None = None,
    lat_name_candidates: tuple[str, ...] = ("lat", "latitude", "y"),
    lon_name_candidates: tuple[str, ...] = ("lon", "longitude", "x"),
    use_scipy: bool = True,
) -> xr.Dataset:
    """Replace invalid predictor values with nearest valid lat/lon neighbors."""
    repaired = ds.copy(deep=True)
    names = list(var_names) if var_names is not None else list(repaired.data_vars)

    for name in names:
        if name not in repaired.data_vars:
            continue

        da = repaired[name]
        if not np.issubdtype(da.dtype, np.number):
            continue

        lat_name = _infer_coord_name(da, lat_name_candidates, "lat")
        lon_name = _infer_coord_name(da, lon_name_candidates, "lon")
        has_coords = lat_name is not None and lon_name is not None

        fill_values = _collect_fill_values(da)
        dims = list(da.dims)

        if has_coords:
            if lat_name not in dims or lon_name not in dims:
                has_coords = False

        if has_coords:
            lat2d, lon2d = _coords_for_distance(da, lat_name, lon_name)
            lon2d = _normalize_longitude(lon2d)
            coord_points = np.column_stack([lat2d.reshape(-1), lon2d.reshape(-1)])
            lat_axis = dims.index(lat_name)
            lon_axis = dims.index(lon_name)
            other_dims = [dim for dim in dims if dim not in (lat_name, lon_name)]
            transpose_order = other_dims + [lat_name, lon_name]
            values = np.asarray(da.transpose(*transpose_order).values).copy()
        else:
            warnings.warn(
                f"{name}: lat/lon coords not found; using index-space nearest fill fallback.",
                RuntimeWarning,
            )
            other_dims = dims[:-2]
            transpose_order = dims
            values = np.asarray(da.values).copy()
            lat_axis = len(dims) - 2
            lon_axis = len(dims) - 1

        trailing_shape = values.shape[-2:]
        if len(trailing_shape) != 2:
            raise ValueError(f"{name}: expected 2 spatial dimensions, got shape {values.shape}")

        index_iter = np.ndindex(values.shape[:-2]) if values.ndim > 2 else [()]
        source_path = str(getattr(ds, "encoding", {}).get("source", "unknown"))
        for idx in index_iter:
            slice_values = values[idx] if idx != () else values
            invalid = _invalid_mask(slice_values, fill_values)
            if not invalid.any():
                continue
            if invalid.all():
                time_idx, level_idx = _time_level_indices(other_dims, idx if idx != () else ())
                raise ValueError(
                    f"All values invalid for variable '{name}' at time index {time_idx}, "
                    f"level index {level_idx}, source '{source_path}'."
                )

            if has_coords and use_scipy and cKDTree is not None:
                flat_values = slice_values.reshape(-1)
                flat_invalid = invalid.reshape(-1)
                valid_points = coord_points[~flat_invalid]
                invalid_points = coord_points[flat_invalid]
                tree = cKDTree(valid_points)
                _, nearest_idx = tree.query(invalid_points, k=1)
                valid_values = flat_values[~flat_invalid]
                flat_values[flat_invalid] = valid_values[nearest_idx]
                filled_slice = flat_values.reshape(slice_values.shape)
            else:
                if has_coords and use_scipy and cKDTree is None:
                    warnings.warn(
                        "SciPy not available; falling back to iterative 4-neighbor propagation.",
                        RuntimeWarning,
                    )
                filled_slice = _fill_with_bfs(slice_values, invalid)

            if idx == ():
                values[...] = filled_slice
            else:
                values[idx] = filled_slice

        repaired_da = xr.DataArray(
            values,
            dims=transpose_order,
            coords={key: da.coords[key] for key in da.coords if key in transpose_order or key not in dims},
            attrs=da.attrs,
            name=da.name,
        ).transpose(*dims)
        repaired[name] = repaired_da.astype(da.dtype, copy=False)

    return repaired

