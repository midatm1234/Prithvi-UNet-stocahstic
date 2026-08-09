"""PyTorch dataset utilities for CORDEX downscaling experiments."""

from __future__ import annotations

import contextlib
import io
import os
import warnings
from bisect import bisect_right
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import Dataset

try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        import xesmf as xe
except Exception as exc:  # pragma: no cover - depends on optional dep
    xe = None
    _XESMF_IMPORT_ERROR = exc
else:  # pragma: no cover - optional import
    _XESMF_IMPORT_ERROR = None


PathLike = Union[str, os.PathLike[str]]


class CordexDownscaleDataset(Dataset):
    """Pair coarse predictors with high-resolution CORDEX targets.

    Parameters
    ----------
    predictor_files : Sequence[PathLike]
        NetCDF files that contain the coarse-resolution predictors.
    target_files : Sequence[PathLike]
        NetCDF files that contain the matching high-resolution targets.
    orography_file : PathLike | None
        NetCDF file containing a static orography field. Required when
        ``use_static`` is True.
    predictor_variables : Sequence[str], optional
        Predictor variable names. Defaults to the canonical u/v/q/t/z at
        850/700/500 hPa.
    target_variables : Sequence[str], optional
        Target variable names. Defaults to all time-dependent variables
        found in the first target file.
    orography_variable : str, optional
        Variable name for the static height field. Defaults to "orog" but
        falls back to the first data variable if not provided.
    time_dim : str, optional
        Name of the time dimension. If omitted it is inferred from the
        first predictor file.
    regrid_method : str, optional
        xESMF regridding method. Defaults to "bilinear".
    dtype : torch.dtype, optional
        Tensor dtype for the returned samples. Defaults to float32.
    """

    DEFAULT_PREDICTORS: Tuple[str, ...] = (
        "u_850",
        "u_700",
        "u_500",
        "v_850",
        "v_700",
        "v_500",
        "q_850",
        "q_700",
        "q_500",
        "t_850",
        "t_700",
        "t_500",
        "z_850",
        "z_700",
        "z_500",
    )

    LAT_CANDIDATES = ("lat", "latitude", "rlat", "y")
    LON_CANDIDATES = ("lon", "longitude", "rlon", "x")

    def __init__(
        self,
        predictor_files: Sequence[PathLike],
        target_files: Sequence[PathLike],
        orography_file: PathLike | None,
        predictor_variables: Optional[Sequence[str]] = None,
        target_variables: Optional[Sequence[str]] = None,
        orography_variable: Optional[str] = "orog",
        time_dim: Optional[str] = None,
        regrid_method: str = "bilinear",
        dtype: torch.dtype = torch.float32,
        crop_size: Optional[Tuple[int, int]] = None,
        random_crop: bool = True,
        random_crop_offset: Optional[Tuple[int, int]] = None,
        seed: Optional[int] = None,
        use_static: bool = True,
        allow_time_mismatch: bool = False,
    ) -> None:
        predictor_paths = [os.fspath(p) for p in predictor_files]
        target_paths = [os.fspath(p) for p in target_files]

        if not predictor_paths:
            raise ValueError("At least one predictor file is required")
        if len(predictor_paths) != len(target_paths):
            raise ValueError("Predictor and target file lists must match in length")

        self.predictor_paths = predictor_paths
        self.target_paths = target_paths
        self.use_static = bool(use_static)
        if self.use_static:
            if orography_file is None:
                raise ValueError("orography_file is required when use_static=True")
            self.orography_path = os.fspath(orography_file)
        else:
            self.orography_path = None
        self.predictor_vars = list(predictor_variables or self.DEFAULT_PREDICTORS)
        self.target_vars = list(target_variables) if target_variables else None
        self.orography_var = orography_variable
        self.dtype = dtype
        self.allow_time_mismatch = bool(allow_time_mismatch)

        (
            self.time_dim,
            self.coarse_lat_name,
            self.coarse_lon_name,
            self.coarse_shape,
            grid_in,
        ) = self._inspect_predictor_template(self.predictor_paths[0], time_dim)

        (
            self.fine_lat_name,
            self.fine_lon_name,
            self.fine_shape,
            self.output_spatial_dims,
            self.target_vars,
            grid_out,
        ) = self._inspect_target_template(self.target_paths[0])

        self.regridder = self._build_regridder(grid_in, grid_out, regrid_method)
        self._orography_tensor = self._prepare_orography() if self.use_static else None

        if crop_size is None:
            self.crop_size = self.fine_shape
        else:
            lat_size = min(int(crop_size[0]), self.fine_shape[0])
            lon_size = min(int(crop_size[1]), self.fine_shape[1])
            self.crop_size = (lat_size, lon_size)

        self.random_crop = random_crop and self.crop_size != self.fine_shape
        if random_crop_offset is None:
            self.random_crop_offset = (0, 0)
        else:
            self.random_crop_offset = (
                max(0, int(random_crop_offset[0])),
                max(0, int(random_crop_offset[1])),
            )
        self._rng = np.random.default_rng(seed)

        # In inference mode, predictor and target calendars can legitimately
        # have different lengths (for example, 7300 no-leap predictor days and
        # 7305 Gregorian target days).  Store an exact date-based target index
        # for each predictor file instead of assuming that equal integer
        # positions refer to the same day.
        self._target_time_indices: List[Optional[np.ndarray]] = []
        self._time_lengths, self._target_time_lengths = self._compute_time_lengths()
        self._cumulative_sizes = self._build_cumulative_sizes(self._time_lengths)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self._cumulative_sizes[-1] if self._cumulative_sizes else 0

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        file_idx, time_idx = self._locate_index(index)

        lat_slice, lon_slice = self._select_crop()

        x = self._load_predictors(self.predictor_paths[file_idx], time_idx, lat_slice, lon_slice)
        target_time_indices = getattr(self, "_target_time_indices", [])
        target_time_idx = time_idx
        if target_time_indices and target_time_indices[file_idx] is not None:
            target_time_idx = int(target_time_indices[file_idx][time_idx])
        y = self._load_targets(
            self.target_paths[file_idx],
            target_time_idx,
            lat_slice,
            lon_slice,
        )

        if self.random_crop_offset != (0, 0):
            x, y = self._apply_random_spatial_offset(x, y)

        return {"x": x, "y": y}

    # ------------------------------------------------------------------
    def _inspect_predictor_template(
        self, path: str, time_dim: Optional[str]
    ) -> Tuple[Optional[str], str, str, Tuple[int, int], xr.Dataset]:
        with xr.open_dataset(path) as ds:
            lat_name = self._infer_coord_name(ds, self.LAT_CANDIDATES)
            lon_name = self._infer_coord_name(ds, self.LON_CANDIDATES)
            inferred_time = time_dim or self._infer_time_dim(ds)

            missing = [var for var in self.predictor_vars if var not in ds.data_vars]
            if missing:
                raise ValueError(
                    f"Predictor file {path} does not contain variables: {missing}"
                )

            lat_da = ds[lat_name].load()
            lon_da = ds[lon_name].load()
            grid_in = self._build_grid(lat_da, lon_da, lat_name, lon_name)
            spatial_shape = self._compute_shape(lat_da, lon_da)

        return inferred_time, lat_name, lon_name, spatial_shape, grid_in

    def _inspect_target_template(
        self, path: str
    ) -> Tuple[str, str, Tuple[int, int], Tuple[str, str], List[str], xr.Dataset]:
        with xr.open_dataset(path) as ds:
            lat_name = self._infer_coord_name(ds, self.LAT_CANDIDATES)
            lon_name = self._infer_coord_name(ds, self.LON_CANDIDATES)
            lat_da = ds[lat_name].load()
            lon_da = ds[lon_name].load()

            if self.target_vars is None:
                candidates = [
                    name
                    for name, var in ds.data_vars.items()
                    if self.time_dim and self.time_dim in var.dims
                ]
                if not candidates:
                    raise ValueError(
                        f"Unable to infer target variables from {path}; please specify them"
                    )
                self.target_vars = candidates
            else:
                missing = [var for var in self.target_vars if var not in ds.data_vars]
                if missing:
                    raise ValueError(
                        f"Target file {path} does not contain variables: {missing}"
                    )

            target_example = ds[self.target_vars[0]]
            spatial_dims = [dim for dim in target_example.dims if dim != self.time_dim]
            if len(spatial_dims) != 2:
                raise ValueError(
                    f"Target variable {self.target_vars[0]} must have two spatial dims"
                )

            grid_out = self._build_grid(lat_da, lon_da, lat_name, lon_name)
            spatial_shape = self._compute_shape(lat_da, lon_da)

        return lat_name, lon_name, spatial_shape, tuple(spatial_dims), self.target_vars, grid_out

    def _compute_time_lengths(self) -> Tuple[List[int], List[int]]:
        lengths: List[int] = []
        target_lengths: List[int] = []
        for predictor_path, target_path in zip(self.predictor_paths, self.target_paths):
            predictor_len = self._read_time_length(predictor_path)
            target_len = self._read_time_length(target_path)
            target_lengths.append(target_len)

            if self.allow_time_mismatch:
                self._target_time_indices.append(
                    self._build_exact_target_time_index(predictor_path, target_path)
                )
            else:
                self._target_time_indices.append(None)
                if predictor_len != target_len:
                    raise ValueError(
                        f"Time dimension mismatch between {predictor_path} and {target_path}"
                    )

            lengths.append(predictor_len)

        return lengths, target_lengths

    def _build_exact_target_time_index(
        self,
        predictor_path: str,
        target_path: str,
    ) -> np.ndarray:
        """Map every predictor timestamp to one unique target timestamp.

        Calendar classes are intentionally ignored when comparing labels: a
        ``cftime.DatetimeNoLeap(1984, 3, 1)`` must match Gregorian
        ``1984-03-01``.  Calendar-only dates such as February 29 may therefore
        exist in the target without a predictor counterpart, but every
        predictor date must be present exactly once in both files.
        """

        if not self.time_dim:
            raise ValueError(
                "Exact inference time alignment requires a predictor time dimension"
            )

        predictor_values = self._read_time_coordinate(predictor_path, role="Predictor")
        target_values = self._read_time_coordinate(target_path, role="Target")
        return _exact_time_index(
            predictor_values,
            target_values,
            predictor_path=predictor_path,
            target_path=target_path,
        )

    def _read_time_coordinate(self, path: str, *, role: str) -> np.ndarray:
        with xr.open_dataset(path) as ds:
            if self.time_dim not in ds.dims:
                raise ValueError(
                    f"{role} file {path} is missing time dimension {self.time_dim!r}"
                )
            if self.time_dim not in ds.coords:
                raise ValueError(
                    f"{role} file {path} is missing time coordinate {self.time_dim!r}"
                )
            values = np.asarray(ds[self.time_dim].values)

        if values.ndim != 1:
            raise ValueError(
                f"{role} time coordinate in {path} must be one-dimensional; "
                f"got shape {values.shape}"
            )
        if values.size == 0:
            raise ValueError(f"{role} file {path} has an empty time coordinate")
        return values

    def _build_cumulative_sizes(self, lengths: Sequence[int]) -> List[int]:
        cumulative: List[int] = []
        total = 0
        for value in lengths:
            total += value
            cumulative.append(total)
        return cumulative

    def _locate_index(self, index: int) -> Tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")

        file_idx = bisect_right(self._cumulative_sizes, index)
        prev_total = self._cumulative_sizes[file_idx - 1] if file_idx > 0 else 0
        time_idx = index - prev_total
        return file_idx, time_idx

    # ------------------------------------------------------------------
    def _load_predictors(
        self, path: str, time_index: int, lat_slice: slice, lon_slice: slice
    ) -> torch.Tensor:
        tensors: List[np.ndarray] = []
        with xr.open_dataset(path) as ds:
            for var in self.predictor_vars:
                da = ds[var]
                if self.time_dim and self.time_dim in da.dims:
                    da = da.isel({self.time_dim: time_index}, drop=True)
                da = self._rename_lat_lon(da, self.coarse_lat_name, self.coarse_lon_name)
                regridded = self.regridder(da)
                arrays = np.nan_to_num(
                    self._to_numpy(regridded), nan=0.0, posinf=0.0, neginf=0.0
                )
                tensors.append(arrays)

        stacked = np.stack(tensors, axis=0)[..., lat_slice, lon_slice]
        stacked_tensor = torch.from_numpy(stacked).to(self.dtype)

        if not self.use_static:
            return stacked_tensor

        static = self._get_orography_crop(lat_slice, lon_slice)
        return torch.cat([stacked_tensor, static], dim=0)

    def _load_targets(
        self,
        path: str,
        time_index: int,
        lat_slice: slice,
        lon_slice: slice,
    ) -> torch.Tensor:
        tensors: List[np.ndarray] = []
        with xr.open_dataset(path) as ds:
            for var in self.target_vars:
                da = ds[var]
                if self.time_dim and self.time_dim in da.dims:
                    da = da.isel({self.time_dim: time_index}, drop=True)
                arrays = self._to_numpy(da)
                invalid = ~np.isfinite(arrays)
                if bool(invalid.any()):
                    # A zero-filled target is indistinguishable from a genuine
                    # dry precipitation pixel and would therefore corrupt both
                    # the deterministic objective and the signed residual seen
                    # by diffusion.  There is no validity mask in the current
                    # batch/loss contract, so fail before training instead of
                    # silently replacing missing targets with physical zeros.
                    raise ValueError(
                        f"Target variable {var!r} in {path} contains "
                        f"{int(invalid.sum())} non-finite value(s) at time index "
                        f"{time_index}; masked targets are not supported."
                    )
                tensors.append(arrays)

        stacked = np.stack(tensors, axis=0)[..., lat_slice, lon_slice]
        return torch.from_numpy(stacked).to(self.dtype)

    # ------------------------------------------------------------------
    def _prepare_orography(self) -> torch.Tensor:
        if self.orography_path is None:
            raise ValueError("Static orography path is not set.")
        with xr.open_dataset(self.orography_path) as ds:
            if self.orography_var and self.orography_var in ds.data_vars:
                da = ds[self.orography_var]
            else:
                first_var = next(iter(ds.data_vars))
                da = ds[first_var]

            if self.time_dim and self.time_dim in da.dims:
                da = da.isel({self.time_dim: 0}, drop=True)

            data = da
            needs_regrid = data.shape[-2:] == self.coarse_shape

            if needs_regrid:
                data = self._rename_lat_lon(data, self.coarse_lat_name, self.coarse_lon_name)
                data = self.regridder(data)
            elif data.shape[-2:] != self.fine_shape:
                raise ValueError(
                    "Static orography grid shape does not match coarse or target grid"
                )

            arr = np.nan_to_num(
                self._to_numpy(data), nan=0.0, posinf=0.0, neginf=0.0
            )
            if arr.ndim > 2:
                arr = np.squeeze(arr)

        return torch.from_numpy(arr).unsqueeze(0).to(self.dtype)

    def _get_orography_crop(self, lat_slice: slice, lon_slice: slice) -> torch.Tensor:
        if self._orography_tensor is None:
            raise ValueError("Static orography tensor is not initialized.")
        return self._orography_tensor[..., lat_slice, lon_slice]

    def _select_crop(self) -> Tuple[slice, slice]:
        if not self.random_crop:
            if self.crop_size == self.fine_shape:
                return slice(None), slice(None)
            lat_start = (self.fine_shape[0] - self.crop_size[0]) // 2
            lon_start = (self.fine_shape[1] - self.crop_size[1]) // 2
        else:
            max_lat = self.fine_shape[0] - self.crop_size[0]
            max_lon = self.fine_shape[1] - self.crop_size[1]
            lat_start = 0 if max_lat <= 0 else int(self._rng.integers(0, max_lat + 1))
            lon_start = 0 if max_lon <= 0 else int(self._rng.integers(0, max_lon + 1))

        lat_slice = slice(lat_start, lat_start + self.crop_size[0])
        lon_slice = slice(lon_start, lon_start + self.crop_size[1])
        return lat_slice, lon_slice

    def _apply_random_spatial_offset(
        self,
        predictors: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        max_dy, max_dx = self.random_crop_offset
        if max_dy <= 0 and max_dx <= 0:
            return predictors, targets

        dy = 0 if max_dy <= 0 else int(self._rng.integers(-max_dy, max_dy + 1))
        dx = 0 if max_dx <= 0 else int(self._rng.integers(-max_dx, max_dx + 1))
        if dy == 0 and dx == 0:
            return predictors, targets

        def _shift(tensor: torch.Tensor) -> torch.Tensor:
            _, h, w = tensor.shape
            pad_y = abs(dy)
            pad_x = abs(dx)
            if pad_y == 0 and pad_x == 0:
                return tensor

            mode = "reflect"
            if h <= 1 or w <= 1:
                mode = "replicate"

            padded = F.pad(
                tensor.unsqueeze(0),
                (pad_x, pad_x, pad_y, pad_y),
                mode=mode,
            )
            y0 = pad_y - dy
            x0 = pad_x - dx
            return padded[..., y0 : y0 + h, x0 : x0 + w].squeeze(0)

        return _shift(predictors), _shift(targets)

    def _build_regridder(
        self, grid_in: xr.Dataset, grid_out: xr.Dataset, method: str
    ):
        if xe is not None:
            try:
                return xe.Regridder(grid_in, grid_out, method=method, periodic=False)
            except Exception as exc:
                warnings.warn(
                    "xESMF regridding failed to initialize; falling back to "
                    f"xarray interpolation. Error: {exc}",
                    RuntimeWarning,
                )
        elif _XESMF_IMPORT_ERROR is not None:
            warnings.warn(
                "xESMF could not be imported; falling back to xarray interpolation "
                f"({type(_XESMF_IMPORT_ERROR).__name__}: {_XESMF_IMPORT_ERROR}).",
                RuntimeWarning,
            )

        return _XarrayRegridder(grid_out, target_shape=self.fine_shape, method=method)

    # ------------------------------------------------------------------
    def _infer_coord_name(self, ds: xr.Dataset, candidates: Iterable[str]) -> str:
        for name in candidates:
            if name in ds.coords or name in ds.data_vars:
                return name
        raise ValueError(f"Unable to locate coordinate names among {candidates}")

    def _infer_time_dim(self, ds: xr.Dataset) -> Optional[str]:
        for dim in ds.dims:
            if "time" in dim.lower():
                return dim
        return None

    def _build_grid(
        self, lat_da: xr.DataArray, lon_da: xr.DataArray, lat_name: str, lon_name: str
    ) -> xr.Dataset:
        lat = self._rename_array(lat_da, lat_name, "lat")
        lon = self._rename_array(lon_da, lon_name, "lon")
        return xr.Dataset({"lat": lat, "lon": lon})

    def _rename_array(self, array: xr.DataArray, old_name: str, new_name: str) -> xr.DataArray:
        result = array
        if old_name != new_name and old_name in result.dims:
            result = result.rename({old_name: new_name})
        if result.name != new_name:
            result = result.rename(new_name)
        return result

    def _compute_shape(
        self, lat_da: xr.DataArray, lon_da: xr.DataArray
    ) -> Tuple[int, int]:
        if lat_da.ndim == 1 and lon_da.ndim == 1:
            return lat_da.sizes[lat_da.dims[0]], lon_da.sizes[lon_da.dims[0]]
        if lat_da.ndim == 2 and lon_da.ndim == 2:
            return lat_da.shape[0], lat_da.shape[1]
        raise ValueError("Unsupported coordinate dimensionality for grid definition")

    def _rename_lat_lon(
        self, data: xr.DataArray, lat_name: str, lon_name: str
    ) -> xr.DataArray:
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

    def _read_time_length(self, path: str) -> int:
        with xr.open_dataset(path) as ds:
            if self.time_dim and self.time_dim in ds.dims:
                return int(ds.sizes[self.time_dim])
        return 1

    def _to_numpy(self, data: Union[xr.DataArray, np.ndarray]) -> np.ndarray:
        if isinstance(data, xr.DataArray):
            array = data.to_numpy()
        else:
            array = np.asarray(data)
        if array.dtype != np.float32:
            array = array.astype(np.float32, copy=False)
        return array


def _time_coordinate_key(value: object) -> Tuple[int, int, int, int, int, int, int, int]:
    """Return a calendar-independent, exact civil-time key."""

    if isinstance(value, np.datetime64):
        if np.isnat(value):
            raise ValueError("time coordinate contains NaT")
        text = np.datetime_as_string(value, unit="ns")
        date_text, _, time_text = text.partition("T")
        year_text, month_text, day_text = date_text.split("-")
        hour = minute = second = nanosecond = 0
        if time_text:
            hour_text, minute_text, second_text = time_text.split(":")
            hour = int(hour_text)
            minute = int(minute_text)
            if "." in second_text:
                whole_seconds, fraction = second_text.split(".", 1)
                second = int(whole_seconds)
                nanosecond = int(fraction.ljust(9, "0")[:9])
            else:
                second = int(second_text)
        return (
            int(year_text),
            int(month_text),
            int(day_text),
            hour,
            minute,
            second,
            nanosecond // 1000,
            nanosecond % 1000,
        )

    required = ("year", "month", "day", "hour", "minute", "second")
    if not all(hasattr(value, field) for field in required):
        raise ValueError(
            "time coordinate values must decode to datetime-like objects; "
            f"got {type(value).__name__}: {value!r}"
        )
    microsecond = int(getattr(value, "microsecond", 0))
    nanosecond = int(getattr(value, "nanosecond", 0))
    return (
        int(getattr(value, "year")),
        int(getattr(value, "month")),
        int(getattr(value, "day")),
        int(getattr(value, "hour")),
        int(getattr(value, "minute")),
        int(getattr(value, "second")),
        microsecond,
        nanosecond,
    )


def _format_time_key(key: Tuple[int, int, int, int, int, int, int, int]) -> str:
    year, month, day, hour, minute, second, microsecond, nanosecond = key
    fraction = microsecond * 1000 + nanosecond
    suffix = f".{fraction:09d}" if fraction else ""
    return f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}{suffix}"


def _exact_time_index(
    predictor_values: Sequence[object],
    target_values: Sequence[object],
    *,
    predictor_path: str = "<predictor>",
    target_path: str = "<target>",
) -> np.ndarray:
    """Build a unique target index for every predictor civil timestamp."""

    def _keys(values: Sequence[object], *, role: str, path: str):
        keys = []
        indices_by_key: dict[Tuple[int, int, int, int, int, int, int, int], List[int]] = {}
        for index, value in enumerate(values):
            try:
                key = _time_coordinate_key(value)
            except ValueError as exc:
                raise ValueError(
                    f"{role} file {path} has an invalid time value at index {index}: {exc}"
                ) from exc
            keys.append(key)
            indices_by_key.setdefault(key, []).append(index)

        duplicates = {
            key: indices for key, indices in indices_by_key.items() if len(indices) > 1
        }
        if duplicates:
            examples = ", ".join(
                f"{_format_time_key(key)} at indices {indices[:4]}"
                for key, indices in list(duplicates.items())[:4]
            )
            raise ValueError(
                f"{role} file {path} contains duplicate time coordinates: {examples}"
            )
        return keys, indices_by_key

    predictor_keys, _ = _keys(
        predictor_values, role="Predictor", path=predictor_path
    )
    _, target_index_by_key = _keys(target_values, role="Target", path=target_path)

    missing = [key for key in predictor_keys if key not in target_index_by_key]
    if missing:
        examples = ", ".join(_format_time_key(key) for key in missing[:8])
        raise ValueError(
            "Target time coordinate does not contain every predictor date: "
            f"{len(missing)} missing date(s) from {target_path}; examples: {examples}. "
            f"Predictor file: {predictor_path}"
        )

    return np.asarray(
        [target_index_by_key[key][0] for key in predictor_keys], dtype=np.int64
    )


class _XarrayRegridder:
    """Small fallback regridder when xESMF isn't available."""

    def __init__(
        self,
        grid_out: xr.Dataset,
        target_shape: Tuple[int, int],
        method: str = "bilinear",
    ) -> None:
        self.target_shape = target_shape
        self.method = "linear" if method in {"bilinear", "linear"} else "nearest"
        self._lat = grid_out["lat"]
        self._lon = grid_out["lon"]
        self._template = xr.Dataset({"lat": self._lat, "lon": self._lon})

        if self._lat.ndim == 1 and self._lon.ndim == 1:
            self._dims = ("lat", "lon")
            self._coords = {"lat": self._lat, "lon": self._lon}
        else:
            # Retain any higher-dimensional coordinate definitions.
            self._dims = self._lat.dims
            self._coords = {
                "lat": (self._lat.dims, self._lat.to_numpy()),
                "lon": (self._lon.dims, self._lon.to_numpy()),
            }

    def __call__(self, data: xr.DataArray) -> Union[xr.DataArray, np.ndarray]:
        try:
            return data.interp(lat=self._lat, lon=self._lon, method=self.method)
        except Exception:
            pass

        try:
            return data.interp_like(self._template, method=self.method)
        except Exception:
            pass

        # Fall back to plain bilinear resize in numpy/torch.
        array = data.to_numpy()
        tensor = torch.from_numpy(array.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0)
        resized = torch.nn.functional.interpolate(  # type: ignore[attr-defined]
            tensor,
            size=self.target_shape,
            mode="bilinear" if self.method == "linear" else "nearest",
            align_corners=False,
        )
        upsampled = resized.squeeze(0).squeeze(0).cpu().numpy()
        return xr.DataArray(upsampled, dims=self._dims, coords=self._coords)
