"""PyTorch dataset utilities for CORDEX downscaling experiments."""

from __future__ import annotations

import os
from bisect import bisect_right
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import xarray as xr
import xesmf as xe
from torch.utils.data import Dataset


PathLike = Union[str, os.PathLike[str]]


class CordexDownscaleDataset(Dataset):
    """Pair coarse predictors with high-resolution CORDEX targets.

    Parameters
    ----------
    predictor_files : Sequence[PathLike]
        NetCDF files that contain the coarse-resolution predictors.
    target_files : Sequence[PathLike]
        NetCDF files that contain the matching high-resolution targets.
    orography_file : PathLike
        NetCDF file containing a static orography field.
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
        orography_file: PathLike,
        predictor_variables: Optional[Sequence[str]] = None,
        target_variables: Optional[Sequence[str]] = None,
        orography_variable: Optional[str] = "orog",
        time_dim: Optional[str] = None,
        regrid_method: str = "bilinear",
        dtype: torch.dtype = torch.float32,
        crop_size: Optional[Tuple[int, int]] = None,
        random_crop: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        predictor_paths = [os.fspath(p) for p in predictor_files]
        target_paths = [os.fspath(p) for p in target_files]

        if not predictor_paths:
            raise ValueError("At least one predictor file is required")
        if len(predictor_paths) != len(target_paths):
            raise ValueError("Predictor and target file lists must match in length")

        self.predictor_paths = predictor_paths
        self.target_paths = target_paths
        self.orography_path = os.fspath(orography_file)
        self.predictor_vars = list(predictor_variables or self.DEFAULT_PREDICTORS)
        self.target_vars = list(target_variables) if target_variables else None
        self.orography_var = orography_variable
        self.dtype = dtype

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

        self.regridder = xe.Regridder(grid_in, grid_out, method=regrid_method, periodic=False)
        self._orography_tensor = self._prepare_orography()

        if crop_size is None:
            self.crop_size = self.fine_shape
        else:
            lat_size = min(int(crop_size[0]), self.fine_shape[0])
            lon_size = min(int(crop_size[1]), self.fine_shape[1])
            self.crop_size = (lat_size, lon_size)

        self.random_crop = random_crop and self.crop_size != self.fine_shape
        self._rng = np.random.default_rng(seed)

        self._time_lengths = self._compute_time_lengths()
        self._cumulative_sizes = self._build_cumulative_sizes(self._time_lengths)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self._cumulative_sizes[-1] if self._cumulative_sizes else 0

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        file_idx, time_idx = self._locate_index(index)

        lat_slice, lon_slice = self._select_crop()

        x = self._load_predictors(self.predictor_paths[file_idx], time_idx, lat_slice, lon_slice)
        y = self._load_targets(self.target_paths[file_idx], time_idx, lat_slice, lon_slice)

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

    def _compute_time_lengths(self) -> List[int]:
        lengths: List[int] = []
        for predictor_path, target_path in zip(self.predictor_paths, self.target_paths):
            predictor_len = self._read_time_length(predictor_path)
            target_len = self._read_time_length(target_path)

            if predictor_len != target_len:
                raise ValueError(
                    f"Time dimension mismatch between {predictor_path} and {target_path}"
                )

            lengths.append(predictor_len)

        return lengths

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

        static = self._get_orography_crop(lat_slice, lon_slice)
        return torch.cat([stacked_tensor, static], dim=0)

    def _load_targets(
        self, path: str, time_index: int, lat_slice: slice, lon_slice: slice
    ) -> torch.Tensor:
        tensors: List[np.ndarray] = []
        with xr.open_dataset(path) as ds:
            for var in self.target_vars:
                da = ds[var]
                if self.time_dim and self.time_dim in da.dims:
                    da = da.isel({self.time_dim: time_index}, drop=True)
                arrays = np.nan_to_num(
                    self._to_numpy(da), nan=0.0, posinf=0.0, neginf=0.0
                )
                tensors.append(arrays)

        stacked = np.stack(tensors, axis=0)[..., lat_slice, lon_slice]
        return torch.from_numpy(stacked).to(self.dtype)

    # ------------------------------------------------------------------
    def _prepare_orography(self) -> torch.Tensor:
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

    def _to_numpy(self, data: xr.DataArray) -> np.ndarray:
        array = data.to_numpy()
        if array.dtype != np.float32:
            array = array.astype(np.float32, copy=False)
        return array
