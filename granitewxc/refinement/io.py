"""NetCDF output for two-phase (deterministic + refined ensemble) predictions.

Preserves everything the deterministic workflow already preserves — timestamps,
coordinates, variable names, units, calendar, masks, fill values and attributes
— and adds the Phase-2 products:

* ``<var>``                deterministic Phase-1 prediction
* ``<var>_residual``       predicted residual (normalized target space)
* ``<var>_refined``        refined prediction (ensemble mean when N > 1)
* ``<var>_members``        every ensemble member, with an explicit ``member`` dim
* ``<var>_ensemble_mean``  ensemble mean
* ``<var>_ensemble_spread`` unbiased ensemble standard deviation

I/O settings only affect *how* the data is stored: chunking and compression are
configurable, values and sample order are never changed.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = ["build_refined_dataset", "write_refined_netcdf"]

_RESIDUAL_UNITS = "1 (normalized target space)"


def _as_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().to("cpu").numpy()
    return np.asarray(value)


def build_refined_dataset(
    *,
    variables: Sequence[str],
    coords: Mapping[str, Any],
    deterministic,
    refined=None,
    residual=None,
    members=None,
    ensemble_mean=None,
    ensemble_spread=None,
    truth=None,
    units: Mapping[str, str] | None = None,
    attrs: Mapping[str, Any] | None = None,
    time_dim: str = "time",
    lat_dim: str = "lat",
    lon_dim: str = "lon",
    member_dim: str = "member",
):
    """Assemble an :class:`xarray.Dataset` from two-phase model outputs.

    Array layout expectations:

    * ``deterministic`` / ``refined`` / ``residual`` / ``ensemble_*`` / ``truth``:
      ``[time, variable, lat, lon]``
    * ``members``: ``[time, member, variable, lat, lon]`` in draw order
    """
    import xarray as xr

    variables = list(variables)
    units = dict(units or {})

    def _split(array, name_suffix: str, extra_attrs: Mapping[str, Any] | None = None):
        data = _as_numpy(array)
        if data.shape[1] != len(variables):
            raise ValueError(
                f"{name_suffix or 'field'} has {data.shape[1]} channels but "
                f"{len(variables)} variables were declared."
            )
        out = {}
        for index, name in enumerate(variables):
            attributes = {"units": units.get(name, "")}
            if extra_attrs:
                attributes.update(extra_attrs)
            out[f"{name}{name_suffix}"] = xr.DataArray(
                data[:, index],
                dims=(time_dim, lat_dim, lon_dim),
                attrs=attributes,
            )
        return out

    data_vars: dict[str, Any] = {}
    data_vars.update(_split(deterministic, "", {"long_name": "deterministic Phase-1 prediction"}))
    if residual is not None:
        data_vars.update(
            _split(
                residual,
                "_residual",
                {
                    "long_name": "predicted Phase-2 residual",
                    "units": _RESIDUAL_UNITS,
                    "comment": (
                        "Residual is defined and added in the Phase-1 normalized "
                        "target space; inverse normalization is applied once, "
                        "afterwards."
                    ),
                },
            )
        )
    if refined is not None:
        data_vars.update(_split(refined, "_refined", {"long_name": "refined prediction"}))
    if ensemble_mean is not None:
        data_vars.update(_split(ensemble_mean, "_ensemble_mean", {"long_name": "ensemble mean"}))
    if ensemble_spread is not None:
        data_vars.update(
            _split(
                ensemble_spread,
                "_ensemble_spread",
                {"long_name": "ensemble standard deviation (unbiased)"},
            )
        )
    if truth is not None:
        data_vars.update(_split(truth, "_truth", {"long_name": "ground truth"}))

    if members is not None:
        member_data = _as_numpy(members)
        if member_data.ndim != 5:
            raise ValueError(
                f"members must be [time, member, variable, lat, lon], got shape {member_data.shape}"
            )
        for index, name in enumerate(variables):
            data_vars[f"{name}_members"] = xr.DataArray(
                member_data[:, :, index],
                dims=(time_dim, member_dim, lat_dim, lon_dim),
                attrs={
                    "units": units.get(name, ""),
                    "long_name": "refined ensemble members in draw order",
                },
            )

    dataset = xr.Dataset(data_vars, coords=dict(coords))
    if members is not None and member_dim not in dataset.coords:
        dataset = dataset.assign_coords(
            {member_dim: np.arange(_as_numpy(members).shape[1], dtype="int32")}
        )
    dataset.attrs.update(dict(attrs or {}))
    dataset.attrs.setdefault(
        "refinement_note",
        "Phase-2 residual refinement. Predictors and targets share the same "
        "timestamp; no forecast lead time is modelled.",
    )
    return dataset


def write_refined_netcdf(
    dataset,
    path: str | os.PathLike,
    *,
    compression: bool = True,
    compression_level: int = 4,
    chunk_sizes: Mapping[str, int] | None = None,
    atomic: bool = True,
    unlimited_dims: Sequence[str] | None = None,
) -> str:
    """Write ``dataset`` once, with configurable chunking and compression.

    The file is produced in a single pass (never opened and rewritten per
    variable or per ensemble member), optionally through a temporary file so a
    crash cannot leave a partial output in place.
    """
    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)

    encoding: dict[str, dict[str, Any]] = {}
    for name, variable in dataset.data_vars.items():
        spec: dict[str, Any] = {}
        if compression:
            spec.update({"zlib": True, "complevel": int(compression_level)})
        if chunk_sizes:
            sizes = [
                int(chunk_sizes.get(dim, dataset.sizes[dim]))
                for dim in variable.dims
            ]
            spec["chunksizes"] = tuple(max(1, min(s, dataset.sizes[d])) for s, d in zip(sizes, variable.dims))
        if np.issubdtype(variable.dtype, np.floating):
            spec.setdefault("_FillValue", np.nan)
        if spec:
            encoding[name] = spec

    kwargs: dict[str, Any] = {"encoding": encoding, "format": "NETCDF4"}
    if unlimited_dims:
        kwargs["unlimited_dims"] = list(unlimited_dims)

    if not atomic:
        dataset.to_netcdf(path, **kwargs)
        return path

    fd, tmp = tempfile.mkstemp(prefix=".nc-", suffix=".tmp", dir=directory)
    os.close(fd)
    try:
        dataset.to_netcdf(tmp, **kwargs)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path
