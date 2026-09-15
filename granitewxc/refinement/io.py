"""NetCDF output for two-phase (deterministic + refined ensemble) predictions.

Preserves everything the deterministic workflow already preserves — timestamps,
coordinates, variable names, units, calendar, masks, fill values and attributes
— and adds the Phase-2 products:

* ``<var>``                deterministic Phase-1 prediction
* ``<var>_residual``       effective predicted residual in physical units
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
    members_unbounded=None,
    member_residuals=None,
    member_residuals_physical=None,
    sampling_states=None,
    sampling_steps=None,
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
    * Member products: ``[time, member, variable, lat, lon]`` in draw order.
    * ``sampling_states``: ``[time, step, member, variable, lat, lon]``;
      ``sampling_steps`` identifies the actual saved integration/noise levels.
    """
    import xarray as xr

    variables = list(variables)
    if not variables or len(variables) != len(set(variables)):
        raise ValueError("variables must be nonempty and unique.")
    units = dict(units or {})
    deterministic_shape = _as_numpy(deterministic).shape
    if len(deterministic_shape) != 4:
        raise ValueError("deterministic must have [time, variable, lat, lon] dimensions.")

    def _split(array, name_suffix: str, extra_attrs: Mapping[str, Any] | None = None):
        data = _as_numpy(array)
        if data.ndim != 4 or data.shape != deterministic_shape:
            raise ValueError(f"{name_suffix or 'field'} shape {data.shape} does not match deterministic {deterministic_shape}.")
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
                    "comment": (
                        "Effective correction in the same physical units as the "
                        "target variable. It equals refined minus deterministic "
                        "after per-member physical reconstruction and constraints."
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
                {"long_name": "physical ensemble standard deviation (unbiased)"},
            )
        )
    if truth is not None:
        data_vars.update(_split(truth, "_truth", {"long_name": "ground truth"}))

    member_count = None
    for product, array, representation in (
        ("members", members, "physical prediction after constraints"),
        ("members_unbounded", members_unbounded, "physical prediction before constraints"),
        ("member_residuals", member_residuals, "normalized signed residual"),
        ("member_residuals_physical", member_residuals_physical, "effective signed physical correction"),
    ):
        if array is None:
            continue
        member_data = _as_numpy(array)
        if member_data.ndim != 5:
            raise ValueError(f"{product} must be [time, member, variable, lat, lon], got {member_data.shape}.")
        if member_data.shape[2] != len(variables):
            raise ValueError(f"{product} has {member_data.shape[2]} channels but {len(variables)} variables were declared.")
        if (member_data.shape[0], *member_data.shape[2:]) != deterministic_shape:
            raise ValueError(f"{product} shape does not match deterministic time/channel/spatial dimensions.")
        if member_data.shape[1] < 1 or (member_count is not None and member_count != member_data.shape[1]):
            raise ValueError("All member products must have the same nonempty member axis.")
        member_count = member_data.shape[1]
        for index, name in enumerate(variables):
            data_vars[f"{name}_{product}"] = xr.DataArray(
                member_data[:, :, index],
                dims=(time_dim, member_dim, lat_dim, lon_dim),
                attrs={
                    "units": "1" if product == "member_residuals" else units.get(name, ""),
                    "long_name": f"{representation}; ensemble members in draw order",
                    "representation": representation,
                },
            )

    if sampling_states is not None:
        states = _as_numpy(sampling_states)
        if states.ndim != 6 or (states.shape[0], *states.shape[3:]) != deterministic_shape:
            raise ValueError("sampling_states must have [time, step, member, variable, lat, lon] dimensions matching deterministic.")
        if member_count is not None and states.shape[2] != member_count:
            raise ValueError("sampling_states member count differs from physical members.")
        steps = np.asarray(sampling_steps)
        if steps.ndim != 1 or steps.size != states.shape[1]:
            raise ValueError("sampling_steps must identify each saved sampling state.")
        coords = dict(coords)
        coords["sampling_step"] = steps
        member_count = states.shape[2]
        for index, name in enumerate(variables):
            data_vars[f"{name}_sampling_states"] = xr.DataArray(
                states[:, :, :, index],
                dims=(time_dim, "sampling_step", member_dim, lat_dim, lon_dim),
                attrs={"units": "1", "representation": "stochastic state in normalized residual coordinates",
                       "comment": "Intermediate states are not physical rainfall or temperature."},
            )

    dataset = xr.Dataset(data_vars, coords=dict(coords))
    if member_count is not None and member_dim not in dataset.coords:
        dataset = dataset.assign_coords(
            {member_dim: np.arange(member_count, dtype="int32")}
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
