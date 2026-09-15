"""Numerical and spatial diagnostics for residual-refinement tensors.

The helpers in this module are intentionally independent of training and
sampling.  They inspect tensors without modifying them and return plain Python
objects that can be written directly to JSON.  Both NumPy arrays and PyTorch
tensors are accepted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

__all__ = ["summarize_refinement_tensor", "summarize_refinement_tensors",
           "boundary_region_masks", "empirical_crps", "negative_value_summary"]


def boundary_region_masks(
    lat: Any, lon: Any, widths: Sequence[int] = (1, 2, 4, 8, 16),
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Geographic edges and corners, independent of latitude ordering.

    Widths count cells, capped at half the shorter axis. Edges include corners:
    regions overlap and must not be summed. The southeast quadrant is defined
    separately by the coordinate midpoints. Missing-data holes do not redefine
    the rectangular domain's boundary.
    """
    latitude, longitude = np.asarray(lat), np.asarray(lon)
    for name, coordinate in (("lat", latitude), ("lon", longitude)):
        if coordinate.ndim != 1 or coordinate.size < 2 or not np.isfinite(coordinate).all():
            raise ValueError(f"{name} must be a finite one-dimensional coordinate with at least two cells.")
        delta = np.diff(coordinate)
        if not ((delta > 0).all() or (delta < 0).all()):
            raise ValueError(f"{name} must be strictly monotonic for geographic boundary diagnostics.")
    height, width = latitude.size, longitude.size
    row, column = np.indices((height, width))
    distance = np.minimum(np.minimum(row, height - 1 - row),
                          np.minimum(column, width - 1 - column))
    south_distance = row if latitude[0] < latitude[-1] else height - 1 - row
    north_distance = height - 1 - south_distance
    west_distance = column if longitude[0] < longitude[-1] else width - 1 - column
    east_distance = width - 1 - west_distance
    masks = {
        "full": np.ones((height, width), dtype=bool),
        "southeast_quadrant": (
            (latitude[:, None] <= (latitude.min() + latitude.max()) / 2)
            & (longitude[None, :] >= (longitude.min() + longitude.max()) / 2)
        ),
    }
    requested = [int(item) for item in widths]
    if any(item < 1 for item in requested):
        raise ValueError("boundary widths must be positive integers.")
    for strip in sorted({min(item, min(height, width) // 2) for item in requested}):
        edges = {"south": south_distance < strip, "north": north_distance < strip,
                 "west": west_distance < strip, "east": east_distance < strip}
        masks[f"boundary_{strip}"] = distance < strip
        masks[f"interior_{strip}"] = distance >= strip
        masks.update({f"{name}_{strip}": mask for name, mask in edges.items()})
        for vertical in ("north", "south"):
            for horizontal in ("east", "west"):
                masks[f"{vertical}{horizontal}_{strip}"] = edges[vertical] & edges[horizontal]
    for ring in range(int(distance.max()) + 1):
        masks[f"distance_{ring}"] = distance == ring
    return masks, distance


def empirical_crps(members: Any, target: Any, *, member_axis: int = 1) -> np.ndarray:
    """CRPS of the issued empirical ensemble using O(M log M) sorting.

    This is not the optional unbiased estimate of infinite-ensemble CRPS.
    Invalid targets or any invalid member yield an invalid score at that cell.
    """
    ensemble = np.moveaxis(_as_numpy(members).astype(np.float64), member_axis, 0)
    truth = _as_numpy(target).astype(np.float64)
    if ensemble.shape[1:] != truth.shape or ensemble.shape[0] < 1:
        raise ValueError("Ensemble and target shapes must agree after removing the member axis.")
    count = ensemble.shape[0]
    ordered = np.sort(ensemble, axis=0)
    weights = (2 * np.arange(count) - count + 1).reshape((count,) + (1,) * truth.ndim)
    pair_term = np.sum(weights * ordered, axis=0) / count**2
    score = np.mean(np.abs(ensemble - truth), axis=0) - pair_term
    return np.where(np.isfinite(truth) & np.isfinite(ensemble).all(axis=0), score, np.nan)


def negative_value_summary(values: Any) -> dict[str, int | float | None]:
    """Quantify negatives without changing them or treating zeros as missing."""
    array = _as_numpy(values)
    finite = array[np.isfinite(array)].astype(np.float64)
    negative = finite[finite < 0]
    return {
        "finite_count": int(finite.size), "negative_count": int(negative.size),
        "negative_fraction": float(negative.size / finite.size) if finite.size else None,
        "mean_negative_magnitude": float(-negative.mean()) if negative.size else 0.0 if finite.size else None,
        "mean_negative_deficit": float(-negative.sum() / finite.size) if finite.size else None,
        "minimum": float(finite.min()) if finite.size else None,
    }


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().to(device="cpu")
    return np.asarray(value)


def _finite_summary(values: np.ndarray) -> dict[str, int | float | None]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    quantiles = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
    result: dict[str, int | float | None] = {
        "finite_count": int(finite.size),
        "nan_count": int(np.isnan(array).sum()),
        "inf_count": int(np.isinf(array).sum()),
    }
    if finite.size == 0:
        result.update(
            {
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "q01": None,
                "q05": None,
                "q25": None,
                "q50": None,
                "q75": None,
                "q95": None,
                "q99": None,
                "zero_fraction": None,
            }
        )
        return result

    q_values = np.quantile(finite, quantiles)
    result.update(
        {
            "min": float(finite.min()),
            "max": float(finite.max()),
            "mean": float(finite.mean()),
            "std": float(finite.std()),
            "q01": float(q_values[0]),
            "q05": float(q_values[1]),
            "q25": float(q_values[2]),
            "q50": float(q_values[3]),
            "q75": float(q_values[4]),
            "q95": float(q_values[5]),
            "q99": float(q_values[6]),
            "zero_fraction": float(np.count_nonzero(finite == 0.0) / finite.size),
        }
    )
    return result


def _spatial_gradient_magnitude(array: np.ndarray) -> float | None:
    if array.ndim < 2 or array.shape[-2] < 2 or array.shape[-1] < 2:
        return None
    values = np.asarray(array, dtype=np.float64)
    dx = values[..., :-1, 1:] - values[..., :-1, :-1]
    dy = values[..., 1:, :-1] - values[..., :-1, :-1]
    magnitude = np.hypot(dx, dy)
    finite = magnitude[np.isfinite(magnitude)]
    return float(finite.mean()) if finite.size else None


def _high_frequency_power_fraction(
    array: np.ndarray,
    *,
    cutoff: float,
) -> float | None:
    """Return power above ``cutoff`` times the axial Nyquist frequency.

    Every leading-index slice is treated as an independent spatial field.  The
    spatial mean is removed before the FFT, so the result measures texture and
    cannot be dominated by a field's climatological offset.
    """
    if not 0.0 < cutoff < 1.0:
        raise ValueError("high_frequency_cutoff must be between 0 and 1.")
    if array.ndim < 2 or array.shape[-2] < 2 or array.shape[-1] < 2:
        return None

    values = np.asarray(array, dtype=np.float64)
    height, width = values.shape[-2:]
    planes = values.reshape((-1, height, width))
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.rfftfreq(width)[None, :]
    radius = np.sqrt(fy * fy + fx * fx)
    high_mask = radius >= (0.5 * cutoff)
    high_power = 0.0
    total_power = 0.0

    for plane in planes:
        finite = np.isfinite(plane)
        if not finite.any():
            continue
        fill = float(plane[finite].mean())
        clean = np.where(finite, plane, fill)
        clean = clean - clean.mean()
        power = np.abs(np.fft.rfft2(clean)) ** 2
        # Account for the negative-frequency half omitted by rFFT.
        multiplicity = np.full(power.shape[1], 2.0)
        multiplicity[0] = 1.0
        if width % 2 == 0:
            multiplicity[-1] = 1.0
        power *= multiplicity[None, :]
        total_power += float(power.sum())
        high_power += float(power[high_mask].sum())

    if total_power <= np.finfo(np.float64).eps:
        return 0.0
    return float(high_power / total_power)


def _lag_one_spatial_autocorrelation(array: np.ndarray) -> float | None:
    """Return the pooled horizontal/vertical lag-one spatial correlation."""
    if array.ndim < 2 or array.shape[-2] < 2 or array.shape[-1] < 2:
        return None
    values = np.asarray(array, dtype=np.float64)
    horizontal_a = values[..., :, :-1].reshape(-1)
    horizontal_b = values[..., :, 1:].reshape(-1)
    vertical_a = values[..., :-1, :].reshape(-1)
    vertical_b = values[..., 1:, :].reshape(-1)
    first = np.concatenate((horizontal_a, vertical_a))
    second = np.concatenate((horizontal_b, vertical_b))
    valid = np.isfinite(first) & np.isfinite(second)
    first = first[valid]
    second = second[valid]
    if first.size < 2:
        return None
    first = first - first.mean()
    second = second - second.mean()
    denominator = np.sqrt(np.dot(first, first) * np.dot(second, second))
    if not np.isfinite(denominator) or denominator <= np.finfo(np.float64).eps:
        return None
    return float(np.dot(first, second) / denominator)


def _normalize_axis(axis: int, ndim: int) -> int:
    normalized = axis + ndim if axis < 0 else axis
    if not 0 <= normalized < ndim:
        raise ValueError(
            f"channel_dim={axis} is invalid for a {ndim}-D tensor."
        )
    return normalized


def _metrics(
    array: np.ndarray,
    cutoff: float,
    wet_threshold: float | None = None,
) -> dict[str, int | float | None]:
    result = _finite_summary(array)
    result["spatial_gradient_magnitude"] = _spatial_gradient_magnitude(array)
    result["lag_one_spatial_autocorrelation"] = _lag_one_spatial_autocorrelation(array)
    result["high_frequency_spectral_power_fraction"] = (
        _high_frequency_power_fraction(array, cutoff=cutoff)
    )
    if wet_threshold is not None:
        finite = np.asarray(array, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        result["wet_threshold"] = float(wet_threshold)
        result["wet_fraction"] = (
            float(np.count_nonzero(finite >= float(wet_threshold)) / finite.size)
            if finite.size
            else None
        )
    return result


def summarize_refinement_tensor(
    name: str,
    tensor: Any,
    *,
    channel_names: Sequence[str] | None = None,
    channel_dim: int = 1,
    dimension_names: Sequence[str] | None = None,
    channel_units: Sequence[str] | Mapping[str, str] | None = None,
    wet_thresholds: Mapping[str, float] | None = None,
    high_frequency_cutoff: float = 0.5,
) -> dict[str, Any]:
    """Summarize one named tensor, including per-channel spatial diagnostics.

    Spatial dimensions are always the final two dimensions.  For standard
    model tensors this corresponds to ``[batch, channel, lat, lon]``.  A clear
    error is raised if declared channel names do not match the tensor layout.
    """
    array = _as_numpy(tensor)
    if array.ndim < 2:
        raise ValueError(
            f"{name!r} must have at least two spatial dimensions; "
            f"got shape {array.shape}."
        )

    names = [str(item) for item in (channel_names or [])]
    dims = [str(item) for item in (dimension_names or [])]
    if dims and len(dims) != array.ndim:
        raise ValueError(
            f"{name!r} has {array.ndim} dimensions, but {len(dims)} dimension "
            "names were supplied."
        )
    if channel_units is None:
        unit_map: dict[str, str] = {}
    elif isinstance(channel_units, Mapping):
        unit_map = {str(key): str(value) for key, value in channel_units.items()}
    else:
        if names and len(channel_units) != len(names):
            raise ValueError(
                f"{name!r} has {len(names)} named channels, but "
                f"{len(channel_units)} units were supplied."
            )
        unit_map = {
            channel_name: str(unit)
            for channel_name, unit in zip(names, channel_units)
        }
    result: dict[str, Any] = {
        "name": str(name),
        "shape": [int(size) for size in array.shape],
        "dtype": str(array.dtype),
        "dimension_names": dims,
        "channel_names": names,
        "channel_units": unit_map,
        "overall": _metrics(array, high_frequency_cutoff),
        "per_channel": {},
    }

    if names:
        axis = _normalize_axis(channel_dim, array.ndim)
        if array.shape[axis] != len(names):
            raise ValueError(
                f"{name!r} has {array.shape[axis]} channels on dimension "
                f"{axis}, but {len(names)} channel names were supplied."
            )
        for index, channel_name in enumerate(names):
            channel = np.take(array, index, axis=axis)
            result["per_channel"][channel_name] = _metrics(
                channel,
                high_frequency_cutoff,
                None if wet_thresholds is None else wet_thresholds.get(channel_name),
            )
            if channel_name in unit_map:
                result["per_channel"][channel_name]["units"] = unit_map[channel_name]
    return result


def summarize_refinement_tensors(
    tensors: Mapping[str, Any],
    *,
    channel_names: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    channel_dim: int | Mapping[str, int] = 1,
    dimension_names: Mapping[str, Sequence[str]] | None = None,
    channel_units: (
        Sequence[str]
        | Mapping[str, str]
        | Mapping[str, Sequence[str] | Mapping[str, str]]
        | None
    ) = None,
    wet_thresholds: (
        Mapping[str, float] | Mapping[str, Mapping[str, float]] | None
    ) = None,
    high_frequency_cutoff: float = 0.5,
) -> dict[str, dict[str, Any]]:
    """Create diagnostics for a mapping of semantic tensor names to values.

    ``channel_names`` may be one sequence shared by every tensor or a mapping
    keyed by tensor name.  This supports the canonical diagnostic set:
    ground truth, Phase 1 prediction, true and normalized residuals, predicted
    normalized and physical residuals, and the final refined prediction.
    """
    report: dict[str, dict[str, Any]] = {}
    for name, value in tensors.items():
        if isinstance(channel_names, Mapping):
            names = channel_names.get(name)
        else:
            names = channel_names
        units_for_tensor = channel_units
        if isinstance(channel_units, Mapping):
            candidate_units = channel_units.get(name)
            if isinstance(candidate_units, Mapping) or (
                isinstance(candidate_units, Sequence)
                and not isinstance(candidate_units, str)
            ):
                units_for_tensor = candidate_units
        wet_thresholds_for_tensor = wet_thresholds
        if isinstance(wet_thresholds, Mapping):
            candidate_thresholds = wet_thresholds.get(name)
            if isinstance(candidate_thresholds, Mapping):
                wet_thresholds_for_tensor = candidate_thresholds
        report[str(name)] = summarize_refinement_tensor(
            str(name),
            value,
            channel_names=names,
            channel_dim=(
                channel_dim.get(name, 1)
                if isinstance(channel_dim, Mapping)
                else channel_dim
            ),
            dimension_names=(
                None if dimension_names is None else dimension_names.get(name)
            ),
            channel_units=units_for_tensor,
            wet_thresholds=wet_thresholds_for_tensor,
            high_frequency_cutoff=high_frequency_cutoff,
        )
    return report
