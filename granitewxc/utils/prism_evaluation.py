"""Streaming metrics for NARR/MERRA downscaling on the canonical PRISM grid.

This module deliberately contains no dataset or model imports.  Evaluation is
performed in physical units and only on finite prediction/target pairs.  The
CLI in :mod:`examples.evaluate_prism_inference` owns file discovery and I/O;
the small, NumPy-only functions here are shared and independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


LEGACY_FLOAT32_COORDINATE_ATOL = 5.0e-6


def validate_evaluation_coordinates(
    canonical: Sequence[float],
    observed: Sequence[float],
    *,
    name: str,
    context: str,
    allow_legacy_float32: bool = False,
    legacy_atol: float = LEGACY_FLOAT32_COORDINATE_ATOL,
) -> bool:
    """Validate one inference coordinate axis against the canonical axis.

    Exact equality is the normal contract.  ``allow_legacy_float32`` is an
    explicit audit-only escape hatch for old inference files that down-cast
    the exact PRISM coordinates to float32.  It never reindexes or interpolates
    values.  The return value is ``True`` only when the legacy path was used.
    """

    expected = np.asarray(canonical, dtype=np.float64)
    actual_raw = np.asarray(observed)
    actual = np.asarray(actual_raw, dtype=np.float64)
    if expected.ndim != 1 or actual.ndim != 1:
        raise ValueError(
            f"{context}: {name} coordinates must be one-dimensional; "
            f"canonical={expected.shape}, observed={actual.shape}"
        )
    if expected.shape != actual.shape:
        raise ValueError(
            f"{context}: {name} coordinate shape {actual.shape} does not "
            f"match canonical shape {expected.shape}"
        )
    if np.array_equal(expected, actual):
        return False

    max_abs = float(np.max(np.abs(expected - actual))) if expected.size else 0.0
    if (
        allow_legacy_float32
        and actual_raw.dtype == np.dtype("float32")
        and np.allclose(expected, actual, rtol=0.0, atol=float(legacy_atol))
    ):
        return True

    unequal = np.flatnonzero(expected != actual)
    first = int(unequal[0]) if unequal.size else -1
    raise ValueError(
        f"{context}: {name} coordinates do not exactly match the canonical "
        f"PRISM grid; first mismatch index={first}, "
        f"canonical={expected[first]!r}, observed={actual[first]!r}, "
        f"max_abs_diff={max_abs:.17g}. Old float32 inference files may be "
        "audited only with --legacy-float32-coordinates; no interpolation is "
        "performed."
    )


def exact_axis_slice(
    source: Sequence[float], canonical: Sequence[float], *, name: str, context: str
) -> slice:
    """Return the contiguous exact slice of ``source`` equal to ``canonical``.

    This is intentionally based on exact values, not nearest-neighbour lookup.
    It therefore catches half-cell shifts, reversed grids, and endpoint loss.
    """

    full = np.asarray(source, dtype=np.float64)
    expected = np.asarray(canonical, dtype=np.float64)
    if full.ndim != 1 or expected.ndim != 1:
        raise ValueError(
            f"{context}: {name} coordinates must be one-dimensional; "
            f"source={full.shape}, canonical={expected.shape}"
        )
    if expected.size == 0:
        raise ValueError(f"{context}: canonical {name} axis is empty")
    if expected.size > full.size:
        raise ValueError(
            f"{context}: canonical {name} axis ({expected.size}) is larger "
            f"than the PRISM source axis ({full.size})"
        )

    candidates = np.flatnonzero(full == expected[0])
    for start_value in candidates:
        start = int(start_value)
        stop = start + int(expected.size)
        if stop <= full.size and np.array_equal(full[start:stop], expected):
            return slice(start, stop)
    raise ValueError(
        f"{context}: canonical {name} coordinates are not an exact contiguous "
        "subset of the PRISM truth coordinates; truth is never interpolated"
    )


@dataclass
class FinalizedVariableMetrics:
    """Final scalar metrics plus maps needed for diagnostics and plotting."""

    scalars: dict[str, float | int]
    prediction_climatology: np.ndarray
    target_climatology: np.ndarray
    bias_climatology: np.ndarray
    temporal_rmse: np.ndarray
    sample_count: np.ndarray


class StreamingVariableMetrics:
    """Accumulate one variable without retaining daily fields in memory."""

    def __init__(self, shape: Sequence[int]) -> None:
        values = tuple(int(v) for v in shape)
        if len(values) != 2 or values[0] <= 0 or values[1] <= 0:
            raise ValueError(f"shape must be positive [lat,lon], got {shape}")
        self.shape = values
        self.prediction_sum = np.zeros(values, dtype=np.float64)
        self.target_sum = np.zeros(values, dtype=np.float64)
        self.squared_error_sum = np.zeros(values, dtype=np.float64)
        self.sample_count = np.zeros(values, dtype=np.uint32)
        self.days_seen = 0

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        pred = np.asarray(prediction, dtype=np.float64)
        truth = np.asarray(target, dtype=np.float64)
        if pred.shape != self.shape or truth.shape != self.shape:
            raise ValueError(
                f"prediction and target must both have shape {self.shape}; "
                f"got {pred.shape} and {truth.shape}"
            )
        finite = np.isfinite(pred) & np.isfinite(truth)
        self.prediction_sum[finite] += pred[finite]
        self.target_sum[finite] += truth[finite]
        delta = pred[finite] - truth[finite]
        self.squared_error_sum[finite] += np.square(delta)
        self.sample_count[finite] += 1
        self.days_seen += 1

    def finalize(self) -> FinalizedVariableMetrics:
        valid = self.sample_count > 0
        if not bool(valid.any()):
            raise ValueError("No finite prediction/PRISM pairs were accumulated")
        denominator = self.sample_count.astype(np.float64)
        pred_clim = np.full(self.shape, np.nan, dtype=np.float64)
        truth_clim = np.full(self.shape, np.nan, dtype=np.float64)
        rmse_map = np.full(self.shape, np.nan, dtype=np.float64)
        pred_clim[valid] = self.prediction_sum[valid] / denominator[valid]
        truth_clim[valid] = self.target_sum[valid] / denominator[valid]
        rmse_map[valid] = np.sqrt(
            self.squared_error_sum[valid] / denominator[valid]
        )
        bias_clim = pred_clim - truth_clim

        n_pairs = int(self.sample_count.sum(dtype=np.uint64))
        pred_total = float(self.prediction_sum.sum(dtype=np.float64))
        truth_total = float(self.target_sum.sum(dtype=np.float64))
        sse_total = float(self.squared_error_sum.sum(dtype=np.float64))
        pred_values = pred_clim[valid]
        truth_values = truth_clim[valid]
        if pred_values.size >= 2 and np.std(pred_values) > 0 and np.std(truth_values) > 0:
            spatial_correlation = float(np.corrcoef(pred_values, truth_values)[0, 1])
        else:
            spatial_correlation = float("nan")

        scalars: dict[str, float | int] = {
            "days": int(self.days_seen),
            "valid_grid_cells": int(valid.sum()),
            "finite_sample_pairs": n_pairs,
            "mean_inference": pred_total / n_pairs,
            "mean_prism": truth_total / n_pairs,
            "mean_bias": (pred_total - truth_total) / n_pairs,
            "space_time_rmse": float(np.sqrt(sse_total / n_pairs)),
            "climatology_rmse": float(np.sqrt(np.mean(np.square(bias_clim[valid])))),
            "spatial_correlation": spatial_correlation,
        }
        return FinalizedVariableMetrics(
            scalars=scalars,
            prediction_climatology=pred_clim,
            target_climatology=truth_clim,
            bias_climatology=bias_clim,
            temporal_rmse=rmse_map,
            sample_count=self.sample_count.copy(),
        )


class StreamingMetricMeans:
    """Accumulate finite daily diagnostic scalars without retaining fields.

    Boundary and spectral artifacts can change sign or location between days
    and disappear from a climatology.  This small accumulator keeps the mean
    of each per-day diagnostic (and a finite-value count for auditability)
    while preserving the evaluator's bounded-memory behavior.
    """

    def __init__(self) -> None:
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self.days_seen = 0

    def update(self, values: Mapping[str, Any]) -> None:
        if not isinstance(values, Mapping):
            raise TypeError(f"daily metrics must be a mapping, got {type(values)!r}")
        for raw_key, raw_value in values.items():
            key = str(raw_key)
            if isinstance(raw_value, (bool, np.bool_)):
                continue
            if not isinstance(raw_value, (int, float, np.integer, np.floating)):
                raise TypeError(
                    f"daily metric {key!r} must be numeric, got {type(raw_value)!r}"
                )
            value = float(raw_value)
            if not np.isfinite(value):
                continue
            self._sums[key] = self._sums.get(key, 0.0) + value
            self._counts[key] = self._counts.get(key, 0) + 1
        self.days_seen += 1

    def finalize(self) -> dict[str, float | int]:
        if self.days_seen <= 0:
            raise ValueError("No daily metric mappings were accumulated")
        result: dict[str, float | int] = {"days": int(self.days_seen)}
        for key in sorted(self._sums):
            count = int(self._counts[key])
            result[key] = self._sums[key] / count
            result[f"{key}_finite_days"] = count
        return result


def _position_set(positions: Iterable[int], size: int) -> list[int]:
    return sorted({int(p) for p in positions if 0 < int(p) < int(size)})


def boundary_gradient_error(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    y_positions: Iterable[int] = (),
    x_positions: Iterable[int] = (),
    half_width: int = 2,
) -> dict[str, float | int]:
    """Compare predicted and true gradients in selected boundary bands.

    The score is ``abs(diff(prediction) - diff(target))``.  Measuring the
    gradient *error* rather than the raw predicted gradient prevents real
    PRISM terrain/coast gradients from being mislabeled as seams.
    """

    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.ndim != 2 or pred.shape != truth.shape:
        raise ValueError(
            f"prediction and target must be same-shape 2-D arrays; "
            f"got {pred.shape} and {truth.shape}"
        )
    if int(half_width) < 0:
        raise ValueError(f"half_width must be non-negative, got {half_width}")

    y_lines = _position_set(y_positions, pred.shape[0])
    x_lines = _position_set(x_positions, pred.shape[1])
    dy = np.abs(np.diff(pred, axis=0) - np.diff(truth, axis=0))
    dx = np.abs(np.diff(pred, axis=1) - np.diff(truth, axis=1))
    y_band = np.zeros(dy.shape, dtype=bool)
    x_band = np.zeros(dx.shape, dtype=bool)
    width = int(half_width)
    for position in y_lines:
        center = position - 1
        lo, hi = max(0, center - width), min(dy.shape[0], center + width + 1)
        y_band[lo:hi, :] = True
    for position in x_lines:
        center = position - 1
        lo, hi = max(0, center - width), min(dx.shape[1], center + width + 1)
        x_band[:, lo:hi] = True

    boundary_values = np.concatenate(
        [dy[y_band & np.isfinite(dy)], dx[x_band & np.isfinite(dx)]]
    )
    interior_values = np.concatenate(
        [dy[(~y_band) & np.isfinite(dy)], dx[(~x_band) & np.isfinite(dx)]]
    )
    boundary_mean = (
        float(np.mean(boundary_values)) if boundary_values.size else float("nan")
    )
    interior_mean = (
        float(np.mean(interior_values)) if interior_values.size else float("nan")
    )
    if np.isfinite(boundary_mean) and np.isfinite(interior_mean):
        ratio = (
            boundary_mean / interior_mean
            if interior_mean > 0.0
            else (float("inf") if boundary_mean > 0.0 else float("nan"))
        )
        excess = boundary_mean - interior_mean
    else:
        ratio = float("nan")
        excess = float("nan")
    return {
        "y_boundary_count": len(y_lines),
        "x_boundary_count": len(x_lines),
        "boundary_gradient_mae": boundary_mean,
        "interior_gradient_mae": interior_mean,
        "boundary_to_interior_ratio": float(ratio),
        "boundary_gradient_excess": float(excess),
    }


def map_native_coordinate_boundaries(
    canonical_axis: Sequence[float], native_centers: Sequence[float]
) -> list[int]:
    """Map native cell boundaries to canonical pixel-edge indices.

    Native centers may be ascending or descending.  Mapping uses insertion
    indices into the monotonic PRISM axis; it does not sample or interpolate a
    target field.
    """

    fine = np.asarray(canonical_axis, dtype=np.float64)
    centers = np.asarray(native_centers, dtype=np.float64)
    if fine.ndim != 1 or centers.ndim != 1:
        raise ValueError("canonical_axis and native_centers must be 1-D")
    centers = centers[np.isfinite(centers)]
    if fine.size < 2 or centers.size < 2:
        return []
    if np.all(np.diff(centers) < 0):
        centers = centers[::-1]
    elif not np.all(np.diff(centers) > 0):
        # Curvilinear center transects can wobble slightly.  Sorting and
        # deduplicating still provides a documented axis-aligned approximation.
        centers = np.unique(centers)
    boundaries = (centers[:-1] + centers[1:]) * 0.5

    descending = bool(fine[0] > fine[-1])
    search_axis = fine[::-1] if descending else fine
    if not np.all(np.diff(search_axis) > 0):
        raise ValueError("canonical_axis must be strictly monotonic")
    indices = np.searchsorted(search_axis, boundaries, side="left")
    if descending:
        indices = fine.size - indices
    return _position_set(indices.tolist(), fine.size)


def median_boundary_period(positions: Iterable[int]) -> float:
    values = np.asarray(sorted({int(v) for v in positions}), dtype=np.float64)
    if values.size < 2:
        return float("nan")
    return float(np.median(np.diff(values)))


def _spectral_summary(
    field: np.ndarray,
    *,
    block_period: float,
    native_period_y: float,
    native_period_x: float,
    relative_bandwidth: float,
    high_frequency_wavelength: float,
) -> dict[str, float]:
    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"spectral fields must be 2-D, got {values.shape}")
    valid = np.isfinite(values)
    if int(valid.sum()) < 4:
        return {
            "total_power": float("nan"),
            "high_frequency_fraction": float("nan"),
            "block_band_fraction": float("nan"),
            "block_axial_fraction": float("nan"),
            "native_axial_fraction": float("nan"),
        }

    centered = np.zeros(values.shape, dtype=np.float64)
    centered[valid] = values[valid] - float(np.mean(values[valid]))
    # The same land mask is used for prediction, truth, and bias.  A separable
    # taper reduces rectangular-domain edge leakage without smoothing the data.
    window = np.multiply.outer(np.hanning(values.shape[0]), np.hanning(values.shape[1]))
    transformed = np.fft.rfft2(centered * window)
    power = np.square(np.abs(transformed))
    # Account for omitted negative-x frequencies in rfft2.
    if power.shape[1] > 1:
        x_weight = np.full(power.shape[1], 2.0, dtype=np.float64)
        x_weight[0] = 1.0
        if values.shape[1] % 2 == 0:
            x_weight[-1] = 1.0
        power *= x_weight[np.newaxis, :]
    power[0, 0] = 0.0
    total = float(np.sum(power))
    if not np.isfinite(total) or total <= 0.0:
        return {
            "total_power": total,
            "high_frequency_fraction": float("nan"),
            "block_band_fraction": float("nan"),
            "block_axial_fraction": float("nan"),
            "native_axial_fraction": float("nan"),
        }

    fy = np.abs(np.fft.fftfreq(values.shape[0]))[:, np.newaxis]
    fx = np.fft.rfftfreq(values.shape[1])[np.newaxis, :]
    radial = np.sqrt(np.square(fy) + np.square(fx))

    def frequency_band(frequency: float, *, minimum_bin: float) -> np.ndarray:
        if not np.isfinite(frequency) or frequency <= 0.0:
            return np.zeros(power.shape, dtype=bool)
        half = max(float(relative_bandwidth) * frequency, minimum_bin)
        return np.abs(radial - frequency) <= half

    min_bin = max(1.0 / values.shape[0], 1.0 / values.shape[1])
    block_frequency = 1.0 / float(block_period)
    block_band = frequency_band(block_frequency, minimum_bin=min_bin)
    block_half = max(float(relative_bandwidth) * block_frequency, min_bin)
    block_axial = (
        (np.abs(fx - block_frequency) <= block_half) & (fy <= block_half)
    ) | ((np.abs(fy - block_frequency) <= block_half) & (fx <= block_half))

    native_axial = np.zeros(power.shape, dtype=bool)
    if np.isfinite(native_period_y) and native_period_y > 1.0:
        native_fy = 1.0 / native_period_y
        half_y = max(float(relative_bandwidth) * native_fy, min_bin)
        native_axial |= (np.abs(fy - native_fy) <= half_y) & (fx <= half_y)
    if np.isfinite(native_period_x) and native_period_x > 1.0:
        native_fx = 1.0 / native_period_x
        half_x = max(float(relative_bandwidth) * native_fx, min_bin)
        native_axial |= (np.abs(fx - native_fx) <= half_x) & (fy <= half_x)

    high_frequency = radial >= 1.0 / float(high_frequency_wavelength)
    return {
        "total_power": total,
        "high_frequency_fraction": float(np.sum(power[high_frequency]) / total),
        "block_band_fraction": float(np.sum(power[block_band]) / total),
        "block_axial_fraction": float(np.sum(power[block_axial]) / total),
        "native_axial_fraction": float(np.sum(power[native_axial]) / total),
    }


def spatial_power_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    block_period: float = 8.0,
    native_period: Sequence[float] = (float("nan"), float("nan")),
    relative_bandwidth: float = 0.15,
    high_frequency_wavelength: float = 16.0,
) -> dict[str, float]:
    """Return normalized spatial/block-scale power diagnostics.

    ``block_period=8`` targets the three-stage UNET bottleneck/upsampling scale.
    ``native_period`` is the approximate coarse-predictor spacing in PRISM
    pixels, ordered ``(lat, lon)``.  Ratios compare normalized band fractions,
    so overall field variance does not dominate the artifact score.
    """

    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.shape != truth.shape or pred.ndim != 2:
        raise ValueError(
            f"prediction and target must be same-shape 2-D arrays; "
            f"got {pred.shape} and {truth.shape}"
        )
    periods = tuple(float(v) for v in native_period)
    if len(periods) != 2:
        raise ValueError(f"native_period must contain (lat,lon), got {native_period}")
    common = np.isfinite(pred) & np.isfinite(truth)
    pred_common = np.where(common, pred, np.nan)
    truth_common = np.where(common, truth, np.nan)
    bias_common = np.where(common, pred - truth, np.nan)

    output: dict[str, float] = {
        "block_period_pixels": float(block_period),
        "native_period_lat_pixels": periods[0],
        "native_period_lon_pixels": periods[1],
    }
    summaries: dict[str, Mapping[str, float]] = {}
    for label, field in (
        ("inference", pred_common),
        ("prism", truth_common),
        ("bias", bias_common),
    ):
        summary = _spectral_summary(
            field,
            block_period=float(block_period),
            native_period_y=periods[0],
            native_period_x=periods[1],
            relative_bandwidth=float(relative_bandwidth),
            high_frequency_wavelength=float(high_frequency_wavelength),
        )
        summaries[label] = summary
        output.update({f"{label}_{key}": value for key, value in summary.items()})

    def safe_ratio(numerator: float, denominator: float) -> float:
        if not np.isfinite(numerator) or not np.isfinite(denominator):
            return float("nan")
        if denominator == 0.0:
            return float("inf") if numerator > 0.0 else float("nan")
        return float(numerator / denominator)

    for key in (
        "high_frequency_fraction",
        "block_band_fraction",
        "block_axial_fraction",
        "native_axial_fraction",
    ):
        output[f"inference_to_prism_{key}_ratio"] = safe_ratio(
            summaries["inference"][key], summaries["prism"][key]
        )
    return output


def prefix_metrics(prefix: str, values: Mapping[str, Any]) -> dict[str, Any]:
    """Prefix metric keys for a flat CSV row while preserving scalar values."""

    return {f"{prefix}_{key}": value for key, value in values.items()}
