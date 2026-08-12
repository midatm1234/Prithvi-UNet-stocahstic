"""Dependency-light metrics for deterministic and stochastic refinement.

The functions in this module operate on NumPy-compatible arrays in *physical
units*.  They deliberately have no model, dataset, xarray, or plotting
dependencies, which makes them suitable for both the NARR--PRISM notebook and
stand-alone streaming/evaluation drivers.

All metrics use only locations where the target, prediction, and optional
mask are valid.  Unless stated otherwise, errors are ``prediction - target``.
Inputs to :func:`deterministic_metrics`, :func:`precipitation_metrics`, and
:func:`temperature_metrics` are for one variable; callers should preserve the
configured variable ordering and call them once per variable.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

__all__ = [
    "deterministic_metrics",
    "ensemble_metrics",
    "precipitation_metrics",
    "strata_masks",
    "stratified_metrics",
    "tasmin_tasmax_metrics",
    "temperature_metrics",
]


MetricMapping = dict[str, float | int]


def _as_float_array(value: Any, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric array") from exc
    return array


def _broadcast_mask(mask: Any | None, shape: tuple[int, ...]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    raw = np.asarray(mask)
    if raw.dtype == np.dtype(bool):
        valid = raw
    elif np.issubdtype(raw.dtype, np.number):
        valid = np.isfinite(raw) & (raw != 0)
    else:
        raise TypeError("mask must contain boolean or numeric values")
    try:
        return np.broadcast_to(valid, shape).copy()
    except ValueError as exc:
        raise ValueError(
            f"mask shape {raw.shape} is not broadcastable to data shape "
            f"{shape}"
        ) from exc


def _paired_arrays(
    prediction: Any,
    target: Any,
    mask: Any | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = _as_float_array(prediction, name="prediction")
    truth = _as_float_array(target, name="target")
    if pred.shape != truth.shape:
        raise ValueError(
            "prediction and target must have identical shapes; "
            f"got {pred.shape} and {truth.shape}"
        )
    valid = _broadcast_mask(mask, pred.shape)
    valid &= np.isfinite(pred) & np.isfinite(truth)
    if not bool(valid.any()):
        raise ValueError("no finite, unmasked prediction/target pairs")
    return pred, truth, valid


def _normalize_axis(axis: int, ndim: int, *, name: str) -> int:
    normalized = int(axis)
    if normalized < 0:
        normalized += ndim
    if not 0 <= normalized < ndim:
        raise ValueError(f"{name} axis {axis} is invalid for {ndim}-D data")
    return normalized


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    x_centered = x - np.mean(x, dtype=np.float64)
    y_centered = y - np.mean(y, dtype=np.float64)
    denominator = np.sqrt(
        np.sum(np.square(x_centered), dtype=np.float64)
        * np.sum(np.square(y_centered), dtype=np.float64)
    )
    if not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan")
    return float(
        np.sum(x_centered * y_centered, dtype=np.float64) / denominator
    )


def _correlations_along_rows(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Return one pairwise-complete correlation for every matrix row."""

    count = valid.sum(axis=1, dtype=np.int64)
    pred_valid = np.where(valid, prediction, 0.0)
    target_valid = np.where(valid, target, 0.0)
    sum_pred = pred_valid.sum(axis=1, dtype=np.float64)
    sum_target = target_valid.sum(axis=1, dtype=np.float64)
    sum_product = (pred_valid * target_valid).sum(axis=1, dtype=np.float64)
    sum_pred_sq = np.square(pred_valid).sum(axis=1, dtype=np.float64)
    sum_target_sq = np.square(target_valid).sum(axis=1, dtype=np.float64)

    safe_count = np.maximum(count, 1)
    covariance = sum_product - sum_pred * sum_target / safe_count
    pred_variance = sum_pred_sq - np.square(sum_pred) / safe_count
    target_variance = sum_target_sq - np.square(sum_target) / safe_count
    # Roundoff can leave a tiny negative value for an otherwise constant row.
    pred_variance = np.maximum(pred_variance, 0.0)
    target_variance = np.maximum(target_variance, 0.0)
    denominator = np.sqrt(pred_variance * target_variance)

    result = np.full(count.shape, np.nan, dtype=np.float64)
    usable = (count >= 2) & (denominator > 0.0) & np.isfinite(denominator)
    result[usable] = covariance[usable] / denominator[usable]
    return np.clip(result, -1.0, 1.0)


def _mean_finite(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if not bool(finite.any()):
        return float("nan")
    return float(np.mean(values[finite], dtype=np.float64))


def _correlation_summaries(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    *,
    sample_axis: int | None,
) -> tuple[float, float, float]:
    """Return temporal, per-sample spatial, and climatology correlations."""

    if sample_axis is None:
        correlation = _pearson(prediction[valid], target[valid])
        return float("nan"), correlation, correlation

    axis = _normalize_axis(sample_axis, prediction.ndim, name="sample")
    pred_samples = np.moveaxis(prediction, axis, 0)
    target_samples = np.moveaxis(target, axis, 0)
    valid_samples = np.moveaxis(valid, axis, 0)
    sample_count = pred_samples.shape[0]

    pred_matrix = pred_samples.reshape(sample_count, -1)
    target_matrix = target_samples.reshape(sample_count, -1)
    valid_matrix = valid_samples.reshape(sample_count, -1)

    spatial = _correlations_along_rows(
        pred_matrix, target_matrix, valid_matrix
    )
    temporal = _correlations_along_rows(
        pred_matrix.T, target_matrix.T, valid_matrix.T
    )

    cell_count = valid_matrix.sum(axis=0, dtype=np.int64)
    usable_cells = cell_count > 0
    if not bool(usable_cells.any()):
        climatology = float("nan")
    else:
        safe_count = np.maximum(cell_count, 1)
        pred_climatology = np.where(valid_matrix, pred_matrix, 0.0).sum(
            axis=0, dtype=np.float64
        ) / safe_count
        target_climatology = np.where(valid_matrix, target_matrix, 0.0).sum(
            axis=0, dtype=np.float64
        ) / safe_count
        climatology = _pearson(
            pred_climatology[usable_cells], target_climatology[usable_cells]
        )
    return _mean_finite(temporal), _mean_finite(spatial), climatology


def _quantile_label(quantile: float) -> str:
    percentage = 100.0 * float(quantile)
    if np.isclose(percentage, round(percentage), rtol=0.0, atol=1.0e-10):
        return f"q{int(round(percentage)):02d}"
    rendered = f"{percentage:.6f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"q{rendered}"


def _validated_quantiles(quantiles: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in quantiles)
    if len(set(values)) != len(values):
        raise ValueError("quantiles must not contain duplicates")
    if any(
        not np.isfinite(value) or not 0.0 <= value <= 1.0
        for value in values
    ):
        raise ValueError("quantiles must be finite values in [0, 1]")
    return values


def _distribution_distances(
    prediction: np.ndarray, target: np.ndarray
) -> tuple[float, float]:
    """Return one-dimensional Wasserstein-1 and two-sample KS distances."""
    pred_sorted = np.sort(prediction)
    target_sorted = np.sort(target)
    # Paired finite filtering gives equal sample counts, so the empirical W1
    # distance is the mean absolute difference of the order statistics.
    wasserstein = float(
        np.mean(np.abs(pred_sorted - target_sorted), dtype=np.float64)
    )
    support = np.sort(np.concatenate((pred_sorted, target_sorted)))
    pred_cdf = np.searchsorted(pred_sorted, support, side="right") / float(
        pred_sorted.size
    )
    target_cdf = np.searchsorted(
        target_sorted, support, side="right"
    ) / float(target_sorted.size)
    ks = float(np.max(np.abs(pred_cdf - target_cdf)))
    return wasserstein, ks


def _safe_ratio(numerator: float | int, denominator: float | int) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def _gradient_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    spatial_axes: Sequence[int] | None,
) -> MetricMapping:
    if prediction.ndim == 0:
        axes: tuple[int, ...] = ()
    elif spatial_axes is None:
        axes = tuple(range(max(0, prediction.ndim - 2), prediction.ndim))
    else:
        axes = tuple(
            _normalize_axis(axis, prediction.ndim, name="spatial")
            for axis in spatial_axes
        )
        if len(set(axes)) != len(axes):
            raise ValueError("spatial_axes must be unique")

    pred_gradients: list[np.ndarray] = []
    target_gradients: list[np.ndarray] = []
    for axis in axes:
        if prediction.shape[axis] < 2:
            continue
        left = [slice(None)] * prediction.ndim
        right = [slice(None)] * prediction.ndim
        left[axis] = slice(None, -1)
        right[axis] = slice(1, None)
        adjacent_valid = valid[tuple(left)] & valid[tuple(right)]
        if not bool(adjacent_valid.any()):
            continue
        pred_difference = np.diff(prediction, axis=axis)
        target_difference = np.diff(target, axis=axis)
        pred_gradients.append(pred_difference[adjacent_valid])
        target_gradients.append(target_difference[adjacent_valid])

    if not pred_gradients:
        return {
            "gradient_valid_count": 0,
            "gradient_mae": float("nan"),
            "gradient_rmse": float("nan"),
            "gradient_agreement": float("nan"),
            "gradient_correlation": float("nan"),
            "gradient_standard_deviation_ratio": float("nan"),
            "prediction_gradient_rms": float("nan"),
            "target_gradient_rms": float("nan"),
            "spatial_roughness_ratio": float("nan"),
            "spatial_smoothing_index": float("nan"),
        }

    pred_gradient = np.concatenate(pred_gradients)
    target_gradient = np.concatenate(target_gradients)
    delta = pred_gradient - target_gradient
    pred_std = float(np.std(pred_gradient))
    target_std = float(np.std(target_gradient))
    pred_norm = float(np.linalg.norm(pred_gradient))
    target_norm = float(np.linalg.norm(target_gradient))
    pred_rms = float(
        np.sqrt(np.mean(np.square(pred_gradient), dtype=np.float64))
    )
    target_rms = float(
        np.sqrt(np.mean(np.square(target_gradient), dtype=np.float64))
    )
    roughness_ratio = _safe_ratio(pred_rms, target_rms)
    if pred_norm == 0.0 and target_norm == 0.0:
        gradient_agreement = 1.0
    elif pred_norm == 0.0 or target_norm == 0.0:
        gradient_agreement = 0.0
    else:
        gradient_agreement = float(
            np.dot(pred_gradient, target_gradient) / (pred_norm * target_norm)
        )
        gradient_agreement = float(np.clip(gradient_agreement, -1.0, 1.0))
    return {
        "gradient_valid_count": int(delta.size),
        "gradient_mae": float(np.mean(np.abs(delta), dtype=np.float64)),
        "gradient_rmse": float(
            np.sqrt(np.mean(np.square(delta), dtype=np.float64))
        ),
        # Cosine agreement remains defined for a constant nonzero gradient;
        # Pearson correlation is also retained for anomaly-pattern analysis.
        "gradient_agreement": gradient_agreement,
        "gradient_correlation": _pearson(pred_gradient, target_gradient),
        "gradient_standard_deviation_ratio": _safe_ratio(pred_std, target_std),
        # Adjacent-grid-difference RMS is a mask-aware, streaming-compatible
        # proxy for small-scale spatial power.  It is preferable to an FFT on
        # this partially masked latitude/longitude domain: zero-filling the
        # mask would create artificial coastline power.  Values below one
        # indicate less resolved spatial variability than the paired target.
        "prediction_gradient_rms": pred_rms,
        "target_gradient_rms": target_rms,
        "spatial_roughness_ratio": roughness_ratio,
        "spatial_smoothing_index": 1.0 - roughness_ratio,
    }


def deterministic_metrics(
    prediction: Any,
    target: Any,
    *,
    mask: Any | None = None,
    quantiles: Sequence[float] = (0.01, 0.05, 0.5, 0.95, 0.99),
    sample_axis: int | None = 0,
    spatial_axes: Sequence[int] | None = None,
) -> MetricMapping:
    """Calculate deterministic metrics for one physical-space variable.

    ``sample_axis`` identifies time/sample for the mean grid-cell temporal and
    mean per-sample spatial-pattern correlations.  Pass ``None`` for a single
    static field.  ``spatial_axes`` controls finite differences for gradient
    agreement and defaults to the last two axes.  ``spatial_roughness_ratio``
    is the prediction/target ratio of adjacent-grid-difference RMS, and
    ``spatial_smoothing_index = 1 - spatial_roughness_ratio``.  The latter is
    positive when a prediction has less small-scale variability than its
    paired target; it is a diagnostic, not proof that all lost power is error.
    """

    pred, truth, valid = _paired_arrays(prediction, target, mask)
    quantile_values = _validated_quantiles(quantiles)
    pred_values = pred[valid]
    target_values = truth[valid]
    error = pred_values - target_values
    mean_bias = float(np.mean(error, dtype=np.float64))
    pred_mean = float(np.mean(pred_values, dtype=np.float64))
    target_mean = float(np.mean(target_values, dtype=np.float64))
    centered_error = (pred_values - pred_mean) - (target_values - target_mean)
    pred_std = float(np.std(pred_values))
    target_std = float(np.std(target_values))
    temporal, spatial_pattern, climatology = _correlation_summaries(
        pred, truth, valid, sample_axis=sample_axis
    )
    wasserstein, ks = _distribution_distances(pred_values, target_values)

    metrics: MetricMapping = {
        "valid_count": int(error.size),
        "prediction_mean": pred_mean,
        "target_mean": target_mean,
        "mean_bias": mean_bias,
        "absolute_bias": abs(mean_bias),
        "mae": float(np.mean(np.abs(error), dtype=np.float64)),
        "rmse": float(np.sqrt(np.mean(np.square(error), dtype=np.float64))),
        "centered_rmse": float(
            np.sqrt(np.mean(np.square(centered_error), dtype=np.float64))
        ),
        "pearson_correlation": _pearson(pred_values, target_values),
        "temporal_correlation": temporal,
        "spatial_pattern_correlation": spatial_pattern,
        "climatological_spatial_correlation": climatology,
        "standard_deviation_ratio": _safe_ratio(pred_std, target_std),
        "distribution_wasserstein_1": wasserstein,
        "distribution_ks_statistic": ks,
    }
    for quantile in quantile_values:
        label = _quantile_label(quantile)
        pred_quantile = float(np.quantile(pred_values, quantile))
        target_quantile = float(np.quantile(target_values, quantile))
        metrics[f"prediction_{label}"] = pred_quantile
        metrics[f"target_{label}"] = target_quantile
        metrics[f"quantile_error_{label}"] = pred_quantile - target_quantile

    metrics.update(_gradient_metrics(pred, truth, valid, spatial_axes))
    return metrics


def precipitation_metrics(
    prediction: Any,
    target: Any,
    *,
    mask: Any | None = None,
    wet_day_threshold: float = 1.0,
    extreme_quantiles: Sequence[float] = (0.9, 0.95, 0.99, 0.999),
    minimum_extreme_samples: int = 1,
) -> MetricMapping:
    """Calculate precipitation occurrence, intensity, and tail metrics.

    Values are expected in a common accumulation unit such as ``mm/day``.
    Wet days use ``value >= wet_day_threshold``.  Extreme RMSE and heavy-event
    bias are conditioned on target values at or above each target quantile.
    """

    threshold = float(wet_day_threshold)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("wet_day_threshold must be finite and nonnegative")
    quantiles = _validated_quantiles(extreme_quantiles)
    minimum_samples = int(minimum_extreme_samples)
    if minimum_samples < 1:
        raise ValueError("minimum_extreme_samples must be at least one")

    pred, truth, valid = _paired_arrays(prediction, target, mask)
    pred_values = pred[valid]
    target_values = truth[valid]
    pred_wet = pred_values >= threshold
    target_wet = target_values >= threshold
    true_positive = int(np.count_nonzero(pred_wet & target_wet))
    false_positive = int(np.count_nonzero(pred_wet & ~target_wet))
    false_negative = int(np.count_nonzero(~pred_wet & target_wet))
    true_negative = int(np.count_nonzero(~pred_wet & ~target_wet))
    count = int(pred_values.size)

    pred_wet_intensity = (
        float(np.mean(pred_values[pred_wet], dtype=np.float64))
        if bool(pred_wet.any())
        else float("nan")
    )
    target_wet_intensity = (
        float(np.mean(target_values[target_wet], dtype=np.float64))
        if bool(target_wet.any())
        else float("nan")
    )
    if bool(target_wet.any()):
        target_wet_error = pred_values[target_wet] - target_values[target_wet]
        target_wet_rmse = float(
            np.sqrt(np.mean(np.square(target_wet_error), dtype=np.float64))
        )
    else:
        target_wet_rmse = float("nan")

    metrics: MetricMapping = {
        "valid_count": count,
        "wet_day_threshold": threshold,
        "prediction_dry_day_frequency": float(np.mean(~pred_wet)),
        "target_dry_day_frequency": float(np.mean(~target_wet)),
        "prediction_wet_day_frequency": float(np.mean(pred_wet)),
        "target_wet_day_frequency": float(np.mean(target_wet)),
        "wet_day_true_positives": true_positive,
        "wet_day_false_positives": false_positive,
        "wet_day_false_negatives": false_negative,
        "wet_day_true_negatives": true_negative,
        "wet_day_precision": _safe_ratio(
            true_positive, true_positive + false_positive
        ),
        "wet_day_recall": _safe_ratio(
            true_positive, true_positive + false_negative
        ),
        # False-alarm rate and false-alarm ratio are both reported because the
        # two terms are sometimes used interchangeably in precipitation work.
        "wet_day_false_alarm_rate": _safe_ratio(
            false_positive, false_positive + true_negative
        ),
        "wet_day_false_alarm_ratio": _safe_ratio(
            false_positive, true_positive + false_positive
        ),
        "prediction_mean_wet_day_intensity": pred_wet_intensity,
        "target_mean_wet_day_intensity": target_wet_intensity,
        "mean_wet_day_intensity_bias": pred_wet_intensity
        - target_wet_intensity,
        "target_wet_day_rmse": target_wet_rmse,
        "prediction_maximum": float(np.max(pred_values)),
        "target_maximum": float(np.max(target_values)),
        "maximum_bias": float(np.max(pred_values) - np.max(target_values)),
        "negative_prediction_frequency": float(np.mean(pred_values < 0.0)),
    }

    for quantile in quantiles:
        label = _quantile_label(quantile)
        pred_quantile = float(np.quantile(pred_values, quantile))
        target_quantile = float(np.quantile(target_values, quantile))
        metrics[f"prediction_{label}"] = pred_quantile
        metrics[f"target_{label}"] = target_quantile
        metrics[f"quantile_error_{label}"] = pred_quantile - target_quantile

        extreme = target_values >= target_quantile
        extreme_count = int(np.count_nonzero(extreme))
        metrics[f"extreme_sample_count_{label}"] = extreme_count
        if extreme_count < minimum_samples:
            metrics[f"heavy_precipitation_bias_{label}"] = float("nan")
            metrics[f"extreme_rmse_{label}"] = float("nan")
        else:
            extreme_error = pred_values[extreme] - target_values[extreme]
            metrics[f"heavy_precipitation_bias_{label}"] = float(
                np.mean(extreme_error, dtype=np.float64)
            )
            metrics[f"extreme_rmse_{label}"] = float(
                np.sqrt(np.mean(np.square(extreme_error), dtype=np.float64))
            )
    return metrics


def _threshold_items(
    thresholds: Mapping[str, float] | Sequence[float] | None,
    *,
    kind: str,
) -> list[tuple[str, float]]:
    if thresholds is None:
        return []
    if isinstance(thresholds, Mapping):
        raw_items = list(thresholds.items())
    else:
        raw_items = [
            (
                (
                    ("minus_" if float(value) < 0.0 else "")
                    + f"{abs(float(value)):g}".replace(".", "p")
                ),
                value,
            )
            for value in thresholds
        ]
    result: list[tuple[str, float]] = []
    seen: set[str] = set()
    for raw_label, raw_value in raw_items:
        label = "".join(
            character if character.isalnum() else "_"
            for character in str(raw_label)
        ).strip("_")
        if not label:
            raise ValueError(f"{kind} threshold labels must not be empty")
        if label in seen:
            raise ValueError(
                f"duplicate normalized {kind} threshold label {label!r}"
            )
        value = float(raw_value)
        if not np.isfinite(value):
            raise ValueError(f"{kind} threshold {raw_label!r} must be finite")
        seen.add(label)
        result.append((label, value))
    return result


def temperature_metrics(
    prediction: Any,
    target: Any,
    *,
    mask: Any | None = None,
    quantiles: Sequence[float] = (0.01, 0.05, 0.95, 0.99),
    lower_thresholds: Mapping[str, float] | Sequence[float] | None = None,
    upper_thresholds: Mapping[str, float] | Sequence[float] | None = None,
) -> MetricMapping:
    """Calculate cold/warm tail and threshold metrics for one temperature."""

    quantile_values = _validated_quantiles(quantiles)
    pred, truth, valid = _paired_arrays(prediction, target, mask)
    pred_values = pred[valid]
    target_values = truth[valid]
    metrics: MetricMapping = {"valid_count": int(pred_values.size)}

    for quantile in quantile_values:
        label = _quantile_label(quantile)
        pred_quantile = float(np.quantile(pred_values, quantile))
        target_quantile = float(np.quantile(target_values, quantile))
        metrics[f"prediction_{label}"] = pred_quantile
        metrics[f"target_{label}"] = target_quantile
        metrics[f"quantile_error_{label}"] = pred_quantile - target_quantile
        if quantile < 0.5:
            extreme = target_values <= target_quantile
            prefix = "cold_extreme_rmse"
        elif quantile > 0.5:
            extreme = target_values >= target_quantile
            prefix = "warm_extreme_rmse"
        else:
            continue
        extreme_error = pred_values[extreme] - target_values[extreme]
        metrics[f"extreme_sample_count_{label}"] = int(extreme_error.size)
        metrics[f"{prefix}_{label}"] = float(
            np.sqrt(np.mean(np.square(extreme_error), dtype=np.float64))
        )

    for label, threshold in _threshold_items(
        lower_thresholds, kind="lower"
    ):
        pred_event = pred_values <= threshold
        target_event = target_values <= threshold
        metrics[f"lower_threshold_{label}"] = threshold
        metrics[f"prediction_below_{label}_frequency"] = float(
            np.mean(pred_event)
        )
        metrics[f"target_below_{label}_frequency"] = float(
            np.mean(target_event)
        )
        metrics[f"below_{label}_frequency_bias"] = float(
            np.mean(pred_event) - np.mean(target_event)
        )

    for label, threshold in _threshold_items(
        upper_thresholds, kind="upper"
    ):
        pred_event = pred_values >= threshold
        target_event = target_values >= threshold
        metrics[f"upper_threshold_{label}"] = threshold
        metrics[f"prediction_above_{label}_frequency"] = float(
            np.mean(pred_event)
        )
        metrics[f"target_above_{label}_frequency"] = float(
            np.mean(target_event)
        )
        metrics[f"above_{label}_frequency_bias"] = float(
            np.mean(pred_event) - np.mean(target_event)
        )
    return metrics


def _ordering_summary(
    tasmin: np.ndarray,
    tasmax: np.ndarray,
    valid: np.ndarray,
    *,
    prefix: str,
) -> MetricMapping:
    excess = tasmin[valid] - tasmax[valid]
    violation = excess > 0.0
    violation_count = int(np.count_nonzero(violation))
    mean_excess = (
        float(np.mean(excess[violation], dtype=np.float64))
        if violation_count
        else 0.0
    )
    maximum_excess = (
        float(np.max(excess[violation])) if violation_count else 0.0
    )
    return {
        f"{prefix}_valid_count": int(excess.size),
        f"{prefix}_tasmin_gt_tasmax_count": violation_count,
        f"{prefix}_tasmin_gt_tasmax_violation_rate": float(
            np.mean(violation)
        ),
        f"{prefix}_tasmin_gt_tasmax_mean_excess": mean_excess,
        f"{prefix}_tasmin_gt_tasmax_maximum_excess": maximum_excess,
    }


def tasmin_tasmax_metrics(
    predicted_tasmin: Any,
    predicted_tasmax: Any,
    *,
    target_tasmin: Any | None = None,
    target_tasmax: Any | None = None,
    mask: Any | None = None,
) -> MetricMapping:
    """Report ``tasmin > tasmax`` violations for predictions and truth."""

    pred_min = _as_float_array(predicted_tasmin, name="predicted_tasmin")
    pred_max = _as_float_array(predicted_tasmax, name="predicted_tasmax")
    if pred_min.shape != pred_max.shape:
        raise ValueError(
            "predicted tasmin and tasmax must have identical shapes; "
            f"got {pred_min.shape} and {pred_max.shape}"
        )
    base_mask = _broadcast_mask(mask, pred_min.shape)
    if (target_tasmin is None) != (target_tasmax is None):
        raise ValueError(
            "target_tasmin and target_tasmax must be provided together"
        )
    truth_min: np.ndarray | None = None
    truth_max: np.ndarray | None = None
    valid = base_mask & np.isfinite(pred_min) & np.isfinite(pred_max)
    if target_tasmin is not None:
        truth_min = _as_float_array(target_tasmin, name="target_tasmin")
        truth_max = _as_float_array(target_tasmax, name="target_tasmax")
        if (
            truth_min.shape != pred_min.shape
            or truth_max.shape != pred_min.shape
        ):
            raise ValueError(
                "target tasmin/tasmax shapes must match predictions; "
                f"got {truth_min.shape}, {truth_max.shape}, and "
                f"{pred_min.shape}"
            )
        # Rates and their difference must use the same dates/grid cells.
        valid &= np.isfinite(truth_min) & np.isfinite(truth_max)
    if not bool(valid.any()):
        raise ValueError("no finite, unmasked tasmin/tasmax pairs")

    metrics = _ordering_summary(pred_min, pred_max, valid, prefix="prediction")
    # Unprefixed aliases make the common prediction-only diagnostic concise.
    metrics["tasmin_gt_tasmax_count"] = metrics[
        "prediction_tasmin_gt_tasmax_count"
    ]
    metrics["tasmin_gt_tasmax_violation_rate"] = metrics[
        "prediction_tasmin_gt_tasmax_violation_rate"
    ]
    metrics["tasmin_gt_tasmax_mean_excess"] = metrics[
        "prediction_tasmin_gt_tasmax_mean_excess"
    ]

    if truth_min is not None and truth_max is not None:
        metrics.update(
            _ordering_summary(truth_min, truth_max, valid, prefix="target")
        )
        metrics["tasmin_gt_tasmax_violation_rate_bias"] = float(
            metrics["prediction_tasmin_gt_tasmax_violation_rate"]
            - metrics["target_tasmin_gt_tasmax_violation_rate"]
        )
    return metrics


def ensemble_metrics(
    members: Any,
    target: Any,
    *,
    member_axis: int = 0,
    mask: Any | None = None,
    coverage_levels: Sequence[float] = (0.5, 0.8, 0.9, 0.95),
) -> MetricMapping:
    """Calculate empirical ensemble scores for one physical-space variable.

    CRPS uses the exact empirical-ensemble identity

    ``mean(|member - target|) - 0.5 * mean(|member_i - member_j|)``.

    Missing members are omitted independently at each target location.  At
    least one finite member is required.  ``member_axis=1`` matches the
    repository NetCDF layout ``[time, member, lat, lon]``.  Central interval
    coverage error is observed coverage minus the nominal level; the mean
    absolute coverage error summarizes reliability across requested levels.
    """

    ensemble = _as_float_array(members, name="members")
    truth = _as_float_array(target, name="target")
    if ensemble.ndim == 0:
        raise ValueError("members must include an explicit member axis")
    axis = _normalize_axis(member_axis, ensemble.ndim, name="member")
    ensemble = np.moveaxis(ensemble, axis, 0)
    if ensemble.shape[1:] != truth.shape:
        raise ValueError(
            "members with the member axis removed must match target shape; "
            f"got {ensemble.shape[1:]} and {truth.shape}"
        )
    ensemble_size = int(ensemble.shape[0])
    if ensemble_size < 1:
        raise ValueError("ensemble must contain at least one member")

    levels = tuple(float(level) for level in coverage_levels)
    if len(set(levels)) != len(levels):
        raise ValueError("coverage_levels must not contain duplicates")
    if any(
        not np.isfinite(level) or not 0.0 < level < 1.0
        for level in levels
    ):
        raise ValueError(
            "coverage_levels must be finite values strictly in (0, 1)"
        )

    location_mask = _broadcast_mask(mask, truth.shape)
    location_mask &= np.isfinite(truth)
    matrix = ensemble.reshape(ensemble_size, -1)
    target_values = truth.reshape(-1)
    location_mask = location_mask.reshape(-1)
    member_count = np.isfinite(matrix).sum(axis=0, dtype=np.int64)
    location_mask &= member_count > 0
    if not bool(location_mask.any()):
        raise ValueError(
            "no finite, unmasked target locations with ensemble members"
        )

    matrix = matrix[:, location_mask]
    target_values = target_values[location_mask]
    member_finite = np.isfinite(matrix)
    member_count = member_finite.sum(axis=0, dtype=np.int64)
    safe_count = np.maximum(member_count, 1)
    filled = np.where(member_finite, matrix, 0.0)
    ensemble_mean = filled.sum(axis=0, dtype=np.float64) / safe_count
    centered = np.where(member_finite, matrix - ensemble_mean[None, :], 0.0)
    variance_numerator = np.square(centered).sum(axis=0, dtype=np.float64)
    spread_valid = member_count > 1
    ensemble_variance = np.full(member_count.shape, np.nan, dtype=np.float64)
    ensemble_variance[spread_valid] = (
        variance_numerator[spread_valid] / (member_count[spread_valid] - 1)
    )
    point_spread = np.sqrt(ensemble_variance)
    mean_error = ensemble_mean - target_values

    first_crps_term = np.where(
        member_finite,
        np.abs(matrix - target_values[None, :]),
        0.0,
    ).sum(axis=0, dtype=np.float64) / safe_count
    pair_absolute_sum = np.zeros(target_values.shape, dtype=np.float64)
    pair_count = np.zeros(target_values.shape, dtype=np.int64)
    for first in range(ensemble_size):
        for second in range(first + 1, ensemble_size):
            pair_valid = member_finite[first] & member_finite[second]
            pair_absolute_sum[pair_valid] += np.abs(
                matrix[first, pair_valid] - matrix[second, pair_valid]
            )
            pair_count[pair_valid] += 1
    # The 0.5 factor and the two symmetric halves of the full pair matrix
    # cancel, leaving the unordered-pair sum divided by M**2.
    crps = first_crps_term - pair_absolute_sum / np.square(safe_count)
    ensemble_mean_rmse = float(
        np.sqrt(np.mean(np.square(mean_error), dtype=np.float64))
    )
    rms_spread = (
        float(
            np.sqrt(
                np.mean(ensemble_variance[spread_valid], dtype=np.float64)
            )
        )
        if bool(spread_valid.any())
        else float("nan")
    )

    point_diversity = np.full(target_values.shape, np.nan, dtype=np.float64)
    has_pair = pair_count > 0
    point_diversity[has_pair] = (
        pair_absolute_sum[has_pair] / pair_count[has_pair]
    )
    metrics: MetricMapping = {
        "valid_count": int(target_values.size),
        "ensemble_size": ensemble_size,
        "minimum_available_members": int(np.min(member_count)),
        "maximum_available_members": int(np.max(member_count)),
        "mean_available_members": float(np.mean(member_count)),
        "spread_valid_count": int(np.count_nonzero(spread_valid)),
        "ensemble_mean_bias": float(np.mean(mean_error, dtype=np.float64)),
        "ensemble_mean_mae": float(
            np.mean(np.abs(mean_error), dtype=np.float64)
        ),
        "ensemble_mean_rmse": ensemble_mean_rmse,
        "empirical_crps": float(np.mean(crps, dtype=np.float64)),
        "mean_ensemble_spread": _mean_finite(point_spread),
        "rms_ensemble_spread": rms_spread,
        "spread_skill_ratio": _safe_ratio(rms_spread, ensemble_mean_rmse),
        "ensemble_member_diversity": _mean_finite(point_diversity),
    }

    quantile_matrix = np.where(member_finite, matrix, np.nan)
    absolute_coverage_errors: list[float] = []
    for level in levels:
        lower_probability = (1.0 - level) / 2.0
        upper_probability = 1.0 - lower_probability
        lower = np.nanquantile(quantile_matrix, lower_probability, axis=0)
        upper = np.nanquantile(quantile_matrix, upper_probability, axis=0)
        covered = (target_values >= lower) & (target_values <= upper)
        label = _quantile_label(level)
        coverage = float(np.mean(covered))
        coverage_error = coverage - level
        metrics[f"prediction_interval_coverage_{label}"] = coverage
        metrics[f"prediction_interval_coverage_error_{label}"] = (
            coverage_error
        )
        metrics[f"prediction_interval_absolute_coverage_error_{label}"] = (
            abs(coverage_error)
        )
        metrics[f"prediction_interval_mean_width_{label}"] = float(
            np.mean(upper - lower, dtype=np.float64)
        )
        absolute_coverage_errors.append(abs(coverage_error))
    metrics["prediction_interval_mean_absolute_coverage_error"] = (
        float(np.mean(absolute_coverage_errors, dtype=np.float64))
        if absolute_coverage_errors
        else float("nan")
    )
    return metrics


def _broadcast_labels(
    labels: Any,
    shape: tuple[int, ...],
    *,
    sample_axis: int | None,
) -> np.ndarray:
    array = np.asarray(labels)
    # A one-dimensional label vector matching the declared sample dimension is
    # interpreted as sample labels before NumPy's trailing-axis broadcasting.
    # This matters whenever, for example, ``n_time == n_lon`` by coincidence.
    if sample_axis is not None and array.ndim == 1 and shape:
        axis = _normalize_axis(sample_axis, len(shape), name="sample")
        if array.shape[0] == shape[axis]:
            reshaped = [1] * len(shape)
            reshaped[axis] = array.shape[0]
            return np.broadcast_to(array.reshape(reshaped), shape)
    try:
        return np.broadcast_to(array, shape)
    except ValueError:
        pass
    raise ValueError(
        f"strata label shape {array.shape} is not broadcastable to data shape "
        f"{shape}"
    )


def _label_is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = np.isnat(value)
    except (TypeError, ValueError):
        missing = False
    if bool(missing):
        return True
    try:
        missing = np.isnan(value)
    except (TypeError, ValueError):
        missing = False
    return bool(missing)


def strata_masks(
    labels: Any,
    shape: Sequence[int],
    *,
    sample_axis: int | None = 0,
) -> dict[str, np.ndarray]:
    """Broadcast categorical labels and return one boolean mask per label.

    Labels may already have the full data shape, be spatial-only (for example
    elevation bands), or contain one label per sample (for example month or
    season).  Missing labels are omitted.  Label order follows first
    appearance, which avoids silently reordering configured region names.
    """

    data_shape = tuple(int(value) for value in shape)
    if any(value < 0 for value in data_shape):
        raise ValueError(f"shape must be nonnegative, got {data_shape}")
    broadcast = _broadcast_labels(labels, data_shape, sample_axis=sample_axis)
    flat = broadcast.reshape(-1)
    ordered_values: list[Any] = []
    keys: set[str] = set()
    for raw_value in flat:
        value = (
            raw_value.item()
            if isinstance(raw_value, np.generic)
            else raw_value
        )
        if _label_is_missing(value):
            continue
        if any(value == existing for existing in ordered_values):
            continue
        key = str(value)
        if key in keys:
            raise ValueError(
                "distinct strata labels have the same string representation "
                f"{key!r}"
            )
        keys.add(key)
        ordered_values.append(value)
    return {
        str(value): np.asarray(broadcast == value, dtype=bool)
        for value in ordered_values
    }


def stratified_metrics(
    prediction: Any,
    target: Any,
    strata: Mapping[str, Any] | Any,
    *,
    metric: Callable[..., Mapping[str, float | int]] = deterministic_metrics,
    mask: Any | None = None,
    sample_axis: int | None = 0,
    metric_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, MetricMapping]]:
    """Evaluate a metric function independently for categorical strata.

    ``strata`` can be one label array or a mapping such as
    ``{"season": seasons, "elevation": elevation_bands}``.  The label mask
    is intersected with ``mask`` before being passed to ``metric``.  This
    helper intentionally evaluates each stratification independently rather
    than constructing an often-sparse Cartesian product.
    """

    truth = np.asarray(target)
    base_mask = _broadcast_mask(mask, truth.shape)
    if isinstance(strata, Mapping):
        strata_mapping = dict(strata)
    else:
        strata_mapping = {"stratum": strata}
    if not strata_mapping:
        raise ValueError("strata must contain at least one label array")
    kwargs = dict(metric_kwargs or {})
    if "mask" in kwargs:
        raise ValueError("pass mask to stratified_metrics, not metric_kwargs")

    results: dict[str, dict[str, MetricMapping]] = {}
    for raw_name, labels in strata_mapping.items():
        name = str(raw_name)
        masks = strata_masks(labels, truth.shape, sample_axis=sample_axis)
        results[name] = {}
        for label, label_mask in masks.items():
            values = metric(
                prediction,
                target,
                mask=base_mask & label_mask,
                **kwargs,
            )
            results[name][label] = {
                str(key): value for key, value in values.items()
            }
    return results
