#!/usr/bin/env python3
"""Streamed validation report for CORDEX deterministic and diffusion outputs.

The diffusion files produced by the legacy experiment contain full-field ensemble
samples.  Consequently, ``diffusion ensemble mean - U-Net`` is reported here as
an *implied legacy correction*, not as a trained residual-diffusion prediction.

The corrected workflow is selected with ``--experiment residual-correction``.
It compares a physical deterministic-baseline sidecar, the paired physical
corrected ensemble, and ground truth on one exact coordinate intersection.  In
that mode ``corrected member - baseline`` is an actual generated residual.

Example
-------
python residual_validation_report.py \
    --analysis-root D:\\CORDEX_ML_analysis \
    --model ACCESS-CM2 \
    --period 1981-2000

python residual_validation_report.py \
    --experiment residual-correction \
    --analysis-root D:\\CORDEX_ML_analysis \
    --corrected-file D:\\CORDEX_ML_analysis\\ResidualDiffusion_Dataset\\Predictions_pr_tasmax_ACCESS-CM2_1981-2000.nc \
    --truth-file D:\\CORDEX_ML_analysis\\GroundTruth_Dataset\\pr_tasmax_ACCESS-CM2_1981-2000.nc \
    --reference-report evaluations\\residual_validation_ACCESS-CM2_1981-2000.json \
    --plot-dir D:\\CORDEX_ML_analysis\\ResidualDiffusion_Dataset
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import xarray as xr


VARIABLES = ("pr", "tasmax")
SPATIAL_DIMS = ("lat", "lon")
QUANTILE_LEVELS = (0.01, 0.05, 0.50, 0.95, 0.99)
RESIDUAL_DEFINITION = (
    "ground_truth_standardized - deterministic_baseline_standardized"
)
BASELINE_ROLE = "joint_residual_diffusion_deterministic_baseline"


def _json_number(value: float | int) -> float | int | None:
    """Return a strict-JSON scalar, replacing non-finite values with null."""

    if isinstance(value, (int, np.integer)):
        return int(value)
    value = float(value)
    return value if math.isfinite(value) else None


def _normalise_units(units: str) -> str:
    return " ".join(units.strip().lower().replace("**", "^").split())


def _unit_conversion(variable: str, units: str, source_name: str) -> tuple[float, str]:
    """Return a multiplicative conversion into the report's canonical units."""

    normalised = _normalise_units(units)
    if variable == "pr":
        flux_units = {
            "kg m-2 s-1",
            "kg m^-2 s^-1",
            "kg/m2/s",
            "kg m-2 sec-1",
        }
        daily_units = {"mm/day", "mm day-1", "mm d-1", "mm day^-1"}
        if normalised in flux_units:
            return 86400.0, "mm/day"
        if normalised in daily_units:
            return 1.0, "mm/day"
    elif variable == "tasmax" and normalised in {"k", "kelvin"}:
        return 1.0, "K"

    raise ValueError(
        f"Unsupported units {units!r} for {variable!r} in {source_name}. "
        "Refusing to compare fields with an implicit unit conversion."
    )


def _require_unique(values: np.ndarray, label: str) -> None:
    if np.unique(values).size != values.size:
        raise ValueError(f"{label} contains duplicate coordinate values")


def _exact_common_coordinate(
    arrays: Iterable[np.ndarray], labels: Iterable[str], coordinate: str
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Find the exact (not tolerance-based) intersection and source indices."""

    arrays = [np.asarray(array) for array in arrays]
    labels = list(labels)
    for array, label in zip(arrays, labels, strict=True):
        _require_unique(array, f"{label}:{coordinate}")

    common = arrays[0]
    for array in arrays[1:]:
        common = np.intersect1d(common, array, assume_unique=True)
    if common.size == 0:
        raise ValueError(f"No exact common {coordinate} values across {labels}")

    indices: list[np.ndarray] = []
    for array in arrays:
        order = np.argsort(array)
        sorted_array = array[order]
        positions = np.searchsorted(sorted_array, common)
        if np.any(positions >= sorted_array.size) or not np.array_equal(
            sorted_array[positions], common
        ):
            raise RuntimeError(f"Internal {coordinate} alignment failure")
        indices.append(order[positions].astype(np.int64, copy=False))
    return common, indices


def _selector(indices: np.ndarray) -> slice | np.ndarray:
    """Prefer a slice for contiguous indices to keep NetCDF reads efficient."""

    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        return indices
    if indices.size == 1:
        return slice(int(indices[0]), int(indices[0]) + 1)
    if np.all(np.diff(indices) == 1):
        return slice(int(indices[0]), int(indices[-1]) + 1)
    return indices


def _coordinate_sha256(values: np.ndarray) -> str:
    """Hash coordinate values including dtype and shape for sample identity."""

    values = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest().upper()


@dataclass
class RunningMoments:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return

        batch_count = int(values.size)
        batch_mean = float(values.mean(dtype=np.float64))
        deviations = values - batch_mean
        batch_m2 = float(np.dot(deviations, deviations))
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
        else:
            delta = batch_mean - self.mean
            total = self.count + batch_count
            self.m2 += batch_m2 + delta * delta * self.count * batch_count / total
            self.mean += delta * batch_count / total
            self.count = total
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))

    @property
    def std(self) -> float:
        return math.sqrt(self.m2 / self.count) if self.count else math.nan

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean": _json_number(self.mean if self.count else math.nan),
            "std_population": _json_number(self.std),
            "min": _json_number(self.minimum if self.count else math.nan),
            "max": _json_number(self.maximum if self.count else math.nan),
        }


@dataclass
class ErrorMetrics:
    count: int = 0
    sum_error: float = 0.0
    sum_absolute_error: float = 0.0
    sum_squared_error: float = 0.0

    def update(self, prediction: np.ndarray, truth: np.ndarray) -> None:
        prediction = np.asarray(prediction, dtype=np.float64)
        truth = np.asarray(truth, dtype=np.float64)
        valid = np.isfinite(prediction) & np.isfinite(truth)
        if not np.any(valid):
            return
        error = prediction[valid] - truth[valid]
        self.count += int(error.size)
        self.sum_error += float(error.sum(dtype=np.float64))
        self.sum_absolute_error += float(np.abs(error).sum(dtype=np.float64))
        self.sum_squared_error += float(np.dot(error, error))

    def as_dict(self) -> dict[str, Any]:
        if not self.count:
            return {"count": 0, "bias": None, "mae": None, "rmse": None}
        return {
            "count": self.count,
            "bias": self.sum_error / self.count,
            "mae": self.sum_absolute_error / self.count,
            "rmse": math.sqrt(self.sum_squared_error / self.count),
        }


@dataclass
class ClimatologyPair:
    prediction_sum: np.ndarray
    truth_sum: np.ndarray
    count: np.ndarray

    @classmethod
    def create(cls, spatial_shape: tuple[int, int]) -> "ClimatologyPair":
        return cls(
            prediction_sum=np.zeros(spatial_shape, dtype=np.float64),
            truth_sum=np.zeros(spatial_shape, dtype=np.float64),
            count=np.zeros(spatial_shape, dtype=np.int64),
        )

    def update(self, prediction: np.ndarray, truth: np.ndarray) -> None:
        prediction = np.asarray(prediction, dtype=np.float64)
        truth = np.asarray(truth, dtype=np.float64)
        valid = np.isfinite(prediction) & np.isfinite(truth)
        self.prediction_sum += np.where(valid, prediction, 0.0).sum(axis=0)
        self.truth_sum += np.where(valid, truth, 0.0).sum(axis=0)
        self.count += valid.sum(axis=0)

    def as_dict(self) -> dict[str, Any]:
        valid = self.count > 0
        if not np.any(valid):
            return {
                "valid_grid_cells": 0,
                "spatial_correlation": None,
                "mean_absolute_bias": None,
            }
        prediction = self.prediction_sum[valid] / self.count[valid]
        truth = self.truth_sum[valid] / self.count[valid]
        if prediction.size < 2 or prediction.std() == 0.0 or truth.std() == 0.0:
            correlation = math.nan
        else:
            correlation = float(np.corrcoef(prediction, truth)[0, 1])
        return {
            "valid_grid_cells": int(valid.sum()),
            "spatial_correlation": _json_number(correlation),
            "mean_absolute_bias": float(np.abs(prediction - truth).mean()),
        }

    def maps(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return prediction climatology, truth climatology, and their bias."""

        valid = self.count > 0
        prediction = np.full(self.count.shape, np.nan, dtype=np.float64)
        truth = np.full(self.count.shape, np.nan, dtype=np.float64)
        prediction[valid] = self.prediction_sum[valid] / self.count[valid]
        truth[valid] = self.truth_sum[valid] / self.count[valid]
        return prediction, truth, prediction - truth


def _climatology_report(pair: ClimatologyPair) -> dict[str, Any]:
    """Add signed and RMS climatological bias to the legacy summary fields."""

    result = pair.as_dict()
    _, _, bias = pair.maps()
    finite = bias[np.isfinite(bias)]
    result.update(
        {
            "mean_bias": _json_number(finite.mean() if finite.size else math.nan),
            "rmse_bias": _json_number(
                math.sqrt(float(np.mean(np.square(finite))))
                if finite.size
                else math.nan
            ),
        }
    )
    return result


@dataclass
class SystematicQuantileSample:
    total_positions: int
    maximum_samples: int
    seed: int
    values: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.maximum_samples <= 0:
            raise ValueError("maximum_samples must be positive")
        self.stride = max(1, math.ceil(self.total_positions / self.maximum_samples))
        self.offset = self.seed % self.stride

    def update(self, values: np.ndarray, global_start: int) -> None:
        flat = np.asarray(values).reshape(-1)
        global_stop = global_start + flat.size
        first = self.offset
        if first < global_start:
            first += math.ceil((global_start - first) / self.stride) * self.stride
        if first >= global_stop:
            return
        local_indices = np.arange(first, global_stop, self.stride, dtype=np.int64)
        local_indices -= global_start
        sampled = np.asarray(flat[local_indices], dtype=np.float64)
        sampled = sampled[np.isfinite(sampled)]
        if sampled.size:
            self.values.append(sampled)

    def quantiles(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.values:
            sample = np.concatenate(self.values)
        else:
            sample = np.empty(0, dtype=np.float64)
        if sample.size:
            quantiles = np.quantile(sample, QUANTILE_LEVELS)
        else:
            quantiles = np.full(len(QUANTILE_LEVELS), np.nan)
        result = {
            f"p{int(level * 100):02d}": _json_number(value)
            for level, value in zip(QUANTILE_LEVELS, quantiles, strict=True)
        }
        metadata = {
            "method": "deterministic systematic flat-index subsample",
            "seed": self.seed,
            "stride": self.stride,
            "offset": self.offset,
            "finite_sample_count": int(sample.size),
            "maximum_requested_samples": self.maximum_samples,
        }
        return result, metadata


@dataclass
class EnsembleMetrics:
    count: int = 0
    sum_std_unbiased: float = 0.0
    sum_variance_unbiased: float = 0.0
    sum_crps: float = 0.0
    ensemble_mean_errors: ErrorMetrics = field(default_factory=ErrorMetrics)

    def update(self, ensemble: np.ndarray, truth: np.ndarray) -> None:
        """Update from arrays shaped (time, ensemble, lat, lon)."""

        ensemble = np.asarray(ensemble, dtype=np.float64)
        truth = np.asarray(truth, dtype=np.float64)
        members = ensemble.shape[1]
        if members < 2:
            raise ValueError("At least two ensemble members are required for spread metrics")

        valid = np.isfinite(truth) & np.all(np.isfinite(ensemble), axis=1)
        if not np.any(valid):
            return
        member_values = np.moveaxis(ensemble, 1, -1)[valid]
        truth_values = truth[valid]
        ensemble_mean = member_values.mean(axis=1)
        self.ensemble_mean_errors.update(ensemble_mean, truth_values)

        variance = member_values.var(axis=1, ddof=1)
        self.sum_std_unbiased += float(np.sqrt(variance).sum(dtype=np.float64))
        self.sum_variance_unbiased += float(variance.sum(dtype=np.float64))

        first_term = np.abs(member_values - truth_values[:, None]).mean(axis=1)
        pairwise_sum = np.zeros(truth_values.shape, dtype=np.float64)
        for first in range(members):
            for second in range(first + 1, members):
                pairwise_sum += np.abs(member_values[:, first] - member_values[:, second])
        crps = first_term - pairwise_sum / (members * members)
        self.sum_crps += float(crps.sum(dtype=np.float64))
        self.count += int(truth_values.size)

    def as_dict(self, ensemble_size: int) -> dict[str, Any]:
        errors = self.ensemble_mean_errors.as_dict()
        if not self.count:
            return {
                "count": 0,
                "ensemble_size": ensemble_size,
                "mean_spread_unbiased": None,
                "rms_spread_unbiased": None,
                "ensemble_mean_rmse_on_complete_cases": None,
                "spread_skill_ratio": None,
                "crps_empirical_ensemble": None,
            }
        rms_spread = math.sqrt(self.sum_variance_unbiased / self.count)
        rmse = float(errors["rmse"])
        return {
            "count": self.count,
            "ensemble_size": ensemble_size,
            "mean_spread_unbiased": self.sum_std_unbiased / self.count,
            "rms_spread_unbiased": rms_spread,
            "ensemble_mean_rmse_on_complete_cases": rmse,
            "spread_skill_ratio": _json_number(rms_spread / rmse if rmse else math.nan),
            "crps_empirical_ensemble": self.sum_crps / self.count,
        }


@dataclass
class PairedMoments:
    """Streaming bivariate moments for required versus generated corrections."""

    count: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_x2: float = 0.0
    sum_y2: float = 0.0
    sum_xy: float = 0.0

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        valid = np.isfinite(x) & np.isfinite(y)
        if not np.any(valid):
            return
        x = x[valid]
        y = y[valid]
        self.count += int(x.size)
        self.sum_x += float(x.sum(dtype=np.float64))
        self.sum_y += float(y.sum(dtype=np.float64))
        self.sum_x2 += float(np.dot(x, x))
        self.sum_y2 += float(np.dot(y, y))
        self.sum_xy += float(np.dot(x, y))

    def as_dict(self) -> dict[str, Any]:
        if not self.count:
            return {key: None for key in (
                "correlation", "ols_slope", "ols_intercept", "origin_slope",
                "required_mean", "correction_mean", "required_std_population",
                "correction_std_population", "std_correction_to_required_ratio",
                "rms_correction_to_required_ratio",
            )} | {"count": 0}
        n = float(self.count)
        mean_x = self.sum_x / n
        mean_y = self.sum_y / n
        var_x = max(0.0, self.sum_x2 / n - mean_x * mean_x)
        var_y = max(0.0, self.sum_y2 / n - mean_y * mean_y)
        covariance = self.sum_xy / n - mean_x * mean_y
        slope = covariance / var_x if var_x else math.nan
        correlation = covariance / math.sqrt(var_x * var_y) if var_x and var_y else math.nan
        rms_x = math.sqrt(self.sum_x2 / n)
        rms_y = math.sqrt(self.sum_y2 / n)
        result = {
            "count": self.count,
            "correlation": _json_number(correlation),
            "ols_slope": _json_number(slope),
            "ols_intercept": _json_number(mean_y - slope * mean_x),
            "origin_slope": _json_number(self.sum_xy / self.sum_x2 if self.sum_x2 else math.nan),
            "required_mean": mean_x,
            "correction_mean": mean_y,
            "required_std_population": math.sqrt(var_x),
            "correction_std_population": math.sqrt(var_y),
            "std_correction_to_required_ratio": _json_number(math.sqrt(var_y / var_x) if var_x else math.nan),
            "rms_correction_to_required_ratio": _json_number(rms_y / rms_x if rms_x else math.nan),
        }
        result.update(
            {
                "x_mean": mean_x,
                "y_mean": mean_y,
                "x_std_population": math.sqrt(var_x),
                "y_std_population": math.sqrt(var_y),
                "std_y_to_x_ratio": result["std_correction_to_required_ratio"],
                "rms_y_to_x_ratio": result["rms_correction_to_required_ratio"],
            }
        )
        return result


@dataclass
class ImprovementMetrics:
    """Counts cases where a correction improves absolute baseline error."""

    improved: int = 0
    degraded: int = 0
    tied: int = 0

    def update(self, corrected: np.ndarray, baseline: np.ndarray, truth: np.ndarray) -> None:
        corrected = np.asarray(corrected, dtype=np.float64)
        baseline = np.asarray(baseline, dtype=np.float64)
        truth = np.asarray(truth, dtype=np.float64)
        valid = np.isfinite(corrected) & np.isfinite(baseline) & np.isfinite(truth)
        if not np.any(valid):
            return
        corrected_error = np.abs(corrected[valid] - truth[valid])
        baseline_error = np.abs(baseline[valid] - truth[valid])
        self.improved += int(np.count_nonzero(corrected_error < baseline_error))
        self.degraded += int(np.count_nonzero(corrected_error > baseline_error))
        self.tied += int(np.count_nonzero(corrected_error == baseline_error))

    def as_dict(self) -> dict[str, Any]:
        count = self.improved + self.degraded + self.tied
        return {
            "count": count,
            "improved": self.improved,
            "degraded": self.degraded,
            "tied": self.tied,
            "strict_improvement_fraction": _json_number(self.improved / count if count else math.nan),
            "degradation_fraction": _json_number(self.degraded / count if count else math.nan),
        }


@dataclass
class DailySpatialRMSEComparison:
    """Compare corrected and baseline spatial RMSE independently for each day."""

    baseline_rmse: RunningMoments = field(default_factory=RunningMoments)
    corrected_rmse: RunningMoments = field(default_factory=RunningMoments)
    improved_days: int = 0
    degraded_days: int = 0
    tied_days: int = 0

    def update(
        self, corrected: np.ndarray, baseline: np.ndarray, truth: np.ndarray
    ) -> None:
        corrected = np.asarray(corrected, dtype=np.float64)
        baseline = np.asarray(baseline, dtype=np.float64)
        truth = np.asarray(truth, dtype=np.float64)
        if corrected.shape != baseline.shape or baseline.shape != truth.shape:
            raise ValueError("Daily RMSE arrays must have identical shapes")
        if corrected.ndim < 2:
            raise ValueError("Daily RMSE arrays require time plus spatial dimensions")
        spatial_axes = tuple(range(1, corrected.ndim))
        valid = np.isfinite(corrected) & np.isfinite(baseline) & np.isfinite(truth)
        counts = valid.sum(axis=spatial_axes)
        usable = counts > 0
        if not np.any(usable):
            return
        baseline_squared = np.where(valid, np.square(baseline - truth), 0.0)
        corrected_squared = np.where(valid, np.square(corrected - truth), 0.0)
        baseline_rmse = np.sqrt(
            baseline_squared.sum(axis=spatial_axes, dtype=np.float64)[usable]
            / counts[usable]
        )
        corrected_rmse = np.sqrt(
            corrected_squared.sum(axis=spatial_axes, dtype=np.float64)[usable]
            / counts[usable]
        )
        self.baseline_rmse.update(baseline_rmse)
        self.corrected_rmse.update(corrected_rmse)
        self.improved_days += int(np.count_nonzero(corrected_rmse < baseline_rmse))
        self.degraded_days += int(np.count_nonzero(corrected_rmse > baseline_rmse))
        self.tied_days += int(np.count_nonzero(corrected_rmse == baseline_rmse))

    def as_dict(self) -> dict[str, Any]:
        count = self.improved_days + self.degraded_days + self.tied_days
        return {
            "comparable_day_count": count,
            "improved_day_count": self.improved_days,
            "degraded_day_count": self.degraded_days,
            "tied_day_count": self.tied_days,
            "improved_day_fraction": _json_number(
                self.improved_days / count if count else math.nan
            ),
            "degraded_day_fraction": _json_number(
                self.degraded_days / count if count else math.nan
            ),
            "baseline_spatial_rmse_across_days": self.baseline_rmse.as_dict(),
            "corrected_spatial_rmse_across_days": self.corrected_rmse.as_dict(),
        }


@dataclass
class PairedSystematicSample:
    """Deterministically retain matched x/y values for a diagnostic scatter."""

    total_positions: int
    maximum_samples: int
    seed: int
    x_values: list[np.ndarray] = field(default_factory=list)
    y_values: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.maximum_samples <= 0:
            raise ValueError("maximum_samples must be positive")
        self.stride = max(1, math.ceil(self.total_positions / self.maximum_samples))
        self.offset = self.seed % self.stride

    def update(self, x: np.ndarray, y: np.ndarray, global_start: int) -> None:
        x = np.asarray(x).reshape(-1)
        y = np.asarray(y).reshape(-1)
        global_stop = global_start + x.size
        first = self.offset
        if first < global_start:
            first += math.ceil((global_start - first) / self.stride) * self.stride
        if first >= global_stop:
            return
        indices = np.arange(first, global_stop, self.stride, dtype=np.int64) - global_start
        sx = np.asarray(x[indices], dtype=np.float64)
        sy = np.asarray(y[indices], dtype=np.float64)
        valid = np.isfinite(sx) & np.isfinite(sy)
        if np.any(valid):
            self.x_values.append(sx[valid])
            self.y_values.append(sy[valid])

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.x_values:
            return np.empty(0), np.empty(0)
        return np.concatenate(self.x_values), np.concatenate(self.y_values)

    def metadata(self) -> dict[str, Any]:
        count = sum(values.size for values in self.x_values)
        return {
            "method": "deterministic systematic paired flat-index subsample",
            "seed": self.seed,
            "stride": self.stride,
            "offset": self.offset,
            "finite_sample_count": count,
            "maximum_requested_samples": self.maximum_samples,
        }


def _read_field(
    data: xr.DataArray,
    time_indices: np.ndarray,
    lat_indices: np.ndarray,
    lon_indices: np.ndarray,
    ensemble: bool = False,
) -> np.ndarray:
    indexers = {
        "time": _selector(time_indices),
        "lat": _selector(lat_indices),
        "lon": _selector(lon_indices),
    }
    selected = data.isel(indexers)
    dimensions = ("time", "ensemble", "lat", "lon") if ensemble else (
        "time",
        "lat",
        "lon",
    )
    return np.asarray(selected.transpose(*dimensions).values)


def _input_metadata(path: Path, dataset: xr.Dataset) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "head_type": dataset.attrs.get("head_type"),
        "ensemble_generation": dataset.attrs.get("ensemble_generation"),
    }


def _distribution_report(
    moments: RunningMoments,
    quantile_sample: SystematicQuantileSample,
    target_moments: RunningMoments,
    comparison_std: float | None = None,
) -> dict[str, Any]:
    report = moments.as_dict()
    quantiles, sample_metadata = quantile_sample.quantiles()
    report.update(quantiles)
    report["quantile_sample"] = sample_metadata
    report["target_std_population_on_same_valid_cases"] = _json_number(target_moments.std)
    report["std_to_target_std_ratio"] = _json_number(
        moments.std / target_moments.std if target_moments.std else math.nan
    )
    if comparison_std is not None:
        report["std_to_true_residual_std_ratio"] = _json_number(
            moments.std / comparison_std if comparison_std else math.nan
        )
    return report


def validate(
    unet_path: Path,
    diffusion_path: Path,
    truth_path: Path,
    chunk_time: int,
    quantile_samples: int,
    quantile_seed: int,
    progress_every: int,
) -> dict[str, Any]:
    for path in (unet_path, diffusion_path, truth_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if chunk_time <= 0:
        raise ValueError("chunk_time must be positive")

    with (
        xr.open_dataset(unet_path, decode_times=False, cache=False) as unet_ds,
        xr.open_dataset(diffusion_path, decode_times=False, cache=False) as diffusion_ds,
        xr.open_dataset(truth_path, decode_times=False, cache=False) as truth_ds,
    ):
        labels = ("unet", "legacy_diffusion", "ground_truth")
        datasets = (unet_ds, diffusion_ds, truth_ds)
        for label, dataset in zip(labels, datasets, strict=True):
            missing = set(VARIABLES + ("time", "lat", "lon")) - set(dataset.variables)
            if missing:
                raise ValueError(f"{label} is missing variables/coordinates: {sorted(missing)}")
        if "ensemble" not in diffusion_ds.dims:
            raise ValueError("Legacy diffusion file has no ensemble dimension")

        common_time, time_indices = _exact_common_coordinate(
            [dataset["time"].values for dataset in datasets], labels, "time"
        )
        common_lat, lat_indices = _exact_common_coordinate(
            [dataset["lat"].values for dataset in datasets], labels, "lat"
        )
        common_lon, lon_indices = _exact_common_coordinate(
            [dataset["lon"].values for dataset in datasets], labels, "lon"
        )
        spatial_shape = (common_lat.size, common_lon.size)
        total_positions = int(common_time.size * common_lat.size * common_lon.size)
        ensemble_size = int(diffusion_ds.sizes["ensemble"])

        report: dict[str, Any] = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "legacy_diffusion_semantics": (
                "The diffusion file is a legacy full-field ensemble. Its ensemble-mean "
                "minus U-Net difference is an implied correction only; it was not trained "
                "as a residual correction."
            ),
            "inputs": {
                "unet": _input_metadata(unet_path, unet_ds),
                "legacy_diffusion": _input_metadata(diffusion_path, diffusion_ds),
                "ground_truth": _input_metadata(truth_path, truth_ds),
            },
            "alignment": {
                "method": "exact coordinate-value intersection (no tolerance)",
                "common": {
                    "time": int(common_time.size),
                    "lat": int(common_lat.size),
                    "lon": int(common_lon.size),
                    "sample_gridpoint_count": total_positions,
                    "time_first_raw": _json_number(common_time[0]),
                    "time_last_raw": _json_number(common_time[-1]),
                    "lat_min": _json_number(common_lat.min()),
                    "lat_max": _json_number(common_lat.max()),
                    "lon_min": _json_number(common_lon.min()),
                    "lon_max": _json_number(common_lon.max()),
                },
                "source_coordinate_counts": {
                    label: {
                        coordinate: int(dataset.sizes[coordinate])
                        for coordinate in ("time", "lat", "lon")
                    }
                    for label, dataset in zip(labels, datasets, strict=True)
                },
                "dropped_coordinate_values": {
                    label: {
                        "time": int(dataset.sizes["time"] - common_time.size),
                        "lat": int(dataset.sizes["lat"] - common_lat.size),
                        "lon": int(dataset.sizes["lon"] - common_lon.size),
                    }
                    for label, dataset in zip(labels, datasets, strict=True)
                },
            },
            "metric_definitions": {
                "bias": "mean(prediction - ground_truth)",
                "mae": "mean(abs(prediction - ground_truth))",
                "rmse": "sqrt(mean((prediction - ground_truth)^2))",
                "spatial_climatology_correlation": (
                    "Pearson correlation across grid cells between paired-time mean fields"
                ),
                "mean_absolute_climatological_bias": (
                    "spatial mean of abs(prediction time mean - truth time mean)"
                ),
                "true_residual": "ground_truth - U-Net",
                "implied_legacy_correction": "legacy diffusion ensemble mean - U-Net",
                "ensemble_spread": (
                    "sample standard deviation across members (ddof=1); RMS spread is "
                    "sqrt(mean(sample variance))"
                ),
                "spread_skill_ratio": "RMS ensemble spread / RMSE of ensemble mean",
                "crps": (
                    "mean_m |x_m-y| - (1/(2 M^2)) sum_m sum_n |x_m-x_n|"
                ),
            },
            "variables": {},
        }

        for variable_index, variable in enumerate(VARIABLES):
            source_factors: dict[str, float] = {}
            canonical_unit = None
            source_units: dict[str, str] = {}
            for label, dataset in zip(labels, datasets, strict=True):
                units = str(dataset[variable].attrs.get("units", ""))
                factor, resulting_unit = _unit_conversion(variable, units, label)
                source_factors[label] = factor
                source_units[label] = units
                if canonical_unit is None:
                    canonical_unit = resulting_unit
                elif canonical_unit != resulting_unit:
                    raise ValueError(f"Canonical unit mismatch for {variable}")

            unet_errors = ErrorMetrics()
            diffusion_errors = ErrorMetrics()
            unet_climatology = ClimatologyPair.create(spatial_shape)
            diffusion_climatology = ClimatologyPair.create(spatial_shape)
            ensemble_metrics = EnsembleMetrics()

            true_residual_moments = RunningMoments()
            implied_correction_moments = RunningMoments()
            truth_for_true_residual = RunningMoments()
            truth_for_implied_correction = RunningMoments()
            true_residual_quantiles = SystematicQuantileSample(
                total_positions, quantile_samples, quantile_seed + variable_index * 2
            )
            implied_correction_quantiles = SystematicQuantileSample(
                total_positions, quantile_samples, quantile_seed + variable_index * 2 + 1
            )

            chunks = math.ceil(common_time.size / chunk_time)
            for chunk_number, start in enumerate(
                range(0, common_time.size, chunk_time), start=1
            ):
                stop = min(start + chunk_time, common_time.size)
                aligned_slice = slice(start, stop)
                unet = _read_field(
                    unet_ds[variable],
                    time_indices[0][aligned_slice],
                    lat_indices[0],
                    lon_indices[0],
                ).astype(np.float64, copy=False)
                diffusion = _read_field(
                    diffusion_ds[variable],
                    time_indices[1][aligned_slice],
                    lat_indices[1],
                    lon_indices[1],
                    ensemble=True,
                ).astype(np.float64, copy=False)
                truth = _read_field(
                    truth_ds[variable],
                    time_indices[2][aligned_slice],
                    lat_indices[2],
                    lon_indices[2],
                ).astype(np.float64, copy=False)

                unet *= source_factors["unet"]
                diffusion *= source_factors["legacy_diffusion"]
                truth *= source_factors["ground_truth"]
                diffusion_mean = np.mean(diffusion, axis=1)

                unet_errors.update(unet, truth)
                diffusion_errors.update(diffusion_mean, truth)
                unet_climatology.update(unet, truth)
                diffusion_climatology.update(diffusion_mean, truth)
                ensemble_metrics.update(diffusion, truth)

                true_residual = truth - unet
                implied_correction = diffusion_mean - unet
                true_valid = np.isfinite(true_residual) & np.isfinite(truth)
                implied_valid = np.isfinite(implied_correction) & np.isfinite(truth)
                true_residual_moments.update(true_residual[true_valid])
                implied_correction_moments.update(implied_correction[implied_valid])
                truth_for_true_residual.update(truth[true_valid])
                truth_for_implied_correction.update(truth[implied_valid])

                flat_global_start = start * common_lat.size * common_lon.size
                true_residual_quantiles.update(true_residual, flat_global_start)
                implied_correction_quantiles.update(implied_correction, flat_global_start)

                if progress_every and (
                    chunk_number % progress_every == 0 or chunk_number == chunks
                ):
                    print(
                        f"[{variable}] chunk {chunk_number}/{chunks} "
                        f"({stop}/{common_time.size} common times)",
                        file=sys.stderr,
                        flush=True,
                    )

            variable_report = {
                "canonical_unit": canonical_unit,
                "source_units": source_units,
                "conversion_factors_to_canonical_unit": source_factors,
                "unet": {
                    **unet_errors.as_dict(),
                    "climatology": unet_climatology.as_dict(),
                },
                "legacy_diffusion_ensemble_mean": {
                    **diffusion_errors.as_dict(),
                    "climatology": diffusion_climatology.as_dict(),
                },
                "true_residual_ground_truth_minus_unet": _distribution_report(
                    true_residual_moments,
                    true_residual_quantiles,
                    truth_for_true_residual,
                ),
                "implied_legacy_correction_diffusion_mean_minus_unet": _distribution_report(
                    implied_correction_moments,
                    implied_correction_quantiles,
                    truth_for_implied_correction,
                    comparison_std=true_residual_moments.std,
                ),
                "legacy_diffusion_ensemble": ensemble_metrics.as_dict(ensemble_size),
            }
            report["variables"][variable] = variable_report

    return report


def _residual_pair_contract(
    corrected_path: Path,
    corrected_ds: xr.Dataset,
    baseline_path: Path,
    baseline_ds: xr.Dataset,
) -> dict[str, Any]:
    """Validate the sidecar links that make a correction physically meaningful."""

    corrected_baseline = Path(str(corrected_ds.attrs.get("baseline_file", ""))).name
    baseline_corrected = Path(
        str(baseline_ds.attrs.get("paired_corrected_prediction", ""))
    ).name
    checks = {
        "corrected_head_type_is_diffusion": corrected_ds.attrs.get("head_type") == "diffusion",
        "residual_definition_matches": (
            corrected_ds.attrs.get("residual_definition") == RESIDUAL_DEFINITION
        ),
        "corrected_names_baseline_sidecar": corrected_baseline == baseline_path.name,
        "baseline_role_matches": baseline_ds.attrs.get("baseline_role") == BASELINE_ROLE,
        "baseline_names_corrected_file": baseline_corrected == corrected_path.name,
        "baseline_has_no_ensemble_dimension": "ensemble" not in baseline_ds.dims,
        "corrected_has_ensemble_dimension": "ensemble" in corrected_ds.dims,
    }
    return {
        "valid": bool(all(checks.values())),
        "checks": checks,
        "residual_definition": corrected_ds.attrs.get("residual_definition"),
        "baseline_role": baseline_ds.attrs.get("baseline_role"),
        "corrected_baseline_file_attribute": corrected_ds.attrs.get("baseline_file"),
        "baseline_corrected_file_attribute": baseline_ds.attrs.get(
            "paired_corrected_prediction"
        ),
    }


def _residual_unit_conversion(
    variable: str,
    label: str,
    dataset: xr.Dataset,
    contract_valid: bool,
    prediction_pr_units: str,
) -> tuple[float, str, dict[str, Any]]:
    """Resolve numeric units without trusting template attributes on predictions."""

    attribute_units = str(dataset[variable].attrs.get("units", ""))
    if variable == "pr" and label in {"baseline", "corrected"}:
        selected = prediction_pr_units
        if selected == "auto":
            if not contract_valid:
                raise ValueError(
                    "Cannot infer precipitation prediction units: the residual-pair "
                    "metadata contract is incomplete. Supply --prediction-pr-units."
                )
            selected = "mm/day"
            method = "residual physical-output contract"
            evidence = (
                "Paired residual-diffusion inference stores inverse-transformed physical "
                "precipitation predictions in mm/day; NetCDF template units are copied "
                "from the flux target and are therefore not numeric-unit evidence."
            )
        else:
            method = "explicit CLI override"
            evidence = "--prediction-pr-units"
        factor, canonical = _unit_conversion(variable, selected, label)
        return factor, canonical, {
            "attribute_units": attribute_units,
            "numeric_units": selected,
            "method": method,
            "evidence": evidence,
            "attribute_used_for_conversion": False,
        }

    factor, canonical = _unit_conversion(variable, attribute_units, label)
    return factor, canonical, {
        "attribute_units": attribute_units,
        "numeric_units": attribute_units,
        "method": "CF units attribute",
        "evidence": f"{label}:{variable}.attrs['units']",
        "attribute_used_for_conversion": True,
    }


def _map_summary(error: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    values = np.asarray(error, dtype=np.float64)[mask]
    values = values[np.isfinite(values)]
    if not values.size:
        return {"count": 0, "bias": None, "mae": None, "rmse": None}
    return {
        "count": int(values.size),
        "bias": float(values.mean()),
        "mae": float(np.abs(values).mean()),
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _side_jump_report(field: np.ndarray) -> dict[str, Any]:
    """Report the outermost-cell jump toward the domain interior on each side."""

    side_values = {
        "south": field[0, 1:-1] - field[1, 1:-1],
        "north": field[-1, 1:-1] - field[-2, 1:-1],
        "west": field[1:-1, 0] - field[1:-1, 1],
        "east": field[1:-1, -1] - field[1:-1, -2],
    }
    return {
        side: {
            "mean_outer_minus_inner": float(values.mean()),
            "mean_absolute_jump": float(np.abs(values).mean()),
            "rms_jump": float(np.sqrt(np.mean(np.square(values)))),
        }
        for side, values in side_values.items()
    }


def _relationship_report(
    moments: PairedMoments, x_definition: str, y_definition: str
) -> dict[str, Any]:
    return {
        "x_definition": x_definition,
        "y_definition": y_definition,
        **moments.as_dict(),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _active_geometry_report(
    spatial_shape: tuple[int, int],
    edge_width: int,
    active_boundary_mode: str,
    padding_mode: str,
    receptive_field_radius: int | None,
    geometry_config: Path | None,
    geometry_evidence_file: Path | None,
    checkpoint_git_commit: str | None,
    checkpoint_padding_source: str | None,
) -> dict[str, Any]:
    """Build explicit geometry provenance without guessing from image patterns."""

    height, width = spatial_shape
    report: dict[str, Any] = {
        "edge_width_cells": edge_width,
        "edge_definition": (
            "minimum grid-cell distance from any outer full-frame boundary "
            "< edge_width_cells"
        ),
        "output_grid_shape": [height, width],
        "active_boundary_mode": active_boundary_mode,
        "active_boundary_mode_source": (
            "explicit --active-boundary-mode CLI assertion for this saved artifact"
        ),
        "padding_mode": padding_mode,
        "padding_mode_source": "explicit --padding-mode CLI assertion",
        "outer_full_frame_padding_boundary_present": active_boundary_mode == "full-frame",
        "internal_tile_boundaries_present": (
            False if active_boundary_mode == "full-frame" else None
        ),
        "internal_stitch_boundaries_present": (
            False if active_boundary_mode == "full-frame" else None
        ),
        "internal_crop_boundaries_present": (
            False if active_boundary_mode == "full-frame" else None
        ),
        "boundary_interpretation": (
            "The audited artifact was evaluated as one full output frame; any visible "
            "rectangle follows the external model-frame/padding boundary, not an "
            "internal tile, overlap, stitch, or crop boundary."
            if active_boundary_mode == "full-frame"
            else "No full-frame claim is made because active boundary mode is not full-frame."
        ),
        "outer_frame_indices": {
            "south_lat_index": 0,
            "north_lat_index": height - 1,
            "west_lon_index": 0,
            "east_lon_index": width - 1,
        },
    }

    if geometry_config is not None:
        import yaml

        config_path = geometry_config.resolve(strict=True)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        data = config.get("data", {})
        input_shape = [
            int(data.get("input_size_lat", 0)),
            int(data.get("input_size_lon", 0)),
        ]
        target_shape = [
            int(data.get("target_size_lat", 0)),
            int(data.get("target_size_lon", 0)),
        ]
        train_crop = [
            int(data.get("train_crop_size_lat", target_shape[0]) or target_shape[0]),
            int(data.get("train_crop_size_lon", target_shape[1]) or target_shape[1]),
        ]
        val_crop = [
            int(data.get("val_crop_size_lat", target_shape[0]) or target_shape[0]),
            int(data.get("val_crop_size_lon", target_shape[1]) or target_shape[1]),
        ]
        model = config.get("model", {})
        downscaling_patch_size = model.get("downscaling_patch_size")
        mask_unit_size = config.get("mask_unit_size")
        unet_upsample_scales = model.get("unet_upsample_scales")
        encoder_decoder_scales = model.get("encoder_decoder_scale_per_stage")
        ratios = [
            target_shape[index] // input_shape[index]
            if input_shape[index] and target_shape[index] % input_shape[index] == 0
            else None
            for index in range(2)
        ]
        factor_supported = (
            target_shape == [height, width]
            and ratios[0] is not None
            and ratios[0] == ratios[1]
        )
        report["configuration_evidence"] = {
            "path": str(config_path),
            "sha256": _sha256_file(config_path),
            "input_low_resolution_shape": input_shape,
            "target_output_shape": target_shape,
            "train_crop_shape": train_crop,
            "validation_crop_shape": val_crop,
            "target_cells_per_low_resolution_cell": ratios,
            "uniform_low_resolution_factor_supported": factor_supported,
            "full_target_crop_supported": (
                train_crop == target_shape and val_crop == target_shape
            ),
            "downscaling_patch_size": downscaling_patch_size,
            "mask_unit_size": mask_unit_size,
            "unet_upsample_scales": unet_upsample_scales,
            "encoder_decoder_scale_per_stage": encoder_decoder_scales,
        }
        if factor_supported:
            report["low_resolution_grid"] = {
                "shape": input_shape,
                "target_cells_per_low_resolution_cell": ratios,
                "description": (
                    f"Configured {input_shape[0]}x{input_shape[1]} low-resolution grid "
                    f"mapped to {height}x{width} output at {ratios[0]}x in each axis."
                ),
            }
        upsample_product = None
        if (
            isinstance(unet_upsample_scales, (list, tuple))
            and unet_upsample_scales
            and all(isinstance(value, int) and value > 0 for value in unet_upsample_scales)
        ):
            upsample_product = math.prod(unet_upsample_scales)
        mask_spacing_supported = (
            isinstance(mask_unit_size, (list, tuple))
            and len(mask_unit_size) >= 2
            and all(isinstance(value, int) and value > 0 for value in mask_unit_size[:2])
            and target_shape == [height, width]
            and height % int(mask_unit_size[0]) == 0
            and width % int(mask_unit_size[1]) == 0
        )
        report["patch_mask_and_upsampling_geometry"] = {
            "downscaling_patch_size": downscaling_patch_size,
            "downscaling_patch_output_spacing_unambiguous": False,
            "downscaling_patch_output_spacing": None,
            "downscaling_patch_interpretation": (
                "The patch size is recorded but not overlaid: predictor regridding, "
                "shallow-feature construction, and decoder upsampling prevent a unique "
                "direct patch-to-output boundary spacing from config alone."
            ),
            "mask_unit_size": mask_unit_size,
            "mask_unit_output_spacing_unambiguous": mask_spacing_supported,
            "mask_unit_output_spacing": (
                [int(mask_unit_size[0]), int(mask_unit_size[1])]
                if mask_spacing_supported
                else None
            ),
            "mask_unit_interpretation": (
                "Top-level mask_unit_size is defined in current backbone pixels; the "
                "configured target/backbone grid and saved output are both 128x128, so "
                "16-cell mask-unit lines are a defensible output-grid overlay."
                if mask_spacing_supported
                else "No output-grid mask-unit spacing is inferred."
            ),
            "unet_upsample_scales": unet_upsample_scales,
            "unet_upsample_product": upsample_product,
            "unet_product_matches_low_resolution_factor": (
                factor_supported and upsample_product == ratios[0]
            ),
            "encoder_decoder_scale_per_stage": encoder_decoder_scales,
        }
        report["defensible_output_grid_overlays"] = {
            "mapped_low_resolution_cell_spacing": ratios if factor_supported else None,
            "mask_unit_spacing": (
                [int(mask_unit_size[0]), int(mask_unit_size[1])]
                if mask_spacing_supported
                else None
            ),
            "downscaling_patch_spacing": None,
        }

    if geometry_evidence_file is not None:
        evidence_path = geometry_evidence_file.resolve(strict=True)
        report["architecture_evidence"] = {
            "path": str(evidence_path),
            "sha256": _sha256_file(evidence_path),
        }

    if checkpoint_git_commit is not None or checkpoint_padding_source is not None:
        if not checkpoint_git_commit or not checkpoint_padding_source:
            raise ValueError(
                "checkpoint_git_commit and checkpoint_padding_source must be supplied together"
            )
        repo_root = Path(__file__).resolve().parents[2]
        source_spec = f"{checkpoint_git_commit}:{checkpoint_padding_source}"
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "show", source_spec],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        source_text = completed.stdout
        three_by_three_lines = [
            line.strip()
            for line in source_text.splitlines()
            if "nn.Conv2d" in line and ", 3" in line
        ]
        all_three_by_three_padding_one = bool(three_by_three_lines) and all(
            "padding=1" in line for line in three_by_three_lines
        )
        explicit_padding_mode = "padding_mode" in source_text
        zero_padding_verified = (
            all_three_by_three_padding_one and not explicit_padding_mode
        )
        if padding_mode == "zero" and not zero_padding_verified:
            raise ValueError(
                "Checkpoint source does not verify the asserted zero-padding geometry"
            )
        report["checkpoint_padding_source_evidence"] = {
            "git_commit": checkpoint_git_commit,
            "source_path": checkpoint_padding_source,
            "git_object": source_spec,
            "source_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest().upper(),
            "detected_3x3_conv_definition_count": len(three_by_three_lines),
            "all_detected_3x3_convs_use_padding_1": all_three_by_three_padding_one,
            "explicit_padding_mode_present": explicit_padding_mode,
            "pytorch_default_padding_mode_when_omitted": "zeros",
            "zero_padding_verified": zero_padding_verified,
            "interpretation": (
                "At the checkpoint commit, every detected 3x3 Conv2d definition uses "
                "padding=1 and no padding_mode is specified; PyTorch therefore uses its "
                "default zero padding."
            ),
        }

    report["receptive_field"] = {
        "radius_output_cells": receptive_field_radius,
        "diameter_output_cells": (
            2 * receptive_field_radius + 1
            if receptive_field_radius is not None
            else None
        ),
    }
    if receptive_field_radius is not None:
        max_distance = min((height - 1) // 2, (width - 1) // 2)
        valid_height = max(0, height - 2 * receptive_field_radius)
        valid_width = max(0, width - 2 * receptive_field_radius)
        padding_independent_empty = (
            padding_mode == "zero"
            and receptive_field_radius > max_distance
        )
        report["receptive_field"].update(
            {
                "maximum_grid_cell_distance_from_outer_frame": max_distance,
                "padding_independent_valid_interior_shape": [valid_height, valid_width],
                "padding_independent_valid_interior_empty": padding_independent_empty,
                "interpretation": (
                    f"With documented zero padding and radius {receptive_field_radius}, "
                    f"the farthest output cell is only {max_distance} cells from an "
                    "outer frame; therefore the padding-independent valid interior is empty."
                    if padding_independent_empty
                    else "No empty-interior conclusion is made for these settings."
                ),
            }
        )
    return report


def _plot_residual_composite(
    variable: str,
    unit: str,
    payload: dict[str, Any],
    output: Path,
    geometry_overlay: bool,
    edge_width: int,
    geometry: dict[str, Any],
) -> None:
    """Write maps and a paired residual scatter in one reproducible figure."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    lon = payload["lon"]
    lat = payload["lat"]
    truth = payload["truth_climatology"]
    baseline = payload["baseline_climatology"]
    corrected = payload["corrected_climatology"]
    baseline_bias = payload["baseline_bias"]
    corrected_bias = payload["corrected_bias"]
    required = payload["required_correction"]
    generated = payload["generated_correction"]
    scatter_x = payload["scatter_x"]
    scatter_y = payload["scatter_y"]

    fig, axes = plt.subplots(3, 3, figsize=(17, 15), constrained_layout=True)
    absolute_fields = (truth, baseline, corrected)
    absolute_vmin = min(
        float(np.nanquantile(values, 0.02)) for values in absolute_fields
    )
    absolute_vmax = max(
        float(np.nanquantile(values, 0.98)) for values in absolute_fields
    )
    if not (
        math.isfinite(absolute_vmin)
        and math.isfinite(absolute_vmax)
        and absolute_vmax > absolute_vmin
    ):
        absolute_vmin = min(float(np.nanmin(values)) for values in absolute_fields)
        absolute_vmax = max(float(np.nanmax(values)) for values in absolute_fields)

    signed_fields = (baseline_bias, required, generated, corrected_bias)
    signed_limit = max(
        float(np.nanquantile(np.abs(values), 0.98)) for values in signed_fields
    )
    if not math.isfinite(signed_limit) or signed_limit == 0.0:
        signed_limit = 1.0

    map_specs = (
        (axes[0, 0], truth, "Ground-truth climatology", "viridis", False),
        (axes[0, 1], baseline, "Deterministic baseline climatology", "viridis", False),
        (axes[0, 2], corrected, "Corrected ensemble-mean climatology", "viridis", False),
        (axes[1, 0], baseline_bias, "Baseline − truth", "RdBu_r", True),
        (axes[1, 1], required, "Required residual: truth − baseline", "RdBu_r", True),
        (axes[1, 2], generated, "Generated residual: corrected mean − baseline", "RdBu_r", True),
        (axes[2, 0], corrected_bias, "Corrected ensemble mean − truth", "RdBu_r", True),
    )
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
    for map_index, (axis, values, title, cmap, diverging) in enumerate(map_specs):
        kwargs: dict[str, Any] = {
            "vmin": -signed_limit if diverging else absolute_vmin,
            "vmax": signed_limit if diverging else absolute_vmax,
        }
        image = axis.imshow(values, origin="lower", extent=extent, cmap=cmap, **kwargs)
        axis.set_title(title)
        axis.set_xlabel("longitude")
        axis.set_ylabel("latitude")
        fig.colorbar(image, ax=axis, shrink=0.78, label=unit)
        if geometry_overlay and edge_width > 0:
            axis.add_patch(
                Rectangle(
                    (extent[0], extent[2]),
                    extent[1] - extent[0],
                    extent[3] - extent[2],
                    fill=False,
                    edgecolor="cyan",
                    linewidth=1.8,
                    label="outer full-frame/padding boundary",
                )
            )
            x0, x1 = float(lon[edge_width]), float(lon[-edge_width - 1])
            y0, y1 = float(lat[edge_width]), float(lat[-edge_width - 1])
            axis.add_patch(
                Rectangle(
                    (x0, y0), x1 - x0, y1 - y0, fill=False,
                    edgecolor="magenta", linewidth=1.4, linestyle="--",
                    label=f"{edge_width}-cell edge-band interior",
                )
            )
            low_resolution = geometry.get("low_resolution_grid", {})
            factors = low_resolution.get("target_cells_per_low_resolution_cell")
            if factors and all(isinstance(value, int) and value > 0 for value in factors):
                for index in range(factors[1], lon.size, factors[1]):
                    position = float((lon[index - 1] + lon[index]) / 2)
                    axis.axvline(
                        position,
                        color="white",
                        alpha=0.22,
                        linewidth=0.45,
                        label=(
                            f"{factors[1]}-cell mapped low-resolution grid"
                            if map_index == 0 and index == factors[1]
                            else None
                        ),
                    )
                for index in range(factors[0], lat.size, factors[0]):
                    position = float((lat[index - 1] + lat[index]) / 2)
                    axis.axhline(position, color="white", alpha=0.22, linewidth=0.45)
            mask_spacing = geometry.get("defensible_output_grid_overlays", {}).get(
                "mask_unit_spacing"
            )
            if mask_spacing and all(
                isinstance(value, int) and value > 0 for value in mask_spacing
            ):
                for index in range(mask_spacing[1], lon.size, mask_spacing[1]):
                    position = float((lon[index - 1] + lon[index]) / 2)
                    axis.axvline(
                        position,
                        color="yellow",
                        alpha=0.72,
                        linewidth=0.75,
                        linestyle=":",
                        label=(
                            f"{mask_spacing[1]}-cell mask-unit grid"
                            if map_index == 0 and index == mask_spacing[1]
                            else None
                        ),
                    )
                for index in range(mask_spacing[0], lat.size, mask_spacing[0]):
                    position = float((lat[index - 1] + lat[index]) / 2)
                    axis.axhline(
                        position,
                        color="yellow",
                        alpha=0.72,
                        linewidth=0.75,
                        linestyle=":",
                    )
            if map_index == 0:
                axis.legend(loc="lower left", fontsize=6)
                boundary_mode = geometry.get("active_boundary_mode", "unknown")
                axis.text(
                    0.01,
                    0.99,
                    f"active mode: {boundary_mode}\nno internal tile/stitch/crop boundaries",
                    transform=axis.transAxes,
                    va="top",
                    ha="left",
                    fontsize=7,
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
                )
                receptive_field = geometry.get("receptive_field", {})
                if receptive_field.get("padding_independent_valid_interior_empty"):
                    axis.text(
                        0.99,
                        0.01,
                        (
                            f"RF radius={receptive_field['radius_output_cells']}: "
                            "padding-independent interior is empty"
                        ),
                        transform=axis.transAxes,
                        va="bottom",
                        ha="right",
                        fontsize=7,
                        color="magenta",
                        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
                    )

    geometry_axis = axes[2, 1]
    geometry_axis.axis("off")
    receptive_field = geometry.get("receptive_field", {})
    low_resolution = geometry.get("low_resolution_grid", {})
    overlays = geometry.get("defensible_output_grid_overlays", {})
    geometry_axis.text(
        0.02,
        0.98,
        "Geometry interpretation\n\n"
        f"Active boundary mode: {geometry.get('active_boundary_mode', 'unknown')}\n"
        "One 128×128 full frame; no tile/stitch/crop boundaries\n"
        f"Padding: {geometry.get('padding_mode', 'unknown')}\n"
        f"Score-network RF: diameter {receptive_field.get('diameter_output_cells', 'unknown')}, "
        f"radius {receptive_field.get('radius_output_cells', 'unknown')}\n"
        f"Padding-independent valid interior empty: "
        f"{receptive_field.get('padding_independent_valid_interior_empty', 'unknown')}\n"
        f"Mapped low-resolution spacing: "
        f"{low_resolution.get('target_cells_per_low_resolution_cell', 'unknown')} cells\n"
        f"Mask-unit spacing: {overlays.get('mask_unit_spacing', 'unknown')} cells\n\n"
        "Cyan: outer full-frame/padding boundary\n"
        f"Magenta dashed: {edge_width}-cell edge-band interior\n"
        "White: mapped low-resolution grid\n"
        "Yellow dotted: mask-unit grid",
        transform=geometry_axis.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        linespacing=1.35,
        bbox={"facecolor": "#f5f5f5", "edgecolor": "#bbbbbb", "pad": 8},
    )

    scatter_axis = axes[2, 2]
    if scatter_x.size:
        scatter_axis.hexbin(scatter_x, scatter_y, gridsize=90, bins="log", mincnt=1)
        bounds = np.nanquantile(np.concatenate([scatter_x, scatter_y]), [0.01, 0.99])
        scatter_axis.plot(bounds, bounds, "k--", linewidth=1, label="ideal y=x")
        fit = payload["residual_fit"]
        slope = fit.get("ols_slope")
        intercept = fit.get("ols_intercept")
        if slope is not None and intercept is not None:
            scatter_axis.plot(
                bounds, intercept + slope * bounds, color="red", linewidth=1.2,
                label=f"OLS slope={slope:.3f}",
            )
        scatter_axis.set_xlim(bounds)
        scatter_axis.set_ylim(bounds)
        scatter_axis.legend(fontsize=8)
    scatter_axis.set_title("Daily generated vs required correction")
    scatter_axis.set_xlabel(f"required correction ({unit})")
    scatter_axis.set_ylabel(f"generated correction ({unit})")
    fig.suptitle(f"Residual-correction audit: {variable}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def _plot_residual_scatter_triptych(
    variable: str,
    unit: str,
    payload: dict[str, Any],
    output: Path,
) -> None:
    """Plot the three sign-explicit residual relationships used in the audit."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    relationships = (
        (
            "generated_vs_required",
            "Generated correction vs required correction",
            "required = truth − baseline",
            "generated = corrected − baseline",
            "identity",
        ),
        (
            "generated_vs_baseline_error",
            "Generated correction vs baseline error",
            "baseline error = baseline − truth",
            "generated = corrected − baseline",
            "negative_identity",
        ),
        (
            "final_vs_original_bias",
            "Final bias vs original baseline bias",
            "original bias = baseline − truth",
            "final bias = corrected − truth",
            "zero",
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.3), constrained_layout=True)
    for axis, (key, title, x_label, y_label, ideal) in zip(
        axes, relationships, strict=True
    ):
        values = payload["scatter_relationships"][key]
        x = values["x"]
        y = values["y"]
        if x.size:
            collection = axis.hexbin(
                x, y, gridsize=90, bins="log", mincnt=1, cmap="viridis"
            )
            fig.colorbar(collection, ax=axis, shrink=0.78, label="log10 count")
            x_bounds = np.nanquantile(x, [0.01, 0.99])
            if ideal == "identity":
                axis.plot(x_bounds, x_bounds, "k--", linewidth=1, label="ideal y=x")
            elif ideal == "negative_identity":
                axis.plot(
                    x_bounds, -x_bounds, "k--", linewidth=1, label="ideal y=−x"
                )
            else:
                axis.axhline(0.0, color="black", linestyle="--", linewidth=1, label="ideal y=0")
            fit = values["fit"]
            slope = fit.get("ols_slope")
            intercept = fit.get("ols_intercept")
            if slope is not None and intercept is not None:
                axis.plot(
                    x_bounds,
                    intercept + slope * x_bounds,
                    color="red",
                    linewidth=1.2,
                    label=(
                        f"OLS slope={slope:.3f}, "
                        f"r={fit.get('correlation', float('nan')):.3f}"
                    ),
                )
            axis.set_xlim(x_bounds)
            axis.legend(fontsize=7)
        axis.set_title(title)
        axis.set_xlabel(f"{x_label} ({unit})")
        axis.set_ylabel(f"{y_label} ({unit})")
        axis.grid(alpha=0.2)
    fig.suptitle(f"Residual sign and bias relationships: {variable}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def validate_residual_correction(
    corrected_path: Path,
    baseline_path: Path,
    truth_path: Path,
    chunk_time: int,
    progress_every: int,
    scatter_samples: int,
    scatter_seed: int,
    edge_width: int,
    prediction_pr_units: str,
    active_boundary_mode: str,
    padding_mode: str,
    receptive_field_radius: int | None,
    geometry_config: Path | None,
    geometry_evidence_file: Path | None,
    checkpoint_git_commit: str | None,
    checkpoint_padding_source: str | None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Audit a paired residual-diffusion ensemble with streaming calculations."""

    for path in (corrected_path, baseline_path, truth_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if chunk_time <= 0:
        raise ValueError("chunk_time must be positive")

    with (
        xr.open_dataset(corrected_path, decode_times=False, cache=False) as corrected_ds,
        xr.open_dataset(baseline_path, decode_times=False, cache=False) as baseline_ds,
        xr.open_dataset(truth_path, decode_times=False, cache=False) as truth_ds,
    ):
        labels = ("corrected", "baseline", "ground_truth")
        datasets = (corrected_ds, baseline_ds, truth_ds)
        for label, dataset in zip(labels, datasets, strict=True):
            missing = set(VARIABLES + ("time", "lat", "lon")) - set(dataset.variables)
            if missing:
                raise ValueError(f"{label} is missing variables/coordinates: {sorted(missing)}")
        if "ensemble" not in corrected_ds.dims:
            raise ValueError("Corrected residual-diffusion file has no ensemble dimension")

        contract = _residual_pair_contract(
            corrected_path, corrected_ds, baseline_path, baseline_ds
        )
        if not contract["valid"]:
            failed = [name for name, passed in contract["checks"].items() if not passed]
            raise ValueError(f"Invalid residual prediction/baseline pair: {failed}")

        common_time, time_indices = _exact_common_coordinate(
            [dataset["time"].values for dataset in datasets], labels, "time"
        )
        common_lat, lat_indices = _exact_common_coordinate(
            [dataset["lat"].values for dataset in datasets], labels, "lat"
        )
        common_lon, lon_indices = _exact_common_coordinate(
            [dataset["lon"].values for dataset in datasets], labels, "lon"
        )
        spatial_shape = (common_lat.size, common_lon.size)
        if edge_width <= 0 or edge_width * 2 >= min(spatial_shape):
            raise ValueError("edge_width must leave a non-empty spatial interior")
        rows, columns = np.indices(spatial_shape)
        edge_distance = np.minimum.reduce(
            [rows, columns, spatial_shape[0] - 1 - rows, spatial_shape[1] - 1 - columns]
        )
        region_masks = {
            "global": np.ones(spatial_shape, dtype=bool),
            f"edge_{edge_width}_cells": edge_distance < edge_width,
            f"interior_from_{edge_width}_cells": edge_distance >= edge_width,
        }
        total_positions = int(common_time.size * common_lat.size * common_lon.size)
        ensemble_size = int(corrected_ds.sizes["ensemble"])
        geometry = _active_geometry_report(
            spatial_shape=spatial_shape,
            edge_width=edge_width,
            active_boundary_mode=active_boundary_mode,
            padding_mode=padding_mode,
            receptive_field_radius=receptive_field_radius,
            geometry_config=geometry_config,
            geometry_evidence_file=geometry_evidence_file,
            checkpoint_git_commit=checkpoint_git_commit,
            checkpoint_padding_source=checkpoint_padding_source,
        )

        report: dict[str, Any] = {
            "schema_version": 3,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "residual-correction",
            "residual_pair_contract": contract,
            "inputs": {
                "corrected": _input_metadata(corrected_path, corrected_ds),
                "baseline": _input_metadata(baseline_path, baseline_ds),
                "ground_truth": _input_metadata(truth_path, truth_ds),
            },
            "alignment": {
                "method": "exact coordinate-value intersection (no tolerance)",
                "common": {
                    "time": int(common_time.size),
                    "lat": int(common_lat.size),
                    "lon": int(common_lon.size),
                    "sample_gridpoint_count": total_positions,
                    "time_sha256": _coordinate_sha256(common_time),
                    "lat_sha256": _coordinate_sha256(common_lat),
                    "lon_sha256": _coordinate_sha256(common_lon),
                    "time_first_raw": _json_number(common_time[0]),
                    "time_last_raw": _json_number(common_time[-1]),
                    "lat_min": _json_number(common_lat.min()),
                    "lat_max": _json_number(common_lat.max()),
                    "lon_min": _json_number(common_lon.min()),
                    "lon_max": _json_number(common_lon.max()),
                },
                "source_coordinate_counts": {
                    label: {dimension: int(dataset.sizes[dimension]) for dimension in SPATIAL_DIMS + ("time",)}
                    for label, dataset in zip(labels, datasets, strict=True)
                },
                "dropped_coordinate_values": {
                    label: {
                        "time": int(dataset.sizes["time"] - common_time.size),
                        "lat": int(dataset.sizes["lat"] - common_lat.size),
                        "lon": int(dataset.sizes["lon"] - common_lon.size),
                    }
                    for label, dataset in zip(labels, datasets, strict=True)
                },
            },
            "geometry": geometry,
            "metric_definitions": {
                "required_correction": "ground_truth - deterministic_baseline",
                "baseline_error_or_original_bias": "deterministic_baseline - ground_truth",
                "negative_baseline_error": "ground_truth - deterministic_baseline",
                "generated_correction": "corrected_prediction - deterministic_baseline",
                "final_bias": "corrected_prediction - ground_truth",
                "residual_correlation": "Pearson correlation over paired daily gridpoint corrections",
                "ols_slope": "generated_correction = intercept + slope * required_correction",
                "strict_improvement_fraction": "fraction with abs(corrected-truth) < abs(baseline-truth)",
                "daily_rmse": "sqrt(mean((prediction-ground_truth)^2)) over all daily gridpoints",
                "per_day_spatial_rmse": (
                    "for each time independently, sqrt(spatial mean((prediction-ground_truth)^2))"
                ),
                "improved_day_fraction": (
                    "fraction of common days whose corrected spatial RMSE is strictly less "
                    "than paired-baseline spatial RMSE"
                ),
                "correction_ratio": "generated correction magnitude divided by required correction magnitude",
                "ensemble_spread": "sample standard deviation across members (ddof=1)",
            },
            "variables": {},
        }
        plot_payloads: dict[str, dict[str, Any]] = {}

        for variable_index, variable in enumerate(VARIABLES):
            factors: dict[str, float] = {}
            unit_resolution: dict[str, Any] = {}
            canonical_unit: str | None = None
            for label, dataset in zip(labels, datasets, strict=True):
                factor, resulting_unit, resolution = _residual_unit_conversion(
                    variable,
                    "ground_truth" if label == "ground_truth" else label,
                    dataset,
                    contract["valid"],
                    prediction_pr_units,
                )
                factors[label] = factor
                unit_resolution[label] = resolution
                if canonical_unit is None:
                    canonical_unit = resulting_unit
                elif resulting_unit != canonical_unit:
                    raise ValueError(f"Canonical unit mismatch for {variable}")

            baseline_errors = {name: ErrorMetrics() for name in region_masks}
            mean_errors = {name: ErrorMetrics() for name in region_masks}
            member_errors = [
                {name: ErrorMetrics() for name in region_masks}
                for _ in range(ensemble_size)
            ]
            mean_improvements = {name: ImprovementMetrics() for name in region_masks}
            member_improvements = [
                {name: ImprovementMetrics() for name in region_masks}
                for _ in range(ensemble_size)
            ]
            mean_residual_fit = {name: PairedMoments() for name in region_masks}
            member_residual_fit = [
                {name: PairedMoments() for name in region_masks}
                for _ in range(ensemble_size)
            ]
            mean_correction_vs_baseline_error = {
                name: PairedMoments() for name in region_masks
            }
            member_correction_vs_baseline_error = [
                {name: PairedMoments() for name in region_masks}
                for _ in range(ensemble_size)
            ]
            mean_final_vs_original_bias = {
                name: PairedMoments() for name in region_masks
            }
            member_final_vs_original_bias = [
                {name: PairedMoments() for name in region_masks}
                for _ in range(ensemble_size)
            ]
            mean_daily_rmse = DailySpatialRMSEComparison()
            member_daily_rmse = [
                DailySpatialRMSEComparison() for _ in range(ensemble_size)
            ]
            ensemble_metrics = {name: EnsembleMetrics() for name in region_masks}
            baseline_climatology = ClimatologyPair.create(spatial_shape)
            mean_climatology = ClimatologyPair.create(spatial_shape)
            member_climatologies = [
                ClimatologyPair.create(spatial_shape) for _ in range(ensemble_size)
            ]
            relationship_samples = {
                name: PairedSystematicSample(
                    total_positions,
                    scatter_samples,
                    scatter_seed + variable_index,
                )
                for name in (
                    "generated_vs_required",
                    "generated_vs_baseline_error",
                    "final_vs_original_bias",
                )
            }

            chunk_count = math.ceil(common_time.size / chunk_time)
            for chunk_number, start in enumerate(
                range(0, common_time.size, chunk_time), start=1
            ):
                stop = min(start + chunk_time, common_time.size)
                aligned_slice = slice(start, stop)
                corrected = _read_field(
                    corrected_ds[variable],
                    time_indices[0][aligned_slice],
                    lat_indices[0],
                    lon_indices[0],
                    ensemble=True,
                ).astype(np.float64, copy=False)
                baseline = _read_field(
                    baseline_ds[variable],
                    time_indices[1][aligned_slice],
                    lat_indices[1],
                    lon_indices[1],
                ).astype(np.float64, copy=False)
                truth = _read_field(
                    truth_ds[variable],
                    time_indices[2][aligned_slice],
                    lat_indices[2],
                    lon_indices[2],
                ).astype(np.float64, copy=False)
                corrected *= factors["corrected"]
                baseline *= factors["baseline"]
                truth *= factors["ground_truth"]
                corrected_mean = corrected.mean(axis=1)
                required_correction = truth - baseline
                baseline_error = baseline - truth
                generated_mean_correction = corrected_mean - baseline
                final_mean_bias = corrected_mean - truth

                mean_daily_rmse.update(corrected_mean, baseline, truth)
                for member in range(ensemble_size):
                    member_daily_rmse[member].update(
                        corrected[:, member], baseline, truth
                    )

                baseline_climatology.update(baseline, truth)
                mean_climatology.update(corrected_mean, truth)
                for member in range(ensemble_size):
                    member_climatologies[member].update(corrected[:, member], truth)

                for region_name, mask in region_masks.items():
                    regional_truth = truth[:, mask]
                    regional_baseline = baseline[:, mask]
                    regional_corrected = corrected[:, :, mask]
                    regional_mean = corrected_mean[:, mask]
                    regional_required = required_correction[:, mask]
                    regional_baseline_error = baseline_error[:, mask]
                    regional_generated_mean = generated_mean_correction[:, mask]
                    regional_final_mean_bias = final_mean_bias[:, mask]
                    baseline_errors[region_name].update(regional_baseline, regional_truth)
                    mean_errors[region_name].update(regional_mean, regional_truth)
                    mean_improvements[region_name].update(
                        regional_mean, regional_baseline, regional_truth
                    )
                    mean_residual_fit[region_name].update(
                        regional_required,
                        regional_generated_mean,
                    )
                    mean_correction_vs_baseline_error[region_name].update(
                        regional_baseline_error,
                        regional_generated_mean,
                    )
                    mean_final_vs_original_bias[region_name].update(
                        regional_baseline_error,
                        regional_final_mean_bias,
                    )
                    ensemble_metrics[region_name].update(
                        regional_corrected, regional_truth
                    )
                    for member in range(ensemble_size):
                        member_prediction = regional_corrected[:, member]
                        member_errors[member][region_name].update(
                            member_prediction, regional_truth
                        )
                        member_improvements[member][region_name].update(
                            member_prediction, regional_baseline, regional_truth
                        )
                        member_residual_fit[member][region_name].update(
                            regional_required,
                            member_prediction - regional_baseline,
                        )
                        member_correction_vs_baseline_error[member][region_name].update(
                            regional_baseline_error,
                            member_prediction - regional_baseline,
                        )
                        member_final_vs_original_bias[member][region_name].update(
                            regional_baseline_error,
                            member_prediction - regional_truth,
                        )

                flat_global_start = start * common_lat.size * common_lon.size
                relationship_samples["generated_vs_required"].update(
                    required_correction,
                    generated_mean_correction,
                    flat_global_start,
                )
                relationship_samples["generated_vs_baseline_error"].update(
                    baseline_error,
                    generated_mean_correction,
                    flat_global_start,
                )
                relationship_samples["final_vs_original_bias"].update(
                    baseline_error,
                    final_mean_bias,
                    flat_global_start,
                )
                if progress_every and (
                    chunk_number % progress_every == 0 or chunk_number == chunk_count
                ):
                    print(
                        f"[{variable}] chunk {chunk_number}/{chunk_count} "
                        f"({stop}/{common_time.size} common times)",
                        file=sys.stderr,
                        flush=True,
                    )

            baseline_map, truth_map, baseline_bias = baseline_climatology.maps()
            corrected_map, corrected_truth_map, corrected_bias = mean_climatology.maps()
            if not np.allclose(truth_map, corrected_truth_map, equal_nan=True):
                raise RuntimeError("Climatology truth maps disagree across paired metrics")
            required_map = truth_map - baseline_map
            generated_map = corrected_map - baseline_map
            spatial_mean_fit = {name: PairedMoments() for name in region_masks}
            for name, mask in region_masks.items():
                spatial_mean_fit[name].update(required_map[mask], generated_map[mask])

            member_map_reports: dict[str, Any] = {}
            correction_fields = {"ensemble_mean": generated_map}
            corrected_bias_fields = {"ensemble_mean": corrected_bias}
            for member, climatology in enumerate(member_climatologies):
                member_map, member_truth, member_bias = climatology.maps()
                if not np.allclose(truth_map, member_truth, equal_nan=True):
                    raise RuntimeError("Member climatology truth maps disagree")
                member_name = f"member_{member}"
                correction_fields[member_name] = member_map - baseline_map
                corrected_bias_fields[member_name] = member_bias

            max_ring = min(16, int(edge_distance.max()) + 1)
            ring_profiles: dict[str, Any] = {}
            for name, correction_field in correction_fields.items():
                member_index = (
                    int(name.split("_")[1]) if name.startswith("member_") else None
                )
                required_fits = (
                    member_residual_fit[member_index]
                    if member_index is not None
                    else mean_residual_fit
                )
                baseline_error_fits = (
                    member_correction_vs_baseline_error[member_index]
                    if member_index is not None
                    else mean_correction_vs_baseline_error
                )
                final_bias_fits = (
                    member_final_vs_original_bias[member_index]
                    if member_index is not None
                    else mean_final_vs_original_bias
                )
                daily_rmse_comparison = (
                    member_daily_rmse[member_index]
                    if member_index is not None
                    else mean_daily_rmse
                )
                ring_profiles[name] = [
                    {
                        "distance_from_outer_frame_cells": ring,
                        "generated_correction_mean": float(
                            correction_field[edge_distance == ring].mean()
                        ),
                        "corrected_bias_mean": float(
                            corrected_bias_fields[name][edge_distance == ring].mean()
                        ),
                        "corrected_bias_mae": float(
                            np.abs(corrected_bias_fields[name][edge_distance == ring]).mean()
                        ),
                    }
                    for ring in range(max_ring)
                ]
                member_map_reports[name] = {
                    "daily_error": (
                        {region: member_errors[int(name.split("_")[1])][region].as_dict() for region in region_masks}
                        if name.startswith("member_")
                        else {region: mean_errors[region].as_dict() for region in region_masks}
                    ),
                    "improvement_vs_baseline": (
                        {region: member_improvements[int(name.split("_")[1])][region].as_dict() for region in region_masks}
                        if name.startswith("member_")
                        else {region: mean_improvements[region].as_dict() for region in region_masks}
                    ),
                    "residual_fit": (
                        {region: required_fits[region].as_dict() for region in region_masks}
                    ),
                    "correction_relationships": {
                        "generated_vs_required_correction": {
                            region: _relationship_report(
                                required_fits[region],
                                "ground_truth - deterministic_baseline",
                                "corrected_prediction - deterministic_baseline",
                            )
                            for region in region_masks
                        },
                        "generated_vs_baseline_error": {
                            region: _relationship_report(
                                baseline_error_fits[region],
                                "deterministic_baseline - ground_truth",
                                "corrected_prediction - deterministic_baseline",
                            )
                            for region in region_masks
                        },
                        "generated_vs_negative_baseline_error": {
                            region: _relationship_report(
                                required_fits[region],
                                "-(deterministic_baseline - ground_truth)",
                                "corrected_prediction - deterministic_baseline",
                            )
                            for region in region_masks
                        },
                        "final_bias_vs_original_bias": {
                            region: _relationship_report(
                                final_bias_fits[region],
                                "deterministic_baseline - ground_truth",
                                "corrected_prediction - ground_truth",
                            )
                            for region in region_masks
                        },
                    },
                    "per_day_spatial_rmse_comparison": daily_rmse_comparison.as_dict(),
                    "climatological_bias": {
                        region: _map_summary(corrected_bias_fields[name], mask)
                        for region, mask in region_masks.items()
                    },
                    "generated_correction_outer_cell_jump": _side_jump_report(
                        correction_field
                    ),
                }

            scatter_arrays = {
                name: sample.arrays()
                for name, sample in relationship_samples.items()
            }
            scatter_fits = {
                "generated_vs_required": mean_residual_fit["global"].as_dict(),
                "generated_vs_baseline_error": (
                    mean_correction_vs_baseline_error["global"].as_dict()
                ),
                "final_vs_original_bias": mean_final_vs_original_bias["global"].as_dict(),
            }
            variable_report = {
                "canonical_unit": canonical_unit,
                "unit_resolution": unit_resolution,
                "conversion_factors_to_canonical_unit": factors,
                "baseline": {
                    "daily_error": {
                        region: metric.as_dict()
                        for region, metric in baseline_errors.items()
                    },
                    "per_day_spatial_rmse": mean_daily_rmse.baseline_rmse.as_dict(),
                    "climatological_bias": {
                        region: _map_summary(baseline_bias, mask)
                        for region, mask in region_masks.items()
                    },
                },
                "corrected": member_map_reports,
                "ensemble": {
                    region: metric.as_dict(ensemble_size)
                    for region, metric in ensemble_metrics.items()
                },
                "spatial_climatology_residual_fit": {
                    region: metric.as_dict()
                    for region, metric in spatial_mean_fit.items()
                },
                "edge_ring_profiles": ring_profiles,
                "scatter_samples": {
                    name: sample.metadata()
                    for name, sample in relationship_samples.items()
                },
            }
            report["variables"][variable] = variable_report
            plot_payloads[variable] = {
                "lon": common_lon,
                "lat": common_lat,
                "truth_climatology": truth_map,
                "baseline_climatology": baseline_map,
                "corrected_climatology": corrected_map,
                "baseline_bias": baseline_bias,
                "corrected_bias": corrected_bias,
                "required_correction": required_map,
                "generated_correction": generated_map,
                "scatter_x": scatter_arrays["generated_vs_required"][0],
                "scatter_y": scatter_arrays["generated_vs_required"][1],
                "residual_fit": mean_residual_fit["global"].as_dict(),
                "scatter_relationships": {
                    name: {
                        "x": scatter_arrays[name][0],
                        "y": scatter_arrays[name][1],
                        "fit": scatter_fits[name],
                    }
                    for name in scatter_arrays
                },
                "canonical_unit": canonical_unit,
            }

        gate_variables: dict[str, Any] = {}
        gate_reasons: list[str] = []
        for variable in VARIABLES:
            metrics = report["variables"][variable]
            baseline_global = metrics["baseline"]["daily_error"]["global"]
            corrected_global = metrics["corrected"]["ensemble_mean"]["daily_error"]["global"]
            absolute_bias_improved = (
                abs(float(corrected_global["bias"]))
                < abs(float(baseline_global["bias"]))
            )
            rmse_improved = (
                float(corrected_global["rmse"]) < float(baseline_global["rmse"])
            )
            accepted = absolute_bias_improved and rmse_improved
            reasons: list[str] = []
            if not absolute_bias_improved:
                reasons.append(
                    f"absolute bias did not improve: {abs(float(baseline_global['bias'])):.6g} "
                    f"-> {abs(float(corrected_global['bias'])):.6g} {metrics['canonical_unit']}"
                )
            if not rmse_improved:
                reasons.append(
                    f"RMSE did not improve: {float(baseline_global['rmse']):.6g} "
                    f"-> {float(corrected_global['rmse']):.6g} {metrics['canonical_unit']}"
                )
            gate_variables[variable] = {
                "accepted": accepted,
                "absolute_bias_improved": absolute_bias_improved,
                "rmse_improved": rmse_improved,
                "baseline_bias": baseline_global["bias"],
                "corrected_ensemble_mean_bias": corrected_global["bias"],
                "baseline_rmse": baseline_global["rmse"],
                "corrected_ensemble_mean_rmse": corrected_global["rmse"],
                "reasons": reasons,
            }
            gate_reasons.extend(f"{variable}: {reason}" for reason in reasons)
        report["deployment_gate"] = {
            "accepted": bool(all(item["accepted"] for item in gate_variables.values())),
            "candidate": "corrected residual-diffusion ensemble mean",
            "reference": "paired deterministic baseline",
            "required_per_variable": [
                "absolute value of global daily-gridpoint bias strictly improves",
                "global daily-gridpoint RMSE strictly improves",
            ],
            "variables": gate_variables,
            "reasons": gate_reasons,
        }

    return report, plot_payloads


def _default_paths(analysis_root: Path, model: str, period: str) -> tuple[Path, Path, Path]:
    suffix = f"{model}_{period}.nc"
    return (
        analysis_root / "UNet_Dataset" / f"Predictions_pr_tasmax_{suffix}",
        analysis_root / "Diffusion_Dataset" / f"Predictions_pr_tasmax_{suffix}",
        analysis_root / "GroundTruth_Dataset" / f"pr_tasmax_{suffix}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=("legacy", "residual-correction"),
        default="legacy",
        help="Validation contract to apply (default: legacy)",
    )
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--model", default="ACCESS-CM2")
    parser.add_argument("--period", default="1981-2000")
    parser.add_argument("--unet-file", type=Path)
    parser.add_argument("--diffusion-file", type=Path)
    parser.add_argument(
        "--corrected-file",
        type=Path,
        help="Paired corrected residual-diffusion ensemble NetCDF",
    )
    parser.add_argument(
        "--baseline-file",
        type=Path,
        help="Deterministic baseline sidecar (default: corrected baseline_file attribute)",
    )
    parser.add_argument("--truth-file", type=Path)
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--plot-dir", type=Path)
    parser.add_argument(
        "--prediction-pr-units",
        choices=("auto", "mm/day", "kg m-2 s-1"),
        default="auto",
        help="Numeric precipitation units in paired prediction files",
    )
    parser.add_argument("--edge-width", type=int, default=4)
    parser.add_argument(
        "--active-boundary-mode",
        choices=("unknown", "full-frame", "overlap-tiled"),
        default="unknown",
        help="Effective runtime boundary mode for the saved artifact",
    )
    parser.add_argument(
        "--padding-mode",
        choices=("unknown", "zero", "replicate", "reflect", "circular"),
        default="unknown",
        help="Spatial padding used by the audited score network checkpoint",
    )
    parser.add_argument("--receptive-field-radius", type=int)
    parser.add_argument(
        "--geometry-config",
        type=Path,
        help="Resolved training config used to verify input/target/crop geometry",
    )
    parser.add_argument(
        "--geometry-evidence-file",
        type=Path,
        help="Versioned architecture/boundary audit supporting geometry assertions",
    )
    parser.add_argument(
        "--checkpoint-git-commit",
        help="Checkpoint commit used to verify historical padding source",
    )
    parser.add_argument(
        "--checkpoint-padding-source",
        help="Repository-relative score-network source path at checkpoint commit",
    )
    parser.add_argument(
        "--geometry-overlay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay the edge/interior boundary on residual diagnostic maps",
    )
    parser.add_argument("--scatter-samples", type=int, default=200_000)
    parser.add_argument("--scatter-seed", type=int, default=20260716)
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON destination (default: evaluations/residual_validation_MODEL_PERIOD.json)",
    )
    parser.add_argument("--chunk-time", type=int, default=32)
    parser.add_argument("--quantile-samples", type=int, default=1_000_000)
    parser.add_argument("--quantile-seed", type=int, default=20260715)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N chunks; use 0 to disable",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    default_unet, default_diffusion, default_truth = _default_paths(
        args.analysis_root, args.model, args.period
    )
    truth_path = args.truth_file or default_truth
    if args.experiment == "legacy":
        unet_path = args.unet_file or default_unet
        diffusion_path = args.diffusion_file or default_diffusion
        output = args.output or (
            Path(__file__).resolve().parent
            / "evaluations"
            / f"residual_validation_{args.model}_{args.period}.json"
        )
        report = validate(
            unet_path=unet_path,
            diffusion_path=diffusion_path,
            truth_path=truth_path,
            chunk_time=args.chunk_time,
            quantile_samples=args.quantile_samples,
            quantile_seed=args.quantile_seed,
            progress_every=args.progress_every,
        )
    else:
        corrected_path = args.corrected_file or args.diffusion_file or default_diffusion
        if args.baseline_file is not None:
            baseline_path = args.baseline_file
        else:
            with xr.open_dataset(corrected_path, decode_times=False, cache=False) as dataset:
                baseline_attribute = str(dataset.attrs.get("baseline_file", ""))
            if not baseline_attribute:
                raise ValueError(
                    "Corrected file has no baseline_file attribute; supply --baseline-file"
                )
            baseline_candidate = Path(baseline_attribute)
            baseline_path = (
                baseline_candidate
                if baseline_candidate.is_absolute()
                else corrected_path.parent / baseline_candidate.name
            )
        output = args.output or (
            Path(__file__).resolve().parent
            / "evaluations"
            / f"residual_correction_{args.model}_{args.period}.json"
        )
        report, plot_payloads = validate_residual_correction(
            corrected_path=corrected_path,
            baseline_path=baseline_path,
            truth_path=truth_path,
            chunk_time=args.chunk_time,
            progress_every=args.progress_every,
            scatter_samples=args.scatter_samples,
            scatter_seed=args.scatter_seed,
            edge_width=args.edge_width,
            prediction_pr_units=args.prediction_pr_units,
            active_boundary_mode=args.active_boundary_mode,
            padding_mode=args.padding_mode,
            receptive_field_radius=args.receptive_field_radius,
            geometry_config=args.geometry_config,
            geometry_evidence_file=args.geometry_evidence_file,
            checkpoint_git_commit=args.checkpoint_git_commit,
            checkpoint_padding_source=args.checkpoint_padding_source,
        )
        plot_dir = args.plot_dir or output.parent
        plot_paths: dict[str, Any] = {}
        for variable, payload in plot_payloads.items():
            plot_path = plot_dir / f"{variable}_residual_correction_diagnostic.png"
            _plot_residual_composite(
                variable=variable,
                unit=str(payload["canonical_unit"]),
                payload=payload,
                output=plot_path,
                geometry_overlay=args.geometry_overlay,
                edge_width=args.edge_width,
                geometry=report["geometry"],
            )
            scatter_plot_path = (
                plot_dir / f"{variable}_residual_scatter_relationships.png"
            )
            _plot_residual_scatter_triptych(
                variable=variable,
                unit=str(payload["canonical_unit"]),
                payload=payload,
                output=scatter_plot_path,
            )
            plot_paths[variable] = {
                "composite": str(plot_path.resolve()),
                "scatter_relationships": str(scatter_plot_path.resolve()),
            }
        report["plots"] = {
            "geometry_overlay": args.geometry_overlay,
            "files": plot_paths,
        }

        if args.reference_report is not None:
            reference_bytes = args.reference_report.read_bytes()
            reference = json.loads(reference_bytes)
            comparisons: dict[str, Any] = {}
            for variable in VARIABLES:
                try:
                    reference_rmse = float(reference["variables"][variable]["unet"]["rmse"])
                except (KeyError, TypeError, ValueError):
                    continue
                current = report["variables"][variable]
                baseline_rmse = float(current["baseline"]["daily_error"]["global"]["rmse"])
                corrected_rmse = float(
                    current["corrected"]["ensemble_mean"]["daily_error"]["global"]["rmse"]
                )
                comparisons[variable] = {
                    "reference_legacy_unet_rmse": reference_rmse,
                    "paired_baseline_rmse": baseline_rmse,
                    "corrected_ensemble_mean_rmse": corrected_rmse,
                    "paired_baseline_to_reference_rmse_ratio": baseline_rmse / reference_rmse,
                    "corrected_to_reference_rmse_ratio": corrected_rmse / reference_rmse,
                }
            report["reference_report"] = {
                "path": str(args.reference_report.resolve()),
                "sha256": hashlib.sha256(reference_bytes).hexdigest().upper(),
                "comparisons": comparisons,
            }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
