"""Canonical daily-error and climatological reductions shared by evaluators."""

import math

import numpy as np


class ErrorAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.error_sum = 0.0
        self.absolute_error_sum = 0.0
        self.squared_error_sum = 0.0
        self.prediction_sum = 0.0
        self.target_sum = 0.0

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        valid = np.isfinite(prediction) & np.isfinite(target)
        if not valid.any():
            return
        predicted = np.asarray(prediction[valid], dtype=np.float64)
        observed = np.asarray(target[valid], dtype=np.float64)
        error = predicted - observed
        self.count += int(error.size)
        self.error_sum += float(error.sum())
        self.absolute_error_sum += float(np.abs(error).sum())
        self.squared_error_sum += float(np.square(error).sum())
        self.prediction_sum += float(predicted.sum())
        self.target_sum += float(observed.sum())

    def result(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {
                "valid_count": 0,
                "mean_bias": None,
                "mae": None,
                "rmse": None,
                "prediction_mean": None,
                "target_mean": None,
            }
        count = float(self.count)
        values = error_metrics_from_sums(
            count, self.error_sum, self.absolute_error_sum,
            self.squared_error_sum, self.prediction_sum, self.target_sum,
        )
        return {"valid_count": self.count, **values}


class MapMean:
    def __init__(self, shape: tuple[int, int]) -> None:
        self.sum = np.zeros(shape, dtype=np.float64)
        self.count = np.zeros(shape, dtype=np.int64)

    def update(self, values: np.ndarray) -> None:
        finite = np.isfinite(values)
        self.sum += np.where(finite, values, 0.0).sum(axis=0)
        self.count += finite.sum(axis=0)

    def result(self) -> np.ndarray:
        result = np.full(self.sum.shape, np.nan, dtype=np.float64)
        np.divide(self.sum, self.count, out=result, where=self.count > 0)
        return result


def correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    valid = np.isfinite(first) & np.isfinite(second)
    if valid.sum() < 2:
        return None
    x = np.asarray(first[valid], dtype=np.float64)
    y = np.asarray(second[valid], dtype=np.float64)
    x -= x.mean()
    y -= y.mean()
    denominator = math.sqrt(float(np.square(x).sum() * np.square(y).sum()))
    if denominator <= np.finfo(np.float64).eps:
        return None
    return float(np.dot(x, y) / denominator)


def map_difference_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, float | None]:
    valid = np.isfinite(prediction) & np.isfinite(target)
    if not valid.any():
        return {
            "mean_bias": None,
            "mean_absolute_bias": None,
            "climatological_rmse": None,
            "pattern_correlation": None,
        }
    difference = prediction[valid] - target[valid]
    return {
        "mean_bias": float(difference.mean()),
        "mean_absolute_bias": float(mean_absolute_climatological_bias_from_sums(np.abs(difference).sum(), difference.size)),
        "climatological_rmse": float(np.sqrt(np.square(difference).mean())),
        "pattern_correlation": correlation(prediction, target),
    }



def error_metrics_from_sums(count, signed, absolute, squared, predicted, observed):
    """Canonical scalar or array reduction; input count is strictly positive."""
    return {
        "mean_bias": signed / count,
        "mae": absolute / count,
        "rmse": np.sqrt(squared / count),
        "prediction_mean": predicted / count,
        "target_mean": observed / count,
    }


def mean_absolute_climatological_bias_from_sums(absolute_gridpoint_bias_sum, valid_gridpoints):
    """Equal valid-gridpoint reduction of already formed temporal-mean biases."""
    return absolute_gridpoint_bias_sum / valid_gridpoints
