"""Deterministic bounded-memory quantiles for large gridded datasets."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class StreamingHistogramQuantile:
    """Accumulate non-negative values into a fixed-resolution histogram.

    PRISM precipitation scalar computation spans billions of finite pixels, so
    concatenating values (or taking a percentile independently per tile/day)
    is neither feasible nor correct.  This accumulator visits every training
    pixel once with constant memory.  Zeros are counted exactly and positive
    quantiles have an absolute discretization error no larger than one bin.
    Values above ``maximum`` remain in the sample count; a requested quantile
    fails clearly only when it actually falls in that overflow tail.
    """

    maximum: float = 512.0
    bins: int = 65536
    counts: np.ndarray = field(init=False, repr=False)
    zero_count: int = field(default=0, init=False)
    overflow_count: int = field(default=0, init=False)
    finite_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.maximum = float(self.maximum)
        self.bins = int(self.bins)
        if not np.isfinite(self.maximum) or self.maximum <= 0.0:
            raise ValueError(f"maximum must be finite and positive, got {self.maximum}")
        if self.bins < 2:
            raise ValueError(f"bins must be at least 2, got {self.bins}")
        self.counts = np.zeros(self.bins, dtype=np.uint64)

    @property
    def bin_width(self) -> float:
        return self.maximum / self.bins

    def update(self, values: object) -> None:
        array = np.asarray(values)
        finite = np.asarray(array[np.isfinite(array)], dtype=np.float64)
        if not finite.size:
            return
        negative = finite < 0.0
        if bool(negative.any()):
            minimum = float(finite[negative].min())
            raise ValueError(
                "Non-negative quantile accumulator received "
                f"{int(negative.sum())} negative value(s), minimum={minimum}"
            )

        self.finite_count += int(finite.size)
        zeros = finite == 0.0
        self.zero_count += int(zeros.sum())
        positive = finite[~zeros]
        if not positive.size:
            return
        overflow = positive > self.maximum
        self.overflow_count += int(overflow.sum())
        in_range = positive[~overflow]
        if in_range.size:
            histogram, _ = np.histogram(
                in_range, bins=self.bins, range=(0.0, self.maximum)
            )
            self.counts += histogram.astype(np.uint64, copy=False)

    def quantile(self, probability: float) -> float:
        probability = float(probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"probability must be in [0,1], got {probability}")
        if self.finite_count == 0:
            raise ValueError("Cannot compute a quantile without finite samples")

        # Match NumPy's default linear-quantile rank. Histogram interpolation is
        # approximate within one bin, while the zero mass remains exact.
        rank = probability * (self.finite_count - 1)
        in_range_count = int(self.counts.sum(dtype=np.uint64))
        if rank >= self.zero_count + in_range_count:
            raise ValueError(
                f"Requested p{100.0 * probability:g} lies in the overflow tail: "
                f"{self.overflow_count}/{self.finite_count} samples exceed "
                f"the histogram maximum {self.maximum}. Increase maximum."
            )

        cumulative = np.cumsum(self.counts, dtype=np.uint64)

        def approximate_order_value(order: int) -> float:
            if order < self.zero_count:
                return 0.0
            positive_order = order - self.zero_count
            if positive_order >= in_range_count:
                raise ValueError(
                    "Quantile interpolation reaches the histogram overflow tail"
                )
            bin_index = int(
                np.searchsorted(cumulative, positive_order, side="right")
            )
            return (bin_index + 0.5) * self.bin_width

        lower_order = int(np.floor(rank))
        upper_order = int(np.ceil(rank))
        lower_value = approximate_order_value(lower_order)
        if lower_order == upper_order:
            return lower_value
        upper_value = approximate_order_value(upper_order)
        fraction = rank - lower_order
        return lower_value + fraction * (upper_value - lower_value)

    def provenance(self) -> dict[str, float | int | str]:
        return {
            "method": "all-training-pixel fixed-range histogram",
            "finite_count": int(self.finite_count),
            "zero_count": int(self.zero_count),
            "overflow_count": int(self.overflow_count),
            "maximum": float(self.maximum),
            "bins": int(self.bins),
            "absolute_error_bound": float(self.bin_width),
        }


def percentile_probability(name: str) -> float:
    """Parse ``p90``/``p95``/``p99`` style scale-stat names."""
    value = str(name).strip().lower()
    if not value.startswith("p"):
        raise ValueError(f"Expected percentile scale_stat such as p95, got {name!r}")
    try:
        percentile = float(value[1:])
    except ValueError as exc:
        raise ValueError(f"Invalid percentile scale_stat {name!r}") from exc
    probability = percentile / 100.0
    if not 0.0 < probability < 1.0:
        raise ValueError(f"Percentile scale_stat must be between p0 and p100, got {name!r}")
    return probability


def resolve_divide_only_scale(
    scale_stat: str,
    *,
    training_mean: float,
    finite_count: int,
    epsilon: float,
    fixed_scale: float | None = None,
    quantile_accumulator: StreamingHistogramQuantile | None = None,
) -> tuple[float, dict[str, float | int | str]]:
    """Resolve a configured divide-only statistic and its provenance."""
    statistic = str(scale_stat).strip().lower()
    if statistic == "mean":
        value = float(training_mean)
        provenance: dict[str, float | int | str] = {
            "method": "all-training-pixel mean",
            "finite_count": int(finite_count),
        }
    elif statistic == "fixed":
        if fixed_scale is None or not np.isfinite(fixed_scale) or fixed_scale <= 0.0:
            raise ValueError("scale_stat=fixed requires a positive finite fixed_scale")
        value = float(fixed_scale)
        provenance = {"method": "configured fixed scale"}
    elif statistic.startswith("p"):
        if quantile_accumulator is None:
            raise ValueError(
                f"scale_stat={statistic} requires a streaming quantile accumulator"
            )
        value = quantile_accumulator.quantile(percentile_probability(statistic))
        provenance = quantile_accumulator.provenance()
    else:
        raise ValueError(f"Unsupported divide_only scale_stat {statistic!r}")

    resolved = max(float(value), float(epsilon))
    return resolved, {
        "scale_stat": statistic,
        "resolved_scale": resolved,
        **provenance,
    }
