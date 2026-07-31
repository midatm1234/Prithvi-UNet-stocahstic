from __future__ import annotations

import numpy as np
import pytest

from granitewxc.utils.streaming_quantile import (
    StreamingHistogramQuantile,
    percentile_probability,
    resolve_divide_only_scale,
)


def test_streaming_histogram_quantile_uses_all_updates_and_exact_zero_mass() -> None:
    accumulator = StreamingHistogramQuantile(maximum=100.0, bins=10000)
    accumulator.update(np.zeros(900, dtype=np.float32))
    accumulator.update(np.arange(1.0, 101.0, dtype=np.float32))

    assert accumulator.quantile(0.5) == 0.0
    expected = float(np.quantile(np.concatenate([np.zeros(900), np.arange(1, 101)]), 0.95))
    assert accumulator.quantile(0.95) == pytest.approx(
        expected, abs=accumulator.bin_width
    )
    assert accumulator.provenance()["finite_count"] == 1000


def test_streaming_quantile_counts_overflow_in_rank_and_rejects_bad_data() -> None:
    accumulator = StreamingHistogramQuantile(maximum=10.0, bins=1000)
    accumulator.update(np.concatenate([np.arange(10.0), np.full(100, 20.0)]))
    with pytest.raises(ValueError, match="overflow tail"):
        accumulator.quantile(0.95)
    with pytest.raises(ValueError, match="negative"):
        accumulator.update([-0.1])


@pytest.mark.parametrize("name, expected", [("p90", 0.9), ("P95", 0.95), ("p99", 0.99)])
def test_percentile_probability(name: str, expected: float) -> None:
    assert percentile_probability(name) == pytest.approx(expected)


def test_divide_only_p95_resolver_does_not_substitute_training_mean() -> None:
    accumulator = StreamingHistogramQuantile(maximum=20.0, bins=20000)
    values = np.concatenate([np.zeros(500), np.linspace(0.1, 10.0, 500)])
    accumulator.update(values)
    scale, provenance = resolve_divide_only_scale(
        "p95",
        training_mean=float(values.mean()),
        finite_count=values.size,
        epsilon=1.0e-6,
        quantile_accumulator=accumulator,
    )

    assert scale == pytest.approx(np.quantile(values, 0.95), abs=accumulator.bin_width)
    assert scale != pytest.approx(values.mean())
    assert provenance["scale_stat"] == "p95"
