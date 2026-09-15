"""Exact whole-year extreme reductions match explicitly repeated fields."""
import numpy as np
import pytest

from granitewxc.refinement.scientific_extremes import (
    YearExtremeStatistics, paired_extreme_changes, weighted_year_quantile,
)


@pytest.mark.parametrize('quantile', [0, .01, .5, .99, 1])
def test_weighted_quantiles_equal_explicit_integer_year_replication(quantile):
    rng = np.random.default_rng(982)
    arrays = [np.sort(rng.normal(size=n)) for n in (0, 7, 13, 22)]
    arrays[2][5:8] = arrays[2][5]  # Include tied observations.
    arrays[2].sort()
    weights = np.array([[0, 1, 1, 1], [4, 0, 2, 1], [0, 3, 0, 0], [1, 0, 0, 2]])
    got = weighted_year_quantile(arrays, weights, quantile)
    expected = [np.quantile(np.concatenate([np.tile(a, w) for a, w in zip(arrays, row)]), quantile)
                for row in weights]
    np.testing.assert_allclose(got, expected, rtol=1e-15, atol=1e-15)


def test_inclusive_events_zeros_mask_and_full_year_fields():
    truth = np.array([[[0., 0., 0.]], [[0., 0., 0.]], [[1., 2., 3.]], [[2., 4., 7.]]])
    baseline = truth + 2
    candidate = truth - 1
    years = np.array([1977, 1977, 2096, 2096])
    valid = np.ones_like(truth, dtype=bool)
    valid[-1, 0, -1] = False
    candidate[-1, 0, -1] = np.nan
    weights = np.array([[1, 1], [2, 0], [0, 2]])
    stats = YearExtremeStatistics(candidate, baseline, truth, years, valid)
    got = stats.evaluate(weights)
    for row, weight in enumerate(weights):
        indices = np.concatenate([np.tile(np.flatnonzero(years == y), w)
                                  for y, w in zip(stats.years, weight)])
        p, b, t = [value[indices][valid[indices]] for value in (candidate, baseline, truth)]
        threshold = np.quantile(t, .99)
        for name, value in (('candidate', p), ('reference', b)):
            assert got['p99_absolute_error'][name][row] == pytest.approx(abs(np.quantile(value, .99)-threshold))
            assert got['observed_p99_event_rmse'][name][row] == pytest.approx(
                np.sqrt(np.mean((value[t >= threshold]-t[t >= threshold])**2)))


def test_training_seed_resampling_does_not_pool_member_or_field_distributions():
    truth = np.arange(24.).reshape(4, 2, 3)
    years = np.array([1977, 1977, 2096, 2096])
    stats = [YearExtremeStatistics(truth+d, truth+4, truth, years) for d in (1., 2., 3.)]
    weights = np.array([[1, 1], [2, 0], [0, 2]])
    seeds = np.array([[0, 1, 2], [2, 2, 2], [1, 0, 0]])
    result = paired_extreme_changes(stats, weights, seeds, .01)
    expected = (np.array([2., 3., 4/3])-4)/4
    for entry in result.values():
        np.testing.assert_allclose(entry['bootstrap_changes'], expected)


def test_nonfinite_candidate_and_invalid_weights_rejected():
    with pytest.raises(ValueError, match='candidate'):
        YearExtremeStatistics([[np.nan]], [[1.]], [[0.]], [1977])
    with pytest.raises(ValueError, match='integer'):
        weighted_year_quantile([[1., 2.]], [[.5]])
