"""Exact pooled extreme metrics under paired whole-year resampling.

Linear empirical quantiles and inclusive observed >= q99 events match the
existing evaluator. Each resampling weight repeats an entire year; no cells,
members, or projected-member compositions are resampled independently.
"""
from __future__ import annotations

import numpy as np

_SIGN = np.uint64(1 << 63)


def _ordered_bits(values):
    bits = np.asarray(values, dtype=np.float64).view(np.uint64)
    return np.where(bits & _SIGN, ~bits, bits ^ _SIGN)


def _ordered_values(bits):
    bits = np.asarray(bits, dtype=np.uint64)
    return np.where(bits & _SIGN, bits ^ _SIGN, ~bits).view(np.float64)


def weighted_year_quantile(sorted_years, year_weights, quantile=.99):
    """Exact NumPy-linear quantile of integer-repeated sorted yearly arrays.

    Search the ordered float64 key space to find replicated order statistics
    without constructing the (potentially hundreds of millions of cells)
    resampled arrays. Vectorization is across resampling replicates.
    """
    arrays = [np.asarray(values, dtype=np.float64) for values in sorted_years]
    weights = np.asarray(year_weights)
    if (weights.ndim != 2 or weights.shape[1] != len(arrays)
            or not np.issubdtype(weights.dtype, np.integer) or np.any(weights < 0)
            or not 0 <= quantile <= 1):
        raise ValueError('Expected nonnegative integer [replicate, year] weights.')
    for values in arrays:
        if values.ndim != 1 or not np.isfinite(values).all() or np.any(values[1:] < values[:-1]):
            raise ValueError('Year arrays must contain sorted finite scalar observations.')
    lengths = np.array([len(values) for values in arrays], dtype=np.int64)
    totals = weights @ lengths
    if np.any(totals <= 0):
        raise ValueError('Every replicate needs at least one valid observation.')
    nonempty = [values for values in arrays if len(values)]
    lowest = min(values[0] for values in nonempty)
    highest = max(values[-1] for values in nonempty)
    position = (totals - 1) * float(quantile)

    def kth(rank):
        low = np.full(len(weights), _ordered_bits(lowest), dtype=np.uint64)
        high = np.full(len(weights), _ordered_bits(highest), dtype=np.uint64)
        while np.any(low < high):
            mid = low + ((high - low) // np.uint64(2))
            threshold = _ordered_values(mid)
            count = np.zeros(len(weights), dtype=np.int64)
            for column, values in enumerate(arrays):
                count += weights[:, column] * np.searchsorted(values, threshold, side='right')
            below = count <= rank
            low = np.where(below & (low < high), mid + np.uint64(1), low)
            high = np.where(~below, mid, high)
        return _ordered_values(low)

    lower = kth(np.floor(position).astype(np.int64))
    upper = kth(np.ceil(position).astype(np.int64))
    fraction = position - np.floor(position)
    return lower + fraction * (upper - lower)


class YearExtremeStatistics:
    """Per-year exact physical ensemble-mean and observed-event distributions."""

    def __init__(self, prediction, baseline, target, years, valid_mask=None):
        prediction, baseline, target = [np.asarray(value, dtype=np.float64)
                                        for value in (prediction, baseline, target)]
        years = np.asarray(years, dtype=np.int64)
        if (prediction.shape != target.shape or baseline.shape != target.shape
                or target.ndim < 2 or years.shape != (len(target),)):
            raise ValueError('Expected paired [time,...spatial] fields and years.')
        valid = np.isfinite(target) & np.isfinite(baseline)
        if valid_mask is not None:
            valid &= np.broadcast_to(np.asarray(valid_mask, dtype=bool), target.shape)
        if np.any(valid & ~np.isfinite(prediction)):
            raise ValueError('Nonfinite candidate cannot shrink the baseline support.')
        self.years = np.unique(years)
        self.sorted = {key: [] for key in ('candidate', 'reference', 'target')}
        self.event_error_suffix = {'candidate': [], 'reference': []}
        for year in self.years:
            mask = valid[years == year]
            p, b, t = [value[years == year][mask] for value in (prediction, baseline, target)]
            order = np.argsort(t, kind='stable')
            self.sorted['target'].append(t[order])
            self.sorted['candidate'].append(np.sort(p))
            self.sorted['reference'].append(np.sort(b))
            for name, value in (('candidate', p), ('reference', b)):
                squared = (value[order] - t[order]) ** 2
                self.event_error_suffix[name].append(
                    np.concatenate((np.cumsum(squared[::-1], dtype=np.float64)[::-1], [0.0])))

    def evaluate(self, year_weights=None):
        weights = (np.ones((1, len(self.years)), dtype=np.int64)
                   if year_weights is None else np.asarray(year_weights))
        target_q = weighted_year_quantile(self.sorted['target'], weights)
        result = {'p99_absolute_error': {}, 'observed_p99_event_rmse': {},
                  'target_p99': target_q}
        event_count = np.zeros(len(weights), dtype=np.int64)
        event_sum = {name: np.zeros(len(weights)) for name in ('candidate', 'reference')}
        for column, values in enumerate(self.sorted['target']):
            # Inclusive event convention, including tied dry/trace observations.
            start = np.searchsorted(values, target_q, side='left')
            event_count += weights[:, column] * (len(values) - start)
            for name in event_sum:
                event_sum[name] += weights[:, column] * self.event_error_suffix[name][column][start]
        if np.any(event_count == 0):
            raise ValueError('Undefined observed-q99 event error: no valid events.')
        for name in event_sum:
            result['p99_absolute_error'][name] = np.abs(
                weighted_year_quantile(self.sorted[name], weights) - target_q)
            result['observed_p99_event_rmse'][name] = np.sqrt(event_sum[name] / event_count)
        return result


def paired_extreme_changes(per_seed, year_weights, seed_indices, floor):
    """Average metric values per independent seed before the paired ratio."""
    if len(per_seed) != seed_indices.shape[1] or floor <= 0:
        raise ValueError('Registered training seeds and positive reference floor required.')
    points = [stats.evaluate() for stats in per_seed]
    draws = [stats.evaluate(year_weights) for stats in per_seed]
    results = {}
    replicate = np.arange(len(year_weights))[:, None]
    for metric in ('p99_absolute_error', 'observed_p99_event_rmse'):
        values = {}
        for product in ('reference', 'candidate'):
            stacked = np.stack([draw[metric][product] for draw in draws], axis=1)
            values[product] = stacked[replicate, seed_indices].mean(axis=1)
        results[metric] = {
            'per_seed': [{'reference': float(point[metric]['reference'][0]),
                          'candidate': float(point[metric]['candidate'][0])} for point in points],
            'bootstrap_changes': (values['candidate'] - values['reference'])
                / np.maximum(np.abs(values['reference']), float(floor)),
        }
    return results
