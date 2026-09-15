"""Evaluation metrics for temporal downscaling, per variable.

Two families, and it matters which one is applicable:

**Event-paired** (:func:`event_paired_metrics`) -- bias, MAE, RMSE, spatial
correlation, tendency error, lagged autocorrelation, spell statistics. These
compare prediction and truth *on the same date* and are only meaningful when the
predictors and targets describe the same physical weather. For the SA case that
was verified empirically (domain-mean ``t_850`` vs ``tasmax`` anomaly correlation
peaks sharply at lag 0), so they apply.

**Distributional** (:func:`distributional_metrics`) -- quantiles, wet-day
frequency, spell-length distributions, seasonal cycle, and climate-change deltas.
These do not assume date-by-date correspondence and are the correct choice when
applying a model to a free-running simulation whose day ``t`` is not the
observed day ``t``.

:func:`evaluate_predictions` refuses to report event-paired scores when the caller
declares the data not event-aligned, rather than computing them and leaving the
reader to notice.

Statistical care:

* Autocorrelation is computed on **deseasonalized** anomalies
  (:func:`deseasonalize`), because raw daily autocorrelation is dominated by the
  annual cycle and would look high even for a model with no day-to-day skill.
* Confidence intervals use a **moving-block bootstrap** over time
  (:func:`block_bootstrap_ci`). An i.i.d. bootstrap over days would treat 7300
  serially correlated days as 7300 independent samples and understate the
  interval by roughly the square root of the decorrelation time.
* Every date is counted once. The caller is responsible for supplying
  de-duplicated series (see
  :meth:`~granitewxc.temporal.sequence_dataset.TemporalSequenceDataset.unique_emission_plan`);
  :func:`assert_unique_dates` checks it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "assert_unique_dates",
    "deseasonalize",
    "block_bootstrap_ci",
    "event_paired_metrics",
    "distributional_metrics",
    "spell_length_stats",
    "wet_dry_transitions",
    "temperature_event_stats",
    "boundary_interior_split",
    "chunk_seam_metric",
    "evaluate_predictions",
]

_EPS = 1.0e-12


def assert_unique_dates(dates: Sequence[Any]) -> None:
    """Raise if any date appears more than once."""
    seen: dict[Any, int] = {}
    for d in dates:
        key = tuple(d) if isinstance(d, (list, tuple, np.ndarray)) else d
        seen[key] = seen.get(key, 0) + 1
    repeats = {k: v for k, v in seen.items() if v > 1}
    if repeats:
        sample = list(repeats.items())[:5]
        raise ValueError(
            f"{len(repeats)} date(s) appear more than once, e.g. {sample}. Overlapping "
            "sequence windows must be de-duplicated before evaluation; counting a date "
            "twice inflates the sample size and shrinks confidence intervals."
        )


def _nan_masked(a: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return a
    out = np.array(a, dtype=np.float64, copy=True)
    out[~mask] = np.nan
    return out


def deseasonalize(
    values: np.ndarray,
    day_of_year: Sequence[int],
    *,
    n_harmonics: int = 3,
) -> np.ndarray:
    """Remove a smooth seasonal cycle along axis 0.

    A harmonic regression (mean + ``n_harmonics`` annual harmonics) is used
    rather than a day-of-year climatology, because a per-calendar-day mean over a
    short record is noisy and would leave residual seasonal structure that
    contaminates the autocorrelation estimate.
    """
    values = np.asarray(values, dtype=np.float64)
    t = np.asarray(day_of_year, dtype=np.float64)
    n = values.shape[0]
    if n < 2 * n_harmonics + 2:
        return values - np.nanmean(values, axis=0, keepdims=True)

    cols = [np.ones(n)]
    for k in range(1, n_harmonics + 1):
        cols.append(np.sin(2 * np.pi * k * t / 365.25))
        cols.append(np.cos(2 * np.pi * k * t / 365.25))
    design = np.stack(cols, axis=1)

    flat = values.reshape(n, -1)
    finite = np.isfinite(flat)
    out = np.empty_like(flat)
    # Solve once for fully-finite columns (the common case) and per-column only
    # where there are gaps, so a few masked cells do not force a slow path.
    all_finite = finite.all(axis=0)
    if all_finite.any():
        beta, *_ = np.linalg.lstsq(design, flat[:, all_finite], rcond=None)
        out[:, all_finite] = flat[:, all_finite] - design @ beta
    for j in np.flatnonzero(~all_finite):
        good = finite[:, j]
        if good.sum() < design.shape[1] + 1:
            out[:, j] = flat[:, j] - np.nanmean(flat[:, j])
            continue
        beta, *_ = np.linalg.lstsq(design[good], flat[good, j], rcond=None)
        out[:, j] = flat[:, j] - design @ beta
    return out.reshape(values.shape)


def block_bootstrap_ci(
    series: np.ndarray,
    statistic,
    *,
    block_length: int = 30,
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    seed: int = 1234,
) -> tuple[float, float, float]:
    """Moving-block bootstrap CI for a statistic of a time series.

    Returns ``(point, lo, hi)``. Blocks of ``block_length`` consecutive days are
    resampled with replacement so within-block serial dependence is preserved;
    ``block_length`` should exceed the decorrelation time of the field (a few
    days for daily temperature).
    """
    series = np.asarray(series, dtype=np.float64)
    n = series.shape[0]
    point = float(statistic(series))
    if n_bootstrap <= 0 or n < 2 * block_length:
        return point, float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_length))
    draws = np.empty(n_bootstrap, dtype=np.float64)
    max_start = n - block_length
    for b in range(n_bootstrap):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_length) for s in starts])[:n]
        draws[b] = statistic(series[idx])
    lo, hi = np.nanpercentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


# ---------------------------------------------------------------------------
# spell / event statistics
# ---------------------------------------------------------------------------
def spell_length_stats(occurrence: np.ndarray) -> dict[str, float]:
    """Mean/max run length of True and False spells along axis 0.

    ``occurrence`` is ``[T, ...]`` boolean. Runs are computed per spatial cell and
    then pooled, so a cell that is permanently dry does not dilute the wet-spell
    statistics of cells that do rain.
    """
    occ = np.asarray(occurrence, dtype=bool)
    t = occ.shape[0]
    flat = occ.reshape(t, -1)
    wet_runs: list[int] = []
    dry_runs: list[int] = []
    for j in range(flat.shape[1]):
        col = flat[:, j]
        if col.size == 0:
            continue
        change = np.flatnonzero(np.diff(col.astype(np.int8)) != 0) + 1
        bounds = np.concatenate([[0], change, [t]])
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            (wet_runs if col[lo] else dry_runs).append(int(hi - lo))
    return {
        "wet_spell_mean": float(np.mean(wet_runs)) if wet_runs else float("nan"),
        "wet_spell_max": float(np.max(wet_runs)) if wet_runs else float("nan"),
        "wet_spell_p90": float(np.percentile(wet_runs, 90)) if wet_runs else float("nan"),
        "dry_spell_mean": float(np.mean(dry_runs)) if dry_runs else float("nan"),
        "dry_spell_max": float(np.max(dry_runs)) if dry_runs else float("nan"),
        "dry_spell_p90": float(np.percentile(dry_runs, 90)) if dry_runs else float("nan"),
    }


def wet_dry_transitions(occurrence: np.ndarray) -> dict[str, float]:
    """Lag-1 transition probabilities ``P(wet|wet)`` and ``P(wet|dry)``.

    These are the statistics a drizzle-everywhere model gets wrong even when its
    marginal wet-day frequency is right, which is why they are reported
    separately from ``wet_day_frequency``.
    """
    occ = np.asarray(occurrence, dtype=bool)
    if occ.shape[0] < 2:
        return {"p_wet_given_wet": float("nan"), "p_wet_given_dry": float("nan")}
    prev, nxt = occ[:-1], occ[1:]
    n_wet = int(prev.sum())
    n_dry = int((~prev).sum())
    return {
        "p_wet_given_wet": float(nxt[prev].mean()) if n_wet else float("nan"),
        "p_wet_given_dry": float(nxt[~prev].mean()) if n_dry else float("nan"),
    }


def temperature_event_stats(
    values: np.ndarray,
    *,
    threshold: float,
    day_of_year: Sequence[int] | None = None,
) -> dict[str, float]:
    """Onset frequency, duration, peak intensity and decay of warm events.

    An event is a run of days above ``threshold`` (an absolute value, typically a
    high percentile of the reference distribution so the same threshold is used
    for prediction and truth).

    ``decay_rate`` is the mean day-over-day drop on the falling side of events;
    together with ``onset_rate`` it distinguishes "gets the number of hot days
    right" from "gets the shape of heat waves right", which a model that simply
    smooths its output can fail while matching the count.
    """
    v = np.asarray(values, dtype=np.float64)
    t = v.shape[0]
    flat = v.reshape(t, -1)
    exceed = flat > threshold

    durations: list[int] = []
    peaks: list[float] = []
    onsets = 0
    rises: list[float] = []
    decays: list[float] = []
    for j in range(flat.shape[1]):
        col = exceed[:, j]
        if not col.any():
            continue
        change = np.flatnonzero(np.diff(col.astype(np.int8)) != 0) + 1
        bounds = np.concatenate([[0], change, [t]])
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            if not col[lo]:
                continue
            onsets += 1
            durations.append(int(hi - lo))
            seg = flat[lo:hi, j]
            peaks.append(float(seg.max()))
            arg = int(np.argmax(seg))
            if arg > 0:
                rises.append(float((seg[arg] - seg[0]) / arg))
            if arg < len(seg) - 1:
                decays.append(float((seg[arg] - seg[-1]) / (len(seg) - 1 - arg)))
    n_cells = max(flat.shape[1], 1)
    return {
        "event_threshold": float(threshold),
        "onset_rate_per_cell_per_day": onsets / (n_cells * max(t, 1)),
        "event_duration_mean": float(np.mean(durations)) if durations else float("nan"),
        "event_duration_p90": float(np.percentile(durations, 90)) if durations else float("nan"),
        "event_peak_mean": float(np.mean(peaks)) if peaks else float("nan"),
        "event_rise_rate": float(np.mean(rises)) if rises else float("nan"),
        "event_decay_rate": float(np.mean(decays)) if decays else float("nan"),
        "exceedance_frequency": float(exceed.mean()),
    }


# ---------------------------------------------------------------------------
# event-paired metrics
# ---------------------------------------------------------------------------
def _autocorr(anom: np.ndarray, lag: int) -> float:
    if anom.shape[0] <= lag:
        return float("nan")
    a, b = anom[lag:], anom[:-lag]
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 8:
        return float("nan")
    a, b = a[good], b[good]
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / max(denom, _EPS))


def event_paired_metrics(
    pred: np.ndarray,
    truth: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    day_of_year: Sequence[int] | None = None,
    lags: Sequence[int] = (1, 2, 3, 5),
    accumulation_windows: Sequence[int] = (3, 5),
    quantiles: Sequence[float] = (0.9, 0.99),
    deseason: bool = True,
) -> dict[str, float]:
    """Date-by-date scores for one variable. ``pred``/``truth`` are ``[T, H, W]``."""
    p = _nan_masked(np.asarray(pred, dtype=np.float64), mask)
    t = _nan_masked(np.asarray(truth, dtype=np.float64), mask)
    diff = p - t
    good = np.isfinite(diff)

    out: dict[str, float] = {
        "n_valid": float(good.sum()),
        "bias": float(np.nanmean(diff)),
        "mae": float(np.nanmean(np.abs(diff))),
        "rmse": float(np.sqrt(np.nanmean(diff**2))),
    }

    # Pooled correlation over all valid (time, space) points.
    pf, tf = p[good], t[good]
    if pf.size > 8:
        pc = pf - pf.mean()
        tc = tf - tf.mean()
        out["pearson_r"] = float(
            (pc * tc).sum() / max(np.sqrt((pc**2).sum() * (tc**2).sum()), _EPS)
        )

    # Per-day spatial correlation, averaged: measures whether the *pattern* of
    # each individual day is right, which pooled correlation can hide.
    daily = []
    for i in range(p.shape[0]):
        a, b = p[i].ravel(), t[i].ravel()
        g = np.isfinite(a) & np.isfinite(b)
        if g.sum() < 16:
            continue
        aa, bb = a[g] - a[g].mean(), b[g] - b[g].mean()
        den = np.sqrt((aa**2).sum() * (bb**2).sum())
        if den > _EPS:
            daily.append(float((aa * bb).sum() / den))
    out["spatial_r_daily_mean"] = float(np.mean(daily)) if daily else float("nan")

    # Day-to-day tendency error: the direct measure of whether evolution is right.
    if p.shape[0] > 1:
        dp, dt = np.diff(p, axis=0), np.diff(t, axis=0)
        de = dp - dt
        out["tendency_mae"] = float(np.nanmean(np.abs(de)))
        out["tendency_rmse"] = float(np.sqrt(np.nanmean(de**2)))
        out["tendency_std_pred"] = float(np.nanstd(dp))
        out["tendency_std_truth"] = float(np.nanstd(dt))
        # A ratio below 1 means the model under-varies day to day, which is the
        # classic symptom of a smoothness-biased objective.
        out["tendency_std_ratio"] = float(
            out["tendency_std_pred"] / max(out["tendency_std_truth"], _EPS)
        )

    # Seasonal-cycle-adjusted autocorrelation of the domain mean.
    doy = list(day_of_year) if day_of_year is not None else list(range(p.shape[0]))
    pm = np.nanmean(p.reshape(p.shape[0], -1), axis=1)
    tm = np.nanmean(t.reshape(t.shape[0], -1), axis=1)
    if deseason:
        pa = deseasonalize(pm[:, None], doy)[:, 0]
        ta = deseasonalize(tm[:, None], doy)[:, 0]
    else:
        pa, ta = pm - np.nanmean(pm), tm - np.nanmean(tm)
    for lag in lags:
        rp, rt = _autocorr(pa, int(lag)), _autocorr(ta, int(lag))
        out[f"autocorr_lag{lag}_pred"] = rp
        out[f"autocorr_lag{lag}_truth"] = rt
        out[f"autocorr_lag{lag}_error"] = float(rp - rt)

    # Multi-day accumulations.
    for w in accumulation_windows:
        w = int(w)
        if p.shape[0] < w:
            continue
        kernel = np.ones(w)
        def roll(x: np.ndarray) -> np.ndarray:
            flat = x.reshape(x.shape[0], -1)
            c = np.cumsum(np.nan_to_num(flat, nan=0.0), axis=0)
            acc = c[w - 1 :] - np.concatenate([np.zeros((1, flat.shape[1])), c[: -w]], axis=0)
            valid = np.cumsum(np.isfinite(flat).astype(np.float64), axis=0)
            vv = valid[w - 1 :] - np.concatenate([np.zeros((1, flat.shape[1])), valid[: -w]], axis=0)
            acc[vv < w - 1e-6] = np.nan
            return acc
        pa_acc, ta_acc = roll(p), roll(t)
        d = pa_acc - ta_acc
        out[f"acc{w}d_bias"] = float(np.nanmean(d))
        out[f"acc{w}d_mae"] = float(np.nanmean(np.abs(d)))
        out[f"acc{w}d_rmse"] = float(np.sqrt(np.nanmean(d**2)))

    # Extremes: quantiles of the pooled distribution.
    for q in quantiles:
        qp = float(np.nanquantile(p, q))
        qt = float(np.nanquantile(t, q))
        out[f"q{q:g}_pred"] = qp
        out[f"q{q:g}_truth"] = qt
        out[f"q{q:g}_bias"] = qp - qt
    out["max_pred"] = float(np.nanmax(p)) if good.any() else float("nan")
    out["max_truth"] = float(np.nanmax(t)) if good.any() else float("nan")
    out["std_pred"] = float(np.nanstd(p))
    out["std_truth"] = float(np.nanstd(t))
    out["std_ratio"] = float(out["std_pred"] / max(out["std_truth"], _EPS))
    return out


def distributional_metrics(
    pred: np.ndarray,
    reference: np.ndarray | None = None,
    *,
    mask: np.ndarray | None = None,
    quantiles: Sequence[float] = (0.01, 0.1, 0.5, 0.9, 0.99),
    wet_threshold: float | None = None,
    lags: Sequence[int] = (1, 2, 3, 5),
    day_of_year: Sequence[int] | None = None,
) -> dict[str, float]:
    """Distribution/persistence summaries that do not assume date pairing.

    Use for a free-running climate application, where day ``t`` of the simulation
    is not day ``t`` of any observation, so bias/RMSE against an observed series
    would be meaningless. When ``reference`` is supplied, the *differences* of
    each summary are also reported.
    """
    p = _nan_masked(np.asarray(pred, dtype=np.float64), mask)
    out: dict[str, float] = {"mean": float(np.nanmean(p)), "std": float(np.nanstd(p))}
    for q in quantiles:
        out[f"q{q:g}"] = float(np.nanquantile(p, q))

    doy = list(day_of_year) if day_of_year is not None else list(range(p.shape[0]))
    pm = np.nanmean(p.reshape(p.shape[0], -1), axis=1)
    pa = deseasonalize(pm[:, None], doy)[:, 0]
    for lag in lags:
        out[f"autocorr_lag{lag}"] = _autocorr(pa, int(lag))

    if wet_threshold is not None:
        occ = p > wet_threshold
        out["wet_day_frequency"] = float(np.nanmean(occ.astype(np.float64)))
        out.update(wet_dry_transitions(occ))
        out.update(spell_length_stats(occ))

    if reference is not None:
        ref = distributional_metrics(
            reference,
            None,
            mask=mask,
            quantiles=quantiles,
            wet_threshold=wet_threshold,
            lags=lags,
            day_of_year=day_of_year,
        )
        for key, value in list(out.items()):
            if key in ref and np.isfinite(value) and np.isfinite(ref[key]):
                out[f"delta_{key}"] = float(value - ref[key])
    return out


# ---------------------------------------------------------------------------
# spatial / seam diagnostics
# ---------------------------------------------------------------------------
def boundary_interior_split(
    pred: np.ndarray,
    truth: np.ndarray,
    *,
    width: int,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    """RMSE separately on the domain boundary band and the interior.

    Precipitation boundary artefacts are a known failure mode in these
    workflows, so a change that improves the interior while degrading the rim
    must be visible rather than averaged away.
    """
    if width <= 0:
        return {}
    p = _nan_masked(np.asarray(pred, dtype=np.float64), mask)
    t = _nan_masked(np.asarray(truth, dtype=np.float64), mask)
    h, w = p.shape[-2:]
    if 2 * width >= min(h, w):
        return {}
    edge = np.ones((h, w), dtype=bool)
    edge[width : h - width, width : w - width] = False
    d = p - t
    return {
        "rmse_boundary": float(np.sqrt(np.nanmean(d[..., edge] ** 2))),
        "rmse_interior": float(np.sqrt(np.nanmean(d[..., ~edge] ** 2))),
        "boundary_width": float(width),
    }


def chunk_seam_metric(
    values: np.ndarray,
    seam_indices: Sequence[int],
    *,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Compare the day-to-day jump at chunk seams with the typical jump.

    A ratio near 1 means chunked inference left no temporal seam. A ratio
    materially above 1 means state was not carried correctly across a chunk
    boundary, which is exactly the artefact chunked inference risks introducing.
    """
    v = _nan_masked(np.asarray(values, dtype=np.float64), mask)
    if v.shape[0] < 2:
        return {}
    jumps = np.abs(np.diff(v, axis=0))
    per_step = np.nanmean(jumps.reshape(jumps.shape[0], -1), axis=1)
    seams = [int(i) - 1 for i in seam_indices if 0 < int(i) < v.shape[0]]
    if not seams:
        return {"seam_jump_ratio": float("nan"), "n_seams": 0.0}
    seam_mean = float(np.nanmean(per_step[seams]))
    others = np.setdiff1d(np.arange(per_step.shape[0]), np.asarray(seams))
    base = float(np.nanmean(per_step[others])) if others.size else float("nan")
    return {
        "seam_jump_mean": seam_mean,
        "nonseam_jump_mean": base,
        "seam_jump_ratio": float(seam_mean / max(base, _EPS)),
        "n_seams": float(len(seams)),
    }


# ---------------------------------------------------------------------------
# top-level
# ---------------------------------------------------------------------------
def evaluate_predictions(
    pred: np.ndarray,
    truth: np.ndarray | None,
    *,
    output_vars: Sequence[str],
    dates: Sequence[Any] | None = None,
    mask: np.ndarray | None = None,
    event_aligned: bool = True,
    wet_threshold: float = 1.0,
    lags: Sequence[int] = (1, 2, 3, 5),
    accumulation_windows: Sequence[int] = (3, 5),
    boundary_width: int = 8,
    seam_indices: Sequence[int] = (),
    temperature_percentile: float = 0.95,
) -> dict[str, Any]:
    """Full per-variable report. ``pred``/``truth`` are ``[T, C, H, W]``.

    When ``event_aligned`` is False, date-paired scores are omitted entirely and
    only distributional summaries are returned, with an explicit note. This is
    the guard against reporting an RMSE between two unrelated realizations of the
    same climate.
    """
    pred = np.asarray(pred)
    if dates is not None:
        assert_unique_dates([tuple(d) if isinstance(d, (list, tuple)) else d for d in dates])
    doy = None
    if dates is not None:
        doy = []
        for d in dates:
            if isinstance(d, (list, tuple)) and len(d) >= 3:
                y, m, day = int(d[0]), int(d[1]), int(d[2])
                cum = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
                doy.append(cum[m - 1] + day)
            else:
                doy.append(len(doy))

    report: dict[str, Any] = {
        "event_aligned": bool(event_aligned),
        "n_dates": int(pred.shape[0]),
        "variables": {},
    }
    if not event_aligned:
        report["note"] = (
            "Predictors and targets are not event-aligned for this configuration, so "
            "date-paired scores (bias/MAE/RMSE/correlation) are not reported: they would "
            "compare two different realizations of the same climate. Distributional, "
            "persistence and extreme-value summaries are reported instead."
        )

    for c, name in enumerate(output_vars):
        var_mask = None if mask is None else mask[:, c]
        p = pred[:, c]
        entry: dict[str, Any] = {}
        t = None if truth is None else np.asarray(truth)[:, c]

        is_precip = str(name).lower() in {"pr", "ppt", "precip", "precipitation", "tp"}
        thr = wet_threshold if is_precip else None

        if event_aligned and t is not None:
            entry["paired"] = event_paired_metrics(
                p,
                t,
                mask=var_mask,
                day_of_year=doy,
                lags=lags,
                accumulation_windows=accumulation_windows,
            )
            entry.update(
                boundary_interior_split(p, t, width=boundary_width, mask=var_mask)
            )

        entry["distribution_pred"] = distributional_metrics(
            p, None, mask=var_mask, wet_threshold=thr, lags=lags, day_of_year=doy
        )
        if t is not None:
            entry["distribution_truth"] = distributional_metrics(
                t, None, mask=var_mask, wet_threshold=thr, lags=lags, day_of_year=doy
            )
            for key, value in entry["distribution_pred"].items():
                ref = entry["distribution_truth"].get(key)
                if ref is not None and np.isfinite(value) and np.isfinite(ref):
                    entry.setdefault("distribution_delta", {})[key] = float(value - ref)

        if not is_precip and t is not None:
            # Use the *truth* percentile as the shared absolute threshold so the
            # comparison is not self-referential.
            thr_t = float(np.nanquantile(_nan_masked(t, var_mask), temperature_percentile))
            entry["events_pred"] = temperature_event_stats(p, threshold=thr_t)
            entry["events_truth"] = temperature_event_stats(t, threshold=thr_t)

        if seam_indices:
            entry["seams"] = chunk_seam_metric(p, seam_indices, mask=var_mask)

        report["variables"][str(name)] = entry
    return report
