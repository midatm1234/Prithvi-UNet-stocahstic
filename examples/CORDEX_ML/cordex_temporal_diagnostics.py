#!/usr/bin/env python
"""Data diagnostics that must pass BEFORE supervised sequence training.

Three questions, answered from the actual files named in a case YAML:

1. **Is the case event-aligned?** Do the coarse predictors and the
   high-resolution targets on the same date describe the same weather? If the
   lagged cross-correlation does not peak at lag 0, day-by-day losses and
   event-paired metrics are invalid and the case must be treated as a
   distributional application instead. This is the check that decides whether
   ``temporal.evaluation.event_paired`` may be true.

2. **How much does history actually add?** A linear variance decomposition gives
   a defensible lower bound on the headroom a temporal model has beyond the
   same-day predictors, and shows which lags carry it. This is what
   ``temporal.context_length`` is chosen from, instead of assuming longer is
   better.

3. **What is the target's own temporal structure?** Lag-`k` autocorrelation of
   deseasonalized target anomalies, i.e. the statistic the temporal model is
   supposed to reproduce.

Everything is computed on the **training** split only, so nothing here consults
validation or test data.

Usage::

    mamba run -n Prithvi python examples/CORDEX_ML/cordex_temporal_diagnostics.py \
        --config examples/CORDEX_ML/SA_downscaling_refinement_T2_ACCESS-CM2_static_temporal_recurrent.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402

from granitewxc.temporal.calendar import describe_time_axis, time_key  # noqa: E402
from granitewxc.temporal.config import parse_temporal_config  # noqa: E402
from granitewxc.temporal.sequence_dataset import date_in_range  # noqa: E402
from granitewxc.temporal.sources import build_frame_source  # noqa: E402
from granitewxc.temporal.training import _split_dates  # noqa: E402
from granitewxc.utils.config import get_config  # noqa: E402

#: Interpretation thresholds.
#:
#: Two verdicts are reported, deliberately, because they answer different questions
#: and the strict one fires on a physically ordinary case:
#:
#: ``strict_lag0_aligned``
#:     The peak is exactly at lag 0 and clears its neighbours by
#:     ``ALIGNMENT_MIN_MARGIN``. This is the right test for a *diagnostic* variable
#:     whose driver is simultaneous, e.g. 850 hPa temperature against tasmax.
#:
#: ``same_day_defensible``
#:     The peak is within +/-1 day and lag 0 retains at least
#:     ``ALIGNMENT_LAG0_RATIO`` of the peak correlation. This is what actually
#:     licenses same-day supervision, and it tolerates a driver that genuinely
#:     leads its target by less than the sampling interval -- 700 hPa humidity leads
#:     daily precipitation, so the lag-0 and lag-(-1) correlations are expected to be
#:     nearly tied. A truly unaligned pair (two different realizations of the same
#:     climate) is flat and near zero at *every* lag and fails both tests.
#:
#: Both are reported; neither is hidden. The strict verdict firing on a variable
#: whose driver leads it is a property of the test, not evidence of misalignment,
#: and the printed lag curve lets a reader judge for themselves.
ALIGNMENT_MIN_PEAK = 0.35
ALIGNMENT_MIN_MARGIN = 0.05
ALIGNMENT_MAX_PEAK_LAG = 1
ALIGNMENT_LAG0_RATIO = 0.90


def _deseasonalize(values: np.ndarray, period: int = 365) -> np.ndarray:
    """Remove a day-of-year climatology along axis 0."""
    doy = np.arange(values.shape[0]) % period
    clim = np.stack([values[doy == d].mean(axis=0) for d in range(period)])
    return values - clim[doy]


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 16:
        return float("nan")
    x, y = a[good] - a[good].mean(), b[good] - b[good].mean()
    den = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / den) if den > 0 else float("nan")


def _r2(columns: list[np.ndarray], y: np.ndarray) -> float:
    design = np.concatenate([np.ones((len(y), 1))] + columns, axis=1)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    return float(1.0 - resid.var() / y.var())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--days", type=int, default=3650, help="Days of the training split to use")
    ap.add_argument("--stride", type=int, default=24, help="Grid-point sampling stride")
    ap.add_argument("--max-lag", type=int, default=4)
    ap.add_argument("--max-history-lags", type=int, default=9)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    config = get_config(args.config)
    cfg = parse_temporal_config(getattr(config, "temporal", None))
    source = build_frame_source(config, "train")
    start, end = _split_dates(config, "train")

    frames = [f for f in source.frames() if date_in_range(f.timestamp, start, end)]
    frames.sort(key=lambda f: time_key(f.timestamp))
    if not frames:
        raise SystemExit(f"No training frames in [{start}, {end}].")
    frames = frames[: int(args.days)]

    report: dict[str, Any] = {
        "config": args.config,
        "case_name": getattr(config, "case_name", None),
        "train_date_range": [start, end],
        "n_days_used": len(frames),
        "output_vars": [str(v) for v in config.data.output_vars],
        "time_axis": describe_time_axis(
            [f.timestamp for f in frames], cfg.cadence_days if cfg else 1.0,
            calendar=source.calendar(),
        ),
    }

    static_channels = (
        int(getattr(config.model, "num_static_channels", 1))
        if bool(getattr(config.data, "use_static", True))
        else 0
    )

    # Stream the frames, retaining only what the diagnostics need: domain means
    # (for the alignment cross-correlation) and a sparse grid-point sample (for
    # the local headroom regression). Holding the full [T, C, H, W] cube would be
    # ~7.6 GB at 3650 days for the SA case, for no benefit.
    print(f"Loading {len(frames)} training days from {args.config} (streaming) ...")
    lat_slice = lon_slice = slice(None)
    probe = source.load_frame(frames[0], lat_slice, lon_slice)
    n_dyn = probe["x"].shape[0] - static_channels
    height, width = probe["y"].shape[-2:]
    ys_idx = np.arange(args.stride // 2, height, args.stride)
    xs_idx = np.arange(args.stride // 2, width, args.stride)

    T = len(frames)
    n_tgt = probe["y"].shape[0]
    xm = np.empty((T, n_dyn), dtype=np.float64)
    ym = np.empty((T, n_tgt), dtype=np.float64)
    Xs = np.empty((T, n_dyn, len(ys_idx), len(xs_idx)), dtype=np.float64)
    Ys = np.empty((T, n_tgt, len(ys_idx), len(xs_idx)), dtype=np.float64)
    for i, frame in enumerate(frames):
        loaded = source.load_frame(frame, lat_slice, lon_slice)
        xd = loaded["x"].numpy()[:n_dyn].astype(np.float64)
        yv = loaded["y"].numpy().astype(np.float64)
        yv[~loaded["valid_mask"].numpy()] = np.nan
        xm[i] = np.nanmean(xd, axis=(1, 2))
        ym[i] = np.nanmean(yv, axis=(1, 2))
        Xs[i] = xd[:, ys_idx][:, :, xs_idx]
        Ys[i] = yv[:, ys_idx][:, :, xs_idx]
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{T}")
    del probe
    predictor_names = [
        f"{v}_{str(l)[:-2] if str(l).endswith('.0') else str(l)}"
        for v in config.data.input_vars
        for l in config.data.input_levels
    ][:n_dyn]
    if len(predictor_names) < n_dyn:
        predictor_names += [f"chan_{i}" for i in range(len(predictor_names), n_dyn)]
    target_names = [str(v) for v in config.data.output_vars]
    report["n_dynamic_predictors"] = int(n_dyn)
    report["predictor_names"] = predictor_names

    # ------------------------------------------------------------------
    # 1. event alignment (domain-mean deseasonalized cross-correlation)
    # ------------------------------------------------------------------
    xm = _deseasonalize(xm)       # [T, Cx]
    ym = _deseasonalize(ym)       # [T, Cy]

    alignment: dict[str, Any] = {}
    for ti, tname in enumerate(target_names):
        # Pick the single predictor most correlated at lag 0 as the probe.
        lag0 = [abs(_corr(xm[:, ci], ym[:, ti])) for ci in range(n_dyn)]
        probe = int(np.nanargmax(lag0))
        curve: dict[str, float] = {}
        for lag in range(-args.max_lag, args.max_lag + 1):
            if lag >= 0:
                a = xm[lag:, probe]
                b = ym[: len(ym) - lag, ti] if lag else ym[:, ti]
            else:
                a = xm[:lag, probe]
                b = ym[-lag:, ti]
            curve[str(lag)] = _corr(a, b)
        peak_lag = int(max(curve, key=lambda k: abs(curve[k])))
        peak_abs = abs(curve[str(peak_lag)])
        lag0_abs = abs(curve["0"])
        neighbours = [abs(curve[str(l)]) for l in (-1, 1) if str(l) in curve]
        margin = lag0_abs - max(neighbours) if neighbours else float("nan")

        strict = bool(
            peak_lag == 0
            and lag0_abs >= ALIGNMENT_MIN_PEAK
            and margin >= ALIGNMENT_MIN_MARGIN
        )
        lag0_ratio = lag0_abs / peak_abs if peak_abs > 0 else float("nan")
        defensible = bool(
            abs(peak_lag) <= ALIGNMENT_MAX_PEAK_LAG
            and peak_abs >= ALIGNMENT_MIN_PEAK
            and lag0_ratio >= ALIGNMENT_LAG0_RATIO
        )

        if strict:
            verdict = (
                "lag-0 peak: predictors and targets describe the same weather; "
                "date-paired losses and metrics are valid"
            )
        elif defensible:
            verdict = (
                f"peak at lag {peak_lag:+d} but lag 0 retains "
                f"{100 * lag0_ratio:.1f}% of the peak correlation. The driver leads "
                "its target by less than one sampling interval, which is physically "
                "expected (e.g. mid-level humidity ahead of daily precipitation). "
                "Same-day supervision and date-paired metrics remain valid; the "
                "strict lag-0 test is mis-specified for a leading driver."
            )
        else:
            verdict = (
                "NOT event-aligned: the lag curve has no clear peak near lag 0. Use "
                "distributional metrics only (--not-event-aligned) and do not apply "
                "day-by-day losses or event-paired scores."
            )

        alignment[tname] = {
            "probe_predictor": predictor_names[probe],
            "lag_correlation": curve,
            "peak_lag": peak_lag,
            "peak_abs": peak_abs,
            "lag0_abs": lag0_abs,
            "lag0_over_peak": lag0_ratio,
            "margin_over_neighbours": margin,
            "strict_lag0_aligned": strict,
            "same_day_defensible": defensible,
            # ``event_aligned`` is the field callers act on.
            "event_aligned": defensible,
            "verdict": verdict,
        }
    report["event_alignment"] = alignment

    # ------------------------------------------------------------------
    # 2. headroom from history (local, deseasonalized, linear)
    # ------------------------------------------------------------------
    Xs = _deseasonalize(Xs)
    Ys = _deseasonalize(Ys)
    L = int(args.max_history_lags) + 1

    headroom: dict[str, Any] = {}
    lag_counts = sorted({0, 1, 2, 3, 5, int(args.max_history_lags)})
    for ti, tname in enumerate(target_names):
        curve: dict[str, float] = {}
        for nl in lag_counts:
            scores = []
            for iy in range(len(ys_idx)):
                for ix in range(len(xs_idx)):
                    yv = Ys[L:, ti, iy, ix]
                    cols = [Xs[L:, :, iy, ix]] + [
                        Xs[L - k : -k, :, iy, ix] for k in range(1, nl + 1)
                    ]
                    scores.append(_r2(cols, yv))
            curve[str(nl)] = float(np.mean(scores))
        same = curve["0"]
        headroom[tname] = {
            "r2_by_history_lags": curve,
            "r2_same_day_only": same,
            "gain_lags_1_3": curve.get("3", float("nan")) - same,
            # Two DIFFERENT normalizations of the same gain. They are easy to
            # confuse and here they differ by an order of magnitude in OPPOSITE
            # directions, so both are reported explicitly:
            #   explained_variance_increase = gain / R2_same_day
            #       "history increases explained variance by X%"
            #   residual_variance_reduction = gain / (1 - R2_same_day)
            #       "history removes X% of what was still unexplained"
            # For pr the first is ~13% and the second ~3%; for tasmax it is the
            # other way round. Quoting the wrong one badly misstates the headroom.
            "explained_variance_increase_lags_1_3": (
                (curve.get("3", float("nan")) - same) / same if same > 0 else float("nan")
            ),
            "residual_variance_reduction_lags_1_3": (
                (curve.get("3", float("nan")) - same) / (1.0 - same) if same < 1 else float("nan")
            ),
            "note": (
                "In-sample linear R2, not cross-validated: each added lag brings "
                f"{n_dyn} free parameters, so the apparent gain beyond ~lag 5 is "
                "largely overfitting. Treat as an upper bound on the LINEAR "
                "contribution of history, and as a lower bound on what a nonlinear "
                "model could extract."
            ),
        }
    report["history_headroom"] = headroom

    # ------------------------------------------------------------------
    # 3. target temporal structure
    # ------------------------------------------------------------------
    persistence: dict[str, Any] = {}
    lags = list(cfg.evaluation.autocorr_lags) if cfg else [1, 2, 3, 5]
    for ti, tname in enumerate(target_names):
        series = ym[:, ti]
        entry = {f"lag{l}": _corr(series[l:], series[:-l]) for l in lags}
        # crude e-folding estimate from lag 1
        r1 = entry.get("lag1", float("nan"))
        entry["efolding_days_from_lag1"] = (
            float(-1.0 / np.log(r1)) if np.isfinite(r1) and 0 < r1 < 1 else float("nan")
        )
        persistence[tname] = entry
    report["target_persistence_deseasonalized"] = persistence

    # ------------------------------------------------------------------
    # context-length recommendation
    # ------------------------------------------------------------------
    efolds = [
        persistence[t].get("efolding_days_from_lag1", float("nan")) for t in target_names
    ]
    finite = [e for e in efolds if np.isfinite(e)]
    suggested = int(np.ceil(2.5 * max(finite))) + 1 if finite else 7
    report["context_length"] = {
        "configured": cfg.context_length if cfg else None,
        "suggested_from_efolding": int(min(max(suggested, 3), 21)),
        "rationale": (
            "About 2.5 e-folding times of the slowest target, plus the current "
            "frame. Longer windows cost linearly in memory and, per Robin & Vrac "
            "(2021), do not monotonically improve temporal statistics."
        ),
    }

    text = json.dumps(report, indent=2, default=str)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"[write] {args.output}")

    print("\n" + "=" * 78)
    for tname, entry in alignment.items():
        flag = "ALIGNED" if entry["event_aligned"] else "NOT ALIGNED"
        strict = "strict" if entry["strict_lag0_aligned"] else "lead<1d"
        print(f"{tname:10s} {flag:12s} ({strict:8s}) peak_lag={entry['peak_lag']:+d} "
              f"|r(0)|={entry['lag0_abs']:.4f} r0/peak={entry['lag0_over_peak']:.3f} "
              f"probe={entry['probe_predictor']}")
    for tname in target_names:
        h = headroom[tname]
        print(f"{tname:10s} R2 same-day={h['r2_same_day_only']:.4f} "
              f"+lags1-3={h['gain_lags_1_3']:+.4f} "
              f"(explained +{100 * h['explained_variance_increase_lags_1_3']:.1f}%, "
              f"unexplained -{100 * h['residual_variance_reduction_lags_1_3']:.1f}%)")
    print("=" * 78)

    all_aligned = all(v["event_aligned"] for v in alignment.values())
    if not all_aligned:
        print(
            "\nWARNING: at least one target is not clearly event-aligned. Set "
            "temporal.evaluation.event_paired: false and evaluate with "
            "--not-event-aligned."
        )
    return 0 if all_aligned else 1


if __name__ == "__main__":
    raise SystemExit(main())
