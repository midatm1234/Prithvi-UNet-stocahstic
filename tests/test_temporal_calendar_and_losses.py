"""Calendar handling, sequence windowing, and temporal-loss behaviour.

The loss tests are the important ones: they pin the property that separates a
*matching* temporal objective from a *smoothness* prior. A term that rewards flat
output would pass a "does it run" test and quietly destroy fronts and extremes, so
each term is checked against an explicit adversary (a constant field, a
drizzle-everywhere field) that must score worse than the truth.
"""

from __future__ import annotations

import cftime
import numpy as np
import pytest
import torch

from granitewxc.temporal.calendar import (
    build_time_features,
    days_in_year,
    describe_time_axis,
    elapsed_days,
    find_discontinuities,
    resolve_calendar,
    time_key,
    year_fraction,
)
from granitewxc.temporal.config import (
    TemporalConfigError,
    parse_temporal_config,
)
from granitewxc.temporal.losses import (
    TemporalLossComputer,
    accumulation_loss,
    lag_autocorrelation_loss,
    occurrence_persistence_loss,
    tendency_loss,
    tmax_tmin_consistency_loss,
)
from granitewxc.temporal.sequence_dataset import (
    FrameRef,
    build_runs,
    build_windows,
    date_in_range,
)


# ---------------------------------------------------------------------------
# calendars
# ---------------------------------------------------------------------------
def _days(calendar: str, start: tuple[int, int, int], n: int, *, skip: set[tuple[int, int]] = frozenset()):
    """Consecutive daily timestamps in ``calendar``, optionally omitting (m, d)."""
    out = []
    current = cftime.datetime(*start, calendar=calendar)
    step = cftime.datetime(*start, calendar=calendar) - cftime.datetime(*start, calendar=calendar)
    one_day = cftime.datetime(2001, 1, 2, calendar=calendar) - cftime.datetime(2001, 1, 1, calendar=calendar)
    while len(out) < n:
        if (current.month, current.day) not in skip:
            out.append(current)
        current = current + one_day
    return out


@pytest.mark.parametrize(
    "name,expected",
    [
        ("standard", "standard"),
        ("gregorian", "standard"),
        ("noleap", "noleap"),
        ("365_day", "noleap"),
        ("360_day", "360_day"),
        ("all_leap", "all_leap"),
        ("proleptic_gregorian", "proleptic_gregorian"),
    ],
)
def test_calendar_aliases(name, expected):
    assert resolve_calendar(name).name == expected


def test_unsupported_calendar_rejected():
    with pytest.raises(ValueError, match="Unsupported calendar"):
        resolve_calendar("mayan")


@pytest.mark.parametrize(
    "calendar,year,expected",
    [
        ("standard", 1999, 365),
        ("standard", 2000, 366),   # divisible by 400
        ("standard", 1900, 365),   # divisible by 100 but not 400
        ("standard", 2004, 366),
        ("noleap", 2000, 365),
        ("all_leap", 1999, 366),
        ("360_day", 2000, 360),
    ],
)
def test_days_in_year(calendar, year, expected):
    assert days_in_year(year, calendar) == expected


def test_year_fraction_uses_the_right_year_length():
    """Seasonal phase must not drift across a leap year."""
    # 1 January is exactly 0.0 in every calendar.
    for cal in ("standard", "noleap", "360_day"):
        first = cftime.datetime(2000, 1, 1, calendar=cal)
        assert year_fraction([first], cal)[0] == pytest.approx(0.0)
    # 31 December of a leap year is 365/366, of a common year 364/365.
    leap = year_fraction([cftime.datetime(2000, 12, 31, calendar="standard")], "standard")[0]
    common = year_fraction([cftime.datetime(1999, 12, 31, calendar="standard")], "standard")[0]
    assert leap == pytest.approx(365 / 366)
    assert common == pytest.approx(364 / 365)


def test_elapsed_days_sees_a_removed_leap_day():
    """A 365-day archive labelled 'standard' must show a 2-day step at 1 March."""
    times = [
        cftime.datetime(2004, 2, 27, calendar="standard"),
        cftime.datetime(2004, 2, 28, calendar="standard"),
        cftime.datetime(2004, 3, 1, calendar="standard"),  # 29 Feb missing
        cftime.datetime(2004, 3, 2, calendar="standard"),
    ]
    deltas = elapsed_days(times, "standard")
    assert np.isnan(deltas[0])
    assert deltas[1] == pytest.approx(1.0)
    assert deltas[2] == pytest.approx(2.0)
    assert deltas[3] == pytest.approx(1.0)

    breaks = find_discontinuities(times, 1.0, calendar="standard")
    assert breaks.tolist() == [True, False, True, False]


def test_discontinuities_flag_duplicates_and_reversals():
    a = cftime.datetime(2001, 1, 1, calendar="noleap")
    b = cftime.datetime(2001, 1, 2, calendar="noleap")
    # duplicate timestamp -> zero step -> flagged
    assert find_discontinuities([a, a, b], 1.0, calendar="noleap").tolist() == [True, True, False]
    # out of order -> negative step -> flagged
    assert find_discontinuities([b, a], 1.0, calendar="noleap").tolist() == [True, True]


def test_hour_of_day_suppressed_for_constant_stamps():
    """A constant 12:00 stamp carries no diurnal information."""
    times = [cftime.datetime(2001, 1, d, 12, calendar="noleap") for d in (1, 2, 3)]
    _, spec = build_time_features(times, cadence_days=1.0, calendar="noleap")
    assert not spec.include_hour_of_day
    assert "hour_sin" not in spec.names

    with pytest.raises(ValueError, match="constant stamp carries no diurnal"):
        build_time_features(
            times, cadence_days=1.0, calendar="noleap", include_hour_of_day=True
        )


def test_hour_of_day_emitted_for_subdaily_archive():
    times = [cftime.datetime(2001, 1, 1, h, calendar="noleap") for h in (0, 6, 12, 18)]
    feats, spec = build_time_features(times, cadence_days=0.25, calendar="noleap")
    assert spec.include_hour_of_day
    assert "hour_sin" in spec.names and "hour_cos" in spec.names
    assert feats.shape == (4, spec.dim)


def test_time_features_are_chunk_invariant():
    """Regression test for a bug found by the real-data check.

    ``state_age`` and ``is_sequence_start`` must depend only on the distance from
    the last state reset and on the configured context length -- never on how many
    frames happen to be in the current array. When they did, chunked inference
    disagreed with a single pass by ~0.02 K on the SA case.
    """
    times = [cftime.datetime(2001, 1, d, calendar="noleap") for d in range(1, 13)]
    ctx = 7
    full, spec = build_time_features(
        times, cadence_days=1.0, calendar="noleap", context_length=ctx
    )
    # Same 12 frames, assembled as chunks of 5 + 4 + 3, each told where it starts.
    pieces = []
    offset = 0
    for size in (5, 4, 3):
        part, _ = build_time_features(
            times[offset : offset + size],
            cadence_days=1.0,
            calendar="noleap",
            context_length=ctx,
            position_offset=offset,
            mark_sequence_start=(offset == 0),
        )
        pieces.append(part)
        offset += size
    chunked = np.concatenate(pieces, axis=0)
    assert chunked.shape == full.shape
    np.testing.assert_array_equal(chunked, full)

    # state_age saturates at 1.0 once the state is fully warmed up.
    age = full[:, list(spec.names).index("state_age")]
    assert age[0] == pytest.approx(0.0)
    assert age[ctx - 1] == pytest.approx(1.0)
    assert age[-1] == pytest.approx(1.0)
    start = full[:, list(spec.names).index("is_sequence_start")]
    assert start[0] == 1.0 and start[1:].sum() == 0.0


def test_gap_feature_scales_large_and_small_gaps():
    """A dropped day and a century-long jump must both be representable."""
    small = [
        cftime.datetime(2004, 2, 28, calendar="standard"),
        cftime.datetime(2004, 3, 1, calendar="standard"),
    ]
    feats, spec = build_time_features(small, cadence_days=1.0, calendar="standard")
    idx = list(spec.names).index("log_gap_excess")
    assert feats[1, idx] == pytest.approx(np.log1p(1.0), rel=1e-5)

    huge = [
        cftime.datetime(1980, 12, 31, calendar="standard"),
        cftime.datetime(2080, 1, 1, calendar="standard"),
    ]
    feats2, _ = build_time_features(huge, cadence_days=1.0, calendar="standard")
    # Compressed, but strictly larger than the one-day-gap case.
    assert feats2[1, idx] > feats[1, idx]
    assert feats2[1, idx] < 15.0


# ---------------------------------------------------------------------------
# runs and windows
# ---------------------------------------------------------------------------
def _refs(times):
    return [FrameRef(0, i, i, t) for i, t in enumerate(times)]


def test_runs_split_at_every_discontinuity():
    times = (
        [cftime.datetime(2001, 1, d, calendar="noleap") for d in range(1, 6)]
        + [cftime.datetime(2001, 2, d, calendar="noleap") for d in range(1, 5)]
    )
    runs = build_runs(_refs(times), cadence_days=1.0, calendar="noleap")
    assert [len(r) for r in runs] == [5, 4]
    assert [r.run_id for r in runs] == [0, 1]


def test_runs_sort_by_actual_time_not_index():
    times = [cftime.datetime(2001, 1, d, calendar="noleap") for d in (3, 1, 2, 4)]
    runs = build_runs(_refs(times), cadence_days=1.0, calendar="noleap")
    assert len(runs) == 1
    assert [time_key(f.timestamp)[2] for f in runs[0].frames] == [1, 2, 3, 4]


def test_duplicate_timestamps_rejected():
    t = cftime.datetime(2001, 1, 1, calendar="noleap")
    with pytest.raises(ValueError, match="duplicate timestamp"):
        build_runs(_refs([t, t]), cadence_days=1.0, calendar="noleap")


def test_windows_never_straddle_a_run_boundary():
    times = (
        [cftime.datetime(2001, 1, d, calendar="noleap") for d in range(1, 8)]
        + [cftime.datetime(2001, 3, d, calendar="noleap") for d in range(1, 4)]
    )
    runs = build_runs(_refs(times), cadence_days=1.0, calendar="noleap")
    windows = build_windows(runs, window_length=5, stride=1)
    # Run 0 has 7 frames -> 3 windows; run 1 has 3 frames -> none (too short).
    assert len(windows) == 3
    for w in windows:
        assert w.run_id == 0
        deltas = elapsed_days([f.timestamp for f in w.frames], "noleap")
        assert np.allclose(deltas[1:], 1.0)


def test_short_runs_are_dropped_not_padded():
    times = [cftime.datetime(2001, 1, d, calendar="noleap") for d in range(1, 4)]
    runs = build_runs(_refs(times), cadence_days=1.0, calendar="noleap")
    assert build_windows(runs, window_length=7, stride=1) == []


def test_date_in_range_handles_absent_calendar_days():
    """Comparing (y, m, d) avoids constructing 29 Feb in a no-leap file."""
    stamp = cftime.datetime(2004, 3, 1, calendar="noleap")
    assert date_in_range(stamp, "2004-02-29", "2004-03-02")
    assert not date_in_range(stamp, "2004-03-02", None)
    assert not date_in_range(stamp, None, "2004-02-28")


# ---------------------------------------------------------------------------
# config validation
# ---------------------------------------------------------------------------
def test_absent_or_disabled_temporal_block_returns_none():
    assert parse_temporal_config(None) is None
    assert parse_temporal_config({}) is None
    assert parse_temporal_config({"enabled": False, "backend": "recurrent"}) is None


def test_unknown_keys_rejected():
    with pytest.raises(TemporalConfigError, match="Unknown key"):
        parse_temporal_config(
            {"enabled": True, "backend": "recurrent", "typo_key": 1,
             "init_from_spatial_checkpoint": "x.ckpt"}
        )


def test_downscaling_mode_forbids_lead_time_and_non_causal():
    base = {"enabled": True, "backend": "recurrent", "init_from_spatial_checkpoint": "x.ckpt"}
    with pytest.raises(TemporalConfigError, match="lead_time_days must be 0"):
        parse_temporal_config({**base, "lead_time_days": 1.0})
    with pytest.raises(TemporalConfigError, match="causal must be true"):
        parse_temporal_config({**base, "causal": False})


def test_forecasting_mode_requires_positive_lead():
    base = {"enabled": True, "backend": "recurrent", "mode": "forecasting",
            "init_from_spatial_checkpoint": "x.ckpt"}
    with pytest.raises(TemporalConfigError, match="lead_time_days must be > 0"):
        parse_temporal_config(base)
    cfg = parse_temporal_config({**base, "lead_time_days": 1.0,
                                 "time_features": {"include_lead_time": True}})
    assert cfg.mode == "forecasting" and cfg.include_lead_time


def test_geometry_constraints():
    base = {"enabled": True, "backend": "recurrent", "init_from_spatial_checkpoint": "x.ckpt"}
    with pytest.raises(TemporalConfigError, match="cannot exceed context_length"):
        parse_temporal_config({**base, "context_length": 4, "output_length": 5})
    with pytest.raises(TemporalConfigError, match="warmup_length"):
        parse_temporal_config({**base, "context_length": 4, "warmup_length": 4})
    with pytest.raises(TemporalConfigError, match="exceeds context_length"):
        parse_temporal_config(
            {**base, "context_length": 5, "warmup_length": 3, "output_length": 4}
        )


def test_checkpoint_source_required():
    with pytest.raises(TemporalConfigError, match="requires either init_from_spatial_checkpoint"):
        parse_temporal_config({"enabled": True, "backend": "recurrent"})


def test_refinement_noise_rho_bounds():
    base = {"enabled": True, "backend": "recurrent", "init_from_spatial_checkpoint": "x.ckpt"}
    with pytest.raises(TemporalConfigError, match="must be < 1.0"):
        parse_temporal_config(
            {**base, "refinement": {"noise": "ar1_correlated", "noise_rho": 1.0}}
        )
    with pytest.raises(TemporalConfigError, match="only meaningful with"):
        parse_temporal_config(
            {**base, "refinement": {"noise": "iid_per_frame", "noise_rho": 0.5}}
        )
    cfg = parse_temporal_config(
        {**base, "refinement": {"noise": "ar1_correlated", "noise_rho": 0.6}}
    )
    assert cfg.refinement.noise_rho == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# losses -- the properties that matter
# ---------------------------------------------------------------------------
def _ramp(b=2, t=6, c=1, h=4, w=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, t, c, h, w, generator=g)


def test_tendency_loss_does_not_prefer_a_constant_field():
    """The decisive test: a flat prediction must score WORSE than the truth.

    ``|d(pred)|`` (a smoothness prior) is minimized by a constant field.
    ``|d(pred) - d(truth)|`` is minimized by the correct evolution. This asserts
    we implemented the second one.
    """
    truth = _ramp(seed=1)
    perfect = truth.clone()
    constant = truth.mean(dim=1, keepdim=True).expand_as(truth).contiguous()

    l_perfect = tendency_loss(perfect, truth, None)
    l_constant = tendency_loss(constant, truth, None)

    assert float(l_perfect) == pytest.approx(0.0, abs=1e-6)
    assert float(l_constant) > 0.1
    assert float(l_constant) > float(l_perfect)


def test_tendency_loss_accounts_for_the_time_interval():
    """A two-day step must not be charged as if it were a one-day change."""
    truth = torch.zeros(1, 3, 1, 2, 2)
    truth[:, 1] = 2.0
    truth[:, 2] = 4.0
    pred = torch.zeros_like(truth)

    regular = torch.ones(1, 3)
    gapped = torch.tensor([[1.0, 2.0, 1.0]])  # frame 1 is two days after frame 0

    l_regular = float(tendency_loss(pred, truth, None, interval_ratio=regular))
    l_gapped = float(tendency_loss(pred, truth, None, interval_ratio=gapped))
    assert l_gapped < l_regular


def test_tendency_loss_masks_pairs_touching_an_invalid_frame():
    truth = _ramp(b=1, t=3, seed=2)
    pred = torch.zeros_like(truth)
    mask = torch.ones_like(truth, dtype=torch.bool)
    mask[:, 1] = False  # middle frame invalid -> both differences dropped
    assert float(tendency_loss(pred, truth, mask)) == pytest.approx(0.0, abs=1e-6)


def test_accumulation_loss_detects_mistimed_events():
    """Same total, wrong day: daily MAE identical, 3-day accumulation differs."""
    truth = torch.zeros(1, 6, 1, 2, 2)
    truth[:, 1] = 10.0
    shifted = torch.zeros_like(truth)
    shifted[:, 4] = 10.0  # same amount, moved outside the first 3-day window

    l_same = accumulation_loss(truth, truth, None, windows=(3,), channels=(0,))
    l_shift = accumulation_loss(shifted, truth, None, windows=(3,), channels=(0,))
    assert float(l_same) == pytest.approx(0.0, abs=1e-6)
    assert float(l_shift) > 1.0


def test_occurrence_loss_penalizes_drizzle_everywhere():
    """Adding light rain everywhere must NOT be a way to lower this loss.

    ``drizzle`` is built to match the truth's wet-day *frequency* while destroying
    its persistence structure. The transition term must catch that.
    """
    torch.manual_seed(0)
    # Truth: persistent wet spells (first half wet, second half dry).
    truth = torch.zeros(1, 8, 1, 6, 6)
    truth[:, :4] = 8.0
    perfect = truth.clone()
    # Drizzle: alternating wet/dry days -> same wet fraction (0.5), no persistence.
    drizzle = torch.zeros_like(truth)
    drizzle[:, ::2] = 8.0

    l_perfect = float(
        occurrence_persistence_loss(perfect, truth, None, channel=0, wet_threshold=1.0)
    )
    l_drizzle = float(
        occurrence_persistence_loss(drizzle, truth, None, channel=0, wet_threshold=1.0)
    )
    assert l_perfect == pytest.approx(0.0, abs=1e-4)
    assert l_drizzle > 0.3, "transition term failed to penalize destroyed persistence"


def test_lag_autocorr_loss_returns_zero_below_min_samples():
    truth = _ramp(b=1, t=4, h=2, w=2, seed=3)
    pred = torch.zeros_like(truth)
    mask = torch.ones_like(truth, dtype=torch.bool)
    # 3 pairs * 1 * 2 * 2 = 12 valid entries, below the threshold.
    out = lag_autocorrelation_loss(pred, truth, mask, lags=(1,), min_samples=10_000)
    assert float(out) == 0.0


def test_lag_autocorr_loss_rewards_matching_persistence():
    t = 40
    base = torch.zeros(1, t, 1, 3, 3)
    # AR(1)-like truth
    prev = torch.zeros(1, 1, 3, 3)
    g = torch.Generator().manual_seed(5)
    for i in range(t):
        prev = 0.8 * prev + torch.randn(1, 1, 3, 3, generator=g)
        base[:, i] = prev
    white = torch.randn(1, t, 1, 3, 3, generator=g)
    l_match = float(lag_autocorrelation_loss(base, base, None, lags=(1,), min_samples=8))
    l_white = float(lag_autocorrelation_loss(white, base, None, lags=(1,), min_samples=8))
    assert l_match == pytest.approx(0.0, abs=1e-5)
    assert l_white > l_match


def test_tmax_tmin_consistency_only_penalizes_violations():
    pred = torch.zeros(1, 2, 2, 3, 3)
    pred[:, :, 0] = 300.0  # tmax
    pred[:, :, 1] = 290.0  # tmin  (consistent)
    ok = tmax_tmin_consistency_loss(pred, None, tmax_channel=0, tmin_channel=1)
    assert float(ok) == pytest.approx(0.0)

    bad = pred.clone()
    bad[:, :, 1] = 310.0  # tmin above tmax
    assert float(tmax_tmin_consistency_loss(bad, None, tmax_channel=0, tmin_channel=1)) > 5.0


def test_loss_computer_balances_variables_by_scale():
    """Precipitation in mm/day must not be swamped by temperature in kelvin."""
    cfg = parse_temporal_config(
        {
            "enabled": True,
            "backend": "recurrent",
            "init_from_spatial_checkpoint": "x.ckpt",
            "losses": {"tendency": {"enabled": True, "weight": 1.0}},
        }
    ).losses
    scales = {"pr": 5.0, "tasmax": 3.0}
    comp = TemporalLossComputer(cfg, output_vars=["pr", "tasmax"], variable_scales=scales)

    truth = torch.zeros(1, 4, 2, 3, 3)
    # Give each channel an error equal to its own scale: after rescaling both
    # contribute the same amount, so neither dominates.
    pred_pr = truth.clone()
    pred_pr[:, :, 0] = 5.0
    pred_tx = truth.clone()
    pred_tx[:, :, 1] = 3.0
    a = comp(pred_pr, truth).per_frame
    b = comp(pred_tx, truth).per_frame
    assert float(a) == pytest.approx(float(b), rel=1e-5)


def test_loss_computer_rejects_misconfigured_variables():
    cfg = parse_temporal_config(
        {
            "enabled": True,
            "backend": "recurrent",
            "init_from_spatial_checkpoint": "x.ckpt",
            "losses": {"occurrence": {"enabled": True, "weight": 0.1, "variable": "nope"}},
        }
    ).losses
    with pytest.raises(ValueError, match="not among output_vars"):
        TemporalLossComputer(cfg, output_vars=["pr", "tasmax"])


def test_loss_computer_reports_every_term():
    cfg = parse_temporal_config(
        {
            "enabled": True,
            "backend": "recurrent",
            "init_from_spatial_checkpoint": "x.ckpt",
            "losses": {
                "tendency": {"enabled": True, "weight": 0.1},
                "accumulation": {"enabled": True, "weight": 0.1, "windows": [3], "predictands": ["pr"]},
                "occurrence": {"enabled": True, "weight": 0.05, "variable": "pr"},
            },
        }
    ).losses
    comp = TemporalLossComputer(cfg, output_vars=["pr", "tasmax"])
    truth = torch.rand(2, 6, 2, 4, 4) * 5
    pred = truth + 0.3
    terms = comp(pred, truth, torch.ones_like(truth, dtype=torch.bool))
    log = terms.to_log()
    for key in ("per_frame", "tendency", "accumulation", "occurrence", "total"):
        assert key in log and np.isfinite(log[key])
    assert log["total"] > 0.0
    assert terms.total.requires_grad is False  # inputs had no grad; sanity


def test_describe_time_axis_reports_real_structure():
    times = (
        [cftime.datetime(2001, 1, d, calendar="noleap") for d in range(1, 6)]
        + [cftime.datetime(2001, 2, d, calendar="noleap") for d in range(1, 4)]
    )
    info = describe_time_axis(times, 1.0, calendar="noleap")
    assert info["n_times"] == 8
    assert info["calendar"] == "noleap"
    assert info["n_discontinuities"] == 2
    assert info["n_duplicate_timestamps"] == 0
    assert 1.0 in info["step_histogram"]
