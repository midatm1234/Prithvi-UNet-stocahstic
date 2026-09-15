"""Focused tests for refinement tensor diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from granitewxc.refinement.diagnostics import (
    summarize_refinement_tensor,
    summarize_refinement_tensors,
)


def test_summary_reports_declared_channels_and_finite_statistics():
    values = np.arange(2 * 2 * 4 * 5, dtype=np.float32).reshape(2, 2, 4, 5)
    values[0, 0, 0, 0] = np.nan
    values[0, 1, 0, 0] = np.inf
    report = summarize_refinement_tensor(
        "physical_residual",
        values,
        channel_names=("pr", "tasmax"),
    )

    assert report["shape"] == [2, 2, 4, 5]
    assert report["channel_names"] == ["pr", "tasmax"]
    assert set(report["per_channel"]) == {"pr", "tasmax"}
    assert report["overall"]["nan_count"] == 1
    assert report["overall"]["inf_count"] == 1
    assert report["overall"]["finite_count"] == values.size - 2
    assert report["overall"]["q01"] <= report["overall"]["q50"]
    assert report["overall"]["q50"] <= report["overall"]["q99"]
    assert report["overall"]["q05"] <= report["overall"]["q25"]
    assert report["overall"]["q25"] <= report["overall"]["q75"]
    assert 0.0 <= report["overall"]["zero_fraction"] <= 1.0
    assert report["overall"]["lag_one_spatial_autocorrelation"] is not None


def test_gradient_and_spectral_fraction_detect_checkerboard_texture():
    y, x = np.mgrid[:16, :16]
    smooth = np.broadcast_to(x, (1, 1, 16, 16)).astype(np.float64)
    checkerboard = ((-1.0) ** (x + y))[None, None]

    smooth_report = summarize_refinement_tensor(
        "smooth",
        smooth,
        channel_names=("pr",),
    )
    noisy_report = summarize_refinement_tensor(
        "checkerboard",
        checkerboard,
        channel_names=("pr",),
    )

    assert smooth_report["overall"][
        "spatial_gradient_magnitude"
    ] == pytest.approx(1.0)
    assert noisy_report["overall"][
        "high_frequency_spectral_power_fraction"
    ] > 0.99
    assert (
        noisy_report["overall"]["high_frequency_spectral_power_fraction"]
        > smooth_report["overall"]["high_frequency_spectral_power_fraction"]
    )


def test_named_tensor_report_and_channel_mismatch_guard():
    values = np.zeros((1, 2, 4, 4), dtype=np.float32)
    report = summarize_refinement_tensors(
        {
            "ground_truth": values,
            "phase1_prediction": values + 1.0,
            "predicted_physical_residual": values,
        },
        channel_names=("pr", "tasmax"),
    )
    assert set(report) == {
        "ground_truth",
        "phase1_prediction",
        "predicted_physical_residual",
    }
    assert report["predicted_physical_residual"]["overall"]["std"] == 0.0
    assert (
        report["predicted_physical_residual"]["overall"]
        ["high_frequency_spectral_power_fraction"]
        == 0.0
    )

    with pytest.raises(ValueError, match="channel names"):
        summarize_refinement_tensor(
            "bad",
            values,
            channel_names=("pr",),
        )


def test_summary_records_layout_units_and_precipitation_wet_fraction():
    values = np.array(
        [[[[0.0, 0.2], [1.0, 2.0]], [[280.0, 281.0], [282.0, 283.0]]]],
        dtype=np.float32,
    )
    report = summarize_refinement_tensor(
        "ground_truth",
        values,
        channel_names=("pr", "tasmax"),
        dimension_names=("batch", "channel", "lat", "lon"),
        channel_units={"pr": "mm/day", "tasmax": "K"},
        wet_thresholds={"pr": 1.0},
    )

    assert report["dimension_names"] == ["batch", "channel", "lat", "lon"]
    assert report["per_channel"]["pr"]["units"] == "mm/day"
    assert report["per_channel"]["tasmax"]["units"] == "K"
    assert report["per_channel"]["pr"]["zero_fraction"] == pytest.approx(0.25)
    assert report["per_channel"]["pr"]["wet_fraction"] == pytest.approx(0.5)
    assert "wet_fraction" not in report["per_channel"]["tasmax"]


def test_named_tensor_report_supports_ensemble_channel_axis():
    members = np.zeros((1, 3, 2, 4, 4), dtype=np.float32)
    report = summarize_refinement_tensors(
        {"members": members},
        channel_names=("pr", "tasmax"),
        channel_dim={"members": 2},
        dimension_names={
            "members": ("batch", "ensemble", "channel", "lat", "lon")
        },
    )
    assert set(report["members"]["per_channel"]) == {"pr", "tasmax"}

    with pytest.raises(ValueError, match="dimension names"):
        summarize_refinement_tensor(
            "bad_dims",
            members,
            channel_names=("pr", "tasmax"),
            channel_dim=2,
            dimension_names=("batch", "channel", "lat", "lon"),
        )


def test_named_tensor_report_supports_per_tensor_units_and_wet_thresholds():
    values = np.array(
        [[[[0.0, 2.0]], [[280.0, 282.0]]]],
        dtype=np.float32,
    )
    report = summarize_refinement_tensors(
        {
            "physical": values,
            "normalized": values,
        },
        channel_names=("pr", "tasmax"),
        channel_units={
            "physical": {"pr": "mm/day", "tasmax": "K"},
            "normalized": {
                "pr": "normalized residual",
                "tasmax": "normalized residual",
            },
        },
        wet_thresholds={"physical": {"pr": 1.0}},
    )

    assert report["physical"]["per_channel"]["pr"]["units"] == "mm/day"
    assert report["physical"]["per_channel"]["pr"]["wet_fraction"] == 0.5
    assert (
        report["normalized"]["per_channel"]["pr"]["units"]
        == "normalized residual"
    )
    assert "wet_fraction" not in report["normalized"]["per_channel"]["pr"]
