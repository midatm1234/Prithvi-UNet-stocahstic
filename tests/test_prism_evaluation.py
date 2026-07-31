import numpy as np
import pytest

from granitewxc.utils.prism_evaluation import (
    StreamingMetricMeans,
    StreamingVariableMetrics,
    boundary_gradient_error,
    exact_axis_slice,
    map_native_coordinate_boundaries,
    spatial_power_metrics,
    validate_evaluation_coordinates,
)


def test_coordinate_contract_is_exact_with_explicit_legacy_float32_escape():
    canonical = np.linspace(-124.45, -115.925, 1024, dtype=np.float64)
    legacy = canonical.astype(np.float32)

    with pytest.raises(ValueError, match="do not exactly match"):
        validate_evaluation_coordinates(
            canonical, legacy, name="longitude", context="old inference"
        )

    assert validate_evaluation_coordinates(
        canonical,
        legacy,
        name="longitude",
        context="old inference",
        allow_legacy_float32=True,
    )

    shifted = legacy.copy()
    shifted[100] += np.float32(2.0e-5)
    with pytest.raises(ValueError, match="do not exactly match"):
        validate_evaluation_coordinates(
            canonical,
            shifted,
            name="longitude",
            context="shifted inference",
            allow_legacy_float32=True,
        )


def test_exact_truth_slice_never_uses_nearest_coordinates():
    source = np.arange(12, dtype=np.float64) * 0.25
    canonical = source[3:9].copy()
    assert exact_axis_slice(
        source, canonical, name="latitude", context="truth"
    ) == slice(3, 9)

    shifted = canonical + 1.0e-10
    with pytest.raises(ValueError, match="never interpolated"):
        exact_axis_slice(source, shifted, name="latitude", context="truth")


def test_streaming_metrics_use_only_finite_prediction_truth_pairs():
    accumulator = StreamingVariableMetrics((2, 2))
    accumulator.update(
        np.array([[2.0, 4.0], [np.nan, 8.0]]),
        np.array([[1.0, 2.0], [3.0, np.nan]]),
    )
    accumulator.update(
        np.array([[4.0, 8.0], [6.0, 10.0]]),
        np.array([[2.0, 4.0], [3.0, 8.0]]),
    )
    result = accumulator.finalize()

    np.testing.assert_allclose(
        result.prediction_climatology, [[3.0, 6.0], [6.0, 10.0]]
    )
    np.testing.assert_allclose(
        result.target_climatology, [[1.5, 3.0], [3.0, 8.0]]
    )
    np.testing.assert_allclose(
        result.temporal_rmse,
        [[np.sqrt(2.5), np.sqrt(10.0)], [3.0, 2.0]],
    )
    assert result.scalars["days"] == 2
    assert result.scalars["finite_sample_pairs"] == 6
    assert result.scalars["mean_bias"] == pytest.approx(14.0 / 6.0)
    assert result.scalars["space_time_rmse"] == pytest.approx(np.sqrt(38.0 / 6.0))
    assert result.scalars["climatology_rmse"] == pytest.approx(
        np.sqrt(np.mean(np.square([1.5, 3.0, 3.0, 2.0])))
    )


def test_boundary_score_detects_artificial_crossover_not_real_target_gradient():
    height, width = 48, 64
    y, x = np.mgrid[:height, :width]
    truth = 0.2 * x + 0.1 * y
    prediction = truth.copy()
    prediction[:, 32:] += 5.0

    at_artifact = boundary_gradient_error(
        prediction,
        truth,
        x_positions=[32],
        half_width=0,
    )
    away_from_artifact = boundary_gradient_error(
        prediction,
        truth,
        x_positions=[16],
        half_width=0,
    )
    assert at_artifact["boundary_gradient_mae"] == pytest.approx(5.0)
    assert at_artifact["boundary_gradient_excess"] > 4.9
    assert away_from_artifact["boundary_gradient_mae"] == pytest.approx(0.0)


def test_daily_boundary_metrics_cannot_cancel_in_a_climatology():
    height, width = 24, 32
    y, x = np.mgrid[:height, :width]
    truth = 0.2 * x + 0.1 * y
    positive_seam = truth.copy()
    negative_seam = truth.copy()
    positive_seam[:, 16:] += 4.0
    negative_seam[:, 16:] -= 4.0

    climatology = StreamingVariableMetrics((height, width))
    daily = StreamingMetricMeans()
    for prediction in (positive_seam, negative_seam):
        climatology.update(prediction, truth)
        daily.update(
            boundary_gradient_error(
                prediction, truth, x_positions=[16], half_width=0
            )
        )

    final = climatology.finalize()
    climatology_score = boundary_gradient_error(
        final.prediction_climatology,
        final.target_climatology,
        x_positions=[16],
        half_width=0,
    )
    daily_score = daily.finalize()
    assert climatology_score["boundary_gradient_mae"] == pytest.approx(0.0)
    assert daily_score["boundary_gradient_mae"] == pytest.approx(4.0)
    assert daily_score["boundary_gradient_mae_finite_days"] == 2


def test_native_centers_map_to_prism_pixel_edges_without_field_resampling():
    fine = np.arange(0.0, 10.0, 0.25)
    native_centers = np.arange(0.0, 10.0, 2.0)
    # Midpoints are 1,3,5,7 degrees -> insertion/pixel-edge indices 4,12,20,28.
    assert map_native_coordinate_boundaries(fine, native_centers) == [4, 12, 20, 28]


def test_block_band_power_is_sensitive_to_an_eight_pixel_pattern():
    size = 128
    x = np.arange(size, dtype=np.float64)
    block_wave = np.sin(2.0 * np.pi * x / 8.0)[None, :]
    prediction = np.repeat(block_wave, size, axis=0)
    truth_wave = np.sin(2.0 * np.pi * x / 32.0)[None, :]
    truth = np.repeat(truth_wave, size, axis=0)

    metrics = spatial_power_metrics(
        prediction, truth, block_period=8.0, native_period=(32.0, 32.0)
    )
    assert metrics["inference_block_axial_fraction"] > 0.9
    assert metrics["prism_block_axial_fraction"] < 0.01
    assert metrics["inference_to_prism_block_axial_fraction_ratio"] > 100.0
    assert metrics["prism_native_axial_fraction"] > 0.9
