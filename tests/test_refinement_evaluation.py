"""Synthetic tests for dependency-light Phase-2 evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from granitewxc.refinement.evaluation import (
    deterministic_metrics,
    ensemble_metrics,
    precipitation_metrics,
    strata_masks,
    stratified_metrics,
    tasmin_tasmax_metrics,
    temperature_metrics,
)


def test_deterministic_metrics_have_known_bias_structure_and_quantiles():
    target = np.arange(24, dtype=np.float64).reshape(3, 2, 4)
    prediction = target + 2.0
    prediction[0, 0, 0] = np.nan
    mask = np.ones((2, 4), dtype=bool)
    mask[1, 3] = False

    metrics = deterministic_metrics(
        prediction,
        target,
        mask=mask,
        quantiles=(0.5,),
    )

    assert metrics["valid_count"] == 20
    assert metrics["mean_bias"] == pytest.approx(2.0)
    assert metrics["absolute_bias"] == pytest.approx(2.0)
    assert metrics["mae"] == pytest.approx(2.0)
    assert metrics["rmse"] == pytest.approx(2.0)
    assert metrics["centered_rmse"] == pytest.approx(0.0)
    assert metrics["pearson_correlation"] == pytest.approx(1.0)
    assert metrics["temporal_correlation"] == pytest.approx(1.0)
    assert metrics["spatial_pattern_correlation"] == pytest.approx(1.0)
    assert metrics["climatological_spatial_correlation"] == pytest.approx(1.0)
    assert metrics["standard_deviation_ratio"] == pytest.approx(1.0)
    assert metrics["distribution_wasserstein_1"] == pytest.approx(2.0)
    assert 0.0 <= metrics["distribution_ks_statistic"] <= 1.0
    assert metrics["quantile_error_q50"] == pytest.approx(2.0)
    assert metrics["gradient_mae"] == pytest.approx(0.0)
    assert metrics["gradient_rmse"] == pytest.approx(0.0)
    assert metrics["gradient_agreement"] == pytest.approx(1.0)


def test_deterministic_metrics_reject_mismatch_and_empty_valid_mask():
    with pytest.raises(ValueError, match="identical shapes"):
        deterministic_metrics(np.zeros((2, 3)), np.zeros((3, 2)))
    with pytest.raises(ValueError, match="no finite"):
        deterministic_metrics(
            np.ones((2, 3)),
            np.ones((2, 3)),
            mask=np.zeros((2, 3), dtype=bool),
        )


def test_gradient_agreement_is_defined_for_an_identical_constant_gradient():
    target = np.arange(8, dtype=np.float64)
    metrics = deterministic_metrics(
        target.copy(),
        target,
        quantiles=(),
        sample_axis=None,
        spatial_axes=(0,),
    )

    assert metrics["gradient_agreement"] == pytest.approx(1.0)
    assert np.isnan(metrics["gradient_correlation"])
    assert metrics["spatial_roughness_ratio"] == pytest.approx(1.0)
    assert metrics["spatial_smoothing_index"] == pytest.approx(0.0)


def test_spatial_smoothing_index_detects_lost_adjacent_scale_variability():
    target = np.indices((6, 8)).sum(axis=0) % 2
    target = target.astype(np.float64)
    prediction = np.full_like(target, np.mean(target))
    mask = np.ones_like(target, dtype=bool)
    mask[0, 0] = False

    metrics = deterministic_metrics(
        prediction,
        target,
        mask=mask,
        quantiles=(),
        sample_axis=None,
    )

    assert metrics["prediction_gradient_rms"] == pytest.approx(0.0)
    assert metrics["target_gradient_rms"] > 0.0
    assert metrics["spatial_roughness_ratio"] == pytest.approx(0.0)
    assert metrics["spatial_smoothing_index"] == pytest.approx(1.0)


def test_precipitation_metrics_separate_occurrence_intensity_and_extremes():
    target = np.array([0.0, 2.0, 5.0, 10.0])
    prediction = np.array([1.5, 3.0, 0.0, 12.0])
    metrics = precipitation_metrics(
        prediction,
        target,
        wet_day_threshold=1.0,
        extreme_quantiles=(0.5,),
    )

    assert metrics["prediction_wet_day_frequency"] == pytest.approx(0.75)
    assert metrics["target_wet_day_frequency"] == pytest.approx(0.75)
    assert metrics["wet_day_true_positives"] == 2
    assert metrics["wet_day_false_positives"] == 1
    assert metrics["wet_day_false_negatives"] == 1
    assert metrics["wet_day_true_negatives"] == 0
    assert metrics["wet_day_precision"] == pytest.approx(2.0 / 3.0)
    assert metrics["wet_day_recall"] == pytest.approx(2.0 / 3.0)
    assert metrics["wet_day_false_alarm_rate"] == pytest.approx(1.0)
    assert metrics["wet_day_false_alarm_ratio"] == pytest.approx(1.0 / 3.0)
    assert metrics["prediction_mean_wet_day_intensity"] == pytest.approx(5.5)
    assert metrics["target_mean_wet_day_intensity"] == pytest.approx(
        17.0 / 3.0
    )
    assert metrics["heavy_precipitation_bias_q50"] == pytest.approx(-1.5)
    assert metrics["extreme_rmse_q50"] == pytest.approx(np.sqrt(14.5))
    assert metrics["prediction_maximum"] == pytest.approx(12.0)
    assert metrics["target_maximum"] == pytest.approx(10.0)


def test_temperature_tail_threshold_and_ordering_metrics_are_known():
    target = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    prediction = target + 2.0
    metrics = temperature_metrics(
        prediction,
        target,
        quantiles=(0.2, 0.8),
        lower_thresholds={"freezing": 0.0},
        upper_thresholds={"hot": 35.0},
    )

    assert metrics["quantile_error_q20"] == pytest.approx(2.0)
    assert metrics["quantile_error_q80"] == pytest.approx(2.0)
    assert metrics["cold_extreme_rmse_q20"] == pytest.approx(2.0)
    assert metrics["warm_extreme_rmse_q80"] == pytest.approx(2.0)
    assert metrics["target_below_freezing_frequency"] == pytest.approx(0.2)
    assert metrics["prediction_below_freezing_frequency"] == pytest.approx(0.0)
    assert metrics["target_above_hot_frequency"] == pytest.approx(0.2)
    assert metrics["prediction_above_hot_frequency"] == pytest.approx(0.2)

    ordering = tasmin_tasmax_metrics(
        predicted_tasmin=np.array([5.0, 15.0, 20.0]),
        predicted_tasmax=np.array([10.0, 12.0, 19.0]),
        target_tasmin=np.array([4.0, 11.0, 18.0]),
        target_tasmax=np.array([9.0, 13.0, 20.0]),
    )
    assert ordering["tasmin_gt_tasmax_count"] == 2
    assert ordering["tasmin_gt_tasmax_violation_rate"] == pytest.approx(
        2.0 / 3.0
    )
    assert ordering["tasmin_gt_tasmax_mean_excess"] == pytest.approx(2.0)
    assert ordering["target_tasmin_gt_tasmax_count"] == 0
    assert ordering["target_tasmin_gt_tasmax_violation_rate"] == pytest.approx(
        0.0
    )


def test_empirical_ensemble_crps_spread_coverage_and_diversity_are_exact():
    # Columns are target locations.  Both ensembles have spread one and the
    # unordered pair distance is two.
    members = np.array([[0.0, 2.0], [2.0, 4.0]])
    target = np.array([0.0, 3.0])
    metrics = ensemble_metrics(
        members,
        target,
        member_axis=0,
        coverage_levels=(0.5,),
    )

    assert metrics["ensemble_size"] == 2
    assert metrics["empirical_crps"] == pytest.approx(0.5)
    assert metrics["ensemble_mean_bias"] == pytest.approx(0.5)
    assert metrics["ensemble_mean_rmse"] == pytest.approx(np.sqrt(0.5))
    assert metrics["mean_ensemble_spread"] == pytest.approx(np.sqrt(2.0))
    assert metrics["rms_ensemble_spread"] == pytest.approx(np.sqrt(2.0))
    assert metrics["spread_skill_ratio"] == pytest.approx(2.0)
    assert metrics["ensemble_member_diversity"] == pytest.approx(2.0)
    assert metrics["prediction_interval_coverage_q50"] == pytest.approx(0.5)
    assert metrics[
        "prediction_interval_coverage_error_q50"
    ] == pytest.approx(0.0)
    assert metrics[
        "prediction_interval_absolute_coverage_error_q50"
    ] == pytest.approx(0.0)
    assert metrics[
        "prediction_interval_mean_absolute_coverage_error"
    ] == pytest.approx(0.0)
    assert metrics["prediction_interval_mean_width_q50"] == pytest.approx(1.0)


def test_interval_coverage_error_exposes_underdispersed_ensemble():
    members = np.array([[0.0, 0.0], [1.0, 1.0]])
    target = np.array([-2.0, 3.0])

    metrics = ensemble_metrics(
        members,
        target,
        coverage_levels=(0.5, 0.8),
    )

    assert metrics["prediction_interval_coverage_q50"] == pytest.approx(0.0)
    assert metrics["prediction_interval_coverage_error_q50"] == pytest.approx(
        -0.5
    )
    assert metrics["prediction_interval_coverage_error_q80"] == pytest.approx(
        -0.8
    )
    assert metrics[
        "prediction_interval_mean_absolute_coverage_error"
    ] == pytest.approx(0.65)


def test_ensemble_metrics_omit_missing_members_per_location():
    members = np.array(
        [
            [[0.0, np.nan], [1.0, 2.0]],
            [[2.0, 3.0], [np.nan, 4.0]],
            [[1.0, 5.0], [3.0, 6.0]],
        ]
    )
    target = np.array([[1.0, 4.0], [2.0, np.nan]])
    metrics = ensemble_metrics(members, target, coverage_levels=(0.8,))

    assert metrics["valid_count"] == 3
    assert metrics["minimum_available_members"] == 2
    assert metrics["maximum_available_members"] == 3
    assert np.isfinite(metrics["empirical_crps"])
    assert np.isfinite(metrics["ensemble_member_diversity"])

    repository_layout = np.moveaxis(members, 0, 1)
    repo_metrics = ensemble_metrics(
        repository_layout,
        target,
        member_axis=1,
        coverage_levels=(0.8,),
    )
    assert repo_metrics["empirical_crps"] == pytest.approx(
        metrics["empirical_crps"]
    )


def test_strata_api_broadcasts_sample_and_spatial_labels_without_reordering():
    target = np.zeros((2, 2, 2), dtype=np.float64)
    prediction = np.stack(
        [np.ones((2, 2), dtype=np.float64), -np.ones((2, 2), dtype=np.float64)]
    )
    season = np.array(["winter", "summer"])
    elevation = np.array([["low", "high"], ["low", "high"]])

    masks = strata_masks(season, target.shape)
    assert list(masks) == ["winter", "summer"]
    assert masks["winter"].sum() == 4
    assert masks["summer"].sum() == 4

    results = stratified_metrics(
        prediction,
        target,
        {"season": season, "elevation": elevation},
        metric_kwargs={"quantiles": (), "sample_axis": 0},
    )
    assert results["season"]["winter"]["mean_bias"] == pytest.approx(1.0)
    assert results["season"]["summer"]["mean_bias"] == pytest.approx(-1.0)
    assert results["elevation"]["low"]["valid_count"] == 4
    assert results["elevation"]["high"]["valid_count"] == 4
