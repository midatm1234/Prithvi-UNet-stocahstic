"""Independent numerical controls for regional/probabilistic refinement scoring."""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from granitewxc.refinement.diagnostics import (
    boundary_region_masks, empirical_crps, negative_value_summary,
    summarize_refinement_tensor,
)
from granitewxc.refinement.io import build_refined_dataset
from examples.CORDEX_ML.utils.evaluate_refinement_outputs import (
    _BoundedTripletSampler, _CalibrationAccumulator, evaluate_variable,
)


def _fields():
    coords = {
        "time": np.asarray(["2001-01-01", "2001-01-02", "2001-07-01"], dtype="datetime64[ns]"),
        "lat": np.linspace(-34, -22, 6), "lon": np.linspace(20, 33, 8),
    }
    truth = np.broadcast_to(np.arange(48).reshape(1, 6, 8), (3, 6, 8)).astype(float)
    baseline = truth + 2
    corrected = truth.copy()
    corrected[:, 0, :] -= 4  # geographic south for ascending latitude
    ensemble = np.stack((corrected - 0.5, corrected + 0.5), axis=1)
    target = xr.DataArray(truth, dims=("time", "lat", "lon"), coords=coords, attrs={"units": "K"})
    phase1 = xr.DataArray(baseline, dims=target.dims, coords=coords, attrs=target.attrs)
    members = xr.DataArray(ensemble, dims=("time", "member", "lat", "lon"),
                           coords={**coords, "member": [0, 1]}, attrs=target.attrs)
    return members, phase1, target


def test_geographic_edges_and_southeast_ignore_coordinate_orientation():
    lat, lon = np.linspace(-35, -20, 7), np.linspace(20, 34, 9)
    forward, distance = boundary_region_masks(lat, lon)
    reverse, reversed_distance = boundary_region_masks(lat[::-1], lon[::-1])
    for name in forward:
        np.testing.assert_array_equal(forward[name], reverse[name][::-1, ::-1])
    np.testing.assert_array_equal(distance, reversed_distance[::-1, ::-1])
    assert forward["south_1"][0].all()
    assert forward["east_1"][:, -1].all()
    assert forward["southeast_quadrant"][0, -1]
    assert not forward["southeast_quadrant"][-1, 0]
    assert forward["boundary_1"].sum() == 2 * 7 + 2 * 9 - 4
    assert set(np.flatnonzero(forward["interior_3"])) == {30, 31, 32}
    assert "boundary_16" not in forward


def test_nonmonotonic_coordinates_are_rejected():
    with pytest.raises(ValueError, match="strictly monotonic"):
        boundary_region_masks([0, 2, 1], [0, 1, 2])


@pytest.mark.parametrize("member_count", [1, 2, 7])
def test_empirical_crps_matches_explicit_pairwise_definition(member_count):
    rng = np.random.default_rng(84)
    members = rng.normal(size=(3, member_count, 4, 5))
    truth = rng.normal(size=(3, 4, 5))
    first = np.abs(members - truth[:, None]).mean(axis=1)
    second = np.abs(members[:, :, None] - members[:, None, :]).mean(axis=(1, 2)) / 2
    np.testing.assert_allclose(empirical_crps(members, truth), first - second, atol=1e-14)
    members[0, 0, 0, 0] = np.nan
    assert np.isnan(empirical_crps(members, truth)[0, 0, 0])


def test_zero_precipitation_ties_are_distributed_over_all_ranks():
    ensemble, truth = np.zeros((2, 3, 2, 2)), np.zeros((2, 2, 2))
    calibration = _CalibrationAccumulator(3)
    calibration.update(ensemble, truth, empirical_crps(ensemble, truth))
    result = calibration.result()
    assert result["crps"] == 0
    assert result["rank_histogram_counts"] == [2, 2, 2, 2]
    negatives = negative_value_summary(np.array([0, -2, 4, np.nan]))
    assert negatives["finite_count"] == 3
    assert negatives["negative_fraction"] == pytest.approx(1 / 3)
    assert negatives["mean_negative_deficit"] == pytest.approx(2 / 3)


def test_boundary_and_seasonal_metrics_expose_local_regression():
    members, baseline, target = _fields()
    metrics, maps = evaluate_variable(members, baseline, target, member_dim="member", variable="tasmax")
    regions = metrics["geographic_regions"]
    assert regions["south_1"]["ensemble_mean"]["mean_bias"] == -4
    assert regions["north_1"]["ensemble_mean"]["mean_bias"] == 0
    assert regions["interior_1"]["ensemble_mean"]["rmse"] == 0
    assert regions["south_1"]["climatology"]["ensemble_mean"]["climatological_rmse"] == 4
    assert metrics["seasonal"]["DJF"]["time_count"] == 2
    assert metrics["seasonal"]["JJA"]["time_count"] == 1
    assert metrics["seasonal"]["MAM"]["daily_skill"]["phase1"]["rmse"] is None
    assert "wet_day" not in regions["full"]
    assert metrics["ensemble"]["calibration"]["valid_count"] == 3 * 6 * 8
    assert maps["valid_sample_count"].min() == 3


def test_pairing_fingerprints_are_chunk_independent_and_detect_changes():
    members, baseline, target = _fields()
    progress = []
    first, _ = evaluate_variable(members, baseline, target, member_dim="member", variable="tasmax", chunk_time=1,
                                 progress_callback=lambda done,total: progress.append((done,total)))
    assert progress == [(1,3),(2,3),(3,3)]
    sensitivity = first["ensemble"]["finite_member_uncertainty"]["full"]
    assert sensitivity["projection_recomputed"] is False
    assert sensitivity["member_independence_assumed"] is False
    assert "conditional" in sensitivity["status"]
    second, _ = evaluate_variable(members, baseline, target, member_dim="member", variable="tasmax", chunk_time=3)
    assert first["pairing"] == second["pairing"]
    assert first["ensemble"]["calibration"]["crps"] == pytest.approx(second["ensemble"]["calibration"]["crps"])
    baseline = baseline.copy()
    baseline.values[0, 0, 0] += 1
    third, _ = evaluate_variable(members, baseline, target, member_dim="member", variable="tasmax")
    assert first["pairing"]["matched_phase1_values_sha256"] != third["pairing"]["matched_phase1_values_sha256"]


def test_uniform_distribution_sampling_avoids_fixed_column_aliasing():
    sampler = _BoundedTripletSampler(100 * 16, 100)
    values = np.tile(np.arange(16), 100)
    sampler.update(values[:501], values[:501], values[:501])
    sampler.update(values[501:], values[501:], values[501:])
    sampled, _, _ = sampler.result()
    assert set(sampled) == set(range(16))
    assert sampled.size == 100


@pytest.mark.parametrize("width", [7, 8])
def test_spectral_fraction_matches_full_fft_hermitian_energy(width):
    rng = np.random.default_rng(102)
    plane = rng.normal(size=(9, width))
    clean = plane - plane.mean()
    power = abs(np.fft.fft2(clean)) ** 2
    fy, fx = np.fft.fftfreq(9)[:, None], np.fft.fftfreq(width)[None, :]
    expected = power[np.hypot(fy, fx) >= 0.25].sum() / power.sum()
    report = summarize_refinement_tensor("test", plane[None, None], channel_names=["pr"])
    assert report["overall"]["high_frequency_spectral_power_fraction"] == pytest.approx(expected)


def test_stage_output_shapes_units_and_signed_values():
    shape = (2, 3, 4, 5)
    base = np.ones(shape)
    members = np.ones((2, 2, 3, 4, 5))
    raw = members.copy()
    raw[:, :, 0] = -2
    states = np.zeros((2, 3, 2, 3, 4, 5))
    ds = build_refined_dataset(
        variables=["pr", "tasmax", "wind"], coords={"time": [0, 1], "lat": range(4), "lon": range(5)},
        deterministic=base, members=members, members_unbounded=raw,
        member_residuals=raw - 1, member_residuals_physical=members - base[:, None],
        sampling_states=states, sampling_steps=[0, 0.5, 1],
        units={"pr": "mm/day", "tasmax": "K", "wind": "m/s"},
    )
    assert float(ds.pr_members_unbounded.min()) == -2
    assert ds.pr_member_residuals.attrs["units"] == "1"
    assert ds.tasmax_members_unbounded.attrs["units"] == "K"
    assert ds.wind_sampling_states.dims == ("time", "sampling_step", "member", "lat", "lon")
    with pytest.raises(ValueError, match="channels"):
        build_refined_dataset(variables=["pr", "tasmax", "wind"], coords={},
                              deterministic=base, members=members[:, :, :2])


@pytest.mark.parametrize("variable,units", [("pr", "mm/day"), ("tasmax", "K")])
def test_saved_preconstraint_members_quantify_changes_without_dropping_zeros(variable, units):
    template, baseline, target = _fields()
    raw = xr.ones_like(template)
    raw.attrs["units"] = units
    raw.values[0, 0, 0, 0] = -2
    post = raw.clip(min=0)
    post.attrs["units"] = units
    baseline = xr.zeros_like(baseline)
    target = xr.zeros_like(target)
    baseline.attrs["units"] = target.attrs["units"] = units
    metrics, _ = evaluate_variable(post, baseline, target, member_dim="member",
                                  variable=variable, unbounded_members=raw)
    full = metrics["pre_postprocessing_diagnostics"]["regions"]["full"]
    assert full["finite_member_count"] == post.size
    assert full["preconstraint_negative_fraction"] == pytest.approx(1 / post.size)
    assert full["mean_physical_change"] == pytest.approx(2 / post.size)
    assert full["modified_member_fraction"] == pytest.approx(1 / post.size)
    assert ("wet_threshold" in full) == (variable == "pr")


def test_paired_comparison_guard_checks_support_baseline_and_seeds():
    from copy import deepcopy
    from examples.CORDEX_ML.utils.evaluate_refinement_outputs import validate_paired_evaluations, RefinementEvaluationError
    report = {
        "variables": {"pr": {"ensemble_size": 2, "pairing": {
            "coordinate_and_time_sha256": "dates", "common_valid_mask_sha256": "mask",
            "matched_phase1_values_sha256": "baseline",
        }}},
        "seed_metadata": {"refinement_seed": 42},
    }
    validate_paired_evaluations([report, deepcopy(report)])
    changed = deepcopy(report)
    changed["variables"]["pr"]["pairing"]["common_valid_mask_sha256"] = "different"
    with pytest.raises(RefinementEvaluationError, match="common_valid_mask"):
        validate_paired_evaluations([report, changed])
    changed = deepcopy(report)
    changed["seed_metadata"] = {}
    with pytest.raises(RefinementEvaluationError, match="seeds"):
        validate_paired_evaluations([report, changed])


def test_direct_evaluator_rejects_reordered_preconstraint_coordinates():
    from examples.CORDEX_ML.utils.evaluate_refinement_outputs import RefinementEvaluationError
    members, baseline, target = _fields()
    reversed_lat = members.isel(lat=slice(None, None, -1))
    with pytest.raises(RefinementEvaluationError, match="coordinate differs"):
        evaluate_variable(members, baseline, target, member_dim="member",
                          variable="tasmax", unbounded_members=reversed_lat)


def test_shared_plot_scales_cover_every_compared_head():
    from examples.CORDEX_ML.utils.evaluate_refinement_outputs import shared_map_limits
    base = {
        "ground_truth": np.array([[0., 1.]]), "phase1": np.array([[1., 2.]]),
        "ensemble_mean": np.array([[0., 1.]]), "member_climatologies": np.array([[[0., 1.]]]),
        "phase1_bias": np.array([[1., 1.]]), "ensemble_mean_bias": np.array([[0., 0.]]),
        "correction": np.array([[-1., -1.]]), "ensemble_spread": np.array([[.2, .3]]),
    }
    other = {**base, "ensemble_mean_bias": np.array([[-4., 3.]]),
             "member_climatologies": np.array([[[-2., 9.]]])}
    limits = shared_map_limits([{"pr": base}, {"pr": other}])
    assert limits["pr"]["field"] == [-2., 9.]
    assert limits["pr"]["difference"] == [-4., 4.]
    assert limits["pr"]["spread"] == [0., .3]


def test_regional_negative_sufficient_statistics_match_explicit_member_gather():
    from examples.CORDEX_ML.utils.evaluate_refinement_outputs import _RegionalAccumulator, _member_negative_fields
    rng = np.random.default_rng(914)
    ensemble = rng.normal(size=(3, 10, 6, 8))
    ensemble[0, :, 0, 0] = np.nan
    target = np.ones((3, 6, 8))
    target[0, 0, 0] = np.nan
    mean = ensemble.mean(axis=1)
    score = np.where(np.isfinite(target), 0.0, np.nan)
    regions, _ = boundary_region_masks(np.arange(6), np.arange(8))
    summaries = _member_negative_fields(ensemble)
    for region in regions.values():
        acc = _RegionalAccumulator(1.0)
        acc.update(mean, target, target, ensemble, score, region,
                   precipitation=True, member_summary=summaries)
        explicit = ensemble[:, :, region]
        explicit = explicit[np.isfinite(explicit)]
        negative = explicit[explicit < 0]
        assert acc.member_count == explicit.size
        assert acc.negative_count == negative.size
        assert acc.negative_sum == pytest.approx(-negative.sum(), abs=1e-12)
        assert acc.minimum == (float(explicit.min()) if explicit.size else None)


def test_batched_calibration_quantiles_match_independent_interval_calls():
    rng = np.random.default_rng(12)
    ensemble = rng.normal(size=(3, 10, 6, 8))
    truth = rng.normal(size=(3, 6, 8))
    calibration = _CalibrationAccumulator(10)
    calibration.update(ensemble, truth, empirical_crps(ensemble, truth))
    result = calibration.result()
    for key, item in result["central_interval_calibration"].items():
        level = float(key)
        low, high = np.quantile(ensemble, [(1-level)/2, (1+level)/2], axis=1)
        assert item["coverage"] == pytest.approx(((truth >= low) & (truth <= high)).mean())
        assert item["mean_width"] == pytest.approx((high-low).mean())


def test_case_config_evaluation_threshold_is_explicit_and_overridable(tmp_path):
    from examples.CORDEX_ML.utils.evaluate_refinement_outputs import resolve_evaluation_wet_threshold
    path=tmp_path/"case.yaml"
    path.write_text("precip_wet_threshold: 0.01\nrefinement:\n  precipitation_wet_threshold: 0.1\n".replace("\\n","\n"))
    threshold,source=resolve_evaluation_wet_threshold(None,path)
    assert threshold == 0.01
    assert source.endswith(":precip_wet_threshold")
    assert resolve_evaluation_wet_threshold(1.0,path)[0] == 1.0
    assert "generic" in resolve_evaluation_wet_threshold(None)[1]
