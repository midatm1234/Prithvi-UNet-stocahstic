"""Synthetic contract tests for CORDEX refinement evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

cftime = pytest.importorskip("cftime")

from examples.CORDEX_ML.utils.evaluate_refinement_outputs import (
    RefinementEvaluationError,
    align_exact_timestamps,
    convert_dataarray_units,
    evaluate_variable,
    evaluate_refinement_datasets,
    evaluate_refinement_files,
    validate_refinement_output,
)
from granitewxc.refinement.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    REFINEMENT_CONTRACT_VERSION,
)


def _state_fingerprint(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(
            value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def _contract_fingerprint(contract):
    encoded = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _synthetic_datasets(
    tmp_path: Path,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    checkpoint_path = tmp_path / "refinement.ckpt"
    phase1_fingerprint = hashlib.sha256(b"synthetic phase1 state").hexdigest()
    contract = {
        "contract_version": REFINEMENT_CONTRACT_VERSION,
        "refinement_type": "diffusion_transformer",
        "residual_contract": "physical_ground_truth_minus_phase1_v1",
        "residual_normalization": {
            "method": "standardize",
            "epsilon": 1.0e-6,
            "minimum_scale": 1.0e-4,
            "require_fitted": True,
        },
    }
    contract_fingerprint = _contract_fingerprint(contract)
    normalizer_state = {
        "residual_normalizer.mean": torch.tensor([[[[0.0]]]], dtype=torch.float64),
        "residual_normalizer.scale": torch.tensor([[[[1.0]]]], dtype=torch.float64),
        "residual_normalizer.count": torch.tensor([[[[48.0]]]], dtype=torch.float64),
        "residual_normalizer._m2": torch.tensor([[[[47.0]]]], dtype=torch.float64),
        "residual_normalizer.fitted": torch.tensor(True),
    }
    checkpoint_payload = {
        "checkpoint_kind": "refinement",
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model": normalizer_state,
        "epoch": 3,
        "global_step": 12,
        "refinement_type": "diffusion_transformer",
        "refinement_contract": contract,
        "refinement_contract_fingerprint": contract_fingerprint,
        "residual_normalizer_state_keys": sorted(normalizer_state),
        "residual_normalizer_state_fingerprint": _state_fingerprint(
            normalizer_state
        ),
        "phase1_checkpoint": str(tmp_path / "phase1.ckpt"),
        "phase1_fingerprint": phase1_fingerprint,
        "resolved_config": {"data": {"output_vars": ["pr"]}},
    }
    torch.save(checkpoint_payload, checkpoint_path)
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    prediction_times = np.asarray(
        [
            cftime.DatetimeNoLeap(2000, 2, 27),
            cftime.DatetimeNoLeap(2000, 2, 28),
            cftime.DatetimeNoLeap(2000, 3, 1),
            cftime.DatetimeNoLeap(2000, 3, 2),
        ],
        dtype=object,
    )
    target_times = np.asarray(
        [
            "2000-02-27",
            "2000-02-28",
            "2000-02-29",
            "2000-03-01",
            "2000-03-02",
        ],
        dtype="datetime64[ns]",
    )
    lat = np.asarray([-31.0, -30.0, -29.0])
    lon = np.asarray([24.0, 25.0, 26.0, 27.0])
    spatial = np.arange(lat.size * lon.size, dtype=np.float64).reshape(
        lat.size,
        lon.size,
    )
    truth = np.stack([spatial + day for day in (1.0, 2.0, 3.0, 4.0)])
    member_zero = truth + 0.25
    member_one = truth + 0.75
    members = np.stack((member_zero, member_one), axis=1)
    phase1 = truth + 2.0
    target_with_leap = np.stack(
        (truth[0], truth[1], spatial + 999.0, truth[2], truth[3]),
        axis=0,
    )

    prediction = xr.Dataset(
        {
            "pr": xr.DataArray(
                members,
                dims=("time", "ensemble", "lat", "lon"),
                attrs={"units": "mm/day"},
            )
        },
        coords={
            "time": prediction_times,
            "ensemble": np.arange(2),
            "lat": lat,
            "lon": lon,
        },
        attrs={
            "prediction_kind": "refinement_ensemble_members",
            "refinement_type": "diffusion_transformer",
            "refinement_case": "synthetic_transformer",
            "refinement_checkpoint": str(checkpoint_path),
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "refinement_contract_version": REFINEMENT_CONTRACT_VERSION,
            "residual_contract": "physical_ground_truth_minus_phase1_v1",
            "refinement_contract_fingerprint": contract_fingerprint,
            "refinement_checkpoint_sha256": checkpoint_sha256,
            "phase1_checkpoint": str(tmp_path / "phase1.ckpt"),
            "phase1_fingerprint": phase1_fingerprint,
            "residual_normalization": (
                '{"channel_names": ["pr"], "count": [48.0], "epsilon": 1e-6, '
                '"fitted": true, "mean": [0.0], "method": "standardize", '
                '"minimum_scale": 0.0001, "require_fitted": true, '
                '"scale": [1.0]}'
            ),
            "ensemble_size": "2",
        },
    )
    baseline = xr.Dataset(
        {
            "pr": xr.DataArray(
                phase1,
                dims=("time", "lat", "lon"),
                attrs={"units": "mm/day"},
            )
        },
        coords={"time": prediction_times, "lat": lat, "lon": lon},
        attrs={
            "prediction_kind": "phase1_unet_deterministic",
            "inference_case": "synthetic_transformer",
            "phase1_checkpoint": str(tmp_path / "phase1.ckpt"),
            "phase1_fingerprint": phase1_fingerprint,
        },
    )
    target = xr.Dataset(
        {
            "pr": xr.DataArray(
                target_with_leap / 86400.0,
                dims=("time", "lat", "lon"),
                attrs={
                    "units": "kg m-2 s-1",
                    "standard_name": "precipitation_flux",
                },
            )
        },
        coords={"time": target_times, "lat": lat, "lon": lon},
    )
    return prediction, baseline, target


def test_output_selection_requires_member_provenance_and_requested_type(
    tmp_path,
):
    prediction, baseline, _ = _synthetic_datasets(tmp_path)
    selected = validate_refinement_output(
        prediction,
        variables=("pr",),
        expected_refinement_type="diffusion_transformer",
        expected_checkpoint=tmp_path / "refinement.ckpt",
    )
    assert selected.member_dim == "ensemble"
    assert selected.variables == {"pr": "pr"}
    assert selected.ensemble_size == 2

    with pytest.raises(RefinementEvaluationError, match="expected_checkpoint is required"):
        validate_refinement_output(prediction, variables=("pr",))

    with pytest.raises(RefinementEvaluationError, match="not requested"):
        validate_refinement_output(
            prediction,
            variables=("pr",),
            expected_refinement_type="flow_matching_transformer",
            expected_checkpoint=tmp_path / "refinement.ckpt",
        )
    with pytest.raises(RefinementEvaluationError, match="prediction_kind"):
        validate_refinement_output(
            baseline,
            variables=("pr",),
            expected_checkpoint=tmp_path / "refinement.ckpt",
        )

    missing_checkpoint = prediction.copy()
    del missing_checkpoint.attrs["refinement_checkpoint"]
    with pytest.raises(
        RefinementEvaluationError,
        match="refinement_checkpoint",
    ):
        validate_refinement_output(
            missing_checkpoint,
            variables=("pr",),
            expected_checkpoint=tmp_path / "refinement.ckpt",
        )

    legacy = prediction.copy()
    legacy.attrs["checkpoint_schema_version"] = 1
    with pytest.raises(RefinementEvaluationError, match="Unsupported.*schema"):
        validate_refinement_output(
            legacy,
            variables=("pr",),
            expected_checkpoint=tmp_path / "refinement.ckpt",
        )

    changed_checkpoint = tmp_path / "refinement.ckpt"
    changed_checkpoint.write_bytes(b"different checkpoint bytes")
    with pytest.raises(RefinementEvaluationError, match="current bytes"):
        validate_refinement_output(
            prediction,
            variables=("pr",),
            expected_checkpoint=changed_checkpoint,
        )


def test_output_selection_rejects_wrong_run_checkpoint_and_tensor_layout(
    tmp_path,
):
    prediction, baseline, target = _synthetic_datasets(tmp_path)

    with pytest.raises(RefinementEvaluationError, match="checkpoint does not match"):
        validate_refinement_output(
            prediction,
            variables=("pr",),
            expected_checkpoint=tmp_path / "different.ckpt",
        )
    with pytest.raises(RefinementEvaluationError, match="case .* does not match"):
        validate_refinement_output(
            prediction,
            variables=("pr",),
            expected_case="some_other_run",
            expected_checkpoint=tmp_path / "refinement.ckpt",
        )

    wrong_layout = prediction.copy()
    wrong_layout["pr"] = prediction["pr"].transpose(
        "ensemble", "time", "lat", "lon"
    )
    with pytest.raises(RefinementEvaluationError, match="must have dimensions"):
        validate_refinement_output(
            wrong_layout,
            variables=("pr",),
            expected_checkpoint=tmp_path / "refinement.ckpt",
        )

    ambiguous = prediction.copy()
    ambiguous["pr_members"] = prediction["pr"]
    with pytest.raises(RefinementEvaluationError, match="exactly one member field"):
        validate_refinement_output(
            ambiguous,
            variables=("pr",),
            expected_checkpoint=tmp_path / "refinement.ckpt",
        )

    wrong_baseline = baseline.copy()
    wrong_baseline.attrs["phase1_fingerprint"] = "different-phase1"
    with pytest.raises(RefinementEvaluationError, match="Phase-1 state"):
        evaluate_refinement_datasets(
            prediction,
            wrong_baseline,
            target,
            variables=("pr",),
            expected_checkpoint=tmp_path / "refinement.ckpt",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(channel_names=["tasmax"]), "channel_names"),
        (lambda value: value.update(scale=[0.0]), "scale must be"),
        (lambda value: value.update(count=[0.0]), "count must be"),
        (lambda value: value.update(mean=[float("nan")]), "mean must contain"),
        (lambda value: value.update(fitted="true"), "fitted must be"),
        (lambda value: value.update(method="target_scaler"), "Unsupported.*method"),
    ],
)
def test_output_selection_validates_residual_normalization_semantics(
    tmp_path,
    mutation,
    message,
):
    prediction, _, _ = _synthetic_datasets(tmp_path)
    normalization = json.loads(prediction.attrs["residual_normalization"])
    mutation(normalization)
    changed = prediction.copy(deep=True)
    changed.attrs["residual_normalization"] = json.dumps(normalization)
    with pytest.raises(RefinementEvaluationError, match=message):
        validate_refinement_output(
            changed,
            variables=("pr",),
            expected_checkpoint=tmp_path / "refinement.ckpt",
        )


def test_output_selection_authenticates_checkpoint_contract_phase1_and_state(
    tmp_path,
):
    prediction, _, _ = _synthetic_datasets(tmp_path)
    checkpoint_path = tmp_path / "refinement.ckpt"

    wrong_contract = prediction.copy(deep=True)
    wrong_contract.attrs["refinement_contract_fingerprint"] = "0" * 64
    with pytest.raises(RefinementEvaluationError, match="contract fingerprint"):
        validate_refinement_output(
            wrong_contract,
            variables=("pr",),
            expected_checkpoint=checkpoint_path,
        )

    wrong_phase1 = prediction.copy(deep=True)
    wrong_phase1.attrs["phase1_fingerprint"] = "1" * 64
    with pytest.raises(RefinementEvaluationError, match="Phase-1 fingerprint"):
        validate_refinement_output(
            wrong_phase1,
            variables=("pr",),
            expected_checkpoint=checkpoint_path,
        )

    wrong_phase1_path = prediction.copy(deep=True)
    wrong_phase1_path.attrs["phase1_checkpoint"] = str(tmp_path / "other-phase1.ckpt")
    with pytest.raises(RefinementEvaluationError, match="Phase-1 checkpoint path"):
        validate_refinement_output(
            wrong_phase1_path,
            variables=("pr",),
            expected_checkpoint=checkpoint_path,
        )

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    payload["model"]["residual_normalizer.scale"] = torch.tensor(
        [[[[2.0]]]], dtype=torch.float64
    )
    torch.save(payload, checkpoint_path)
    tampered = prediction.copy(deep=True)
    tampered.attrs["refinement_checkpoint_sha256"] = hashlib.sha256(
        checkpoint_path.read_bytes()
    ).hexdigest()
    with pytest.raises(RefinementEvaluationError, match="state fingerprint"):
        validate_refinement_output(
            tampered,
            variables=("pr",),
            expected_checkpoint=checkpoint_path,
        )


def test_exact_calendar_intersection_drops_leap_day_without_date_shift(
    tmp_path,
):
    prediction, baseline, target = _synthetic_datasets(tmp_path)
    aligned = align_exact_timestamps(prediction, baseline, target)

    assert aligned.report["matched_count"] == 4
    assert aligned.report["target_dropped_count"] == 1
    np.testing.assert_array_equal(
        aligned.target["time"].values,
        np.asarray(
            ["2000-02-27", "2000-02-28", "2000-03-01", "2000-03-02"],
            dtype="datetime64[ns]",
        ),
    )
    expected_daily_means = np.asarray((6.5, 7.5, 8.5, 9.5)) / 86400.0
    np.testing.assert_allclose(
        aligned.target["pr"].mean(("lat", "lon")).values,
        expected_daily_means,
    )

    wrong_period = target.assign_coords(
        time=np.asarray(
            [
                "1961-01-01",
                "1961-01-02",
                "1961-01-03",
                "1961-01-04",
                "1961-01-05",
            ],
            dtype="datetime64[ns]",
        )
    )
    with pytest.raises(RefinementEvaluationError, match="missing 4"):
        align_exact_timestamps(prediction, baseline, wrong_period)


def test_precipitation_flux_conversion_is_applied_exactly_once(tmp_path):
    _, _, target = _synthetic_datasets(tmp_path)
    converted, first = convert_dataarray_units(
        target["pr"],
        "mm/day",
        variable="pr",
    )
    converted_again, second = convert_dataarray_units(
        converted,
        "mm/day",
        variable="pr",
    )

    assert first["applied"] is True
    assert first["factor"] == 86400.0
    assert second["applied"] is False
    assert second["factor"] == 1.0
    np.testing.assert_allclose(converted_again.values, converted.values)
    assert converted_again.attrs["units"] == "mm/day"


def test_tasmax_kelvin_celsius_conversion_uses_declared_units():
    kelvin = xr.DataArray(
        np.asarray([273.15, 300.0]),
        dims=("sample",),
        attrs={"units": "K"},
    )
    celsius, to_celsius = convert_dataarray_units(
        kelvin,
        "degC",
        variable="tasmax",
    )
    restored, to_kelvin = convert_dataarray_units(
        celsius,
        "K",
        variable="tasmax",
    )
    np.testing.assert_allclose(celsius.values, [0.0, 26.85], atol=1.0e-12)
    np.testing.assert_allclose(restored.values, kelvin.values, atol=1.0e-12)
    assert to_celsius["offset"] == -273.15
    assert to_kelvin["offset"] == 273.15


def test_synthetic_evaluation_metrics_and_artifacts(tmp_path):
    pytest.importorskip("matplotlib")
    prediction, baseline, target = _synthetic_datasets(tmp_path)
    summary, maps = evaluate_refinement_datasets(
        prediction,
        baseline,
        target,
        variables=("pr",),
        expected_refinement_type="diffusion_transformer",
        expected_checkpoint=tmp_path / "refinement.ckpt",
        chunk_time=2,
        distribution_samples=10_000,
        spectral_bins=4,
    )
    metrics = summary["variables"]["pr"]
    assert metrics["ensemble_mean_skill"]["mean_bias"] == pytest.approx(0.5)
    assert metrics["ensemble_mean_skill"]["mae"] == pytest.approx(0.5)
    assert metrics["ensemble_mean_skill"]["rmse"] == pytest.approx(0.5)
    assert metrics["phase1_skill"]["rmse"] == pytest.approx(2.0)
    assert metrics["skill_change_refined_minus_phase1"][
        "rmse"
    ] == pytest.approx(-1.5)
    assert metrics["unit_conversion"]["target_to_prediction_units"][
        "factor"
    ] == 86400.0
    assert maps["pr"]["member_climatologies"].shape == (2, 3, 4)

    prediction_path = tmp_path / "prediction.nc"
    baseline_path = tmp_path / "baseline.nc"
    target_path = tmp_path / "target.nc"
    prediction.to_netcdf(prediction_path)
    baseline.to_netcdf(baseline_path)
    target.to_netcdf(target_path)
    output_dir = tmp_path / "evaluation"
    file_summary = evaluate_refinement_files(
        prediction_path=prediction_path,
        baseline_path=baseline_path,
        target_path=target_path,
        variables=("pr",),
        expected_refinement_type="diffusion_transformer",
        expected_checkpoint=tmp_path / "refinement.ckpt",
        output_dir=output_dir,
        chunk_time=2,
        distribution_samples=10_000,
        spectral_bins=4,
    )

    for path in file_summary["artifacts"].values():
        assert Path(path).is_file()
    assert (output_dir / "refinement_evaluation_summary.json").is_file()
    assert (output_dir / "refinement_evaluation_metrics.csv").is_file()
    assert (output_dir / "pr_refinement_8_panel.png").is_file()
    assert (output_dir / "pr_representative_members.png").is_file()


def test_evaluation_uses_one_common_validity_mask_for_all_products():
    time = np.asarray(["2000-01-01"], dtype="datetime64[ns]")
    lat = np.asarray([-30.0, -29.0])
    lon = np.asarray([24.0, 25.0])
    truth = np.asarray([[[0.0, 2.0], [np.nan, 0.0]]])
    phase1 = np.asarray([[[0.0, 2.0], [100.0, 0.0]]])
    members_values = np.stack((phase1, phase1), axis=1)
    members_values[0, 1, 1, 1] = np.nan
    coords = {"time": time, "lat": lat, "lon": lon}
    members = xr.DataArray(
        members_values,
        dims=("time", "ensemble", "lat", "lon"),
        coords={**coords, "ensemble": [0, 1]},
        attrs={"units": "mm/day"},
    )
    baseline = xr.DataArray(
        phase1,
        dims=("time", "lat", "lon"),
        coords=coords,
        attrs={"units": "mm/day"},
    )
    target = xr.DataArray(
        truth,
        dims=("time", "lat", "lon"),
        coords=coords,
        attrs={"units": "mm/day"},
    )

    metrics, maps = evaluate_variable(
        members,
        baseline,
        target,
        member_dim="ensemble",
        variable="pr",
        chunk_time=1,
        distribution_samples=100,
        spectral_bins=2,
        wet_day_threshold=1.0,
    )
    assert metrics["evaluation_mask"] == {
        "definition": (
            "finite ground truth, Phase-1 prediction, and every physical "
            "ensemble member"
        ),
        "total_count": 4,
        "target_finite_count": 3,
        "phase1_finite_count": 4,
        "all_members_finite_count": 3,
        "common_valid_count": 2,
        "excluded_count": 2,
    }
    assert metrics["ensemble_mean_skill"]["valid_count"] == 2
    assert metrics["phase1_skill"]["valid_count"] == 2
    assert metrics["wet_day"]["ensemble_mean_frequency"] == pytest.approx(0.5)
    assert metrics["wet_day"]["phase1_frequency"] == pytest.approx(0.5)
    assert metrics["wet_day"]["target_frequency"] == pytest.approx(0.5)
    assert metrics["wet_day"]["threshold_units"] == "mm/day"
    assert metrics["wet_day"]["ensemble_mean_contingency"] == {
        "valid_count": 2,
        "target_wet_count": 1,
        "target_dry_count": 1,
        "false_wet_count": 0,
        "missed_wet_count": 0,
        "false_wet_day_rate": 0.0,
        "missed_wet_day_rate": 0.0,
    }
    assert (
        metrics["wet_day"]["phase1_contingency"]
        == metrics["wet_day"]["ensemble_mean_contingency"]
    )
    for name in ("ground_truth", "phase1", "ensemble_mean", "ensemble_spread"):
        assert np.isnan(maps[name][1, 0])
        assert np.isnan(maps[name][1, 1])
    assert np.isnan(maps["member_climatologies"][:, 1, :]).all()

    with pytest.raises(RefinementEvaluationError, match="finite and non-negative"):
        evaluate_variable(
            members,
            baseline,
            target,
            member_dim="ensemble",
            variable="pr",
            wet_day_threshold=-1.0,
        )


def test_evaluation_rejects_non_daily_precipitation_output_units(tmp_path):
    prediction, baseline, target = _synthetic_datasets(tmp_path)
    prediction["pr"].attrs["units"] = "kg m-2 s-1"
    baseline["pr"].attrs["units"] = "kg m-2 s-1"
    with pytest.raises(RefinementEvaluationError, match="must use physical daily units"):
        evaluate_refinement_datasets(
            prediction,
            baseline,
            target,
            variables=("pr",),
            expected_refinement_type="diffusion_transformer",
            expected_checkpoint=tmp_path / "refinement.ckpt",
        )
