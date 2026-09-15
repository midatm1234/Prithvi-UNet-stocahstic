from __future__ import annotations

import os
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import xarray as xr


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples/CORDEX_ML/notebooks/SA_downscaling_inference_T2_ACCESS-CM2_static.py"
)


@pytest.fixture(scope="module")
def helpers() -> SimpleNamespace:
    previous_cwd = Path.cwd()
    try:
        namespace = runpy.run_path(str(SCRIPT))
    finally:
        os.chdir(previous_cwd)
    names = (
        "_attach_cf_auxiliaries",
        "_build_constraint_diagnostics_dataset",
        "_build_netcdf_encoding",
        "_check_output_disk_space",
        "_concat_time_auxiliary",
        "_load_template_cf_auxiliaries",
        "_model_product_global_attrs",
        "_physical_output_variable_attrs",
        "_planned_output_artifacts",
        "_prediction_output_root",
        "_refinement_output_header",
        "_refuse_existing_output_artifacts",
        "_run_full_inference",
    )
    return SimpleNamespace(**{name: namespace[name] for name in names})


def test_output_root_can_use_another_drive(tmp_path, helpers, monkeypatch):
    namespace = helpers._prediction_output_root.__globals__
    run_dir = tmp_path / "run"
    subdir = Path("historical/perfect")
    monkeypatch.setitem(namespace, "REFINEMENT_OUTPUT_ROOT", None)
    assert helpers._prediction_output_root(run_dir, "case", subdir) == (
        run_dir / "predictions/case" / subdir
    )
    alternate = tmp_path / "alternate"
    monkeypatch.setitem(namespace, "REFINEMENT_OUTPUT_ROOT", str(alternate))
    assert helpers._prediction_output_root(run_dir, "case", subdir) == (
        alternate / "case" / subdir
    )


@pytest.mark.parametrize("save_pickles", [False, True])
def test_output_budget_includes_whole_bundle_and_rejects_low_space(
    tmp_path, helpers, monkeypatch, save_pickles
):
    namespace = helpers._check_output_disk_space.__globals__
    monkeypatch.setitem(namespace, "SAVE_PICKLE_OUTPUTS", save_pickles)
    field_bytes = 100 * 20 * 30 * 2 * 4
    expected = int(field_bytes * (14 + (11 if save_pickles else 0)) * 1.05)
    expected += namespace["OUTPUT_DISK_RESERVE_BYTES"]
    args = dict(
        sample_count=100, target_shape=(20, 30), variable_count=2, ensemble_size=10
    )
    monkeypatch.setattr(
        namespace["shutil"], "disk_usage", lambda path: SimpleNamespace(free=expected)
    )
    assert helpers._check_output_disk_space(tmp_path, **args) == expected
    artifacts = helpers._planned_output_artifacts(tmp_path, Path("output.nc"))
    assert ("ensemble_pickle" in artifacts) == save_pickles
    assert ("pre_inverse_pickle" in artifacts) == save_pickles
    monkeypatch.setattr(
        namespace["shutil"], "disk_usage",
        lambda path: SimpleNamespace(free=expected - 1),
    )
    with pytest.raises(OSError, match="GRANITE_REFINEMENT_OUTPUT_ROOT") as exc:
        helpers._check_output_disk_space(tmp_path, **args)
    assert exc.value.errno == 28
    assert not list(tmp_path.iterdir())


def test_compressed_prediction_roundtrip_preserves_float32(tmp_path, helpers):
    import h5py

    values = np.random.default_rng(42).normal(size=(3, 2, 4, 5)).astype(np.float32)
    values[0, 0, 0, 0] = np.nan
    values[1, 0, 0, 0] = np.float32(1e-9)
    dataset = xr.Dataset({"tasmax": (("time", "ensemble", "lat", "lon"), values)})
    path = tmp_path / "prediction.nc"
    dataset.to_netcdf(
        path, engine="h5netcdf", encoding=helpers._build_netcdf_encoding(["tasmax"])
    )
    with h5py.File(path) as stored:
        assert stored["tasmax"].compression == "gzip"
        assert stored["tasmax"].shuffle
        assert stored["tasmax"].chunks is not None
    with xr.open_dataset(path, engine="h5netcdf") as restored:
        assert restored.tasmax.dtype == np.float32
        np.testing.assert_array_equal(restored.tasmax.values, values)


def test_existing_case_artifact_is_never_overwritten(tmp_path, helpers):
    artifacts = helpers._planned_output_artifacts(
        tmp_path,
        Path("Predictions_pr_tasmax_ACCESS-CM2_1981-2000.nc"),
    )
    assert artifacts["prediction_netcdf"].parent == tmp_path
    assert artifacts["baseline_netcdf"].name.endswith(".baseline.nc")
    assert artifacts["constraint_diagnostics_netcdf"].name.endswith(
        ".constraint_diagnostics.nc"
    )
    assert len(set(artifacts.values())) == len(artifacts)
    helpers._refuse_existing_output_artifacts(artifacts)

    for artifact_name in artifacts:
        case_root = tmp_path / artifact_name
        case_root.mkdir()
        case_artifacts = helpers._planned_output_artifacts(
            case_root,
            Path("Predictions_pr_tasmax_ACCESS-CM2_1981-2000.nc"),
        )
        case_artifacts[artifact_name].touch()
        with pytest.raises(FileExistsError, match="Refusing to overwrite") as exc:
            helpers._refuse_existing_output_artifacts(case_artifacts)
        assert artifact_name in str(exc.value)
        assert str(case_artifacts[artifact_name]) in str(exc.value)


@pytest.mark.parametrize("stub", [Path("nested/output.nc"), Path("output.pkl")])
def test_output_stub_must_be_one_netcdf_filename(tmp_path, helpers, stub):
    with pytest.raises(ValueError, match="PREDICTION_OUTPUT_NAMES"):
        helpers._planned_output_artifacts(tmp_path, stub)


def test_schema2_output_header_preserves_new_cases_and_rejects_legacy(
    helpers,
    monkeypatch,
):
    function_globals = helpers._refinement_output_header.__globals__
    config = SimpleNamespace(
        model=SimpleNamespace(refinement={"type": "flow_matching_unet"})
    )
    monkeypatch.setitem(function_globals, "REFINEMENT_OUTPUT_HEADER", None)
    assert helpers._refinement_output_header(
        config,
        "SA_T2_flow_matching_unet_schema2_precip_mean_fixed",
    ) == "SA_T2_flow_matching_unet_schema2_precip_mean_fixed"
    assert helpers._refinement_output_header(
        config,
        "SA_T2_flow_matching_unet_schema2_fixed",
    ) == "SA_T2_flow_matching_unet_schema2_fixed"
    assert helpers._refinement_output_header(
        config,
        "SA_T2_flow_matching_unet",
    ) == "SA_T2_flow_matching_unet_schema2_fixed"

    for header in (
        "SA_T2_flow_matching_transformer_schema2_precip_mean_fixed",
        "SA-T2.flow_schema2_experiment-2_fixed",
    ):
        monkeypatch.setitem(function_globals, "REFINEMENT_OUTPUT_HEADER", header)
        assert helpers._refinement_output_header(config, "ignored") == header

    for unsafe_or_legacy in (
        "SA_T2_flow_matching_unet",
        "SA_T2_flow_matching_unet_schema1_fixed",
        "SA_T2_flow_matching_unet_schema2_precip_mean",
        "nested/SA_T2_flow_matching_unet_schema2_fixed",
        "SA:T2_schema2_fixed",
    ):
        monkeypatch.setitem(
            function_globals,
            "REFINEMENT_OUTPUT_HEADER",
            unsafe_or_legacy,
        )
        with pytest.raises(ValueError, match="GRANITE_REFINEMENT_OUTPUT_HEADER"):
            helpers._refinement_output_header(config, "ignored")


def _cf_template() -> xr.Dataset:
    time = xr.DataArray(
        np.array([0, 1], dtype=np.int32),
        dims="time",
        attrs={"standard_name": "time", "bounds": "time_bnds"},
    )
    lat = xr.DataArray(
        np.array([-30.0, -29.0], dtype=np.float32),
        dims="lat",
        attrs={"units": "degrees_north", "bounds": "lat_bnds"},
    )
    lon = xr.DataArray(
        np.array([25.0, 26.0, 27.0], dtype=np.float32),
        dims="lon",
        attrs={"units": "degrees_east", "bounds": "lon_bnds"},
    )
    return xr.Dataset(
        data_vars={
            "pr": xr.DataArray(
                np.zeros((2, 2, 3), dtype=np.float32),
                dims=("time", "lat", "lon"),
                attrs={"units": "kg m-2 s-1", "grid_mapping": "crs"},
            ),
            "time_bnds": xr.DataArray(
                np.array([[-1, 0], [0, 1]], dtype=np.int32),
                dims=("time", "bnds"),
            ),
            "lat_bnds": xr.DataArray(
                np.array([[-30.5, -29.5], [-29.5, -28.5]], dtype=np.float32),
                dims=("lat", "bnds"),
            ),
            "lon_bnds": xr.DataArray(
                np.array(
                    [[24.5, 25.5], [25.5, 26.5], [26.5, 27.5]],
                    dtype=np.float32,
                ),
                dims=("lon", "bnds"),
            ),
            "crs": xr.DataArray(
                np.int32(0),
                attrs={"grid_mapping_name": "latitude_longitude"},
            ),
        },
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={
            "Conventions": "CF-1.11",
            "title": "CCAM simulation data",
            "source": "truth-producing climate simulation",
            "history": "truth preprocessing history",
        },
    )


def test_physical_output_units_are_channel_aligned_and_required(helpers):
    attrs = helpers._physical_output_variable_attrs(
        {
            "pr": {"units": "kg m-2 s-1", "standard_name": "precipitation_flux"},
            "tasmax": {"units": "K", "standard_name": "air_temperature"},
        },
        ["pr", "tasmax"],
        ["mm/day", "K"],
    )
    assert attrs["pr"]["units"] == "mm/day"
    assert attrs["pr"]["source_units"] == "kg m-2 s-1"
    assert "exactly once" in attrs["pr"]["unit_conversion"]
    assert attrs["tasmax"]["units"] == "K"
    assert "source_units" not in attrs["tasmax"]

    with pytest.raises(ValueError, match="align one-to-one"):
        helpers._physical_output_variable_attrs({}, ["pr", "tasmax"], ["mm/day"])
    with pytest.raises(ValueError, match="missing.*tasmax"):
        helpers._physical_output_variable_attrs({}, ["pr", "tasmax"], ["mm/day", ""])


def test_constraint_sidecar_round_trip_has_units_provenance_and_layout(
    tmp_path,
    helpers,
):
    template = _cf_template()
    template["tasmax"] = xr.DataArray(
        np.full((2, 2, 3), 300.0, dtype=np.float32),
        dims=("time", "lat", "lon"),
        attrs={
            "units": "K",
            "standard_name": "air_temperature",
            "grid_mapping": "crs",
        },
    )
    target_vars = ["pr", "tasmax"]
    target_attrs = helpers._physical_output_variable_attrs(
        {name: dict(template[name].attrs) for name in target_vars},
        target_vars,
        ["mm/day", "K"],
    )
    coords = {name: template[name] for name in ("time", "lat", "lon")}
    auxiliaries = helpers._load_template_cf_auxiliaries(
        template,
        target_attrs,
        coords,
    )

    shape = (2, 2, 2, 3)
    unbounded = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    clipped = unbounded + np.float32(0.25)
    shift = clipped - unbounded
    provenance = {
        "checkpoint_schema_version": 2,
        "refinement_contract_version": 2,
        "residual_contract": "target_residual = ground_truth - phase1",
        "refinement_contract_fingerprint": "contract-sha",
        "refinement_checkpoint_sha256": "checkpoint-sha",
        "phase1_checkpoint": "phase1.ckpt",
        "phase1_fingerprint": "phase1-sha",
        "refinement_config_path": "flow.yaml",
        "residual_normalization": "{}",
        "refinement_seed": 17,
    }
    dataset = helpers._build_constraint_diagnostics_dataset(
        target_vars=target_vars,
        unbounded_ensemble_means=unbounded,
        memberwise_clipped_means=clipped,
        memberwise_clipping_mean_shifts=shift,
        coords=coords,
        time_dim="time",
        lat_dim="lat",
        lon_dim="lon",
        target_attrs=target_attrs,
        template_attrs=dict(template.attrs),
        target_template_path=tmp_path / "truth.nc",
        refinement_type="flow_matching_transformer",
        refinement_case="sa_flow_schema2_fixed",
        output_provenance=provenance,
        nonnegative_ensemble_strategy="mean_preserving",
        cf_auxiliaries=auxiliaries,
    )

    diagnostic_names = [
        f"{name}_{kind}"
        for name in target_vars
        for kind in (
            "unbounded_ensemble_mean",
            "memberwise_clipped_mean",
            "memberwise_clipping_mean_shift",
        )
    ]
    for diagnostic_name in diagnostic_names:
        variable = dataset[diagnostic_name]
        assert variable.dims == ("time", "lat", "lon")
        assert variable.dtype == np.float32
        assert variable.attrs["refinement_space"] == "physical"
        assert "standard_name" not in variable.attrs
    assert dataset["pr_unbounded_ensemble_mean"].attrs["units"] == "mm/day"
    assert dataset["pr_unbounded_ensemble_mean"].attrs["source_units"] == "kg m-2 s-1"
    assert dataset["tasmax_unbounded_ensemble_mean"].attrs["units"] == "K"
    assert dataset.attrs["product_kind"] == "refinement_constraint_diagnostics"
    assert dataset.attrs["refinement_contract_fingerprint"] == "contract-sha"
    assert dataset.attrs["phase1_fingerprint"] == "phase1-sha"
    assert dataset.attrs["nonnegative_ensemble_strategy"] == "mean_preserving"
    assert dataset.attrs["source"].startswith("Granite-WxC flow_matching_transformer")
    assert dataset.attrs["source"] != template.attrs["source"]
    np.testing.assert_array_equal(
        dataset["tasmax_memberwise_clipping_mean_shift"],
        shift[:, 1],
    )

    path = tmp_path / "prediction.constraint_diagnostics.nc"
    dataset.to_netcdf(
        path,
        engine="h5netcdf",
        encoding=helpers._build_netcdf_encoding(diagnostic_names),
    )
    with xr.open_dataset(path, engine="h5netcdf") as restored:
        assert restored.attrs["product_kind"] == "refinement_constraint_diagnostics"
        assert restored.attrs["refinement_checkpoint_sha256"] == "checkpoint-sha"
        assert restored["pr_unbounded_ensemble_mean"].attrs["units"] == "mm/day"
        assert restored["tasmax_unbounded_ensemble_mean"].attrs["units"] == "K"
        assert restored["pr_unbounded_ensemble_mean"].dims == ("time", "lat", "lon")
        assert {"crs", "lat_bnds", "lon_bnds", "time_bnds"} <= set(
            restored.variables
        )


def test_constraint_sidecar_rejects_layout_or_missing_units(helpers, tmp_path):
    coords = {
        "time": np.arange(2),
        "lat": np.arange(2),
        "lon": np.arange(3),
    }
    common = {
        "target_vars": ["pr", "tasmax"],
        "memberwise_clipped_means": np.zeros((2, 2, 2, 3), dtype=np.float32),
        "memberwise_clipping_mean_shifts": np.zeros((2, 2, 2, 3), dtype=np.float32),
        "coords": coords,
        "time_dim": "time",
        "lat_dim": "lat",
        "lon_dim": "lon",
        "template_attrs": {},
        "target_template_path": tmp_path / "truth.nc",
        "refinement_type": "flow_matching_unet",
        "refinement_case": "sa_flow_schema2_fixed",
        "output_provenance": {},
        "nonnegative_ensemble_strategy": "mean_preserving",
        "cf_auxiliaries": {},
    }
    with pytest.raises(ValueError, match="layout"):
        helpers._build_constraint_diagnostics_dataset(
            unbounded_ensemble_means=np.zeros((2, 2, 3), dtype=np.float32),
            target_attrs={"pr": {"units": "mm/day"}, "tasmax": {"units": "K"}},
            **common,
        )
    with pytest.raises(ValueError, match="missing.*tasmax"):
        helpers._build_constraint_diagnostics_dataset(
            unbounded_ensemble_means=np.zeros((2, 2, 2, 3), dtype=np.float32),
            target_attrs={"pr": {"units": "mm/day"}, "tasmax": {"units": ""}},
            **common,
        )


def test_cf_auxiliaries_round_trip_without_changing_physical_layout(
    tmp_path,
    helpers,
):
    template = _cf_template()
    output = xr.Dataset(
        coords={
            "time": template["time"],
            "ensemble": np.arange(2, dtype=np.int32),
            "lat": template["lat"],
            "lon": template["lon"],
        }
    )
    output["pr"] = xr.DataArray(
        np.ones((2, 2, 2, 3), dtype=np.float32),
        dims=("time", "ensemble", "lat", "lon"),
        coords=output.coords,
        attrs={"units": "mm/day", "grid_mapping": "crs"},
    )
    auxiliaries = helpers._load_template_cf_auxiliaries(
        template,
        {"pr": dict(output["pr"].attrs)},
        {name: output[name] for name in ("time", "lat", "lon")},
    )
    copied, removed = helpers._attach_cf_auxiliaries(output, auxiliaries)

    assert set(copied) == {"crs", "lat_bnds", "lon_bnds", "time_bnds"}
    assert removed == []
    assert output["pr"].dims == ("time", "ensemble", "lat", "lon")
    assert output["pr"].dtype == np.float32
    assert output["pr"].attrs["units"] == "mm/day"
    for variable, attribute in (
        ("pr", "grid_mapping"),
        ("time", "bounds"),
        ("lat", "bounds"),
        ("lon", "bounds"),
    ):
        reference = output[variable].attrs[attribute]
        assert reference in output.variables

    path = tmp_path / "cf_output.nc"
    output.to_netcdf(path, engine="h5netcdf")
    with xr.open_dataset(path, engine="h5netcdf") as restored:
        assert restored["pr"].dims == ("time", "ensemble", "lat", "lon")
        assert restored["pr"].attrs["units"] == "mm/day"
        assert restored["pr"].attrs["grid_mapping"] == "crs"
        assert {"crs", "lat_bnds", "lon_bnds", "time_bnds"} <= set(
            restored.variables
        )


def test_incompatible_or_missing_cf_references_are_removed(helpers):
    output = xr.Dataset(
        data_vars={
            "pr": xr.DataArray(
                np.zeros((2, 2, 3), dtype=np.float32),
                dims=("time", "lat", "lon"),
                attrs={"units": "mm/day", "grid_mapping": "missing_crs"},
            )
        },
        coords={
            "time": xr.DataArray(
                np.arange(2),
                dims="time",
                attrs={"bounds": "time_bnds"},
            ),
            "lat": np.arange(2),
            "lon": xr.DataArray(
                np.arange(3),
                dims="lon",
                attrs={"bounds": "missing_lon_bnds"},
            ),
        },
    )
    incompatible = {
        "time_bnds": xr.DataArray(
            np.zeros((3, 2), dtype=np.float32),
            dims=("time", "bnds"),
        )
    }
    copied, removed = helpers._attach_cf_auxiliaries(output, incompatible)
    assert copied == []
    assert set(removed) == {"pr:grid_mapping", "time:bounds", "lon:bounds"}
    assert "grid_mapping" not in output["pr"].attrs
    assert "bounds" not in output["time"].attrs
    assert "bounds" not in output["lon"].attrs


def test_global_metadata_identifies_model_products_not_truth(helpers, tmp_path):
    truth = {
        "Conventions": "CF-1.11",
        "title": "CCAM simulation data",
        "source": "truth-producing climate simulation",
        "history": "truth preprocessing history",
    }
    common = {
        "target_template_path": tmp_path / "truth.nc",
        "refinement_type": "flow_matching_transformer",
        "refinement_case": "sa_flow_schema2_fixed",
    }
    ensemble = helpers._model_product_global_attrs(
        truth,
        product_kind="refinement_ensemble_members",
        **common,
    )
    baseline = helpers._model_product_global_attrs(
        truth,
        product_kind="phase1_unet_deterministic",
        **common,
    )
    constraints = helpers._model_product_global_attrs(
        truth,
        product_kind="refinement_constraint_diagnostics",
        **common,
    )

    assert ensemble["Conventions"] == "CF-1.11"
    assert ensemble["title"].startswith("Granite-WxC stochastic")
    assert ensemble["source"].startswith("Granite-WxC")
    assert ensemble["history"] != truth["history"]
    assert ensemble["input_target_template_title"] == truth["title"]
    assert "input_target_template_history" not in ensemble
    assert baseline["title"].startswith("Granite-WxC deterministic")
    assert baseline["source"] == "Granite-WxC Phase-1 U-Net inference"
    assert constraints["title"].startswith(
        "Granite-WxC residual-refinement physical-constraint"
    )
    assert constraints["product_kind"] == "refinement_constraint_diagnostics"
    assert constraints["source"].startswith("Granite-WxC flow_matching_transformer")


def test_mocked_two_phase_output_constraint_fields_are_collected(
    helpers,
    monkeypatch,
):
    class MockRefinementModel:
        def __init__(self, *, omit_shift: bool = False):
            self._last_checkpoint_loaded = True
            self.phase1 = SimpleNamespace(
                n_input_timestamps=1,
                input_scalers_mu=torch.zeros(1),
                input_scalers_sigma=torch.ones(1),
                input_scalers_epsilon=1e-6,
            )
            self.refinement_config = SimpleNamespace(seed=41)
            self.omit_shift = omit_shift
            self.last_output = None
            self.last_predict_call = None

        def eval(self):
            return self

        def predict(self, batch, *, ensemble_size, seed, return_members):
            batch_size, _, height, width = batch["x"].shape
            channels = batch["y"].shape[1]
            deterministic = torch.full(
                (batch_size, channels, height, width),
                2.0,
                device=batch["x"].device,
            )
            member_offsets = torch.tensor([-0.5, 0.0, 0.5], device=batch["x"].device)
            members = deterministic[:, None] + member_offsets.view(1, 3, 1, 1, 1)
            unbounded = deterministic + 0.125
            clipped = deterministic + 0.25
            shift = clipped - unbounded
            output = SimpleNamespace(
                deterministic=deterministic,
                refined=members.mean(dim=1),
                refined_normalized=torch.zeros_like(deterministic),
                members=members,
                unbounded_ensemble_mean=unbounded,
                memberwise_clipped_mean=clipped,
                memberwise_clipping_mean_shift=None if self.omit_shift else shift,
            )
            self.last_output = output
            self.last_predict_call = (ensemble_size, seed, return_members)
            return output

    function_globals = helpers._run_full_inference.__globals__
    monkeypatch.setitem(function_globals, "REFINEMENT_ENSEMBLE_SIZE", 3)
    batch = {
        "x": torch.ones(2, 1, 2, 3),
        "y": torch.zeros(2, 2, 2, 3),
        "__target_valid_mask": torch.ones(2, 2, 2, 3, dtype=torch.bool),
    }
    model = MockRefinementModel()
    result = helpers._run_full_inference(
        [batch],
        model,
        torch.device("cpu"),
        ["pr", "tasmax"],
        SimpleNamespace(enabled=False),
    )
    assert model.last_predict_call == (3, 41, True)
    torch.testing.assert_close(
        result["unbounded_ensemble_means"],
        model.last_output.unbounded_ensemble_mean,
    )
    torch.testing.assert_close(
        result["memberwise_clipped_means"],
        model.last_output.memberwise_clipped_mean,
    )
    torch.testing.assert_close(
        result["memberwise_clipping_mean_shifts"],
        model.last_output.memberwise_clipping_mean_shift,
    )

    with pytest.raises(RuntimeError, match="physical-constraint diagnostics"):
        helpers._run_full_inference(
            [batch],
            MockRefinementModel(omit_shift=True),
            torch.device("cpu"),
            ["pr", "tasmax"],
            SimpleNamespace(enabled=False),
        )


def test_predictor_time_bounds_are_concatenated(tmp_path, helpers):
    paths = []
    for index in range(2):
        time = np.array([index * 2, index * 2 + 1], dtype=np.int32)
        dataset = xr.Dataset(
            data_vars={
                "time_bnds": xr.DataArray(
                    np.stack([time, time + 1], axis=-1),
                    dims=("time", "bnds"),
                )
            },
            coords={"time": time},
        )
        path = tmp_path / f"predictor_{index}.nc"
        dataset.to_netcdf(path, engine="h5netcdf")
        paths.append(str(path))

    combined = helpers._concat_time_auxiliary(paths, "time", "time_bnds")
    assert combined is not None
    assert combined.dims == ("time", "bnds")
    np.testing.assert_array_equal(combined[:, 0], np.arange(4))
    assert helpers._concat_time_auxiliary(paths, "time", "missing") is None
