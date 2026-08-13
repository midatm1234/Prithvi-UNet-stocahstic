from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import xarray as xr

from examples.CORDEX_ML.utils.diffusion_inference import (
    add_predictions_to_dataset,
    as_ensemble_mean,
    infer_batch_ensemble,
    infer_head_type,
    reset_ensemble_generators,
    residual_transformation_stages,
    resolve_ensemble_size,
    variable_array_map,
)


class _FakeModel(torch.nn.Module):
    diffusion_enabled = True


class _LinearTargetTransformModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "mu", torch.tensor([0.0, 10.0]).view(1, 2, 1, 1)
        )
        self.register_buffer(
            "sigma", torch.tensor([5.0, 2.0]).view(1, 2, 1, 1)
        )

    def _encode_targets_std(self, physical):
        return (physical - self.mu) / self.sigma

    def _decode_targets_std(self, standardized):
        return standardized * self.sigma + self.mu


@pytest.fixture(scope="module")
def inference_script():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "CORDEX_ML"
        / "notebooks"
        / "SA_downscaling_inference_T2_ACCESS-CM2_static.py"
    )
    spec = importlib.util.spec_from_file_location(
        "cordex_residual_diffusion_inference_test_module", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    original_cwd = os.getcwd()
    try:
        spec.loader.exec_module(module)
    finally:
        os.chdir(original_cwd)
    return module


def _write_target_fixture(path: Path, *, pr_units: str | None) -> None:
    dataset = xr.Dataset(
        {
            "pr": (("time", "lat", "lon"), np.zeros((1, 2, 3), dtype=np.float32)),
            "tasmax": (("time", "lat", "lon"), np.full((1, 2, 3), 300.0, dtype=np.float32)),
        },
        coords={"time": [0], "lat": [-30.0, -29.0], "lon": [27.0, 28.0, 29.0]},
    )
    if pr_units is not None:
        dataset["pr"].attrs["units"] = pr_units
    dataset["tasmax"].attrs["units"] = "K"
    dataset.to_netcdf(path, engine="h5netcdf")


def _fake_infer(*, model, batch, cfg):
    del model, cfg
    b, _, h, w = batch["x"].shape
    sample = torch.randn(b, 2, h, w, device=batch["x"].device)
    return sample, sample + 1.0, sample + 2.0


def test_residual_stage_report_preserves_normalized_and_physical_spaces():
    model = _LinearTargetTransformModel()
    baseline_std = torch.tensor(
        [[[[1.0, 2.0]], [[-1.0, 0.5]]]], dtype=torch.float32
    )
    baseline_physical = model._decode_targets_std(baseline_std)
    truth_physical = baseline_physical + torch.tensor(
        [[[[2.5, -5.0]], [[1.0, -2.0]]]], dtype=torch.float32
    )
    generated_std = torch.tensor(
        [
            [
                [[[0.1, -0.2]], [[0.5, -0.25]]],
                [[[-0.3, 0.4]], [[0.0, 0.75]]],
            ]
        ],
        dtype=torch.float32,
    )
    full_std = baseline_std[:, None] + generated_std
    final_physical = model._decode_targets_std(
        full_std.flatten(0, 1)
    ).unflatten(0, (1, 2))

    stages = residual_transformation_stages(
        model=model,
        truth_physical=truth_physical,
        final_physical=final_physical,
        full_standardized=full_std,
        generated_residual_standardized=generated_std,
        baseline_standardized=baseline_std,
        baseline_physical=baseline_physical,
    )

    torch.testing.assert_close(
        stages["physical_unet_prediction"], baseline_physical
    )
    torch.testing.assert_close(
        stages["normalized_true_residual"],
        model._encode_targets_std(truth_physical) - baseline_std,
    )
    torch.testing.assert_close(
        stages["predicted_normalized_residual"], generated_std
    )
    torch.testing.assert_close(
        stages["denormalized_predicted_residual"],
        final_physical - baseline_physical[:, None],
    )


def test_residual_stage_report_rejects_member_dependent_baseline():
    model = _LinearTargetTransformModel()
    truth = torch.ones(1, 2, 1, 2)
    residual = torch.zeros(1, 2, 2, 1, 2)
    full = torch.zeros_like(residual)
    full[:, 1] = 0.1
    final = model._decode_targets_std(full.flatten(0, 1)).unflatten(0, (1, 2))

    import pytest

    with pytest.raises(RuntimeError, match="baseline changed across"):
        residual_transformation_stages(
            model=model,
            truth_physical=truth,
            final_physical=final,
            full_standardized=full,
            generated_residual_standardized=residual,
            baseline_standardized=torch.zeros_like(truth),
        )


def test_diffusion_checkpoint_metadata_wins_over_config_default():
    checkpoint = {"metadata": {"head_type": "diffusion"}}
    config = SimpleNamespace(model=SimpleNamespace())
    assert infer_head_type(checkpoint=checkpoint, config=config, model=None) == "diffusion"


def test_diffusion_checkpoint_state_keys_trigger_detection():
    checkpoint = {"model": {"diffusion_head.net.0.weight": torch.zeros(1)}}
    config = SimpleNamespace(model=SimpleNamespace())
    assert infer_head_type(checkpoint=checkpoint, config=config, model=None) == "diffusion"


def test_diffusion_ensemble_size_honors_explicit_value():
    config = SimpleNamespace(inference=SimpleNamespace(ensemble_size=3))
    assert resolve_ensemble_size(config, head_type="diffusion") == 3
    assert resolve_ensemble_size(config, head_type="deterministic") == 3


def test_diffusion_ensemble_size_defaults_to_one():
    config = SimpleNamespace(inference=SimpleNamespace())
    assert resolve_ensemble_size(config, head_type="diffusion") == 1


def test_diffusion_ensemble_uses_independent_seeded_samples():
    batch = {"x": torch.zeros(1, 1, 3, 4)}
    out, pre_inverse, raw = infer_batch_ensemble(
        model=_FakeModel(),
        batch=batch,
        infer_batch=_fake_infer,
        boundary_cfg=None,
        head_type="diffusion",
        ensemble_size=10,
        base_seed=123,
        device=torch.device("cpu"),
    )
    assert out.shape == (1, 10, 2, 3, 4)
    assert pre_inverse.shape == out.shape
    assert raw.shape == out.shape
    assert not torch.allclose(out[:, 0], out[:, 1])

    out2, _, _ = infer_batch_ensemble(
        model=_FakeModel(),
        batch=batch,
        infer_batch=_fake_infer,
        boundary_cfg=None,
        head_type="diffusion",
        ensemble_size=10,
        base_seed=123,
        device=torch.device("cpu"),
    )
    assert torch.allclose(out, out2)


def test_diffusion_member_streams_advance_across_batches_and_replay_by_seed():
    """Dates must not reuse noise, but a fresh run must replay every member."""
    batch = {"x": torch.zeros(1, 1, 3, 4)}

    def run_two_batches(model):
        outputs = []
        for _ in range(2):
            out, _, _ = infer_batch_ensemble(
                model=model,
                batch=batch,
                infer_batch=_fake_infer,
                boundary_cfg=None,
                head_type="diffusion",
                ensemble_size=3,
                base_seed=987,
                device=torch.device("cpu"),
            )
            outputs.append(out)
        return outputs

    first_run = run_two_batches(_FakeModel())
    replay = run_two_batches(_FakeModel())

    # The old implementation reseeded every call, making these identical and
    # imposing the same initial spatial noise pattern on every date.
    assert not torch.allclose(first_run[0], first_run[1])

    # Member-specific streams still replay exactly from the same base seed.
    assert torch.equal(first_run[0], replay[0])
    assert torch.equal(first_run[1], replay[1])


def test_reset_ensemble_generators_makes_scenario_independent_of_run_order():
    batch = {"x": torch.zeros(1, 1, 3, 4)}
    model = _FakeModel()

    def sample():
        return infer_batch_ensemble(
            model=model,
            batch=batch,
            infer_batch=_fake_infer,
            boundary_cfg=None,
            head_type="diffusion",
            ensemble_size=3,
            base_seed=987,
            device=torch.device("cpu"),
        )[0]

    scenario_alone = sample()
    _ = sample()  # A different scenario advances every member stream.
    assert reset_ensemble_generators(model) is True
    scenario_after_reset = sample()

    assert torch.equal(scenario_alone, scenario_after_reset)
    assert reset_ensemble_generators(model) is True
    assert reset_ensemble_generators(model) is False


def test_deterministic_path_keeps_previous_shape():
    batch = {"x": torch.zeros(2, 1, 3, 4)}
    out, _, _ = infer_batch_ensemble(
        model=torch.nn.Module(),
        batch=batch,
        infer_batch=_fake_infer,
        boundary_cfg=None,
        head_type="deterministic",
        ensemble_size=10,
        base_seed=123,
        device=torch.device("cpu"),
    )
    assert out.shape == (2, 2, 3, 4)


@pytest.mark.parametrize("ensemble_size", [1, 3])
def test_ensemble_mean_always_removes_ensemble_dimension(ensemble_size):
    values = np.arange(2 * ensemble_size * 2 * 3 * 4, dtype=np.float32).reshape(
        2, ensemble_size, 2, 3, 4
    )

    reduced = as_ensemble_mean(values)

    assert reduced.shape == (2, 2, 3, 4)
    np.testing.assert_allclose(reduced, values.mean(axis=1))


def test_ensemble_mean_rejects_empty_ensemble_dimension():
    with pytest.raises(ValueError, match="empty dimension"):
        as_ensemble_mean(np.empty((2, 0, 2, 3, 4), dtype=np.float32))


def test_diffusion_netcdf_dataset_has_ensemble_dimension():
    outputs = np.zeros((5, 10, 2, 3, 4), dtype=np.float32)
    coords = {
        "time": np.arange(5),
        "lat": np.arange(3),
        "lon": np.arange(4),
    }
    ds = xr.Dataset(coords=coords)
    ds = add_predictions_to_dataset(
        prediction_ds=ds,
        target_vars=["pr", "tasmax"],
        outputs_np=outputs,
        coords=coords,
        time_dim="time",
        lat_dim="lat",
        lon_dim="lon",
        target_attrs={"pr": {"units": "mm"}, "tasmax": {"units": "K"}},
        head_type="diffusion",
        ensemble_size=10,
        base_seed=42,
    )
    assert ds.sizes["ensemble"] == 10
    assert ds["pr"].dims == ("time", "ensemble", "lat", "lon")
    assert ds.attrs["head_type"] == "diffusion"
    assert ds.attrs["ensemble_generation"] == "diffusion_sampling"


def test_variable_array_map_accepts_ensemble_and_deterministic_arrays():
    deterministic = np.zeros((5, 2, 3, 4), dtype=np.float32)
    ensemble = np.zeros((5, 10, 2, 3, 4), dtype=np.float32)
    assert variable_array_map(["pr", "tasmax"], deterministic)["pr"].shape == (5, 3, 4)
    assert variable_array_map(["pr", "tasmax"], ensemble)["pr"].shape == (5, 10, 3, 4)


def test_inference_truth_units_convert_flux_to_mm_per_day(
    inference_script, tmp_path
):
    target_path = tmp_path / "truth_flux.nc"
    _write_target_fixture(target_path, pr_units="kg m-2 s-1")

    provenance = inference_script._resolve_target_unit_provenance(
        [str(target_path)], ["pr", "tasmax"]
    )

    assert provenance["pr"]["canonical_units"] == "mm/day"
    assert provenance["pr"]["conversion_factor"] == 86_400.0
    assert provenance["tasmax"]["canonical_units"] == "K"
    assert provenance["tasmax"]["conversion_factor"] == 1.0


def test_inference_truth_conversion_preserves_tensor_contract(inference_script):
    target = torch.tensor(
        [[[[1.0e-5, 2.0e-5]], [[300.0, 301.0]]]], dtype=torch.float64
    )
    provenance = {
        "pr": {"conversion_factor": 86_400.0},
        "tasmax": {"conversion_factor": 1.0},
    }

    converted = inference_script._canonicalize_target_tensor(
        target, ["pr", "tasmax"], provenance
    )

    assert converted.shape == target.shape
    assert converted.dtype == target.dtype
    assert converted.device == target.device
    torch.testing.assert_close(converted[:, 0], target[:, 0] * 86_400.0)
    torch.testing.assert_close(converted[:, 1], target[:, 1])


def test_inference_rejects_mixed_or_unsupported_truth_units(
    inference_script, tmp_path
):
    flux_path = tmp_path / "truth_flux.nc"
    daily_path = tmp_path / "truth_daily.nc"
    unsupported_path = tmp_path / "truth_unsupported.nc"
    missing_path = tmp_path / "truth_missing.nc"
    _write_target_fixture(flux_path, pr_units="kg m-2 s-1")
    _write_target_fixture(daily_path, pr_units="mm/day")
    _write_target_fixture(unsupported_path, pr_units="m/day")
    _write_target_fixture(missing_path, pr_units=None)

    with pytest.raises(ValueError, match="Mixed units"):
        inference_script._resolve_target_unit_provenance(
            [str(flux_path), str(daily_path)], ["pr", "tasmax"]
        )
    with pytest.raises(ValueError, match="Unsupported units"):
        inference_script._resolve_target_unit_provenance(
            [str(unsupported_path)], ["pr", "tasmax"]
        )
    with pytest.raises(ValueError, match="has no units attribute"):
        inference_script._resolve_target_unit_provenance(
            [str(missing_path)], ["pr", "tasmax"]
        )


def test_inference_loop_converts_truth_but_not_model_batch(
    inference_script, monkeypatch
):
    class DiagnosticModel(torch.nn.Module):
        n_input_timestamps = 1
        input_scalers_epsilon = 1.0e-6

        def __init__(self):
            super().__init__()
            self.register_buffer("input_scalers_mu", torch.zeros(1))
            self.register_buffer("input_scalers_sigma", torch.ones(1))

    source_y = torch.cat(
        (
            torch.full((1, 1, 2, 3), 1.0e-5),
            torch.full((1, 1, 2, 3), 300.0),
        ),
        dim=1,
    )
    batch = {"x": torch.zeros(1, 1, 2, 3), "y": source_y.clone()}
    model_y_seen = []

    def fake_ensemble(**kwargs):
        model_y_seen.append(kwargs["batch"]["y"].detach().cpu().clone())
        x = kwargs["batch"]["x"]
        output = torch.zeros(x.shape[0], 2, x.shape[-2], x.shape[-1])
        return output, output.clone(), output.clone()

    monkeypatch.setattr(inference_script, "infer_batch_ensemble", fake_ensemble)
    result = inference_script._run_full_inference(
        [batch],
        DiagnosticModel(),
        torch.device("cpu"),
        ["pr", "tasmax"],
        SimpleNamespace(),
        {
            "pr": {"conversion_factor": 86_400.0},
            "tasmax": {"conversion_factor": 1.0},
        },
    )

    torch.testing.assert_close(model_y_seen[0], source_y)
    torch.testing.assert_close(
        result["targets"][:, 0], source_y[:, 0] * 86_400.0
    )
    torch.testing.assert_close(result["targets"][:, 1], source_y[:, 1])


def test_boundary_provenance_honors_tile_halo_and_origin(inference_script):
    cfg = SimpleNamespace(
        enabled=True,
        force_full_frame=False,
        tile_size=(96, 96),
        overlap=(32, 32),
        halo=(16, 24),
        tile_origin=(7, 11),
        blend_window="hann",
        blend_sigma=0.35,
        deblock=SimpleNamespace(
            enabled=False, boundary_width=2, strength=0.1, kernel_size=3
        ),
    )
    provenance = inference_script._boundary_runtime_provenance(cfg)

    assert provenance["mode"] == "overlap_tiled"
    assert provenance["force_full_frame"] is False
    assert provenance["tile_size"] == [96, 96]
    assert provenance["overlap"] == [32, 32]
    assert provenance["halo"] == [16, 24]
    assert provenance["tile_origin"] == [7, 11]
    assert provenance["tile_stride"] == [64, 64]
    assert "boundary_cfg.force_full_frame = True" not in Path(
        inference_script.__file__
    ).read_text(encoding="utf-8")


def test_runtime_diffusion_provenance_records_alpha_schedule_padding_and_seeds(
    inference_script
):
    class RealisticDiffusionHead(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            self.score_model = torch.nn.Sequential(
                torch.nn.Conv2d(
                    2, 2, 3, padding=1, padding_mode="replicate"
                )
            )

        def score_fn(self, x, cond, timestep):
            del cond, timestep
            return self.score_model(x)

    cfg = SimpleNamespace(
        sde="vpsde",
        beta_min=0.1,
        beta_max=20.0,
        num_scales=1000,
        continuous=True,
        sampling_method="ddim",
        eta=0.2,
        predictor="euler_maruyama",
        corrector="none",
        num_sampling_steps=256,
        prediction_type="epsilon",
        residual_diffusion=True,
        residual_application_scale=0.25,
        padding_mode="replicate",
        zero_init_output=True,
    )
    head = RealisticDiffusionHead(cfg)
    assert callable(head.score_fn)
    assert not isinstance(head.score_fn, torch.nn.Module)
    assert isinstance(head.score_model, torch.nn.Module)
    model = torch.nn.Module()
    model.diffusion_head = head
    checkpoint = {
        "metadata": {
            "compatibility": {
                "residual_contract": {
                    "definition": "target_normalized - baseline_normalized",
                    "sign": "target_minus_baseline",
                    "space": "target_normalized_space",
                }
            }
        }
    }

    provenance = inference_script._diffusion_runtime_provenance(
        model,
        head_type="diffusion",
        ensemble_size=3,
        base_seed=42,
        checkpoint=checkpoint,
        inference_batch_size=16,
        residual_alpha_provenance={
            "environment_variable": "SA_RESIDUAL_ALPHA",
            "source": "environment:SA_RESIDUAL_ALPHA",
            "value": 0.25,
            "configured_value": 0.0,
            "override_is_set": True,
        },
    )

    assert provenance["member_seeds"] == [42, 43, 44]
    assert provenance["sampling"]["sampling_method"] == "ddim"
    assert provenance["sampling"]["prediction_type"] == "epsilon"
    assert provenance["sampling"]["num_sampling_steps"] == 256
    assert provenance["residual_contract"]["alpha"] == 0.25
    assert provenance["residual_contract"]["alpha_source"] == (
        "environment:SA_RESIDUAL_ALPHA"
    )
    assert provenance["inference_batch_size"] == 16
    assert provenance["rng_stream_scope"] == "reset_per_scenario_run"
    assert provenance["batch_size_invariant"] is False
    assert provenance["exact_reproduction_requires_same_batching"] is True
    assert provenance["score_network"]["configured_padding_mode"] == "replicate"
    assert provenance["score_network"]["effective_padded_conv_modes"] == [
        "replicate"
    ]


def test_residual_alpha_runtime_override_uses_config_or_environment(
    inference_script,
):
    configured = SimpleNamespace(
        model=SimpleNamespace(
            diffusion={
                "residual_diffusion": True,
                "residual_application_scale": 0.0,
            }
        )
    )
    configured_provenance = inference_script._apply_residual_alpha_runtime_override(
        configured, {}
    )
    assert configured_provenance["source"] == "resolved_config"
    assert configured_provenance["value"] == 0.0

    overridden = SimpleNamespace(
        model=SimpleNamespace(
            diffusion={
                "residual_diffusion": True,
                "residual_application_scale": 0.0,
            }
        )
    )
    override_provenance = inference_script._apply_residual_alpha_runtime_override(
        overridden, {"SA_RESIDUAL_ALPHA": "0.35"}
    )
    assert overridden.model.diffusion["residual_application_scale"] == 0.35
    assert override_provenance["source"] == "environment:SA_RESIDUAL_ALPHA"
    assert override_provenance["value"] == 0.35
    assert override_provenance["configured_value"] == 0.0
    assert override_provenance["override_is_set"] is True


@pytest.mark.parametrize("raw_value", ["", "not-a-number", "nan", "-0.01", "1.01"])
def test_residual_alpha_runtime_override_rejects_invalid_values(
    inference_script, raw_value
):
    config = SimpleNamespace(
        model=SimpleNamespace(diffusion={"residual_diffusion": True})
    )
    with pytest.raises(ValueError, match="Residual application scale"):
        inference_script._apply_residual_alpha_runtime_override(
            config, {"SA_RESIDUAL_ALPHA": raw_value}
        )


def test_residual_alpha_runtime_override_rejects_non_residual_run(inference_script):
    config = SimpleNamespace(
        model=SimpleNamespace(diffusion={"residual_diffusion": False})
    )
    with pytest.raises(ValueError, match="residual_diffusion is not enabled"):
        inference_script._apply_residual_alpha_runtime_override(
            config, {"SA_RESIDUAL_ALPHA": "0.25"}
        )


def test_scalar_provenance_hashes_exact_runtime_files(inference_script, tmp_path):
    fields = {
        "inputs_mean": "input_mu",
        "inputs_std": "input_sigma",
        "targets_mean": "target_mu",
        "targets_std": "target_sigma",
    }
    model_cfg = SimpleNamespace()
    checkpoint_scalars = {}
    for index, (scalar_name, attr_name) in enumerate(fields.items()):
        path = tmp_path / f"{scalar_name}.npy"
        np.save(path, np.asarray([index], dtype=np.float32))
        setattr(model_cfg, attr_name, str(path))
        checkpoint_scalars[scalar_name] = {
            "sha256": inference_script._sha256_path(path)
        }
    checkpoint = {
        "metadata": {
            "compatibility": {"normalization_scalars": checkpoint_scalars}
        }
    }

    provenance = inference_script._normalization_scalar_provenance(
        SimpleNamespace(model=model_cfg), checkpoint
    )

    assert list(provenance) == list(fields)
    for scalar_name, record in provenance.items():
        assert record["path"] == str((tmp_path / f"{scalar_name}.npy").resolve())
        assert record["sha256"] == checkpoint_scalars[scalar_name]["sha256"]
        assert record["size_bytes"] > 0


def test_saved_netcdf_contains_safe_complete_runtime_provenance(
    inference_script, tmp_path, monkeypatch
):
    predictor_path = tmp_path / "predictor.nc"
    target_path = tmp_path / "target.nc"
    xr.Dataset(coords={"time": [0]}).to_netcdf(
        predictor_path, engine="h5netcdf"
    )
    _write_target_fixture(target_path, pr_units="kg m-2 s-1")
    target_units = inference_script._resolve_target_unit_provenance(
        [str(target_path)], ["pr", "tasmax"]
    )
    boundary = inference_script._boundary_runtime_provenance(
        SimpleNamespace(
            enabled=False,
            force_full_frame=True,
            tile_size=(96, 96),
            overlap=(32, 32),
            halo=(16, 16),
            tile_origin=(0, 0),
            blend_window="hann",
            blend_sigma=0.35,
            deblock=SimpleNamespace(
                enabled=False, boundary_width=2, strength=0.1, kernel_size=3
            ),
        )
    )
    scalar_record = {
        "inputs_mean": {
            "path": str(tmp_path / "inputs_mean.npy"),
            "sha256": "a" * 64,
            "size_bytes": 128,
        }
    }
    diffusion_runtime = {
        "base_seed": 42,
        "ensemble_size": 2,
        "member_seeds": [42, 43],
        "inference_batch_size": 16,
        "rng_stream_scope": "reset_per_scenario_run",
        "batch_size_invariant": False,
        "exact_reproduction_requires_same_batching": True,
        "sampling": {
            "sde": "vpsde",
            "sampling_method": "ddim",
            "num_sampling_steps": 256,
            "prediction_type": "epsilon",
        },
        "residual_contract": {
            "enabled": True,
            "definition": "target_normalized - baseline_normalized",
            "sign": "target_minus_baseline",
            "space": "target_normalized_space",
            "reconstruction": "baseline + alpha * sampled_residual",
            "residual_application_scale": 0.25,
            "alpha": 0.25,
            "alpha_source": "environment:SA_RESIDUAL_ALPHA",
            "alpha_environment_variable": "SA_RESIDUAL_ALPHA",
            "alpha_override_is_set": True,
        },
        "score_network": {
            "configured_padding_mode": "replicate",
            "effective_padded_conv_modes": ["replicate"],
            "conv2d_count": 2,
            "padded_conv2d_count": 2,
        },
    }
    model_provenance = {
        "checkpoint_path": str(tmp_path / "last.ckpt"),
        "checkpoint_sha256": "b" * 64,
        "checkpoint_epoch": 9,
        "checkpoint_global_step": 4570,
        "config_snapshot_path": str(tmp_path / "resolved.yaml"),
        "config_snapshot_sha256": "c" * 64,
        "config_fingerprint_sha256": "d" * 64,
        "git_commit": "e" * 40,
        "normalization_scalars": scalar_record,
        "runtime_overrides": {
            "residual_alpha": {
                "environment_variable": "SA_RESIDUAL_ALPHA",
                "source": "environment:SA_RESIDUAL_ALPHA",
                "value": 0.25,
                "configured_value": 0.0,
                "override_is_set": True,
            }
        },
        "diffusion_runtime": diffusion_runtime,
    }
    outputs = np.zeros((1, 2, 2, 2, 3), dtype=np.float32)
    baseline = np.zeros((1, 2, 2, 3), dtype=np.float32)
    monkeypatch.setattr(inference_script, "SAVE_DIAGNOSTICS_JSON", False)
    monkeypatch.setattr(inference_script, "SAVE_DISTRIBUTION_PLOT", False)

    inference_script._save_run_outputs(
        idx=0,
        output_root=tmp_path,
        prediction_output_stub=Path("predictions.nc"),
        outputs_np=outputs,
        outputs_pre_inverse_np=outputs,
        baseline_outputs_np=baseline,
        target_vars=["pr", "tasmax"],
        predictor_paths=[str(predictor_path)],
        target_template_paths=[str(target_path)],
        time_dim="time",
        lat_dim="lat",
        lon_dim="lon",
        lat_name="lat",
        lon_name="lon",
        head_type="diffusion",
        ensemble_size=2,
        base_seed=42,
        diagnostics_payload={},
        predicted_var_values={"pr": outputs[:, :, 0], "tasmax": outputs[:, :, 1]},
        predicted_pre_inverse_var_values={
            "pr": outputs[:, :, 0],
            "tasmax": outputs[:, :, 1],
        },
        target_var_values={"pr": baseline[:, 0], "tasmax": baseline[:, 1]},
        transformation_stage_samples={},
        target_unit_provenance=target_units,
        boundary_provenance=boundary,
        model_provenance=model_provenance,
    )

    with xr.open_dataset(tmp_path / "predictions.nc", engine="h5netcdf") as saved:
        assert json.loads(saved.attrs["output_variable_order"]) == ["pr", "tasmax"]
        assert json.loads(saved.attrs["canonical_output_units"]) == {
            "pr": "mm/day",
            "tasmax": "K",
        }
        assert saved["pr"].attrs["units"] == "mm/day"
        assert saved["pr"].attrs["source_target_units"] == "kg m-2 s-1"
        assert saved["pr"].attrs["source_target_to_canonical_factor"] == 86_400.0
        assert saved.attrs["checkpoint_sha256"] == "b" * 64
        assert saved.attrs["scalar_inputs_mean_sha256"] == "a" * 64
        assert saved.attrs["diffusion_sampling_method"] == "ddim"
        assert saved.attrs["diffusion_num_sampling_steps"] == 256
        assert saved.attrs["residual_application_scale"] == 0.25
        assert saved.attrs["residual_alpha"] == 0.25
        assert saved.attrs["residual_alpha_source"] == (
            "environment:SA_RESIDUAL_ALPHA"
        )
        assert saved.attrs["residual_alpha_environment_variable"] == (
            "SA_RESIDUAL_ALPHA"
        )
        assert saved.attrs["residual_alpha_override_is_set"] == 1
        assert saved.attrs["inference_batch_size"] == 16
        assert saved.attrs["diffusion_rng_stream_scope"] == (
            "reset_per_scenario_run"
        )
        assert saved.attrs[
            "diffusion_exact_reproduction_requires_same_batching"
        ] == 1
        assert saved.attrs["score_padding_mode"] == "replicate"
        assert json.loads(saved.attrs["boundary_halo"]) == [16, 16]
        assert json.loads(saved.attrs["boundary_tile_origin"]) == [0, 0]
        assert saved.attrs["boundary_force_full_frame"] == 1
