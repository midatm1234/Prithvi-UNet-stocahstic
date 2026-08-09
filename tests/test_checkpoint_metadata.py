from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from granitewxc.utils.checkpoint_metadata import (
    CHECKPOINT_METADATA_SCHEMA,
    CheckpointCompatibilityError,
    build_checkpoint_metadata,
    build_checkpoint_strict_compatibility,
    validate_checkpoint_compatibility,
)


def _residual_config(tmp_path: Path) -> SimpleNamespace:
    scalar_paths = {}
    for name in ("inputs_mean", "inputs_std", "targets_mean", "targets_std"):
        path = tmp_path / f"{name}.npy"
        path.write_bytes(f"scalar:{name}".encode())
        scalar_paths[name] = str(path)
    (tmp_path / "metadata.json").write_text(
        '{"sample_selection":{"policy":"training-only"}}',
        encoding="utf-8",
    )

    initializer = tmp_path / "unet_initializer.pt"
    initializer.write_bytes(b"initializer-state")
    model = SimpleNamespace(
        head_type="diffusion",
        decoder_type="unet",
        input_mu=scalar_paths["inputs_mean"],
        input_sigma=scalar_paths["inputs_std"],
        target_mu=scalar_paths["targets_mean"],
        target_sigma=scalar_paths["targets_std"],
        diffusion={
            "sde": "vpsde",
            "beta_min": 0.1,
            "beta_max": 20.0,
            "num_scales": 1000,
            "sampling_method": "ddim",
            "num_sampling_steps": 64,
            "eta": 0.2,
            "residual_diffusion": True,
            "prediction_type": "epsilon",
            "clean_x0_reconstruction_weight": 1.0,
            "clean_x0_inverse_snr_cap": 100.0,
            "noise_conditioning_scale": 1.0,
            "padding_mode": "replicate",
            "residual_application_scale": 0.0,
        },
    )
    return SimpleNamespace(
        case_name="case_residual",
        job_id="job_residual",
        path_model_weights=str(initializer),
        model=model,
        data=SimpleNamespace(
            output_vars=["pr", "tasmax"],
            scalers=scalar_paths,
            training_predictor_paths=[str(tmp_path / "train_predictors.nc")],
            training_target_paths=[str(tmp_path / "train_targets.nc")],
            validation_predictor_paths=[str(tmp_path / "validation_predictors.nc")],
            validation_target_paths=[str(tmp_path / "validation_targets.nc")],
            static_path=str(tmp_path / "static.nc"),
            use_static=True,
            target_size_lat=128,
            target_size_lon=128,
            train_crop_size_lat=128,
            train_crop_size_lon=128,
            input_surface_vars=["pr", "tasmax"],
            input_static_surface_vars=["orog"],
        ),
        predictands={
            "pr": {"normalization": {"method": "divide_only", "mode": "global"}},
            "tasmax": {"normalization": {"method": "zscore", "mode": "gridpoint"}},
        },
        precip_model="hurdle",
        precip_wet_threshold=0.01,
        loss={
            "base": "rmse",
            "deterministic_weight": 1.0,
            "diffusion": {"weight": 1.0},
        },
        inference={"ensemble_size": 3, "base_seed": 42},
    )


def _checkpoint(config: SimpleNamespace) -> dict:
    return {"metadata": build_checkpoint_metadata(config, epoch=2, global_step=17)}


def test_residual_metadata_records_contract_and_file_hashes(tmp_path: Path) -> None:
    config = _residual_config(tmp_path)
    metadata = _checkpoint(config)["metadata"]

    assert metadata["schema"] == CHECKPOINT_METADATA_SCHEMA
    assert metadata["epoch_index"] == 2
    assert metadata["completed_epochs"] == 3
    assert metadata["global_step"] == 17
    compatibility = metadata["compatibility"]
    assert compatibility["output_variables"] == ["pr", "tasmax"]
    assert compatibility["data_identity"]["target_size_lat"] == 128
    assert compatibility["data_identity"]["use_static"] is True
    assert compatibility["data_identity"]["training_predictor_paths"] == [
        str(tmp_path / "train_predictors.nc")
    ]
    assert compatibility["residual_contract"] == {
        "enabled": True,
        "definition": "target_normalized - stop_gradient(deterministic_baseline_normalized)",
        "sign": "target_minus_baseline",
        "space": "target_normalized_space",
        "reconstruction": "deterministic_baseline_normalized + residual_application_scale * sampled_residual_normalized",
        "residual_application_scale_policy": "runtime_configured_in_[0,1]",
        "score_conditioning_gradient": "stopped",
        "deterministic_baseline_training": "jointly_optimized_by_yaml_loss",
    }
    assert compatibility["diffusion"]["prediction_type"] == "epsilon"
    assert compatibility["diffusion"]["clean_x0_reconstruction_weight"] == 1.0
    assert compatibility["diffusion"]["clean_x0_inverse_snr_cap"] == 100.0
    assert compatibility["diffusion"]["sampling_method"] == "ddim"
    assert compatibility["diffusion"]["padding_mode"] == "replicate"
    assert compatibility["diffusion"]["zero_init_output"] is True
    assert "residual_application_scale" not in compatibility["diffusion"]
    assert metadata["deployment_defaults"]["residual_application_scale"] == 0.0
    assert compatibility["baseline_semantics"]["canonical_output_units"] == {
        "pr": "mm/day",
        "tasmax": "K",
    }
    assert compatibility["spatial_processing"]["target_size"] == [128, 128]
    assert compatibility["loss_weights"] == {"deterministic": 1.0, "diffusion": 1.0}
    assert compatibility["normalization_scalars"]["targets_std"]["sha256"]
    assert compatibility["normalization_scalars"]["metadata"]["sha256"]
    assert compatibility["initializer_files"]["path_model_weights"]["sha256"]
    strict = metadata["strict_compatibility"]
    assert strict == build_checkpoint_strict_compatibility(config)
    assert "case_name" not in strict
    assert "job_id" not in strict
    assert "initializer_files" not in strict
    assert "inference_randomness" not in strict
    assert "training_predictor_paths" not in strict["data_identity"]
    assert strict["data_identity"]["training_predictor_files"] == [
        "train_predictors.nc"
    ]
    assert "path" not in strict["normalization_scalars"]["targets_std"]
    assert metadata["compatibility_provenance_sha256"]

    validate_checkpoint_compatibility({"metadata": metadata}, config)


def test_residual_validation_rejects_output_order_change(tmp_path: Path) -> None:
    config = _residual_config(tmp_path)
    checkpoint = _checkpoint(config)
    edited = deepcopy(config)
    edited.data.output_vars = ["tasmax", "pr"]

    with pytest.raises(CheckpointCompatibilityError, match="output_variables"):
        validate_checkpoint_compatibility(checkpoint, edited)


def test_residual_validation_rejects_dataset_geometry_change(tmp_path: Path) -> None:
    config = _residual_config(tmp_path)
    checkpoint = _checkpoint(config)
    edited = deepcopy(config)
    edited.data.target_size_lat = 96

    with pytest.raises(
        CheckpointCompatibilityError,
        match="data_identity.target_size_lat",
    ):
        validate_checkpoint_compatibility(checkpoint, edited)


def test_residual_validation_rejects_validation_mode_change(tmp_path: Path) -> None:
    config = _residual_config(tmp_path)
    config.validation_enabled = False
    checkpoint = _checkpoint(config)
    edited = deepcopy(config)
    edited.validation_enabled = True

    with pytest.raises(
        CheckpointCompatibilityError,
        match="data_identity.validation_enabled",
    ):
        validate_checkpoint_compatibility(checkpoint, edited)


def test_residual_validation_rejects_baseline_training_policy_change(
    tmp_path: Path,
) -> None:
    config = _residual_config(tmp_path)
    checkpoint = _checkpoint(config)
    edited = deepcopy(config)
    edited.training = {"freeze_deterministic_baseline": True}

    with pytest.raises(
        CheckpointCompatibilityError,
        match="deterministic_baseline_training",
    ):
        validate_checkpoint_compatibility(checkpoint, edited)


def test_residual_validation_rejects_changed_scalar_contents(tmp_path: Path) -> None:
    config = _residual_config(tmp_path)
    checkpoint = _checkpoint(config)
    Path(config.model.target_sigma).write_bytes(b"changed-target-sigma-with-new-size")

    with pytest.raises(CheckpointCompatibilityError, match="normalization_scalars.targets_std"):
        validate_checkpoint_compatibility(checkpoint, config)


def test_residual_validation_rejects_sampling_change(tmp_path: Path) -> None:
    config = _residual_config(tmp_path)
    checkpoint = _checkpoint(config)
    edited = deepcopy(config)
    edited.model.diffusion["eta"] = 0.0

    with pytest.raises(CheckpointCompatibilityError, match="diffusion.eta"):
        validate_checkpoint_compatibility(checkpoint, edited)


def test_residual_validation_allows_alpha_only_runtime_change(tmp_path: Path) -> None:
    config = _residual_config(tmp_path)
    checkpoint = _checkpoint(config)
    edited = deepcopy(config)
    edited.model.diffusion["residual_application_scale"] = 0.35

    validate_checkpoint_compatibility(checkpoint, edited)
    edited_metadata = build_checkpoint_metadata(edited)
    assert (
        edited_metadata["deployment_defaults"]["residual_application_scale"]
        == pytest.approx(0.35)
    )


def test_residual_validation_allows_runtime_and_location_provenance_changes(
    tmp_path: Path,
) -> None:
    original_dir = tmp_path / "original"
    relocated_dir = tmp_path / "relocated"
    original_dir.mkdir()
    relocated_dir.mkdir()
    config = _residual_config(original_dir)
    checkpoint = _checkpoint(config)
    relocated = _residual_config(relocated_dir)
    relocated.case_name = "different_case"
    relocated.job_id = "different_job"
    relocated.inference = {"ensemble_size": 9, "base_seed": 123456}
    relocated.data.training_predictor_paths = [
        r"Z:\moved\train_predictors.nc"
    ]
    relocated.data.training_target_paths = [r"Z:\moved\train_targets.nc"]
    relocated.data.validation_predictor_paths = [
        r"Z:\moved\validation_predictors.nc"
    ]
    relocated.data.validation_target_paths = [
        r"Z:\moved\validation_targets.nc"
    ]
    relocated.data.static_path = r"Z:\moved\static.nc"
    different_initializer = relocated_dir / "different_initializer_name.pt"
    different_initializer.write_bytes(b"unrelated initializer provenance")
    relocated.path_model_weights = str(different_initializer)

    validate_checkpoint_compatibility(checkpoint, relocated)


def test_residual_validation_rejects_portable_training_dataset_identity_change(
    tmp_path: Path,
) -> None:
    config = _residual_config(tmp_path)
    checkpoint = _checkpoint(config)
    edited = deepcopy(config)
    edited.data.training_predictor_paths = [
        str(tmp_path / "different_train_predictors.nc")
    ]

    with pytest.raises(
        CheckpointCompatibilityError, match="training_predictor_files"
    ):
        validate_checkpoint_compatibility(checkpoint, edited)


def test_residual_validation_detects_tampered_recorded_provenance(
    tmp_path: Path,
) -> None:
    config = _residual_config(tmp_path)
    checkpoint = _checkpoint(config)
    checkpoint["metadata"]["compatibility"]["case_name"] = "tampered"

    with pytest.raises(
        CheckpointCompatibilityError, match="recorded compatibility provenance"
    ):
        validate_checkpoint_compatibility(checkpoint, config)


def test_residual_validation_rejects_clean_x0_objective_change(tmp_path: Path) -> None:
    config = _residual_config(tmp_path)
    checkpoint = _checkpoint(config)
    edited = deepcopy(config)
    edited.model.diffusion["clean_x0_reconstruction_weight"] = 0.0

    with pytest.raises(
        CheckpointCompatibilityError,
        match="diffusion.clean_x0_reconstruction_weight",
    ):
        validate_checkpoint_compatibility(checkpoint, edited)


def test_residual_config_rejects_legacy_checkpoint_without_metadata(tmp_path: Path) -> None:
    config = _residual_config(tmp_path)

    with pytest.raises(CheckpointCompatibilityError, match="missing the required 'metadata'"):
        validate_checkpoint_compatibility({"model": {}}, config)
