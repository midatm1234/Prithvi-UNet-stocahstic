import json
from types import SimpleNamespace

import pytest

from granitewxc.utils import normalization
from granitewxc.utils.prism_checkpoint import (
    CONTRACT_KEY,
    CONTRACT_SCHEMA_VERSION,
    build_prism_checkpoint_contract,
    validate_prism_checkpoint_contract,
)
from granitewxc.utils.prism_preprocessed import split_source_artifact_signature


def _config(tmp_path, *, case_name="narr_case", skip_source="dynamic"):
    return SimpleNamespace(
        case_name=case_name,
        data=SimpleNamespace(
            type="narr_prism",
            preprocessed_dir=str(tmp_path),
            scalar_dir=str(tmp_path / "legacy_scalars"),
            use_preprocessed=True,
            train_crop_size_lat=256,
            train_crop_size_lon=256,
            training_tile_stride_lat=192,
            training_tile_stride_lon=192,
            training_halo_lat=32,
            training_halo_lon=32,
            regrid_method="bilinear",
            input_vars=[
                "air_850",
                "elev",
                "mask_air_850",
                "mask_elev",
            ],
            input_levels=[1],
            output_vars=["ppt", "tmax"],
            target_variables=["ppt", "tmax"],
            predictor_variables={"air": [850]},
            static_elevation_file="elevation.nc",
            n_input_timestamps=1,
            use_static=False,
        ),
        model=SimpleNamespace(
            decoder_skip_source=skip_source,
            backbone_attention_scope="windowed_local" if skip_source == "dynamic" else "legacy_global",
            backbone_residual_mode=(
                "pre_conv_add" if skip_source == "dynamic" else "legacy_ignored"
            ),
            residual_connection=True,
            decoder_upsampling_mode="bilinear",
            embed_dim=64,
        ),
        mask_unit_size=[16, 16],
        backbone_use=True,
        predictands={
            "ppt": {
                "nonnegativity": {"enabled": True, "method": "softplus"},
                "normalization": {
                    "method": "divide_only",
                    "mode": "global",
                    "scale_stat": "p95",
                },
            },
            "tmax": {
                "nonnegativity": {"enabled": False, "method": "none"},
                "normalization": {"method": "standardize", "mode": "gridpoint"},
            },
        },
        precip_model="hurdle",
    )


def test_all_prism_pipelines_reject_checkpoint_without_contract(tmp_path):
    config = _config(tmp_path)
    with pytest.raises(ValueError, match="must be retrained"):
        validate_prism_checkpoint_contract(config, {"model": {}}, role="test")

    fully_legacy = _config(tmp_path, skip_source="legacy")
    fully_legacy.data.use_preprocessed = False
    fully_legacy.data.training_halo_lat = 0
    fully_legacy.data.training_halo_lon = 0
    assert fully_legacy.model.backbone_residual_mode == "legacy_ignored"
    with pytest.raises(ValueError, match="cannot be verified"):
        validate_prism_checkpoint_contract(
            fully_legacy, {"model": {}}, role="test"
        )

    # The residual implementation is independently model-semantic: even a
    # config with legacy data/decoder/attention settings may not load an
    # uncontracted checkpoint after opting into the corrected merge.
    residual_only = _config(tmp_path, skip_source="legacy")
    residual_only.data.use_preprocessed = False
    residual_only.data.training_halo_lat = 0
    residual_only.data.training_halo_lon = 0
    residual_only.model.backbone_residual_mode = "pre_conv_add"
    with pytest.raises(ValueError, match="must be retrained"):
        validate_prism_checkpoint_contract(
            residual_only, {"model": {}}, role="test"
        )


def test_checkpoint_contract_round_trip_and_case_mismatch(tmp_path):
    config = _config(tmp_path)
    contract = build_prism_checkpoint_contract(config)
    assert contract is not None
    assert contract["schema_version"] == CONTRACT_SCHEMA_VERSION == 4
    checkpoint = {CONTRACT_KEY: contract}
    assert validate_prism_checkpoint_contract(config, checkpoint, role="test") == contract

    wrong_case = _config(tmp_path, case_name="other_case")
    with pytest.raises(ValueError, match="case_name"):
        validate_prism_checkpoint_contract(wrong_case, checkpoint, role="test")


def test_contract_rejects_same_shape_channel_and_behavior_changes(tmp_path):
    config = _config(tmp_path)
    contract = build_prism_checkpoint_contract(config)
    checkpoint = {CONTRACT_KEY: contract}

    reordered = _config(tmp_path)
    reordered.data.output_vars = ["tmax", "ppt"]
    reordered.data.target_variables = ["tmax", "ppt"]
    with pytest.raises(ValueError, match="channels"):
        validate_prism_checkpoint_contract(reordered, checkpoint, role="test")

    no_residual = _config(tmp_path)
    no_residual.model.residual_connection = False
    with pytest.raises(ValueError, match="model_topology"):
        validate_prism_checkpoint_contract(no_residual, checkpoint, role="test")

    legacy_residual_implementation = _config(tmp_path)
    legacy_residual_implementation.model.backbone_residual_mode = "legacy_ignored"
    with pytest.raises(ValueError, match="model_topology"):
        validate_prism_checkpoint_contract(
            legacy_residual_implementation, checkpoint, role="test"
        )

    different_transform = _config(tmp_path)
    different_transform.predictands["tmax"]["normalization"]["mode"] = "global"
    with pytest.raises(ValueError, match="normalization"):
        validate_prism_checkpoint_contract(
            different_transform, checkpoint, role="test"
        )


def test_contract_preserves_predictor_mapping_order(tmp_path):
    config = _config(tmp_path)
    config.data.predictor_variables = {"air": [500, 850], "hgt": [500]}
    config.data.input_vars = [
        "air_500",
        "air_850",
        "hgt_500",
        "elev",
        "mask_air_500",
        "mask_air_850",
        "mask_hgt_500",
        "mask_elev",
    ]
    contract = build_prism_checkpoint_contract(config)
    assert contract["channels"]["ordered_predictors"] == [
        ["air", 500.0],
        ["air", 850.0],
        ["hgt", 500.0],
    ]

    reordered = _config(tmp_path)
    reordered.data.predictor_variables = {"hgt": [500], "air": [500, 850]}
    reordered.data.input_vars = [
        "hgt_500",
        "air_500",
        "air_850",
        "elev",
        "mask_hgt_500",
        "mask_air_500",
        "mask_air_850",
        "mask_elev",
    ]
    with pytest.raises(ValueError, match="channels"):
        validate_prism_checkpoint_contract(
            reordered, {CONTRACT_KEY: contract}, role="test"
        )


def test_checkpoint_contract_binds_training_source_artifact_split(tmp_path):
    config = _config(tmp_path)
    scalar_dir = normalization.resolve_scalar_dir(config, for_writing=True)
    scalar_dir.mkdir(parents=True)

    def write_source_manifest(daily_signatures):
        manifest = {
            "scalers": {},
            "predictor_preprocessing_signature": "b" * 64,
            normalization.TRAINING_SOURCE_ARTIFACT_SIGNATURES_KEY: (
                daily_signatures
            ),
            normalization.TRAINING_SOURCE_ARTIFACT_SPLIT_SIGNATURE_KEY: (
                split_source_artifact_signature(daily_signatures)
            ),
        }
        (scalar_dir / normalization.MANIFEST_NAME).write_text(
            json.dumps(manifest), encoding="utf-8"
        )

    write_source_manifest({"2000-01-01": "a" * 64})
    contract = build_prism_checkpoint_contract(config)
    assert contract is not None
    source_split = contract["coordinates_and_artifacts"][
        "training_source_artifact_split_signature"
    ]
    assert source_split == split_source_artifact_signature(
        {"2000-01-01": "a" * 64}
    )

    write_source_manifest({"2000-01-01": "c" * 64})
    with pytest.raises(ValueError, match="coordinates_and_artifacts"):
        validate_prism_checkpoint_contract(
            config, {CONTRACT_KEY: contract}, role="test"
        )


def test_legacy_model_mode_remains_available_with_a_full_contract(tmp_path):
    config = _config(tmp_path, skip_source="legacy")
    config.data.use_preprocessed = False
    config.data.training_halo_lat = 0
    config.data.training_halo_lon = 0
    contract = build_prism_checkpoint_contract(config)
    assert contract is not None
    assert contract["model_topology"]["decoder_skip_source"] == "legacy"
    assert contract["model_topology"]["backbone_residual_mode"] == "legacy_ignored"
    assert (
        validate_prism_checkpoint_contract(
            config, {CONTRACT_KEY: contract}, role="test"
        )
        == contract
    )
