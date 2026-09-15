"""Configuration-schema tests for the two-phase stochastic refinement."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from granitewxc.refinement.config import (
    REFINEMENT_TYPES,
    ConfigValidationError,
    resolve_performance_config,
    resolve_refinement_config,
)


def test_missing_refinement_section_is_deterministic():
    cfg = resolve_refinement_config({"model": {"embed_dim": 1024}})
    assert cfg.enabled is False
    assert cfg.type == "none"
    assert cfg.is_active is False


def test_none_type_preserves_deterministic_behaviour():
    cfg = resolve_refinement_config({"refinement": {"type": "none"}})
    assert cfg.is_active is False
    assert cfg.type == "none"


def test_enabled_false_preserves_deterministic_behaviour():
    cfg = resolve_refinement_config({"refinement": {"enabled": False, "type": "diffusion_unet"}})
    assert cfg.is_active is False


@pytest.mark.parametrize("name", [t for t in REFINEMENT_TYPES if t != "none"])
def test_each_refinement_type_resolves(name):
    cfg = resolve_refinement_config({"refinement": {"type": name}})
    assert cfg.type == name
    assert cfg.is_active is True
    assert cfg.is_diffusion == name.startswith("diffusion")
    assert cfg.is_flow_matching == name.startswith("flow_matching")
    assert cfg.uses_transformer == name.endswith("transformer")


def test_refinement_section_may_live_under_model():
    cfg = resolve_refinement_config({"model": {"refinement": {"type": "flow_matching_unet"}}})
    assert cfg.type == "flow_matching_unet"


def test_invalid_refinement_name_raises():
    with pytest.raises(ConfigValidationError, match="Unknown refinement.type"):
        resolve_refinement_config({"refinement": {"type": "diffusion_lstm"}})


def test_unknown_refinement_key_raises():
    with pytest.raises(ConfigValidationError, match="Unknown key"):
        resolve_refinement_config({"refinement": {"type": "none", "lead_time": 6}})


def test_invalid_transformer_head_combination_raises():
    with pytest.raises(ConfigValidationError, match="divisible by"):
        resolve_refinement_config(
            {"refinement": {"type": "diffusion_transformer",
                            "transformer": {"embedding_dim": 100, "num_heads": 8}}}
        )


def test_odd_head_dim_raises():
    with pytest.raises(ConfigValidationError, match="head dimension"):
        resolve_refinement_config(
            {"refinement": {"type": "diffusion_transformer",
                            "transformer": {"embedding_dim": 12, "num_heads": 4}}}
        )


def test_invalid_patch_configuration_raises():
    with pytest.raises(ConfigValidationError, match="patch_size"):
        resolve_refinement_config(
            {"refinement": {"type": "diffusion_transformer",
                            "transformer": {"patch_size": [2, 2, 2]}}}
        )
    with pytest.raises(ConfigValidationError, match="not both"):
        resolve_refinement_config(
            {"refinement": {"type": "diffusion_transformer",
                            "transformer": {"patch_size": 4, "patch_height": 2}}}
        )


def test_patch_size_pair_is_rectangular():
    cfg = resolve_refinement_config(
        {"refinement": {"type": "diffusion_transformer", "transformer": {"patch_size": [2, 8]}}}
    )
    assert cfg.transformer.patch_size == (2, 8)


def test_inference_steps_cannot_exceed_training_timesteps():
    with pytest.raises(ConfigValidationError, match="cannot exceed"):
        resolve_refinement_config(
            {"refinement": {"type": "diffusion_unet",
                            "diffusion": {"training_timesteps": 100, "inference_steps": 200}}}
        )


def test_joint_finetuning_conflicts_with_explicit_freeze():
    with pytest.raises(ConfigValidationError, match="joint_finetuning"):
        resolve_refinement_config(
            {"refinement": {"type": "diffusion_unet", "joint_finetuning": True, "freeze_phase1": True}}
        )


def test_active_refinement_rejects_joint_finetuning():
    with pytest.raises(ConfigValidationError, match="requires a frozen Phase-1"):
        resolve_refinement_config(
            {"refinement": {"type": "diffusion_unet", "joint_finetuning": True}}
        )


def test_all_conditioning_disabled_raises():
    with pytest.raises(ConfigValidationError, match="conditioning"):
        resolve_refinement_config(
            {"refinement": {"type": "diffusion_unet",
                            "conditioning": {k: False for k in
                                             ("deterministic_output", "input_predictors",
                                              "prithvi_features", "unet_features",
                                              "static_fields", "masks")}}}
        )


def test_unsupported_performance_option_raises():
    with pytest.raises(ConfigValidationError, match="Unknown key"):
        resolve_performance_config({"performance": {"use_fast_math": True}})
    with pytest.raises(ConfigValidationError, match="precision.mode"):
        resolve_performance_config({"performance": {"precision": {"mode": "int8"}}})


def test_performance_defaults_are_accuracy_preserving():
    perf = resolve_performance_config({})
    assert perf.precision.mode == "fp32"
    assert perf.precision.allow_tf32 is False
    assert perf.compile.enabled is False
    assert perf.phase1_cache.enabled is False
    assert perf.io.atomic_checkpoints is True


def test_phase1_cache_requires_a_path():
    with pytest.raises(ConfigValidationError, match="no 'path'"):
        resolve_performance_config({"performance": {"phase1_cache": {"enabled": True}}})


def test_phase1_cache_incompatible_with_joint_finetuning():
    cfg = {
        "refinement": {"type": "diffusion_unet", "joint_finetuning": True},
        "performance": {"phase1_cache": {"enabled": True, "path": "/tmp/cache"}},
    }
    with pytest.raises(ConfigValidationError, match="joint_finetuning"):
        resolve_performance_config(cfg)


def test_netcdf_compression_level_bounds():
    with pytest.raises(ConfigValidationError, match="netcdf_compression_level"):
        resolve_performance_config({"performance": {"io": {"netcdf_compression_level": 12}}})


def test_no_lead_time_field_is_accepted_anywhere():
    """Any attempt to smuggle in a forecast lead time must be rejected."""
    for section in ("refinement", "performance"):
        with pytest.raises(ConfigValidationError):
            if section == "refinement":
                resolve_refinement_config({"refinement": {"type": "none", "lead_time_hours": 24}})
            else:
                resolve_performance_config({"performance": {"lead_time_hours": 24}})


def test_train_on_residual_false_is_rejected():
    with pytest.raises(ConfigValidationError, match="ground_truth - frozen_phase1"):
        resolve_refinement_config(
            {"refinement": {"type": "diffusion_unet", "train_on_residual": False}}
        )


def test_transformer_is_zero_initialized_and_serializes_every_semantic_field():
    cfg = resolve_refinement_config({"refinement": {"type": "diffusion_transformer"}})
    assert cfg.transformer.zero_init_output is True
    serialized = cfg.to_dict()
    assert serialized["transformer"]["zero_init_output"] is True
    restored = resolve_refinement_config({"refinement": serialized})
    assert restored.to_dict() == serialized


def test_diffusion_defaults_to_direct_clean_residual_prediction():
    cfg = resolve_refinement_config(
        {"refinement": {"type": "diffusion_transformer"}}
    )
    assert cfg.diffusion.prediction_type == "sample"


def test_all_shipped_diffusion_recipes_lock_clean_residual_prediction():
    root = Path(__file__).resolve().parents[1]
    recipes = (
        "examples/CORDEX_ML/NZ_T1_ACCESS-CM2_static_diffusion_unet.yaml",
        "examples/CORDEX_ML/SA_T2_ACCESS-CM2_static_diffusion_unet.yaml",
        "examples/CORDEX_ML/SA_T2_ACCESS-CM2_static_diffusion_transformer.yaml",
        "examples/MERRA_PRISM/MERRA_PRISM_diffusion_unet.yaml",
        "examples/NARR_PRISM/NARR_PRISM_diffusion_unet.yaml",
        "examples/NARR_PRISM/NARR_PRISM_diffusion_transformer.yaml",
    )
    for relative in recipes:
        path = root / relative
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        cfg = resolve_refinement_config(payload)
        assert cfg.diffusion.prediction_type == "sample", path
        assert cfg.diffusion.clip_sample is False, path


def test_residual_normalization_defaults_to_training_statistics():
    cfg = resolve_refinement_config({"refinement": {"type": "flow_matching_transformer"}})
    assert cfg.residual_normalization.method == "standardize"
    assert cfg.residual_normalization.require_fitted is True
    assert cfg.reconstruction_loss_weight == pytest.approx(0.25)
    assert cfg.multiscale_loss_weight == pytest.approx(0.10)
    assert cfg.gradient_loss_weight == pytest.approx(0.05)
    assert cfg.mean_bias_loss_weight == pytest.approx(0.01)
    assert cfg.flow_matching.mean_path_loss_weight == 0.0
    assert cfg.nonnegative_ensemble_strategy == "memberwise"


def test_flow_mean_path_and_nonnegative_ensemble_strategy_round_trip():
    cfg = resolve_refinement_config(
        {
            "refinement": {
                "type": "flow_matching_unet",
                "nonnegative_ensemble_strategy": "mean_preserving",
                "flow_matching": {"mean_path_loss_weight": 0.25},
            }
        }
    )
    assert cfg.flow_matching.mean_path_loss_weight == pytest.approx(0.25)
    assert cfg.nonnegative_ensemble_strategy == "mean_preserving"
    restored = resolve_refinement_config({"refinement": cfg.to_dict()})
    assert restored.to_dict() == cfg.to_dict()


@pytest.mark.parametrize(
    ("section", "value", "message"),
    (
        ("flow_matching", {"mean_path_loss_weight": -0.1}, "mean_path_loss_weight"),
        ("refinement", {"nonnegative_ensemble_strategy": "truncate"}, "nonnegative_ensemble_strategy"),
    ),
)
def test_invalid_flow_mean_path_and_ensemble_strategy_raise(section, value, message):
    refinement = {"type": "flow_matching_unet"}
    if section == "flow_matching":
        refinement[section] = value
    else:
        refinement.update(value)
    with pytest.raises(ConfigValidationError, match=message):
        resolve_refinement_config({"refinement": refinement})
