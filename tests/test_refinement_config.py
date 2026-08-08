"""Configuration-schema tests for the two-phase stochastic refinement."""

from __future__ import annotations

import warnings

import pytest

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


def test_joint_finetuning_implies_unfrozen_phase1():
    cfg = resolve_refinement_config(
        {"refinement": {"type": "diffusion_unet", "joint_finetuning": True}}
    )
    assert cfg.joint_finetuning is True
    assert cfg.freeze_phase1 is False


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


def test_train_on_residual_false_warns():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = resolve_refinement_config(
            {"refinement": {"type": "diffusion_unet", "train_on_residual": False}}
        )
    assert cfg.train_on_residual is False
    assert any("train_on_residual" in str(w.message) for w in caught)
