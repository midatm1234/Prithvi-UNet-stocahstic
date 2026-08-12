"""Unified two-phase Prithvi-UNet stochastic refinement.

Public API::

    from granitewxc.refinement import (
        RefinementConfig, PerformanceConfig,
        resolve_refinement_config, resolve_performance_config,
        TwoPhaseDownscalingModel, build_two_phase_model,
        build_refiner, available_refiners,
    )

Phase 1 is the existing deterministic Prithvi-WxC encoder + UNet decoder.
Phase 2 is an optional stochastic *residual* refiner selected by
``refinement.type``: ``none``, ``diffusion_unet``, ``flow_matching_unet``,
``diffusion_transformer`` or ``flow_matching_transformer``.

This is a spatial downscaling / bias-correction problem: predictors and targets
always share the same timestamp, there is no forecast lead time, and Transformer
attention operates over 2-D spatial tokens only.
"""

from granitewxc.refinement.base import (
    ResidualRefiner,
    available_refiners,
    build_refiner,
    masked_loss,
    register_refiner,
)
from granitewxc.refinement.config import (
    AuxiliaryLossConfig,
    REFINEMENT_TYPES,
    ConfigValidationError,
    PerformanceConfig,
    RefinementConfig,
    ResidualNormalizationConfig,
    config_fingerprint,
    resolve_performance_config,
    resolve_refinement_config,
)
from granitewxc.refinement.target_space import NormalizedTargetSpace

# Importing the concrete refiners populates the registry.
from granitewxc.refinement import diffusion as _diffusion  # noqa: F401
from granitewxc.refinement import flow_matching as _flow_matching  # noqa: F401

from granitewxc.refinement.training import RefinementTrainer, RefinementTrainState
from granitewxc.refinement.two_phase import (
    TwoPhaseDownscalingModel,
    TwoPhaseOutput,
    build_two_phase_model,
)

__all__ = [
    "REFINEMENT_TYPES",
    "ConfigValidationError",
    "AuxiliaryLossConfig",
    "NormalizedTargetSpace",
    "PerformanceConfig",
    "RefinementConfig",
    "ResidualNormalizationConfig",
    "RefinementTrainState",
    "RefinementTrainer",
    "ResidualRefiner",
    "TwoPhaseDownscalingModel",
    "TwoPhaseOutput",
    "available_refiners",
    "build_refiner",
    "build_two_phase_model",
    "config_fingerprint",
    "masked_loss",
    "register_refiner",
    "resolve_performance_config",
    "resolve_refinement_config",
]
