"""Normalized configuration schema for the two-phase Prithvi-UNet workflow.

This module turns the free-form YAML mappings used across the CORDEX_ML,
MERRA_PRISM and NARR_PRISM cases into a single validated, hashable schema that
is shared by training *and* inference.

Backward compatibility contract
-------------------------------
* A YAML file without a ``model.refinement`` (or top-level ``refinement``)
  section resolves to :class:`RefinementConfig` with ``enabled=False`` and
  ``type="none"``.  In that state the two-phase wrapper is a pass-through and
  the deterministic Prithvi-UNet behaviour is bit-for-bit preserved.
* ``refinement.enabled: false`` and ``refinement.type: none`` are equivalent
  and both preserve deterministic behaviour.
* Unknown refinement names, invalid Transformer geometry and unsupported
  performance options raise :class:`ConfigValidationError` rather than being
  silently coerced.

Scientific-time contract
------------------------
There is **no** predictor/target lead time in this problem.  The only scalar
"time" inputs in this schema are

* ``diffusion.training_timesteps`` / ``diffusion.inference_steps`` -- indices of
  a mathematical noising process, and
* ``flow_matching.integration_steps`` -- the discretisation of the flow ODE
  integration coordinate ``t in [0, 1]``.

Neither is a physical timestamp, a forecast horizon, or a sequence position.
Dataset timestamps never enter this section.
"""

from __future__ import annotations

import copy
import hashlib
import json
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

__all__ = [
    "ConfigValidationError",
    "REFINEMENT_TYPES",
    "ConditioningConfig",
    "DiffusionConfig",
    "FlowMatchingConfig",
    "TransformerConfig",
    "RefinementConfig",
    "PerformanceConfig",
    "resolve_refinement_config",
    "resolve_performance_config",
    "config_fingerprint",
]


class ConfigValidationError(ValueError):
    """Raised for any invalid or mutually incompatible configuration."""


#: Every supported Phase-2 refinement type.  ``none`` disables Phase 2.
REFINEMENT_TYPES: tuple[str, ...] = (
    "none",
    "diffusion_unet",
    "flow_matching_unet",
    "diffusion_transformer",
    "flow_matching_transformer",
)

_REFINEMENT_ALIASES = {
    "": "none",
    "none": "none",
    "off": "none",
    "disabled": "none",
    "deterministic": "none",
    "diffusion": "diffusion_unet",
    "diffusion_unet": "diffusion_unet",
    "unet_diffusion": "diffusion_unet",
    "flow": "flow_matching_unet",
    "flow_matching": "flow_matching_unet",
    "flow_matching_unet": "flow_matching_unet",
    "unet_flow_matching": "flow_matching_unet",
    "diffusion_transformer": "diffusion_transformer",
    "transformer_diffusion": "diffusion_transformer",
    "dit": "diffusion_transformer",
    "flow_matching_transformer": "flow_matching_transformer",
    "transformer_flow_matching": "flow_matching_transformer",
}

_DIFFUSION_TYPES = frozenset({"diffusion_unet", "diffusion_transformer"})
_FLOW_TYPES = frozenset({"flow_matching_unet", "flow_matching_transformer"})
_TRANSFORMER_TYPES = frozenset({"diffusion_transformer", "flow_matching_transformer"})

_PREDICTION_TYPES = ("epsilon", "velocity", "sample")
_SCHEDULES = ("cosine", "linear", "scaled_linear")
_SOLVERS = ("euler", "heun", "midpoint")
_SOURCE_DISTRIBUTIONS = ("gaussian",)
_POSITIONAL_ENCODINGS = ("learned_2d", "sincos_2d")
_ATTENTION_IMPLEMENTATIONS = ("auto", "sdpa", "math")
_PRECISION_MODES = ("fp32", "bf16", "fp16")


def _as_mapping(value: Any) -> dict[str, Any]:
    """Coerce YAML/namespace-ish objects into a plain ``dict``."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    if hasattr(value, "__dict__"):
        return {k: v for k, v in vars(value).items() if not k.startswith("__")}
    raise ConfigValidationError(
        f"Expected a mapping for a configuration section, got {type(value).__name__}"
    )


def _reject_unknown(section: str, data: Mapping[str, Any], known: Sequence[str]) -> None:
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ConfigValidationError(
            f"Unknown key(s) {unknown} in '{section}'. Supported keys: {sorted(known)}."
        )


def _as_bool(section: str, key: str, value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "on", "1"}:
            return True
        if text in {"false", "no", "off", "0"}:
            return False
    raise ConfigValidationError(f"{section}.{key} must be a boolean, got {value!r}")


def _as_int(section: str, key: str, value: Any, default: int, *, minimum: int = 1) -> int:
    if value is None:
        return default
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(f"{section}.{key} must be an integer, got {value!r}") from exc
    if out < minimum:
        raise ConfigValidationError(f"{section}.{key} must be >= {minimum}, got {out}")
    return out


def _as_float(
    section: str,
    key: str,
    value: Any,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(f"{section}.{key} must be a number, got {value!r}") from exc
    if minimum is not None and out < minimum:
        raise ConfigValidationError(f"{section}.{key} must be >= {minimum}, got {out}")
    if maximum is not None and out > maximum:
        raise ConfigValidationError(f"{section}.{key} must be <= {maximum}, got {out}")
    return out


def _as_choice(section: str, key: str, value: Any, default: str, choices: Sequence[str]) -> str:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text not in choices:
        raise ConfigValidationError(
            f"{section}.{key} must be one of {list(choices)}, got {value!r}"
        )
    return text


# ---------------------------------------------------------------------------
# Sub-sections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConditioningConfig:
    """Which spatial fields are concatenated into the Phase-2 conditioning."""

    deterministic_output: bool = True
    input_predictors: bool = True
    prithvi_features: bool = False
    unet_features: bool = False
    static_fields: bool = True
    masks: bool = True

    _KEYS = (
        "deterministic_output",
        "input_predictors",
        "prithvi_features",
        "unet_features",
        "static_fields",
        "masks",
    )

    @classmethod
    def from_mapping(cls, data: Any) -> "ConditioningConfig":
        raw = _as_mapping(data)
        _reject_unknown("refinement.conditioning", raw, cls._KEYS)
        defaults = cls()
        kwargs = {
            key: _as_bool("refinement.conditioning", key, raw.get(key), getattr(defaults, key))
            for key in cls._KEYS
        }
        out = cls(**kwargs)
        if not any(getattr(out, key) for key in cls._KEYS):
            raise ConfigValidationError(
                "refinement.conditioning disables every input; the refiner would have "
                "no conditioning at all. Enable at least one conditioning source."
            )
        return out

    def to_dict(self) -> dict[str, bool]:
        return {key: getattr(self, key) for key in self._KEYS}


@dataclass(frozen=True)
class DiffusionConfig:
    """Discrete-time DDPM settings for the ``diffusion_*`` refiners.

    ``training_timesteps`` and ``inference_steps`` index a *noise* process. They
    are unrelated to dataset timestamps and carry no forecast semantics.
    """

    training_timesteps: int = 1000
    inference_steps: int = 50
    prediction_type: str = "epsilon"
    schedule: str = "cosine"
    beta_start: float = 1e-4
    beta_end: float = 0.02
    cosine_s: float = 8e-3
    clip_sample: bool = False
    clip_sample_range: float = 10.0
    eta: float = 0.0

    _KEYS = (
        "training_timesteps",
        "inference_steps",
        "prediction_type",
        "schedule",
        "beta_start",
        "beta_end",
        "cosine_s",
        "clip_sample",
        "clip_sample_range",
        "eta",
    )

    @classmethod
    def from_mapping(cls, data: Any) -> "DiffusionConfig":
        raw = _as_mapping(data)
        _reject_unknown("refinement.diffusion", raw, cls._KEYS)
        d = cls()
        sec = "refinement.diffusion"
        out = cls(
            training_timesteps=_as_int(sec, "training_timesteps", raw.get("training_timesteps"), d.training_timesteps),
            inference_steps=_as_int(sec, "inference_steps", raw.get("inference_steps"), d.inference_steps),
            prediction_type=_as_choice(sec, "prediction_type", raw.get("prediction_type"), d.prediction_type, _PREDICTION_TYPES),
            schedule=_as_choice(sec, "schedule", raw.get("schedule"), d.schedule, _SCHEDULES),
            beta_start=_as_float(sec, "beta_start", raw.get("beta_start"), d.beta_start, minimum=0.0),
            beta_end=_as_float(sec, "beta_end", raw.get("beta_end"), d.beta_end, minimum=0.0),
            cosine_s=_as_float(sec, "cosine_s", raw.get("cosine_s"), d.cosine_s, minimum=0.0),
            clip_sample=_as_bool(sec, "clip_sample", raw.get("clip_sample"), d.clip_sample),
            clip_sample_range=_as_float(sec, "clip_sample_range", raw.get("clip_sample_range"), d.clip_sample_range, minimum=0.0),
            eta=_as_float(sec, "eta", raw.get("eta"), d.eta, minimum=0.0, maximum=1.0),
        )
        if out.inference_steps > out.training_timesteps:
            raise ConfigValidationError(
                "refinement.diffusion.inference_steps "
                f"({out.inference_steps}) cannot exceed training_timesteps "
                f"({out.training_timesteps})."
            )
        if out.schedule in {"linear", "scaled_linear"} and out.beta_end <= out.beta_start:
            raise ConfigValidationError(
                "refinement.diffusion.beta_end must be greater than beta_start for "
                f"the '{out.schedule}' schedule."
            )
        return out

    def to_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self._KEYS}


@dataclass(frozen=True)
class FlowMatchingConfig:
    """Conditional flow-matching settings for the ``flow_matching_*`` refiners.

    ``t`` is the interpolation coordinate of the probability path, integrated
    from 0 (source distribution) to 1 (residual distribution).  It is a purely
    mathematical integration variable.
    """

    integration_steps: int = 50
    solver: str = "euler"
    source_distribution: str = "gaussian"
    stochastic_initialization: bool = True
    sigma_min: float = 1e-4
    time_sampling: str = "uniform"
    logit_normal_mean: float = -0.5
    logit_normal_std: float = 1.2

    _KEYS = (
        "integration_steps",
        "solver",
        "source_distribution",
        "stochastic_initialization",
        "sigma_min",
        "time_sampling",
        "logit_normal_mean",
        "logit_normal_std",
    )

    @classmethod
    def from_mapping(cls, data: Any) -> "FlowMatchingConfig":
        raw = _as_mapping(data)
        _reject_unknown("refinement.flow_matching", raw, cls._KEYS)
        d = cls()
        sec = "refinement.flow_matching"
        return cls(
            integration_steps=_as_int(sec, "integration_steps", raw.get("integration_steps"), d.integration_steps),
            solver=_as_choice(sec, "solver", raw.get("solver"), d.solver, _SOLVERS),
            source_distribution=_as_choice(sec, "source_distribution", raw.get("source_distribution"), d.source_distribution, _SOURCE_DISTRIBUTIONS),
            stochastic_initialization=_as_bool(sec, "stochastic_initialization", raw.get("stochastic_initialization"), d.stochastic_initialization),
            sigma_min=_as_float(sec, "sigma_min", raw.get("sigma_min"), d.sigma_min, minimum=0.0, maximum=0.5),
            time_sampling=_as_choice(sec, "time_sampling", raw.get("time_sampling"), d.time_sampling, ("uniform", "logit_normal")),
            logit_normal_mean=_as_float(sec, "logit_normal_mean", raw.get("logit_normal_mean"), d.logit_normal_mean),
            logit_normal_std=_as_float(sec, "logit_normal_std", raw.get("logit_normal_std"), d.logit_normal_std, minimum=1e-6),
        )

    def to_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self._KEYS}


@dataclass(frozen=True)
class TransformerConfig:
    """Spatial-only Transformer geometry.

    Attention is applied across *spatial* tokens (2-D patches of the target
    grid) from a single sample and a single timestamp.  There is deliberately
    no temporal axis, no causal mask and no lead-time embedding.
    """

    patch_size: tuple[int, int] = (4, 4)
    embedding_dim: int = 256
    num_heads: int = 8
    num_blocks: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    positional_encoding: str = "learned_2d"
    max_tokens_lat: int = 256
    max_tokens_lon: int = 256
    gradient_checkpointing: bool = False
    optimized_attention: str = "auto"

    _KEYS = (
        "patch_size",
        "patch_height",
        "patch_width",
        "embedding_dim",
        "num_heads",
        "num_blocks",
        "mlp_ratio",
        "dropout",
        "positional_encoding",
        "max_tokens_lat",
        "max_tokens_lon",
        "gradient_checkpointing",
        "optimized_attention",
    )

    @classmethod
    def from_mapping(cls, data: Any) -> "TransformerConfig":
        raw = _as_mapping(data)
        _reject_unknown("refinement.transformer", raw, cls._KEYS)
        d = cls()
        sec = "refinement.transformer"

        patch = raw.get("patch_size")
        if patch is None:
            ph = _as_int(sec, "patch_height", raw.get("patch_height"), d.patch_size[0])
            pw = _as_int(sec, "patch_width", raw.get("patch_width"), d.patch_size[1])
        elif isinstance(patch, (list, tuple)):
            if len(patch) != 2:
                raise ConfigValidationError(
                    f"{sec}.patch_size must be an int or a [height, width] pair, got {patch!r}"
                )
            ph = _as_int(sec, "patch_size[0]", patch[0], d.patch_size[0])
            pw = _as_int(sec, "patch_size[1]", patch[1], d.patch_size[1])
            if raw.get("patch_height") is not None or raw.get("patch_width") is not None:
                raise ConfigValidationError(
                    f"{sec}: set either patch_size or patch_height/patch_width, not both."
                )
        else:
            shared = _as_int(sec, "patch_size", patch, d.patch_size[0])
            ph = pw = shared
            if raw.get("patch_height") is not None or raw.get("patch_width") is not None:
                raise ConfigValidationError(
                    f"{sec}: set either patch_size or patch_height/patch_width, not both."
                )

        embedding_dim = _as_int(sec, "embedding_dim", raw.get("embedding_dim"), d.embedding_dim)
        num_heads = _as_int(sec, "num_heads", raw.get("num_heads"), d.num_heads)
        if embedding_dim % num_heads != 0:
            raise ConfigValidationError(
                f"{sec}.embedding_dim ({embedding_dim}) must be divisible by "
                f"num_heads ({num_heads}); head_dim would be "
                f"{embedding_dim / num_heads:.3f}."
            )

        out = cls(
            patch_size=(ph, pw),
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_blocks=_as_int(sec, "num_blocks", raw.get("num_blocks"), d.num_blocks),
            mlp_ratio=_as_float(sec, "mlp_ratio", raw.get("mlp_ratio"), d.mlp_ratio, minimum=0.5),
            dropout=_as_float(sec, "dropout", raw.get("dropout"), d.dropout, minimum=0.0, maximum=1.0),
            positional_encoding=_as_choice(sec, "positional_encoding", raw.get("positional_encoding"), d.positional_encoding, _POSITIONAL_ENCODINGS),
            max_tokens_lat=_as_int(sec, "max_tokens_lat", raw.get("max_tokens_lat"), d.max_tokens_lat),
            max_tokens_lon=_as_int(sec, "max_tokens_lon", raw.get("max_tokens_lon"), d.max_tokens_lon),
            gradient_checkpointing=_as_bool(sec, "gradient_checkpointing", raw.get("gradient_checkpointing"), d.gradient_checkpointing),
            optimized_attention=_as_choice(sec, "optimized_attention", raw.get("optimized_attention"), d.optimized_attention, _ATTENTION_IMPLEMENTATIONS),
        )
        if int(embedding_dim // num_heads) % 2 != 0:
            raise ConfigValidationError(
                f"{sec}: head dimension ({embedding_dim // num_heads}) must be even so "
                "the 2-D sin/cos positional encoding can be split across axes."
            )
        return out

    def to_dict(self) -> dict[str, Any]:
        data = {
            "patch_size": list(self.patch_size),
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "num_blocks": self.num_blocks,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout,
            "positional_encoding": self.positional_encoding,
            "max_tokens_lat": self.max_tokens_lat,
            "max_tokens_lon": self.max_tokens_lon,
            "gradient_checkpointing": self.gradient_checkpointing,
            "optimized_attention": self.optimized_attention,
        }
        return data


@dataclass(frozen=True)
class UNetRefinerConfig:
    """Geometry of the convolutional (UNet) residual refiners."""

    hidden_channels: int = 64
    num_levels: int = 3
    time_embedding_dim: int = 128
    dropout: float = 0.0
    bottleneck_attention: bool = True
    attention_heads: int = 4
    zero_init_output: bool = True

    _KEYS = (
        "hidden_channels",
        "num_levels",
        "time_embedding_dim",
        "dropout",
        "bottleneck_attention",
        "attention_heads",
        "zero_init_output",
    )

    @classmethod
    def from_mapping(cls, data: Any) -> "UNetRefinerConfig":
        raw = _as_mapping(data)
        _reject_unknown("refinement.unet", raw, cls._KEYS)
        d = cls()
        sec = "refinement.unet"
        out = cls(
            hidden_channels=_as_int(sec, "hidden_channels", raw.get("hidden_channels"), d.hidden_channels),
            num_levels=_as_int(sec, "num_levels", raw.get("num_levels"), d.num_levels),
            time_embedding_dim=_as_int(sec, "time_embedding_dim", raw.get("time_embedding_dim"), d.time_embedding_dim, minimum=2),
            dropout=_as_float(sec, "dropout", raw.get("dropout"), d.dropout, minimum=0.0, maximum=1.0),
            bottleneck_attention=_as_bool(sec, "bottleneck_attention", raw.get("bottleneck_attention"), d.bottleneck_attention),
            attention_heads=_as_int(sec, "attention_heads", raw.get("attention_heads"), d.attention_heads),
            zero_init_output=_as_bool(sec, "zero_init_output", raw.get("zero_init_output"), d.zero_init_output),
        )
        if out.time_embedding_dim % 2 != 0:
            raise ConfigValidationError(
                f"{sec}.time_embedding_dim must be even (sin/cos pairs), got "
                f"{out.time_embedding_dim}."
            )
        bottleneck_channels = out.hidden_channels * (2 ** (out.num_levels - 1))
        if out.bottleneck_attention and bottleneck_channels % out.attention_heads != 0:
            raise ConfigValidationError(
                f"{sec}: bottleneck channels ({bottleneck_channels}) must be divisible by "
                f"attention_heads ({out.attention_heads})."
            )
        return out

    def to_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self._KEYS}


@dataclass(frozen=True)
class RefinementConfig:
    """Fully-resolved Phase-2 configuration."""

    enabled: bool = False
    type: str = "none"
    checkpoint: str | None = None
    freeze_phase1: bool = True
    joint_finetuning: bool = False
    train_on_residual: bool = True
    ensemble_size: int = 1
    loss: str = "mse"
    seed: int | None = None
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    flow_matching: FlowMatchingConfig = field(default_factory=FlowMatchingConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    unet: UNetRefinerConfig = field(default_factory=UNetRefinerConfig)

    _KEYS = (
        "enabled",
        "type",
        "checkpoint",
        "freeze_phase1",
        "joint_finetuning",
        "train_on_residual",
        "ensemble_size",
        "loss",
        "seed",
        "conditioning",
        "diffusion",
        "flow_matching",
        "transformer",
        "unet",
    )

    # -- derived helpers -------------------------------------------------
    @property
    def is_active(self) -> bool:
        return bool(self.enabled) and self.type != "none"

    @property
    def is_diffusion(self) -> bool:
        return self.type in _DIFFUSION_TYPES

    @property
    def is_flow_matching(self) -> bool:
        return self.type in _FLOW_TYPES

    @property
    def uses_transformer(self) -> bool:
        return self.type in _TRANSFORMER_TYPES

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "type": self.type,
            "checkpoint": self.checkpoint,
            "freeze_phase1": self.freeze_phase1,
            "joint_finetuning": self.joint_finetuning,
            "train_on_residual": self.train_on_residual,
            "ensemble_size": self.ensemble_size,
            "loss": self.loss,
            "seed": self.seed,
            "conditioning": self.conditioning.to_dict(),
            "diffusion": self.diffusion.to_dict(),
            "flow_matching": self.flow_matching.to_dict(),
            "transformer": self.transformer.to_dict(),
            "unet": self.unet.to_dict(),
        }


@dataclass(frozen=True)
class DataloaderPerfConfig:
    num_workers: int | None = None  # ``None`` == "auto"
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    non_blocking_transfer: bool = True

    _KEYS = (
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
        "non_blocking_transfer",
    )

    @classmethod
    def from_mapping(cls, data: Any) -> "DataloaderPerfConfig":
        raw = _as_mapping(data)
        _reject_unknown("performance.dataloader", raw, cls._KEYS)
        d = cls()
        sec = "performance.dataloader"
        workers_raw = raw.get("num_workers")
        if workers_raw is None or str(workers_raw).strip().lower() == "auto":
            workers = None
        else:
            workers = _as_int(sec, "num_workers", workers_raw, 0, minimum=0)
        return cls(
            num_workers=workers,
            pin_memory=_as_bool(sec, "pin_memory", raw.get("pin_memory"), d.pin_memory),
            persistent_workers=_as_bool(sec, "persistent_workers", raw.get("persistent_workers"), d.persistent_workers),
            prefetch_factor=_as_int(sec, "prefetch_factor", raw.get("prefetch_factor"), d.prefetch_factor),
            non_blocking_transfer=_as_bool(sec, "non_blocking_transfer", raw.get("non_blocking_transfer"), d.non_blocking_transfer),
        )

    def to_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self._KEYS}


@dataclass(frozen=True)
class PrecisionPerfConfig:
    mode: str = "fp32"
    allow_tf32: bool = False

    _KEYS = ("mode", "allow_tf32")

    @classmethod
    def from_mapping(cls, data: Any) -> "PrecisionPerfConfig":
        raw = _as_mapping(data)
        _reject_unknown("performance.precision", raw, cls._KEYS)
        d = cls()
        return cls(
            mode=_as_choice("performance.precision", "mode", raw.get("mode"), d.mode, _PRECISION_MODES),
            allow_tf32=_as_bool("performance.precision", "allow_tf32", raw.get("allow_tf32"), d.allow_tf32),
        )

    def to_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self._KEYS}


@dataclass(frozen=True)
class CompilePerfConfig:
    enabled: bool = False
    phase1: bool = False
    refinement: bool = False
    mode: str = "default"

    _KEYS = ("enabled", "phase1", "refinement", "mode")

    @classmethod
    def from_mapping(cls, data: Any) -> "CompilePerfConfig":
        raw = _as_mapping(data)
        _reject_unknown("performance.compile", raw, cls._KEYS)
        d = cls()
        sec = "performance.compile"
        return cls(
            enabled=_as_bool(sec, "enabled", raw.get("enabled"), d.enabled),
            phase1=_as_bool(sec, "phase1", raw.get("phase1"), d.phase1),
            refinement=_as_bool(sec, "refinement", raw.get("refinement"), d.refinement),
            mode=_as_choice(sec, "mode", raw.get("mode"), d.mode, ("default", "reduce-overhead", "max-autotune")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self._KEYS}


@dataclass(frozen=True)
class Phase1CachePerfConfig:
    enabled: bool = False
    path: str | None = None
    include_deterministic_output: bool = True
    include_prithvi_features: bool = False
    include_unet_features: bool = False
    validate_cache: bool = True
    validate_samples: int = 4
    validate_tolerance: float = 1e-5

    _KEYS = (
        "enabled",
        "path",
        "include_deterministic_output",
        "include_prithvi_features",
        "include_unet_features",
        "validate_cache",
        "validate_samples",
        "validate_tolerance",
    )

    @classmethod
    def from_mapping(cls, data: Any) -> "Phase1CachePerfConfig":
        raw = _as_mapping(data)
        _reject_unknown("performance.phase1_cache", raw, cls._KEYS)
        d = cls()
        sec = "performance.phase1_cache"
        path = raw.get("path")
        out = cls(
            enabled=_as_bool(sec, "enabled", raw.get("enabled"), d.enabled),
            path=None if path in (None, "", "null") else str(path),
            include_deterministic_output=_as_bool(sec, "include_deterministic_output", raw.get("include_deterministic_output"), d.include_deterministic_output),
            include_prithvi_features=_as_bool(sec, "include_prithvi_features", raw.get("include_prithvi_features"), d.include_prithvi_features),
            include_unet_features=_as_bool(sec, "include_unet_features", raw.get("include_unet_features"), d.include_unet_features),
            validate_cache=_as_bool(sec, "validate_cache", raw.get("validate_cache"), d.validate_cache),
            validate_samples=_as_int(sec, "validate_samples", raw.get("validate_samples"), d.validate_samples, minimum=0),
            validate_tolerance=_as_float(sec, "validate_tolerance", raw.get("validate_tolerance"), d.validate_tolerance, minimum=0.0),
        )
        if out.enabled and not out.path:
            raise ConfigValidationError(
                "performance.phase1_cache.enabled is true but no 'path' was provided."
            )
        return out

    def to_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self._KEYS}


@dataclass(frozen=True)
class EnsemblePerfConfig:
    batch_members: bool = True
    chunk_size: int | None = None  # ``None`` == "auto"

    _KEYS = ("batch_members", "chunk_size")

    @classmethod
    def from_mapping(cls, data: Any) -> "EnsemblePerfConfig":
        raw = _as_mapping(data)
        _reject_unknown("performance.ensemble", raw, cls._KEYS)
        d = cls()
        sec = "performance.ensemble"
        chunk_raw = raw.get("chunk_size")
        if chunk_raw is None or str(chunk_raw).strip().lower() == "auto":
            chunk = None
        else:
            chunk = _as_int(sec, "chunk_size", chunk_raw, 1)
        return cls(
            batch_members=_as_bool(sec, "batch_members", raw.get("batch_members"), d.batch_members),
            chunk_size=chunk,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"batch_members": self.batch_members, "chunk_size": self.chunk_size}


@dataclass(frozen=True)
class IoPerfConfig:
    atomic_checkpoints: bool = True
    netcdf_compression: bool = True
    netcdf_compression_level: int = 4

    _KEYS = ("atomic_checkpoints", "netcdf_compression", "netcdf_compression_level")

    @classmethod
    def from_mapping(cls, data: Any) -> "IoPerfConfig":
        raw = _as_mapping(data)
        _reject_unknown("performance.io", raw, cls._KEYS)
        d = cls()
        sec = "performance.io"
        level = _as_int(sec, "netcdf_compression_level", raw.get("netcdf_compression_level"), d.netcdf_compression_level, minimum=1)
        if level > 9:
            raise ConfigValidationError(
                f"{sec}.netcdf_compression_level must be in [1, 9], got {level}."
            )
        return cls(
            atomic_checkpoints=_as_bool(sec, "atomic_checkpoints", raw.get("atomic_checkpoints"), d.atomic_checkpoints),
            netcdf_compression=_as_bool(sec, "netcdf_compression", raw.get("netcdf_compression"), d.netcdf_compression),
            netcdf_compression_level=level,
        )

    def to_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self._KEYS}


@dataclass(frozen=True)
class PerformanceConfig:
    """Workflow-only settings.

    Nothing in this section may change model architecture, loss, sampler,
    solver, ensemble size or evaluation data.  Unsupported requests raise or
    warn instead of silently degrading to a different numerical behaviour.
    """

    profile: bool = False
    dataloader: DataloaderPerfConfig = field(default_factory=DataloaderPerfConfig)
    precision: PrecisionPerfConfig = field(default_factory=PrecisionPerfConfig)
    compile: CompilePerfConfig = field(default_factory=CompilePerfConfig)
    phase1_cache: Phase1CachePerfConfig = field(default_factory=Phase1CachePerfConfig)
    ensemble: EnsemblePerfConfig = field(default_factory=EnsemblePerfConfig)
    io: IoPerfConfig = field(default_factory=IoPerfConfig)

    _KEYS = ("profile", "dataloader", "precision", "compile", "phase1_cache", "ensemble", "io")

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "dataloader": self.dataloader.to_dict(),
            "precision": self.precision.to_dict(),
            "compile": self.compile.to_dict(),
            "phase1_cache": self.phase1_cache.to_dict(),
            "ensemble": self.ensemble.to_dict(),
            "io": self.io.to_dict(),
        }


# ---------------------------------------------------------------------------
# Resolution entry points
# ---------------------------------------------------------------------------


def _locate_section(config: Any, name: str) -> Any:
    """Find ``name`` on ``config`` or on ``config.model`` (both are supported)."""
    if config is None:
        return None
    holders = [config]
    if isinstance(config, Mapping):
        holders.append(config.get("model"))
    else:
        holders.append(getattr(config, "model", None))
    for holder in holders:
        if holder is None:
            continue
        if isinstance(holder, Mapping):
            if name in holder:
                return holder[name]
        elif hasattr(holder, name):
            value = getattr(holder, name)
            if value is not None:
                return value
    return None


def resolve_refinement_config(config: Any) -> RefinementConfig:
    """Normalize the ``refinement`` section of an experiment configuration.

    Accepts an :class:`~granitewxc.utils.config.ExperimentConfig`, a raw YAML
    mapping, or ``None``.  A missing section yields the deterministic default.
    """
    raw_section = _locate_section(config, "refinement")
    if raw_section is None:
        return RefinementConfig()

    raw = _as_mapping(raw_section)
    _reject_unknown("refinement", raw, RefinementConfig._KEYS)

    raw_type = raw.get("type")
    if raw_type is None:
        resolved_type = "none"
    else:
        key = str(raw_type).strip().lower()
        if key not in _REFINEMENT_ALIASES:
            raise ConfigValidationError(
                f"Unknown refinement.type {raw_type!r}. Supported values: "
                f"{list(REFINEMENT_TYPES)}."
            )
        resolved_type = _REFINEMENT_ALIASES[key]

    enabled_raw = raw.get("enabled")
    if enabled_raw is None:
        enabled = resolved_type != "none"
    else:
        enabled = _as_bool("refinement", "enabled", enabled_raw, True)

    if enabled and resolved_type == "none":
        # Explicit "enabled but type=none" is contradictory only if the user
        # also asked for stochastic settings; treat it as deterministic and warn.
        warnings.warn(
            "refinement.enabled is true but refinement.type is 'none'; "
            "running the deterministic Prithvi-UNet without Phase 2.",
            RuntimeWarning,
            stacklevel=2,
        )
        enabled = False

    checkpoint = raw.get("checkpoint")
    checkpoint = None if checkpoint in (None, "", "null") else str(checkpoint)

    freeze_phase1 = _as_bool("refinement", "freeze_phase1", raw.get("freeze_phase1"), True)
    joint = _as_bool("refinement", "joint_finetuning", raw.get("joint_finetuning"), False)
    if joint and freeze_phase1 and raw.get("freeze_phase1") is not None:
        raise ConfigValidationError(
            "refinement.joint_finetuning=true is incompatible with "
            "refinement.freeze_phase1=true. Joint fine-tuning must be requested "
            "explicitly and requires an unfrozen Phase 1."
        )
    if joint:
        freeze_phase1 = False

    train_on_residual = _as_bool("refinement", "train_on_residual", raw.get("train_on_residual"), True)
    if not train_on_residual and enabled:
        warnings.warn(
            "refinement.train_on_residual=false makes Phase 2 predict the full "
            "normalized target instead of the residual. This is a scientific "
            "change, not an optimization.",
            RuntimeWarning,
            stacklevel=2,
        )

    ensemble_size = _as_int("refinement", "ensemble_size", raw.get("ensemble_size"), 1)
    seed_raw = raw.get("seed")
    seed = None if seed_raw is None else int(seed_raw)

    cfg = RefinementConfig(
        enabled=enabled,
        type=resolved_type if enabled else "none",
        checkpoint=checkpoint,
        freeze_phase1=freeze_phase1,
        joint_finetuning=joint,
        train_on_residual=train_on_residual,
        ensemble_size=ensemble_size,
        loss=_as_choice("refinement", "loss", raw.get("loss"), "mse", ("mse", "l1", "huber")),
        seed=seed,
        conditioning=ConditioningConfig.from_mapping(raw.get("conditioning")),
        diffusion=DiffusionConfig.from_mapping(raw.get("diffusion")),
        flow_matching=FlowMatchingConfig.from_mapping(raw.get("flow_matching")),
        transformer=TransformerConfig.from_mapping(raw.get("transformer")),
        unet=UNetRefinerConfig.from_mapping(raw.get("unet")),
    )

    if cfg.is_active and cfg.type in _DIFFUSION_TYPES and raw.get("flow_matching"):
        warnings.warn(
            "refinement.flow_matching settings are ignored for "
            f"refinement.type={cfg.type!r}.",
            RuntimeWarning,
            stacklevel=2,
        )
    if cfg.is_active and cfg.type in _FLOW_TYPES and raw.get("diffusion"):
        warnings.warn(
            "refinement.diffusion settings are ignored for "
            f"refinement.type={cfg.type!r}.",
            RuntimeWarning,
            stacklevel=2,
        )
    return cfg


def resolve_performance_config(config: Any) -> PerformanceConfig:
    """Normalize the ``performance`` section of an experiment configuration."""
    raw_section = _locate_section(config, "performance")
    if raw_section is None:
        return PerformanceConfig()

    raw = _as_mapping(raw_section)
    _reject_unknown("performance", raw, PerformanceConfig._KEYS)

    cfg = PerformanceConfig(
        profile=_as_bool("performance", "profile", raw.get("profile"), False),
        dataloader=DataloaderPerfConfig.from_mapping(raw.get("dataloader")),
        precision=PrecisionPerfConfig.from_mapping(raw.get("precision")),
        compile=CompilePerfConfig.from_mapping(raw.get("compile")),
        phase1_cache=Phase1CachePerfConfig.from_mapping(raw.get("phase1_cache")),
        ensemble=EnsemblePerfConfig.from_mapping(raw.get("ensemble")),
        io=IoPerfConfig.from_mapping(raw.get("io")),
    )

    refinement = resolve_refinement_config(config)
    if cfg.phase1_cache.enabled and refinement.joint_finetuning:
        raise ConfigValidationError(
            "performance.phase1_cache.enabled is incompatible with "
            "refinement.joint_finetuning: cached conditioning would be stale as "
            "soon as Phase-1 weights are updated."
        )
    if cfg.phase1_cache.enabled and not refinement.freeze_phase1:
        raise ConfigValidationError(
            "performance.phase1_cache.enabled requires refinement.freeze_phase1=true."
        )
    if cfg.phase1_cache.include_prithvi_features and not refinement.conditioning.prithvi_features:
        warnings.warn(
            "performance.phase1_cache.include_prithvi_features is enabled but "
            "refinement.conditioning.prithvi_features is false; the cached "
            "features would never be read.",
            RuntimeWarning,
            stacklevel=2,
        )
    if cfg.compile.enabled and not (cfg.compile.phase1 or cfg.compile.refinement):
        warnings.warn(
            "performance.compile.enabled is true but neither compile.phase1 nor "
            "compile.refinement is enabled; nothing will be compiled.",
            RuntimeWarning,
            stacklevel=2,
        )
    return cfg


def config_fingerprint(payload: Any) -> str:
    """Stable SHA-256 fingerprint of a JSON-serialisable configuration payload."""

    def _default(obj: Any) -> Any:
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        if isinstance(obj, (set, frozenset)):
            return sorted(obj)
        return str(obj)

    blob = json.dumps(copy.deepcopy(payload), sort_keys=True, default=_default)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
