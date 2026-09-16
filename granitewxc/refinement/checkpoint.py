"""Checkpoint compatibility for the two-phase Prithvi-UNet.

Design rules
------------
* Existing deterministic checkpoints (CORDEX_ML / MERRA_PRISM / NARR_PRISM) are
  **never** modified, renamed or converted in place. They are read-only inputs.
* Phase-1 module names are preserved. The only change is the ``phase1.`` prefix
  introduced by :class:`~granitewxc.refinement.two_phase.TwoPhaseDownscalingModel`.
  :func:`migrate_phase1_state_dict` performs that mapping explicitly, together
  with the historical ``module.``/``_orig_mod.`` wrapper prefixes.
* Phase-1 weights are loaded **strictly** after migration. ``strict=False`` is
  never used as a blanket escape hatch: :func:`load_phase1_state_dict` verifies
  that every missing key belongs to the newly introduced Phase-2 module and that
  no Phase-1 key is unexpected or shape-mismatched.
* Refinement-only checkpoints record the identity (SHA-256 over the Phase-1
  tensors) of the Phase-1 checkpoint they were trained against.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import warnings
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import torch

__all__ = [
    "CHECKPOINT_KIND_PHASE1",
    "CHECKPOINT_KIND_REFINEMENT",
    "CHECKPOINT_KIND_COMBINED",
    "CHECKPOINT_SCHEMA_VERSION",
    "REFINEMENT_CONTRACT_VERSION",
    "StateDictReport",
    "extract_model_state",
    "strip_wrapper_prefixes",
    "migrate_phase1_state_dict",
    "phase1_state_fingerprint",
    "load_phase1_state_dict",
    "load_refinement_state_dict",
    "save_checkpoint_atomic",
    "build_refinement_checkpoint",
    "validate_phase1_reference",
]

CHECKPOINT_KIND_PHASE1 = "phase1"
CHECKPOINT_KIND_REFINEMENT = "refinement"
CHECKPOINT_KIND_COMBINED = "combined"

# Schema 2 introduces the explicit physical-residual scientific contract and
# persisted residual-normalizer state. Schema-1 Phase-2 weights modeled a
# different normalized-target delta and must never be reinterpreted as physical
# residuals. Versioned schema-1 Phase-1 checkpoints remain valid read-only
# deterministic inputs.
CHECKPOINT_SCHEMA_VERSION = 2
# Contract 2 uses a regularized flow path, clean-residual diffusion by default,
# endpoint-inclusive DDIM indexing, and detached correction-gate calibration.
# Reject earlier schema-2 experiments rather than reinterpret their weights.
REFINEMENT_CONTRACT_VERSION = 2
_SUPPORTED_PHASE1_CHECKPOINT_SCHEMAS = frozenset({1, CHECKPOINT_SCHEMA_VERSION})

_REFINER_PREFIX = "refiner."
_RESIDUAL_NORMALIZER_PREFIX = "residual_normalizer."
_REFINEMENT_STATE_PREFIXES = (_REFINER_PREFIX, _RESIDUAL_NORMALIZER_PREFIX)

# These settings change implementation efficiency or initialization, but do
# not change the function represented by a fully-loaded state dict. All other
# active backbone settings are part of the scientific compatibility contract.
_NON_SCIENTIFIC_BACKBONE_KEYS = frozenset(
    {"gradient_checkpointing", "optimized_attention"}
)

#: Wrapper prefixes historically produced by DDP / FSDP / ``torch.compile``.
_WRAPPER_PREFIXES = ("module.", "_orig_mod.")

#: Explicit legacy -> current Phase-1 key renames. Empty today: the NARR_PRISM
#: deterministic architecture is byte-compatible with this branch. Any future
#: rename must be added here (never handled by ``strict=False``).
LEGACY_PHASE1_KEY_RENAMES: dict[str, str] = {}


@dataclass
class StateDictReport:
    """Outcome of a controlled state-dict load."""

    loaded: int = 0
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    shape_mismatched: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = field(default_factory=list)
    renamed: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"loaded={self.loaded} missing={len(self.missing)} "
            f"unexpected={len(self.unexpected)} shape_mismatched={len(self.shape_mismatched)} "
            f"renamed={len(self.renamed)}"
        )


def extract_model_state(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Return the model tensors from any of the checkpoint layouts in use."""
    if checkpoint is None:
        raise ValueError("Checkpoint is empty.")
    # migrate_phase1_state_dict returns (state_dict, renames). If a caller
    # accidentally passes that tuple here instead of the raw checkpoint, recover
    # gracefully by using the first element.
    if isinstance(checkpoint, tuple) and len(checkpoint) == 2 and isinstance(checkpoint[0], Mapping):
        checkpoint = checkpoint[0]
    state = checkpoint
    if isinstance(checkpoint, Mapping):
        for key in ("model", "state_dict", "model_state_dict", "phase1"):
            if key in checkpoint and isinstance(checkpoint[key], Mapping):
                state = checkpoint[key]
                break
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    if not isinstance(state, Mapping):
        raise ValueError(f"Could not locate a state dict in checkpoint of type {type(checkpoint)!r}.")
    return {str(k): v for k, v in state.items()}


def strip_wrapper_prefixes(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove ``module.`` / ``_orig_mod.`` wrapper prefixes (possibly nested)."""
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in _WRAPPER_PREFIXES:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        out[new_key] = value
    return out


def migrate_phase1_state_dict(
    state: Mapping[str, torch.Tensor],
    *,
    target_prefix: str = "phase1.",
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Map a legacy deterministic state dict onto the two-phase wrapper.

    Returns ``(migrated_state, renames)`` where ``renames`` documents every key
    whose name changed, so the migration is auditable.
    """
    stripped = strip_wrapper_prefixes(state)
    migrated: dict[str, torch.Tensor] = {}
    renames: dict[str, str] = {}
    for key, value in stripped.items():
        new_key = LEGACY_PHASE1_KEY_RENAMES.get(key, key)
        if new_key.startswith("refiner."):
            # Already a two-phase checkpoint: keep Phase-2 keys untouched.
            migrated[new_key] = value
            continue
        if not new_key.startswith(target_prefix):
            new_key = target_prefix + new_key
        migrated[new_key] = value
        if new_key != key:
            renames[key] = new_key
    return migrated, renames


def phase1_state_fingerprint(state: Mapping[str, torch.Tensor]) -> str:
    """Stable SHA-256 identity of a Phase-1 tensor collection.

    Hashes ``(key, dtype, shape, raw bytes)`` for every tensor in sorted key
    order, so it is independent of dict ordering and of the wrapper prefix.
    """
    digest = hashlib.sha256()
    normalized = strip_wrapper_prefixes(state)
    for key in sorted(normalized):
        value = normalized[key]
        if not torch.is_tensor(value):
            continue
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _canonical_value(value: Any) -> Any:
    """Convert config metadata to a stable, JSON-serializable representation."""
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "item"):
        item = value.item()
        if item is None or isinstance(item, (bool, int, float, str)):
            return item
    raise TypeError(
        "Refinement checkpoint contract values must be JSON-compatible; "
        f"got {type(value)!r}."
    )


def _config_section(section: Any, *, ignored: Iterable[str] = ()) -> dict[str, Any]:
    ignored_keys = {str(key) for key in ignored}
    if isinstance(section, Mapping):
        keys = section.keys()
        getter = section.__getitem__
    elif callable(getattr(section, "to_dict", None)):
        return _config_section(section.to_dict(), ignored=ignored_keys)
    else:
        keys = [key for key in getattr(section, "_KEYS", ()) if hasattr(section, key)]
        getter = lambda key: getattr(section, key)  # noqa: E731 - compact adapter
    return {
        str(key): _canonical_value(getter(key))
        for key in keys
        if str(key) not in ignored_keys
    }


def _residual_normalization_contract(config: Any) -> dict[str, Any]:
    section = getattr(config, "residual_normalization", None)
    if section is None:
        raise RuntimeError(
            "Active refinement config has no residual_normalization section; "
            "refusing to assume normalization semantics."
        )
    contract = {
        "method": str(getattr(section, "method")),
        "epsilon": float(getattr(section, "epsilon")),
        "minimum_scale": float(getattr(section, "minimum_scale")),
        "require_fitted": bool(getattr(section, "require_fitted")),
    }
    # Omitted at the (backward-compatible) default so existing schema-2
    # checkpoints fitted before this option existed keep an identical
    # contract fingerprint; any non-default opt-in remains fingerprinted and
    # must match on load.
    if bool(getattr(section, "signed_log_nonnegative_channels", False)):
        contract["signed_log_nonnegative_channels"] = True
        contract["signed_log_scale"] = float(getattr(section, "signed_log_scale"))
    if bool(getattr(section, "signed_sqrt_nonnegative_channels", False)):
        contract["signed_sqrt_nonnegative_channels"] = True
        contract["signed_sqrt_scale"] = float(getattr(section, "signed_sqrt_scale"))
    return contract


def _build_scientific_contract(config: Any) -> dict[str, Any]:
    refinement_type = str(getattr(config, "type", "")).strip()
    if refinement_type.startswith("diffusion_"):
        family = "diffusion"
        process = _config_section(getattr(config, "diffusion"))
        if not process.get("velocity_loss_channels"):
            process.pop("velocity_loss_channels", None)
        # Existing checkpoints predate the optional numerical integrator.
        if process.get("solver", "ddim") == "ddim":
            process.pop("solver", None)
    elif refinement_type.startswith("flow_matching_"):
        family = "flow_matching"
        process = _config_section(getattr(config, "flow_matching"))
        # Schema-2 checkpoints written before mean-path supervision existed do
        # not contain this key.  An explicit zero is exactly the old objective,
        # so omit it from the scientific contract for backward compatibility;
        # any positive opt-in remains fingerprinted and must match on load.
        if not process.get("mean_path_channel_weights"):
            process.pop("mean_path_channel_weights", None)
        if float(process.get("mean_path_loss_weight", 0.0)) == 0.0:
            process.pop("mean_path_loss_weight", None)
    else:
        raise RuntimeError(
            "A refinement checkpoint requires an active diffusion or flow-matching "
            f"model, got refinement type {refinement_type!r}."
        )

    if refinement_type.endswith("_transformer"):
        architecture = "transformer"
        backbone = _config_section(
            getattr(config, "transformer"), ignored=_NON_SCIENTIFIC_BACKBONE_KEYS
        )
    elif refinement_type.endswith("_unet"):
        architecture = "unet"
        backbone = _config_section(
            getattr(config, "unet"), ignored=_NON_SCIENTIFIC_BACKBONE_KEYS
        )
    else:
        raise RuntimeError(
            f"Could not determine refinement architecture from {refinement_type!r}."
        )

    conditioning = _config_section(getattr(config, "conditioning"))
    if not bool(conditioning.get("normalize_predictors", False)):
        conditioning.pop("normalize_predictors", None)

    # Default legacy geometry has the exact function represented by older
    # checkpoints, whose backbone metadata predates this optional setting.
    if backbone.get("spatial_alignment", "legacy") == "legacy":
        backbone.pop("spatial_alignment", None)

    return _canonical_value(
        {
            "contract_version": REFINEMENT_CONTRACT_VERSION,
            "refinement_type": refinement_type,
            "residual_contract": "physical_ground_truth_minus_phase1_v1",
            **({"process_preconditioning": {"version": "gaussian_source_endpoint_v1", "channels": list(config.process_preconditioning_channels)}}
               if config.process_preconditioning_channels else {}),
            **({"unit_correction_gate_channels": list(config.unit_correction_gate_channels)} if config.unit_correction_gate_channels else {}),
            **({"process_boundary_balance": {"version": "half_global_half_seven_frame_bands_v1", "channels": list(config.process_boundary_balance_channels)}} if config.process_boundary_balance_channels else {}),
            **({"clean_boundary_loss": {"version": "physical_huber_exponential_frame_distance_v1",
                "channels": list(config.clean_boundary_channels), "weight": config.clean_boundary_weight,
                "alpha": config.clean_boundary_alpha, "length": config.clean_boundary_length,
                "delta": config.clean_boundary_delta}} if config.clean_boundary_weight > 0 else {}),
            **({"native_only_channels": list(config.native_only_channels)} if config.native_only_channels else {}),
            "train_on_residual": bool(getattr(config, "train_on_residual")),
            "reconstruction_loss_weight": float(
                getattr(config, "reconstruction_loss_weight", 0.0)
            ),
            "multiscale_loss_weight": float(
                getattr(config, "multiscale_loss_weight", 0.0)
            ),
            "gradient_loss_weight": float(
                getattr(config, "gradient_loss_weight", 0.0)
            ),
            "mean_bias_loss_weight": float(
                getattr(config, "mean_bias_loss_weight", 0.0)
            ),
            **(
                {
                    "nonnegative_ensemble_strategy": str(
                        getattr(config, "nonnegative_ensemble_strategy")
                    )
                }
                if str(
                    getattr(config, "nonnegative_ensemble_strategy", "memberwise")
                )
                != "memberwise"
                else {}
            ),
            "loss": str(getattr(config, "loss")),
            "freeze_phase1": bool(getattr(config, "freeze_phase1")),
            "joint_finetuning": bool(getattr(config, "joint_finetuning")),
            "conditioning": conditioning,
            "residual_normalization": _residual_normalization_contract(config),
            "process": {"family": family, "config": process},
            "network": {"architecture": architecture, "config": backbone},
        }
    )


def _model_scientific_contract(model: torch.nn.Module) -> dict[str, Any]:
    config = getattr(model, "refinement_config", None)
    if config is None:
        raise RuntimeError(
            "Cannot validate a refinement checkpoint: model has no refinement_config."
        )
    return _build_scientific_contract(config)


def _target_space_contract(model: torch.nn.Module) -> dict[str, Any]:
    """Record channel semantics, including Phase-1 nonpersistent buffers.

    A weight fingerprint alone cannot detect a same-width variable reorder or
    a changed normalization transform when semantic buffers are nonpersistent.
    """
    phase1 = getattr(model, "phase1", None)
    if phase1 is None:
        raise RuntimeError("Refinement target-space provenance requires a Phase-1 model.")
    names = getattr(phase1, "output_var_names", None)
    if names is None:
        names = getattr(phase1, "predictands", None)
    scalers = {
        name: getattr(phase1, name)
        for name in ("output_scalers_mu", "output_scalers_sigma")
        if torch.is_tensor(getattr(phase1, name, None))
    }
    contract = {
        "channel_names": list(names) if names is not None else None,
        "channel_count": int(phase1.output_scalers_sigma.shape[1]),
        "phase1_output_scalers_fingerprint": _state_fingerprint(scalers),
        "phase1_output_scalers_shapes": {key: list(value.shape) for key, value in scalers.items()},
        "residual_sign": "target_minus_phase1",
        "residual_space": "physical",
        "residual_transform": model.refinement_config.residual_normalization.method,
    }
    if bool(getattr(model.refinement_config.conditioning, "normalize_predictors", False)):
        conditioning_scalers = {
            name: getattr(phase1, name)
            for name in (
                "input_scalers_mu", "input_scalers_sigma",
                "static_input_scalers_mu", "static_input_scalers_sigma",
                "static_output_scalers_mu", "static_output_scalers_sigma",
            )
            if torch.is_tensor(getattr(phase1, name, None))
        }
        contract["conditioning_normalization"] = {
            "scalers_fingerprint": _state_fingerprint(conditioning_scalers),
            "scaler_names": sorted(conditioning_scalers),
            "input_epsilon": float(getattr(phase1, "input_scalers_epsilon", 0.0)),
            "static_epsilon": float(getattr(phase1, "static_input_scalers_epsilon", 0.0)),
            "input_timestamps": int(getattr(phase1, "n_input_timestamps", 1)),
        }
    for name in (
        "predictand_scaling_method_codes",
        "predictand_nonneg_enabled_mask",
        "predictand_nonneg_method_codes",
    ):
        values = getattr(phase1, name, None)
        contract[name] = values.detach().cpu().reshape(-1).tolist() if values is not None else None
    return _canonical_value(contract)


def _validate_target_space_contract(model: torch.nn.Module, checkpoint: Mapping[str, Any]) -> None:
    recorded = checkpoint.get("target_space_contract")
    if recorded is None:
        warnings.warn(
            "Legacy refinement checkpoint has no target_space_contract: channel ordering, "
            "Phase-1 transforms and scaler identity cannot be independently verified. "
            "Use its original case configuration and verify the Phase-1 fingerprint; "
            "do not substitute variables or statistics.",
            RuntimeWarning,
            stacklevel=3,
        )
        return
    if not isinstance(recorded, Mapping):
        raise RuntimeError("Refinement target_space_contract must be a mapping.")
    if checkpoint.get("target_space_contract_fingerprint") != _contract_fingerprint(recorded):
        raise RuntimeError("Refinement target-space contract fingerprint mismatch.")
    differences = _contract_differences(_target_space_contract(model), recorded)
    if differences:
        raise RuntimeError(
            "Refinement target-space contract mismatch; weights were not loaded. "
            + "; ".join(differences[:12])
        )


def _contract_fingerprint(contract: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _canonical_value(contract), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _state_fingerprint(state: Mapping[str, torch.Tensor]) -> str:
    """Stable fingerprint for a named collection of persistent tensors."""
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if not torch.is_tensor(value):
            raise RuntimeError(
                f"Checkpoint state {key!r} is not a tensor ({type(value)!r})."
            )
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _contract_differences(expected: Any, actual: Any, path: str = "") -> list[str]:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        differences: list[str] = []
        expected_keys, actual_keys = set(expected), set(actual)
        for key in sorted(expected_keys - actual_keys):
            differences.append(f"{path}{key}: missing from checkpoint")
        for key in sorted(actual_keys - expected_keys):
            differences.append(f"{path}{key}: unexpected checkpoint setting")
        for key in sorted(expected_keys & actual_keys):
            differences.extend(
                _contract_differences(expected[key], actual[key], f"{path}{key}.")
            )
        return differences
    if expected != actual:
        leaf = path[:-1] if path.endswith(".") else path
        return [f"{leaf}: model={expected!r}, checkpoint={actual!r}"]
    return []


def _validate_normalizer_state(
    checkpoint: Mapping[str, Any],
    incoming: Mapping[str, torch.Tensor],
    contract: Mapping[str, Any],
) -> None:
    normalizer_state = {
        key: value
        for key, value in incoming.items()
        if key.startswith(_RESIDUAL_NORMALIZER_PREFIX)
    }
    normalization = contract["residual_normalization"]
    method = str(normalization["method"])

    recorded_keys = checkpoint.get("residual_normalizer_state_keys")
    if not isinstance(recorded_keys, (list, tuple)):
        raise RuntimeError(
            "Refinement checkpoint does not record residual_normalizer_state_keys."
        )
    actual_keys = sorted(normalizer_state)
    if sorted(str(key) for key in recorded_keys) != actual_keys:
        raise RuntimeError(
            "Residual-normalizer state keys do not match checkpoint metadata: "
            f"metadata={sorted(str(key) for key in recorded_keys)!r}, "
            f"state={actual_keys!r}."
        )

    recorded_fingerprint = checkpoint.get("residual_normalizer_state_fingerprint")
    actual_fingerprint = _state_fingerprint(normalizer_state)
    if not isinstance(recorded_fingerprint, str) or recorded_fingerprint != actual_fingerprint:
        raise RuntimeError(
            "Residual-normalizer state fingerprint mismatch; refusing potentially "
            "corrupt or substituted normalization statistics."
        )

    if method == "identity":
        if normalizer_state:
            raise RuntimeError(
                "Identity residual normalization must not carry persistent statistics."
            )
        return
    if method != "standardize":
        raise RuntimeError(f"Unsupported residual normalization method {method!r}.")

    required = {
        f"{_RESIDUAL_NORMALIZER_PREFIX}{name}" for name in ("mean", "scale", "count", "fitted")
    }
    missing = sorted(required - set(normalizer_state))
    if missing:
        raise RuntimeError(
            f"Standardized residual checkpoint is missing statistics: {missing}."
        )
    mean = normalizer_state[f"{_RESIDUAL_NORMALIZER_PREFIX}mean"]
    scale = normalizer_state[f"{_RESIDUAL_NORMALIZER_PREFIX}scale"]
    count = normalizer_state[f"{_RESIDUAL_NORMALIZER_PREFIX}count"]
    fitted = normalizer_state[f"{_RESIDUAL_NORMALIZER_PREFIX}fitted"]
    minimum_scale = float(normalization["minimum_scale"])
    for key, value in normalizer_state.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise RuntimeError(
                f"Residual-normalizer statistic {key!r} contains NaN or infinite values."
            )
    if not torch.isfinite(mean).all():
        raise RuntimeError("Residual-normalizer mean contains NaN or infinite values.")
    if not torch.isfinite(scale).all() or bool(torch.any(scale < minimum_scale)):
        raise RuntimeError(
            "Residual-normalizer scale is non-finite or below configured minimum_scale."
        )
    if not torch.isfinite(count).all() or bool(torch.any(count <= 0)):
        raise RuntimeError("Residual-normalizer count must be finite and positive.")
    m2 = normalizer_state.get(f"{_RESIDUAL_NORMALIZER_PREFIX}_m2")
    if m2 is not None and bool(torch.any(m2 < 0)):
        raise RuntimeError("Residual-normalizer accumulated variance must be non-negative.")
    if bool(normalization["require_fitted"]) and not bool(torch.as_tensor(fitted).bool().all()):
        raise RuntimeError(
            "Residual-normalizer checkpoint is not fitted but require_fitted=true."
        )


def _validate_refinement_checkpoint_contract(
    model: torch.nn.Module,
    checkpoint: Any,
    incoming: Mapping[str, torch.Tensor],
) -> None:
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError(
            "Refinement weights must be loaded from a versioned checkpoint payload; "
            "bare state dicts are not scientifically verifiable."
        )

    kind = checkpoint.get("checkpoint_kind")
    if kind not in {CHECKPOINT_KIND_REFINEMENT, CHECKPOINT_KIND_COMBINED}:
        raise RuntimeError(
            "Refinement loader requires checkpoint_kind='refinement' or 'combined', "
            f"got {kind!r}."
        )
    schema = checkpoint.get("checkpoint_schema_version")
    if isinstance(schema, bool) or not isinstance(schema, int):
        raise RuntimeError(
            f"Refinement checkpoint has invalid schema version {schema!r}."
        )
    if schema != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported refinement checkpoint schema {schema}; the physical-residual "
            f"contract requires schema {CHECKPOINT_SCHEMA_VERSION}. Schema-1 Phase-2 "
            "weights must be retrained, not reinterpreted."
        )

    expected = _model_scientific_contract(model)
    expected_type = expected["refinement_type"]
    checkpoint_type = checkpoint.get("refinement_type")
    if not isinstance(checkpoint_type, str) or not checkpoint_type:
        raise RuntimeError("Refinement checkpoint does not record refinement_type.")
    if checkpoint_type != expected_type:
        raise RuntimeError(
            "Refinement checkpoint type mismatch: "
            f"model={expected_type!r}, checkpoint={checkpoint_type!r}."
        )

    raw_contract = checkpoint.get("refinement_contract")
    if not isinstance(raw_contract, Mapping):
        raise RuntimeError("Refinement checkpoint does not record refinement_contract.")
    actual = _canonical_value(raw_contract)
    if actual.get("contract_version") != REFINEMENT_CONTRACT_VERSION:
        raise RuntimeError(
            "Unsupported refinement scientific-contract version "
            f"{actual.get('contract_version')!r}."
        )
    recorded_fingerprint = checkpoint.get("refinement_contract_fingerprint")
    actual_fingerprint = _contract_fingerprint(actual)
    if not isinstance(recorded_fingerprint, str) or recorded_fingerprint != actual_fingerprint:
        raise RuntimeError(
            "Refinement scientific-contract fingerprint mismatch; refusing a "
            "tampered or corrupt checkpoint."
        )

    if actual.get("refinement_type") != checkpoint_type:
        raise RuntimeError(
            "Checkpoint refinement_type disagrees with its scientific contract: "
            f"metadata={checkpoint_type!r}, contract={actual.get('refinement_type')!r}."
        )
    differences = _contract_differences(expected, actual)
    if differences:
        raise RuntimeError(
            "Refinement scientific/config contract mismatch; weights were not loaded. "
            + "; ".join(differences[:12])
        )

    _validate_normalizer_state(checkpoint, incoming, actual)
    _validate_target_space_contract(model, checkpoint)


def load_phase1_state_dict(
    model: torch.nn.Module,
    checkpoint: Any,
    *,
    target_prefix: str = "phase1.",
    allow_missing_refinement: bool = True,
    skip_keys: Iterable[str] = (),
) -> StateDictReport:
    """Load a deterministic Phase-1 checkpoint into a two-phase wrapper.

    Missing keys are tolerated only when they belong to the newly introduced
    Phase-2 module (``refiner.*``) and ``allow_missing_refinement`` is set. Any
    other missing key, any unexpected key and any shape mismatch raises.

    ``skip_keys`` may name Phase-1 buffers/parameters that are intentionally
    rebuilt from the current configuration (e.g. normalization scalers that are
    coordinate-dependent). Skipped keys are reported, never silently dropped.
    """
    if isinstance(checkpoint, Mapping) and "checkpoint_kind" in checkpoint:
        kind = checkpoint.get("checkpoint_kind")
        if kind != CHECKPOINT_KIND_PHASE1:
            raise RuntimeError(
                "Phase-1 loader requires checkpoint_kind='phase1'; it will not "
                f"silently extract Phase 1 from {kind!r} weights."
            )
        schema = checkpoint.get("checkpoint_schema_version")
        if schema not in _SUPPORTED_PHASE1_CHECKPOINT_SCHEMAS:
            raise RuntimeError(
                f"Unsupported Phase-1 checkpoint schema {schema!r}; supported "
                f"versions are {sorted(_SUPPORTED_PHASE1_CHECKPOINT_SCHEMAS)}."
            )

    raw = extract_model_state(checkpoint)
    stripped = strip_wrapper_prefixes(raw)
    phase2_keys = sorted(
        key
        for key in stripped
        if any(key.startswith(prefix) for prefix in _REFINEMENT_STATE_PREFIXES)
    )
    if phase2_keys:
        raise RuntimeError(
            "Phase-1 loader refuses checkpoint keys belonging to Phase 2, "
            f"e.g. {phase2_keys[:8]}. Use the refinement/combined loader explicitly."
        )
    migrated, renames = migrate_phase1_state_dict(stripped, target_prefix=target_prefix)

    skip = {str(k) for k in skip_keys}
    if skip:
        migrated = {
            k: v
            for k, v in migrated.items()
            if not any(part in k for part in skip)
        }

    model_state = model.state_dict()
    report = StateDictReport(renamed=renames)

    for key, value in migrated.items():
        target = model_state.get(key)
        if target is None:
            report.unexpected.append(key)
            continue
        if torch.is_tensor(value) and tuple(target.shape) != tuple(value.shape):
            report.shape_mismatched.append((key, tuple(target.shape), tuple(value.shape)))
            continue
        model_state[key] = value
        report.loaded += 1

    provided = set(migrated)
    for key in model_state:
        if key not in provided:
            report.missing.append(key)

    if report.shape_mismatched:
        details = ", ".join(f"{k}: model{m} vs ckpt{c}" for k, m, c in report.shape_mismatched[:8])
        raise RuntimeError(
            f"Refusing to load checkpoint: {len(report.shape_mismatched)} shape mismatch(es). {details}"
        )
    if report.unexpected:
        raise RuntimeError(
            f"Refusing to load checkpoint: {len(report.unexpected)} unexpected key(s), "
            f"e.g. {sorted(report.unexpected)[:8]}"
        )

    unexplained = [
        key
        for key in report.missing
        if not any(key.startswith(prefix) for prefix in _REFINEMENT_STATE_PREFIXES)
    ]
    unexplained = [k for k in unexplained if not any(part in k for part in skip)]
    if unexplained:
        raise RuntimeError(
            f"Refusing to load checkpoint: {len(unexplained)} missing Phase-1 key(s), "
            f"e.g. {sorted(unexplained)[:8]}"
        )
    if report.missing and not allow_missing_refinement:
        raise RuntimeError(
            f"Checkpoint is missing {len(report.missing)} key(s) and "
            "allow_missing_refinement is disabled."
        )

    # ``strict=False`` here is safe *because* every missing key was proven above
    # to belong to the newly introduced refinement module.
    model.load_state_dict(model_state, strict=not report.missing)
    return report


def load_refinement_state_dict(
    model: torch.nn.Module,
    checkpoint: Any,
    *,
    prefix: str = "refiner.",
) -> StateDictReport:
    """Load Phase-2 weights without touching Phase-1 weights.

    Refiner and residual-normalizer keys are applied together; the Phase-1
    sub-module is left exactly as it was. Metadata and the complete scientific
    contract are validated before any tensor is applied.
    """
    raw = extract_model_state(checkpoint)
    stripped = strip_wrapper_prefixes(raw)
    state_prefixes = (prefix, _RESIDUAL_NORMALIZER_PREFIX)
    incoming = {
        key: value
        for key, value in stripped.items()
        if any(key.startswith(state_prefix) for state_prefix in state_prefixes)
    }
    if not incoming:
        # Bare refiner state dict (no wrapper prefix).
        incoming = {prefix + k: v for k, v in stripped.items()}

    _validate_refinement_checkpoint_contract(model, checkpoint, incoming)

    model_state = model.state_dict()
    report = StateDictReport()
    phase1_before = {k: v for k, v in model_state.items() if k.startswith("phase1.")}

    for key, value in incoming.items():
        target = model_state.get(key)
        if target is None:
            report.unexpected.append(key)
            continue
        if torch.is_tensor(value) and tuple(target.shape) != tuple(value.shape):
            report.shape_mismatched.append((key, tuple(target.shape), tuple(value.shape)))
            continue
        model_state[key] = value
        report.loaded += 1

    refinement_keys = {
        key
        for key in model_state
        if any(key.startswith(state_prefix) for state_prefix in state_prefixes)
    }
    report.missing = sorted(refinement_keys - set(incoming))

    if report.shape_mismatched:
        raise RuntimeError(
            f"Refusing to load refinement checkpoint: shape mismatch(es) "
            f"{report.shape_mismatched[:8]}"
        )
    if report.unexpected:
        raise RuntimeError(
            f"Refusing to load refinement checkpoint: unexpected key(s) "
            f"{sorted(report.unexpected)[:8]}"
        )
    if report.missing:
        raise RuntimeError(
            f"Refusing to load refinement checkpoint: missing key(s) "
            f"{report.missing[:8]}"
        )

    model.load_state_dict(model_state, strict=True)
    after = {k: v for k, v in model.state_dict().items() if k.startswith("phase1.")}
    for key, before in phase1_before.items():
        if torch.is_tensor(before) and not torch.equal(before, after[key]):
            raise RuntimeError(
                f"Loading a refinement checkpoint changed Phase-1 weight {key!r}. "
                "This must never happen."
            )
    return report


def save_checkpoint_atomic(payload: Mapping[str, Any], path: str | os.PathLike, *, atomic: bool = True) -> str:
    """Persist a checkpoint, optionally via a same-directory temporary file.

    Atomic writes prevent a crash mid-save from leaving a truncated checkpoint
    that would break resume.
    """
    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    if not atomic:
        torch.save(payload, path)
        return path
    fd, tmp = tempfile.mkstemp(prefix=".ckpt-", suffix=".tmp", dir=directory)
    os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def build_refinement_checkpoint(
    model: torch.nn.Module,
    *,
    kind: str = CHECKPOINT_KIND_REFINEMENT,
    phase1_checkpoint: str | None = None,
    phase1_fingerprint: str | None = None,
    resolved_config: Mapping[str, Any] | None = None,
    epoch: int = 0,
    global_step: int = 0,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a checkpoint payload that records everything needed for resume.

    ``kind`` distinguishes ``phase1`` / ``refinement`` / ``combined`` payloads. A
    ``refinement`` checkpoint deliberately excludes the (unchanged, frozen)
    Phase-1 weights and instead records the Phase-1 checkpoint path and
    fingerprint, so it stays small while remaining verifiable.
    """
    if kind not in {CHECKPOINT_KIND_PHASE1, CHECKPOINT_KIND_REFINEMENT, CHECKPOINT_KIND_COMBINED}:
        raise ValueError(f"Unsupported checkpoint kind {kind!r}")

    full_state = model.state_dict()
    if kind == CHECKPOINT_KIND_REFINEMENT:
        model_state = {
            key: value
            for key, value in full_state.items()
            if any(key.startswith(prefix) for prefix in _REFINEMENT_STATE_PREFIXES)
        }
    elif kind == CHECKPOINT_KIND_PHASE1:
        model_state = {k: v for k, v in full_state.items() if k.startswith("phase1.")}
    else:
        model_state = dict(full_state)

    payload: dict[str, Any] = {
        "checkpoint_kind": kind,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model": model_state,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "phase1_checkpoint": phase1_checkpoint,
        "phase1_fingerprint": phase1_fingerprint,
        "resolved_config": dict(resolved_config or {}),
        "rng_state": {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    if kind in {CHECKPOINT_KIND_REFINEMENT, CHECKPOINT_KIND_COMBINED}:
        contract = _model_scientific_contract(model)
        target_contract = _target_space_contract(model)
        normalizer_state = {
            key: value
            for key, value in model_state.items()
            if key.startswith(_RESIDUAL_NORMALIZER_PREFIX)
        }
        payload.update(
            {
                "refinement_type": contract["refinement_type"],
                "refinement_contract": contract,
                "refinement_contract_fingerprint": _contract_fingerprint(contract),
                "target_space_contract": target_contract,
                "target_space_contract_fingerprint": _contract_fingerprint(target_contract),
                "residual_normalizer_state_keys": sorted(normalizer_state),
                "residual_normalizer_state_fingerprint": _state_fingerprint(normalizer_state),
            }
        )
        _validate_normalizer_state(payload, model_state, contract)
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        sched_state = scheduler.state_dict()
        payload["scheduler"] = {k: v for k, v in sched_state.items() if k != "anneal_func"}
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if extra:
        extra_values = dict(extra)
        reserved = set(payload) & set(extra_values)
        conflicts = [key for key in sorted(reserved) if extra_values[key] != payload[key]]
        if conflicts:
            raise ValueError(
                "Checkpoint extra metadata cannot override reserved values: "
                f"{conflicts}."
            )
        payload.update(extra_values)
    return payload


def validate_phase1_reference(
    checkpoint: Mapping[str, Any],
    phase1_state: Mapping[str, torch.Tensor],
    *,
    strict: bool = True,
) -> bool:
    """Check that a refinement checkpoint matches the loaded Phase-1 weights."""
    expected = checkpoint.get("phase1_fingerprint")
    if not expected:
        if strict:
            raise RuntimeError(
                "Refinement checkpoint does not record a Phase-1 fingerprint; refusing "
                "to assume compatibility. Re-save it with build_refinement_checkpoint()."
            )
        return False
    actual = phase1_state_fingerprint(phase1_state)
    if actual != expected:
        message = (
            "Phase-1 identity mismatch: the refinement checkpoint was trained against "
            f"Phase-1 fingerprint {expected[:16]}... but the loaded Phase 1 is "
            f"{actual[:16]}...  (referenced checkpoint: "
            f"{checkpoint.get('phase1_checkpoint')!r})"
        )
        if strict:
            raise RuntimeError(message)
        return False
    return True
