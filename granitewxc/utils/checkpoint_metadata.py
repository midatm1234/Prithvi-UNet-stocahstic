"""Checkpoint provenance and compatibility checks for residual diffusion.

Residual diffusion checkpoints are only meaningful together with the exact
baseline definition, target normalization, score parameterization, and
sampler configuration used during training.  This module records those inputs
as plain, immutable checkpoint metadata and provides the inference-side guard
that rejects a semantically incompatible config before model construction.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


CHECKPOINT_METADATA_SCHEMA = "granitewxc.cordex.checkpoint"
CHECKPOINT_METADATA_VERSION = 2
REPO_ROOT = Path(__file__).resolve().parents[2]

_FILE_HASH_CACHE: dict[tuple[str, int, int], str] = {}
_DIFFUSION_ALIASES = {
    "diffusion",
    "diffusion_head",
    "score",
    "score_sde",
    "sde",
}
_MISSING = object()


class CheckpointCompatibilityError(RuntimeError):
    """Raised before inference when checkpoint and residual config differ."""


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _as_plain(value: Any) -> Any:
    """Convert config containers into stable JSON-compatible primitives."""

    if isinstance(value, Mapping):
        return {str(key): _as_plain(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_as_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _as_plain(value.to_dict())
    if hasattr(value, "__dict__"):
        return _as_plain(vars(value))
    return str(value)


def _sha256_file(path: Path) -> str:
    resolved = path.expanduser().resolve(strict=True)
    stat = resolved.stat()
    cache_key = (str(resolved), int(stat.st_size), int(stat.st_mtime_ns))
    cached = _FILE_HASH_CACHE.get(cache_key)
    if cached is not None:
        return cached

    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _FILE_HASH_CACHE[cache_key] = value
    return value


def _file_record(path_value: Any, *, config_field: str) -> dict[str, Any] | None:
    if path_value in (None, ""):
        return None
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        # Match the training/inference path resolver. Resolving relative to the
        # process CWD produced false "missing initializer" records even though
        # training loaded the file from the repository root.
        path = REPO_ROOT / path
    path = path.resolve(strict=False)
    record: dict[str, Any] = {
        "config_field": config_field,
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": None,
        "sha256": None,
    }
    if path.is_file():
        record["size_bytes"] = int(path.stat().st_size)
        record["sha256"] = _sha256_file(path)
    return record


def _scalar_files(config: Any) -> dict[str, Any]:
    model_cfg = _get(config, "model")
    data_cfg = _get(config, "data")
    data_scalers = _get(data_cfg, "scalers", {})
    fields = {
        "inputs_mean": ("input_mu", "inputs_mean"),
        "inputs_std": ("input_sigma", "inputs_std"),
        "targets_mean": ("target_mu", "targets_mean"),
        "targets_std": ("target_sigma", "targets_std"),
    }
    records: dict[str, Any] = {}
    for semantic_name, (model_name, data_name) in fields.items():
        value = _get(model_cfg, model_name)
        source = f"model.{model_name}"
        if value in (None, ""):
            value = _get(data_scalers, data_name)
            source = f"data.scalers.{data_name}"
        records[semantic_name] = _file_record(value, config_field=source)
    target_record = records.get("targets_std")
    if isinstance(target_record, Mapping) and target_record.get("path"):
        metadata_path = Path(str(target_record["path"])).parent / "metadata.json"
        records["metadata"] = _file_record(
            metadata_path,
            config_field="sibling of target scalar files",
        )
    return records


def _initializer_files(config: Any) -> dict[str, Any]:
    model_cfg = _get(config, "model")
    candidates = (
        ("path_model_weights", _get(config, "path_model_weights")),
        ("load_model", _get(config, "load_model")),
        ("model.path_model_weights", _get(model_cfg, "path_model_weights")),
        ("model.unet_checkpoint_path", _get(model_cfg, "unet_checkpoint_path")),
        ("model.baseline_checkpoint_path", _get(model_cfg, "baseline_checkpoint_path")),
    )
    records: dict[str, Any] = {}
    seen_paths: set[str] = set()
    for field, value in candidates:
        record = _file_record(value, config_field=field)
        if record is None or record["path"] in seen_paths:
            continue
        seen_paths.add(record["path"])
        records[field] = record
    return records


def _head_type(config: Any, model: Any = None) -> str:
    model_cfg = _get(config, "model")
    value = _get(model_cfg, "head_type", _get(model_cfg, "decoder_type"))
    if value is None and model is not None:
        value = getattr(model, "head_type", None)
    if value is None and bool(getattr(model, "diffusion_enabled", False)):
        value = "diffusion"
    normalized = str(value or "deterministic").strip().lower()
    return "diffusion" if normalized in _DIFFUSION_ALIASES else normalized


def _resolved_diffusion_config(config: Any) -> dict[str, Any]:
    # Use the same resolver as the model so omitted YAML fields are recorded as
    # their actual runtime defaults instead of ambiguous null values.
    from granitewxc.decoders.diffusion_head import DiffusionHeadConfig

    resolved = DiffusionHeadConfig.from_config(config)
    fields = (
        "sde",
        "beta_min",
        "beta_max",
        "sigma_min",
        "sigma_max",
        "num_scales",
        "continuous",
        "likelihood_weighting",
        "reduce_mean",
        "eps",
        "noise_conditioning_scale",
        "prediction_type",
        "clean_x0_reconstruction_weight",
        "clean_x0_inverse_snr_cap",
        "sampling_method",
        "predictor",
        "corrector",
        "snr",
        "n_corrector_steps",
        "num_sampling_steps",
        "probability_flow",
        "denoise",
        "sampling_eps",
        "eta",
        "projected_cond_channels",
        "residual_magnitude_guard_multiple",
        "residual_guard_min_count",
        "base_channels",
        "channel_multipliers",
        "num_res_blocks",
        "time_embed_dim",
        "dropout",
        "fourier_scale",
        "padding_mode",
        "zero_init_output",
        "residual_application_scale",
    )
    payload = {field: _as_plain(getattr(resolved, field)) for field in fields}
    if resolved.residual_mean_enabled:
        payload["residual_mean_enabled"] = True
        payload["residual_mean_channels"] = int(resolved.residual_mean_channels)
    return payload


def residual_diffusion_enabled(config: Any) -> bool:
    model_cfg = _get(config, "model")
    diffusion_cfg = _get(model_cfg, "diffusion", {})
    return _head_type(config) == "diffusion" and bool(
        _get(diffusion_cfg, "residual_diffusion", _get(model_cfg, "residual_diffusion", False))
    )


def _loss_weights(config: Any) -> dict[str, float]:
    loss_cfg = _get(config, "loss", {})
    diffusion_cfg = _get(loss_cfg, "diffusion", {})
    return {
        "deterministic": float(_get(loss_cfg, "deterministic_weight", 1.0)),
        "diffusion": float(_get(diffusion_cfg, "weight", 1.0)),
    }


def _baseline_semantics(config: Any) -> dict[str, Any]:
    from granitewxc.utils.predictands import _coerce_mapping as _cm, _parse_predictand_spec

    model_cfg = _get(config, "model")
    top = _as_plain(config)
    precipitation = {
        key: value
        for key, value in top.items()
        if key == "precip_model" or key.startswith("precip_")
    }
    bernoulli_gamma = _get(model_cfg, "bernoulli_gamma")
    if bernoulli_gamma is not None:
        precipitation["model.bernoulli_gamma"] = _as_plain(bernoulli_gamma)

    # Canonicalize predictands through the same spec parser used at training time so
    # that the snapshot always matches the PredictandSpec.to_dict() form stored in the
    # checkpoint — regardless of whether build_predictand_specs has already been called.
    # Without this, raw-YAML configs (inference path) have a separate "normalization"
    # sub-block and un-aliased method names (e.g. "standardize") that differ from the
    # canonicalized "scaling"-only form produced after build_predictand_specs mutates
    # config.predictands during training.
    raw_predictands = _cm(_get(config, "predictands", {}))
    canonical: dict[str, Any] = {}
    for var_name, raw_cfg in raw_predictands.items():
        try:
            spec = _parse_predictand_spec(var_name, _cm(raw_cfg))
            canonical[var_name] = spec.to_dict()
        except Exception:
            canonical[var_name] = _as_plain(raw_cfg)

    return {
        "predictands": _as_plain(canonical),
        "precipitation": precipitation,
        "canonical_output_units": {
            name: ("mm/day" if name == "pr" else "K" if name == "tasmax" else None)
            for name in canonical
        },
    }


def _spatial_processing(config: Any) -> dict[str, Any]:
    """Record geometry choices that can alter edge and grid behavior."""

    model_cfg = _get(config, "model", {})
    data_cfg = _get(config, "data", {})
    inference_cfg = _get(config, "inference", {})
    boundary = _get(inference_cfg, "boundary_mitigation", {})
    return {
        "input_size": [
            _get(data_cfg, "input_size_lat"),
            _get(data_cfg, "input_size_lon"),
        ],
        "target_size": [
            _get(data_cfg, "target_size_lat"),
            _get(data_cfg, "target_size_lon"),
        ],
        "training_crop_size": [
            _get(data_cfg, "train_crop_size_lat"),
            _get(data_cfg, "train_crop_size_lon"),
        ],
        "downsample_factor": _get(data_cfg, "downsample_factor"),
        "patch_size": _as_plain(_get(model_cfg, "downscaling_patch_size")),
        "encoder_decoder_upsampling_mode": _get(
            model_cfg, "encoder_decoder_upsampling_mode"
        ),
        "decoder_upsampling_mode": _get(model_cfg, "decoder_upsampling_mode"),
        "output_scaler_resize_mode": _get(model_cfg, "output_scaler_resize_mode"),
        "output_scaler_align_corners": _get(
            model_cfg, "output_scaler_align_corners"
        ),
        "unet_upsample_scales": _as_plain(_get(model_cfg, "unet_upsample_scales")),
        "unet_decoder_kernel_size": _as_plain(
            _get(model_cfg, "unet_decoder_kernel_size")
        ),
        "inference_boundary": {
            "enabled": bool(_get(boundary, "enabled", False)),
            "force_full_frame": bool(
                _get(boundary, "force_full_frame", _get(inference_cfg, "force_full_frame", False))
            ),
            "tile_size": _as_plain(
                _get(boundary, "tile_size", _get(inference_cfg, "inference_tile_size"))
            ),
            "overlap": _as_plain(
                _get(boundary, "overlap", _get(inference_cfg, "inference_overlap"))
            ),
            "halo": _as_plain(
                _get(boundary, "halo", _get(inference_cfg, "inference_halo", [0, 0]))
            ),
            "tile_origin": _as_plain(
                _get(
                    boundary,
                    "tile_origin",
                    _get(inference_cfg, "inference_tile_origin", [0, 0]),
                )
            ),
            "blend_window": _get(
                boundary,
                "blend_window",
                _get(boundary, "blend_mode", _get(inference_cfg, "inference_blend_window", "hann")),
            ),
        },
    }


def _data_identity(config: Any) -> dict[str, Any]:
    """Record dataset/domain paths and geometry without hashing large files."""
    data_cfg = _get(config, "data", {})

    def paths(name: str) -> list[str]:
        values = _get(data_cfg, name, []) or []
        if isinstance(values, (str, Path)):
            values = [values]
        return [str(value) for value in values]

    return {
        "validation_enabled": bool(_get(config, "validation_enabled", True)),
        "training_predictor_paths": paths("training_predictor_paths"),
        "training_target_paths": paths("training_target_paths"),
        "validation_predictor_paths": paths("validation_predictor_paths"),
        "validation_target_paths": paths("validation_target_paths"),
        "test_predictor_paths": paths("test_predictor_paths"),
        "test_target_paths": paths("test_target_paths"),
        "use_static": bool(_get(data_cfg, "use_static", False)),
        "static_path": str(_get(data_cfg, "static_path", "") or ""),
        "target_size_lat": _get(data_cfg, "target_size_lat"),
        "target_size_lon": _get(data_cfg, "target_size_lon"),
        "train_crop_size_lat": _get(data_cfg, "train_crop_size_lat"),
        "train_crop_size_lon": _get(data_cfg, "train_crop_size_lon"),
        "crop_factor": _get(data_cfg, "crop_factor"),
        "validation_holdout_fraction": _get(
            data_cfg, "validation_holdout_fraction"
        ),
        "validation_holdout_strategy": _get(
            data_cfg, "validation_holdout_strategy"
        ),
        "train_random_crop_offset": _as_plain(
            _get(data_cfg, "train_random_crop_offset")
        ),
        "input_surface_vars": list(
            _get(data_cfg, "input_surface_vars", []) or []
        ),
        "input_static_surface_vars": list(
            _get(data_cfg, "input_static_surface_vars", []) or []
        ),
        "output_vars": list(_get(data_cfg, "output_vars", []) or []),
    }


def build_checkpoint_compatibility(config: Any, model: Any = None) -> dict[str, Any]:
    """Return the complete recorded compatibility and provenance contract."""

    residual_enabled = residual_diffusion_enabled(config)
    resolved_diffusion = (
        _resolved_diffusion_config(config)
        if _head_type(config, model) == "diffusion"
        else None
    )
    # Alpha is a runtime deployment/calibration knob, not a learned score-model
    # semantic. Keep it in checkpoint provenance (see ``deployment_defaults``)
    # but out of the strict compatibility fingerprint so held-out evaluation can
    # explicitly test a nonzero correction without fabricating a new checkpoint.
    if resolved_diffusion is not None:
        resolved_diffusion.pop("residual_application_scale", None)
    residual_contract = {
        "enabled": residual_enabled,
        "definition": (
            "target_normalized - stop_gradient(deterministic_baseline_normalized)"
            if residual_enabled
            else None
        ),
        "sign": "target_minus_baseline" if residual_enabled else None,
        "space": "target_normalized_space" if residual_enabled else None,
        "reconstruction": (
            "deterministic_baseline_normalized + residual_application_scale * sampled_residual_normalized"
            if residual_enabled
            else None
        ),
        "residual_application_scale_policy": (
            "runtime_configured_in_[0,1]" if residual_enabled else None
        ),
        "score_conditioning_gradient": "stopped" if residual_enabled else None,
        "deterministic_baseline_training": (
            (
                "frozen"
                if bool(
                    _get(
                        _get(config, "training", {}),
                        "freeze_deterministic_baseline",
                        False,
                    )
                )
                else "jointly_optimized_by_yaml_loss"
            )
            if residual_enabled
            else None
        ),
    }
    if residual_enabled and bool(
        _get(
            _get(_get(config, "model"), "diffusion", {}),
            "residual_mean_enabled",
            False,
        )
    ):
        residual_contract["residual_mean_decomposition"] = (
            "supervised_conditional_mean_plus_diffused_innovation"
        )
    loss_cfg = _get(config, "loss", {})
    inference_cfg = _get(config, "inference", {})
    return {
        "case_name": str(_get(config, "case_name", "") or ""),
        "job_id": str(_get(config, "job_id", "") or ""),
        "head_type": _head_type(config, model),
        "output_variables": list(_get(_get(config, "data"), "output_vars", []) or []),
        "data_identity": _data_identity(config),
        "residual_contract": residual_contract,
        "diffusion": resolved_diffusion,
        "loss_weights": _loss_weights(config),
        "loss_configuration": _as_plain(loss_cfg),
        "baseline_semantics": _baseline_semantics(config),
        "spatial_processing": _spatial_processing(config),
        "normalization_scalars": _scalar_files(config),
        "initializer_files": _initializer_files(config),
        "inference_randomness": {
            "ensemble_size": int(_get(inference_cfg, "ensemble_size", 1)),
            "base_seed": int(_get(inference_cfg, "base_seed", _get(inference_cfg, "ensemble_seed", 42))),
        },
    }


def _portable_path_name(value: Any) -> str:
    """Return a location-independent file identifier for a recorded path."""

    if value in (None, ""):
        return ""
    # Checkpoints can move between Windows and POSIX hosts. Normalize both
    # separator conventions without asking the current host's Path class to
    # interpret the source host's absolute path syntax.
    return str(value).rstrip("/\\").replace("\\", "/").rsplit("/", 1)[-1]


def _strict_compatibility_projection(
    compatibility: Mapping[str, Any],
) -> dict[str, Any]:
    """Select architecture/training semantics for fail-closed comparison.

    The complete ``compatibility`` mapping is retained and separately hashed as
    provenance. Runtime placement/identity fields must not make a trained model
    unusable after moving a run directory or changing an evaluation seed.
    Dataset basenames remain in the strict contract as portable role identities;
    scalar contents remain protected by SHA-256 rather than absolute location.
    """

    full = _as_plain(compatibility)
    data_identity = dict(full.get("data_identity", {}) or {})
    for role in (
        "training_predictor_paths",
        "training_target_paths",
        "validation_predictor_paths",
        "validation_target_paths",
    ):
        values = data_identity.pop(role, []) or []
        data_identity[f"{role.removesuffix('_paths')}_files"] = [
            _portable_path_name(value) for value in values
        ]
    # Test scenarios are selected at inference time and are provenance only.
    data_identity.pop("test_predictor_paths", None)
    data_identity.pop("test_target_paths", None)
    static_path = data_identity.pop("static_path", "")
    data_identity["static_file"] = _portable_path_name(static_path)

    scalar_records: dict[str, Any] = {}
    for name, raw_record in (full.get("normalization_scalars", {}) or {}).items():
        if not isinstance(raw_record, Mapping):
            scalar_records[str(name)] = raw_record
            continue
        scalar_records[str(name)] = {
            key: raw_record.get(key)
            for key in ("exists", "size_bytes", "sha256")
        }

    diffusion = full.get("diffusion")
    if isinstance(diffusion, Mapping):
        diffusion = dict(diffusion)
        # Alpha has always been a deployment gate, including for older schema-v2
        # records that stored it inside the resolved diffusion mapping.
        diffusion.pop("residual_application_scale", None)

    return {
        "head_type": full.get("head_type"),
        "output_variables": full.get("output_variables"),
        "data_identity": data_identity,
        "residual_contract": full.get("residual_contract"),
        "diffusion": diffusion,
        "loss_weights": full.get("loss_weights"),
        "loss_configuration": full.get("loss_configuration"),
        "baseline_semantics": full.get("baseline_semantics"),
        "spatial_processing": full.get("spatial_processing"),
        "normalization_scalars": scalar_records,
    }


def build_checkpoint_strict_compatibility(
    config: Any, model: Any = None
) -> dict[str, Any]:
    """Return only immutable architecture and training semantics."""

    return _strict_compatibility_projection(
        build_checkpoint_compatibility(config, model)
    )


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(_as_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_metadata() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]

    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(repo_root), *args],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip() or None

    status = run("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status) if status is not None else None,
    }


def build_checkpoint_metadata(
    config: Any,
    model: Any = None,
    *,
    epoch: int | None = None,
    global_step: int | None = None,
) -> dict[str, Any]:
    """Build serializable provenance embedded in every new checkpoint."""

    compatibility = build_checkpoint_compatibility(config, model)
    strict_compatibility = _strict_compatibility_projection(compatibility)
    source_control = _git_metadata()
    deployment_defaults = {"residual_application_scale": None}
    if residual_diffusion_enabled(config):
        from granitewxc.decoders.diffusion_head import DiffusionHeadConfig

        deployment_defaults["residual_application_scale"] = float(
            DiffusionHeadConfig.from_config(config).residual_application_scale
        )

    metadata = {
        "schema": CHECKPOINT_METADATA_SCHEMA,
        "schema_version": CHECKPOINT_METADATA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "epoch_index": int(epoch) if epoch is not None else None,
        "completed_epochs": max(int(epoch) + 1, 0) if epoch is not None else None,
        "global_step": int(global_step) if global_step is not None else None,
        "git_commit": source_control["commit"],
        "git_branch": source_control["branch"],
        "source_control": source_control,
        "compatibility": compatibility,
        "strict_compatibility": strict_compatibility,
        "config_fingerprint_sha256": _fingerprint(strict_compatibility),
        "compatibility_provenance_sha256": _fingerprint(compatibility),
        # Recorded for reproducibility but intentionally not part of the strict
        # training-semantic fingerprint. Every inference artifact records the
        # actual runtime alpha separately.
        "deployment_defaults": deployment_defaults,
        # Backward-compatible identity fields used by head detection helpers.
        "head_type": compatibility["head_type"],
        "decoder_type": _get(_get(config, "model"), "decoder_type"),
        "diffusion_head": compatibility["head_type"] == "diffusion",
    }
    return metadata


def _diff(expected: Any, actual: Any, path: str = "compatibility") -> list[str]:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        differences: list[str] = []
        for key in sorted(set(expected) | set(actual), key=str):
            child = f"{path}.{key}"
            if key not in expected:
                differences.append(f"{child}: unexpected checkpoint value {actual[key]!r}")
            elif key not in actual:
                differences.append(f"{child}: missing from checkpoint (expected {expected[key]!r})")
            else:
                differences.extend(_diff(expected[key], actual[key], child))
        return differences
    if isinstance(expected, list) and isinstance(actual, list):
        if expected == actual:
            return []
        return [f"{path}: checkpoint={actual!r}, config={expected!r}"]
    if expected != actual:
        return [f"{path}: checkpoint={actual!r}, config={expected!r}"]
    return []


def validate_checkpoint_compatibility(checkpoint: Any, config: Any) -> None:
    """Hard-reject residual checkpoints/configs whose semantics do not match.

    Legacy deterministic and full-field diffusion checkpoints remain readable.
    As soon as either side declares residual diffusion, versioned compatibility
    metadata is mandatory and every strict architecture/training field must
    match the active immutable config snapshot. Runtime/provenance fields remain
    recorded and integrity-checked but do not block portable inference.
    """

    if not isinstance(checkpoint, Mapping):
        if residual_diffusion_enabled(config):
            raise CheckpointCompatibilityError(
                "Residual-diffusion inference requires a dictionary checkpoint with versioned metadata."
            )
        return

    metadata = checkpoint.get("metadata")
    checkpoint_residual = bool(
        _get(_get(_get(metadata, "compatibility", {}), "residual_contract", {}), "enabled", False)
    )
    config_residual = residual_diffusion_enabled(config)
    if not (checkpoint_residual or config_residual):
        return

    if not isinstance(metadata, Mapping):
        raise CheckpointCompatibilityError(
            "Residual-diffusion checkpoint is missing the required 'metadata' mapping; refusing legacy/ambiguous restoration."
        )
    if metadata.get("schema") != CHECKPOINT_METADATA_SCHEMA:
        raise CheckpointCompatibilityError(
            "Residual-diffusion checkpoint metadata schema is missing or unsupported: "
            f"{metadata.get('schema')!r}."
        )
    if metadata.get("schema_version") != CHECKPOINT_METADATA_VERSION:
        raise CheckpointCompatibilityError(
            "Residual-diffusion checkpoint metadata version mismatch: "
            f"checkpoint={metadata.get('schema_version')!r}, supported={CHECKPOINT_METADATA_VERSION}."
        )

    saved = metadata.get("compatibility")
    if not isinstance(saved, Mapping):
        raise CheckpointCompatibilityError("Residual-diffusion checkpoint metadata has no compatibility contract.")
    saved_strict = metadata.get("strict_compatibility")
    saved_fingerprint = metadata.get("config_fingerprint_sha256")
    if isinstance(saved_strict, Mapping):
        if saved_fingerprint != _fingerprint(saved_strict):
            raise CheckpointCompatibilityError(
                "Residual-diffusion checkpoint strict compatibility metadata "
                "failed its own SHA-256 fingerprint check."
            )
        provenance_fingerprint = metadata.get("compatibility_provenance_sha256")
        if provenance_fingerprint != _fingerprint(saved):
            raise CheckpointCompatibilityError(
                "Residual-diffusion checkpoint recorded compatibility provenance "
                "failed its own SHA-256 fingerprint check."
            )
        projected_saved = _strict_compatibility_projection(saved)
        projection_differences = _diff(
            projected_saved,
            saved_strict,
            path="strict_compatibility",
        )
        if projection_differences:
            raise CheckpointCompatibilityError(
                "Residual-diffusion checkpoint strict compatibility contract "
                "does not match its recorded provenance."
            )
    else:
        # Backward compatibility for schema-v2 checkpoints written before the
        # strict/provenance split: verify the historical full fingerprint, then
        # compare only its projected semantic contract.
        if saved_fingerprint != _fingerprint(saved):
            raise CheckpointCompatibilityError(
                "Residual-diffusion checkpoint compatibility metadata failed "
                "its own SHA-256 fingerprint check."
            )
        saved_strict = _strict_compatibility_projection(saved)

    expected = build_checkpoint_strict_compatibility(config)
    differences = _diff(expected, saved_strict, path="strict_compatibility")
    if differences:
        preview = "\n  - ".join(differences[:20])
        suffix = f"\n  ... and {len(differences) - 20} more" if len(differences) > 20 else ""
        raise CheckpointCompatibilityError(
            "Residual-diffusion checkpoint/config mismatch; refusing inference:\n  - "
            f"{preview}{suffix}"
        )


__all__ = [
    "CHECKPOINT_METADATA_SCHEMA",
    "CHECKPOINT_METADATA_VERSION",
    "CheckpointCompatibilityError",
    "build_checkpoint_compatibility",
    "build_checkpoint_strict_compatibility",
    "build_checkpoint_metadata",
    "residual_diffusion_enabled",
    "validate_checkpoint_compatibility",
]
