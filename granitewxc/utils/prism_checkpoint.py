"""Semantic checkpoint provenance for NARR/MERRA-to-PRISM models.

Tensor shapes are not a sufficient compatibility check for these pipelines:
reordering same-shaped channels, changing a predictand transform, or switching
the decoder/attention behavior can all load successfully and still produce a
scientifically invalid field.  The contract below is intentionally composed of
JSON-like primitives so it survives ``torch.save`` without custom classes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from granitewxc.utils import normalization
from granitewxc.utils.predictands import (
    PHYSICALLY_NONNEGATIVE_VARS,
    canonicalize_precip_model,
    canonicalize_scaling_method,
)
from granitewxc.utils.prism_grid import load_canonical_grid


CONTRACT_KEY = "prism_pipeline_contract"
CONTRACT_SCHEMA_VERSION = 4
PRISM_DATA_TYPES = {"narr_prism", "merra_prism"}


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _primitive(value: Any) -> Any:
    """Convert config containers and scalar wrappers to stable primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_primitive(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _primitive(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "to_dict"):
        return _primitive(value.to_dict())
    if hasattr(value, "__dict__"):
        return _primitive(vars(value))
    return str(value)


def _ordered_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [_primitive(item) for item in value]
    return [_primitive(value)]


def _ordered_predictors(data: Any) -> list[list[Any]]:
    """Expand the YAML mapping while preserving its channel-forming order."""
    predictor_variables = _get(data, "predictor_variables", {}) or {}
    if not isinstance(predictor_variables, Mapping):
        raise TypeError("data.predictor_variables must be an ordered mapping")
    expanded: list[list[Any]] = []
    for variable, levels in predictor_variables.items():
        for level in _ordered_list(levels):
            expanded.append([str(variable), float(level)])
    return expanded


def _canonical_predictands(config: Any, output_vars: list[str]) -> list[dict[str, Any]]:
    """Resolve the output-transform semantics in channel order.

    This mirrors the public predictand parser without mutating the live config
    (checkpoint saving and validation must be side-effect free).
    """
    raw_predictands = _get(config, "predictands", {}) or {}
    resolved: list[dict[str, Any]] = []
    for name in output_vars:
        raw = _get(raw_predictands, name, {}) or {}
        raw_normalization = _get(raw, "normalization", {}) or {}
        raw_scaling = _get(raw, "scaling", {}) or {}
        normalization_block = raw_normalization or raw_scaling

        lower_name = str(name).lower()
        allow_negative = bool(
            _get(
                raw,
                "allow_negative_value",
                _get(raw_normalization, "allow_negative_value", False),
            )
        )
        physically_nonnegative = lower_name in PHYSICALLY_NONNEGATIVE_VARS
        default_nonnegative = physically_nonnegative and not allow_negative
        raw_nonnegative = _get(raw, "nonnegativity", {}) or {}
        nonnegative_enabled = bool(
            _get(raw_nonnegative, "enabled", default_nonnegative)
        )
        nonnegative_method = str(
            _get(
                raw_nonnegative,
                "method",
                "softplus" if nonnegative_enabled else "none",
            )
        ).strip().lower()
        if not nonnegative_enabled:
            nonnegative_method = "none"

        default_method = "divide_only" if default_nonnegative else "zscore"
        scaling_method = canonicalize_scaling_method(
            _get(normalization_block, "method", default_method)
        )
        default_stat = "p95" if scaling_method == "divide_only" else "mean"
        fixed_scale = _get(
            normalization_block,
            "fixed_scale",
            _get(raw_scaling, "fixed_scale", None),
        )
        resolved.append(
            {
                "name": str(name),
                "allow_negative_value": allow_negative,
                "nonnegativity": {
                    "enabled": nonnegative_enabled,
                    "method": nonnegative_method,
                },
                "scaling": {
                    "method": scaling_method,
                    "mode": str(
                        _get(
                            normalization_block,
                            "mode",
                            _get(raw_scaling, "mode", "global"),
                        )
                    ).strip().lower(),
                    "eps_std": float(
                        _get(
                            normalization_block,
                            "eps_std",
                            _get(raw_scaling, "eps_std", 1.0e-6),
                        )
                    ),
                    "scale_stat": str(
                        _get(
                            normalization_block,
                            "scale_stat",
                            _get(raw_scaling, "scale_stat", default_stat),
                        )
                    ).strip().lower(),
                    "fixed_scale": (
                        None if fixed_scale is None else float(fixed_scale)
                    ),
                },
            }
        )
    return resolved


def is_prism_config(config: Any) -> bool:
    return str(_get(_get(config, "data", {}), "type", "")).lower() in PRISM_DATA_TYPES


def build_prism_checkpoint_contract(config: Any) -> dict[str, Any] | None:
    """Describe coordinate, channel, topology, and decoding semantics."""
    if not is_prism_config(config):
        return None
    data = _get(config, "data", {}) or {}
    model = _get(config, "model", {}) or {}
    normalization_cfg = _get(config, "normalization", {}) or {}
    dates = _get(config, "dates", {}) or {}

    output_vars = [
        str(value)
        for value in _ordered_list(
            _get(data, "output_vars", _get(data, "target_variables", []))
        )
    ]
    target_variables = [
        str(value) for value in _ordered_list(_get(data, "target_variables", output_vars))
    ]
    input_vars = [str(value) for value in _ordered_list(_get(data, "input_vars", []))]
    ordered_predictors = _ordered_predictors(data)
    if bool(_get(data, "use_preprocessed", False)) and input_vars and ordered_predictors:
        data_channel_names = [
            f"{variable}_{int(level)}" for variable, level in ordered_predictors
        ]
        if _get(data, "static_elevation_file", None):
            data_channel_names.append("elev")
        expected_input_vars = [
            *data_channel_names,
            *(f"mask_{name}" for name in data_channel_names),
        ]
        if input_vars != expected_input_vars:
            raise ValueError(
                "data.input_vars does not match the ordered preprocessed predictor, "
                "elevation, and validity-mask channels: "
                f"configured={input_vars}, expected={expected_input_vars}"
            )

    scalar_dir = normalization.resolve_scalar_dir(config, for_writing=False)
    manifest = normalization.load_manifest(scalar_dir) or {}
    canonical_grid = load_canonical_grid(
        normalization.case_preprocess_dir(config), required=False
    )

    decoder_mode = str(
        _get(
            model,
            "decoder_upsampling_mode",
            _get(model, "encoder_decoder_upsampling_mode", "pixel_shuffle"),
        )
    ).strip().lower()
    if decoder_mode not in {"pixel_shuffle", "bilinear", "nearest", "conv_transpose"}:
        decoder_mode = "pixel_shuffle"

    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "identity": {
            "data_type": str(_get(data, "type", "")).lower(),
            "case_name": normalization.get_case_name(config),
            "training_dates": _primitive(_get(dates, "training", {})),
            "validation_dates": _primitive(_get(dates, "validation", {})),
        },
        "coordinates_and_artifacts": {
            "use_preprocessed": bool(_get(data, "use_preprocessed", False)),
            "training_core": [
                int(_get(data, "train_crop_size_lat", 0)),
                int(_get(data, "train_crop_size_lon", 0)),
            ],
            "training_stride": [
                int(_get(data, "training_tile_stride_lat", 0)),
                int(_get(data, "training_tile_stride_lon", 0)),
            ],
            "training_halo": [
                int(_get(data, "training_halo_lat", 0)),
                int(_get(data, "training_halo_lon", 0)),
            ],
            "regrid_method": str(_get(data, "regrid_method", "bilinear")).lower(),
            "grid": (
                canonical_grid.manifest_entry() if canonical_grid is not None else None
            ),
            "scalers": {
                name: {
                    "shape": entry.get("shape"),
                    "sha256": entry.get("sha256"),
                }
                for name, entry in (manifest.get("scalers") or {}).items()
            },
            "predictor_preprocessing_signature": manifest.get(
                "predictor_preprocessing_signature"
            ),
            "training_source_artifact_split_signature": manifest.get(
                normalization.TRAINING_SOURCE_ARTIFACT_SPLIT_SIGNATURE_KEY
            ),
        },
        "channels": {
            "input_vars": input_vars,
            "input_levels": _ordered_list(_get(data, "input_levels", [])),
            "ordered_predictors": ordered_predictors,
            "output_vars": output_vars,
            "target_variables": target_variables,
            "n_input_timestamps": int(_get(data, "n_input_timestamps", 1)),
            "use_static": bool(_get(data, "use_static", False)),
            "input_static_surface_vars": _ordered_list(
                _get(data, "input_static_surface_vars", [])
            ),
        },
        "normalization": {
            "predictor_method": str(
                _get(normalization_cfg, "predictor_method", "standardize")
            ).lower(),
            "predictor_mode": str(
                _get(normalization_cfg, "predictor_mode", "global")
            ).lower(),
            "target_method": str(
                _get(normalization_cfg, "target_method", "per_variable")
            ).lower(),
            "predictands": _canonical_predictands(config, output_vars),
        },
        "model_topology": {
            "embed_dim": _primitive(_get(model, "embed_dim", None)),
            "n_blocks_encoder": _primitive(_get(model, "n_blocks_encoder", None)),
            "mlp_multiplier": _primitive(_get(model, "mlp_multiplier", None)),
            "n_heads": _primitive(_get(model, "n_heads", None)),
            "dropout_rate": _primitive(_get(model, "dropout_rate", None)),
            "drop_path": _primitive(_get(model, "drop_path", None)),
            "downscaling_patch_size": _ordered_list(
                _get(model, "downscaling_patch_size", [])
            ),
            "downscaling_embed_dim": _primitive(
                _get(model, "downscaling_embed_dim", None)
            ),
            "encoder_decoder_type": _primitive(
                _get(model, "encoder_decoder_type", None)
            ),
            "encoder_decoder_upsampling_mode": _primitive(
                _get(model, "encoder_decoder_upsampling_mode", None)
            ),
            "encoder_decoder_kernel_size_per_stage": _primitive(
                _get(model, "encoder_decoder_kernel_size_per_stage", [])
            ),
            "encoder_decoder_scale_per_stage": _primitive(
                _get(model, "encoder_decoder_scale_per_stage", [])
            ),
            "encoder_decoder_conv_channels": _primitive(
                _get(model, "encoder_decoder_conv_channels", None)
            ),
            "unet": bool(_get(model, "unet", True)),
            "unet_upsample_scales": _ordered_list(
                _get(model, "unet_upsample_scales", [2, 2, 2])
            ),
            "unet_decoder_kernel_size": _ordered_list(
                _get(model, "unet_decoder_kernel_size", [3, 3, 3])
            ),
            "decoder_upsampling_mode": decoder_mode,
            "decoder_skip_source": str(
                _get(model, "decoder_skip_source", "legacy")
            ).lower(),
            "backbone_attention_scope": str(
                _get(model, "backbone_attention_scope", "legacy_global")
            ).lower(),
            "backbone_residual_mode": str(
                _get(model, "backbone_residual_mode", "legacy_ignored")
            ).lower(),
            "residual": _primitive(_get(model, "residual", None)),
            "residual_connection": bool(_get(model, "residual_connection", False)),
            "num_static_channels": int(_get(model, "num_static_channels", 0) or 0),
            "static_embedding_scale": float(
                _get(model, "static_embedding_scale", 1.0)
            ),
            "static_skip_scale": float(_get(model, "static_skip_scale", 1.0)),
            "static_dropout_p": float(_get(model, "static_dropout_p", 0.0)),
            "output_scaler_resize_mode": str(
                _get(model, "output_scaler_resize_mode", "bilinear")
            ).lower(),
            "output_scaler_align_corners": bool(
                _get(model, "output_scaler_align_corners", False)
            ),
            "loss_type": _primitive(_get(model, "loss_type", "patch_rmse_loss")),
            "backbone_use": bool(_get(config, "backbone_use", True)),
            "mask_unit_size": _ordered_list(_get(config, "mask_unit_size", [])),
            "finetune_w_static": bool(_get(config, "finetune_w_static", False)),
        },
        "precipitation": {
            "model": canonicalize_precip_model(
                _get(config, "precip_model", _get(config, "precip_head_type", "single_head"))
            ),
            "wet_threshold": float(_get(config, "precip_wet_threshold", 0.0)),
            "occurrence_probability_threshold": float(
                _get(config, "precip_occurrence_prob_threshold", 0.5)
            ),
            "lambda_occurrence": float(_get(config, "precip_lambda_occurrence", 1.0)),
            "lambda_amount": float(_get(config, "precip_lambda_amount", 1.0)),
            "amount_loss_type": str(
                _get(config, "precip_amount_loss_type", "smoothl1")
            ).lower(),
        },
    }


def validate_prism_checkpoint_contract(
    config: Any,
    checkpoint: Mapping[str, Any],
    *,
    role: str,
) -> dict[str, Any] | None:
    """Reject a checkpoint trained with different PRISM pipeline semantics."""
    expected = build_prism_checkpoint_contract(config)
    if expected is None:
        return None
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"[{role}] checkpoint must be a mapping with provenance")
    observed = checkpoint.get(CONTRACT_KEY)
    if observed is None:
        raise ValueError(
            f"[{role}] checkpoint has no {CONTRACT_KEY!r}. Its channel order, "
            "case-scoped scaler bytes/grid, regridding semantics, model behavior, "
            "and tiling context cannot be verified against the current PRISM "
            "pipeline. It must be retrained and saved with the full pipeline "
            "contract; matching tensor shapes alone are not scientifically "
            "compatible."
        )
    if not isinstance(observed, Mapping):
        raise ValueError(f"[{role}] invalid {CONTRACT_KEY}: expected a mapping")
    if observed.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise ValueError(
            f"[{role}] unsupported checkpoint PRISM contract schema "
            f"{observed.get('schema_version')!r}"
        )

    if dict(observed) != expected:
        keys = sorted(set(observed) | set(expected))
        mismatch_keys = [key for key in keys if observed.get(key) != expected.get(key)]
        details = "; ".join(
            f"{key}: checkpoint={observed.get(key)!r}, config={expected.get(key)!r}"
            for key in mismatch_keys
        )
        raise ValueError(
            f"[{role}] checkpoint PRISM pipeline contract mismatch ({details}). "
            "Use the exact training case/config artifacts or retrain."
        )
    return dict(observed)
