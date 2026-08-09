#!/usr/bin/env python
"""Python version of SA_downscaling_inference_T2_ACCESS-CM2_static.ipynb.

Fill in the parameter lists below (length NUM_RUNS) to loop over multiple
inference runs without editing the script each time.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import subprocess
import warnings
from concurrent.futures import ThreadPoolExecutor, Future
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import xarray as xr
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Ensure local modules resolve when running as a script.
SCRIPT_DIR = Path(__file__).resolve()
REPO_ROOT = SCRIPT_DIR.parents[3]
PROJECT_DIR = REPO_ROOT / "examples/CORDEX_ML"
for path in (REPO_ROOT, PROJECT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.chdir(PROJECT_DIR)

from cordex_inference import CordexWrappedDataset, build_inference_dataset, build_predictor_names  # noqa: E402
from utils.nearest_fill import repair_invalid_by_nearest_xr, summarize_invalid_counts_xr  # noqa: E402
from utils.predictand_runtime import assert_nonnegative_outputs, resolve_predictand_specs  # noqa: E402
from utils.inference_blending import infer_batch_with_boundary_mitigation, resolve_boundary_mitigation_settings  # noqa: E402
from utils.diffusion_inference import (  # noqa: E402
    add_predictions_to_dataset,
    infer_batch_ensemble,
    infer_head_type,
    reset_ensemble_generators,
    residual_transformation_stages,
    resolve_base_seed,
    resolve_ensemble_size,
    variable_array_map,
)
from utils.quantization_diagnostics import (  # noqa: E402
    RunningStats,
    format_compact_table,
    quantization_detector,
    save_distribution_plot,
    save_json,
)
from granitewxc.models.model import get_finetune_model_UNET  # noqa: E402
from granitewxc.utils.checkpoint_metadata import validate_checkpoint_compatibility  # noqa: E402
from granitewxc.utils.config import get_config  # noqa: E402
from sa_params import (  # noqa: E402
    UserParams,
    export_params,
    resolve_checkpoint,
    resolve_existing_run_dir,
    validate_paths,
)
from run_utils import assert_no_eccc_reference, load_run_manifest  # noqa: E402


_CANONICAL_TARGET_UNITS = {
    "pr": "mm/day",
    "tasmax": "K",
}


def _iter_runtime_model_wrappers(model: torch.nn.Module):
    """Yield a model and common DDP/compile wrappers without following cycles."""

    pending = [model]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in visited:
            continue
        visited.add(identity)
        yield current
        for attribute in ("module", "_orig_mod"):
            candidate = getattr(current, attribute, None)
            if candidate is not None:
                pending.append(candidate)


def _runtime_model_with_attribute(
    model: torch.nn.Module, attribute: str
) -> torch.nn.Module:
    return next(
        (
            candidate
            for candidate in _iter_runtime_model_wrappers(model)
            if hasattr(candidate, attribute)
        ),
        model,
    )


def _normalized_unit_token(value: object) -> str:
    """Return a conservative token used only for supported-unit lookup."""

    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="strict")
    text = str(value).strip().lower()
    text = (
        text.replace("−", "-")
        .replace("⁻", "-")
        .replace("²", "2")
        .replace("·", "")
        .replace("**", "^")
    )
    return "".join(text.split())


def _target_unit_rule(var_name: str, source_units: object) -> dict[str, object]:
    """Resolve one explicitly supported source-to-canonical conversion."""

    name = str(var_name).strip().lower()
    token = _normalized_unit_token(source_units)
    if name == "pr":
        flux_tokens = {
            "kgm-2s-1",
            "kgm^-2s^-1",
            "kg/m2/s",
            "mms-1",
            "mms^-1",
            "mm/s",
        }
        daily_tokens = {
            "mm/day",
            "mmday-1",
            "mmday^-1",
            "mmd-1",
            "mmd^-1",
            "kgm-2day-1",
            "kgm^-2day^-1",
            "kg/m2/day",
        }
        if token in flux_tokens:
            return {
                "source_unit_key": "precipitation_depth_per_second",
                "canonical_units": _CANONICAL_TARGET_UNITS[name],
                "conversion_factor": 86_400.0,
            }
        if token in daily_tokens:
            return {
                "source_unit_key": "precipitation_depth_per_day",
                "canonical_units": _CANONICAL_TARGET_UNITS[name],
                "conversion_factor": 1.0,
            }
    elif name == "tasmax" and token in {"k", "kelvin", "degk", "degreekelvin"}:
        return {
            "source_unit_key": "kelvin",
            "canonical_units": _CANONICAL_TARGET_UNITS[name],
            "conversion_factor": 1.0,
        }
    elif name not in _CANONICAL_TARGET_UNITS:
        raise ValueError(
            f"No canonical inference-target unit contract is defined for {var_name!r}."
        )

    raise ValueError(
        f"Unsupported units {source_units!r} for inference target {var_name!r}; "
        f"expected a unit convertible to {_CANONICAL_TARGET_UNITS[name]!r}."
    )


def _resolve_target_unit_provenance(
    target_paths: list[str], target_vars: list[str]
) -> dict[str, dict[str, object]]:
    """Validate source units across files and return canonical conversion metadata."""

    if not target_paths:
        raise ValueError("At least one inference target path is required for unit validation.")
    if not target_vars:
        raise ValueError("At least one inference target variable is required for unit validation.")

    records: dict[str, list[dict[str, object]]] = {name: [] for name in target_vars}
    for raw_path in target_paths:
        path = Path(raw_path).expanduser().resolve(strict=True)
        with xr.open_dataset(path, decode_times=False) as dataset:
            for name in target_vars:
                if name not in dataset.data_vars:
                    raise ValueError(f"Inference target {path} is missing variable {name!r}.")
                source_units = dataset[name].attrs.get("units")
                if source_units in (None, ""):
                    raise ValueError(
                        f"Inference target variable {name!r} in {path} has no units attribute."
                    )
                rule = _target_unit_rule(name, source_units)
                records[name].append(
                    {
                        **rule,
                        "source_units": str(source_units),
                        "source_path": str(path),
                    }
                )

    provenance: dict[str, dict[str, object]] = {}
    for name, per_file in records.items():
        source_keys = {str(record["source_unit_key"]) for record in per_file}
        factors = {float(record["conversion_factor"]) for record in per_file}
        canonical_units = {str(record["canonical_units"]) for record in per_file}
        if len(source_keys) != 1 or len(factors) != 1 or len(canonical_units) != 1:
            detail = [
                {
                    "path": record["source_path"],
                    "units": record["source_units"],
                    "factor": record["conversion_factor"],
                }
                for record in per_file
            ]
            raise ValueError(
                f"Mixed units for inference target variable {name!r}: {detail}. "
                "Use one source-unit convention per inference run."
            )
        provenance[name] = {
            "source_units": sorted(
                {str(record["source_units"]) for record in per_file}
            ),
            "source_unit_key": next(iter(source_keys)),
            "canonical_units": next(iter(canonical_units)),
            "conversion_factor": next(iter(factors)),
            "source_paths": [str(record["source_path"]) for record in per_file],
        }
    return provenance


def _canonicalize_target_tensor(
    target: torch.Tensor,
    target_vars: list[str],
    unit_provenance: dict[str, dict[str, object]],
) -> torch.Tensor:
    """Convert target channels without changing rank, shape, device, or dtype."""

    if target.ndim != 4:
        raise ValueError(f"Inference targets must have shape [B,V,H,W], got {tuple(target.shape)}.")
    if target.shape[1] != len(target_vars):
        raise ValueError(
            f"Inference target channel count {target.shape[1]} does not match "
            f"target_vars={target_vars}."
        )
    missing = [name for name in target_vars if name not in unit_provenance]
    if missing:
        raise ValueError(f"Missing unit provenance for inference targets: {missing}.")
    factors = target.new_tensor(
        [float(unit_provenance[name]["conversion_factor"]) for name in target_vars]
    ).view(1, -1, 1, 1)
    return target * factors


def _canonical_target_attrs(
    target_attrs: dict[str, dict[str, object]],
    unit_provenance: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Replace inherited truth units with the units of the numeric predictions."""

    result: dict[str, dict[str, object]] = {}
    for name, attrs in target_attrs.items():
        updated = dict(attrs)
        conversion = unit_provenance.get(name)
        if conversion is not None:
            source_units = list(conversion["source_units"])
            updated["units"] = str(conversion["canonical_units"])
            updated["canonical_units"] = str(conversion["canonical_units"])
            updated["source_target_units"] = (
                source_units[0] if len(source_units) == 1 else json.dumps(source_units)
            )
            updated["source_target_to_canonical_factor"] = float(
                conversion["conversion_factor"]
            )
        result[name] = updated
    return result


def _output_runtime_contract(
    target_vars: list[str],
    unit_provenance: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Describe the channel/variable order and numeric units of saved outputs."""

    missing = [name for name in target_vars if name not in unit_provenance]
    if missing:
        raise ValueError(f"Missing output-unit provenance for variables: {missing}.")
    return {
        "variable_order": list(target_vars),
        "canonical_units": {
            name: str(unit_provenance[name]["canonical_units"])
            for name in target_vars
        },
        "numeric_storage": "physical_canonical_units",
    }


def _boundary_runtime_provenance(boundary_cfg: object) -> dict[str, object]:
    """Describe the effective boundary path without silently overriding config."""

    enabled = bool(getattr(boundary_cfg, "enabled", False))
    force_full_frame = bool(getattr(boundary_cfg, "force_full_frame", False))
    tile_size = getattr(boundary_cfg, "tile_size", None)
    overlap = getattr(boundary_cfg, "overlap", (0, 0))
    halo = getattr(boundary_cfg, "halo", (0, 0))
    tile_origin = getattr(boundary_cfg, "tile_origin", (0, 0))
    mode = "overlap_tiled" if enabled and not force_full_frame else "full_frame"
    tile_stride = (
        [
            max(1, int(tile_size[0]) - int(overlap[0])),
            max(1, int(tile_size[1]) - int(overlap[1])),
        ]
        if tile_size is not None
        else None
    )
    deblock = getattr(boundary_cfg, "deblock", None)
    return {
        "mode": mode,
        "enabled": enabled,
        "force_full_frame": force_full_frame,
        "tile_size": list(tile_size) if tile_size is not None else None,
        "overlap": list(overlap),
        "halo": list(halo),
        "tile_origin": list(tile_origin),
        "tile_stride": tile_stride,
        "blend_window": str(getattr(boundary_cfg, "blend_window", "uniform")),
        "blend_sigma": float(getattr(boundary_cfg, "blend_sigma", 0.0)),
        "deblock_enabled": bool(getattr(deblock, "enabled", False)),
        "deblock_boundary_width": int(getattr(deblock, "boundary_width", 0)),
        "deblock_strength": float(getattr(deblock, "strength", 0.0)),
        "deblock_kernel_size": int(getattr(deblock, "kernel_size", 0)),
    }


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalization_scalar_provenance(
    config: object, checkpoint: dict
) -> dict[str, dict[str, object]]:
    """Hash the exact scalar files used to reconstruct the inference model."""

    scalar_fields = {
        "inputs_mean": "input_mu",
        "inputs_std": "input_sigma",
        "targets_mean": "target_mu",
        "targets_std": "target_sigma",
    }
    metadata = checkpoint.get("metadata", {}) if isinstance(checkpoint, dict) else {}
    compatibility = metadata.get("compatibility", {})
    checkpoint_scalars = (
        compatibility.get("normalization_scalars", {})
        if isinstance(compatibility, dict)
        else {}
    )
    model_config = getattr(config, "model", None)
    result: dict[str, dict[str, object]] = {}
    for scalar_name, config_attr in scalar_fields.items():
        raw_path = getattr(model_config, config_attr, None)
        if raw_path in (None, ""):
            raise ValueError(
                f"Missing model.{config_attr} while constructing scalar provenance."
            )
        path = Path(str(raw_path)).expanduser().resolve(strict=True)
        sha256 = _sha256_path(path)
        checkpoint_record = checkpoint_scalars.get(scalar_name, {})
        expected_sha256 = (
            checkpoint_record.get("sha256")
            if isinstance(checkpoint_record, dict)
            else None
        )
        if expected_sha256 and str(expected_sha256).lower() != sha256.lower():
            raise RuntimeError(
                f"Scalar hash mismatch for {scalar_name}: runtime {sha256}, "
                f"checkpoint {expected_sha256}."
            )
        result[scalar_name] = {
            "config_field": f"model.{config_attr}",
            "path": str(path),
            "sha256": sha256,
            "size_bytes": int(path.stat().st_size),
            "checkpoint_sha256": (
                str(expected_sha256) if expected_sha256 is not None else None
            ),
        }
    return result


def _apply_residual_alpha_runtime_override(
    config: object,
    environ: dict[str, str] | None = None,
) -> dict[str, object]:
    """Resolve and apply the optional ``SA_RESIDUAL_ALPHA`` deployment gate."""

    environment = os.environ if environ is None else environ
    model_cfg = getattr(config, "model", None)
    diffusion_cfg = (
        model_cfg.get("diffusion")
        if isinstance(model_cfg, dict)
        else getattr(model_cfg, "diffusion", None)
    )

    if isinstance(diffusion_cfg, dict):
        residual_enabled = bool(diffusion_cfg.get("residual_diffusion", False))
        alpha_is_configured = "residual_application_scale" in diffusion_cfg
        configured_value = diffusion_cfg.get("residual_application_scale", 0.0)
    else:
        residual_enabled = bool(
            getattr(diffusion_cfg, "residual_diffusion", False)
        )
        alpha_is_configured = bool(
            diffusion_cfg is not None
            and hasattr(diffusion_cfg, "residual_application_scale")
        )
        configured_value = getattr(
            diffusion_cfg, "residual_application_scale", 0.0
        )

    environment_variable = "SA_RESIDUAL_ALPHA"
    override_is_set = environment_variable in environment
    raw_override = environment.get(environment_variable)
    if override_is_set and not residual_enabled:
        raise ValueError(
            f"{environment_variable} is set, but model.diffusion.residual_diffusion "
            "is not enabled. Remove the override or use a residual-diffusion run."
        )

    raw_value = raw_override if override_is_set else configured_value
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        source = environment_variable if override_is_set else "resolved configuration"
        raise ValueError(
            f"Residual application scale from {source} must be numeric, got "
            f"{raw_value!r}."
        ) from exc
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        source = environment_variable if override_is_set else "resolved configuration"
        raise ValueError(
            f"Residual application scale from {source} must be finite and in "
            f"[0, 1], got {raw_value!r}."
        )

    if diffusion_cfg is not None:
        if isinstance(diffusion_cfg, dict):
            diffusion_cfg["residual_application_scale"] = value
        else:
            setattr(diffusion_cfg, "residual_application_scale", value)

    return {
        "environment_variable": environment_variable,
        "source": (
            f"environment:{environment_variable}"
            if override_is_set
            else ("resolved_config" if alpha_is_configured else "safe_default")
        ),
        "value": value,
        "configured_value": float(configured_value),
        "override_is_set": bool(override_is_set),
    }


def _diffusion_runtime_provenance(
    model: torch.nn.Module,
    *,
    head_type: str,
    ensemble_size: int,
    base_seed: int,
    checkpoint: dict,
    inference_batch_size: int | None = None,
    residual_alpha_provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    """Capture the effective sampler, residual gate, and score padding contract."""

    runtime_model = _runtime_model_with_attribute(model, "diffusion_head")
    head = getattr(runtime_model, "diffusion_head", None)
    cfg = getattr(head, "cfg", None)
    stochastic_diffusion = str(head_type).strip().lower() == "diffusion"
    rng_provenance: dict[str, object] = {
        "rng_stream_scope": (
            "reset_per_scenario_run"
            if stochastic_diffusion
            else "not_applicable_deterministic"
        ),
        "batch_size_invariant": not stochastic_diffusion,
        "exact_reproduction_requires_same_batching": stochastic_diffusion,
    }
    if inference_batch_size is not None:
        if int(inference_batch_size) <= 0:
            raise ValueError("inference_batch_size must be positive when provided.")
        rng_provenance["inference_batch_size"] = int(inference_batch_size)
    if residual_alpha_provenance is not None:
        rng_provenance["runtime_overrides"] = {
            "residual_alpha": dict(residual_alpha_provenance)
        }

    if cfg is None:
        return {
            "head_type": str(head_type),
            "enabled": False,
            "ensemble_size": int(ensemble_size),
            "base_seed": int(base_seed),
            "member_seeds": [int(base_seed)],
            **rng_provenance,
        }

    fields = (
        "sde",
        "beta_min",
        "beta_max",
        "sigma_min",
        "sigma_max",
        "num_scales",
        "continuous",
        "sampling_method",
        "eta",
        "predictor",
        "corrector",
        "snr",
        "n_corrector_steps",
        "num_sampling_steps",
        "probability_flow",
        "denoise",
        "sampling_eps",
        "prediction_type",
        "noise_conditioning_scale",
        "residual_magnitude_guard_multiple",
        "residual_guard_min_count",
        "zero_init_output",
    )
    sampling = {name: getattr(cfg, name) for name in fields if hasattr(cfg, name)}
    for name, value in list(sampling.items()):
        if isinstance(value, np.generic):
            sampling[name] = value.item()
        elif isinstance(value, tuple):
            sampling[name] = list(value)

    # Real diffusion heads expose ``score_fn`` as a bound callable that wraps
    # the actual convolutional module stored in ``score_model``.  Only inspect
    # a verified nn.Module; calling ``.modules()`` on the bound method would
    # fail during real-model provenance collection.
    score_model = getattr(head, "score_model", None)
    if not isinstance(score_model, torch.nn.Module):
        legacy_score_module = getattr(head, "score_fn", None)
        score_model = (
            legacy_score_module
            if isinstance(legacy_score_module, torch.nn.Module)
            else None
        )
    score_convs = (
        [
            module
            for module in score_model.modules()
            if isinstance(module, torch.nn.Conv2d)
        ]
        if score_model is not None
        else []
    )
    padded_score_convs = [
        module
        for module in score_convs
        if any(int(value) > 0 for value in tuple(module.padding))
    ]
    effective_padding_modes = sorted(
        {str(module.padding_mode) for module in padded_score_convs}
    )

    metadata = checkpoint.get("metadata", {}) if isinstance(checkpoint, dict) else {}
    compatibility = metadata.get("compatibility", {})
    checkpoint_residual = (
        compatibility.get("residual_contract", {})
        if isinstance(compatibility, dict)
        else {}
    )
    residual_enabled = bool(getattr(cfg, "residual_diffusion", False))
    residual_application_scale = float(
        getattr(cfg, "residual_application_scale", 0.0)
    )
    residual_contract = (
        dict(checkpoint_residual) if isinstance(checkpoint_residual, dict) else {}
    )
    residual_contract.update(
        {
            "enabled": residual_enabled,
            "definition": residual_contract.get(
                "definition",
                "target_normalized - stop_gradient(deterministic_baseline_normalized)",
            ),
            "reconstruction": (
                "deterministic_baseline_normalized + "
                "residual_application_scale * sampled_residual_normalized"
            ),
            "residual_application_scale": residual_application_scale,
            "alpha": residual_application_scale,
        }
    )
    if residual_alpha_provenance is not None:
        residual_contract.update(
            {
                "alpha_source": residual_alpha_provenance.get("source"),
                "alpha_environment_variable": residual_alpha_provenance.get(
                    "environment_variable"
                ),
                "alpha_override_is_set": residual_alpha_provenance.get(
                    "override_is_set"
                ),
            }
        )
    return {
        "head_type": str(head_type),
        "enabled": True,
        "ensemble_size": int(ensemble_size),
        "base_seed": int(base_seed),
        "member_seeds": [
            int(base_seed) + member_idx for member_idx in range(int(ensemble_size))
        ],
        **rng_provenance,
        "sampling": sampling,
        "residual_contract": residual_contract,
        "score_network": {
            "configured_padding_mode": str(
                getattr(cfg, "padding_mode", "unspecified")
            ),
            "effective_padded_conv_modes": effective_padding_modes,
            "conv2d_count": len(score_convs),
            "padded_conv2d_count": len(padded_score_convs),
        },
    }


# ===================== USER PARAMETERS (EDIT ME) =====================
NUM_RUNS = 12

# Each list must have NUM_RUNS entries.
TEST_SPLITS = ["test/historical/predictors/perfect","test/historical/predictors/perfect",
        "test/historical/predictors/imperfect", "test/historical/predictors/imperfect",
        "test/mid_century/predictors/perfect","test/mid_century/predictors/perfect",
        "test/mid_century/predictors/imperfect", "test/mid_century/predictors/imperfect",
        "test/end_century/predictors/perfect","test/end_century/predictors/perfect",
        "test/end_century/predictors/imperfect", "test/end_century/predictors/imperfect"
]

PREDICTOR_FILES = ["ACCESS-CM2_1981-2000.nc","NorESM2-MM_1981-2000.nc",
        "ACCESS-CM2_1981-2000.nc","NorESM2-MM_1981-2000.nc",
        "ACCESS-CM2_2041-2060.nc","NorESM2-MM_2041-2060.nc",
        "ACCESS-CM2_2041-2060.nc","NorESM2-MM_2041-2060.nc",
        "ACCESS-CM2_2080-2099.nc","NorESM2-MM_2080-2099.nc",
        "ACCESS-CM2_2080-2099.nc","NorESM2-MM_2080-2099.nc"
]

# Each predictor run is paired with its matching held-out target.  Note that
# NorESM predictor files use "NorESM2-MM", while the target filenames use the
# CORDEX archive spelling "Nor-ESM2-MM".
TARGET_FILES = [
        "test/historical/target/pr_tasmax_ACCESS-CM2_1981-2000.nc",
        "test/historical/target/pr_tasmax_Nor-ESM2-MM_1981-2000.nc",
        "test/historical/target/pr_tasmax_ACCESS-CM2_1981-2000.nc",
        "test/historical/target/pr_tasmax_Nor-ESM2-MM_1981-2000.nc",
        "test/mid_century/target/pr_tasmax_ACCESS-CM2_2041-2060.nc",
        "test/mid_century/target/pr_tasmax_Nor-ESM2-MM_2041-2060.nc",
        "test/mid_century/target/pr_tasmax_ACCESS-CM2_2041-2060.nc",
        "test/mid_century/target/pr_tasmax_Nor-ESM2-MM_2041-2060.nc",
        "test/end_century/target/pr_tasmax_ACCESS-CM2_2080-2099.nc",
        "test/end_century/target/pr_tasmax_Nor-ESM2-MM_2080-2099.nc",
        "test/end_century/target/pr_tasmax_ACCESS-CM2_2080-2099.nc",
        "test/end_century/target/pr_tasmax_Nor-ESM2-MM_2080-2099.nc",
]

PREDICTION_OUTPUT_NAMES = ["Predictions_pr_tasmax_ACCESS-CM2_1981-2000.nc", "Predictions_pr_tasmax_NorESM2-MM_1981-2000.nc",
        "Predictions_pr_tasmax_ACCESS-CM2_1981-2000.nc", "Predictions_pr_tasmax_NorESM2-MM_1981-2000.nc",
        "Predictions_pr_tasmax_ACCESS-CM2_2041-2060.nc", "Predictions_pr_tasmax_NorESM2-MM_2041-2060.nc",
        "Predictions_pr_tasmax_ACCESS-CM2_2041-2060.nc", "Predictions_pr_tasmax_NorESM2-MM_2041-2060.nc",
        "Predictions_pr_tasmax_ACCESS-CM2_2080-2099.nc", "Predictions_pr_tasmax_NorESM2-MM_2080-2099.nc",
        "Predictions_pr_tasmax_ACCESS-CM2_2080-2099.nc", "Predictions_pr_tasmax_NorESM2-MM_2080-2099.nc"
]

# Relative to the case-specific run directory (runs_root/<case_name>);
# _case_inference_root() joins each entry under that run dir at runtime.
INFERENCE_OUTPUT_ROOTS = ["predictions/historical/perfect/",
        "predictions/historical/perfect/",
        "predictions/historical/imperfect/",
        "predictions/historical/imperfect/",
        "predictions/mid-century/perfect/",
        "predictions/mid-century/perfect/",
        "predictions/mid-century/imperfect/",
        "predictions/mid-century/imperfect/",
        "predictions/end-century/perfect/",
        "predictions/end-century/perfect/",
        "predictions/end-century/imperfect/",
        "predictions/end-century/imperfect/"
]

# Fixed config for the fine-tuned model
REPO_ROOT = REPO_ROOT.resolve()
PROJECT_DIR = PROJECT_DIR.resolve()
DATASET_ROOT = REPO_ROOT / "granite-geospatial-wxc-downscaling/CORDEX/SA_domain"
if not DATASET_ROOT.exists():
    DATASET_ROOT = Path("D:/CORDEX/SA_domain")
CONFIG_PATH = PROJECT_DIR / "SA_T2_ACCESS-CM2_static_twophase.yaml"
RUNS_ROOT = PROJECT_DIR / "runs/SA_T2_ACCESS-CM2_static_train"

TRAIN_SPLIT = "train/Emulator_hist_future"

FINETUNE_RUN_NAME = getattr(get_config(str(CONFIG_PATH)), "case_name", None)  # set None to auto-pick latest
USE_STATIC = True
STATIC_PATH = DATASET_ROOT / TRAIN_SPLIT / "predictors" / "Static_fields.nc"

REPAIR_INVALID_INPUTS = True

DEVICE_TARGET = "cuda"
NUM_WORKERS = 4
INFERENCE_BATCH_SIZE = 16  # inference-only; does not affect model weights
PREFERRED_CHECKPOINT = "last"
MIN_FREE_GB = 8  # minimum free GPU memory to consider "idle"
MAX_GPUS = 1  # set to an int to cap how many GPUs to expose
RESPECT_CUDA_VISIBLE_DEVICES = True  # ignore preset CUDA_VISIBLE_DEVICES when auto-selecting idle GPUs
ENABLE_MIXED_PRECISION = True  # bfloat16/float16 autocast; FORCE_OUTPUT_FLOAT32 casts outputs back
FORCE_OUTPUT_FLOAT32 = True
USE_TORCH_COMPILE = False  # ~30-60s warm-up on first run; pays off across 12 runs
NETCDF_OUTPUT_DTYPE = "float32"
DIAGNOSTIC_MAX_SAMPLES = 250_000
DIAGNOSTIC_ROUND_DECIMALS = 6
DIAGNOSTIC_TIME_WINDOW = 5
DIAGNOSTIC_SPATIAL_WINDOW = 20
DIAGNOSTIC_STAGE_SAMPLE_COUNT = 3
SAVE_DIAGNOSTICS_JSON = True
SAVE_DISTRIBUTION_PLOT = True

# Set to run a subset: e.g. (0, 1) for only run 1, (1, 12) for runs 2-12, None for all.
#RUN_RANGE: tuple[int, int] | None = (0, 1)
RUN_RANGE: tuple[int, int] | None = (0, 1)

DIST_ENABLED = False
DIST_RANK = 0
DIST_WORLD_SIZE = 1
DIST_LOCAL_RANK = 0

# ================================================================


def _check_list_lengths() -> None:
    lists = {
        "TEST_SPLITS": TEST_SPLITS,
        "PREDICTOR_FILES": PREDICTOR_FILES,
        "TARGET_FILES": TARGET_FILES,
        "PREDICTION_OUTPUT_NAMES": PREDICTION_OUTPUT_NAMES,
        "INFERENCE_OUTPUT_ROOTS": INFERENCE_OUTPUT_ROOTS,
    }
    for name, values in lists.items():
        if len(values) != NUM_RUNS:
            raise ValueError(f"{name} must have {NUM_RUNS} entries (got {len(values)})")

    for index, (test_split, predictor_file, target_file) in enumerate(
        zip(TEST_SPLITS, PREDICTOR_FILES, TARGET_FILES), start=1
    ):
        predictor_root = DATASET_ROOT / test_split
        if not predictor_root.is_dir():
            raise FileNotFoundError(
                f"Run {index:02d} predictor directory does not exist: {predictor_root}"
            )
        if predictor_file:
            predictor_path = predictor_root / predictor_file
            if not predictor_path.is_file():
                raise FileNotFoundError(
                    f"Run {index:02d} predictor file does not exist: {predictor_path}"
                )

        target_path = DATASET_ROOT / target_file
        if not target_path.is_file():
            raise FileNotFoundError(
                f"Run {index:02d} target file does not exist: {target_path}"
            )


def _candidate_runs_roots(preferred_runs_root: Path) -> list[Path]:
    """Return fallback candidates for run roots (runs_v6 -> runs)."""
    try:
        rel = preferred_runs_root.resolve().relative_to(PROJECT_DIR.resolve())
    except Exception:
        return [preferred_runs_root]
    if not rel.parts:
        return [preferred_runs_root]
    tail = Path(*rel.parts[1:]) if len(rel.parts) > 1 else Path()
    preferred_base = rel.parts[0]
    base_order = [name for name in (preferred_base, "runs", "runs_v6") if name in ("runs", "runs_v6")]
    base_order = list(dict.fromkeys(base_order))
    return [PROJECT_DIR / base / tail for base in base_order]


def _select_runs_root(preferred_runs_root: Path, run_name_hint: str | None) -> Path:
    """Choose a run root that actually contains the requested run/manifest."""
    candidates = _candidate_runs_roots(preferred_runs_root)

    if run_name_hint:
        for candidate in candidates:
            manifest = candidate / run_name_hint / "run_manifest.json"
            if manifest.exists():
                return candidate

    for candidate in candidates:
        if not candidate.exists():
            continue
        for child in candidate.iterdir():
            if child.is_dir() and (child / "run_manifest.json").exists():
                return candidate

    return preferred_runs_root


def _case_inference_root(entry, run_dir: Path) -> Path:
    """Root an inference output path under the case-specific run directory.

    Preserves the meaningful '.../predictions/<scenario>/<kind>/' substructure
    while discarding any hardcoded absolute prefix, so every inference output
    lands under the case folder (runs_root/<case_name>).
    """
    text = str(entry).replace("\\", "/")
    marker = "predictions/"
    lowered = text.lower()
    if marker in lowered:
        return run_dir / text[lowered.index(marker):]
    return run_dir / "predictions"


def _resolve_inputs(values):
    return [str(Path(p).resolve()) for p in values]


def _select_idle_gpus(min_free_gb=8, max_gpus=None, respect_visible_devices=True):
    visible = None
    visible_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_env and respect_visible_devices:
        visible = [int(v) for v in visible_env.split(",") if v.strip() != ""]
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception as exc:
        print(f"GPU auto-selection skipped (nvidia-smi unavailable): {exc}")
        return None
    rows = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        idx_str, free_str = line.split(",")
        idx = int(idx_str.strip())
        free_mb = int(free_str.strip())
        rows.append((idx, free_mb))
    rows.sort(key=lambda x: x[1], reverse=True)
    if visible is not None:
        rows = [row for row in rows if row[0] in visible]
    selected = [idx for idx, free_mb in rows if free_mb >= min_free_gb * 1024]
    if max_gpus is not None:
        selected = selected[:max_gpus]
    if not selected and visible is not None:
        return visible
    return selected


def _configure_idle_gpus() -> None:
    selected = _select_idle_gpus(MIN_FREE_GB, MAX_GPUS, RESPECT_CUDA_VISIBLE_DEVICES)
    if selected:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in selected)
        print(f"Using idle GPUs (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})")
    else:
        print("Using default CUDA_VISIBLE_DEVICES (no idle GPU filter applied).")


def _select_device() -> torch.device:
    target = (DEVICE_TARGET or "auto").lower()
    if target == "cpu":
        return torch.device("cpu")

    if DIST_ENABLED:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed inference requested but CUDA is unavailable.")
        return torch.device(f"cuda:{DIST_LOCAL_RANK}")

    if target == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device_target is 'cuda' but no CUDA device is available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _collect_scaler_dtype_summary(model: torch.nn.Module) -> dict[str, str]:
    summary: dict[str, str] = {}
    for name in (
        "input_scalers_mu",
        "input_scalers_sigma",
        "static_input_scalers_mu",
        "static_input_scalers_sigma",
        "output_scalers_mu",
        "output_scalers_sigma",
    ):
        tensor = getattr(model, name, None)
        if torch.is_tensor(tensor):
            summary[name] = str(tensor.dtype)
    return summary


def _normalize_predictors_for_diagnostics(batch: dict[str, torch.Tensor], model: torch.nn.Module) -> torch.Tensor:
    x = batch["x"].float()
    batch_size, _, height, width = x.shape
    n_input_timestamps = int(getattr(model, "n_input_timestamps", 1))
    x_sep_time = x.view(batch_size, n_input_timestamps, -1, height, width)
    mu = model.input_scalers_mu.view(1, 1, -1, 1, 1).to(device=x.device, dtype=x.dtype)
    sigma = model.input_scalers_sigma.view(1, 1, -1, 1, 1).to(device=x.device, dtype=x.dtype)
    eps = float(getattr(model, "input_scalers_epsilon", 1e-6))
    x_norm = (x_sep_time - mu) / (sigma + eps)
    return x_norm.reshape(batch_size, -1, height, width)


def _to_pre_inverse_output(pred_inverse: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    pred = pred_inverse.float()
    sigma = model.output_scalers_sigma.to(device=pred.device, dtype=pred.dtype)
    mu = getattr(model, "output_scalers_mu", None)
    if mu is None:
        mu = torch.zeros_like(sigma)
    else:
        mu = mu.to(device=pred.device, dtype=pred.dtype)
    return (pred - mu) / (sigma + 1e-12)


def _run_full_inference(
    dataloader,
    model,
    device,
    target_vars,
    boundary_cfg,
    target_unit_provenance,
    head_type="deterministic",
    ensemble_size=1,
    base_seed=42,
):
    # A model is reused across the scenario loop. Restart member RNG streams at
    # every scenario boundary so a scenario does not depend on which run was
    # evaluated before it; streams still advance normally across its batches.
    reset_ensemble_generators(model)
    predictions = []
    pre_inverse_predictions = []
    baseline_predictions = []
    targets = []
    transformation_sample_parts: dict[str, list[np.ndarray]] = {}
    transformation_samples_remaining = max(0, int(DIAGNOSTIC_STAGE_SAMPLE_COUNT))

    diagnostic_model = _runtime_model_with_attribute(model, "diffusion_head")
    diffusion_head = getattr(diagnostic_model, "diffusion_head", None)
    residual_mode = bool(
        diffusion_head is not None
        and getattr(getattr(diffusion_head, "cfg", None), "residual_diffusion", False)
    )
    residual_stage_names = (
        "physical_unet_prediction",
        "physical_ground_truth",
        "physical_true_residual",
        "normalized_true_residual",
        "raw_predicted_normalized_residual",
        "predicted_normalized_residual",
        "applied_normalized_residual",
        "raw_denormalized_predicted_residual",
        "denormalized_predicted_residual",
        "applied_denormalized_residual",
        "final_physical_prediction",
    )

    stage_stats = {
        "raw_predictors": RunningStats(
            max_samples=DIAGNOSTIC_MAX_SAMPLES,
            round_decimals=DIAGNOSTIC_ROUND_DECIMALS,
        ),
        "normalized_predictors": RunningStats(
            max_samples=DIAGNOSTIC_MAX_SAMPLES,
            round_decimals=DIAGNOSTIC_ROUND_DECIMALS,
        ),
        "raw_model_outputs_pre_inverse": RunningStats(
            max_samples=DIAGNOSTIC_MAX_SAMPLES,
            round_decimals=DIAGNOSTIC_ROUND_DECIMALS,
        ),
        "inverse_outputs": RunningStats(
            max_samples=DIAGNOSTIC_MAX_SAMPLES,
            round_decimals=DIAGNOSTIC_ROUND_DECIMALS,
        ),
    }
    if residual_mode:
        stage_stats.update(
            {
                name: RunningStats(
                    max_samples=DIAGNOSTIC_MAX_SAMPLES,
                    round_decimals=DIAGNOSTIC_ROUND_DECIMALS,
                )
                for name in residual_stage_names
            }
        )
    stage_stats_by_var = {
        "raw_model_outputs_pre_inverse": {
            name: RunningStats(max_samples=DIAGNOSTIC_MAX_SAMPLES, round_decimals=DIAGNOSTIC_ROUND_DECIMALS)
            for name in target_vars
        },
        "inverse_outputs": {
            name: RunningStats(max_samples=DIAGNOSTIC_MAX_SAMPLES, round_decimals=DIAGNOSTIC_ROUND_DECIMALS)
            for name in target_vars
        },
        "targets": {
            name: RunningStats(max_samples=DIAGNOSTIC_MAX_SAMPLES, round_decimals=DIAGNOSTIC_ROUND_DECIMALS)
            for name in target_vars
        },
    }
    if residual_mode:
        stage_stats_by_var.update(
            {
                stage: {
                    name: RunningStats(
                        max_samples=DIAGNOSTIC_MAX_SAMPLES,
                        round_decimals=DIAGNOSTIC_ROUND_DECIMALS,
                    )
                    for name in target_vars
                }
                for stage in residual_stage_names
            }
        )

    autocast_enabled = bool(ENABLE_MIXED_PRECISION and device.type == "cuda")
    autocast_dtype = (
        torch.bfloat16 if (autocast_enabled and torch.cuda.is_bf16_supported()) else torch.float16
    )

    def autocast_context():
        return (
            torch.amp.autocast('cuda', dtype=autocast_dtype, enabled=autocast_enabled)
            if autocast_enabled
            else nullcontext()
        )

    with torch.no_grad():
        model.eval()
        for batch in tqdm(dataloader, desc="Running inference", leave=False):
            if "x" not in batch or "y" not in batch:
                raise KeyError("Inference batch must include 'x' and 'y'")

            stage_stats["raw_predictors"].add(batch["x"].numpy())
            target_canonical_cpu = _canonicalize_target_tensor(
                batch["y"].float(), target_vars, target_unit_provenance
            )
            target_cpu = target_canonical_cpu.numpy()
            for var_idx, var_name in enumerate(target_vars):
                stage_stats_by_var["targets"][var_name].add(target_cpu[:, var_idx])

            batch = {
                key: value.to(device=device, dtype=torch.float32, non_blocking=True)
                for key, value in batch.items()
            }

            normalized_x = _normalize_predictors_for_diagnostics(batch, model)
            stage_stats["normalized_predictors"].add(normalized_x.detach().cpu().numpy())

            out, out_pre_inverse, out_raw_model = infer_batch_ensemble(
                model=model,
                batch=batch,
                infer_batch=infer_batch_with_boundary_mitigation,
                boundary_cfg=boundary_cfg,
                head_type=head_type,
                ensemble_size=ensemble_size,
                base_seed=base_seed,
                device=device,
                autocast_context=autocast_context,
                force_float32=FORCE_OUTPUT_FLOAT32,
            )

            out_cpu = out.detach().cpu()
            out_pre_inverse_cpu = out_pre_inverse.detach().cpu()

            predictions.append(out_cpu)
            pre_inverse_predictions.append(out_pre_inverse_cpu)
            targets.append(target_canonical_cpu)

            stage_stats["raw_model_outputs_pre_inverse"].add(out_pre_inverse_cpu.numpy())
            stage_stats["inverse_outputs"].add(out_cpu.numpy())
            for var_idx, var_name in enumerate(target_vars):
                if out_pre_inverse_cpu.ndim == 5:
                    pre_inverse_var = out_pre_inverse_cpu[:, :, var_idx].numpy()
                    output_var = out_cpu[:, :, var_idx].numpy()
                else:
                    pre_inverse_var = out_pre_inverse_cpu[:, var_idx].numpy()
                    output_var = out_cpu[:, var_idx].numpy()
                stage_stats_by_var["raw_model_outputs_pre_inverse"][var_name].add(pre_inverse_var)
                stage_stats_by_var["inverse_outputs"][var_name].add(output_var)

            if residual_mode:
                truth_physical = target_canonical_cpu.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                baseline_model = _runtime_model_with_attribute(
                    model, "get_last_diffusion_baseline"
                )
                baseline_getter = getattr(
                    baseline_model, "get_last_diffusion_baseline", None
                )
                if not callable(baseline_getter):
                    raise RuntimeError(
                        "Residual diagnostics require the model to expose its "
                        "direct deterministic baseline; subtraction-based recovery "
                        "is intentionally unsupported."
                    )
                direct_baseline = baseline_getter()
                if direct_baseline is None:
                    raise RuntimeError(
                        "Residual sampling did not publish a direct deterministic "
                        "baseline for the current batch."
                    )
                baseline_physical, baseline_standardized = direct_baseline
                residual_stages = residual_transformation_stages(
                    model=model,
                    truth_physical=truth_physical,
                    final_physical=out.float(),
                    full_standardized=out_pre_inverse.float(),
                    generated_residual_standardized=out_raw_model.float(),
                    baseline_standardized=baseline_standardized.float(),
                    baseline_physical=baseline_physical.float(),
                )
                baseline_predictions.append(
                    residual_stages["physical_unet_prediction"]
                    .detach()
                    .cpu()
                    .float()
                )
                for stage_name, stage_tensor in residual_stages.items():
                    stage_cpu = stage_tensor.detach().cpu().float()
                    stage_array = stage_cpu.numpy()
                    stage_stats[stage_name].add(stage_array)
                    for var_idx, var_name in enumerate(target_vars):
                        per_variable = (
                            stage_array[:, :, var_idx]
                            if stage_array.ndim == 5
                            else stage_array[:, var_idx]
                        )
                        stage_stats_by_var[stage_name][var_name].add(per_variable)

                    if transformation_samples_remaining > 0:
                        take = min(
                            transformation_samples_remaining,
                            int(stage_array.shape[0]),
                        )
                        transformation_sample_parts.setdefault(stage_name, []).append(
                            stage_array[:take].copy()
                        )
                if transformation_samples_remaining > 0:
                    transformation_samples_remaining -= min(
                        transformation_samples_remaining,
                        int(target_canonical_cpu.shape[0]),
                    )

    if not predictions:
        raise RuntimeError("Inference produced no predictions.")

    return {
        "predictions": torch.cat(predictions, dim=0),
        "predictions_pre_inverse": torch.cat(pre_inverse_predictions, dim=0),
        "baseline_predictions": (
            torch.cat(baseline_predictions, dim=0)
            if baseline_predictions
            else None
        ),
        "targets": torch.cat(targets, dim=0),
        "stage_stats": {key: value.finalize() for key, value in stage_stats.items()},
        "stage_stats_by_var": {
            stage: {name: stats.finalize() for name, stats in per_var.items()}
            for stage, per_var in stage_stats_by_var.items()
        },
        "transformation_stage_samples": {
            stage: np.concatenate(parts, axis=0)
            for stage, parts in transformation_sample_parts.items()
            if parts
        },
    }


def _concat_time_coordinate(paths, time_key):
    arrays = []
    attrs = None
    encoding = None
    for path in paths:
        with xr.open_dataset(path) as ds:
            if time_key not in ds.coords and time_key not in ds.data_vars:
                raise KeyError(f"Time dimension '{time_key}' missing in {path}")
            da = ds[time_key]
            arrays.append(da)
            if attrs is None:
                attrs = dict(da.attrs)
            if encoding is None and hasattr(da, "encoding"):
                encoding = dict(da.encoding)
    combined = xr.concat(arrays, dim=time_key)
    if attrs:
        combined.attrs.update(attrs)
    if encoding:
        combined.encoding.update(encoding)
    return combined.load()


def _netcdf_engine() -> str:
    """Return 'h5netcdf' if h5py is available, otherwise 'netcdf4'."""
    try:
        import h5py  # noqa: F401
        return "h5netcdf"
    except ImportError:
        return "netcdf4"


def _netcdf_safe_attr_value(value: object) -> object:
    """Convert boolean metadata to standard NetCDF-compatible integer values."""
    if isinstance(value, (bool, np.bool_)):
        return np.int8(value)
    if isinstance(value, np.ndarray) and value.dtype.kind == "b":
        return value.astype(np.int8)
    if isinstance(value, (list, tuple)):
        array = np.asarray(value)
        if array.dtype.kind == "b":
            return array.astype(np.int8)
    return value


def _netcdf_safe_attrs(attrs: dict[str, object]) -> dict[str, object]:
    return {key: _netcdf_safe_attr_value(value) for key, value in attrs.items()}


def _sanitize_data_attrs(attrs: dict[str, object]) -> dict[str, object]:
    clean_attrs = _netcdf_safe_attrs(attrs)
    for key in ("scale_factor", "add_offset", "_FillValue", "missing_value", "dtype"):
        clean_attrs.pop(key, None)
    return clean_attrs


def _sanitize_dataset_attrs_for_netcdf(dataset: xr.Dataset) -> xr.Dataset:
    """Normalize global, data-variable, and coordinate metadata before saving."""
    dataset.attrs = _netcdf_safe_attrs(dict(dataset.attrs))
    for variable_name in dataset.variables:
        dataset[variable_name].attrs = _netcdf_safe_attrs(
            dict(dataset[variable_name].attrs)
        )
    return dataset


def _build_netcdf_encoding(var_names: list[str]) -> dict[str, dict[str, object]]:
    fill = np.float32(np.nan)
    return {
        name: {
            "dtype": NETCDF_OUTPUT_DTYPE,
            "_FillValue": fill,
        }
        for name in var_names
    }


def _build_var_array_map(var_names: list[str], values: np.ndarray) -> dict[str, np.ndarray]:
    return variable_array_map(var_names, values)


def _compute_quantization_summary(var_values: dict[str, np.ndarray]) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for var_name, values in var_values.items():
        summary[var_name] = quantization_detector(
            values,
            time_window=DIAGNOSTIC_TIME_WINDOW,
            spatial_window=DIAGNOSTIC_SPATIAL_WINDOW,
            round_decimals=DIAGNOSTIC_ROUND_DECIMALS,
        )
    return summary


def _build_diagnostic_rows(
    stage_stats: dict[str, dict[str, object]],
    stage_stats_by_var: dict[str, dict[str, dict[str, object]]],
    quant_summary: dict[str, dict[str, dict[str, object]]],
) -> list[dict[str, object]]:
    rows = [
        {"stage": "raw_predictors", "var": "all", "stats": stage_stats["raw_predictors"], "quantization": {}},
        {
            "stage": "normalized_predictors",
            "var": "all",
            "stats": stage_stats["normalized_predictors"],
            "quantization": {},
        },
    ]
    preferred_order = (
        "physical_unet_prediction",
        "physical_ground_truth",
        "physical_true_residual",
        "normalized_true_residual",
        "predicted_normalized_residual",
        "denormalized_predicted_residual",
        "final_physical_prediction",
        "raw_model_outputs_pre_inverse",
        "inverse_outputs",
        "targets",
    )
    ordered_stages = [name for name in preferred_order if name in stage_stats_by_var]
    ordered_stages.extend(
        name for name in stage_stats_by_var if name not in ordered_stages
    )
    for stage_name in ordered_stages:
        for var_name, stats in stage_stats_by_var.get(stage_name, {}).items():
            rows.append(
                {
                    "stage": stage_name,
                    "var": var_name,
                    "stats": stats,
                    "quantization": quant_summary.get(stage_name, {}).get(var_name, {}),
                }
            )
    return rows



def _repair_predictor_files_if_needed(
    predictor_paths,
    *,
    output_root: Path,
    predictor_var_names,
    run_index: int,
):
    resolved_paths = [str(Path(path).resolve()) for path in predictor_paths]
    if not REPAIR_INVALID_INPUTS:
        print("[repair] Predictor invalid-value repair disabled by REPAIR_INVALID_INPUTS=False")
        return resolved_paths

    repaired_dir = output_root / "_repaired_predictors" / f"run_{run_index:02d}"
    repaired_dir.mkdir(parents=True, exist_ok=True)

    final_paths: list[str] = []
    for predictor_path in resolved_paths:
        source_path = Path(predictor_path)
        with xr.open_dataset(source_path) as ds:
            predictor_ds = ds.load()

        before_counts = summarize_invalid_counts_xr(predictor_ds, var_names=predictor_var_names)
        invalid_before_total = sum(before_counts.values())

        if invalid_before_total > 0:
            print("Invalid predictor values detected (NaN/inf/fill). Auto-repair enabled.")
            predictor_ds.encoding["source"] = str(source_path)
            repaired_ds = repair_invalid_by_nearest_xr(
                predictor_ds,
                var_names=predictor_var_names,
            )
            after_counts = summarize_invalid_counts_xr(repaired_ds, var_names=predictor_var_names)

            repaired_path = repaired_dir / source_path.name
            repaired_ds.to_netcdf(repaired_path, engine=_netcdf_engine())
            final_paths.append(str(repaired_path.resolve()))
        else:
            after_counts = before_counts
            final_paths.append(str(source_path))

        keys = sorted(set(before_counts) | set(after_counts))
        for key in keys:
            print(
                f"[repair] {source_path.name} {key}: "
                f"invalid_count_before={before_counts.get(key, 0)} "
                f"invalid_count_after={after_counts.get(key, 0)}"
            )

    return final_paths

def _build_dataloader(config, predictor_paths, target_paths, device):
    base_dataset = build_inference_dataset(config, predictor_paths, target_paths)
    dataset = CordexWrappedDataset(base_dataset)
    batch_size = getattr(config, "batch_size", 1)
    num_workers = config.dl_num_workers
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )


def _apply_twophase_phase2_overrides(config: object) -> bool:
    """Apply Phase 2 overrides to a twophase config snapshot for inference.

    The resolved config snapshot is written before training, so it captures
    Phase 1 defaults (head_type=deterministic, original loss weights).  When
    the loaded checkpoint is a Phase 2 (diffusion) artifact, apply the same
    overrides that _apply_phase_overrides uses during training so that
    validate_checkpoint_compatibility sees a matching config.

    Returns True if overrides were applied, False if this is not a twophase run.
    """
    import copy

    two_phase = getattr(config, "two_phase_training", None)
    if two_phase is None:
        return False
    phase2_cfg = (
        two_phase.get("phase2", {}) if isinstance(two_phase, dict)
        else getattr(two_phase, "phase2", {}) or {}
    )
    if not phase2_cfg:
        return False

    # Phase 2 always uses the diffusion head with the frozen U-Net baseline.
    config.model.head_type = "diffusion"

    training_raw = getattr(config, "training", None)
    if hasattr(training_raw, "__dict__"):
        training_dict = dict(vars(training_raw))
    elif isinstance(training_raw, dict):
        training_dict = dict(training_raw)
    else:
        training_dict = {}
    training_dict["freeze_deterministic_baseline"] = True
    config.training = training_dict

    # Deep-merge the Phase 2 loss overrides (e.g. disable corrected_mean_weight).
    phase2_loss = (
        phase2_cfg.get("loss", {}) if isinstance(phase2_cfg, dict)
        else getattr(phase2_cfg, "loss", {}) or {}
    )
    if phase2_loss:
        base_loss = getattr(config, "loss", {})
        if isinstance(base_loss, dict):
            merged = copy.deepcopy(base_loss)
        elif hasattr(base_loss, "__dict__"):
            merged = copy.deepcopy(vars(base_loss))
        else:
            merged = {}
        for key, value in (phase2_loss if isinstance(phase2_loss, dict) else vars(phase2_loss)).items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        config.loss = merged

    print("[config] Applied two-phase Phase 2 overrides: head_type=diffusion, freeze_deterministic_baseline=True")
    return True


def _load_model_and_config() -> tuple[
    str, Path, Path, object, torch.nn.Module, Path, str, int, int, dict[str, object]
]:
    runs_root = _select_runs_root(RUNS_ROOT, FINETUNE_RUN_NAME)
    if runs_root != RUNS_ROOT:
        print(f"[runs] Using fallback runs root: {runs_root} (preferred {RUNS_ROOT})")

    base_params = UserParams(
        repo_root=REPO_ROOT,
        project_dir=PROJECT_DIR,
        runs_root=runs_root,
        config_path=CONFIG_PATH,
        inference_run_name=FINETUNE_RUN_NAME,
        preferred_checkpoint=PREFERRED_CHECKPOINT,
        device_target=DEVICE_TARGET,
        batch_size=INFERENCE_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        use_static=USE_STATIC,
    )

    validate_paths(base_params)
    run_name, run_dir = resolve_existing_run_dir(base_params)
    manifest = load_run_manifest(run_dir)
    resolved_config_path = Path(manifest["config_snapshot"]).resolve()
    # A checkpoint must be reconstructed from the immutable resolved snapshot
    # written for that run. The editable base YAML may have changed since
    # training and previously could silently select different scalers/scheduler.
    config_source_path = resolved_config_path

    os.chdir(PROJECT_DIR)
    config = get_config(str(config_source_path))
    assert_no_eccc_reference(config_source_path)
    print(f"[config] source={config_source_path}")

    manifest_scalars = manifest.get("scalars", {})
    if isinstance(manifest_scalars, dict):
        model_scaler_map = {
            "model.input_mu": "input_mu",
            "model.input_sigma": "input_sigma",
            "model.target_mu": "target_mu",
            "model.target_sigma": "target_sigma",
        }
        for key, attr in model_scaler_map.items():
            path = manifest_scalars.get(key)
            if path:
                setattr(config.model, attr, path)
        data_scalers = manifest_scalars.get("data.scalers")
        if data_scalers:
            config.data.scalers = data_scalers

    if NUM_WORKERS is not None:
        config.dl_num_workers = int(NUM_WORKERS)
    if INFERENCE_BATCH_SIZE is not None:
        config.batch_size = int(INFERENCE_BATCH_SIZE)
    config.device_target = DEVICE_TARGET
    if USE_STATIC is not None:
        config.data.use_static = bool(USE_STATIC)
    if STATIC_PATH:
        config.data.static_path = str(Path(STATIC_PATH).resolve())

    residual_alpha_provenance = _apply_residual_alpha_runtime_override(config)
    print(
        "[runtime] residual_application_scale="
        f"{residual_alpha_provenance['value']:g} "
        f"source={residual_alpha_provenance['source']}"
    )

    checkpoint_path = run_dir / "checkpoints" / "last.ckpt"
    if not checkpoint_path.exists():
        checkpoint_path = resolve_checkpoint(base_params, run_dir)
    assert_no_eccc_reference(checkpoint_path)

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    # The config snapshot is written before training, capturing Phase 1 defaults.
    # Promote it to Phase 2 semantics so the compatibility check matches the
    # Phase 2 checkpoint (diffusion head, frozen baseline, adjusted loss weights).
    _apply_twophase_phase2_overrides(config)

    # Residual diffusion has an exact, versioned contract (normalization,
    # residual sign/space, score parameterization, and sampler). Validate that
    # contract before constructing or restoring a model so an edited YAML can
    # never silently reinterpret trained weights.
    validate_checkpoint_compatibility(checkpoint, config)

    model = get_finetune_model_UNET(config)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model_state = model.state_dict()
    weights_have_module_prefix = all(key.startswith("module.") for key in state_dict.keys())
    model_expects_module_prefix = all(key.startswith("module.") for key in model_state.keys())

    if model_expects_module_prefix and not weights_have_module_prefix:
        state_dict = state_dict.__class__((("module." + key), value) for key, value in state_dict.items())
    elif weights_have_module_prefix and not model_expects_module_prefix:
        prefix_len = len("module.")
        state_dict = state_dict.__class__((key[prefix_len:], value) for key, value in state_dict.items())

    scaler_key_parts = (
        "input_scalers_",
        "output_scalers_",
        "static_input_scalers_",
        "static_output_scalers_",
    )
    state_dict = state_dict.__class__(
        (key, value)
        for key, value in state_dict.items()
        if not any(part in key for part in scaler_key_parts)
    )
    expected_non_scaler = {
        key for key in model_state if not any(part in key for part in scaler_key_parts)
    }
    checkpoint_keys = set(state_dict)
    missing = sorted(expected_non_scaler - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - expected_non_scaler)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model mismatch after filtering scaler tensors. "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )

    # Scaler tensors are intentionally reconstructed from the immutable scalar
    # files whose SHA-256 values were validated above. Fill only those known
    # keys, then require a genuinely strict state-dict restoration.
    strict_state_dict = state_dict.__class__(state_dict)
    for key, value in model_state.items():
        if any(part in key for part in scaler_key_parts):
            strict_state_dict[key] = value
    model.load_state_dict(strict_state_dict, strict=True)
    head_type = infer_head_type(checkpoint=checkpoint, config=config, model=model)
    ensemble_size = resolve_ensemble_size(config, head_type=head_type)
    base_seed = resolve_base_seed(config)
    print(f"[inference] head_type={head_type}, ensemble_size={ensemble_size}, base_seed={base_seed}")

    model_provenance = _log_checkpoint_identity(
        checkpoint_path, checkpoint, config_source_path
    )
    model_provenance["normalization_scalars"] = _normalization_scalar_provenance(
        config, checkpoint
    )
    model_provenance["runtime_overrides"] = {
        "residual_alpha": residual_alpha_provenance
    }
    model_provenance["diffusion_runtime"] = _diffusion_runtime_provenance(
        model,
        head_type=head_type,
        ensemble_size=ensemble_size,
        base_seed=base_seed,
        checkpoint=checkpoint,
        inference_batch_size=int(config.batch_size),
        residual_alpha_provenance=residual_alpha_provenance,
    )

    return (
        run_name,
        run_dir,
        checkpoint_path,
        config,
        model,
        runs_root,
        head_type,
        ensemble_size,
        base_seed,
        model_provenance,
    )


def _log_checkpoint_identity(
    checkpoint_path: Path, checkpoint: dict, config_snapshot_path: Path
) -> dict[str, object]:
    """Print the validated checkpoint's path, content hash, and saved identity."""
    resolved = checkpoint_path.resolve()
    checkpoint_sha256 = _sha256_path(resolved)
    config_snapshot_path = config_snapshot_path.resolve(strict=True)
    config_snapshot_sha256 = _sha256_path(config_snapshot_path)
    metadata = checkpoint.get("metadata", {}) if isinstance(checkpoint, dict) else {}
    source_control = metadata.get("source_control", {})
    print(f"[ckpt] path={resolved}")
    print(f"[ckpt] sha256={checkpoint_sha256}")
    print(
        "[ckpt] "
        f"epoch={checkpoint.get('epoch', 'n/a')} "
        f"global_step={checkpoint.get('global_step', checkpoint.get('step', 'n/a'))} "
        f"contract={metadata.get('config_fingerprint_sha256', 'n/a')} "
        f"git_commit={metadata.get('git_commit', 'n/a')}"
    )
    print(
        f"[config] sha256={config_snapshot_sha256} path={config_snapshot_path}"
    )
    return {
        "checkpoint_path": str(resolved),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": (
            checkpoint.get("epoch", metadata.get("epoch_index"))
            if isinstance(checkpoint, dict)
            else metadata.get("epoch_index")
        ),
        "checkpoint_global_step": (
            checkpoint.get(
                "global_step", checkpoint.get("step", metadata.get("global_step"))
            )
            if isinstance(checkpoint, dict)
            else metadata.get("global_step")
        ),
        "config_snapshot_path": str(config_snapshot_path),
        "config_snapshot_sha256": config_snapshot_sha256,
        "config_fingerprint_sha256": metadata.get("config_fingerprint_sha256"),
        "git_commit": metadata.get("git_commit") or (
            source_control.get("commit") if isinstance(source_control, dict) else None
        ),
        "git_branch": (
            source_control.get("branch", metadata.get("git_branch"))
            if isinstance(source_control, dict)
            else metadata.get("git_branch")
        ),
        "git_dirty": (
            source_control.get("dirty", metadata.get("git_dirty"))
            if isinstance(source_control, dict)
            else metadata.get("git_dirty")
        ),
        "checkpoint_metadata_schema": metadata.get("schema"),
        "checkpoint_metadata_schema_version": metadata.get("schema_version"),
        "checkpoint_created_utc": metadata.get("created_utc"),
        "checkpoint_head_type": metadata.get("head_type"),
    }


def _save_run_outputs(
    *,
    idx: int,
    output_root: Path,
    prediction_output_stub: Path,
    outputs_np: np.ndarray,
    outputs_pre_inverse_np: np.ndarray,
    baseline_outputs_np: np.ndarray | None,
    target_vars: list[str],
    predictor_paths: list[str],
    target_template_paths: list[str],
    time_dim: str,
    lat_dim: str,
    lon_dim: str,
    lat_name: str,
    lon_name: str,
    head_type: str,
    ensemble_size: int,
    base_seed: int,
    diagnostics_payload: dict,
    predicted_var_values: dict[str, np.ndarray],
    predicted_pre_inverse_var_values: dict[str, np.ndarray],
    target_var_values: dict[str, np.ndarray],
    transformation_stage_samples: dict[str, np.ndarray],
    target_unit_provenance: dict[str, dict[str, object]],
    boundary_provenance: dict[str, object],
    model_provenance: dict[str, object],
) -> None:
    import pickle

    baseline_output_path = output_root / prediction_output_stub.with_suffix(
        ".baseline.nc"
    )
    if baseline_outputs_np is not None:
        diagnostics_payload["full_domain_baseline_artifact"] = {
            "path": str(baseline_output_path.resolve()),
            "shape": list(baseline_outputs_np.shape),
            "ensemble_dimension": False,
        }

    if transformation_stage_samples:
        stage_artifact_path = output_root / prediction_output_stub.with_suffix(
            ".residual_stages.npz"
        )
        np.savez_compressed(stage_artifact_path, **transformation_stage_samples)
        diagnostics_payload["transformation_stage_sample_artifact"] = {
            "path": str(stage_artifact_path.resolve()),
            "sample_count": int(
                next(iter(transformation_stage_samples.values())).shape[0]
            ),
            "stages": list(transformation_stage_samples),
        }
        print(
            f"[diag] Saved residual transformation-stage samples to "
            f"{stage_artifact_path}"
        )

    if SAVE_DIAGNOSTICS_JSON:
        diagnostics_path = output_root / prediction_output_stub.with_suffix(".diagnostics.json")
        save_json(diagnostics_payload, diagnostics_path)
        print(f"[diag] Saved diagnostics JSON to {diagnostics_path}")

    if SAVE_DISTRIBUTION_PLOT:
        if "tasmax" in predicted_var_values and "pr" in predicted_var_values:
            plot_path = output_root / prediction_output_stub.with_suffix(".distribution.png")
            save_distribution_plot(
                target_values=target_var_values,
                predicted_values=predicted_var_values,
                predicted_raw_values=predicted_pre_inverse_var_values,
                output_path=plot_path,
            )
            print(f"[diag] Saved distribution comparison plot to {plot_path}")
        else:
            print("[diag] Skipped distribution plot; expected vars 'pr' and 'tasmax' were not both present.")

    predictor_time_coord = _concat_time_coordinate(predictor_paths, time_dim)
    if outputs_np.shape[0] != predictor_time_coord.sizes[time_dim]:
        raise ValueError(
            f"Prediction time dimension {outputs_np.shape[0]} does not match "
            f"predictor timestamps {predictor_time_coord.sizes[time_dim]}"
        )

    if not target_template_paths:
        raise FileNotFoundError("Target template paths missing; ensure config.data.test_target_paths is set")

    with xr.open_dataset(target_template_paths[0], engine=_netcdf_engine()) as template_ds:
        template_attrs = dict(template_ds.attrs)
        target_attrs = {
            name: _sanitize_data_attrs(dict(template_ds[name].attrs))
            for name in target_vars
            if name in template_ds.data_vars
        }
        lat_coord = template_ds[lat_name].load()
        lon_coord = template_ds[lon_name].load()

    target_attrs = _canonical_target_attrs(target_attrs, target_unit_provenance)
    output_contract = _output_runtime_contract(target_vars, target_unit_provenance)
    scalar_provenance = model_provenance.get("normalization_scalars", {})
    diffusion_runtime = model_provenance.get("diffusion_runtime", {})
    sampling_provenance = (
        diffusion_runtime.get("sampling", {})
        if isinstance(diffusion_runtime, dict)
        else {}
    )
    residual_contract = (
        diffusion_runtime.get("residual_contract", {})
        if isinstance(diffusion_runtime, dict)
        else {}
    )
    score_network = (
        diffusion_runtime.get("score_network", {})
        if isinstance(diffusion_runtime, dict)
        else {}
    )
    runtime_attrs: dict[str, object] = {
        "model_provenance": json.dumps(model_provenance, sort_keys=True),
        "target_unit_provenance": json.dumps(
            target_unit_provenance, sort_keys=True
        ),
        "output_variable_order": json.dumps(output_contract["variable_order"]),
        "canonical_output_units": json.dumps(
            output_contract["canonical_units"], sort_keys=True
        ),
        "output_numeric_storage": output_contract["numeric_storage"],
        "normalization_scalar_provenance": json.dumps(
            scalar_provenance, sort_keys=True
        ),
        "diffusion_runtime_provenance": json.dumps(
            diffusion_runtime, sort_keys=True
        ),
        "inference_boundary_provenance": json.dumps(
            boundary_provenance, sort_keys=True
        ),
        "boundary_mode": boundary_provenance["mode"],
        "boundary_enabled": boundary_provenance["enabled"],
        "boundary_force_full_frame": boundary_provenance["force_full_frame"],
        "boundary_tile_size": json.dumps(boundary_provenance["tile_size"]),
        "boundary_overlap": json.dumps(boundary_provenance["overlap"]),
        "boundary_halo": json.dumps(boundary_provenance["halo"]),
        "boundary_tile_origin": json.dumps(boundary_provenance["tile_origin"]),
        "boundary_tile_stride": json.dumps(boundary_provenance["tile_stride"]),
    }
    for key in (
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_epoch",
        "checkpoint_global_step",
        "config_snapshot_path",
        "config_snapshot_sha256",
        "config_fingerprint_sha256",
        "git_commit",
        "git_branch",
        "git_dirty",
        "checkpoint_metadata_schema",
        "checkpoint_metadata_schema_version",
        "checkpoint_created_utc",
        "checkpoint_head_type",
    ):
        value = model_provenance.get(key)
        if value is not None:
            runtime_attrs[key] = value
    if isinstance(scalar_provenance, dict):
        for scalar_name, record in scalar_provenance.items():
            if not isinstance(record, dict):
                continue
            for field in ("path", "sha256", "size_bytes"):
                value = record.get(field)
                if value is not None:
                    runtime_attrs[f"scalar_{scalar_name}_{field}"] = value
    if isinstance(diffusion_runtime, dict):
        runtime_attrs["diffusion_base_seed"] = int(
            diffusion_runtime.get("base_seed", base_seed)
        )
        runtime_attrs["diffusion_ensemble_size"] = int(
            diffusion_runtime.get("ensemble_size", ensemble_size)
        )
        runtime_attrs["diffusion_member_seeds"] = json.dumps(
            diffusion_runtime.get("member_seeds", [base_seed])
        )
        inference_batch_size = diffusion_runtime.get("inference_batch_size")
        if inference_batch_size is not None:
            runtime_attrs["inference_batch_size"] = int(inference_batch_size)
        runtime_attrs["diffusion_rng_stream_scope"] = diffusion_runtime.get(
            "rng_stream_scope", "unspecified"
        )
        runtime_attrs["diffusion_batch_size_invariant"] = bool(
            diffusion_runtime.get("batch_size_invariant", False)
        )
        runtime_attrs["diffusion_exact_reproduction_requires_same_batching"] = bool(
            diffusion_runtime.get(
                "exact_reproduction_requires_same_batching", True
            )
        )
    if isinstance(sampling_provenance, dict):
        for field in (
            "sde",
            "beta_min",
            "beta_max",
            "sigma_min",
            "sigma_max",
            "num_scales",
            "continuous",
            "sampling_method",
            "eta",
            "predictor",
            "corrector",
            "num_sampling_steps",
            "prediction_type",
        ):
            value = sampling_provenance.get(field)
            if value is not None:
                runtime_attrs[f"diffusion_{field}"] = value
    if isinstance(residual_contract, dict):
        residual_attr_names = {
            "enabled": "residual_diffusion",
            "definition": "residual_definition",
            "sign": "residual_sign",
            "space": "residual_space",
            "reconstruction": "residual_reconstruction",
            "residual_application_scale": "residual_application_scale",
            "alpha": "residual_alpha",
            "alpha_source": "residual_alpha_source",
            "alpha_environment_variable": "residual_alpha_environment_variable",
            "alpha_override_is_set": "residual_alpha_override_is_set",
        }
        for field in (
            "enabled",
            "definition",
            "sign",
            "space",
            "reconstruction",
            "residual_application_scale",
            "alpha",
            "alpha_source",
            "alpha_environment_variable",
            "alpha_override_is_set",
        ):
            value = residual_contract.get(field)
            if value is not None:
                runtime_attrs[residual_attr_names[field]] = value
    if isinstance(score_network, dict):
        padding_mode = score_network.get("configured_padding_mode")
        if padding_mode is not None:
            runtime_attrs["score_padding_mode"] = padding_mode
        runtime_attrs["score_effective_padded_conv_modes"] = json.dumps(
            score_network.get("effective_padded_conv_modes", [])
        )
        for field in ("conv2d_count", "padded_conv2d_count"):
            value = score_network.get(field)
            if value is not None:
                runtime_attrs[f"score_{field}"] = value

    coords = {time_dim: predictor_time_coord, lat_dim: lat_coord, lon_dim: lon_coord}
    prediction_ds = xr.Dataset(coords=coords)
    prediction_ds.attrs.update(template_attrs)
    prediction_ds.attrs.update(runtime_attrs)
    if baseline_outputs_np is not None:
        prediction_ds.attrs.update(
            {
                "baseline_file": baseline_output_path.name,
            }
        )
    prediction_ds = add_predictions_to_dataset(
        prediction_ds=prediction_ds,
        target_vars=target_vars,
        outputs_np=outputs_np,
        coords=coords,
        time_dim=time_dim,
        lat_dim=lat_dim,
        lon_dim=lon_dim,
        target_attrs=target_attrs,
        head_type=head_type,
        ensemble_size=ensemble_size,
        base_seed=base_seed,
    )
    prediction_output_path = output_root / prediction_output_stub
    prediction_ds = _sanitize_dataset_attrs_for_netcdf(prediction_ds)
    prediction_ds.to_netcdf(
        prediction_output_path,
        engine=_netcdf_engine(),
        encoding=_build_netcdf_encoding(target_vars),
    )
    print(
        f"[{idx + 1:02d}/{NUM_RUNS}] Saved predictions to {prediction_output_path} "
        f"({predictor_time_coord.values[0]} -> {predictor_time_coord.values[-1]})"
    )

    if baseline_outputs_np is not None:
        expected_baseline_shape = (
            outputs_np.shape[0],
            len(target_vars),
            int(lat_coord.size),
            int(lon_coord.size),
        )
        if tuple(baseline_outputs_np.shape) != expected_baseline_shape:
            raise ValueError(
                "Deterministic baseline shape does not match output coordinates: "
                f"{tuple(baseline_outputs_np.shape)} != {expected_baseline_shape}."
            )
        baseline_ds = xr.Dataset(coords=coords)
        baseline_ds.attrs.update(template_attrs)
        baseline_ds.attrs.update(runtime_attrs)
        baseline_ds.attrs.update(
            {
                "head_type": "deterministic",
                "baseline_role": (
                    "joint_residual_diffusion_deterministic_baseline"
                ),
                "paired_corrected_prediction": prediction_output_path.name,
            }
        )
        baseline_ds = add_predictions_to_dataset(
            prediction_ds=baseline_ds,
            target_vars=target_vars,
            outputs_np=baseline_outputs_np,
            coords=coords,
            time_dim=time_dim,
            lat_dim=lat_dim,
            lon_dim=lon_dim,
            target_attrs=target_attrs,
            head_type="deterministic",
            ensemble_size=1,
            base_seed=base_seed,
        )
        baseline_ds = _sanitize_dataset_attrs_for_netcdf(baseline_ds)
        baseline_ds.to_netcdf(
            baseline_output_path,
            engine=_netcdf_engine(),
            encoding=_build_netcdf_encoding(target_vars),
        )
        print(
            f"[{idx + 1:02d}/{NUM_RUNS}] Saved jointly trained deterministic "
            f"baseline to {baseline_output_path}"
        )

    pickle_path = output_root / prediction_output_stub.with_suffix(".pkl")
    with open(pickle_path, "wb") as handle:
        pickle.dump(outputs_np, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[{idx + 1:02d}/{NUM_RUNS}] Saved raw predictions array to {pickle_path}")

    pre_inverse_pickle_path = output_root / prediction_output_stub.with_suffix(".pre_inverse.pkl")
    with open(pre_inverse_pickle_path, "wb") as handle:
        pickle.dump(outputs_pre_inverse_np, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"[{idx + 1:02d}/{NUM_RUNS}] Saved pre-inverse predictions array to "
        f"{pre_inverse_pickle_path}"
    )


def main() -> None:
    global DIST_ENABLED, DIST_RANK, DIST_WORLD_SIZE, DIST_LOCAL_RANK

    logging.disable(logging.CRITICAL)
    warnings.simplefilter(action="ignore", category=FutureWarning)

    _check_list_lengths()

    DIST_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
    DIST_RANK = int(os.environ.get("RANK", "0"))
    DIST_LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
    DIST_ENABLED = DIST_WORLD_SIZE > 1

    if DIST_ENABLED and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    if DIST_ENABLED:
        print(
            f"[dist] inference enabled: rank={DIST_RANK}/{DIST_WORLD_SIZE}, "
            f"local_rank={DIST_LOCAL_RANK}"
        )
    else:
        _configure_idle_gpus()

    torch.jit.enable_onednn_fusion(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False  # deterministic disables benchmark; inference doesn't need it
        torch.cuda.manual_seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    device = _select_device()
    (
        run_name,
        run_dir,
        checkpoint_path,
        config,
        model,
        runs_root_used,
        head_type,
        ensemble_size,
        base_seed,
        model_provenance,
    ) = _load_model_and_config()
    model.to(device)
    if USE_TORCH_COMPILE and hasattr(torch, "compile"):
        print("[compile] Compiling model with torch.compile (mode='reduce-overhead')…")
        model = torch.compile(model, mode="reduce-overhead")
        print("[compile] Done.")
    scaler_dtype_summary = _collect_scaler_dtype_summary(model)
    print(f"[diag] scaler dtypes: {scaler_dtype_summary}")
    print(
        f"[diag] mixed_precision_enabled={ENABLE_MIXED_PRECISION}, "
        f"force_output_float32={FORCE_OUTPUT_FLOAT32}"
    )

    _full = range(DIST_RANK, NUM_RUNS, DIST_WORLD_SIZE) if DIST_ENABLED else range(NUM_RUNS)
    run_indices = [i for i in _full if RUN_RANGE is None or RUN_RANGE[0] <= i < RUN_RANGE[1]]
    print(f"[runs] executing indices {list(run_indices)} of {NUM_RUNS} total")
    save_future: Future | None = None
    with ThreadPoolExecutor(max_workers=1) as save_executor:
        for idx in run_indices:
            test_split = TEST_SPLITS[idx]
            predictor_file = PREDICTOR_FILES[idx]
            target_file = TARGET_FILES[idx]
            prediction_output_name = PREDICTION_OUTPUT_NAMES[idx]
            inference_output_root = _case_inference_root(INFERENCE_OUTPUT_ROOTS[idx], run_dir)

            if predictor_file:
                test_predictors = [DATASET_ROOT / test_split / predictor_file]
            else:
                test_predictors = sorted((DATASET_ROOT / test_split).glob("*.nc"))
            test_targets = [DATASET_ROOT / target_file]

            params = UserParams(
                repo_root=REPO_ROOT,
                project_dir=PROJECT_DIR,
                runs_root=runs_root_used,
                config_path=CONFIG_PATH,
                inference_run_name=run_name,
                inference_output_root=inference_output_root,
                inference_predictor_root=DATASET_ROOT / test_split,
                test_predictor_paths=test_predictors,
                test_target_paths=test_targets,
                use_static=USE_STATIC,
                checkpoint_path=None,
                preferred_checkpoint=PREFERRED_CHECKPOINT,
                device_target=DEVICE_TARGET,
                num_workers=NUM_WORKERS,
                batch_size=INFERENCE_BATCH_SIZE,
            )

            validate_paths(params, require_inference=True)

            test_predictor_paths = _resolve_inputs(params.test_predictor_paths) if params.test_predictor_paths else []
            if not test_predictor_paths and params.inference_predictor_root:
                test_predictor_paths = [
                    str(path.resolve()) for path in Path(params.inference_predictor_root).glob("*.nc")
                ]
            if not test_predictor_paths:
                raise FileNotFoundError("No inference predictor files configured in USER PARAMETERS.")

            output_root = Path(params.inference_output_root or (run_dir / "predictions"))
            output_root.mkdir(parents=True, exist_ok=True)

            predictor_var_names = build_predictor_names(config)
            test_predictor_paths = _repair_predictor_files_if_needed(
                test_predictor_paths,
                output_root=output_root,
                predictor_var_names=predictor_var_names,
                run_index=idx + 1,
            )

            config.data.test_predictor_paths = test_predictor_paths
            config.data.test_target_paths = _resolve_inputs(params.test_target_paths)

            prediction_output_stub = Path(
                prediction_output_name or f"{run_name}_predictions_{idx + 1:02d}.nc"
            )

            print(f"[{idx + 1:02d}/{NUM_RUNS}] Using predictors from {Path(test_predictor_paths[0]).parent}")
            print(f"[{idx + 1:02d}/{NUM_RUNS}] Output directory: {output_root}")

            test_dl = _build_dataloader(config, config.data.test_predictor_paths, config.data.test_target_paths, device)
            effective_inference_batch_size = int(test_dl.batch_size or 1)
            diffusion_runtime = model_provenance.get("diffusion_runtime", {})
            if isinstance(diffusion_runtime, dict):
                recorded_batch_size = diffusion_runtime.get("inference_batch_size")
                if (
                    recorded_batch_size is not None
                    and int(recorded_batch_size) != effective_inference_batch_size
                ):
                    raise RuntimeError(
                        "Runtime DataLoader batch size differs from recorded diffusion "
                        f"provenance: {effective_inference_batch_size} != "
                        f"{recorded_batch_size}."
                    )
                diffusion_runtime["inference_batch_size"] = (
                    effective_inference_batch_size
                )

            base_dataset = test_dl.dataset.base
            target_vars = list(base_dataset.target_vars)
            predictand_specs = resolve_predictand_specs(config, target_vars)
            predictor_paths = list(base_dataset.predictor_paths)
            target_template_paths = list(base_dataset.target_paths)
            target_unit_provenance = _resolve_target_unit_provenance(
                target_template_paths, target_vars
            )
            output_contract = _output_runtime_contract(
                target_vars, target_unit_provenance
            )
            print(
                "[units] "
                + ", ".join(
                    f"{name}: {conversion['source_units']} x "
                    f"{conversion['conversion_factor']:g} -> "
                    f"{conversion['canonical_units']}"
                    for name, conversion in target_unit_provenance.items()
                )
            )

            boundary_cfg = resolve_boundary_mitigation_settings(config)
            boundary_provenance = _boundary_runtime_provenance(boundary_cfg)
            if boundary_provenance["mode"] == "full_frame":
                reason = (
                    "force_full_frame=True"
                    if boundary_provenance["force_full_frame"]
                    else "boundary_mitigation.enabled=False"
                )
                print(f"[boundary] mode=full_frame ({reason}; tiling/blending disabled)")
            else:
                print(
                    f"[boundary] mode=overlap_tiled, tile_size={boundary_cfg.tile_size}, "
                    f"overlap={boundary_cfg.overlap}, halo={boundary_cfg.halo}, "
                    f"tile_origin={boundary_cfg.tile_origin}, "
                    f"blend_mode={boundary_cfg.blend_mode}, "
                    f"deblock_enabled={boundary_cfg.deblock.enabled}"
                )

            params_json = export_params(
                params,
                output_root / prediction_output_stub.with_suffix(".json"),
                extra={
                    "run_name": run_name,
                    "checkpoint": str(checkpoint_path),
                    "model_provenance": model_provenance,
                    "target_unit_provenance": target_unit_provenance,
                    "output_contract": output_contract,
                    "boundary_provenance": boundary_provenance,
                    "inference_batch_size": effective_inference_batch_size,
                    "runtime_overrides": model_provenance.get(
                        "runtime_overrides", {}
                    ),
                },
            )
            print(f"[{idx + 1:02d}/{NUM_RUNS}] Saved parameter snapshot to {params_json}")

            # Wait for previous run's save before starting GPU inference (frees memory).
            if save_future is not None:
                save_future.result()
                save_future = None

            inference_result = _run_full_inference(
                test_dl,
                model,
                device,
                target_vars,
                boundary_cfg,
                target_unit_provenance,
                head_type=head_type,
                ensemble_size=ensemble_size,
                base_seed=base_seed,
            )
            full_outputs = inference_result["predictions"]
            full_outputs_pre_inverse = inference_result["predictions_pre_inverse"]
            full_baseline_outputs = inference_result["baseline_predictions"]
            full_targets = inference_result["targets"]

            outputs_np = full_outputs.numpy().astype(np.float32, copy=False)
            outputs_pre_inverse_np = full_outputs_pre_inverse.numpy().astype(np.float32, copy=False)
            baseline_outputs_np = (
                full_baseline_outputs.numpy().astype(np.float32, copy=False)
                if full_baseline_outputs is not None
                else None
            )
            targets_np = full_targets.numpy().astype(np.float32, copy=False)
            output_var_axis = 2 if outputs_np.ndim == 5 else 1
            if outputs_np.shape[output_var_axis] != len(target_vars):
                raise ValueError(
                    f"Model produced {outputs_np.shape[output_var_axis]} channels but target_vars expects {len(target_vars)}"
                )

            predicted_var_values = _build_var_array_map(target_vars, outputs_np)
            predicted_pre_inverse_var_values = _build_var_array_map(target_vars, outputs_pre_inverse_np)
            target_var_values = _build_var_array_map(target_vars, targets_np)

            assert_nonnegative_outputs(predicted_var_values, predictand_specs, eps=1e-8)

            quantization_summary = {
                "raw_model_outputs_pre_inverse": _compute_quantization_summary(predicted_pre_inverse_var_values),
                "inverse_outputs": _compute_quantization_summary(predicted_var_values),
                "targets": _compute_quantization_summary(target_var_values),
            }
            diagnostic_rows = _build_diagnostic_rows(
                inference_result["stage_stats"],
                inference_result["stage_stats_by_var"],
                quantization_summary,
            )
            print(f"[diag] Quantization summary for run {idx + 1:02d}")
            print(format_compact_table(diagnostic_rows))

            diagnostics_payload = {
                "run_index": idx + 1,
                "num_runs": NUM_RUNS,
                "predictor_paths": test_predictor_paths,
                "target_template_paths": target_template_paths,
                "stage_stats": inference_result["stage_stats"],
                "stage_stats_by_var": inference_result["stage_stats_by_var"],
                "quantization_detector": quantization_summary,
                "scaler_dtypes": scaler_dtype_summary,
                "mixed_precision_enabled": bool(ENABLE_MIXED_PRECISION),
                "force_output_float32": bool(FORCE_OUTPUT_FLOAT32),
                "head_type": head_type,
                "ensemble_size": int(ensemble_size),
                "base_seed": int(base_seed),
                "inference_batch_size": effective_inference_batch_size,
                "runtime_overrides": model_provenance.get("runtime_overrides", {}),
                "model_provenance": model_provenance,
                "normalization_scalar_provenance": model_provenance.get(
                    "normalization_scalars", {}
                ),
                "diffusion_runtime_provenance": model_provenance.get(
                    "diffusion_runtime", {}
                ),
                "residual_contract": model_provenance.get(
                    "diffusion_runtime", {}
                ).get("residual_contract", {}),
                "target_unit_provenance": target_unit_provenance,
                "output_contract": output_contract,
                "boundary_provenance": boundary_provenance,
                "predictands": {name: spec.to_dict() for name, spec in predictand_specs.items()},
            }
            time_dim = base_dataset.time_dim or "time"
            lat_dim, lon_dim = base_dataset.output_spatial_dims
            lat_name = base_dataset.fine_lat_name
            lon_name = base_dataset.fine_lon_name

            save_future = save_executor.submit(
                _save_run_outputs,
                idx=idx,
                output_root=output_root,
                prediction_output_stub=prediction_output_stub,
                outputs_np=outputs_np,
                outputs_pre_inverse_np=outputs_pre_inverse_np,
                baseline_outputs_np=baseline_outputs_np,
                target_vars=target_vars,
                predictor_paths=predictor_paths,
                target_template_paths=target_template_paths,
                time_dim=time_dim,
                lat_dim=lat_dim,
                lon_dim=lon_dim,
                lat_name=lat_name,
                lon_name=lon_name,
                head_type=head_type,
                ensemble_size=ensemble_size,
                base_seed=base_seed,
                diagnostics_payload=diagnostics_payload,
                predicted_var_values=predicted_var_values,
                predicted_pre_inverse_var_values=predicted_pre_inverse_var_values,
                target_var_values=target_var_values,
                transformation_stage_samples=inference_result[
                    "transformation_stage_samples"
                ],
                target_unit_provenance=target_unit_provenance,
                boundary_provenance=boundary_provenance,
                model_provenance=model_provenance,
            )
            print(f"[{idx + 1:02d}/{NUM_RUNS}] Save dispatched to background thread.")

        # Wait for the last run's save to complete.
        if save_future is not None:
            save_future.result()

    if DIST_ENABLED and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
