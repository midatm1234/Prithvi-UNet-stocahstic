#!/usr/bin/env python
"""Strict, calendar-aware evaluation of CORDEX refinement ensembles.

This evaluator consumes the physical ensemble-member NetCDF written by the
two-phase CORDEX workflow, the matching Phase-1 baseline file, and an explicit
ground-truth file.  It never guesses a target from a training configuration and
never aligns records by array position.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import xarray as xr

from granitewxc.refinement.diagnostics import boundary_region_masks, empirical_crps

from granitewxc.refinement.metrics import (
    ErrorAccumulator as _ErrorAccumulator,
    MapMean as _MapMean,
    correlation as _correlation,
    map_difference_metrics as _map_difference_metrics,
)

from granitewxc.refinement.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    REFINEMENT_CONTRACT_VERSION,
)


SUPPORTED_REFINEMENT_TYPES = (
    "diffusion_unet",
    "diffusion_transformer",
    "flow_matching_unet",
    "flow_matching_transformer",
)


class RefinementEvaluationError(ValueError):
    """Raised when an input violates the scientific evaluation contract."""


@dataclass(frozen=True)
class RefinementOutputSelection:
    member_dim: str
    variables: dict[str, str]
    refinement_type: str
    refinement_case: str
    refinement_checkpoint: str
    checkpoint_schema_version: int
    refinement_contract_version: int
    residual_contract: str
    refinement_contract_fingerprint: str
    refinement_checkpoint_sha256: str
    phase1_checkpoint: str
    phase1_fingerprint: str
    residual_normalizer_state_fingerprint: str
    residual_normalization: dict[str, Any]
    ensemble_size: int


@dataclass(frozen=True)
class AlignedDatasets:
    prediction: xr.Dataset
    baseline: xr.Dataset
    target: xr.Dataset
    report: dict[str, Any]


@dataclass(frozen=True)
class UnitConversion:
    source_units: str
    target_units: str
    factor: float = 1.0
    offset: float = 0.0
    reason: str = "units already match"

    @property
    def applied(self) -> bool:
        return self.factor != 1.0 or self.offset != 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_units": self.source_units,
            "target_units": self.target_units,
            "factor": float(self.factor),
            "offset": float(self.offset),
            "applied": bool(self.applied),
            "reason": self.reason,
        }


def _require_attr(dataset: xr.Dataset, name: str, source: str) -> str:
    value = str(dataset.attrs.get(name, "")).strip()
    if not value:
        raise RefinementEvaluationError(
            f"{source} is missing required global attribute {name!r}."
        )
    return value


def _require_int_attr(dataset: xr.Dataset, name: str, source: str) -> int:
    value = _require_attr(dataset, name, source)
    try:
        return int(value)
    except ValueError as exc:
        raise RefinementEvaluationError(
            f"{source} attribute {name!r} must be an integer, got {value!r}."
        ) from exc


def _require_coordinate(dataset: xr.Dataset, name: str, source: str) -> None:
    if name not in dataset.coords:
        raise RefinementEvaluationError(
            f"{source} is missing required coordinate {name!r}."
        )
    if dataset[name].dims != (name,):
        raise RefinementEvaluationError(
            f"{source} coordinate {name!r} must be one-dimensional; "
            f"got dimensions {dataset[name].dims}."
        )
    if dataset.sizes[name] == 0:
        raise RefinementEvaluationError(
            f"{source} coordinate {name!r} is empty."
        )


def _member_variable_name(dataset: xr.Dataset, variable: str) -> str:
    candidates = [
        name
        for name in (variable, f"{variable}_members")
        if name in dataset
    ]
    if len(candidates) != 1:
        raise RefinementEvaluationError(
            f"Refinement output must contain exactly one member field for "
            f"{variable!r}; found {candidates or 'none'}."
        )
    return candidates[0]


def _resolved_path_text(value: str) -> str:
    return str(Path(value).expanduser().resolve(strict=False)).casefold()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    text = str(value or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise RefinementEvaluationError(
            f"{label} must be a 64-character hexadecimal SHA-256 digest, "
            f"got {text!r}."
        )
    return text


def _contract_fingerprint(contract: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            contract,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RefinementEvaluationError(
            "Expected checkpoint contains a non-canonical refinement contract."
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _state_fingerprint(state: Mapping[str, Any]) -> str:
    """Match the schema-v2 checkpoint tensor fingerprint exactly."""
    import torch

    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if not torch.is_tensor(value):
            raise RefinementEvaluationError(
                f"Expected checkpoint state {key!r} is not a tensor."
            )
        digest.update(str(key).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(
            value.detach()
            .to("cpu")
            .contiguous()
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
    return digest.hexdigest()


def _load_expected_checkpoint(path: Path) -> Mapping[str, Any]:
    import torch

    try:
        try:
            payload = torch.load(str(path), map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - PyTorch < 2.0
            payload = torch.load(str(path), map_location="cpu")
    except Exception as exc:
        raise RefinementEvaluationError(
            f"Expected refinement checkpoint is not a readable PyTorch payload: {path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise RefinementEvaluationError(
            "Expected refinement checkpoint payload must be a mapping."
        )
    return payload


def _numeric_vector(
    normalization: Mapping[str, Any],
    key: str,
    *,
    channel_count: int,
) -> np.ndarray:
    value = normalization.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != channel_count:
        raise RefinementEvaluationError(
            f"residual_normalization.{key} must contain exactly one value per "
            f"output channel ({channel_count}), got {value!r}."
        )
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RefinementEvaluationError(
            f"residual_normalization.{key} must be numeric."
        ) from exc
    if array.shape != (channel_count,) or not np.isfinite(array).all():
        raise RefinementEvaluationError(
            f"residual_normalization.{key} must contain only finite values."
        )
    return array


def _validate_checkpoint_and_normalization(
    *,
    checkpoint_payload: Mapping[str, Any],
    checkpoint_schema: int,
    refinement_type: str,
    residual_contract: str,
    contract_fingerprint: str,
    phase1_checkpoint: str,
    phase1_fingerprint: str,
    residual_normalization: Mapping[str, Any],
    variables: Sequence[str],
) -> str:
    """Authenticate output metadata against schema-v2 checkpoint contents."""
    import torch

    checkpoint_kind = str(checkpoint_payload.get("checkpoint_kind", ""))
    if checkpoint_kind not in {"refinement", "combined"}:
        raise RefinementEvaluationError(
            "Expected checkpoint must be a schema-v2 refinement or combined payload, "
            f"got checkpoint_kind={checkpoint_kind!r}."
        )
    if checkpoint_payload.get("checkpoint_schema_version") != checkpoint_schema:
        raise RefinementEvaluationError(
            "Output and expected checkpoint schema versions do not match."
        )
    if checkpoint_payload.get("refinement_type") != refinement_type:
        raise RefinementEvaluationError(
            "Output refinement_type does not match expected checkpoint metadata."
        )

    contract = checkpoint_payload.get("refinement_contract")
    if not isinstance(contract, Mapping):
        raise RefinementEvaluationError(
            "Expected checkpoint does not contain a refinement_contract mapping."
        )
    computed_contract_fingerprint = _contract_fingerprint(contract)
    checkpoint_contract_fingerprint = _require_sha256(
        checkpoint_payload.get("refinement_contract_fingerprint"),
        label="checkpoint refinement_contract_fingerprint",
    )
    if checkpoint_contract_fingerprint != computed_contract_fingerprint:
        raise RefinementEvaluationError(
            "Expected checkpoint refinement contract fingerprint does not "
            "authenticate its stored contract."
        )
    if contract_fingerprint != checkpoint_contract_fingerprint:
        raise RefinementEvaluationError(
            "Output refinement contract fingerprint does not match the expected "
            "checkpoint."
        )
    if int(contract.get("contract_version", -1)) != REFINEMENT_CONTRACT_VERSION:
        raise RefinementEvaluationError(
            "Expected checkpoint has an unsupported refinement contract version."
        )
    if contract.get("refinement_type") != refinement_type:
        raise RefinementEvaluationError(
            "Expected checkpoint contract and refinement_type metadata disagree."
        )
    if contract.get("residual_contract") != residual_contract:
        raise RefinementEvaluationError(
            "Output residual contract does not match the expected checkpoint."
        )

    checkpoint_phase1 = _require_sha256(
        checkpoint_payload.get("phase1_fingerprint"),
        label="checkpoint phase1_fingerprint",
    )
    if phase1_fingerprint != checkpoint_phase1:
        raise RefinementEvaluationError(
            "Output Phase-1 fingerprint does not match the expected checkpoint."
        )
    checkpoint_phase1_path = str(
        checkpoint_payload.get("phase1_checkpoint", "")
    ).strip()
    if checkpoint_phase1_path and _resolved_path_text(
        checkpoint_phase1_path
    ) != _resolved_path_text(phase1_checkpoint):
        raise RefinementEvaluationError(
            "Output Phase-1 checkpoint path does not match expected checkpoint metadata."
        )

    names = residual_normalization.get("channel_names")
    expected_names = [str(variable) for variable in variables]
    if not isinstance(names, list) or [str(name) for name in names] != expected_names:
        raise RefinementEvaluationError(
            "residual_normalization.channel_names must exactly match requested "
            f"output variables and order: expected {expected_names!r}, got {names!r}."
        )
    if residual_normalization.get("fitted") is not True:
        raise RefinementEvaluationError(
            "residual_normalization.fitted must be the boolean true."
        )
    method = str(residual_normalization.get("method", ""))
    if method not in {"standardize", "identity"}:
        raise RefinementEvaluationError(
            f"Unsupported residual_normalization.method {method!r}."
        )
    normalizer_contract = contract.get("residual_normalization")
    if not isinstance(normalizer_contract, Mapping):
        raise RefinementEvaluationError(
            "Expected checkpoint contract has no residual_normalization mapping."
        )
    if method != str(normalizer_contract.get("method", "")):
        raise RefinementEvaluationError(
            "Output residual-normalization method does not match the checkpoint."
        )
    for key in ("epsilon", "minimum_scale"):
        try:
            value = float(residual_normalization[key])
            expected = float(normalizer_contract[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise RefinementEvaluationError(
                f"residual_normalization.{key} must be a finite numeric value."
            ) from exc
        if not math.isfinite(value) or value < 0.0 or value != expected:
            raise RefinementEvaluationError(
                f"residual_normalization.{key} does not match the checkpoint contract."
            )
    if residual_normalization.get("require_fitted") is not bool(
        normalizer_contract.get("require_fitted")
    ):
        raise RefinementEvaluationError(
            "residual_normalization.require_fitted does not match the checkpoint."
        )

    mean = _numeric_vector(
        residual_normalization, "mean", channel_count=len(expected_names)
    )
    scale = _numeric_vector(
        residual_normalization, "scale", channel_count=len(expected_names)
    )
    count = _numeric_vector(
        residual_normalization, "count", channel_count=len(expected_names)
    )
    minimum_scale = float(normalizer_contract["minimum_scale"])
    if np.any(scale <= 0.0) or np.any(scale < minimum_scale):
        raise RefinementEvaluationError(
            "residual_normalization.scale must be finite, positive, and no smaller "
            "than the checkpoint minimum_scale."
        )
    if method == "standardize":
        if np.any(count <= 0.0):
            raise RefinementEvaluationError(
                "Standardized residual_normalization.count must be finite and positive."
            )
        if np.any(count != np.floor(count)):
            raise RefinementEvaluationError(
                "Standardized residual_normalization.count must contain sample counts."
            )
    if method == "identity" and (
        np.any(count != 0.0) or np.any(mean != 0.0) or np.any(scale != 1.0)
    ):
        raise RefinementEvaluationError(
            "Identity residual normalization must record count=0, mean=0, scale=1."
        )

    resolved = checkpoint_payload.get("resolved_config")
    if isinstance(resolved, Mapping):
        data = resolved.get("data")
        if isinstance(data, Mapping) and data.get("output_vars") is not None:
            checkpoint_names = [str(name) for name in data["output_vars"]]
            if checkpoint_names != expected_names:
                raise RefinementEvaluationError(
                    "Requested output variables/order do not match expected checkpoint "
                    f"configuration: {expected_names!r} != {checkpoint_names!r}."
                )

    model_state = checkpoint_payload.get("model")
    if not isinstance(model_state, Mapping):
        raise RefinementEvaluationError("Expected checkpoint has no model state mapping.")
    normalizer_state = {
        str(key): value
        for key, value in model_state.items()
        if str(key).startswith("residual_normalizer.")
    }
    recorded_keys = checkpoint_payload.get("residual_normalizer_state_keys")
    if not isinstance(recorded_keys, (list, tuple)) or sorted(
        str(key) for key in recorded_keys
    ) != sorted(normalizer_state):
        raise RefinementEvaluationError(
            "Expected checkpoint residual-normalizer state keys do not match its model state."
        )
    computed_state_fingerprint = _state_fingerprint(normalizer_state)
    recorded_state_fingerprint = _require_sha256(
        checkpoint_payload.get("residual_normalizer_state_fingerprint"),
        label="checkpoint residual_normalizer_state_fingerprint",
    )
    if recorded_state_fingerprint != computed_state_fingerprint:
        raise RefinementEvaluationError(
            "Expected checkpoint residual-normalizer state fingerprint does not "
            "authenticate its stored tensors."
        )

    if method == "standardize":
        required = {
            f"residual_normalizer.{name}" for name in ("mean", "scale", "count", "fitted")
        }
        if not required.issubset(normalizer_state):
            raise RefinementEvaluationError(
                "Expected checkpoint is missing standardized residual-normalizer tensors."
            )
        checkpoint_vectors = {
            key: normalizer_state[f"residual_normalizer.{key}"]
            .detach()
            .to(device="cpu", dtype=torch.float64)
            .reshape(-1)
            .numpy()
            for key in ("mean", "scale", "count")
        }
        for key, output_values in (("mean", mean), ("scale", scale), ("count", count)):
            if checkpoint_vectors[key].shape != output_values.shape or not np.array_equal(
                checkpoint_vectors[key], output_values
            ):
                raise RefinementEvaluationError(
                    f"Output residual_normalization.{key} does not match expected "
                    "checkpoint tensors."
                )
        fitted = normalizer_state["residual_normalizer.fitted"]
        if fitted.numel() != 1 or not bool(fitted.detach().cpu().bool().item()):
            raise RefinementEvaluationError(
                "Expected checkpoint residual normalizer is not fitted."
            )
    elif normalizer_state:
        raise RefinementEvaluationError(
            "Identity residual normalization must not persist normalizer tensors."
        )
    return recorded_state_fingerprint


def validate_refinement_output(
    dataset: xr.Dataset,
    *,
    variables: Sequence[str],
    expected_refinement_type: str | None = None,
    expected_checkpoint: str | Path | None = None,
    expected_case: str | None = None,
) -> RefinementOutputSelection:
    """Strictly select physical member variables and verify provenance."""
    source = "refinement output"
    for coordinate in ("time", "lat", "lon"):
        _require_coordinate(dataset, coordinate, source)

    prediction_kind = _require_attr(dataset, "prediction_kind", source)
    if prediction_kind != "refinement_ensemble_members":
        raise RefinementEvaluationError(
            "Expected prediction_kind='refinement_ensemble_members', got "
            f"{prediction_kind!r}. Refusing to evaluate a baseline or "
            "normalized-space artifact as a physical refinement ensemble."
        )

    refinement_type = _require_attr(dataset, "refinement_type", source)
    if refinement_type not in SUPPORTED_REFINEMENT_TYPES:
        raise RefinementEvaluationError(
            f"Unsupported refinement_type {refinement_type!r}; expected one "
            "of "
            f"{SUPPORTED_REFINEMENT_TYPES}."
        )
    if (
        expected_refinement_type
        and refinement_type != expected_refinement_type
    ):
        raise RefinementEvaluationError(
            f"Selected output is {refinement_type!r}, not requested "
            f"{expected_refinement_type!r}."
        )

    refinement_case = _require_attr(dataset, "refinement_case", source)
    checkpoint = _require_attr(dataset, "refinement_checkpoint", source)
    checkpoint_schema = _require_int_attr(
        dataset, "checkpoint_schema_version", source
    )
    if checkpoint_schema != CHECKPOINT_SCHEMA_VERSION:
        raise RefinementEvaluationError(
            f"Unsupported refinement checkpoint schema {checkpoint_schema}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION}."
        )
    contract_version = _require_int_attr(
        dataset, "refinement_contract_version", source
    )
    if contract_version != REFINEMENT_CONTRACT_VERSION:
        raise RefinementEvaluationError(
            f"Unsupported refinement contract version {contract_version}; "
            f"expected {REFINEMENT_CONTRACT_VERSION}."
        )
    residual_contract = _require_attr(dataset, "residual_contract", source)
    if residual_contract != "physical_ground_truth_minus_phase1_v1":
        raise RefinementEvaluationError(
            "Refinement output does not use the required physical residual "
            f"contract: {residual_contract!r}."
        )
    contract_fingerprint = _require_sha256(
        _require_attr(dataset, "refinement_contract_fingerprint", source),
        label="refinement_contract_fingerprint",
    )
    checkpoint_sha256 = _require_sha256(
        _require_attr(dataset, "refinement_checkpoint_sha256", source),
        label="refinement_checkpoint_sha256",
    )
    phase1_checkpoint = _require_attr(dataset, "phase1_checkpoint", source)
    phase1_fingerprint = _require_sha256(
        _require_attr(dataset, "phase1_fingerprint", source),
        label="phase1_fingerprint",
    )
    normalization_text = _require_attr(dataset, "residual_normalization", source)
    try:
        residual_normalization = json.loads(normalization_text)
    except json.JSONDecodeError as exc:
        raise RefinementEvaluationError(
            "refinement output residual_normalization is not valid JSON."
        ) from exc
    if not isinstance(residual_normalization, dict):
        raise RefinementEvaluationError(
            "refinement output residual_normalization must be a JSON object."
        )
    if expected_case and refinement_case != expected_case:
        raise RefinementEvaluationError(
            f"Selected output case {refinement_case!r} does not match "
            f"{expected_case!r}."
        )
    if expected_checkpoint is None:
        raise RefinementEvaluationError(
            "expected_checkpoint is required for schema-v2 output evaluation; "
            "embedded NetCDF provenance is not self-authenticating."
        )
    if _resolved_path_text(checkpoint) != _resolved_path_text(str(expected_checkpoint)):
        raise RefinementEvaluationError(
            "Selected output checkpoint does not match the requested "
            f"checkpoint: {checkpoint!r} != {str(expected_checkpoint)!r}."
        )
    expected_path = Path(expected_checkpoint).expanduser().resolve()
    if not expected_path.is_file():
        raise RefinementEvaluationError(
            f"Expected refinement checkpoint does not exist: {expected_path}"
        )
    actual_sha256 = _sha256_file(expected_path)
    if actual_sha256 != checkpoint_sha256:
        raise RefinementEvaluationError(
            "Selected output was not generated from the current bytes at "
            f"the expected checkpoint path: output={checkpoint_sha256}, "
            f"file={actual_sha256}."
        )
    checkpoint_payload = _load_expected_checkpoint(expected_path)
    normalizer_state_fingerprint = _validate_checkpoint_and_normalization(
        checkpoint_payload=checkpoint_payload,
        checkpoint_schema=checkpoint_schema,
        refinement_type=refinement_type,
        residual_contract=residual_contract,
        contract_fingerprint=contract_fingerprint,
        phase1_checkpoint=phase1_checkpoint,
        phase1_fingerprint=phase1_fingerprint,
        residual_normalization=residual_normalization,
        variables=variables,
    )

    member_dims = [
        name for name in ("ensemble", "member") if name in dataset.dims
    ]
    if len(member_dims) != 1:
        raise RefinementEvaluationError(
            "Refinement output must have exactly one 'ensemble' or 'member' "
            f"dimension; found {member_dims or 'none'}."
        )
    member_dim = member_dims[0]
    ensemble_size = int(dataset.sizes[member_dim])
    if ensemble_size < 1:
        raise RefinementEvaluationError("Refinement ensemble is empty.")
    declared_size = _require_attr(dataset, "ensemble_size", source)
    try:
        declared_size_int = int(declared_size)
    except ValueError as exc:
        raise RefinementEvaluationError(
            f"ensemble_size must be an integer, got {declared_size!r}."
        ) from exc
    if declared_size_int != ensemble_size:
        raise RefinementEvaluationError(
            f"ensemble_size attribute is {declared_size_int}, but dimension "
            f"{member_dim!r} has length {ensemble_size}."
        )

    selected: dict[str, str] = {}
    expected_dims = ("time", member_dim, "lat", "lon")
    for variable in variables:
        name = _member_variable_name(dataset, variable)
        if dataset[name].dims != expected_dims:
            raise RefinementEvaluationError(
                f"Member field {name!r} must have dimensions {expected_dims}, "
                f"got {dataset[name].dims}."
            )
        units = str(dataset[name].attrs.get("units", "")).strip()
        if not units:
            raise RefinementEvaluationError(
                f"Member field {name!r} has no physical units attribute."
            )
        selected[str(variable)] = name

    return RefinementOutputSelection(
        member_dim=member_dim,
        variables=selected,
        refinement_type=refinement_type,
        refinement_case=refinement_case,
        refinement_checkpoint=checkpoint,
        checkpoint_schema_version=checkpoint_schema,
        refinement_contract_version=contract_version,
        residual_contract=residual_contract,
        refinement_contract_fingerprint=contract_fingerprint,
        refinement_checkpoint_sha256=checkpoint_sha256,
        phase1_checkpoint=phase1_checkpoint,
        phase1_fingerprint=phase1_fingerprint,
        residual_normalizer_state_fingerprint=normalizer_state_fingerprint,
        residual_normalization=residual_normalization,
        ensemble_size=ensemble_size,
    )


def validate_baseline_output(
    dataset: xr.Dataset,
    *,
    variables: Sequence[str],
    refinement_case: str,
    phase1_checkpoint: str,
    phase1_fingerprint: str,
) -> None:
    source = "Phase-1 baseline output"
    for coordinate in ("time", "lat", "lon"):
        _require_coordinate(dataset, coordinate, source)
    kind = _require_attr(dataset, "prediction_kind", source)
    if kind != "phase1_unet_deterministic":
        raise RefinementEvaluationError(
            "Expected a Phase-1 baseline with "
            f"prediction_kind='phase1_unet_deterministic', got {kind!r}."
        )
    inference_case = str(dataset.attrs.get("inference_case", "")).strip()
    if inference_case and inference_case != refinement_case:
        raise RefinementEvaluationError(
            f"Baseline case {inference_case!r} does not match refinement case "
            f"{refinement_case!r}."
        )
    baseline_checkpoint = _require_attr(dataset, "phase1_checkpoint", source)
    baseline_fingerprint = _require_attr(dataset, "phase1_fingerprint", source)
    if _resolved_path_text(baseline_checkpoint) != _resolved_path_text(
        phase1_checkpoint
    ):
        raise RefinementEvaluationError(
            "Baseline and refinement output name different Phase-1 checkpoints: "
            f"{baseline_checkpoint!r} != {phase1_checkpoint!r}."
        )
    if baseline_fingerprint != phase1_fingerprint:
        raise RefinementEvaluationError(
            "Baseline and refinement output have different Phase-1 state "
            "fingerprints."
        )
    for variable in variables:
        if variable not in dataset:
            raise RefinementEvaluationError(
                f"Phase-1 baseline is missing variable {variable!r}."
            )
        if dataset[variable].dims != ("time", "lat", "lon"):
            raise RefinementEvaluationError(
                f"Baseline {variable!r} must have dimensions "
                "('time', 'lat', 'lon'), got "
                f"{dataset[variable].dims}."
            )
        if not str(dataset[variable].attrs.get("units", "")).strip():
            raise RefinementEvaluationError(
                f"Baseline variable {variable!r} has no units attribute."
            )


def validate_target(
    dataset: xr.Dataset,
    *,
    variables: Sequence[str],
) -> None:
    source = "ground-truth target"
    for coordinate in ("time", "lat", "lon"):
        _require_coordinate(dataset, coordinate, source)
    for variable in variables:
        if variable not in dataset:
            raise RefinementEvaluationError(
                f"Ground-truth target is missing variable {variable!r}."
            )
        if dataset[variable].dims != ("time", "lat", "lon"):
            raise RefinementEvaluationError(
                f"Target {variable!r} must have dimensions "
                "('time', 'lat', 'lon'), got "
                f"{dataset[variable].dims}."
            )
        if not str(dataset[variable].attrs.get("units", "")).strip():
            raise RefinementEvaluationError(
                f"Target variable {variable!r} has no units attribute."
            )


def _validate_same_spatial_grid(
    reference: xr.Dataset,
    candidate: xr.Dataset,
    source: str,
) -> None:
    for coordinate in ("lat", "lon"):
        expected = np.asarray(reference[coordinate].values)
        actual = np.asarray(candidate[coordinate].values)
        if expected.shape != actual.shape or not np.allclose(
            expected,
            actual,
            rtol=0.0,
            atol=1.0e-8,
            equal_nan=True,
        ):
            raise RefinementEvaluationError(
                f"{source} {coordinate} grid or orientation does not exactly "
                "match the refinement output."
            )


def _timestamp_key(value: Any) -> tuple[int, int, int, int, int, int, int]:
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            raise RefinementEvaluationError("Time coordinate contains NaT.")
        text = np.datetime_as_string(value, unit="us")
        match = re.fullmatch(
            r"(-?\d+)-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})\.(\d{6})",
            text,
        )
        if not match:
            raise RefinementEvaluationError(
                f"Could not parse NumPy timestamp {text!r}."
            )
        parts = tuple(int(part) for part in match.groups())
        return parts  # type: ignore[return-value]

    required = ("year", "month", "day")
    if all(hasattr(value, item) for item in required):
        return (
            int(value.year),
            int(value.month),
            int(value.day),
            int(getattr(value, "hour", 0)),
            int(getattr(value, "minute", 0)),
            int(getattr(value, "second", 0)),
            int(getattr(value, "microsecond", 0)),
        )
    raise RefinementEvaluationError(
        f"Unsupported timestamp value {value!r} ({type(value).__name__})."
    )


def _timestamp_text(key: tuple[int, ...]) -> str:
    return (
        f"{key[0]:04d}-{key[1]:02d}-{key[2]:02d}T"
        f"{key[3]:02d}:{key[4]:02d}:{key[5]:02d}.{key[6]:06d}"
    )


def _timestamp_keys(dataset: xr.Dataset, source: str) -> list[tuple[int, ...]]:
    keys = [_timestamp_key(value) for value in dataset["time"].values]
    if len(keys) != len(set(keys)):
        raise RefinementEvaluationError(
            f"{source} time coordinate contains duplicate timestamps."
        )
    return keys


def _calendar_name(dataset: xr.Dataset) -> str:
    time = dataset["time"]
    return str(
        time.encoding.get("calendar", time.attrs.get("calendar", "standard"))
    )


def align_exact_timestamps(
    prediction: xr.Dataset,
    baseline: xr.Dataset,
    target: xr.Dataset,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    require_all_prediction_times: bool = True,
) -> AlignedDatasets:
    """Align calendars by exact civil timestamps, never positional indices.

    Matching civil timestamps lets a no-leap prediction safely use a Gregorian
    target: leap days are explicitly omitted, while every post-February record
    still maps to its true date.
    """
    pred_keys = _timestamp_keys(prediction, "refinement output")
    base_keys = _timestamp_keys(baseline, "Phase-1 baseline")
    target_keys = _timestamp_keys(target, "ground-truth target")
    if pred_keys != base_keys:
        raise RefinementEvaluationError(
            "Phase-1 baseline timestamps do not exactly match the refinement "
            "output in both values and order."
        )

    target_index = {key: index for index, key in enumerate(target_keys)}
    selected_prediction: list[int] = []
    selected_target: list[int] = []
    missing: list[tuple[int, ...]] = []
    for index, key in enumerate(pred_keys):
        date_text = _timestamp_text(key)[:10]
        if start_date and date_text < start_date:
            continue
        if end_date and date_text > end_date:
            continue
        target_position = target_index.get(key)
        if target_position is None:
            missing.append(key)
            continue
        selected_prediction.append(index)
        selected_target.append(target_position)

    if missing and require_all_prediction_times:
        preview = ", ".join(_timestamp_text(key) for key in missing[:3])
        raise RefinementEvaluationError(
            f"Ground-truth target is missing {len(missing)} prediction "
            f"timestamps (first: {preview})."
        )
    if not selected_prediction:
        raise RefinementEvaluationError(
            "Prediction, baseline, and target have no exact timestamps "
            "in common."
        )

    common_keys = [pred_keys[index] for index in selected_prediction]
    report = {
        "matched_count": len(common_keys),
        "prediction_count": len(pred_keys),
        "baseline_count": len(base_keys),
        "target_count": len(target_keys),
        "prediction_dropped_count": len(pred_keys) - len(common_keys),
        "target_dropped_count": len(target_keys) - len(common_keys),
        "first_timestamp": _timestamp_text(common_keys[0]),
        "last_timestamp": _timestamp_text(common_keys[-1]),
        "prediction_calendar": _calendar_name(prediction),
        "baseline_calendar": _calendar_name(baseline),
        "target_calendar": _calendar_name(target),
        "alignment": "exact civil timestamp intersection",
    }
    return AlignedDatasets(
        prediction=prediction.isel(time=selected_prediction),
        baseline=baseline.isel(time=selected_prediction),
        target=target.isel(time=selected_target),
        report=report,
    )


def _unit_key(units: str) -> str:
    value = units.strip().lower().replace("−", "-").replace("⁻", "-")
    value = value.replace("**", "").replace("^", "")
    value = re.sub(r"\s+", "", value)
    return value


_PRECIP_DAILY_UNITS = {
    "mm/day",
    "mmday-1",
    "mmd-1",
    "mmperday",
}
_PRECIP_FLUX_UNITS = {
    "kgm-2s-1",
    "kg/m2/s",
    "kgm-2/sec",
    "mms-1",
    "mm/s",
}
_KELVIN_UNITS = {"k", "kelvin", "degreeskelvin"}
_CELSIUS_UNITS = {"degc", "c", "celsius", "degreecelsius", "degreescelsius"}


def _is_precipitation(variable: str, data: xr.DataArray | None = None) -> bool:
    name = variable.casefold()
    if name in {"pr", "precip", "precipitation", "ppt", "tp"}:
        return True
    standard_name = "" if data is None else str(
        data.attrs.get("standard_name", "")
    ).casefold()
    return "precipitation" in standard_name


def resolve_unit_conversion(
    source_units: str,
    target_units: str,
    *,
    variable: str,
    source: xr.DataArray | None = None,
) -> UnitConversion:
    source_key = _unit_key(source_units)
    target_key = _unit_key(target_units)
    if source_key == target_key:
        return UnitConversion(source_units, target_units)

    if _is_precipitation(variable, source):
        if (
            source_key in _PRECIP_FLUX_UNITS
            and target_key in _PRECIP_DAILY_UNITS
        ):
            return UnitConversion(
                source_units,
                target_units,
                factor=86400.0,
                reason="water flux converted from per-second to mm/day",
            )
        if (
            source_key in _PRECIP_DAILY_UNITS
            and target_key in _PRECIP_FLUX_UNITS
        ):
            return UnitConversion(
                source_units,
                target_units,
                factor=1.0 / 86400.0,
                reason=(
                    "precipitation converted from mm/day to per-second flux"
                ),
            )

    if source_key in _KELVIN_UNITS and target_key in _CELSIUS_UNITS:
        return UnitConversion(
            source_units,
            target_units,
            offset=-273.15,
            reason="Kelvin converted to degrees Celsius",
        )
    if source_key in _CELSIUS_UNITS and target_key in _KELVIN_UNITS:
        return UnitConversion(
            source_units,
            target_units,
            offset=273.15,
            reason="degrees Celsius converted to Kelvin",
        )

    raise RefinementEvaluationError(
        f"No scientifically defined conversion for {variable!r}: "
        f"{source_units!r} -> {target_units!r}."
    )


def convert_dataarray_units(
    data: xr.DataArray,
    target_units: str,
    *,
    variable: str,
) -> tuple[xr.DataArray, dict[str, Any]]:
    """Convert a field once based on its declared source and target units."""
    source_units = str(data.attrs.get("units", "")).strip()
    if not source_units:
        raise RefinementEvaluationError(
            f"Cannot convert {variable!r}: source units are missing."
        )
    conversion = resolve_unit_conversion(
        source_units,
        target_units,
        variable=variable,
        source=data,
    )
    converted = data * conversion.factor + conversion.offset
    converted.attrs = dict(data.attrs)
    converted.attrs["units"] = target_units
    converted.attrs["evaluation_unit_conversion"] = conversion.reason
    return converted, conversion.as_dict()






class _BoundedTripletSampler:
    def __init__(self, total_size: int, maximum: int) -> None:
        self.maximum = max(1, int(maximum))
        self.stride = None  # retained as an explicit legacy report field
        # A fixed linear stride can repeatedly select the same columns and
        # miss narrow perimeter artifacts. Uniform indices avoid that alias.
        self.positions = np.sort(np.random.default_rng(0).choice(
            int(total_size), size=min(int(total_size), self.maximum), replace=False
        ))
        self.seen = 0
        self.refined: list[np.ndarray] = []
        self.baseline: list[np.ndarray] = []
        self.target: list[np.ndarray] = []
        self.kept = 0

    def update(
        self,
        refined: np.ndarray,
        baseline: np.ndarray,
        target: np.ndarray,
    ) -> None:
        refined_flat = refined.reshape(-1)
        baseline_flat = baseline.reshape(-1)
        target_flat = target.reshape(-1)
        first = np.searchsorted(self.positions, self.seen)
        last = np.searchsorted(self.positions, self.seen + refined_flat.size)
        selected = self.positions[first:last] - self.seen
        valid = (np.isfinite(refined_flat[selected]) & np.isfinite(baseline_flat[selected])
                 & np.isfinite(target_flat[selected]))
        selected = selected[valid]
        if selected.size:
            self.refined.append(refined_flat[selected].astype(np.float64))
            self.baseline.append(baseline_flat[selected].astype(np.float64))
            self.target.append(target_flat[selected].astype(np.float64))
            self.kept += int(selected.size)
        self.seen += refined_flat.size

    def result(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        def combine(items: list[np.ndarray]) -> np.ndarray:
            if not items:
                return np.empty(0, dtype=np.float64)
            return np.concatenate(items)

        return (
            combine(self.refined),
            combine(self.baseline),
            combine(self.target),
        )


def _finite_mean(values: Iterable[float | None]) -> float | None:
    array = np.asarray(
        [value for value in values if value is not None],
        dtype=np.float64,
    )
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else None




def _daily_spatial_correlations(
    prediction: np.ndarray,
    target: np.ndarray,
) -> list[float | None]:
    return [
        _correlation(prediction[index], target[index])
        for index in range(prediction.shape[0])
    ]


def _daily_lag_one_autocorrelations(values: np.ndarray) -> list[float | None]:
    results: list[float | None] = []
    for field in values:
        horizontal = _correlation(field[:, :-1], field[:, 1:])
        vertical = _correlation(field[:-1, :], field[1:, :])
        results.append(_finite_mean((horizontal, vertical)))
    return results


def _gradient_fields(values: np.ndarray) -> np.ndarray:
    dx = values[..., :-1, 1:] - values[..., :-1, :-1]
    dy = values[..., 1:, :-1] - values[..., :-1, :-1]
    return np.hypot(dx, dy)


def _safe_ensemble_mean(members: np.ndarray) -> np.ndarray:
    finite = np.isfinite(members)
    count = finite.sum(axis=1)
    total = np.where(finite, members, 0.0).sum(axis=1)
    result = np.full(total.shape, np.nan, dtype=np.float64)
    np.divide(total, count, out=result, where=count > 0)
    return result


def _safe_ensemble_spread(members: np.ndarray) -> np.ndarray:
    mean = _safe_ensemble_mean(members)
    finite = np.isfinite(members)
    count = finite.sum(axis=1)
    squared = np.where(
        finite,
        np.square(members - mean[:, None]),
        0.0,
    ).sum(axis=1)
    spread = np.full(mean.shape, np.nan, dtype=np.float64)
    np.divide(squared, count - 1, out=spread, where=count > 1)
    return np.sqrt(spread)




def _radial_spectrum(field: np.ndarray, bins: int) -> dict[str, Any]:
    values = np.asarray(field, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return {
            "normalized_radial_frequency_edges": [],
            "power_fraction": [],
            "high_frequency_power_fraction": None,
        }
    fill = float(values[finite].mean())
    clean = np.where(finite, values, fill)
    clean -= clean.mean()
    height, width = clean.shape
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    normalized_radius = np.sqrt(fy * fy + fx * fx) / 0.5
    power = np.abs(np.fft.fft2(clean)) ** 2
    power[0, 0] = 0.0
    edges = np.linspace(0.0, math.sqrt(2.0), int(bins) + 1)
    bin_power = np.zeros(int(bins), dtype=np.float64)
    for index in range(int(bins)):
        if index == bins - 1:
            mask = (normalized_radius >= edges[index]) & (
                normalized_radius <= edges[index + 1]
            )
        else:
            mask = (normalized_radius >= edges[index]) & (
                normalized_radius < edges[index + 1]
            )
        bin_power[index] = float(power[mask].sum())
    total = float(power.sum())
    fractions = bin_power / total if total > 0.0 else np.zeros_like(bin_power)
    high = float(power[normalized_radius >= 0.5].sum())
    return {
        "normalized_radial_frequency_edges": [float(item) for item in edges],
        "power_fraction": [float(item) for item in fractions],
        "high_frequency_power_fraction": high / total if total > 0.0 else 0.0,
    }


def _spectral_comparison(
    prediction: np.ndarray,
    target: np.ndarray,
    bins: int,
) -> dict[str, Any]:
    predicted = _radial_spectrum(prediction, bins)
    observed = _radial_spectrum(target, bins)
    pred_fraction = np.asarray(predicted["power_fraction"], dtype=np.float64)
    target_fraction = np.asarray(observed["power_fraction"], dtype=np.float64)
    if pred_fraction.size and target_fraction.size:
        epsilon = 1.0e-12
        log_rmse = float(
            np.sqrt(
                np.mean(
                    np.square(
                        np.log10(pred_fraction + epsilon)
                        - np.log10(target_fraction + epsilon)
                    )
                )
            )
        )
    else:
        log_rmse = None
    return {
        "prediction": predicted,
        "target": observed,
        "log_power_fraction_rmse": log_rmse,
    }


def _quantile_metrics(
    refined: np.ndarray,
    baseline: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    levels = np.asarray((0.01, 0.05, 0.50, 0.95, 0.99))
    if target.size == 0:
        return {
            "sample_count": 0,
            "levels": [float(item) for item in levels],
            "ensemble_mean": [],
            "phase1": [],
            "target": [],
            "ensemble_mean_quantile_mae": None,
            "phase1_quantile_mae": None,
            "p95_error_ensemble_mean": None,
            "p99_error_ensemble_mean": None,
            "p95_error_phase1": None,
            "p99_error_phase1": None,
        }
    refined_q = np.quantile(refined, levels)
    baseline_q = np.quantile(baseline, levels)
    target_q = np.quantile(target, levels)
    return {
        "sample_count": int(target.size),
        "levels": [float(item) for item in levels],
        "ensemble_mean": [float(item) for item in refined_q],
        "phase1": [float(item) for item in baseline_q],
        "target": [float(item) for item in target_q],
        "ensemble_mean_quantile_mae": float(
            np.mean(np.abs(refined_q - target_q))
        ),
        "phase1_quantile_mae": float(
            np.mean(np.abs(baseline_q - target_q))
        ),
        "p95_error_ensemble_mean": float(refined_q[-2] - target_q[-2]),
        "p99_error_ensemble_mean": float(refined_q[-1] - target_q[-1]),
        "p95_error_phase1": float(baseline_q[-2] - target_q[-2]),
        "p99_error_phase1": float(baseline_q[-1] - target_q[-1]),
        "p99_observed_event_threshold": float(target_q[-1]),
        "p99_observed_event_frequency": float(np.mean(target >= target_q[-1])),
        "p99_predicted_event_frequency_ensemble_mean": float(np.mean(refined >= target_q[-1])),
        "p99_predicted_event_frequency_phase1": float(np.mean(baseline >= target_q[-1])),
        "observed_p99_event_rmse_ensemble_mean": float(np.sqrt(np.mean(
            np.square(refined[target >= target_q[-1]] - target[target >= target_q[-1]])
        ))),
        "observed_p99_event_rmse_phase1": float(np.sqrt(np.mean(
            np.square(baseline[target >= target_q[-1]] - target[target >= target_q[-1]])
        ))),
    }


def _wet_day_metrics(
    refined: np.ndarray,
    baseline: np.ndarray,
    target: np.ndarray,
    threshold: float,
) -> dict[str, float | None]:
    def summarize(values: np.ndarray) -> tuple[float | None, float | None]:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return None, None
        wet = finite >= threshold
        frequency = float(wet.mean())
        intensity = float(finite[wet].mean()) if wet.any() else None
        return frequency, intensity

    refined_frequency, refined_intensity = summarize(refined)
    baseline_frequency, baseline_intensity = summarize(baseline)
    target_frequency, target_intensity = summarize(target)
    return {
        "threshold": float(threshold),
        "ensemble_mean_frequency": refined_frequency,
        "phase1_frequency": baseline_frequency,
        "target_frequency": target_frequency,
        "ensemble_mean_intensity": refined_intensity,
        "phase1_intensity": baseline_intensity,
        "target_intensity": target_intensity,
        "ensemble_mean_frequency_error": (
            None
            if refined_frequency is None or target_frequency is None
            else refined_frequency - target_frequency
        ),
        "phase1_frequency_error": (
            None
            if baseline_frequency is None or target_frequency is None
            else baseline_frequency - target_frequency
        ),
        "ensemble_mean_intensity_error": (
            None
            if refined_intensity is None or target_intensity is None
            else refined_intensity - target_intensity
        ),
        "phase1_intensity_error": (
            None
            if baseline_intensity is None or target_intensity is None
            else baseline_intensity - target_intensity
        ),
    }


class _WetAccumulator:
    def __init__(self, threshold: float) -> None:
        self.threshold = float(threshold)
        self.finite_count = 0
        self.wet_count = 0
        self.wet_sum = 0.0

    def update(self, values: np.ndarray) -> None:
        finite = values[np.isfinite(values)]
        self.finite_count += int(finite.size)
        wet = finite >= self.threshold
        self.wet_count += int(wet.sum())
        if wet.any():
            self.wet_sum += float(finite[wet].sum())

    def result(self) -> tuple[float | None, float | None]:
        if self.finite_count == 0:
            return None, None
        frequency = self.wet_count / self.finite_count
        intensity = self.wet_sum / self.wet_count if self.wet_count else None
        return (
            float(frequency),
            None if intensity is None else float(intensity),
        )


class _WetContingencyAccumulator:
    """Accumulate paired wet/dry errors on one explicit validity support."""

    def __init__(self, threshold: float) -> None:
        self.threshold = float(threshold)
        self.valid_count = 0
        self.target_wet_count = 0
        self.target_dry_count = 0
        self.false_wet_count = 0
        self.missed_wet_count = 0

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        valid = np.isfinite(prediction) & np.isfinite(target)
        predicted_wet = prediction[valid] >= self.threshold
        target_wet = target[valid] >= self.threshold
        target_dry = ~target_wet
        self.valid_count += int(valid.sum())
        self.target_wet_count += int(target_wet.sum())
        self.target_dry_count += int(target_dry.sum())
        self.false_wet_count += int((predicted_wet & target_dry).sum())
        self.missed_wet_count += int((~predicted_wet & target_wet).sum())

    def result(self) -> dict[str, float | int | None]:
        return {
            "valid_count": self.valid_count,
            "target_wet_count": self.target_wet_count,
            "target_dry_count": self.target_dry_count,
            "false_wet_count": self.false_wet_count,
            "missed_wet_count": self.missed_wet_count,
            "false_wet_day_rate": (
                None
                if self.target_dry_count == 0
                else float(self.false_wet_count / self.target_dry_count)
            ),
            "missed_wet_day_rate": (
                None
                if self.target_wet_count == 0
                else float(self.missed_wet_count / self.target_wet_count)
            ),
        }


def _exact_wet_metrics(
    refined: _WetAccumulator,
    baseline: _WetAccumulator,
    target: _WetAccumulator,
    refined_contingency: _WetContingencyAccumulator,
    baseline_contingency: _WetContingencyAccumulator,
) -> dict[str, Any]:
    refined_frequency, refined_intensity = refined.result()
    baseline_frequency, baseline_intensity = baseline.result()
    target_frequency, target_intensity = target.result()

    def difference(first: float | None, second: float | None) -> float | None:
        return None if first is None or second is None else first - second

    return {
        "threshold": refined.threshold,
        "ensemble_mean_frequency": refined_frequency,
        "phase1_frequency": baseline_frequency,
        "target_frequency": target_frequency,
        "ensemble_mean_intensity": refined_intensity,
        "phase1_intensity": baseline_intensity,
        "target_intensity": target_intensity,
        "ensemble_mean_frequency_error": difference(
            refined_frequency,
            target_frequency,
        ),
        "phase1_frequency_error": difference(
            baseline_frequency,
            target_frequency,
        ),
        "ensemble_mean_intensity_error": difference(
            refined_intensity,
            target_intensity,
        ),
        "phase1_intensity_error": difference(
            baseline_intensity,
            target_intensity,
        ),
        "ensemble_mean_contingency": refined_contingency.result(),
        "phase1_contingency": baseline_contingency.result(),
    }


def _metric_delta(
    refined: dict[str, Any],
    baseline: dict[str, Any],
    name: str,
) -> float | None:
    first = refined.get(name)
    second = baseline.get(name)
    if first is None or second is None:
        return None
    return float(first - second)


def _member_negative_fields(ensemble):
    # Sufficient statistics retain exactly the same negative counts/deficits
    # while avoiding a ten-member gather for each geographic region.
    negative = ensemble < 0
    return {
        "count": negative.sum(axis=1),
        "deficit": np.where(negative, -ensemble, 0.0).sum(axis=1),
        "minimum": ensemble.min(axis=1),
    }


class _RegionalAccumulator:
    """Exact daily metrics on an externally specified geographic support."""

    def __init__(self, wet_threshold: float) -> None:
        self.errors = {name: _ErrorAccumulator() for name in ("phase1", "ensemble_mean")}
        self.wet = {name: _WetAccumulator(wet_threshold)
                    for name in ("phase1", "ensemble_mean", "target")}
        self.crps_sum = 0.0
        self.count = 0
        self.negative_count = 0
        self.member_count = 0
        self.negative_sum = 0.0
        self.minimum = None

    def update(self, mean, baseline, target, ensemble, crps, region, *, precipitation, member_summary=None):
        observed = target[:, region]
        refined, phase1 = mean[:, region], baseline[:, region]
        self.errors["phase1"].update(phase1, observed)
        self.errors["ensemble_mean"].update(refined, observed)
        scores = crps[:, region]
        valid = np.isfinite(scores)
        self.crps_sum += float(scores[valid].sum())
        self.count += int(valid.sum())
        if precipitation:
            for name, values in (("phase1", phase1), ("ensemble_mean", refined),
                                 ("target", observed)):
                self.wet[name].update(values)
            if member_summary is None:
                member_summary = _member_negative_fields(ensemble)
            support = np.isfinite(observed)
            self.member_count += int(support.sum()) * ensemble.shape[1]
            self.negative_count += int(member_summary["count"][:, region][support].sum())
            self.negative_sum += float(member_summary["deficit"][:, region][support].sum())
            minima = member_summary["minimum"][:, region][support]
            if minima.size:
                minimum = float(minima.min())
                self.minimum = minimum if self.minimum is None else min(self.minimum, minimum)

    def result(self, maps, region, *, precipitation):
        result = {
            "spatial_cell_count": int(region.sum()),
            "phase1": self.errors["phase1"].result(),
            "ensemble_mean": self.errors["ensemble_mean"].result(),
            "climatology": {
                name: _map_difference_metrics(maps[name][region], maps["ground_truth"][region])
                for name in ("phase1", "ensemble_mean")
            },
            "empirical_crps": self.crps_sum / self.count if self.count else None,
        }
        if precipitation:
            result["wet_day"] = {"threshold": self.wet["target"].threshold}
            for name, accumulator in self.wet.items():
                frequency, intensity = accumulator.result()
                result["wet_day"][name] = {"frequency": frequency, "intensity": intensity}
            result["saved_physical_member_negatives"] = {
                "finite_count": self.member_count, "negative_count": self.negative_count,
                "negative_fraction": self.negative_count / self.member_count if self.member_count else None,
                "mean_negative_magnitude": self.negative_sum / self.negative_count if self.negative_count else 0.0 if self.member_count else None,
                "mean_negative_deficit": self.negative_sum / self.member_count if self.member_count else None,
                "minimum": self.minimum,
            }
        return result


class _ConstraintAccumulator:
    """Exact member-level changes; available only when both stages were saved."""

    def __init__(self, threshold):
        self.threshold = threshold
        self.count = self.negative_count = self.changed_count = 0
        self.raw_wet_count = self.post_wet_count = 0
        self.negative_sum = self.shift_sum = self.absolute_shift_sum = 0.0

    def update(self, raw, post, common_valid, region):
        valid = np.broadcast_to(common_valid[:, None], post.shape)[:, :, region]
        before, after = raw[:, :, region][valid], post[:, :, region][valid]
        if not np.isfinite(before).all():
            raise RefinementEvaluationError("Preconstraint members are invalid on the common evaluation mask.")
        negative = before < 0
        shift = after - before
        self.count += int(before.size)
        self.negative_count += int(negative.sum())
        self.negative_sum += float(-before[negative].sum())
        self.changed_count += int(np.count_nonzero(shift))
        self.shift_sum += float(shift.sum())
        self.absolute_shift_sum += float(np.abs(shift).sum())
        if self.threshold is not None:
            self.raw_wet_count += int((before >= self.threshold).sum())
            self.post_wet_count += int((after >= self.threshold).sum())

    def result(self):
        def mean(value):
            return value / self.count if self.count else None
        result = {
            "finite_member_count": self.count,
            "preconstraint_negative_fraction": mean(self.negative_count),
            "preconstraint_negative_mean_magnitude": self.negative_sum / self.negative_count if self.negative_count else 0.0 if self.count else None,
            "preconstraint_mean_negative_deficit": mean(self.negative_sum),
            "modified_member_fraction": mean(self.changed_count),
            "mean_physical_change": mean(self.shift_sum),
            "mean_absolute_physical_change": mean(self.absolute_shift_sum),
            "wet_threshold": self.threshold,
            "preconstraint_member_wet_frequency": mean(self.raw_wet_count),
            "postprocessed_member_wet_frequency": mean(self.post_wet_count),
        }
        if self.threshold is None:
            for key in ("wet_threshold", "preconstraint_member_wet_frequency", "postprocessed_member_wet_frequency"):
                result.pop(key)
        return result


class _CalibrationAccumulator:
    def __init__(self, members):
        self.count = 0
        self.crps_sum = 0.0
        self.rank_histogram = np.zeros(members + 1, dtype=np.float64)
        self.coverage_count = {str(level): 0 for level in (0.5, 0.8, 0.9, 0.95)}
        self.interval_width_sum = {str(level): 0.0 for level in (0.5, 0.8, 0.9, 0.95)}
        self.member_count = members

    def update(self, ensemble, target, score):
        valid = np.isfinite(target) & np.isfinite(ensemble).all(axis=1)
        values = np.moveaxis(ensemble, 1, 0)[:, valid]
        truth = target[valid]
        self.count += int(truth.size)
        self.crps_sum += float(score[valid].sum())
        if not truth.size:
            return
        below = (values < truth).sum(axis=0)
        equal = (values == truth).sum(axis=0)
        # Spread exact ties uniformly across admissible ranks, so valid dry
        # days do not all land in the lowest bin.
        for rank in range(self.member_count + 1):
            self.rank_histogram[rank] += float(np.where(
                (below <= rank) & (rank <= below + equal), 1.0 / (equal + 1), 0.0
            ).sum())
        if self.member_count < 2:
            return
        keys = list(self.coverage_count)
        levels = [point for key in keys for point in ((1-float(key))/2, (1+float(key))/2)]
        bounds = np.quantile(values, levels, axis=0)
        for index, key in enumerate(keys):
            low, high = bounds[2*index:2*index+2]
            self.coverage_count[key] += int(((truth >= low) & (truth <= high)).sum())
            self.interval_width_sum[key] += float((high - low).sum())

    def result(self):
        return {
            "score": "CRPS of the issued finite empirical ensemble (not fair CRPS)",
            "crps": self.crps_sum / self.count if self.count else None,
            "valid_count": self.count,
            "rank_histogram_counts": self.rank_histogram.tolist(),
            "rank_ties": "fractional uniform allocation across tied admissible ranks",
            "central_interval_calibration": {
                key: {
                    "coverage": self.coverage_count[key] / self.count if self.count and self.member_count > 1 else None,
                    "mean_width": self.interval_width_sum[key] / self.count if self.count and self.member_count > 1 else None,
                } for key in self.coverage_count
            },
            "interval_note": "Finite-ensemble linear empirical quantiles; nominal levels alone do not guarantee calibrated coverage.",
        }


def _sampling_uncertainty(member_maps, target_map, baseline_map, region):
    """Conditional sensitivity to resampling saved physical member fields."""
    count = member_maps.shape[0]
    valid = region & np.isfinite(target_map) & np.isfinite(baseline_map) & np.isfinite(member_maps).all(axis=0)
    if count < 2 or not valid.any():
        return {"status": "unavailable: at least two members and valid cells required"}
    values = member_maps[:, valid]
    truth = target_map[valid]
    original = float(np.abs(baseline_map[valid] - truth).mean())
    rng = np.random.default_rng(1759)
    changes = []
    for _ in range(200):
        selected = rng.integers(0, count, size=count)
        changes.append(float(np.abs(values[selected].mean(axis=0) - truth).mean()) - original)
    low, high = np.quantile(changes, [0.025, 0.975])
    return {
        "status": "conditional saved-member resampling sensitivity; not an independent sampler-seed confidence interval",
        "projection_recomputed": False,
        "member_independence_assumed": False,
        "interpretation": (
            "Mean-preserving ensemble projection couples processed members. "
            "These percentiles condition on the saved projected ensemble; "
            "they do not rerun projection or estimate independent new-seed variability."
        ),
        "method": "resample whole physical member climatologies with replacement",
        "resamples": 200, "bootstrap_seed": 1759,
        "climatological_absolute_bias_delta_95_percentile_interval": [float(low), float(high)],
        "fraction_of_resamples_improving_climatological_absolute_bias": float((np.asarray(changes) < 0).mean()),
        "independent_seed_group_replication": "not evaluated by a single output file",
    }


def evaluate_variable(
    members: xr.DataArray,
    baseline: xr.DataArray,
    target: xr.DataArray,
    *,
    member_dim: str,
    variable: str,
    chunk_time: int = 32,
    distribution_samples: int = 1_000_000,
    spectral_bins: int = 8,
    wet_day_threshold: float = 1.0,
    unbounded_members: xr.DataArray | None = None,
    progress_callback: Any = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Compute metrics while loading only time chunks into memory."""
    if chunk_time < 1:
        raise RefinementEvaluationError("chunk_time must be at least 1.")
    if distribution_samples < 1:
        raise RefinementEvaluationError("distribution_samples must be at least 1.")
    if spectral_bins < 2:
        raise RefinementEvaluationError("spectral_bins must be at least 2.")
    is_precipitation = _is_precipitation(variable, members)
    if is_precipitation and (
        not math.isfinite(float(wet_day_threshold))
        or float(wet_day_threshold) < 0.0
    ):
        raise RefinementEvaluationError(
            "wet_day_threshold must be finite and non-negative for precipitation."
        )
    expected_member_dims = ("time", member_dim, "lat", "lon")
    if members.dims != expected_member_dims:
        raise RefinementEvaluationError(
            f"{variable!r} member dimensions changed after selection: "
            f"{members.dims} != {expected_member_dims}."
        )

    for label, field in (("baseline", baseline), ("target", target), ("preconstraint members", unbounded_members)):
        if field is None:
            continue
        expected_dims = members.dims if label == "preconstraint members" else ("time", "lat", "lon")
        if field.dims != expected_dims:
            raise RefinementEvaluationError(f"{label} dimensions must be {expected_dims}, got {field.dims}.")
        for axis in ("time", "lat", "lon"):
            if field.sizes[axis] != members.sizes[axis]:
                raise RefinementEvaluationError(f"{label} {axis} size differs from the ensemble.")
            first, second = field[axis].values, members[axis].values
            equal = ([_timestamp_key(value) for value in first] == [_timestamp_key(value) for value in second]
                     if axis == "time" else np.array_equal(first, second))
            if not equal:
                raise RefinementEvaluationError(f"{label} {axis} coordinate differs from the ensemble.")
    time_count = int(members.sizes["time"])
    member_count = int(members.sizes[member_dim])
    shape = (int(members.sizes["lat"]), int(members.sizes["lon"]))
    if min(shape) < 2:
        raise RefinementEvaluationError(
            "Spatial-gradient and spectral evaluation require at least a "
            "2x2 grid."
        )

    region_masks, boundary_distance = boundary_region_masks(members["lat"].values, members["lon"].values)
    regions = {name: _RegionalAccumulator(wet_day_threshold) for name in region_masks}
    calibration = _CalibrationAccumulator(member_count)
    constraints = None
    if unbounded_members is not None:
        if unbounded_members.dims != members.dims or unbounded_members.shape != members.shape:
            raise RefinementEvaluationError("Preconstraint and physical members must have identical dimensions.")
        if str(unbounded_members.attrs.get("units")) != str(members.attrs.get("units")):
            raise RefinementEvaluationError("Preconstraint and physical members must use identical physical units.")
        constraints = {name: _ConstraintAccumulator(wet_day_threshold if is_precipitation else None) for name in region_masks}
    support_digest, baseline_digest = hashlib.sha256(), hashlib.sha256()
    coordinate_digest = hashlib.sha256()
    for coordinate in ("lat", "lon"):
        coordinate_digest.update(np.asarray(members[coordinate].values, dtype="<f8").tobytes())
    coordinate_digest.update(json.dumps([_timestamp_key(value) for value in members["time"].values]).encode("ascii"))
    months = np.asarray([_timestamp_key(value)[1] for value in members["time"].values])
    season_months = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
    seasons = {
        name: {"errors": {key: _ErrorAccumulator() for key in ("phase1", "ensemble_mean")},
               "maps": {key: _MapMean(shape) for key in ("phase1", "ensemble_mean", "ground_truth")},
               "time_count": 0}
        for name in season_months
    }
    member_samplers = [
        _BoundedTripletSampler(time_count * shape[0] * shape[1], max(1, distribution_samples // member_count))
        for _ in range(member_count)
    ]
    member_wet = [_WetAccumulator(wet_day_threshold) for _ in range(member_count)]
    member_autocorrelation = [[] for _ in range(member_count)]
    correction_autocorrelation = [[] for _ in range(member_count)]

    refined_error = _ErrorAccumulator()
    baseline_error = _ErrorAccumulator()
    member_errors = [_ErrorAccumulator() for _ in range(member_count)]
    refined_gradient_error = _ErrorAccumulator()
    baseline_gradient_error = _ErrorAccumulator()
    climatologies = {
        "ground_truth": _MapMean(shape),
        "phase1": _MapMean(shape),
        "ensemble_mean": _MapMean(shape),
        "ensemble_spread": _MapMean(shape),
    }
    member_climatologies = [_MapMean(shape) for _ in range(member_count)]
    total_points = time_count * shape[0] * shape[1]
    sampler = _BoundedTripletSampler(total_points, distribution_samples)
    daily_pattern_refined: list[float | None] = []
    daily_pattern_baseline: list[float | None] = []
    autocorrelation_refined: list[float | None] = []
    autocorrelation_baseline: list[float | None] = []
    autocorrelation_target: list[float | None] = []
    wet_refined = _WetAccumulator(wet_day_threshold)
    wet_baseline = _WetAccumulator(wet_day_threshold)
    wet_target = _WetAccumulator(wet_day_threshold)
    wet_refined_contingency = _WetContingencyAccumulator(wet_day_threshold)
    wet_baseline_contingency = _WetContingencyAccumulator(wet_day_threshold)
    spread_sum = 0.0
    spread_squared_sum = 0.0
    spread_count = 0
    diversity_squared_sum = 0.0
    diversity_count = 0
    nonfinite = {
        "member_nan_count": 0,
        "member_inf_count": 0,
        "phase1_nan_count": 0,
        "phase1_inf_count": 0,
        "target_nan_count": 0,
        "target_inf_count": 0,
    }
    target_finite_count = 0
    phase1_finite_count = 0
    all_members_finite_count = 0
    common_valid_count = 0

    for start in range(0, time_count, chunk_time):
        stop = min(start + chunk_time, time_count)
        member_chunk = np.asarray(
            members.isel(time=slice(start, stop)).values,
            dtype=np.float64,
        )
        baseline_chunk = np.asarray(
            baseline.isel(time=slice(start, stop)).values,
            dtype=np.float64,
        )
        target_chunk = np.asarray(
            target.isel(time=slice(start, stop)).values,
            dtype=np.float64,
        )
        nonfinite["member_nan_count"] += int(np.isnan(member_chunk).sum())
        nonfinite["member_inf_count"] += int(np.isinf(member_chunk).sum())
        nonfinite["phase1_nan_count"] += int(np.isnan(baseline_chunk).sum())
        nonfinite["phase1_inf_count"] += int(np.isinf(baseline_chunk).sum())
        nonfinite["target_nan_count"] += int(np.isnan(target_chunk).sum())
        nonfinite["target_inf_count"] += int(np.isinf(target_chunk).sum())
        if (
            nonfinite["member_inf_count"]
            or nonfinite["phase1_inf_count"]
            or nonfinite["target_inf_count"]
        ):
            raise RefinementEvaluationError(
                f"Infinite values detected while evaluating {variable!r}."
            )

        ensemble_mean = _safe_ensemble_mean(member_chunk)
        ensemble_spread = _safe_ensemble_spread(member_chunk)
        target_finite = np.isfinite(target_chunk)
        phase1_finite = np.isfinite(baseline_chunk)
        all_members_finite = np.isfinite(member_chunk).all(axis=1)
        common_valid = target_finite & phase1_finite & all_members_finite
        target_finite_count += int(target_finite.sum())
        phase1_finite_count += int(phase1_finite.sum())
        all_members_finite_count += int(all_members_finite.sum())
        common_valid_count += int(common_valid.sum())
        support_digest.update(common_valid.astype(np.uint8).tobytes())
        # Hash actual matched Phase-1 fields, including invalid locations, to
        # verify reuse across independently evaluated heads and seed groups.
        baseline_digest.update(np.where(np.isfinite(baseline_chunk), baseline_chunk, np.nan).astype("<f8").tobytes())

        target_evaluation = np.where(common_valid, target_chunk, np.nan)
        phase1_evaluation = np.where(common_valid, baseline_chunk, np.nan)
        mean_evaluation = np.where(common_valid, ensemble_mean, np.nan)
        spread_evaluation = np.where(common_valid, ensemble_spread, np.nan)
        member_evaluation = np.where(
            common_valid[:, None, :, :],
            member_chunk,
            np.nan,
        )

        if constraints is not None:
            unbounded_chunk = np.asarray(unbounded_members.isel(time=slice(start, stop)).values, dtype=np.float64)
            for name, region in region_masks.items():
                constraints[name].update(unbounded_chunk, member_chunk, common_valid, region)
        crps = empirical_crps(member_evaluation, target_evaluation)
        calibration.update(member_evaluation, target_evaluation, crps)
        member_summary = _member_negative_fields(member_evaluation) if is_precipitation else None
        for name, region in region_masks.items():
            regions[name].update(mean_evaluation, phase1_evaluation, target_evaluation,
                                 member_evaluation, crps, region, precipitation=is_precipitation,
                                 member_summary=member_summary)
        for name, calendar_months in season_months.items():
            selected = np.isin(months[start:stop], calendar_months)
            season = seasons[name]
            season["time_count"] += int(selected.sum())
            for key, values in (("phase1", phase1_evaluation), ("ensemble_mean", mean_evaluation)):
                season["errors"][key].update(values[selected], target_evaluation[selected])
                season["maps"][key].update(values[selected])
            season["maps"]["ground_truth"].update(target_evaluation[selected])
        for index in range(member_count):
            member_samplers[index].update(member_evaluation[:, index], phase1_evaluation, target_evaluation)
            if is_precipitation:
                member_wet[index].update(member_evaluation[:, index])
            member_autocorrelation[index].extend(_daily_lag_one_autocorrelations(member_evaluation[:, index]))
            correction_autocorrelation[index].extend(_daily_lag_one_autocorrelations(member_evaluation[:, index] - phase1_evaluation))

        refined_error.update(mean_evaluation, target_evaluation)
        baseline_error.update(phase1_evaluation, target_evaluation)
        for index in range(member_count):
            member_errors[index].update(
                member_evaluation[:, index],
                target_evaluation,
            )

        climatologies["ground_truth"].update(target_evaluation)
        climatologies["phase1"].update(phase1_evaluation)
        climatologies["ensemble_mean"].update(mean_evaluation)
        climatologies["ensemble_spread"].update(spread_evaluation)
        for index in range(member_count):
            member_climatologies[index].update(member_evaluation[:, index])

        refined_gradient_error.update(
            _gradient_fields(mean_evaluation),
            _gradient_fields(target_evaluation),
        )
        baseline_gradient_error.update(
            _gradient_fields(phase1_evaluation),
            _gradient_fields(target_evaluation),
        )
        daily_pattern_refined.extend(
            _daily_spatial_correlations(mean_evaluation, target_evaluation)
        )
        daily_pattern_baseline.extend(
            _daily_spatial_correlations(phase1_evaluation, target_evaluation)
        )
        autocorrelation_refined.extend(
            _daily_lag_one_autocorrelations(mean_evaluation)
        )
        autocorrelation_baseline.extend(
            _daily_lag_one_autocorrelations(phase1_evaluation)
        )
        autocorrelation_target.extend(
            _daily_lag_one_autocorrelations(target_evaluation)
        )
        sampler.update(mean_evaluation, phase1_evaluation, target_evaluation)
        wet_refined.update(mean_evaluation)
        wet_baseline.update(phase1_evaluation)
        wet_target.update(target_evaluation)
        wet_refined_contingency.update(mean_evaluation, target_evaluation)
        wet_baseline_contingency.update(phase1_evaluation, target_evaluation)

        finite_spread = spread_evaluation[np.isfinite(spread_evaluation)]
        spread_sum += float(finite_spread.sum())
        spread_squared_sum += float(np.square(finite_spread).sum())
        spread_count += int(finite_spread.size)
        for first in range(member_count):
            for second in range(first + 1, member_count):
                first_member = member_evaluation[:, first]
                second_member = member_evaluation[:, second]
                valid = np.isfinite(first_member) & np.isfinite(second_member)
                difference = first_member[valid] - second_member[valid]
                diversity_squared_sum += float(np.square(difference).sum())
                diversity_count += int(difference.size)

        if progress_callback is not None:
            progress_callback(stop, time_count)

    maps = {
        name: accumulator.result()
        for name, accumulator in climatologies.items()
    }
    maps["member_climatologies"] = np.stack(
        [accumulator.result() for accumulator in member_climatologies],
        axis=0,
    )
    maps["phase1_bias"] = maps["phase1"] - maps["ground_truth"]
    maps["ensemble_mean_bias"] = (
        maps["ensemble_mean"] - maps["ground_truth"]
    )
    maps["correction"] = maps["ensemble_mean"] - maps["phase1"]
    maps["valid_sample_count"] = climatologies["ground_truth"].count.copy()
    maps["distance_from_boundary_cells"] = boundary_distance

    refined_metrics = refined_error.result()
    baseline_metrics = baseline_error.result()
    refined_metrics["mean_daily_spatial_pattern_correlation"] = _finite_mean(
        daily_pattern_refined
    )
    baseline_metrics["mean_daily_spatial_pattern_correlation"] = _finite_mean(
        daily_pattern_baseline
    )
    refined_metrics["mean_daily_lag1_spatial_autocorrelation"] = _finite_mean(
        autocorrelation_refined
    )
    baseline_metrics["mean_daily_lag1_spatial_autocorrelation"] = _finite_mean(
        autocorrelation_baseline
    )
    refined_metrics["spatial_gradient"] = refined_gradient_error.result()
    baseline_metrics["spatial_gradient"] = baseline_gradient_error.result()

    refined_climatology = _map_difference_metrics(
        maps["ensemble_mean"],
        maps["ground_truth"],
    )
    baseline_climatology = _map_difference_metrics(
        maps["phase1"],
        maps["ground_truth"],
    )
    refined_samples, baseline_samples, target_samples = sampler.result()
    refined_rmse = refined_metrics.get("rmse")
    rms_spread = (
        math.sqrt(spread_squared_sum / spread_count) if spread_count else None
    )
    spread_skill_ratio = (
        None
        if rms_spread is None or refined_rmse in (None, 0.0)
        else rms_spread / float(refined_rmse)
    )
    member_maps = maps["member_climatologies"]
    pairwise_climatology_correlations = []
    for first in range(member_count):
        for second in range(first + 1, member_count):
            pairwise_climatology_correlations.append(
                _correlation(member_maps[first], member_maps[second])
            )

    metrics: dict[str, Any] = {
        "units": str(members.attrs.get("units", "")),
        "time_count": time_count,
        "ensemble_size": member_count,
        "ensemble_mean_skill": refined_metrics,
        "phase1_skill": baseline_metrics,
        "skill_change_refined_minus_phase1": {
            "mean_bias": _metric_delta(
                refined_metrics,
                baseline_metrics,
                "mean_bias",
            ),
            "mae": _metric_delta(refined_metrics, baseline_metrics, "mae"),
            "rmse": _metric_delta(refined_metrics, baseline_metrics, "rmse"),
            "mean_daily_spatial_pattern_correlation": _metric_delta(
                refined_metrics,
                baseline_metrics,
                "mean_daily_spatial_pattern_correlation",
            ),
        },
        "climatology": {
            "ensemble_mean": refined_climatology,
            "phase1": baseline_climatology,
        },
        "distribution_quantiles": _quantile_metrics(
            refined_samples,
            baseline_samples,
            target_samples,
        ),
        "distribution_sampling": {
            "method": "uniform global cell indices without replacement; fixed seed 0; paired across products",
            "seed": 0,
            "maximum_samples": int(distribution_samples),
            "stride": sampler.stride,
        },
        "spatial_autocorrelation": {
            "ensemble_mean": _finite_mean(autocorrelation_refined),
            "phase1": _finite_mean(autocorrelation_baseline),
            "target": _finite_mean(autocorrelation_target),
        },
        "spectral_power_by_scale": {
            "ensemble_mean": _spectral_comparison(
                maps["ensemble_mean"],
                maps["ground_truth"],
                spectral_bins,
            ),
            "phase1": _spectral_comparison(
                maps["phase1"],
                maps["ground_truth"],
                spectral_bins,
            ),
        },
        "ensemble": {
            "mean_spread": spread_sum / spread_count if spread_count else None,
            "rms_spread": rms_spread,
            "spread_skill_ratio": spread_skill_ratio,
            "mean_pairwise_member_rms_difference": (
                math.sqrt(diversity_squared_sum / diversity_count)
                if diversity_count
                else 0.0
            ),
            "mean_pairwise_member_climatology_correlation": _finite_mean(
                pairwise_climatology_correlations
            ),
            "individual_member_skill": [
                item.result() for item in member_errors
            ],
        },
        "evaluation_mask": {
            "definition": (
                "finite ground truth, Phase-1 prediction, and every physical "
                "ensemble member"
            ),
            "total_count": int(total_points),
            "target_finite_count": int(target_finite_count),
            "phase1_finite_count": int(phase1_finite_count),
            "all_members_finite_count": int(all_members_finite_count),
            "common_valid_count": int(common_valid_count),
            "excluded_count": int(total_points - common_valid_count),
        },
        "nonfinite_counts": nonfinite,
    }
    metrics["pairing"] = {
        "coordinate_and_time_sha256": coordinate_digest.hexdigest(),
        "common_valid_mask_sha256": support_digest.hexdigest(),
        "matched_phase1_values_sha256": baseline_digest.hexdigest(),
        "requirement": "All three fingerprints, ensemble sizes, and draw seeds must match across compared outputs.",
    }
    metrics["geographic_regions"] = {
        name: accumulator.result(maps, region_masks[name], precipitation=is_precipitation)
        for name, accumulator in regions.items()
    }
    metrics["geographic_region_definition"] = {
        "edge_orientation": "geographic latitude/longitude values, independent of array orientation",
        "distance": "integer distance in cells from rectangular outer boundary; not invalid-data holes",
        "requested_widths": [1, 2, 4, 8, 16],
        "effective_widths": sorted(int(name.split("_")[1]) for name in region_masks if name.startswith("boundary_")),
        "overlap": "edges include corners; regions overlap and must not be added",
        "southeast_quadrant": {
            "latitude_max": float((members["lat"].min() + members["lat"].max()) / 2),
            "longitude_min": float((members["lon"].min() + members["lon"].max()) / 2),
            "purpose": "predeclared broad southeast region; not a fitted dry-bias mask",
        },
        "weighting": "equal valid cell/time weights; climatological metrics give equal weight to valid gridpoints",
    }
    metrics["seasonal"] = {}
    for season_name, season in seasons.items():
        seasonal_maps = {key: value.result() for key, value in season["maps"].items()}
        metrics["seasonal"][season_name] = {
            "time_count": season["time_count"],
            "daily_skill": {key: value.result() for key, value in season["errors"].items()},
            "regional_climatology": {
                region_name: {
                    key: _map_difference_metrics(seasonal_maps[key][region], seasonal_maps["ground_truth"][region])
                    for key in ("phase1", "ensemble_mean")
                } for region_name, region in region_masks.items() if not region_name.startswith("distance_")
            },
        }
    metrics["ensemble"]["calibration"] = calibration.result()
    metrics["ensemble"]["member_distributions"] = []
    for index, member_sampler in enumerate(member_samplers):
        distribution = _quantile_metrics(*member_sampler.result())
        # This helper's legacy key names describe the compared field; rename
        # them explicitly for individual-member distributions.
        distribution = {key.replace("ensemble_mean", "member"): value for key, value in distribution.items()}
        item = {
            "member_index": index, "quantiles": distribution,
            "daily_lag1_spatial_autocorrelation": _finite_mean(member_autocorrelation[index]),
            "correction_daily_lag1_spatial_autocorrelation": _finite_mean(correction_autocorrelation[index]),
        }
        if is_precipitation:
            frequency, intensity = member_wet[index].result()
            item["wet_day"] = {"threshold": wet_day_threshold, "frequency": frequency, "intensity": intensity}
        metrics["ensemble"]["member_distributions"].append(item)
    metrics["ensemble"]["finite_member_uncertainty"] = {
        name: _sampling_uncertainty(maps["member_climatologies"], maps["ground_truth"], maps["phase1"], region)
        for name, region in region_masks.items()
        if name in ("full", "southeast_quadrant") or name.startswith(("boundary_", "interior_"))
    }
    if is_precipitation:
        metrics["pre_postprocessing_diagnostics"] = {
            "status": "unavailable from postprocessed predictions alone",
            "required_product": "physical members before constraints plus corresponding postprocessed members",
            "saved_member_negatives": "reported by region; these do not establish preconstraint negative frequency",
        }
    if is_precipitation:
        metrics["wet_day"] = _exact_wet_metrics(
            wet_refined,
            wet_baseline,
            wet_target,
            wet_refined_contingency,
            wet_baseline_contingency,
        )
        metrics["wet_day"]["threshold_units"] = metrics["units"]
    if constraints is not None:
        metrics["pre_postprocessing_diagnostics"] = {
            "status": "paired preconstraint and postprocessed physical members",
            "regions": {name: accumulator.result() for name, accumulator in constraints.items()},
        }
    return metrics, maps


def evaluate_refinement_datasets(
    prediction: xr.Dataset,
    baseline: xr.Dataset,
    target: xr.Dataset,
    *,
    variables: Sequence[str],
    expected_refinement_type: str | None = None,
    expected_checkpoint: str | Path | None = None,
    expected_case: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    require_all_prediction_times: bool = True,
    chunk_time: int = 32,
    distribution_samples: int = 1_000_000,
    spectral_bins: int = 8,
    wet_day_threshold: float = 1.0,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    """Validate, align, convert, and evaluate already-open datasets."""
    variables = [str(variable) for variable in variables]
    if not variables or len(variables) != len(set(variables)):
        raise RefinementEvaluationError(
            "variables must be a non-empty list without duplicates."
        )
    selection = validate_refinement_output(
        prediction,
        variables=variables,
        expected_refinement_type=expected_refinement_type,
        expected_checkpoint=expected_checkpoint,
        expected_case=expected_case,
    )
    validate_baseline_output(
        baseline,
        variables=variables,
        refinement_case=selection.refinement_case,
        phase1_checkpoint=selection.phase1_checkpoint,
        phase1_fingerprint=selection.phase1_fingerprint,
    )
    validate_target(target, variables=variables)
    _validate_same_spatial_grid(prediction, baseline, "Phase-1 baseline")
    _validate_same_spatial_grid(prediction, target, "ground-truth target")
    aligned = align_exact_timestamps(
        prediction,
        baseline,
        target,
        start_date=start_date,
        end_date=end_date,
        require_all_prediction_times=require_all_prediction_times,
    )

    summary: dict[str, Any] = {
        "evaluation_contract": "CORDEX physical residual-refinement v2 regional-probabilistic",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "refinement_type": selection.refinement_type,
        "refinement_case": selection.refinement_case,
        "refinement_checkpoint": selection.refinement_checkpoint,
        "checkpoint_schema_version": selection.checkpoint_schema_version,
        "refinement_contract_version": selection.refinement_contract_version,
        "residual_contract": selection.residual_contract,
        "refinement_contract_fingerprint": (
            selection.refinement_contract_fingerprint
        ),
        "refinement_checkpoint_sha256": selection.refinement_checkpoint_sha256,
        "phase1_checkpoint": selection.phase1_checkpoint,
        "phase1_fingerprint": selection.phase1_fingerprint,
        "residual_normalizer_state_fingerprint": (
            selection.residual_normalizer_state_fingerprint
        ),
        "residual_normalization": selection.residual_normalization,
        "prediction_kind": "refinement_ensemble_members",
        "ensemble_size": selection.ensemble_size,
        "ensemble_aggregation": (
            "arithmetic mean of individually reconstructed physical members"
        ),
        "evaluation_split": str(prediction.attrs.get("evaluation_split", "unavailable: verify the actual experiment split")),
        "seed_metadata": {key: str(value) for key, value in prediction.attrs.items()
                          if "seed" in key.lower()},
        "time_alignment": aligned.report,
        "variables": {},
    }
    maps_by_variable: dict[str, dict[str, np.ndarray]] = {}
    for variable in variables:
        member_field = aligned.prediction[selection.variables[variable]]
        prediction_units = str(member_field.attrs["units"]).strip()
        if (
            _is_precipitation(variable, member_field)
            and _unit_key(prediction_units) not in _PRECIP_DAILY_UNITS
        ):
            raise RefinementEvaluationError(
                f"Precipitation refinement output {variable!r} must use physical "
                f"daily units (mm/day), got {prediction_units!r}."
            )
        converted_baseline, baseline_conversion = convert_dataarray_units(
            aligned.baseline[variable],
            prediction_units,
            variable=variable,
        )
        converted_target, target_conversion = convert_dataarray_units(
            aligned.target[variable],
            prediction_units,
            variable=variable,
        )
        metrics, maps = evaluate_variable(
            member_field,
            converted_baseline,
            converted_target,
            member_dim=selection.member_dim,
            variable=variable,
            chunk_time=chunk_time,
            distribution_samples=distribution_samples,
            spectral_bins=spectral_bins,
            wet_day_threshold=wet_day_threshold,
            unbounded_members=aligned.prediction.get(f"{variable}_members_unbounded"),
        )
        metrics["unit_conversion"] = {
            "phase1_to_prediction_units": baseline_conversion,
            "target_to_prediction_units": target_conversion,
        }
        summary["variables"][variable] = metrics
        maps_by_variable[variable] = maps
    return summary, maps_by_variable


def validate_paired_evaluations(
    summaries: Sequence[Mapping[str, Any]], *, require_paired_seeds: bool = True,
) -> None:
    """Reject comparisons that changed dates, masks, baseline, members or seeds.

    Run this on complete per-output summaries before making before/after
    improvement claims. Across heads, seeds need not match, but all three data
    fingerprints and the number of ensemble members still must.
    """
    if len(summaries) < 2:
        raise RefinementEvaluationError("At least two evaluation summaries are required for a paired comparison.")
    reference = summaries[0]
    for candidate in summaries[1:]:
        if set(reference["variables"]) != set(candidate["variables"]):
            raise RefinementEvaluationError("Compared outputs do not contain the same target variables.")
        for variable in reference["variables"]:
            first, second = reference["variables"][variable], candidate["variables"][variable]
            if first["ensemble_size"] != second["ensemble_size"]:
                raise RefinementEvaluationError(f"Ensemble sizes differ for {variable}.")
            for key in ("coordinate_and_time_sha256", "common_valid_mask_sha256", "matched_phase1_values_sha256"):
                if not first.get("pairing", {}).get(key) or first["pairing"][key] != second.get("pairing", {}).get(key):
                    raise RefinementEvaluationError(f"Paired comparison failed for {variable}: {key}.")
        if require_paired_seeds:
            first_seeds = {key: str(value) for key, value in reference.get("seed_metadata", {}).items()}
            second_seeds = {key: str(value) for key, value in candidate.get("seed_metadata", {}).items()}
            if not first_seeds or first_seeds != second_seeds:
                raise RefinementEvaluationError("Before/after draw seeds are missing or differ.")


def _finite_limits(arrays: Sequence[np.ndarray]) -> tuple[float, float]:
    finite_arrays = [array[np.isfinite(array)] for array in arrays]
    finite_arrays = [array for array in finite_arrays if array.size]
    if not finite_arrays:
        return 0.0, 1.0
    minimum = min(float(array.min()) for array in finite_arrays)
    maximum = max(float(array.max()) for array in finite_arrays)
    if minimum == maximum:
        delta = max(1.0e-12, abs(minimum) * 1.0e-6)
        return minimum - delta, maximum + delta
    return minimum, maximum


def shared_map_limits(outputs: Sequence[Mapping[str, Mapping[str, np.ndarray]]]) -> dict[str, dict[str, list[float]]]:
    """Compute one plot scale per variable across every before/after head."""
    variables = {variable for output in outputs for variable in output}
    limits = {}
    for variable in variables:
        fields, differences, spreads = [], [], []
        for output in outputs:
            if variable not in output:
                continue
            maps = output[variable]
            fields.extend(maps[key] for key in ("ground_truth", "phase1", "ensemble_mean", "member_climatologies"))
            differences.extend(maps[key] for key in ("phase1_bias", "ensemble_mean_bias", "correction"))
            spreads.append(maps["ensemble_spread"])
        low, high = _finite_limits(differences)
        bound = max(abs(low), abs(high), 1e-12)
        limits[variable] = {
            "field": list(_finite_limits(fields)),
            "difference": [-bound, bound],
            "spread": [0.0, max(_finite_limits(spreads)[1], 1e-12)],
        }
    return limits


def _draw_map(
    axis: Any,
    field: np.ndarray,
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
) -> Any:
    image = axis.pcolormesh(
        lon,
        lat,
        field,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    axis.set_title(title, fontsize=10)
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    return image


def plot_evaluation_maps(
    maps: dict[str, np.ndarray],
    *,
    variable: str,
    units: str,
    refinement_type: str,
    lat: np.ndarray,
    lon: np.ndarray,
    output_path: str | Path,
    limits: Mapping[str, Sequence[float]] | None = None,
) -> Path:
    """Write the requested eight maps with shared comparable color scales."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    member_maps = maps["member_climatologies"]
    field_arrays = [
        maps["ground_truth"],
        maps["phase1"],
        maps["ensemble_mean"],
        *list(member_maps),
    ]
    field_min, field_max = _finite_limits(field_arrays)
    difference_arrays = [
        maps["phase1_bias"],
        maps["ensemble_mean_bias"],
        maps["correction"],
    ]
    difference_min, difference_max = _finite_limits(difference_arrays)
    difference_limit = max(abs(difference_min), abs(difference_max), 1.0e-12)
    _, spread_max = _finite_limits([maps["ensemble_spread"]])
    spread_max = max(spread_max, 1.0e-12)
    if limits is not None:
        field_min, field_max = (float(value) for value in limits["field"])
        difference_low, difference_high = (float(value) for value in limits["difference"])
        if difference_low != -difference_high:
            raise RefinementEvaluationError("Difference plot limits must be symmetric about zero.")
        difference_limit = difference_high
        spread_min, spread_max = (float(value) for value in limits["spread"])
        if spread_min != 0 or spread_max <= 0 or field_min >= field_max or difference_limit <= 0:
            raise RefinementEvaluationError("Plot limits must be finite ordered physical ranges with zero-based spread.")
        if not np.isfinite([field_min, field_max, difference_limit, spread_max]).all():
            raise RefinementEvaluationError("Plot limits must be finite.")

    figure, axes = plt.subplots(
        2,
        4,
        figsize=(20, 10),
        constrained_layout=True,
    )
    panels = (
        (
            "ground_truth",
            "Ground truth climatology",
            "viridis",
            field_min,
            field_max,
        ),
        (
            "phase1",
            "Phase 1 U-Net climatology",
            "viridis",
            field_min,
            field_max,
        ),
        (
            "ensemble_mean",
            f"{refinement_type} ensemble-mean climatology",
            "viridis",
            field_min,
            field_max,
        ),
        (
            "phase1_bias",
            "Phase 1 minus ground truth",
            "BrBG",
            -difference_limit,
            difference_limit,
        ),
        (
            "ensemble_mean_bias",
            f"{refinement_type} minus ground truth",
            "BrBG",
            -difference_limit,
            difference_limit,
        ),
        (
            "correction",
            "Refinement correction relative to Phase 1",
            "BrBG",
            -difference_limit,
            difference_limit,
        ),
        (
            "representative",
            "Representative ensemble member 0 climatology",
            "viridis",
            field_min,
            field_max,
        ),
        (
            "ensemble_spread",
            "Mean spatial standard deviation across members",
            "magma",
            0.0,
            spread_max,
        ),
    )
    images: list[Any] = []
    for axis, (key, title, cmap, vmin, vmax) in zip(axes.flat, panels):
        field = member_maps[0] if key == "representative" else maps[key]
        images.append(
            _draw_map(
                axis,
                field,
                lat=lat,
                lon=lon,
                title=title,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
        )
    figure.colorbar(
        images[0],
        ax=[axes.flat[index] for index in (0, 1, 2, 6)],
        label=f"{variable} ({units})",
        shrink=0.85,
    )
    figure.colorbar(
        images[3],
        ax=[axes.flat[index] for index in (3, 4, 5)],
        label=f"Difference ({units})",
        shrink=0.85,
    )
    figure.colorbar(
        images[7],
        ax=axes.flat[7],
        label=f"Ensemble spread ({units})",
        shrink=0.85,
    )
    figure.suptitle(
        f"{variable}: matched-period climatology and residual-refinement "
        "diagnostics",
        fontsize=14,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_representative_members(
    maps: dict[str, np.ndarray],
    *,
    variable: str,
    units: str,
    lat: np.ndarray,
    lon: np.ndarray,
    output_path: str | Path,
    maximum_members: int = 8,
    field_limits: Sequence[float] | None = None,
) -> Path:
    """Write individual member climatologies without spatial filtering."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    members = maps["member_climatologies"]
    count = min(int(members.shape[0]), int(maximum_members))
    indices = np.linspace(0, members.shape[0] - 1, count, dtype=int)
    field_min, field_max = _finite_limits(
        [
            maps["ground_truth"],
            maps["phase1"],
            maps["ensemble_mean"],
            *list(members),
        ]
    )
    if field_limits is not None:
        field_min, field_max = (float(value) for value in field_limits)
    columns = min(4, count)
    rows = int(math.ceil(count / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5 * columns, 4.5 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for plot_index, member_index in enumerate(indices):
        image = _draw_map(
            axes.flat[plot_index],
            members[member_index],
            lat=lat,
            lon=lon,
            title=f"Member {member_index} climatology",
            cmap="viridis",
            vmin=field_min,
            vmax=field_max,
        )
    for axis in axes.flat[count:]:
        axis.set_visible(False)
    if image is not None:
        figure.colorbar(
            image,
            ax=list(axes.flat[:count]),
            label=f"{variable} ({units})",
            shrink=0.85,
        )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _flatten_scalars(
    value: Any,
    *,
    prefix: str = "",
) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten_scalars(item, prefix=child)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _flatten_scalars(item, prefix=f"{prefix}[{index}]")
    elif value is None or isinstance(value, (str, int, float, bool)):
        yield prefix, value


def plot_boundary_distance(metrics, *, variable, units, output_path):
    """Compare exact distance rings without cropping or smoothing predictions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regions = metrics["geographic_regions"]
    rings = sorted(int(name.split("_")[1]) for name in regions if name.startswith("distance_"))
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for model, label in (("phase1", "Phase 1"), ("ensemble_mean", "Refinement ensemble mean")):
        for axis, metric, title in zip(
            axes, ("mean_bias", "mean_absolute_bias", "climatological_rmse"),
            ("Signed climatological bias", "Mean absolute gridpoint bias", "Climatological RMSE"),
        ):
            values = [regions[f"distance_{ring}"]["climatology"][model][metric] for ring in rings]
            axis.plot(rings, values, label=label)
            axis.set_title(title)
            axis.set_xlabel("Distance from boundary (cells)")
            axis.set_ylabel(units)
            axis.grid(alpha=0.25)
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].legend()
    figure.suptitle(variable)
    path = Path(output_path)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def write_evaluation_outputs(
    summary: dict[str, Any],
    maps_by_variable: dict[str, dict[str, np.ndarray]],
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    output_dir: str | Path,
    make_plots: bool = True,
    plot_limits: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    if plot_limits is None:
        plot_limits = shared_map_limits([maps_by_variable])
    summary["plot_limits"] = plot_limits
    for variable, maps in maps_by_variable.items():
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", variable)
        map_path = output / f"{safe_name}_diagnostic_maps.npz"
        np.savez_compressed(map_path, lat=np.asarray(lat), lon=np.asarray(lon), **maps)
        artifacts[f"{variable}_numerical_maps"] = str(map_path.resolve())
    if make_plots:
        for variable, maps in maps_by_variable.items():
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", variable)
            units = str(summary["variables"][variable]["units"])
            overview = plot_evaluation_maps(
                maps,
                variable=variable,
                units=units,
                refinement_type=str(summary["refinement_type"]),
                lat=np.asarray(lat),
                lon=np.asarray(lon),
                output_path=output / f"{safe_name}_refinement_8_panel.png",
                limits=plot_limits[variable],
            )
            members = plot_representative_members(
                maps,
                variable=variable,
                units=units,
                lat=np.asarray(lat),
                lon=np.asarray(lon),
                output_path=output / f"{safe_name}_representative_members.png",
                field_limits=plot_limits[variable]["field"],
            )
            if "geographic_regions" in summary["variables"][variable]:
                distance_plot = plot_boundary_distance(
                    summary["variables"][variable], variable=variable, units=units,
                    output_path=output / f"{safe_name}_error_by_boundary_distance.png",
                )
                artifacts[f"{variable}_boundary_distance_plot"] = str(distance_plot.resolve())
            artifacts[f"{variable}_eight_panel_plot"] = str(overview.resolve())
            artifacts[f"{variable}_representative_members_plot"] = str(
                members.resolve()
            )

    json_path = output / "refinement_evaluation_summary.json"
    csv_path = output / "refinement_evaluation_metrics.csv"
    artifacts["summary_json"] = str(json_path.resolve())
    artifacts["metrics_csv"] = str(csv_path.resolve())
    summary["artifacts"] = artifacts
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("scope", "metric", "value"),
        )
        writer.writeheader()
        for variable, metrics in summary["variables"].items():
            for name, value in _flatten_scalars(metrics):
                writer.writerow(
                    {"scope": variable, "metric": name, "value": value}
                )
    return artifacts


def evaluate_refinement_files(
    *,
    prediction_path: str | Path,
    baseline_path: str | Path,
    target_path: str | Path,
    variables: Sequence[str],
    expected_refinement_type: str,
    output_dir: str | Path,
    expected_checkpoint: str | Path | None = None,
    expected_case: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    require_all_prediction_times: bool = True,
    chunk_time: int = 32,
    distribution_samples: int = 1_000_000,
    spectral_bins: int = 8,
    wet_day_threshold: float = 1.0,
    wet_day_threshold_source: str = "explicit function argument or generic1mm/day default",
    make_plots: bool = True,
    plot_limits: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
) -> dict[str, Any]:
    """Evaluate explicit NetCDF files and write reproducible artifacts."""
    prediction_path = Path(prediction_path).expanduser().resolve()
    baseline_path = Path(baseline_path).expanduser().resolve()
    target_path = Path(target_path).expanduser().resolve()
    paths = (prediction_path, baseline_path, target_path)
    if len(set(paths)) != len(paths):
        raise RefinementEvaluationError(
            "Prediction, Phase-1 baseline, and target must be distinct files."
        )
    for path in paths:
        if not path.is_file():
            raise RefinementEvaluationError(
                f"Input NetCDF does not exist: {path}"
            )

    with (
        xr.open_dataset(prediction_path) as prediction,
        xr.open_dataset(baseline_path) as baseline,
        xr.open_dataset(target_path) as target,
    ):
        summary, maps = evaluate_refinement_datasets(
            prediction,
            baseline,
            target,
            variables=variables,
            expected_refinement_type=expected_refinement_type,
            expected_checkpoint=expected_checkpoint,
            expected_case=expected_case,
            start_date=start_date,
            end_date=end_date,
            require_all_prediction_times=require_all_prediction_times,
            chunk_time=chunk_time,
            distribution_samples=distribution_samples,
            spectral_bins=spectral_bins,
            wet_day_threshold=wet_day_threshold,
        )
        summary["wet_day_threshold_source"] = wet_day_threshold_source
        summary["inputs"] = {
            "prediction": str(prediction_path),
            "phase1_baseline": str(baseline_path),
            "ground_truth": str(target_path),
        }
        lat = np.asarray(prediction["lat"].values)
        lon = np.asarray(prediction["lon"].values)

    write_evaluation_outputs(
        summary,
        maps,
        lat=lat,
        lon=lon,
        output_dir=output_dir,
        make_plots=make_plots,
        plot_limits=plot_limits,
    )
    return summary


def resolve_evaluation_wet_threshold(
    explicit: float | None, case_config: str | Path | None = None,
) -> tuple[float, str]:
    """Resolve the evaluation threshold without conflating it with loss gates."""
    if explicit is not None:
        threshold, source = float(explicit), "explicit --wet-day-threshold"
    elif case_config is not None:
        import yaml
        path = Path(case_config).expanduser().resolve()
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(config, Mapping) or "precip_wet_threshold" not in config:
            raise RefinementEvaluationError(
                "Case config lacks precip_wet_threshold; supply --wet-day-threshold explicitly."
            )
        threshold, source = float(config["precip_wet_threshold"]), str(path) + ":precip_wet_threshold"
    else:
        threshold, source = 1.0, "generic1mm/day convention; not inferred from an experiment config"
    if not math.isfinite(threshold) or threshold < 0:
        raise RefinementEvaluationError("Wet-day threshold must be finite and non-negative.")
    return threshold, source


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a physical CORDEX residual-refinement ensemble against "
            "an explicit Phase-1 baseline and ground-truth NetCDF."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--variables", nargs="+", required=True)
    parser.add_argument(
        "--expected-refinement-type",
        required=True,
        choices=SUPPORTED_REFINEMENT_TYPES,
    )
    parser.add_argument(
        "--expected-checkpoint",
        required=True,
        help=(
            "Exact schema-v2 checkpoint used to generate the output. Its bytes, "
            "scientific contract, Phase-1 identity, and normalizer state are "
            "authenticated before evaluation."
        ),
    )
    parser.add_argument("--expected-case", default=None)
    parser.add_argument(
        "--start-date",
        default=None,
        help="Inclusive YYYY-MM-DD",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Inclusive YYYY-MM-DD",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-time", type=int, default=32)
    parser.add_argument("--plot-limits-json", help="Shared per-variable field/difference/spread limits for every compared output.")
    parser.add_argument("--distribution-samples", type=int, default=1_000_000)
    parser.add_argument("--spectral-bins", type=int, default=8)
    parser.add_argument("--wet-day-threshold", type=float, default=None,
                        help="Physical precipitation threshold in mm/day; overrides --case-config.")
    parser.add_argument("--case-config", help="Read precip_wet_threshold from the actual experiment YAML; otherwise1mm/day is a generic convention.")
    parser.add_argument(
        "--allow-incomplete-time-coverage",
        action="store_true",
        help="Allow prediction timestamps that have no exact target match.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write JSON/CSV only.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    wet_threshold, wet_threshold_source = resolve_evaluation_wet_threshold(args.wet_day_threshold, args.case_config)
    summary = evaluate_refinement_files(
        prediction_path=args.prediction,
        baseline_path=args.baseline,
        target_path=args.target,
        variables=args.variables,
        expected_refinement_type=args.expected_refinement_type,
        output_dir=args.output_dir,
        expected_checkpoint=args.expected_checkpoint,
        expected_case=args.expected_case,
        start_date=args.start_date,
        end_date=args.end_date,
        require_all_prediction_times=not args.allow_incomplete_time_coverage,
        chunk_time=args.chunk_time,
        distribution_samples=args.distribution_samples,
        spectral_bins=args.spectral_bins,
        wet_day_threshold=wet_threshold,
        wet_day_threshold_source=wet_threshold_source,
        make_plots=not args.no_plots,
        plot_limits=json.loads(Path(args.plot_limits_json).read_text(encoding="utf-8")) if args.plot_limits_json else None,
    )
    print(json.dumps(summary["artifacts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
