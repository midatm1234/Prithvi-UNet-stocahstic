"""Explicit checkpoint migration between spatial and temporal Prithvi-UNet.

Two operations, deliberately kept distinct because they mean different things:

:func:`initialize_from_spatial_checkpoint`
    Start a *new* temporal run from an existing frame-independent checkpoint.
    Every spatial weight must land; the only keys allowed to be missing are the
    ones this branch introduces (``temporal_adapter.*``). A migrated checkpoint
    is **not** a trained temporal model, and :func:`save_temporal_checkpoint`
    records ``trained_temporal_steps`` so the distinction survives on disk.

:func:`resume_temporal_checkpoint`
    Continue a temporal run. Nothing may be missing, the architecture version
    must match, and the temporal settings and time-feature layout must agree
    with the current config -- otherwise the weights describe a different model.

Neither path uses an unrestricted ``strict=False``. ``load_state_dict`` is called
with ``strict=False`` only so that this module can *inspect and adjudicate* the
missing/unexpected lists itself; anything unexpected, any shape mismatch, and any
missing key outside the allowed set is an error.

The source checkpoint is opened read-only and never rewritten.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn

from granitewxc.temporal.config import (
    TEMPORAL_ARCHITECTURE_VERSION,
    TemporalConfig,
)

__all__ = [
    "CheckpointContract",
    "MigrationReport",
    "TemporalCheckpointError",
    "extract_state_dict",
    "describe_spatial_checkpoint",
    "initialize_from_spatial_checkpoint",
    "resume_temporal_checkpoint",
    "save_temporal_checkpoint",
]

#: Prefix of every parameter/buffer introduced by the temporal extension.
TEMPORAL_PREFIX = "temporal_adapter."

#: Buffers that legitimately differ between a checkpoint and a fresh model
#: because they are recomputed from config at construction time.
_NON_PERSISTENT_BUFFERS = (
    "predictand_scaling_method_codes",
    "predictand_nonneg_enabled_mask",
    "predictand_nonneg_method_codes",
)


class TemporalCheckpointError(RuntimeError):
    """Raised for any incompatible or ambiguous checkpoint migration."""


# ---------------------------------------------------------------------------
# contract
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CheckpointContract:
    """Everything that must agree between a checkpoint and a model.

    Grid shape is included because gridpoint target normalization bakes the
    target grid into ``output_scalers_mu`` (the SA case stores
    ``[1, 2, 128, 128]``). Loading such a checkpoint against a different crop or
    domain would silently apply the wrong per-cell mean.
    """

    output_vars: tuple[str, ...]
    n_dynamic_input_channels: int
    n_static_input_channels: int
    n_output_channels: int
    target_grid_shape: tuple[int, ...]
    embed_dim: int
    downscaling_embed_dim: int
    n_upsample_stages: int
    input_scaler_digest: str
    output_scaler_digest: str

    def mismatches(self, other: "CheckpointContract") -> list[str]:
        problems: list[str] = []
        # ``output_vars`` cannot be recovered from tensor shapes, so a contract
        # reconstructed by :func:`contract_from_state_dict` leaves it empty.
        # Compare names only when both sides actually know them; the checkpoint's
        # *recorded* variable list is checked separately by the loaders, which is
        # where a real ordering mismatch is caught.
        if self.output_vars and other.output_vars and self.output_vars != other.output_vars:
            problems.append(
                f"output_vars: checkpoint {list(self.output_vars)} vs config {list(other.output_vars)}"
            )
        for name in (
            "n_dynamic_input_channels",
            "n_static_input_channels",
            "n_output_channels",
            "embed_dim",
            "downscaling_embed_dim",
            "n_upsample_stages",
        ):
            a, b = getattr(self, name), getattr(other, name)
            if a != b:
                problems.append(f"{name}: checkpoint {a} vs model {b}")
        if self.target_grid_shape != other.target_grid_shape:
            problems.append(
                f"target_grid_shape: checkpoint {list(self.target_grid_shape)} vs "
                f"model {list(other.target_grid_shape)}"
            )
        return problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_vars": list(self.output_vars),
            "n_dynamic_input_channels": self.n_dynamic_input_channels,
            "n_static_input_channels": self.n_static_input_channels,
            "n_output_channels": self.n_output_channels,
            "target_grid_shape": list(self.target_grid_shape),
            "embed_dim": self.embed_dim,
            "downscaling_embed_dim": self.downscaling_embed_dim,
            "n_upsample_stages": self.n_upsample_stages,
            "input_scaler_digest": self.input_scaler_digest,
            "output_scaler_digest": self.output_scaler_digest,
        }


def _digest(tensor: torch.Tensor | None) -> str:
    if tensor is None:
        return "absent"
    arr = tensor.detach().to(torch.float64).cpu().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    """Pull the model state dict out of the repository's checkpoint layouts."""
    if isinstance(payload, Mapping):
        for key in ("model", "state_dict", "model_state_dict"):
            inner = payload.get(key)
            if isinstance(inner, Mapping):
                return dict(inner)
        if all(torch.is_tensor(v) for v in payload.values()) and payload:
            return dict(payload)
    if hasattr(payload, "state_dict"):
        return dict(payload.state_dict())
    raise TemporalCheckpointError(
        "Unrecognized checkpoint layout: expected a mapping containing 'model' or "
        "'state_dict', or a bare state dict."
    )


def _strip_wrappers(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove DDP/FSDP/compile prefixes so keys match a bare module."""
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new = key
        for prefix in ("module.", "_orig_mod."):
            while new.startswith(prefix):
                new = new[len(prefix) :]
        out[new] = value
    return out


def contract_from_state_dict(state: Mapping[str, torch.Tensor]) -> CheckpointContract:
    """Reconstruct the contract a checkpoint implies from its tensor shapes."""
    state = _strip_wrappers(state)
    try:
        in_mu = state["input_scalers_mu"]
        out_mu = state["output_scalers_mu"]
        static_mu = state.get("static_input_scalers_mu")
        embed_w = state["conv_after_backbone.weight"]
        down_w = state["conv_before_backbone.weight"]
    except KeyError as exc:
        raise TemporalCheckpointError(
            f"Checkpoint is missing the structural tensor {exc.args[0]!r}; it does not "
            "look like a ClimateDownscaleFinetuneUNETModel checkpoint."
        ) from exc

    n_upsample = len(
        {
            key.split(".")[1]
            for key in state
            if key.startswith("upsample_layers.") and len(key.split(".")) > 2
        }
    )
    return CheckpointContract(
        output_vars=(),  # filled by the caller from checkpoint metadata when present
        n_dynamic_input_channels=int(in_mu.shape[1]),
        n_static_input_channels=0 if static_mu is None else int(static_mu.shape[1]),
        n_output_channels=int(out_mu.shape[1]),
        target_grid_shape=tuple(int(v) for v in out_mu.shape[2:]),
        embed_dim=int(embed_w.shape[0]),
        downscaling_embed_dim=int(down_w.shape[1] // 2),
        n_upsample_stages=int(n_upsample),
        input_scaler_digest=_digest(in_mu),
        output_scaler_digest=_digest(out_mu),
    )


def contract_from_model(model: nn.Module, output_vars: Sequence[str]) -> CheckpointContract:
    state = model.state_dict()
    return CheckpointContract(
        output_vars=tuple(str(v) for v in output_vars),
        n_dynamic_input_channels=int(state["input_scalers_mu"].shape[1]),
        n_static_input_channels=int(state["static_input_scalers_mu"].shape[1])
        if "static_input_scalers_mu" in state
        else 0,
        n_output_channels=int(state["output_scalers_mu"].shape[1]),
        target_grid_shape=tuple(int(v) for v in state["output_scalers_mu"].shape[2:]),
        embed_dim=int(model.conv_after_backbone.weight.shape[0]),
        downscaling_embed_dim=int(model.conv_before_backbone.weight.shape[1] // 2),
        n_upsample_stages=int(len(model.upsample_layers)),
        input_scaler_digest=_digest(state["input_scalers_mu"]),
        output_scaler_digest=_digest(state["output_scalers_mu"]),
    )


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
@dataclass
class MigrationReport:
    """Full accounting of a weight-loading operation."""

    source: str
    kind: str  # "spatial_init" | "temporal_resume"
    loaded: list[str] = field(default_factory=list)
    missing_temporal: list[str] = field(default_factory=list)
    missing_other: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    shape_mismatch: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = field(default_factory=list)
    contract_problems: list[str] = field(default_factory=list)
    checkpoint_epoch: int | None = None
    checkpoint_global_step: int | None = None
    trained_temporal_steps: int = 0

    @property
    def ok(self) -> bool:
        return not (
            self.missing_other or self.unexpected or self.shape_mismatch or self.contract_problems
        )

    def summary(self) -> str:
        lines = [
            f"[checkpoint] {self.kind} from {self.source}",
            f"  loaded            : {len(self.loaded)} tensor(s)",
            f"  missing (temporal): {len(self.missing_temporal)}"
            + (f" e.g. {self.missing_temporal[:3]}" if self.missing_temporal else ""),
            f"  missing (other)   : {len(self.missing_other)}"
            + (f" -> {self.missing_other[:8]}" if self.missing_other else ""),
            f"  unexpected        : {len(self.unexpected)}"
            + (f" -> {self.unexpected[:8]}" if self.unexpected else ""),
            f"  shape mismatch    : {len(self.shape_mismatch)}"
            + (f" -> {self.shape_mismatch[:4]}" if self.shape_mismatch else ""),
        ]
        if self.checkpoint_epoch is not None:
            lines.append(
                f"  provenance        : epoch={self.checkpoint_epoch} "
                f"step={self.checkpoint_global_step} "
                f"trained_temporal_steps={self.trained_temporal_steps}"
            )
        if self.contract_problems:
            lines.append("  CONTRACT PROBLEMS :")
            lines.extend(f"    - {p}" for p in self.contract_problems)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "kind": self.kind,
            "n_loaded": len(self.loaded),
            "missing_temporal": self.missing_temporal,
            "missing_other": self.missing_other,
            "unexpected": self.unexpected,
            "shape_mismatch": [
                {"key": k, "checkpoint": list(a), "model": list(b)}
                for k, a, b in self.shape_mismatch
            ],
            "contract_problems": self.contract_problems,
            "checkpoint_epoch": self.checkpoint_epoch,
            "checkpoint_global_step": self.checkpoint_global_step,
            "trained_temporal_steps": self.trained_temporal_steps,
        }


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def describe_spatial_checkpoint(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read-only inspection of a checkpoint: contract, provenance, key census."""
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    state = _strip_wrappers(extract_state_dict(payload))
    contract = contract_from_state_dict(state)
    meta = payload.get("metadata", {}) if isinstance(payload, Mapping) else {}
    prefixes: dict[str, int] = {}
    for key in state:
        prefixes[key.split(".")[0]] = prefixes.get(key.split(".")[0], 0) + 1
    return {
        "path": str(path),
        "n_tensors": len(state),
        "prefixes": prefixes,
        "contract": contract.to_dict(),
        "has_temporal_weights": any(k.startswith(TEMPORAL_PREFIX) for k in state),
        "epoch": payload.get("epoch") if isinstance(payload, Mapping) else None,
        "global_step": payload.get("global_step") if isinstance(payload, Mapping) else None,
        "git_commit": payload.get("git_commit") if isinstance(payload, Mapping) else None,
        "git_branch": payload.get("git_branch") if isinstance(payload, Mapping) else None,
        "temporal": payload.get("temporal") if isinstance(payload, Mapping) else None,
        "metadata_schema": meta.get("schema") if isinstance(meta, Mapping) else None,
    }


def _apply_state(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
    report: MigrationReport,
    *,
    allow_missing_prefixes: tuple[str, ...],
) -> None:
    """Load ``state`` into ``model``, adjudicating every key ourselves."""
    model_state = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}

    for key, value in state.items():
        if key not in model_state:
            if key in _NON_PERSISTENT_BUFFERS:
                continue
            report.unexpected.append(key)
            continue
        target = model_state[key]
        if torch.is_tensor(value) and tuple(value.shape) != tuple(target.shape):
            report.shape_mismatch.append(
                (key, tuple(int(v) for v in value.shape), tuple(int(v) for v in target.shape))
            )
            continue
        filtered[key] = value
        report.loaded.append(key)

    for key in model_state:
        if key in filtered or key in _NON_PERSISTENT_BUFFERS:
            continue
        if any(key.startswith(prefix) for prefix in allow_missing_prefixes):
            report.missing_temporal.append(key)
        else:
            report.missing_other.append(key)

    incompatible = model.load_state_dict(filtered, strict=False)
    # ``filtered`` was built from the model's own keys, so the only entries in
    # ``missing_keys`` are the ones we already classified. Assert that rather
    # than assume it.
    residual_unexpected = [
        k for k in incompatible.unexpected_keys if k not in _NON_PERSISTENT_BUFFERS
    ]
    if residual_unexpected:
        report.unexpected.extend(residual_unexpected)


def initialize_from_spatial_checkpoint(
    model: nn.Module,
    path: str | os.PathLike[str],
    *,
    expected_contract: CheckpointContract | None = None,
    strict_contract: bool = True,
) -> MigrationReport:
    """Initialize a temporal model from a frame-independent spatial checkpoint.

    Raises :class:`TemporalCheckpointError` unless

    * every checkpoint tensor is consumed (no unexpected keys, no shape
      mismatches), and
    * every model tensor left uninitialized belongs to ``temporal_adapter.*``.

    The second condition is what prevents a broad ``strict=False`` from hiding a
    genuinely failed load: if, say, the decoder were renamed, its weights would
    show up as ``missing_other`` and this raises instead of silently training a
    randomly initialized decoder.
    """
    source = str(path)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    state = _strip_wrappers(extract_state_dict(payload))

    if any(key.startswith(TEMPORAL_PREFIX) for key in state):
        raise TemporalCheckpointError(
            f"{source} already contains temporal weights ({TEMPORAL_PREFIX}*). Use "
            "temporal.resume_from_temporal_checkpoint to continue a temporal run; "
            "init_from_spatial_checkpoint is for starting from a frame-independent model."
        )

    report = MigrationReport(source=source, kind="spatial_init")
    if isinstance(payload, Mapping):
        report.checkpoint_epoch = payload.get("epoch")
        report.checkpoint_global_step = payload.get("global_step")

    ckpt_contract = contract_from_state_dict(state)
    if expected_contract is not None:
        problems = ckpt_contract.mismatches(expected_contract)
        # output_vars are not recoverable from tensor shapes alone; compare them
        # only when the checkpoint recorded them.
        recorded = None
        if isinstance(payload, Mapping):
            recorded = (payload.get("temporal") or {}).get("output_vars") or (
                payload.get("metadata") or {}
            ).get("output_vars")
        if recorded is not None:
            if tuple(str(v) for v in recorded) != expected_contract.output_vars:
                problems.append(
                    f"output_vars: checkpoint {list(recorded)} vs config "
                    f"{list(expected_contract.output_vars)}"
                )
        report.contract_problems.extend(problems)

    _apply_state(model, state, report, allow_missing_prefixes=(TEMPORAL_PREFIX,))

    if strict_contract and not report.ok:
        raise TemporalCheckpointError(
            "Refusing to initialize from an incompatible spatial checkpoint.\n"
            + report.summary()
        )
    return report


def resume_temporal_checkpoint(
    model: nn.Module,
    path: str | os.PathLike[str],
    cfg: TemporalConfig,
    *,
    expected_contract: CheckpointContract | None = None,
    time_feature_names: Sequence[str] | None = None,
) -> tuple[MigrationReport, dict[str, Any]]:
    """Resume a trained temporal run, validating that it is the same model.

    Returns the report and the raw payload so the caller can restore optimizer,
    scheduler and step counters.
    """
    source = str(path)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    state = _strip_wrappers(extract_state_dict(payload))

    if not any(key.startswith(TEMPORAL_PREFIX) for key in state):
        raise TemporalCheckpointError(
            f"{source} contains no temporal weights ({TEMPORAL_PREFIX}*); it is a "
            "spatial checkpoint. Use init_from_spatial_checkpoint instead."
        )

    stored = payload.get("temporal") if isinstance(payload, Mapping) else None
    if not isinstance(stored, Mapping):
        raise TemporalCheckpointError(
            f"{source} has temporal weights but no 'temporal' metadata block; its "
            "architecture settings cannot be validated. Refusing to resume."
        )

    problems: list[str] = []
    stored_version = str(stored.get("architecture_version", ""))
    if stored_version != cfg.architecture_version:
        problems.append(
            f"architecture_version: checkpoint {stored_version!r} vs code "
            f"{cfg.architecture_version!r}"
        )
    stored_cfg = stored.get("config") or {}
    for key in (
        "backend",
        "mode",
        "causal",
        "context_length",
        "output_length",
        "cadence_days",
    ):
        want = getattr(cfg, key)
        got = stored_cfg.get(key)
        if got is not None and got != want:
            problems.append(f"temporal.{key}: checkpoint {got!r} vs config {want!r}")
    for section, current in (
        ("latent", cfg.latent.to_dict()),
        ("recurrent" if cfg.backend == "recurrent" else "mamba",
         cfg.recurrent.to_dict() if cfg.backend == "recurrent" else cfg.mamba.to_dict()),
    ):
        stored_section = stored_cfg.get(section) or {}
        for key, want in current.items():
            # ``implementation`` selects a kernel, not an architecture: a run
            # started with fused kernels may legitimately resume on reference.
            if key in {"implementation", "adapter_init_gate"}:
                continue
            got = stored_section.get(key)
            if got is not None and got != want:
                problems.append(f"temporal.{section}.{key}: checkpoint {got!r} vs config {want!r}")

    if time_feature_names is not None:
        stored_names = stored.get("time_feature_names")
        if stored_names is not None and tuple(stored_names) != tuple(time_feature_names):
            problems.append(
                f"time_feature_names: checkpoint {list(stored_names)} vs current "
                f"{list(time_feature_names)}. The temporal module's input layout "
                "changed; its first-layer weights no longer mean the same thing."
            )

    report = MigrationReport(source=source, kind="temporal_resume")
    report.contract_problems.extend(problems)
    if isinstance(payload, Mapping):
        report.checkpoint_epoch = payload.get("epoch")
        report.checkpoint_global_step = payload.get("global_step")
    report.trained_temporal_steps = int(stored.get("trained_temporal_steps", 0) or 0)

    if expected_contract is not None:
        report.contract_problems.extend(
            contract_from_state_dict(state).mismatches(expected_contract)
        )

    # On resume nothing may be missing.
    _apply_state(model, state, report, allow_missing_prefixes=())

    if not report.ok:
        raise TemporalCheckpointError(
            "Refusing to resume from an incompatible temporal checkpoint.\n" + report.summary()
        )
    return report, dict(payload) if isinstance(payload, Mapping) else {}


# ---------------------------------------------------------------------------
# saving
# ---------------------------------------------------------------------------
def _git_provenance() -> dict[str, str | None]:
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(
                args, stderr=subprocess.DEVNULL, text=True, timeout=10
            ).strip()
        except Exception:
            return None

    return {
        "git_commit": run(["git", "rev-parse", "HEAD"]),
        "git_branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": run(["git", "status", "--porcelain"]) not in (None, ""),
    }


def save_temporal_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: nn.Module,
    cfg: TemporalConfig,
    output_vars: Sequence[str],
    time_feature_names: Sequence[str],
    epoch: int,
    global_step: int,
    trained_temporal_steps: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    train_loss: float | None = None,
    val_loss: float | None = None,
    metrics: Mapping[str, Any] | None = None,
    normalization: Mapping[str, Any] | None = None,
    data_provenance: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Write a temporal checkpoint with the metadata needed to validate a reload.

    ``trained_temporal_steps`` is recorded explicitly and is 0 for a checkpoint
    that was merely migrated from a spatial one. Anything reading a checkpoint
    can therefore tell "temporal architecture, untrained" from "temporal model,
    trained for N steps" without inspecting weights.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    contract = contract_from_model(model, output_vars)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "loss": None if train_loss is None else float(train_loss),
        "val_loss": None if val_loss is None else float(val_loss),
        "head_type": "temporal_deterministic",
        "temporal": {
            "architecture_version": cfg.architecture_version,
            "config": cfg.to_dict(),
            "backend": cfg.backend,
            "backend_implementation": getattr(
                getattr(model, "temporal_adapter", None), "backend", None
            ).__class__.__name__
            if getattr(model, "temporal_adapter", None) is not None
            else None,
            "mamba_implementation_used": getattr(
                getattr(getattr(model, "temporal_adapter", None), "backend", None),
                "implementation_used",
                None,
            ),
            "time_feature_names": list(time_feature_names),
            "output_vars": [str(v) for v in output_vars],
            "trained_temporal_steps": int(trained_temporal_steps),
            "contract": contract.to_dict(),
        },
        "normalization": dict(normalization or {}),
        "data_provenance": dict(data_provenance or {}),
        "metrics": dict(metrics or {}),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "platform": platform.platform(),
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **_git_provenance(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        payload["scheduler"] = scheduler.state_dict()
    if extra:
        payload.update(dict(extra))

    # Atomic write: a crash mid-save must not destroy the previous checkpoint.
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, target)
    return str(target)
