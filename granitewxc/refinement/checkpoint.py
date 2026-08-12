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
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import torch

__all__ = [
    "CHECKPOINT_KIND_PHASE1",
    "CHECKPOINT_KIND_REFINEMENT",
    "CHECKPOINT_KIND_COMBINED",
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


def _classify(keys: Iterable[str], prefix: str) -> tuple[set[str], set[str]]:
    inside, outside = set(), set()
    for key in keys:
        (inside if key.startswith(prefix) else outside).add(key)
    return inside, outside


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
    raw = extract_model_state(checkpoint)
    migrated, renames = migrate_phase1_state_dict(raw, target_prefix=target_prefix)

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

    unexplained = [k for k in report.missing if not k.startswith("refiner.")]
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

    Only keys under ``prefix`` are applied; the Phase-1 sub-module is left
    exactly as it was.
    """
    raw = extract_model_state(checkpoint)
    stripped = strip_wrapper_prefixes(raw)
    incoming = {k: v for k, v in stripped.items() if k.startswith(prefix)}
    if not incoming:
        # Bare refiner state dict (no wrapper prefix).
        incoming = {prefix + k: v for k, v in stripped.items()}

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

    refiner_keys, _ = _classify(model_state, prefix)
    report.missing = sorted(refiner_keys - set(incoming))

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
        model_state = {k: v for k, v in full_state.items() if k.startswith("refiner.")}
    elif kind == CHECKPOINT_KIND_PHASE1:
        model_state = {k: v for k, v in full_state.items() if k.startswith("phase1.")}
    else:
        model_state = dict(full_state)

    payload: dict[str, Any] = {
        "checkpoint_kind": kind,
        "checkpoint_schema_version": 1,
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
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        sched_state = scheduler.state_dict()
        payload["scheduler"] = {k: v for k, v in sched_state.items() if k != "anneal_func"}
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if extra:
        payload.update(dict(extra))
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
