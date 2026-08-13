"""Dataset-agnostic Phase-2 (stochastic refinement) training loop.

Supported workflows
-------------------
1. Load an existing deterministic Phase-1 checkpoint and train only Phase 2
   (the default: Phase 1 frozen, in ``eval()``, run without gradients).
2. Resume an interrupted Phase-2 run from a refinement checkpoint.
3. Joint Phase-1 + Phase-2 fine-tuning, only when
   ``refinement.joint_finetuning: true`` is set explicitly.

Everything needed for an exact resume is persisted: model, optimizer,
scheduler, gradient-scaler, epoch, global step, RNG states, the fully resolved
configuration, the case identifier, the precision settings, the refinement type
and the Phase-1 checkpoint identity.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

import torch
from tqdm import tqdm

from granitewxc.refinement.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CHECKPOINT_KIND_COMBINED,
    CHECKPOINT_KIND_REFINEMENT,
    REFINEMENT_SEMANTICS_VERSION,
    build_refinement_checkpoint,
    load_refinement_state_dict,
    phase1_state_fingerprint,
    save_checkpoint_atomic,
    validate_phase1_reference,
)
from granitewxc.refinement.two_phase import TwoPhaseDownscalingModel

__all__ = ["RefinementTrainState", "RefinementTrainer"]

_PRIVATE_BATCH_KEYS = ("__scaler_offset", "__input_scaler_offset", "__output_scaler_offset", "__output_crop")


@dataclass
class RefinementTrainState:
    epoch: int = 0
    global_step: int = 0
    best_val_loss: float | None = None
    train_loss_history: list[float] = field(default_factory=list)
    val_loss_history: list[float] = field(default_factory=list)


def _move_batch(batch: Mapping[str, Any], device: torch.device, non_blocking: bool) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=non_blocking)
        else:
            moved[key] = value
    return moved


def _loader_steps(
    loader: Iterable[Mapping[str, Any]], limit_steps: int
) -> int | None:
    """Return the number of batches that this run will actually consume."""
    try:
        available = len(loader)  # type: ignore[arg-type]
    except (TypeError, NotImplementedError):
        available = None
    if limit_steps > 0:
        return min(limit_steps, available) if available is not None else limit_steps
    return available


def _flatten_contract(
    value: Any, prefix: str = ""
) -> dict[str, Any]:
    """Flatten a nested resume contract for concise mismatch diagnostics."""
    if isinstance(value, Mapping):
        flattened: dict[str, Any] = {}
        for key in sorted(value, key=str):
            name = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_contract(value[key], name))
        return flattened
    return {prefix: value}


def _require_exact_contract(
    label: str,
    saved: Mapping[str, Any],
    current: Mapping[str, Any],
) -> None:
    """Reject a resume when any persisted numerical/runtime setting differs."""
    saved_flat = _flatten_contract(saved)
    current_flat = _flatten_contract(current)
    differences = []
    for key in sorted(set(saved_flat) | set(current_flat)):
        old = saved_flat.get(key, "<missing>")
        new = current_flat.get(key, "<missing>")
        if old != new:
            differences.append(f"{key}: saved={old!r}, current={new!r}")
    if differences:
        detail = "; ".join(differences[:12])
        if len(differences) > 12:
            detail += f"; ... ({len(differences) - 12} more)"
        raise RuntimeError(
            f"Refinement checkpoint {label} does not match the current run; "
            f"refusing an inexact resume ({detail})."
        )


class RefinementTrainer:
    """Minimal, explicit Phase-2 trainer.

    Args:
        model: an initialised :class:`TwoPhaseDownscalingModel`.
        optimizer / scheduler / scaler: standard PyTorch objects.
        checkpoint_dir: destination for ``last.ckpt`` / ``best.ckpt`` /
            ``epoch_XXX.ckpt``.
        phase1_checkpoint: path of the deterministic checkpoint Phase 1 was
            loaded from. Recorded (with its fingerprint) in every refinement
            checkpoint so a mismatch is detected on resume.
        resolved_config: the fully normalized configuration, saved with the run.
    """

    def __init__(
        self,
        model: TwoPhaseDownscalingModel,
        optimizer: torch.optim.Optimizer,
        *,
        scheduler: Any = None,
        scaler: Any = None,
        device: torch.device | str = "cpu",
        checkpoint_dir: str = "./refinement_checkpoints",
        phase1_checkpoint: str | None = None,
        resolved_config: Mapping[str, Any] | None = None,
        case_name: str = "",
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float | None = None,
        seed: int | None = None,
        warmup_steps: int = 0,
        log_every: int = 20,
        logger: Callable[[str], None] = print,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.device = torch.device(device)
        self.checkpoint_dir = checkpoint_dir
        self.phase1_checkpoint = phase1_checkpoint
        self.resolved_config = dict(resolved_config or {})
        # Schema-2 checkpoints always carry complete model/runtime semantics,
        # even when a low-level caller omits the optional workflow mapping.
        self.resolved_config.setdefault(
            "refinement", model.refinement_config.to_dict()
        )
        self.resolved_config.setdefault(
            "performance", model.performance_config.to_dict()
        )
        self.case_name = case_name
        self.accum = max(1, int(gradient_accumulation_steps))
        self.max_grad_norm = max_grad_norm
        self.log_every = max(1, int(log_every))
        self.warmup_steps = max(0, int(warmup_steps))
        self._base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        if self.warmup_steps:
            for group, base_lr in zip(
                self.optimizer.param_groups, self._base_lrs, strict=True
            ):
                group["lr"] = base_lr / float(self.warmup_steps)
        self.log = logger
        self.state = RefinementTrainState()

        self.atomic = model.performance_config.io.atomic_checkpoints
        self.non_blocking = model.performance_config.dataloader.non_blocking_transfer

        self._phase1_fingerprint = phase1_state_fingerprint(
            {k[len("phase1.") :]: v for k, v in model.state_dict().items() if k.startswith("phase1.")}
        )
        self._generator = torch.Generator(device="cpu")
        if seed is not None:
            self._generator.manual_seed(int(seed))
            torch.manual_seed(int(seed))

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def _payload(self, kind: str) -> dict[str, Any]:
        payload = build_refinement_checkpoint(
            self.model,
            kind=kind,
            phase1_checkpoint=self.phase1_checkpoint,
            phase1_fingerprint=self._phase1_fingerprint,
            resolved_config=self.resolved_config,
            epoch=self.state.epoch,
            global_step=self.state.global_step,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            extra={
                "case_name": self.case_name,
                "refinement_type": self.model.refinement_config.type,
                "precision": self.model.performance_config.precision.to_dict(),
                "train_loss_history": list(self.state.train_loss_history),
                "val_loss_history": list(self.state.val_loss_history),
                "best_val_loss": self.state.best_val_loss,
                "generator_state": self._generator.get_state(),
            },
        )
        return payload

    def save(self, *, is_best: bool = False, kind: str | None = None, epoch_file: bool = True) -> str:
        kind = kind or (
            CHECKPOINT_KIND_COMBINED
            if not self.model.phase1_inference_only
            else CHECKPOINT_KIND_REFINEMENT
        )
        payload = self._payload(kind)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        last = os.path.join(self.checkpoint_dir, "last.ckpt")
        save_checkpoint_atomic(payload, last, atomic=self.atomic)
        if epoch_file:
            save_checkpoint_atomic(
                payload,
                os.path.join(self.checkpoint_dir, f"epoch_{self.state.epoch:03d}.ckpt"),
                atomic=self.atomic,
            )
        if is_best:
            save_checkpoint_atomic(
                payload, os.path.join(self.checkpoint_dir, "best.ckpt"), atomic=self.atomic
            )
        self.log(f"[refinement] saved {kind} checkpoint -> {last}")
        return last

    def resume(self, path: str, *, strict_phase1_reference: bool = True) -> RefinementTrainState:
        """Restore model, optimizer, scheduler, scaler, epoch, step and RNG."""
        payload = torch.load(path, map_location="cpu", weights_only=False)
        schema = payload.get("checkpoint_schema_version")
        semantics = payload.get("refinement_semantics_version")
        if (
            schema != CHECKPOINT_SCHEMA_VERSION
            or semantics != REFINEMENT_SEMANTICS_VERSION
        ):
            raise RuntimeError(
                "Phase-2 checkpoint predates the encoded-final-Phase1 "
                "residual semantics and cannot be resumed safely: "
                f"schema={schema!r}, semantics={semantics!r}; expected "
                f"schema={CHECKPOINT_SCHEMA_VERSION}, "
                f"semantics={REFINEMENT_SEMANTICS_VERSION!r}. Retrain or "
                "explicitly migrate the refinement checkpoint."
            )
        saved_type = payload.get("refinement_type")
        if saved_type and saved_type != self.model.refinement_config.type:
            raise RuntimeError(
                f"Refinement checkpoint type {saved_type!r} cannot resume a "
                f"{self.model.refinement_config.type!r} model."
            )
        saved_refinement = (payload.get("resolved_config") or {}).get(
            "refinement"
        )
        if not isinstance(saved_refinement, Mapping):
            raise RuntimeError(
                "Schema-2 refinement checkpoint lacks the complete "
                "resolved_config.refinement contract."
            )
        current_refinement = self.model.refinement_config.to_dict()
        # Paths and requested inference ensemble size may legitimately move;
        # training/model semantics may not. Schema 2 was introduced with the
        # current full mapping, so absent fields are not treated as defaults.
        saved_refinement = dict(saved_refinement)
        for mutable in ("checkpoint", "ensemble_size"):
            saved_refinement.pop(mutable, None)
            current_refinement.pop(mutable, None)
        if saved_refinement != current_refinement:
            raise RuntimeError(
                "Refinement checkpoint configuration does not match the current "
                "training/model configuration; refusing a partial or reinitialized resume."
            )

        # Schema-2 Phase-2 checkpoints persist the complete numerical training
        # contract. Validate it before loading any model/optimizer state so a
        # changed warmup, accumulation factor, cosine horizon, epoch budget or
        # step limit cannot silently alter an interrupted trajectory.
        saved_resolved = payload.get("resolved_config") or {}
        saved_training = saved_resolved.get("training")
        current_training = self.resolved_config.get("training")
        if not isinstance(saved_training, Mapping):
            raise RuntimeError(
                "Schema-2 refinement checkpoint must contain a complete "
                "resolved_config.training mapping."
            )
        elif not isinstance(current_training, Mapping):
            raise RuntimeError(
                "Refinement checkpoint records a training contract, but the current "
                "trainer does not provide resolved_config.training; refusing an "
                "unverifiable resume."
            )
        else:
            _require_exact_contract(
                "training contract", saved_training, current_training
            )

        saved_performance = saved_resolved.get("performance")
        current_performance = self.model.performance_config.to_dict()
        if not isinstance(saved_performance, Mapping):
            raise RuntimeError(
                "Schema-2 refinement checkpoint must contain a complete "
                "resolved_config.performance mapping."
            )
        else:
            _require_exact_contract(
                "performance contract", saved_performance, current_performance
            )

        saved_precision = payload.get("precision")
        current_precision = self.model.performance_config.precision.to_dict()
        if not isinstance(saved_precision, Mapping):
            raise RuntimeError(
                "Schema-2 refinement checkpoint must contain complete precision "
                "metadata."
            )
        else:
            _require_exact_contract(
                "precision contract", saved_precision, current_precision
            )
        phase1_state = {
            k[len("phase1.") :]: v
            for k, v in self.model.state_dict().items()
            if k.startswith("phase1.")
        }
        validate_phase1_reference(payload, phase1_state, strict=strict_phase1_reference)

        if payload.get("checkpoint_kind") == CHECKPOINT_KIND_COMBINED:
            self.model.load_state_dict(payload["model"], strict=True)
        else:
            load_refinement_state_dict(self.model, payload)

        if "optimizer" in payload and payload["optimizer"] is not None:
            self.optimizer.load_state_dict(payload["optimizer"])
        if self.scheduler is not None and payload.get("scheduler"):
            self.scheduler.load_state_dict(payload["scheduler"])
        if self.scaler is not None and payload.get("scaler"):
            self.scaler.load_state_dict(payload["scaler"])

        self.state.epoch = int(payload.get("epoch", 0))
        self.state.global_step = int(payload.get("global_step", 0))
        self.state.best_val_loss = payload.get("best_val_loss")
        self.state.train_loss_history = list(payload.get("train_loss_history", []))
        self.state.val_loss_history = list(payload.get("val_loss_history", []))

        rng = payload.get("rng_state") or {}
        if rng.get("cpu") is not None:
            torch.set_rng_state(rng["cpu"].to(torch.uint8) if torch.is_tensor(rng["cpu"]) else rng["cpu"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(rng["cuda"])
            except Exception as exc:  # pragma: no cover - device count may differ
                self.log(f"[refinement] warning: could not restore CUDA RNG state ({exc})")
        if payload.get("generator_state") is not None:
            self._generator.set_state(payload["generator_state"])

        self.log(
            f"[refinement] resumed from {path} at epoch={self.state.epoch} "
            f"step={self.state.global_step}"
        )
        return self.state

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _autocast(self):
        mode = self.model.performance_config.precision.mode
        if mode == "fp32" or self.device.type != "cuda":
            return torch.autocast(device_type="cpu", enabled=False)
        dtype = torch.bfloat16 if mode == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def train_one_epoch(
        self,
        loader: Iterable[Mapping[str, Any]],
        limit_steps: int = 0,
        epoch: int = 0,
        progress_bar: Any | None = None,
    ) -> float:
        self.model.train()
        sampler = getattr(loader, "sampler", None)
        set_epoch = getattr(sampler, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)
        total, count = 0.0, 0
        self.optimizer.zero_grad(set_to_none=True)

        n = _loader_steps(loader, limit_steps)
        ep_label = f"Ep {epoch+1:03d} train"
        owns_bar = progress_bar is None
        bar = (
            progress_bar
            if progress_bar is not None
            else tqdm(
                total=n,
                desc=ep_label,
                unit="batch",
                ascii=" #",
                ncols=100,
                leave=True,
                mininterval=1.0,
            )
        )
        bar.set_postfix(stage="train", refresh=True)
        try:
            for step, batch in enumerate(loader):
                if limit_steps and step >= limit_steps:
                    break
                batch = _move_batch(batch, self.device, self.non_blocking)
                with self._autocast():
                    output = self.model.training_step(batch, generator=self._generator)
                    loss = output.losses["loss"]
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite refinement loss at step {step}: {loss.item()}"
                    )

                # Scale by the actual accumulation group size. The final partial
                # group must not be discarded or underweighted (important for smoke
                # tests and tiny-overfit runs where batches < accumulation steps).
                if n is not None:
                    group_start = (step // self.accum) * self.accum
                    group_size = min(self.accum, n - group_start)
                else:
                    group_size = self.accum
                scaled = loss / max(1, group_size)
                if self.scaler is not None and self.scaler.is_enabled():
                    self.scaler.scale(scaled).backward()
                else:
                    scaled.backward()

                group_boundary = (step + 1) % self.accum == 0
                final_known_batch = n is not None and (step + 1) == n
                if group_boundary or final_known_batch:
                    if self.max_grad_norm:
                        if self.scaler is not None and self.scaler.is_enabled():
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            self.max_grad_norm,
                        )
                    if self.scaler is not None and self.scaler.is_enabled():
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.state.global_step += 1
                    if self.state.global_step < self.warmup_steps:
                        factor = float(self.state.global_step + 1) / float(
                            self.warmup_steps
                        )
                        for group, base_lr in zip(
                            self.optimizer.param_groups, self._base_lrs, strict=True
                        ):
                            group["lr"] = base_lr * factor
                    elif self.scheduler is not None:
                        self.scheduler.step()

                total += float(loss.detach())
                count += 1
                bar.update(1)
                if count % self.log_every == 0:
                    bar.set_postfix(
                        stage="train",
                        train_loss=f"{total / count:.4f}",
                        lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    )
        finally:
            if owns_bar:
                bar.close()
        # Iterable loaders without __len__ cannot announce the final group. If
        # one remains, rescale its gradients from /accum to /actual_count and
        # flush it once rather than silently losing the update.
        if n is None and count % self.accum:
            remainder = count % self.accum
            factor = float(self.accum) / float(remainder)
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(factor)
            if self.max_grad_norm:
                if self.scaler is not None and self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], self.max_grad_norm
                )
            if self.scaler is not None and self.scaler.is_enabled():
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.state.global_step += 1
            if self.state.global_step < self.warmup_steps:
                factor = float(self.state.global_step + 1) / float(self.warmup_steps)
                for group, base_lr in zip(
                    self.optimizer.param_groups, self._base_lrs, strict=True
                ):
                    group["lr"] = base_lr * factor
            elif self.scheduler is not None:
                self.scheduler.step()
        return total / max(1, count)

    @torch.no_grad()
    def validate(
        self,
        loader: Iterable[Mapping[str, Any]],
        limit_steps: int = 0,
        epoch: int = 0,
        progress_bar: Any | None = None,
    ) -> float:
        self.model.eval()
        total, count = 0.0, 0
        n = _loader_steps(loader, limit_steps)
        owns_bar = progress_bar is None
        bar = (
            progress_bar
            if progress_bar is not None
            else tqdm(
                total=n,
                desc=f"Ep {epoch+1:03d} val  ",
                unit="batch",
                ascii=" #",
                ncols=100,
                leave=True,
                mininterval=1.0,
            )
        )
        bar.set_postfix(stage="validation", refresh=True)
        try:
            for step, batch in enumerate(loader):
                if limit_steps and step >= limit_steps:
                    break
                batch = _move_batch(batch, self.device, self.non_blocking)
                with self._autocast():
                    output = self.model.training_step(batch, generator=self._generator)
                    loss = output.losses["loss"]
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        "Non-finite refinement validation loss at step "
                        f"{step}: {loss.item()}"
                    )
                total += float(loss.detach())
                count += 1
                bar.update(1)
                if count % self.log_every == 0:
                    bar.set_postfix(
                        stage="validation", val_loss=f"{total / count:.4f}"
                    )
        finally:
            if owns_bar:
                bar.close()
        return total / max(1, count)

    def fit(
        self,
        train_loader: Iterable[Mapping[str, Any]],
        val_loader: Iterable[Mapping[str, Any]] | None = None,
        *,
        num_epochs: int = 1,
        limit_steps_train: int = 0,
        limit_steps_valid: int = 0,
        save_every: int = 1,
    ) -> RefinementTrainState:
        final_epoch = self.state.epoch + num_epochs
        for ep in range(self.state.epoch, final_epoch):
            train_steps = _loader_steps(train_loader, limit_steps_train)
            val_steps = (
                _loader_steps(val_loader, limit_steps_valid)
                if val_loader is not None
                else 0
            )
            progress_total = (
                train_steps + val_steps
                if train_steps is not None and val_steps is not None
                else None
            )
            epoch_bar = tqdm(
                total=progress_total,
                desc=f"Epoch {ep+1:03d}/{final_epoch:03d}",
                unit="batch",
                ascii=" #",
                ncols=100,
                leave=True,
                mininterval=1.0,
            )
            try:
                train_loss = self.train_one_epoch(
                    train_loader,
                    limit_steps=limit_steps_train,
                    epoch=ep,
                    progress_bar=epoch_bar,
                )
                self.state.train_loss_history.append(train_loss)
                val_loss = None
                if val_loader is not None:
                    epoch_bar.set_postfix(
                        stage="validation", train_loss=f"{train_loss:.4f}"
                    )
                    val_loss = self.validate(
                        val_loader,
                        limit_steps=limit_steps_valid,
                        epoch=ep,
                        progress_bar=epoch_bar,
                    )
                    self.state.val_loss_history.append(val_loss)
                final_postfix = {
                    "stage": "complete",
                    "train_loss": f"{train_loss:.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                }
                if val_loss is not None:
                    final_postfix["val_loss"] = f"{val_loss:.4f}"
                epoch_bar.set_postfix(**final_postfix)
            finally:
                epoch_bar.close()
            is_best = val_loss is not None and (
                self.state.best_val_loss is None or val_loss < self.state.best_val_loss
            )
            if is_best:
                self.state.best_val_loss = val_loss
            val_str = "n/a" if val_loss is None else f"{val_loss:.4f}"
            best_str = "" if self.state.best_val_loss is None else f"  best={self.state.best_val_loss:.4f}"
            marker = "  *** new best ***" if is_best else ""
            print(f"Ep {ep+1:03d}  train={train_loss:.4f}  val={val_str}{best_str}{marker}")
            # ``state.epoch`` is advanced *before* saving so a checkpoint records
            # the number of completed epochs, i.e. the epoch to resume at.
            self.state.epoch += 1
            epoch_file = save_every > 0 and self.state.epoch % save_every == 0
            self.save(is_best=is_best, epoch_file=epoch_file)
        return self.state
