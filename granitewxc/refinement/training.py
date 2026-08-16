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
    CHECKPOINT_KIND_COMBINED,
    CHECKPOINT_KIND_REFINEMENT,
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


class RefinementTrainer:
    """Minimal, explicit Phase-2 trainer.

    Args:
        model: an initialised :class:`TwoPhaseDownscalingModel`.
        optimizer / scheduler / scaler: standard PyTorch objects.
        checkpoint_dir: destination for ``last.ckpt`` and ``best.ckpt``.
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
        self.case_name = case_name
        self.accum = max(1, int(gradient_accumulation_steps))
        self.max_grad_norm = max_grad_norm
        self.log_every = max(1, int(log_every))
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

    def save(self, *, is_best: bool = False, kind: str | None = None) -> str:
        kind = kind or (
            CHECKPOINT_KIND_COMBINED
            if self.model.refinement_config.joint_finetuning
            else CHECKPOINT_KIND_REFINEMENT
        )
        payload = self._payload(kind)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        last = os.path.join(self.checkpoint_dir, "last.ckpt")
        save_checkpoint_atomic(payload, last, atomic=self.atomic)
        if is_best:
            save_checkpoint_atomic(
                payload, os.path.join(self.checkpoint_dir, "best.ckpt"), atomic=self.atomic
            )
        self.log(f"[refinement] saved {kind} checkpoint -> {last}")
        return last

    def resume(self, path: str, *, strict_phase1_reference: bool = True) -> RefinementTrainState:
        """Restore model, optimizer, scheduler, scaler, epoch, step and RNG."""
        payload = torch.load(path, map_location="cpu", weights_only=False)
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
    ) -> float:
        self.model.train()
        total, count = 0.0, 0
        self.optimizer.zero_grad(set_to_none=True)

        n = limit_steps if limit_steps > 0 else (len(loader) if hasattr(loader, "__len__") else None)
        ep_label = f"Ep {epoch+1:03d} train"
        bar = tqdm(total=n, desc=ep_label, unit="batch", ascii=" #", ncols=100, leave=True)
        for step, batch in enumerate(loader):
            if limit_steps and step >= limit_steps:
                break
            batch = _move_batch(batch, self.device, self.non_blocking)
            with self._autocast():
                output = self.model.training_step(batch, generator=self._generator)
                loss = output.losses["loss"]
            if not torch.isfinite(loss):
                bar.close()
                raise RuntimeError(f"Non-finite refinement loss at step {step}: {loss.item()}")

            scaled = loss / self.accum
            if self.scaler is not None and self.scaler.is_enabled():
                self.scaler.scale(scaled).backward()
            else:
                scaled.backward()

            if (step + 1) % self.accum == 0:
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
                if self.scheduler is not None:
                    self.scheduler.step()
                self.state.global_step += 1

            total += float(loss.detach())
            count += 1
            bar.update(1)
            if count % self.log_every == 0:
                bar.set_postfix(
                    loss=f"{total / count:.4f}",
                    lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                )
                bar.refresh()
        bar.close()
        return total / max(1, count)

    @torch.no_grad()
    def validate(
        self,
        loader: Iterable[Mapping[str, Any]],
        limit_steps: int = 0,
        epoch: int = 0,
    ) -> float:
        self.model.eval()
        total, count = 0.0, 0
        n = limit_steps if limit_steps > 0 else (len(loader) if hasattr(loader, "__len__") else None)
        bar = tqdm(total=n, desc=f"Ep {epoch+1:03d} val  ", unit="batch", ascii=" #",
                   ncols=100, leave=True)
        for step, batch in enumerate(loader):
            if limit_steps and step >= limit_steps:
                break
            batch = _move_batch(batch, self.device, self.non_blocking)
            with self._autocast():
                output = self.model.training_step(batch, generator=self._generator)
            total += float(output.losses["loss"].detach())
            count += 1
            bar.update(1)
            if count % self.log_every == 0:
                bar.set_postfix(val_loss=f"{total / count:.4f}")
                bar.refresh()
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
        save_every: int | None = None,
    ) -> RefinementTrainState:
        # ``save_every`` is retained as a no-op for compatibility with older
        # notebooks. Refinement training intentionally keeps only best.ckpt
        # and the resumable last.ckpt.
        _ = save_every
        for ep in range(self.state.epoch, self.state.epoch + num_epochs):
            train_loss = self.train_one_epoch(
                train_loader, limit_steps=limit_steps_train, epoch=ep
            )
            self.state.train_loss_history.append(train_loss)
            val_loss = None
            if val_loader is not None:
                val_loss = self.validate(
                    val_loader, limit_steps=limit_steps_valid, epoch=ep
                )
                self.state.val_loss_history.append(val_loss)
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
            self.save(is_best=is_best)
        return self.state
