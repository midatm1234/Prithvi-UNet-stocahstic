"""Training loop for the temporal Prithvi-UNet.

Deliberate choices:

* **The per-frame loss is the repository's existing loss**
  (:func:`granitewxc.models.loss.build_loss_fn`), called once per emitted frame
  with exactly the batch layout the frame trainer uses. Temporal terms are added
  on top. This keeps the temporal run comparable to the frame-independent
  baseline: any measured difference comes from the model, not from a rewritten
  objective.
* **Invalid targets are restored to NaN before the per-frame loss.** The sequence
  dataset stores a finite-filled ``y`` plus an explicit mask; the existing loss
  does its own ``isfinite`` masking, so re-inserting NaN reproduces the frame
  trainer's masking behaviour byte-for-byte rather than approximating it.
* **Staged unfreezing is re-applied every epoch** and the resulting trainable
  groups are logged, so the run record shows what was actually optimized.
* **Validation selects the checkpoint.** Loss weights and early stopping look at
  validation only; the held-out test period is never consulted during training.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from granitewxc.models.loss import build_loss_fn
from granitewxc.temporal.checkpoint import (
    CheckpointContract,
    MigrationReport,
    contract_from_model,
    initialize_from_spatial_checkpoint,
    resume_temporal_checkpoint,
    save_temporal_checkpoint,
)
from granitewxc.temporal.config import TemporalConfig
from granitewxc.temporal.losses import TemporalLossComputer, TemporalLossTerms
from granitewxc.temporal.model import (
    TemporalSequenceModel,
    apply_freeze_policy,
    attach_temporal_adapter,
    build_param_groups,
)
from granitewxc.temporal.sequence_dataset import (
    TemporalSequenceDataset,
    collate_sequences,
)

__all__ = [
    "TemporalTrainingState",
    "resolve_crop_size",
    "resolve_static_channels",
    "seed_everything",
    "build_sequence_dataloaders",
    "build_temporal_model",
    "restore_nan_targets",
    "variable_scales_from_model",
    "train_temporal_model",
]


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch, including CUDA."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def restore_nan_targets(
    y: torch.Tensor, valid_mask: torch.Tensor | None
) -> torch.Tensor:
    """Re-insert NaN where the target was invalid.

    The sequence dataset returns a finite-filled ``y`` (so collation and the
    temporal terms are well behaved) plus an explicit validity mask. The existing
    per-frame loss detects invalid cells with ``torch.isfinite``, so putting the
    NaNs back is what makes its masking identical to the frame trainer's.
    """
    if valid_mask is None:
        return y
    return torch.where(valid_mask, y, torch.full_like(y, float("nan")))


def variable_scales_from_model(
    model: torch.nn.Module, output_vars: Sequence[str]
) -> dict[str, float]:
    """Characteristic magnitude per target variable, from the training scalers.

    Uses ``output_scalers_sigma`` -- the same statistics the model was fitted
    with, computed on the training split only. Deriving these from the current
    batch instead would make the objective drift between batches.
    """
    scales: dict[str, float] = {}
    sigma = getattr(model, "output_scalers_sigma", None)
    if sigma is None:
        return {name: 1.0 for name in output_vars}
    sigma = sigma.detach()
    for idx, name in enumerate(output_vars):
        if idx >= sigma.shape[1]:
            scales[str(name)] = 1.0
            continue
        value = float(sigma[0, idx].abs().mean().item())
        scales[str(name)] = value if value > 0 else 1.0
    return scales


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def _split_dates(config: Any, split: str) -> tuple[str | None, str | None]:
    """Resolve the date bounds for a split.

    Looks in ``temporal_splits`` first (the temporal configs declare explicit,
    non-overlapping ranges), then falls back to the legacy ``dates:`` block used
    by the NARR/PRISM workflow.
    """
    explicit = getattr(config, "temporal_splits", None)
    if isinstance(explicit, Mapping) and split in explicit:
        entry = explicit[split] or {}
        return entry.get("start"), entry.get("end")
    legacy = getattr(config, "dates", None)
    if isinstance(legacy, Mapping):
        alias = {"train": "training", "validation": "validation", "test": "inference"}
        entry = legacy.get(alias.get(split, split)) or {}
        return entry.get("start"), entry.get("end")
    return None, None


def resolve_static_channels(config: Any) -> int:
    """Number of trailing channels of ``x`` that are time-invariant static fields.

    ``num_static_channels: 0`` is a **meaningful value**, not a missing one: the
    NARR/PRISM configs route elevation and the predictor masks through the dynamic
    channel list instead of a separate static tensor. Coercing 0 to 1 would strip a
    real predictor channel off every frame.
    """
    if not bool(getattr(config.data, "use_static", True)):
        return 0
    return int(getattr(config.model, "num_static_channels", 1))


def resolve_crop_size(config: Any, source: Any = None) -> tuple[int, int]:
    """Spatial crop for sequence windows.

    The two cases declare their geometry differently and neither is wrong:

    * CORDEX sets ``target_size_lat/lon`` *and* ``train_crop_size_lat/lon``.
    * NARR/PRISM sets only ``train_crop_size_lat/lon`` -- the full target grid comes
      from the PRISM file, so there is no ``target_size_*`` key at all.

    So we prefer the training crop, fall back to ``target_size_*`` when present, and
    finally to the frame source's actual fine grid. Reading ``target_size_lat``
    unconditionally raises ``AttributeError`` on the NARR configs.
    """
    data = config.data
    lat = getattr(data, "train_crop_size_lat", None)
    lon = getattr(data, "train_crop_size_lon", None)
    if lat is None or lon is None:
        lat = getattr(data, "target_size_lat", lat)
        lon = getattr(data, "target_size_lon", lon)
    if lat is None or lon is None:
        if source is None:
            raise ValueError(
                "Cannot resolve a crop size: the config defines neither "
                "train_crop_size_lat/lon nor target_size_lat/lon, and no frame "
                "source was supplied to read the target grid from."
            )
        fine = source.fine_shape()
        lat, lon = int(fine[0]), int(fine[1])
    return int(lat), int(lon)


def build_sequence_dataloaders(
    config: Any,
    cfg: TemporalConfig,
    *,
    splits: Sequence[str] = ("train", "validation"),
    batch_size: int = 1,
    num_workers: int = 0,
    verbose: bool = True,
) -> dict[str, DataLoader]:
    """Build sequence dataloaders for the requested splits.

    Splits are narrowed by date *before* windows are constructed, so no window
    can straddle a split boundary.
    """
    from granitewxc.temporal.sources import build_frame_source

    static_channels = resolve_static_channels(config)

    loaders: dict[str, DataLoader] = {}
    for split in splits:
        source = build_frame_source(config, split)
        crop = resolve_crop_size(config, source)
        start, end = _split_dates(config, split)
        dataset = TemporalSequenceDataset(
            source,
            window_length=cfg.context_length,
            stride=cfg.sequence_stride if split == "train" else cfg.output_length,
            cadence_days=cfg.cadence_days,
            crop_size=crop,
            random_crop=(split == "train"),
            seed=cfg.seed,
            static_channels=static_channels,
            date_start=start,
            date_end=end,
            include_hour_of_day=cfg.include_hour_of_day,
            include_lead_time=cfg.include_lead_time,
            lead_time_days=cfg.lead_time_days,
        )
        if verbose:
            print(f"[data:{split}] {json.dumps(dataset.describe(), default=str)}")
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=collate_sequences,
            drop_last=(split == "train"),
        )
    return loaders


def resolve_time_feature_dim(loader: DataLoader) -> tuple[int, list[str]]:
    """Read the temporal-metadata width and channel names from one sample.

    Taking this from the data rather than from a hard-coded constant means the
    adapter's input layout always matches what the dataset actually emits (for
    example, hour-of-day channels appear only for a sub-daily archive).
    """
    dataset = loader.dataset
    sample = dataset[0]
    spec = getattr(dataset, "time_feature_spec", None)
    dim = int(sample["time_features"].shape[-1])
    names = list(spec.names) if spec is not None else [f"f{i}" for i in range(dim)]
    return dim, names


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def build_temporal_model(
    config: Any,
    cfg: TemporalConfig,
    *,
    time_feature_dim: int,
    time_feature_names: Sequence[str],
    device: torch.device | str = "cpu",
    verbose: bool = True,
) -> tuple[TemporalSequenceModel, MigrationReport, dict[str, Any]]:
    """Build the spatial model, attach the adapter, and load weights.

    Exactly one of ``init_from_spatial_checkpoint`` /
    ``resume_from_temporal_checkpoint`` drives weight loading, and the two are
    reported distinctly so a run log never confuses "migrated" with "trained".
    """
    from granitewxc.models.model import get_finetune_model_UNET

    base = get_finetune_model_UNET(config)
    adapter = attach_temporal_adapter(base, cfg, time_feature_dim=time_feature_dim)
    contract = contract_from_model(base, list(config.data.output_vars))

    resume_payload: dict[str, Any] = {}
    if cfg.resume_from_temporal_checkpoint:
        report, resume_payload = resume_temporal_checkpoint(
            base,
            cfg.resume_from_temporal_checkpoint,
            cfg,
            expected_contract=contract,
            time_feature_names=time_feature_names,
        )
    elif cfg.init_from_spatial_checkpoint:
        report = initialize_from_spatial_checkpoint(
            base, cfg.init_from_spatial_checkpoint, expected_contract=contract
        )
    else:  # unreachable: config validation requires one of the two
        raise ValueError("temporal config supplied no checkpoint to load")

    if verbose:
        print(report.summary())

    base = base.to(device)
    runner = TemporalSequenceModel(base, cfg, adapter=base.temporal_adapter)
    return runner, report, resume_payload


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
@dataclass
class TemporalTrainingState:
    epoch: int = 0
    global_step: int = 0
    trained_temporal_steps: int = 0
    best_val: float = math.inf
    history: list[dict[str, Any]] = field(default_factory=list)


def _per_frame_loss(
    loss_fn: Callable[..., torch.Tensor],
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor | None,
    frame_batches: Sequence[Mapping[str, Any]],
) -> torch.Tensor:
    """Average the existing spatial loss over the emitted frames."""
    total = predictions.sum() * 0.0
    n = predictions.shape[1]
    for i in range(n):
        y = restore_nan_targets(
            targets[:, i], None if valid_mask is None else valid_mask[:, i]
        )
        frame = dict(frame_batches[i])
        frame["y"] = y
        total = total + loss_fn(predictions[:, i], frame)
    return total / max(n, 1)


def train_temporal_model(
    config: Any,
    cfg: TemporalConfig,
    *,
    device: torch.device | str | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    max_steps_per_epoch: int | None = None,
    max_val_steps: int | None = None,
    num_epochs: int | None = None,
    batch_size: int | None = None,
    num_workers: int = 0,
    verbose: bool = True,
) -> dict[str, Any]:
    """Fine-tune the temporal adapter (and optionally more) on real sequences."""
    seed_everything(cfg.seed)
    device = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    out_dir = Path(
        output_dir
        or getattr(config, "checkpoint_dir", None)
        or Path(getattr(config, "path_experiment", "./experiments")) / "temporal"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.jsonl"

    bs = int(batch_size or getattr(config, "batch_size", 1) or 1)
    epochs = int(num_epochs if num_epochs is not None else getattr(config, "num_epochs", 1))

    loaders = build_sequence_dataloaders(
        config, cfg, batch_size=bs, num_workers=num_workers, verbose=verbose
    )
    time_dim, time_names = resolve_time_feature_dim(loaders["train"])
    if verbose:
        print(f"[temporal] time features ({time_dim}): {time_names}")

    runner, report, resume_payload = build_temporal_model(
        config,
        cfg,
        time_feature_dim=time_dim,
        time_feature_names=time_names,
        device=device,
        verbose=verbose,
    )
    model = runner.base
    output_vars = [str(v) for v in config.data.output_vars]

    loss_fn = build_loss_fn(config, output_vars)
    scales = variable_scales_from_model(model, output_vars)
    if verbose:
        print(f"[temporal] variable scales (from training scalers): {scales}")
    temporal_loss = TemporalLossComputer(
        cfg.losses, output_vars=output_vars, variable_scales=scales, device=device
    )

    state = TemporalTrainingState()
    if resume_payload:
        state.epoch = int(resume_payload.get("epoch", 0) or 0) + 1
        state.global_step = int(resume_payload.get("global_step", 0) or 0)
        state.trained_temporal_steps = int(
            (resume_payload.get("temporal") or {}).get("trained_temporal_steps", 0) or 0
        )

    accum = max(1, int(getattr(config, "gradient_accumulation_steps", 1) or 1))
    max_norm = float(getattr(config, "max_grad_norm", 0.0) or 0.0)

    optimizer: torch.optim.Optimizer | None = None
    trainable_log: dict[str, bool] = {}

    for epoch in range(state.epoch, epochs):
        # Re-apply the freeze policy each epoch and rebuild the optimizer so the
        # parameter groups always match what is actually trainable.
        new_trainable = apply_freeze_policy(model, cfg, epoch)
        if new_trainable != trainable_log or optimizer is None:
            groups = build_param_groups(model, cfg)
            optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
            trainable_log = new_trainable
            if verbose:
                desc = ", ".join(f"{k}={'train' if v else 'frozen'}" for k, v in new_trainable.items())
                lrs = ", ".join(f"{g['name']}@{g['lr']:.2e}" for g in groups)
                print(f"[epoch {epoch}] trainable: {desc} | groups: {lrs}")
            if resume_payload.get("optimizer") and epoch == state.epoch:
                try:
                    optimizer.load_state_dict(resume_payload["optimizer"])
                    if verbose:
                        print("[epoch] restored optimizer state from checkpoint")
                except (ValueError, KeyError) as exc:
                    print(
                        f"[epoch] optimizer state not restored ({exc}); the parameter "
                        "groups changed relative to the checkpoint."
                    )

        train_ds = loaders["train"].dataset
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch)

        runner.train()
        running: dict[str, float] = {}
        n_batches = 0
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loaders["train"]):
            if max_steps_per_epoch is not None and step >= max_steps_per_epoch:
                break
            batch = {
                k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
            out = runner(batch)
            emitted = out.emitted_indices
            frame_batches = [
                {
                    key: batch[key]
                    for key in ("static_x", "static_y")
                    if key in batch
                }
                for _ in emitted
            ]
            per_frame = _per_frame_loss(
                loss_fn, out.predictions, out.target_frames, out.valid_mask, frame_batches
            )
            terms = temporal_loss(
                out.predictions,
                out.target_frames,
                out.valid_mask,
                interval_ratio=out.interval_ratio,
                per_frame_loss=per_frame,
            )
            loss = terms.total / accum
            loss.backward()

            if (step + 1) % accum == 0:
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], max_norm
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                state.global_step += 1
                state.trained_temporal_steps += 1

            for key, value in terms.to_log().items():
                running[key] = running.get(key, 0.0) + value
            n_batches += 1

        train_metrics = {k: v / max(n_batches, 1) for k, v in running.items()}
        train_metrics["seconds"] = time.time() - t0
        train_metrics["n_batches"] = n_batches

        val_metrics = evaluate_split(
            runner,
            loaders.get("validation"),
            loss_fn=loss_fn,
            temporal_loss=temporal_loss,
            device=device,
            max_steps=max_val_steps,
        )

        record = {
            "epoch": epoch,
            "global_step": state.global_step,
            "trained_temporal_steps": state.trained_temporal_steps,
            "trainable": trainable_log,
            "train": train_metrics,
            "validation": val_metrics,
        }
        state.history.append(record)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
        if verbose:
            print(
                f"[epoch {epoch}] train.total={train_metrics.get('total', float('nan')):.5f} "
                f"val.total={val_metrics.get('total', float('nan')):.5f} "
                f"({train_metrics['seconds']:.1f}s, {n_batches} batches)"
            )

        common = dict(
            model=model,
            cfg=cfg,
            output_vars=output_vars,
            time_feature_names=time_names,
            epoch=epoch,
            global_step=state.global_step,
            trained_temporal_steps=state.trained_temporal_steps,
            optimizer=optimizer,
            train_loss=train_metrics.get("total"),
            val_loss=val_metrics.get("total"),
            metrics=record,
            normalization={"variable_scales": scales},
            data_provenance={
                split: loader.dataset.describe() for split, loader in loaders.items()
            },
            extra={"migration_report": report.to_dict()},
        )
        save_temporal_checkpoint(out_dir / "last.ckpt", **common)
        current = val_metrics.get("total")
        if current is not None and current < state.best_val:
            state.best_val = current
            save_temporal_checkpoint(out_dir / "best.ckpt", **common)
            if verbose:
                print(f"[epoch {epoch}] new best validation total={current:.5f}")

    return {
        "output_dir": str(out_dir),
        "history": state.history,
        "best_val": state.best_val,
        "migration_report": report.to_dict(),
        "time_feature_names": time_names,
        "variable_scales": scales,
        "trained_temporal_steps": state.trained_temporal_steps,
    }


@torch.no_grad()
def evaluate_split(
    runner: TemporalSequenceModel,
    loader: DataLoader | None,
    *,
    loss_fn: Callable[..., torch.Tensor],
    temporal_loss: TemporalLossComputer,
    device: torch.device | str,
    max_steps: int | None = None,
) -> dict[str, float]:
    """Loss-only evaluation used for checkpoint selection."""
    if loader is None:
        return {}
    runner.eval()
    running: dict[str, float] = {}
    n = 0
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        batch = {
            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        out = runner(batch)
        frame_batches = [
            {key: batch[key] for key in ("static_x", "static_y") if key in batch}
            for _ in out.emitted_indices
        ]
        per_frame = _per_frame_loss(
            loss_fn, out.predictions, out.target_frames, out.valid_mask, frame_batches
        )
        terms = temporal_loss(
            out.predictions,
            out.target_frames,
            out.valid_mask,
            interval_ratio=out.interval_ratio,
            per_frame_loss=per_frame,
        )
        for key, value in terms.to_log().items():
            running[key] = running.get(key, 0.0) + value
        n += 1
    return {k: v / max(n, 1) for k, v in running.items()}
