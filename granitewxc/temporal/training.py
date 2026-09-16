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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from granitewxc.models.loss import build_loss_fn
from granitewxc.temporal.checkpoint import (
    CheckpointContract,
    TemporalCheckpointError,
    capture_rng_state,
    restore_rng_state,
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
    attach_native_pair_adapter,
    attach_temporal_adapter,
    build_param_groups,
    classify_parameter,
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


class EpochSequenceSampler(Sampler[int]):
    """Reproducible epoch permutation with a resumable, consumed-sample offset.

    Sequence crops already derive from (seed, epoch, sample index). Keeping order
    independent of process RNG permits resuming without reloading earlier data.
    """

    def __init__(self, dataset: Any, seed: int) -> None:
        self.dataset, self.seed = dataset, int(seed)
        self.epoch, self.start_index = 0, 0

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.dataset), generator=generator).tolist()
        return iter(order[self.start_index:])

    def __len__(self) -> int:
        return max(0, len(self.dataset) - self.start_index)


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
            sampler=EpochSequenceSampler(dataset, cfg.seed) if split == "train" else None,
            generator=torch.Generator().manual_seed(cfg.seed + (0 if split == "train" else 1)),
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
    if cfg.backend == "native_pair":
        # Temporal information enters before the transformer, so there is no
        # bottleneck adapter and no hidden state; the time-feature dimension is
        # not part of this pathway's contract.
        adapter = attach_native_pair_adapter(base, cfg)
    else:
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
            base,
            cfg.init_from_spatial_checkpoint,
            expected_contract=contract,
            n_input_timestamps=cfg.native_pair.n_input_timestamps
            if cfg.backend == "native_pair"
            else 1,
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
    microbatches: int = 0
    backward_passes: int = 0
    accumulation_cycles: int = 0
    skipped_nonfinite_updates: int = 0
    skipped_amp_overflow_updates: int = 0
    target_samples: int = 0
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


def _update_optimizer_groups(
    optimizer: torch.optim.Optimizer | None, groups: list[dict[str, Any]]
) -> torch.optim.Optimizer:
    """Apply staged unfreezing without discarding existing Adam moments."""
    if not groups:
        raise ValueError("The temporal freeze policy leaves no trainable parameters")
    updated = torch.optim.AdamW(groups, weight_decay=0.0)
    if optimizer is not None:
        for group in updated.param_groups:
            for parameter in group["params"]:
                if parameter in optimizer.state:
                    updated.state[parameter] = optimizer.state[parameter]
    return updated


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
    checkpoint_interval_updates: int = 100,
    before_training_callback: Callable[..., None] | None = None,
    optimizer_step_callback: Callable[..., None] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Fine-tune on sequences using a successful-optimizer-update budget.

    ``max_steps_per_epoch`` caps actual successful optimizer updates, not
    microbatches. An exhausted loader flushes a partial accumulation cycle using
    its actual sample count; data are never repeated to fill the budget.
    Observer callbacks receive ``(model, optimizer, state)``; the update callback
    runs after ``step`` and before gradients are cleared. ``progress_callback``
    receives JSON-serializable epoch_start, batch, validation, and epoch_end
    events. Epoch numbers are 1-based; completed counts training microbatches,
    including those restored on resume, while updates counts successful updates
    in the current epoch. Total is the available microbatch count capped by the
    remaining update budget, assuming no further skipped updates; it expands
    when updates are skipped and becomes completed once training ends. Loss is
    the running training mean (null if unavailable/nonfinite). Validation marks
    the start of validation; epoch_end follows successful checkpoint writes and
    includes val_loss and status='complete'. Callback exceptions propagate.
    Without a callback, verbose mode displays one tqdm bar per epoch.
    The inherited numerical protocol is FP32 without a scheduler or AMP scaler.
    """
    if max_steps_per_epoch is not None and max_steps_per_epoch < 1:
        raise ValueError("max_steps_per_epoch must allow at least one optimizer update")
    if checkpoint_interval_updates < 0:
        raise ValueError("checkpoint_interval_updates must be nonnegative")
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
    saved_training = resume_payload.get("training_state") or {}
    resume_status = "new_run"
    if resume_payload:
        if resume_payload.get("scheduler") or resume_payload.get("scaler"):
            raise TemporalCheckpointError(
                "This FP32 trainer has no scheduler or AMP scaler; refusing to discard recorded state"
            )
        if saved_training.get("schema") == "temporal_training_v2":
            for key, value in saved_training["counters"].items():
                if hasattr(state, key):
                    setattr(state, key, value)
            state.epoch = int(saved_training["next_epoch"])
            resume_status = "exact_epoch_boundary" if saved_training["epoch_complete"] else "exact_update_boundary"
        else:
            state.epoch = int(resume_payload.get("epoch", 0) or 0) + 1
            state.global_step = int(resume_payload.get("global_step", 0) or 0)
            state.trained_temporal_steps = int(
                (resume_payload.get("temporal") or {}).get("trained_temporal_steps", 0) or 0
            )
            state.best_val = float(resume_payload.get("val_loss") or math.inf)
            resume_status = "approximate_legacy_missing_rng_and_data_order"
            print("[resume] Legacy checkpoint lacks RNG/data-order/counter state; this is an approximate restart")

    accum = max(1, int(getattr(config, "gradient_accumulation_steps", 1) or 1))
    max_norm = float(getattr(config, "max_grad_norm", 0.0) or 0.0)
    data_provenance = {split: loader.dataset.describe() for split, loader in loaders.items()}
    protocol = {
        "gradient_accumulation_steps": accum, "batch_size": bs,
        "num_workers": num_workers, "max_steps_per_epoch": max_steps_per_epoch,
        "max_val_steps": max_val_steps, "precision": "float32",
        "scheduler": None, "scaler": None, "data_provenance": data_provenance,
        "seed": cfg.seed, "max_grad_norm": max_norm,
        "native_pair": cfg.native_pair.to_dict() if cfg.backend == "native_pair" else None,
        "freeze": cfg.freeze.to_dict(), "losses": cfg.losses.to_dict(),
    }
    if saved_training:
        recorded_protocol = saved_training.get("protocol", {})
        if any(protocol.get(key) != value for key, value in recorded_protocol.items()):
            raise TemporalCheckpointError("Exact resume requires the same training/data protocol")
        if set(recorded_protocol) != set(protocol):
            resume_status += "_partial_protocol_metadata"
            print("[resume] Full RNG/optimizer/data state restored, but earlier protocol metadata is incomplete")
    if saved_training and not saved_training["epoch_complete"] and num_workers != 0:
        raise TemporalCheckpointError("Exact mid-epoch resume currently requires num_workers=0")

    # Pretraining-aligned auxiliary objectives. Off unless the config asks for
    # them, so the architectural change can be measured on its own first.
    pretext_enabled = cfg.backend == "native_pair" and cfg.native_pair.pretext.any_enabled
    # Masking draws from a dedicated, explicitly seeded generator so the mask
    # sequence is reproducible and independent of the crop/data-order RNG.
    pretext_generator: torch.Generator | None = None
    if pretext_enabled:
        pretext_generator = torch.Generator(device=torch.device(device))
        pretext_generator.manual_seed(int(cfg.seed) + 8191)
        if verbose:
            p = cfg.native_pair.pretext
            print(
                f"[pretext] masked_reconstruction={p.masked_reconstruction_enabled}"
                f"(w={p.masked_reconstruction_weight}, ratio={p.mask_ratio}) "
                f"transition={p.transition_enabled}"
                f"(w={p.transition_weight}, lead={p.transition_lead_steps} step(s))"
            )

    optimizer: torch.optim.Optimizer | None = None
    trainable_log: dict[str, bool] = {}
    previous_lr_scale: dict[str, float] = {}
    initial_epoch = state.epoch
    resumed_mid_epoch = bool(saved_training and not saved_training["epoch_complete"])

    def save_checkpoint(path: Path, epoch: int, epoch_complete: bool,
                        epoch_progress: dict[str, Any], record: dict[str, Any]) -> None:
        sampler = loaders["train"].sampler
        reproducible_order = isinstance(sampler, EpochSequenceSampler)
        runtime = {
            "schema": "temporal_training_v2", "counters": asdict(state),
            "epoch_complete": epoch_complete, "next_epoch": epoch + int(epoch_complete),
            "protocol": protocol, "epoch_progress": epoch_progress,
            "data_order": "epoch_seed_permutation" if reproducible_order else "external_loader",
            "loader_generators": {
                split: loader.generator.get_state() for split, loader in loaders.items()
                if loader.generator is not None
            },
            "pretext_generator": pretext_generator.get_state() if pretext_generator is not None else None,
        }
        save_temporal_checkpoint(
            path, model=model, cfg=cfg, output_vars=output_vars,
            time_feature_names=time_names, epoch=epoch, global_step=state.global_step,
            trained_temporal_steps=state.trained_temporal_steps, optimizer=optimizer,
            train_loss=record.get("train", {}).get("total"),
            val_loss=record.get("validation", {}).get("total"), metrics=record,
            normalization={"variable_scales": scales}, data_provenance=data_provenance,
            training_state=runtime, rng_state=capture_rng_state(),
            extra={"migration_report": report.to_dict(), "resume_status": resume_status},
        )

    owned_progress = None
    if progress_callback is None and verbose:
        from granitewxc.temporal.progress import EpochProgress

        owned_progress = EpochProgress()
        progress_callback = owned_progress
    progress_status = "interrupted"
    try:
        for epoch in range(initial_epoch, epochs):
            state.epoch = epoch
            new_trainable = apply_freeze_policy(model, cfg, epoch)
            lr_scale = {}
            for stage in cfg.freeze.unfreeze_schedule:
                if epoch >= stage.epoch:
                    for group in stage.modules:
                        names = ("temporal", "decoder", "encoder", "backbone", "other") if group == "all" else (group,)
                        lr_scale.update({name: stage.lr_scale for name in names})
            if new_trainable != trainable_log or lr_scale != previous_lr_scale or optimizer is None:
                groups = build_param_groups(model, cfg, lr_scale=lr_scale)
                previous_lr_scale = lr_scale
                optimizer = _update_optimizer_groups(optimizer, groups)
                trainable_log = new_trainable
                if verbose:
                    desc = ", ".join(f"{k}={'train' if v else 'frozen'}" for k, v in new_trainable.items())
                    lrs = ", ".join(f"{g['name']}@{g['lr']:.2e}" for g in groups)
                    print(f"[epoch {epoch}] trainable: {desc} | groups: {lrs}")
                if resume_payload and epoch == initial_epoch:
                    stored_optimizer = resume_payload.get("optimizer")
                    if not stored_optimizer:
                        raise TemporalCheckpointError("Training resume requires optimizer state")
                    old_groups = stored_optimizer.get("param_groups", [])
                    new_groups = optimizer.param_groups
                    # Restore by recorded parameter names, including moments of groups
                    # newly unfrozen at this epoch. Positional remapping is unsafe.
                    if saved_training:
                        named = dict(model.named_parameters())
                        seen = set()
                        for old_group in old_groups:
                            names = old_group.get("param_names")
                            if names is None or len(names) != len(old_group["params"]):
                                raise TemporalCheckpointError("Optimizer checkpoint lacks parameter identity")
                            for name, index in zip(names, old_group["params"]):
                                if name not in named or classify_parameter(name) == "normalization" or name in seen:
                                    raise TemporalCheckpointError(f"Invalid optimizer parameter identity: {name}")
                                seen.add(name)
                                if index in stored_optimizer["state"]:
                                    optimizer.state[named[name]] = {
                                        key: (value.to(named[name].device) if torch.is_tensor(value) else value)
                                        for key, value in stored_optimizer["state"][index].items()
                                    }
                        for group in new_groups:
                            prior = next((old for old in old_groups if old.get("name") == group["name"]), None)
                            if prior is not None:
                                for key, value in prior.items():
                                    if key not in {"params", "param_names", "name", "lr", "initial_lr"}:
                                        group[key] = value
                    else:
                        try:
                            parameter_names = [group["param_names"] for group in optimizer.param_groups]
                            optimizer.load_state_dict(stored_optimizer)
                            for group, names in zip(optimizer.param_groups, parameter_names):
                                group["param_names"] = names
                        except (ValueError, KeyError) as exc:
                            raise TemporalCheckpointError("Legacy optimizer groups are incompatible; cannot resume") from exc

            train_ds = loaders["train"].dataset
            if hasattr(train_ds, "set_epoch"):
                train_ds.set_epoch(epoch)
            progress = saved_training.get("epoch_progress", {}) if resumed_mid_epoch and epoch == initial_epoch else {}
            running: dict[str, float] = dict(progress.get("running", {}))
            n_batches = int(progress.get("n_batches", 0))
            backbone_evals = int(progress.get("backbone_evals", 0))
            epoch_updates = int(progress.get("optimizer_updates", 0))
            epoch_start_counters = progress.get("start_counters", asdict(state))
            sampler = loaders["train"].sampler
            if isinstance(sampler, EpochSequenceSampler):
                sampler.epoch, sampler.start_index = epoch, n_batches * bs
            elif n_batches:
                raise TemporalCheckpointError("Mid-epoch resume requires EpochSequenceSampler")
            if before_training_callback is not None and epoch == initial_epoch:
                before_training_callback(model, optimizer, state)
            if saved_training and epoch == initial_epoch:
                for split, generator_state in saved_training.get("loader_generators", {}).items():
                    if loaders[split].generator is None:
                        raise TemporalCheckpointError("Cannot restore missing data-loader generator")
                    loaders[split].generator.set_state(generator_state.cpu())
                if pretext_generator is not None and saved_training.get("pretext_generator") is not None:
                    pretext_generator.set_state(saved_training["pretext_generator"].cpu())
                if "rng_state" not in resume_payload:
                    raise TemporalCheckpointError("Exact resume checkpoint is missing process RNG state")
                restore_rng_state(resume_payload["rng_state"])

            runner.train()
            t0 = time.time()
            optimizer.zero_grad(set_to_none=True)
            pending = 0
            pending_samples = 0
            pending_nonfinite = False
            available_batches = n_batches + len(loaders["train"])

            def emit_progress(event: str, *, training_complete: bool = False,
                              **fields: Any) -> None:
                if progress_callback is None:
                    return
                total_batches = available_batches
                if max_steps_per_epoch is not None:
                    remaining = max(0, (max_steps_per_epoch - epoch_updates) * accum - pending)
                    total_batches = min(total_batches, n_batches + remaining)
                if training_complete:
                    total_batches = n_batches
                loss = running.get("total")
                loss = loss / n_batches if loss is not None and n_batches else None
                progress_callback({
                    "event": event, "epoch": epoch + 1, "epochs": epochs,
                    "total": total_batches, "completed": n_batches, "updates": epoch_updates,
                    "loss": loss if loss is not None and math.isfinite(loss) else None,
                    **fields,
                })

            emit_progress("epoch_start")
            trainable_parameters = [p for group in optimizer.param_groups for p in group["params"]]
            temporal_parameters = [p for group in optimizer.param_groups if group["name"] == "temporal"
                                   and group["lr"] != 0 for p in group["params"]]

            def update() -> None:
                nonlocal pending, pending_samples, pending_nonfinite, epoch_updates
                if not pending:
                    return
                state.accumulation_cycles += 1
                # Backpropagate a sample sum, then normalize by the true cycle size.
                # This also handles the final short cycle and unequal batch sizes.
                for parameter in trainable_parameters:
                    if parameter.grad is not None:
                        parameter.grad.div_(pending_samples)
                finite = not pending_nonfinite and all(
                    bool(torch.isfinite(parameter.grad).all()) for parameter in trainable_parameters
                    if parameter.grad is not None
                )
                if finite:
                    if max_norm > 0:
                        torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm)
                    has_temporal_gradient = any(p.grad is not None for p in temporal_parameters)
                    optimizer.step()
                    state.global_step += 1
                    state.trained_temporal_steps += int(has_temporal_gradient)
                    epoch_updates += 1
                    if optimizer_step_callback is not None:
                        optimizer_step_callback(model, optimizer, state)
                else:
                    state.skipped_nonfinite_updates += 1
                optimizer.zero_grad(set_to_none=True)
                pending, pending_samples, pending_nonfinite = 0, 0, False

            iterator = iter(loaders["train"])
            if resumed_mid_epoch and epoch == initial_epoch and loaders["train"].generator is not None:
                # Iterator creation consumes a worker-base seed. No workers exist in
                # supported mid-epoch resumes; undo that extra draw for future epochs.
                loaders["train"].generator.set_state(saved_training["loader_generators"]["train"].cpu())
            for batch in iterator:
                if max_steps_per_epoch is not None and epoch_updates >= max_steps_per_epoch:
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
                total = terms.total
                pretext_log: dict[str, float] = {}
                if pretext_enabled:
                    from granitewxc.temporal.native_pair import compute_pretext_losses

                    # Applied to the last emitted frame only; see
                    # ``compute_pretext_losses`` for why one frame per window.
                    pretext_terms, pretext_forwards = compute_pretext_losses(
                        runner.base,
                        runner.adapter,
                        batch,
                        emitted[-1],
                        cfg,
                        generator=pretext_generator,
                    )
                    pretext_total = pretext_terms.total
                    if pretext_total is not None:
                        total = total + pretext_total
                    pretext_log = {
                        k: float(v.detach().item()) for k, v in pretext_terms.items()
                    }
                    backbone_evals += pretext_forwards
                samples = int(out.predictions.shape[0])
                pending += 1
                pending_samples += samples
                state.microbatches += 1
                state.target_samples += samples * len(emitted)
                if bool(torch.isfinite(total)):
                    (total * samples).backward()
                    state.backward_passes += 1
                else:
                    pending_nonfinite = True
                for key, value in terms.to_log().items():
                    running[key] = running.get(key, 0.0) + value
                for key, value in pretext_log.items():
                    running[key] = running.get(key, 0.0) + value
                backbone_evals += int(out.backbone_evaluations)
                n_batches += 1
                if pending == accum:
                    previous_step = state.global_step
                    update()
                    if checkpoint_interval_updates and state.global_step > previous_step and state.global_step % checkpoint_interval_updates == 0:
                        save_checkpoint(out_dir / "last.ckpt", epoch, False, {
                            "running": running, "n_batches": n_batches, "backbone_evals": backbone_evals,
                            "optimizer_updates": epoch_updates, "start_counters": epoch_start_counters,
                        }, {})
                emit_progress("batch")
                if max_steps_per_epoch is not None and epoch_updates >= max_steps_per_epoch:
                    break
            update()
            if state.global_step == epoch_start_counters["global_step"]:
                raise RuntimeError("Training epoch produced no successful optimizer updates; no trained checkpoint was saved")
            train_metrics = {k: v / max(n_batches, 1) for k, v in running.items()}
            train_metrics.update({
                "backbone_evaluations": backbone_evals,
                "backbone_evaluations_per_microbatch": backbone_evals / max(n_batches, 1),
                "backbone_evaluations_per_optimizer_update": backbone_evals / max(epoch_updates, 1),
                "seconds": time.time() - t0, "n_batches": n_batches,
                "optimizer_updates": epoch_updates, "requested_optimizer_updates": max_steps_per_epoch,
                "budget_exhausted": max_steps_per_epoch is not None and epoch_updates >= max_steps_per_epoch,
                "data_exhausted": max_steps_per_epoch is None or epoch_updates < max_steps_per_epoch,
                "step_unit": "successful_optimizer_update", "precision": "float32",
            })
            for key in ("microbatches", "backward_passes", "accumulation_cycles", "target_samples",
                        "skipped_nonfinite_updates", "skipped_amp_overflow_updates"):
                train_metrics[key] = getattr(state, key) - epoch_start_counters[key]
            emit_progress("validation", training_complete=True)
            val_metrics = evaluate_split(
                runner, loaders.get("validation"), loss_fn=loss_fn,
                temporal_loss=temporal_loss, device=device, max_steps=max_val_steps,
            )
            record = {
                "epoch": epoch, "global_step": state.global_step,
                "trained_temporal_steps": state.trained_temporal_steps,
                "trainable": trainable_log,
                "optimizer_groups": [{"name": g["name"], "lr": g["lr"],
                                      "parameter_count": sum(p.numel() for p in g["params"]),
                                      "param_names": g["param_names"]} for g in optimizer.param_groups],
                "resume_status": resume_status, "train": train_metrics, "validation": val_metrics,
            }
            state.history.append(record)
            current = val_metrics.get("total")
            improved = current is not None and math.isfinite(current) and current < state.best_val
            if improved:
                state.best_val = current
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
            if verbose:
                print(f"[epoch {epoch}] train.total={train_metrics.get('total', float('nan')):.5f} "
                      f"val.total={val_metrics.get('total', float('nan')):.5f} "
                      f"({train_metrics['seconds']:.1f}s, {n_batches} microbatches, {epoch_updates} updates)")
            save_checkpoint(out_dir / "last.ckpt", epoch, True, {}, record)
            if improved:
                save_checkpoint(out_dir / "best.ckpt", epoch, True, {}, record)
            emit_progress("epoch_end", training_complete=True,
                          val_loss=current if current is not None and math.isfinite(current) else None,
                          status="complete")
        progress_status = "complete"
    finally:
        if owned_progress is not None:
            owned_progress.close(status=progress_status)

    return {
        "output_dir": str(out_dir), "history": state.history, "best_val": state.best_val,
        "migration_report": report.to_dict(), "time_feature_names": time_names,
        "variable_scales": scales, "trained_temporal_steps": state.trained_temporal_steps,
        "global_step": state.global_step, "counters": asdict(state), "resume_status": resume_status,
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
