#!/usr/bin/env python
"""Train a CORDEX Phase-2 residual refiner without a notebook.

This entry point deliberately reuses the CORDEX dataset/model construction and
the core :class:`granitewxc.refinement.training.RefinementTrainer`.  It does not
implement a second training loop or a second checkpoint format.

Examples
--------
Train from scratch using every setting in a refinement YAML::

    python examples/CORDEX_ML/cordex_refinement_training.py \
        --config examples/CORDEX_ML/SA_T2_ACCESS-CM2_static_diffusion_transformer.yaml \
        --device cuda:0

Run a deliberately tiny, deterministic overfit check::

    python examples/CORDEX_ML/cordex_refinement_training.py \
        --config examples/CORDEX_ML/SA_T2_ACCESS-CM2_static_diffusion_transformer.yaml \
        --device cuda:0 --tiny-overfit --epochs 20

Resume is opt-in.  A bare ``--resume`` selects ``last.ckpt`` in the resolved
checkpoint directory; an explicit value selects that exact checkpoint.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parents[1]
_LAST_CHECKPOINT = "__checkpoint_dir_last__"


class _LimitedLoader:
    """Re-iterable view over the first ``limit`` batches of a loader.

    Passing this view to ``RefinementTrainer.fit`` also bounds its training-only
    residual-statistics pass.  This matters for quick smoke/overfit commands:
    ``limit_steps_train`` alone intentionally does not truncate statistic
    fitting in the core trainer.
    """

    def __init__(self, loader: Iterable[Mapping[str, Any]], limit: int) -> None:
        if limit < 1:
            raise ValueError("A limited loader requires at least one batch.")
        self.loader = loader
        self.limit = int(limit)

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        for index, batch in enumerate(self.loader):
            if index >= self.limit:
                break
            yield batch

    def __len__(self) -> int:
        if hasattr(self.loader, "__len__"):
            return min(self.limit, len(self.loader))  # type: ignore[arg-type]
        return self.limit


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not parsed > 0.0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def _case_index(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("case index must be >= 0")
    return parsed


def _validation_fraction(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 0.5:
        raise argparse.ArgumentTypeError("validation fraction must be in (0, 0.5)")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train one CORDEX stochastic residual-refinement head on a strictly "
            "loaded, frozen Phase-1 UNet."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to a Phase-2 CORDEX YAML file.")
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device (auto, cpu, cuda, cuda:0, ...).",
    )
    parser.add_argument(
        "--epochs",
        type=_nonnegative_int,
        default=None,
        help="Override the YAML target epoch count (resume trains only the remainder).",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=None,
        help="Override the YAML batch size for bounded validation or training.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=_positive_int,
        default=None,
        help="Override the YAML optimizer accumulation interval.",
    )
    parser.add_argument(
        "--learning-rate",
        type=_positive_float,
        default=None,
        help="Override the YAML learning rate (useful for bounded tiny-overfit diagnostics).",
    )
    parser.add_argument(
        "--limit-train",
        "--max-train-batches",
        dest="limit_train",
        type=_nonnegative_int,
        default=None,
        help=(
            "Bound training to this many batches per epoch. An explicit nonzero "
            "value also bounds the residual-statistics prepass."
        ),
    )
    parser.add_argument(
        "--limit-valid",
        "--max-valid-batches",
        "--max-val-batches",
        dest="limit_valid",
        type=_nonnegative_int,
        default=None,
        help="Bound validation to this many batches per epoch.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help=(
            "Override checkpoint directory. Relative paths are resolved under "
            "examples/CORDEX_ML."
        ),
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const=_LAST_CHECKPOINT,
        default=None,
        metavar="CHECKPOINT",
        help=(
            "Resume explicitly. With no path, use CHECKPOINT_DIR/last.ckpt; "
            "relative paths are resolved under examples/CORDEX_ML."
        ),
    )
    parser.add_argument(
        "--case-index",
        "--dataset-index",
        dest="case_index",
        type=_case_index,
        default=None,
        help="Select one paired predictor/target file from each configured split.",
    )
    parser.add_argument(
        "--tiny-overfit",
        action="store_true",
        help=(
            "Use one deterministic full-domain training batch for both train and "
            "validation; intended only as a transformer coherence smoke test."
        ),
    )
    parser.add_argument(
        "--validation-fraction",
        type=_validation_fraction,
        default=0.10,
        help=(
            "Chronological holdout fraction when YAML training and validation "
            "paths are identical. This prevents validation leakage."
        ),
    )
    return parser


def _resolve_config_path(value: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.resolve()


def _resolve_repo_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _resolve_project_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def _move_batch(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    import torch

    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint_payload(path: Path) -> Mapping[str, Any]:
    import torch

    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - PyTorch < 2.0
        payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"Checkpoint payload must be a mapping: {path}")
    return payload


def _checkpoint_belongs_to_run(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> bool:
    """Reject a stale best checkpoint even when it shares a directory."""
    identity_keys = (
        "checkpoint_schema_version",
        "checkpoint_kind",
        "refinement_type",
        "refinement_contract_fingerprint",
        "residual_normalizer_state_fingerprint",
        "phase1_fingerprint",
        "case_name",
        "resolved_config",
    )
    if any(candidate.get(key) != reference.get(key) for key in identity_keys):
        return False
    try:
        return (
            int(candidate.get("epoch", -1)) <= int(reference.get("epoch", -1))
            and int(candidate.get("global_step", -1))
            <= int(reference.get("global_step", -1))
        )
    except (TypeError, ValueError):
        return False


def _first_batch_value(value: Any, batch_size: int) -> Any:
    import torch

    if torch.is_tensor(value):
        selected = value[0] if value.ndim > 0 and value.shape[0] == batch_size else value
        selected = selected.detach().to("cpu")
        return selected.item() if selected.numel() == 1 else selected.tolist()
    if isinstance(value, (list, tuple)) and len(value) == batch_size:
        return value[0]
    return value


def _diagnostic_sample_identity(
    batch: Mapping[str, Any],
    *,
    batch_size: int,
) -> dict[str, Any]:
    identity = {
        key[len("__sample_") :]: _first_batch_value(value, batch_size)
        for key, value in batch.items()
        if key.startswith("__sample_")
    }
    identity["identity_available"] = bool(identity)
    identity["loader_batch_index"] = 0
    return identity


def _write_diagnostic_report(
    report: Mapping[str, Any],
    *,
    convenience_path: Path,
    checkpoint_sha256: str,
) -> Path:
    """Write one immutable checkpoint-bound report plus a convenience copy."""
    immutable_path = convenience_path.with_name(
        f"{convenience_path.stem}-{checkpoint_sha256}{convenience_path.suffix}"
    )
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    convenience_path.parent.mkdir(parents=True, exist_ok=True)
    if immutable_path.exists():
        print(f"[diagnostics] immutable report already exists; retained {immutable_path}")
    else:
        # Exclusive creation makes a repeated invocation unable to silently
        # replace evidence already bound to these exact checkpoint bytes.
        with immutable_path.open("x", encoding="utf-8") as handle:
            handle.write(text)
        print(f"[diagnostics] wrote immutable report {immutable_path}")
    if convenience_path.exists():
        print(f"[diagnostics] refreshing convenience report {convenience_path}")
    convenience_path.write_text(text, encoding="utf-8")
    return immutable_path


def _resolve_device(requested: str) -> Any:
    import torch

    value = str(requested).strip().lower()
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {requested!r} was requested, but CUDA is unavailable.")
    if device.type == "cuda" and device.index is not None:
        if device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {device.index} is unavailable; "
                f"torch reports {torch.cuda.device_count()} device(s)."
            )
    return device


def _split_paths(config: Any, split: str) -> tuple[list[Any], list[Any]]:
    predictors = list(getattr(config.data, f"{split}_predictor_paths", []) or [])
    targets = list(getattr(config.data, f"{split}_target_paths", []) or [])
    if bool(predictors) != bool(targets):
        raise ValueError(
            f"data.{split}_predictor_paths and data.{split}_target_paths must "
            "either both be supplied or both be empty."
        )
    if len(predictors) != len(targets):
        raise ValueError(
            f"{split} predictor/target path counts differ: "
            f"{len(predictors)} != {len(targets)}."
        )
    return predictors, targets


def _select_case(config: Any, index: int | None) -> None:
    if index is None:
        return
    selected_any = False
    for split in ("training", "validation"):
        predictors, targets = _split_paths(config, split)
        if not predictors:
            continue
        if index >= len(predictors):
            raise IndexError(
                f"--case-index {index} is outside the {split} split "
                f"with {len(predictors)} paired file(s)."
            )
        setattr(config.data, f"{split}_predictor_paths", [predictors[index]])
        setattr(config.data, f"{split}_target_paths", [targets[index]])
        selected_any = True
    if not selected_any:
        raise ValueError("--case-index was supplied, but no dataset path lists are configured.")


def _crop_and_offset(config: Any, *, validation: bool) -> tuple[tuple[int, int], tuple[int, int]]:
    target = (int(config.data.target_size_lat), int(config.data.target_size_lon))
    prefix = "val" if validation else "train"
    crop = (
        int(getattr(config.data, f"{prefix}_crop_size_lat", target[0])),
        int(getattr(config.data, f"{prefix}_crop_size_lon", target[1])),
    )
    if validation:
        return crop, (0, 0)
    raw_offset = getattr(config.data, "train_random_crop_offset", (0, 0))
    if isinstance(raw_offset, (int, float)):
        offset = (max(0, int(raw_offset)), max(0, int(raw_offset)))
    elif isinstance(raw_offset, (list, tuple)) and len(raw_offset) == 2:
        offset = (max(0, int(raw_offset[0])), max(0, int(raw_offset[1])))
    else:
        raise ValueError("data.train_random_crop_offset must be a scalar or a pair.")
    return crop, offset


def _chronological_split_indices(
    length: int,
    validation_fraction: float,
) -> tuple[range, range]:
    """Return disjoint, ordered train/validation indices."""
    if length < 2:
        raise ValueError("A held-out split requires at least two samples.")
    validation_count = max(1, int(round(length * validation_fraction)))
    validation_count = min(validation_count, length - 1)
    split = length - validation_count
    return range(0, split), range(split, length)


def _subset_loader(loader: Any, indices: range, *, shuffle: bool) -> Any:
    from torch.utils.data import DataLoader, Subset

    dataset = Subset(loader.dataset, indices)
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": loader.batch_size,
        "shuffle": shuffle,
        "num_workers": loader.num_workers,
        "pin_memory": loader.pin_memory,
        "collate_fn": loader.collate_fn,
        "drop_last": loader.drop_last,
    }
    if loader.num_workers > 0:
        kwargs["persistent_workers"] = bool(loader.persistent_workers)
        if loader.prefetch_factor is not None:
            kwargs["prefetch_factor"] = loader.prefetch_factor
    return DataLoader(**kwargs)


def _build_loaders(
    config: Any,
    *,
    use_gpu: bool,
    tiny_overfit: bool,
    validation_fraction: float,
) -> tuple[Any, Any, str]:
    from cordex_training import build_dataloader, get_dataloaders

    train_predictors, train_targets = _split_paths(config, "training")
    if not train_predictors:
        raise ValueError("The YAML does not define a training predictor/target pair.")
    val_predictors, val_targets = _split_paths(config, "validation")

    if tiny_overfit:
        full_size = (int(config.data.target_size_lat), int(config.data.target_size_lon))
        loader = build_dataloader(
            config,
            train_predictors,
            train_targets,
            shuffle=False,
            use_gpu=use_gpu,
            distributed=False,
            rank=0,
            world_size=1,
            crop_size=full_size,
            random_crop=False,
            random_crop_offset=(0, 0),
        )
        tiny = _LimitedLoader(loader, 1)
        return tiny, tiny, "training batch (tiny-overfit mode)"

    if val_predictors:
        train_loader, val_loader = get_dataloaders(config, use_gpu=use_gpu)
        same_paths = (
            [str(item) for item in train_predictors]
            == [str(item) for item in val_predictors]
            and [str(item) for item in train_targets]
            == [str(item) for item in val_targets]
        )
        if same_paths:
            if len(train_loader.dataset) != len(val_loader.dataset):
                raise RuntimeError(
                    "Identical validation paths produced different dataset lengths."
                )
            train_indices, val_indices = _chronological_split_indices(
                len(train_loader.dataset), validation_fraction
            )
            train_loader = _subset_loader(
                train_loader, train_indices, shuffle=True
            )
            val_loader = _subset_loader(val_loader, val_indices, shuffle=False)
            print(
                "[data] identical validation paths detected; using disjoint "
                f"chronological holdout ({len(train_indices)} train, "
                f"{len(val_indices)} validation samples)."
            )
            source = "automatic chronological holdout from identical YAML paths"
        else:
            source = "YAML validation predictor/target paths"
        return train_loader, val_loader, source

    train_crop, offset = _crop_and_offset(config, validation=False)
    train_loader = build_dataloader(
        config,
        train_predictors,
        train_targets,
        shuffle=True,
        use_gpu=use_gpu,
        distributed=False,
        rank=0,
        world_size=1,
        crop_size=train_crop,
        random_crop=True,
        random_crop_offset=offset,
    )
    return train_loader, None, "none (no validation paths configured)"


def _build_scaler(torch_module: Any, *, enabled: bool) -> Any:
    amp = getattr(torch_module, "amp", None)
    scaler_type = getattr(amp, "GradScaler", None) if amp is not None else None
    if scaler_type is not None:
        try:
            return scaler_type("cuda", enabled=enabled)
        except TypeError:
            return scaler_type(enabled=enabled)
    from torch.cuda.amp import GradScaler

    return GradScaler(enabled=enabled)


def _gate_values(model: Any) -> list[float] | None:
    gate = getattr(getattr(model, "refiner", None), "correction_gate", None)
    if gate is None:
        return None
    return [float(value) for value in gate.detach().float().cpu().reshape(-1)]


def _print_gate(model: Any, label: str) -> None:
    values = _gate_values(model)
    if values is None:
        print(f"[gate] {label}: not used by this refinement architecture")
    else:
        print(f"[gate] {label}: {values}")


def _paired_skill_metrics(prediction: Any, truth: Any) -> dict[str, float | int | None]:
    """Fixed-batch physical skill with explicit finite-pair handling."""
    import torch

    predicted = prediction.detach().float().reshape(-1)
    observed = truth.detach().float().reshape(-1)
    valid = torch.isfinite(predicted) & torch.isfinite(observed)
    predicted = predicted[valid]
    observed = observed[valid]
    count = int(predicted.numel())
    if count == 0:
        return {
            "finite_count": 0,
            "correlation": None,
            "mae": None,
            "rmse": None,
            "bias": None,
        }
    error = predicted - observed
    correlation = None
    if count >= 2:
        predicted_centered = predicted - predicted.mean()
        observed_centered = observed - observed.mean()
        denominator = (
            predicted_centered.square().sum().sqrt()
            * observed_centered.square().sum().sqrt()
        )
        if float(denominator) > 0.0:
            correlation = float(
                (predicted_centered * observed_centered).sum() / denominator
            )
    return {
        "finite_count": count,
        "correlation": correlation,
        "mae": float(error.abs().mean()),
        "rmse": float(error.square().mean().sqrt()),
        "bias": float(error.mean()),
    }


def _skill_delta(
    refined: Mapping[str, float | int | None],
    phase1: Mapping[str, float | int | None],
) -> dict[str, float | None]:
    delta: dict[str, float | None] = {}
    for key in ("correlation", "mae", "rmse", "bias"):
        refined_value = refined.get(key)
        phase1_value = phase1.get(key)
        delta[key] = (
            None
            if refined_value is None or phase1_value is None
            else float(refined_value) - float(phase1_value)
        )
    refined_bias = refined.get("bias")
    phase1_bias = phase1.get("bias")
    delta["absolute_bias"] = (
        None
        if refined_bias is None or phase1_bias is None
        else abs(float(refined_bias)) - abs(float(phase1_bias))
    )
    return delta


def _diagnostic_target_units(loader: Any, names: Sequence[str]) -> dict[str, str]:
    """Recover channel-aligned physical units through loader/dataset wrappers."""

    current = loader
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        units = getattr(current, "target_units", None)
        if units is not None and len(units) == len(names):
            return {str(name): str(unit) for name, unit in zip(names, units)}
        next_value = None
        for attribute in ("dataset", "base", "loader"):
            candidate = getattr(current, attribute, None)
            if candidate is not None and candidate is not current:
                next_value = candidate
                break
        current = next_value
    return {}


def _compact_tensor_summary(value: Any) -> dict[str, Any] | None:
    """Serialize sampled time coordinates and scalar objective terms safely."""

    import torch

    if not torch.is_tensor(value):
        return None
    array = value.detach().float().reshape(-1).cpu()
    finite = array[torch.isfinite(array)]
    return {
        "shape": [int(size) for size in value.shape],
        "finite_count": int(finite.numel()),
        "nan_count": int(torch.isnan(array).sum()),
        "inf_count": int(torch.isinf(array).sum()),
        "min": None if finite.numel() == 0 else float(finite.min()),
        "max": None if finite.numel() == 0 else float(finite.max()),
        "mean": None if finite.numel() == 0 else float(finite.mean()),
    }


def _print_diagnostics(
    model: Any,
    loader: Any,
    device: Any,
    *,
    seed: int,
    output_path: Path | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_payload: Mapping[str, Any] | None = None,
    checkpoint_selection: str | None = None,
    split_context: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    import torch

    from granitewxc.refinement.diagnostics import summarize_refinement_tensors

    if loader is None:
        print("[diagnostics] skipped: no validation or training loader is available")
        return None
    raw_batch = next(iter(loader))
    batch_size = int(raw_batch["y"].shape[0])
    sample_identity = _diagnostic_sample_identity(
        raw_batch,
        batch_size=batch_size,
    )
    # Sample provenance is for the report, not a model input. Strip only these
    # private keys; scaler/crop metadata remains part of the model contract.
    raw_batch = {
        key: value
        for key, value in raw_batch.items()
        if not key.startswith("__sample_")
    }
    batch = _move_batch(raw_batch, device)
    # Diagnostics need one representative sample, not an expensive reverse
    # process over the entire training batch. Preserve private scalar metadata
    # while slicing every batched tensor consistently.
    batch = {
        key: value[:1]
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch_size
        else value
        for key, value in batch.items()
    }
    model.eval()
    diagnostic_ensemble_size = max(
        1,
        int(getattr(getattr(model, "refinement_config", None), "ensemble_size", 1)),
    )
    with torch.no_grad():
        training_view = model.training_step(batch, generator=torch.Generator().manual_seed(seed))
        prediction = model.predict(
            batch,
            ensemble_size=diagnostic_ensemble_size,
            seed=seed,
            return_members=True,
        )
    if model.training or model.phase1.training or model.refiner.training:
        raise RuntimeError("Diagnostic inference unexpectedly left a model component in train mode.")
    names = list(
        getattr(model.phase1, "output_var_names", None)
        or getattr(model.phase1, "predictands", [])
        or []
    )
    if not names:
        names = [f"channel_{index}" for index in range(batch["y"].shape[1])]
    raw_normalized = prediction.residual
    raw_physical = model.residual_normalizer.denormalize(
        raw_normalized.to(prediction.deterministic.dtype)
    )
    tensors: dict[str, Any] = {
        "ground_truth": batch["y"],
        "phase1_prediction": prediction.deterministic,
        "true_residual_physical": training_view.residual_target_physical,
        "true_residual_normalized": training_view.residual_target,
        "predicted_residual_normalized_raw": raw_normalized,
        "predicted_residual_physical_raw": raw_physical,
        "effective_correction_physical": prediction.residual_physical,
        "final_refined_prediction": prediction.refined,
    }
    optional_prediction_tensors = {
        "physical_ensemble_members": prediction.members,
        "unbounded_physical_ensemble_members": prediction.members_unbounded,
        "normalized_residual_ensemble_members": prediction.member_residuals,
        "effective_physical_correction_members": prediction.member_residuals_physical,
        "physical_ensemble_spread": prediction.ensemble_spread,
        "unbounded_ensemble_mean": prediction.unbounded_ensemble_mean,
        "historical_memberwise_clipped_mean": prediction.memberwise_clipped_mean,
        "memberwise_clipping_mean_shift": prediction.memberwise_clipping_mean_shift,
    }
    tensors.update(
        {name: value for name, value in optional_prediction_tensors.items() if value is not None}
    )
    loss_tensors: dict[str, Any] = {}
    process_coordinates: dict[str, Any] = {}
    for key, value in training_view.losses.items():
        if torch.is_tensor(value) and value.ndim >= 4:
            if value.shape[1] == len(names):
                loss_tensors[f"objective_{key}"] = value
        elif torch.is_tensor(value):
            summary = _compact_tensor_summary(value)
            if summary is not None:
                process_coordinates[key] = summary
    tensors.update(loss_tensors)

    physical_units = _diagnostic_target_units(loader, names)
    normalized_units = {name: "normalized residual" for name in names}
    physical_tensor_names = {
        "ground_truth",
        "phase1_prediction",
        "true_residual_physical",
        "predicted_residual_physical_raw",
        "effective_correction_physical",
        "final_refined_prediction",
        "physical_ensemble_members",
        "unbounded_physical_ensemble_members",
        "effective_physical_correction_members",
        "physical_ensemble_spread",
        "unbounded_ensemble_mean",
        "historical_memberwise_clipped_mean",
        "memberwise_clipping_mean_shift",
    }
    channel_units = {
        name: (physical_units if name in physical_tensor_names else normalized_units)
        for name in tensors
    }
    channel_dims = {
        name: (2 if getattr(value, "ndim", 0) == 5 else 1)
        for name, value in tensors.items()
    }
    dimension_names = {
        name: (
            ("batch", "ensemble", "channel", "lat", "lon")
            if getattr(value, "ndim", 0) == 5
            else ("batch", "channel", "lat", "lon")
        )
        for name, value in tensors.items()
    }
    wet_field_names = {
        "ground_truth",
        "phase1_prediction",
        "final_refined_prediction",
        "physical_ensemble_members",
        "unbounded_physical_ensemble_members",
        "unbounded_ensemble_mean",
        "historical_memberwise_clipped_mean",
    }
    wet_thresholds = {
        name: {"pr": 1.0}
        for name in tensors
        if name in wet_field_names and "pr" in names
    }
    report = summarize_refinement_tensors(
        tensors,
        channel_names=names,
        channel_dim=channel_dims,
        dimension_names=dimension_names,
        channel_units=channel_units,
        wet_thresholds=wet_thresholds,
    )
    report["sampled_process_coordinates_and_losses"] = process_coordinates
    if prediction.members_unbounded is not None:
        nonnegative_mask = getattr(
            model.phase1,
            "predictand_nonneg_enabled_mask",
            torch.zeros(len(names), dtype=torch.bool),
        ).detach().to(device=prediction.members_unbounded.device, dtype=torch.bool)
        constraint_channels: dict[str, Any] = {}
        for channel, name in enumerate(names):
            values = prediction.members_unbounded[:, :, channel]
            finite = torch.isfinite(values)
            below_zero = finite & (values < 0.0)
            finite_count = int(finite.sum())
            constraint_channels[name] = {
                "nonnegative_constraint_enabled": bool(nonnegative_mask[channel]),
                "unbounded_member_cell_count": finite_count,
                "unbounded_member_cells_below_zero": int(below_zero.sum()),
                "unbounded_member_fraction_below_zero": (
                    None
                    if finite_count == 0
                    else float(below_zero.sum() / finite_count)
                ),
                "members_with_any_negative_cell": int(
                    below_zero.flatten(start_dim=2).any(dim=2).sum()
                ),
            }
        report["physical_constraint_diagnostics"] = {
            "strategy": str(
                model.refinement_config.nonnegative_ensemble_strategy
            ),
            "constraint_application": (
                "once, after inverse residual normalization and one Phase-1 addition"
            ),
            "per_channel": constraint_channels,
        }
    comparisons: dict[str, dict[str, float | int | None]] = {}
    target_normalized = training_view.residual_target
    for channel, name in enumerate(names):
        truth = target_normalized[:, channel].float().reshape(-1)
        sample = raw_normalized[:, channel].float().reshape(-1)
        valid = torch.isfinite(truth) & torch.isfinite(sample)
        truth = truth[valid]
        sample = sample[valid]
        if truth.numel() < 2:
            comparisons[name] = {"finite_count": int(truth.numel()), "correlation": None}
            continue
        truth_centered = truth - truth.mean()
        sample_centered = sample - sample.mean()
        denominator = truth_centered.square().sum().sqrt() * sample_centered.square().sum().sqrt()
        correlation = (
            float((truth_centered * sample_centered).sum() / denominator)
            if float(denominator) > 0.0
            else None
        )
        comparisons[name] = {
            "finite_count": int(truth.numel()),
            "correlation": correlation,
            "rmse": float((sample - truth).square().mean().sqrt()),
            "mae": float((sample - truth).abs().mean()),
            "predicted_to_true_std_ratio": float(
                sample.std(unbiased=False) / truth.std(unbiased=False).clamp(min=1.0e-12)
            ),
        }
    report["raw_sampler_vs_true_residual_normalized"] = {
        "per_channel": comparisons
    }
    physical_skill: dict[str, Any] = {}
    for channel, name in enumerate(names):
        phase1_skill = _paired_skill_metrics(
            prediction.deterministic[:, channel],
            batch["y"][:, channel],
        )
        refined_skill = _paired_skill_metrics(
            prediction.refined[:, channel],
            batch["y"][:, channel],
        )
        physical_skill[name] = {
            "phase1_vs_truth": phase1_skill,
            "refined_vs_truth": refined_skill,
            "refined_minus_phase1": _skill_delta(refined_skill, phase1_skill),
        }
    report["fixed_seed_physical_skill"] = {
        "space": "physical",
        "semantics": (
            "refined is reconstructed from the raw normalized sample via one "
            "inverse residual transform, the explicit correction gate when "
            "present, one Phase-1 residual addition, and final physical constraints"
        ),
        "correction_gate_values": _gate_values(model),
        "per_channel": physical_skill,
    }
    if checkpoint_path is None or checkpoint_payload is None:
        raise RuntimeError(
            "Persisted diagnostics require the exact selected checkpoint path and payload."
        )
    checkpoint_path = checkpoint_path.expanduser().resolve()
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    normalizer_payload = model.residual_normalization_metadata()
    report["provenance"] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_selection": checkpoint_selection,
        "checkpoint_kind": checkpoint_payload.get("checkpoint_kind"),
        "checkpoint_schema_version": checkpoint_payload.get(
            "checkpoint_schema_version"
        ),
        "refinement_contract_version": (
            checkpoint_payload.get("refinement_contract") or {}
        ).get("contract_version"),
        "refinement_contract_fingerprint": checkpoint_payload.get(
            "refinement_contract_fingerprint"
        ),
        "residual_normalizer_state_fingerprint": checkpoint_payload.get(
            "residual_normalizer_state_fingerprint"
        ),
        "phase1_checkpoint": checkpoint_payload.get("phase1_checkpoint"),
        "phase1_fingerprint": checkpoint_payload.get("phase1_fingerprint"),
        "epoch": checkpoint_payload.get("epoch"),
        "global_step": checkpoint_payload.get("global_step"),
        "best_val_loss": checkpoint_payload.get("best_val_loss"),
        "case_name": checkpoint_payload.get("case_name"),
        "refinement_type": checkpoint_payload.get("refinement_type"),
        "residual_normalization": normalizer_payload,
        "resolved_config": checkpoint_payload.get("resolved_config"),
        "split": dict(split_context or {}),
        "sample_identity": sample_identity,
        "seed": int(seed),
        "model_mode": "eval",
        "ensemble_size": diagnostic_ensemble_size,
        "raw_sampler_metrics_are_ungated": True,
        "physical_skill_uses_effective_gated_refinement": True,
    }
    if output_path is not None:
        immutable_path = _write_diagnostic_report(
            report,
            convenience_path=output_path,
            checkpoint_sha256=checkpoint_sha256,
        )
        print(
            f"[diagnostics] convenience={output_path} immutable={immutable_path}"
        )
    print("[diagnostics] one fixed-seed batch:")
    console_report = copy.deepcopy(report)
    console_report["provenance"]["resolved_config"] = (
        "<persisted in diagnostics JSON; omitted from console>"
    )
    print(json.dumps(console_report, indent=2, sort_keys=True, allow_nan=False))
    return report


def _resolved_config_payload(raw: Mapping[str, Any], config: Any, args: argparse.Namespace) -> dict[str, Any]:
    payload = copy.deepcopy(dict(raw))
    payload["num_epochs"] = int(config.num_epochs)
    payload["batch_size"] = int(config.batch_size)
    payload["gradient_accumulation_steps"] = int(
        getattr(config, "gradient_accumulation_steps", 1)
    )
    payload["learning_rate"] = float(config.learning_rate)
    payload["limit_steps_train"] = int(getattr(config, "limit_steps_train", 0))
    payload["limit_steps_valid"] = int(getattr(config, "limit_steps_valid", 0))
    payload["checkpoint_dir"] = str(config.checkpoint_dir)
    if args.case_index is not None:
        data = payload.setdefault("data", {})
        for split in ("training", "validation"):
            predictors, targets = _split_paths(config, split)
            data[f"{split}_predictor_paths"] = list(predictors)
            data[f"{split}_target_paths"] = list(targets)
    payload["runtime_cli"] = {
        "device": str(args.device),
        "tiny_overfit": bool(args.tiny_overfit),
        "case_index": args.case_index,
        "epochs_override": args.epochs,
        "batch_size_override": args.batch_size,
        "gradient_accumulation_steps_override": (
            args.gradient_accumulation_steps
        ),
        "learning_rate_override": args.learning_rate,
        "max_train_batches": args.limit_train,
        "max_validation_batches": args.limit_valid,
        "explicit_resume": args.resume is not None,
        "validation_fraction": args.validation_fraction,
    }
    return payload


def run(args: argparse.Namespace) -> int:
    # Imports are intentionally delayed so ``--help`` remains available in a
    # lightweight environment and this module can be imported by CLI tests.
    import torch
    import yaml
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))

    from granitewxc.models.model import get_finetune_model_UNET
    from granitewxc.refinement.checkpoint import (
        CHECKPOINT_KIND_COMBINED,
        CHECKPOINT_KIND_REFINEMENT,
        CHECKPOINT_SCHEMA_VERSION,
        load_phase1_state_dict,
        load_refinement_state_dict,
        validate_phase1_reference,
    )
    from granitewxc.refinement.training import RefinementTrainer
    from granitewxc.refinement.two_phase import build_two_phase_model
    from granitewxc.utils.config import get_config
    from granitewxc.utils.predictands import build_predictand_specs

    config_path = _resolve_config_path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_config, Mapping):
        raise ValueError(f"Expected a YAML mapping in {config_path}.")
    config = get_config(str(config_path))
    _select_case(config, args.case_index)

    device = _resolve_device(args.device)
    use_gpu = device.type == "cuda"
    config.device_target = device.type
    if args.epochs is not None:
        config.num_epochs = int(args.epochs)
    if args.batch_size is not None:
        config.batch_size = int(args.batch_size)
    if args.gradient_accumulation_steps is not None:
        config.gradient_accumulation_steps = int(
            args.gradient_accumulation_steps
        )
    if args.learning_rate is not None:
        config.learning_rate = float(args.learning_rate)
    if args.limit_train is not None:
        config.limit_steps_train = int(args.limit_train)
    if args.limit_valid is not None:
        config.limit_steps_valid = int(args.limit_valid)
    if args.tiny_overfit:
        config.batch_size = 1
        config.dl_num_workers = 0
        config.dl_prefetch_size = 0
        config.dl_pin_memory = False

    configured_checkpoint_dir = args.checkpoint_dir or getattr(config, "checkpoint_dir", None)
    if not configured_checkpoint_dir:
        refinement = getattr(config.model, "refinement", {}) or {}
        refiner_type = refinement.get("type", "unknown") if isinstance(refinement, Mapping) else "unknown"
        configured_checkpoint_dir = Path("experiments/refinement_checkpoints") / str(refiner_type)
    checkpoint_dir = _resolve_project_path(configured_checkpoint_dir)
    if args.tiny_overfit and args.checkpoint_dir is None:
        # A smoke test must never overwrite a long-running scientific run just
        # because the caller reused its YAML.
        checkpoint_dir = checkpoint_dir / "tiny_overfit"
    config.checkpoint_dir = str(checkpoint_dir)

    resume_path: Path | None = None
    if args.resume is not None:
        resume_path = (
            checkpoint_dir / "last.ckpt"
            if args.resume == _LAST_CHECKPOINT
            else _resolve_project_path(args.resume)
        )
        if not resume_path.is_file():
            raise FileNotFoundError(f"Requested refinement checkpoint not found: {resume_path}")
    elif (checkpoint_dir / "last.ckpt").exists() or (checkpoint_dir / "best.ckpt").exists():
        raise FileExistsError(
            f"Checkpoint directory already contains a run: {checkpoint_dir}. "
            "Use --resume or choose a new --checkpoint-dir; fresh training will "
            "not overwrite existing Phase-2 checkpoints."
        )

    phase1_config = getattr(config.model, "phase1", None)
    if not isinstance(phase1_config, Mapping) or not phase1_config.get("checkpoint"):
        raise ValueError("model.phase1.checkpoint must name the deterministic checkpoint.")
    phase1_path = _resolve_repo_path(str(phase1_config["checkpoint"]))
    if not phase1_path.is_file():
        raise FileNotFoundError(f"Phase-1 checkpoint not found: {phase1_path}")

    refinement_config = getattr(config.model, "refinement", {}) or {}
    seed = int(refinement_config.get("seed", 42)) if isinstance(refinement_config, Mapping) else 42
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_loader, val_loader, validation_source = _build_loaders(
        config,
        use_gpu=use_gpu,
        tiny_overfit=args.tiny_overfit,
        validation_fraction=args.validation_fraction,
    )
    if args.limit_train not in (None, 0) and not args.tiny_overfit:
        train_loader = _LimitedLoader(train_loader, args.limit_train)
        config.limit_steps_train = 0
    if args.limit_valid not in (None, 0) and val_loader is not None and not args.tiny_overfit:
        val_loader = _LimitedLoader(val_loader, args.limit_valid)
        config.limit_steps_valid = 0

    if len(train_loader) < 1:
        raise RuntimeError("The configured training loader is empty.")
    if val_loader is not None and len(val_loader) < 1:
        raise RuntimeError("The configured validation loader is empty.")

    sample = next(iter(train_loader), None)
    if sample is None:
        raise RuntimeError("The configured training loader is empty.")

    if not getattr(config.data, "input_static_surface_vars", None):
        config.data.input_static_surface_vars = []
    specs = build_predictand_specs(config, output_vars=list(config.data.output_vars))
    print(f"[config] {config_path}")
    print(f"[device] {device}")
    print(f"[data] train_batches={len(train_loader)} validation_source={validation_source}")
    print(f"[data] val_batches={len(val_loader) if val_loader is not None else 0}")
    print("[predictands] " + ", ".join(spec.name for spec in specs))
    print(f"[checkpoint] Phase 1={phase1_path}")
    print(f"[checkpoint] Phase 2 directory={checkpoint_dir}")

    phase1 = get_finetune_model_UNET(config)
    model = build_two_phase_model(phase1, config).to(device)
    if not model.refinement_config.is_active:
        raise ValueError("The selected YAML does not enable a Phase-2 refinement head.")
    if not model.phase1_frozen or any(parameter.requires_grad for parameter in model.phase1.parameters()):
        raise RuntimeError("Phase-2 training requires Phase 1 to be frozen before loading.")
    model.phase1.eval()

    try:
        phase1_payload = torch.load(str(phase1_path), map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0
        phase1_payload = torch.load(str(phase1_path), map_location="cpu")
    report = load_phase1_state_dict(model, phase1_payload)
    print(f"[checkpoint] strict Phase-1 load: {report.summary()}")

    init_batch = _move_batch(sample, device)
    model.initialize_from_batch(init_batch)
    del init_batch
    if model.refiner is None:
        raise RuntimeError("The active refiner was not initialized from the training batch.")
    if model.phase1.training or not model.phase1_frozen:
        raise RuntimeError("Phase 1 must remain frozen and in eval mode after refiner initialization.")

    trainable = model.trainable_parameters()
    if not trainable:
        raise RuntimeError("No trainable Phase-2 parameters were found.")
    if any(parameter.requires_grad for parameter in model.phase1.parameters()):
        raise RuntimeError("A Phase-1 parameter became trainable during initialization.")
    _print_gate(model, "before fit/resume")

    accum = max(1, int(getattr(config, "gradient_accumulation_steps", 1)))
    train_batches = max(1, len(train_loader))
    configured_limit = int(getattr(config, "limit_steps_train", 0))
    effective_batches = min(train_batches, configured_limit) if configured_limit > 0 else train_batches
    optimizer_steps = max(1, math.ceil(effective_batches / accum))
    total_steps = max(1, int(config.num_epochs) * optimizer_steps)
    optimizer = AdamW(trainable, lr=float(config.learning_rate))
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=float(getattr(config, "min_lr", 0.0)),
    )
    precision_mode = str(model.performance_config.precision.mode).lower()
    scaler = _build_scaler(torch, enabled=use_gpu and precision_mode == "fp16")

    resolved_payload = _resolved_config_payload(raw_config, config, args)
    trainer = RefinementTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        checkpoint_dir=str(checkpoint_dir),
        phase1_checkpoint=str(phase1_path),
        resolved_config=resolved_payload,
        case_name=str(getattr(config, "case_name", config_path.stem)),
        gradient_accumulation_steps=accum,
        max_grad_norm=float(getattr(config, "max_grad_norm", 1.0)),
        seed=seed,
        log_every=int(getattr(config, "log_every", 50)),
    )

    if resume_path is not None:
        trainer.resume(str(resume_path), strict_phase1_reference=True)
        _print_gate(model, "after resume")

    target_epochs = int(config.num_epochs)
    completed_epochs = int(trainer.state.epoch)
    remaining_epochs = max(0, target_epochs - completed_epochs)
    best_path = checkpoint_dir / "best.ckpt"
    best_sha_before = _sha256_file(best_path) if best_path.is_file() else None
    print(
        f"[train] target_epochs={target_epochs} completed={completed_epochs} "
        f"remaining={remaining_epochs} accumulation={accum}"
    )
    state = trainer.fit(
        train_loader,
        val_loader,
        num_epochs=remaining_epochs,
        limit_steps_train=int(getattr(config, "limit_steps_train", 0)),
        limit_steps_valid=int(getattr(config, "limit_steps_valid", 0)),
    )
    _print_gate(model, "after fit")
    print(
        "[normalization] "
        + json.dumps(
            model.residual_normalization_metadata(),
            sort_keys=True,
            default=str,
        )
    )
    gate_values = _gate_values(model)
    if (
        state.epoch > 0
        and gate_values is not None
        and max(map(abs, gate_values), default=0.0) <= 1.0e-12
    ):
        raise RuntimeError(
            "A trained Transformer checkpoint still has an exactly-zero correction gate; "
            "the refinement cannot change Phase 1."
        )

    if remaining_epochs > 0:
        last_path = checkpoint_dir / "last.ckpt"
        if not last_path.is_file():
            raise RuntimeError(f"Training completed without expected checkpoint: {last_path}")
        saved = _load_checkpoint_payload(last_path)
        if saved.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise RuntimeError(
                f"Saved checkpoint schema={saved.get('checkpoint_schema_version')!r}, "
                f"expected {CHECKPOINT_SCHEMA_VERSION}."
            )
        if saved.get("checkpoint_kind") not in {
            CHECKPOINT_KIND_REFINEMENT,
            CHECKPOINT_KIND_COMBINED,
        }:
            raise RuntimeError(f"Unexpected saved checkpoint kind: {saved.get('checkpoint_kind')!r}")
        print(
            f"[checkpoint] verified schema-{CHECKPOINT_SCHEMA_VERSION} "
            f"{saved['checkpoint_kind']} checkpoint: {last_path}"
        )

    last_path = checkpoint_dir / "last.ckpt"
    if remaining_epochs > 0 and last_path.is_file():
        diagnostic_checkpoint = last_path
        checkpoint_selection = "last checkpoint from current invocation"
    elif resume_path is not None and resume_path.is_file():
        diagnostic_checkpoint = resume_path
        checkpoint_selection = "explicit resume checkpoint (no new epochs)"
    elif last_path.is_file():
        diagnostic_checkpoint = last_path
        checkpoint_selection = "last checkpoint"
    else:
        diagnostic_checkpoint = None
        checkpoint_selection = "unavailable"
    if diagnostic_checkpoint is None:
        raise RuntimeError("No refinement checkpoint is available for diagnostics.")
    diagnostic_payload = _load_checkpoint_payload(diagnostic_checkpoint)

    best_sha_after = _sha256_file(best_path) if best_path.is_file() else None
    best_was_selected_now = (
        remaining_epochs > 0
        and best_sha_after is not None
        and best_sha_after != best_sha_before
    )
    if best_was_selected_now:
        candidate = _load_checkpoint_payload(best_path)
        if _checkpoint_belongs_to_run(candidate, diagnostic_payload):
            diagnostic_checkpoint = best_path
            diagnostic_payload = candidate
            checkpoint_selection = "best checkpoint selected during current invocation"
        else:
            print(
                "[diagnostics] rejected updated best.ckpt because its run "
                "identity does not match current last.ckpt; using last.ckpt"
            )
    elif best_path.is_file():
        print(
            "[diagnostics] best.ckpt was not selected during this invocation; "
            f"using {diagnostic_checkpoint.name} to avoid stale-best ambiguity"
        )
    validate_phase1_reference(
        diagnostic_payload, model.phase1.state_dict(), strict=True
    )
    diagnostic_report = load_refinement_state_dict(model, diagnostic_payload)
    model.eval()
    print(
        f"[diagnostics] strict selected-checkpoint load: {diagnostic_report.summary()} "
        f"from {diagnostic_checkpoint} ({checkpoint_selection})"
    )
    _print_gate(model, "diagnostic checkpoint")

    diagnostic_loader = val_loader if val_loader is not None else train_loader
    if args.tiny_overfit:
        configured_split = "training"
    else:
        configured_split = "validation" if val_loader is not None else "training"
    diagnostic_predictors, diagnostic_targets = _split_paths(config, configured_split)
    split_context = {
        "loader_role": "validation" if val_loader is not None else "training",
        "configured_split": configured_split,
        "source": validation_source if val_loader is not None else "training loader",
        "predictor_paths": [str(path) for path in diagnostic_predictors],
        "target_paths": [str(path) for path in diagnostic_targets],
    }
    _print_diagnostics(
        model,
        diagnostic_loader,
        device,
        seed=seed + 10_003,
        output_path=checkpoint_dir / "diagnostics.json",
        checkpoint_path=diagnostic_checkpoint,
        checkpoint_payload=diagnostic_payload,
        checkpoint_selection=checkpoint_selection,
        split_context=split_context,
    )
    best = "n/a" if state.best_val_loss is None else f"{state.best_val_loss:.8g}"
    print(
        f"[done] epoch={state.epoch} global_step={state.global_step} "
        f"best_val_loss={best}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] == "--scientific-experiment":
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from granitewxc.refinement.scientific_entrypoints import dispatch_scientific
        return dispatch_scientific(values, expected_operation="train")
    args = build_parser().parse_args(values)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
