"""Phase-2 (stochastic refinement) entry point for the NARR-to-PRISM workflow.

Subcommands
-----------
``train``
    Load an existing deterministic Phase-1 checkpoint, freeze it, and train the
    configured Phase-2 refiner on residuals. Resumable.

``infer``
    Run deterministic and refined ensemble inference over the exact configured
    inference dates and write one canonical-grid NetCDF file per day.

``describe``
    Print the fully resolved configuration and model summary without running
    anything.

The deterministic Phase-1 checkpoint is opened read-only and never modified;
Phase-2 checkpoints are written to ``checkpoint_dir`` from the YAML.

Examples::

    python examples/NARR_PRISM/narr_prism_refinement.py train \
        --config examples/NARR_PRISM/NARR_PRISM_diffusion_unet.yaml

    python examples/NARR_PRISM/narr_prism_refinement.py train \
        --config examples/NARR_PRISM/NARR_PRISM_diffusion_unet.yaml --resume

    python examples/NARR_PRISM/narr_prism_refinement.py infer \
        --config examples/NARR_PRISM/NARR_PRISM_flow_matching_unet.yaml \
        --refinement-checkpoint .../refinement_checkpoints/flow_matching_unet/best.ckpt \
        --ensemble-size 10 --output /tmp/narr_refined
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
import random
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from granitewxc.refinement import TwoPhaseDownscalingModel  # noqa: E402
from granitewxc.refinement.checkpoint import (  # noqa: E402
    load_phase1_state_dict,
    phase1_state_fingerprint,
)
from granitewxc.refinement.config import (  # noqa: E402
    resolve_performance_config,
    resolve_refinement_config,
)
from granitewxc.refinement.training import RefinementTrainer  # noqa: E402
from granitewxc.utils.config import get_config  # noqa: E402
from granitewxc.utils.normalization import (  # noqa: E402
    apply_scalar_paths,
    assert_scalars_available,
    log_case_context,
)
from granitewxc.utils.prism_checkpoint import (  # noqa: E402
    validate_prism_checkpoint_contract,
)
from narr_prism_training import create_finetune_model, get_dataloaders  # noqa: E402
from narr_prism_utils import get_case_name, load_yaml  # noqa: E402


def _resolve(path: str | os.PathLike) -> str:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    return str((REPO_ROOT / candidate).resolve())


def _phase1_checkpoint(config, override: str | None) -> str | None:
    if override:
        return _resolve(override)
    phase1_cfg = getattr(config.model, "phase1", None) or {}
    if isinstance(phase1_cfg, dict):
        path = phase1_cfg.get("checkpoint")
    else:
        path = getattr(phase1_cfg, "checkpoint", None)
    return _resolve(path) if path else None


def _refinement_checkpoint(config, override: str | None) -> str | None:
    """Resolve a Phase-2 checkpoint, preferring an explicit CLI override."""
    if override:
        return _resolve(override)
    refinement_cfg = getattr(config.model, "refinement", None) or {}
    if isinstance(refinement_cfg, dict):
        path = refinement_cfg.get("checkpoint")
    else:
        path = getattr(refinement_cfg, "checkpoint", None)
    return _resolve(path) if path else None


def _require_checkpoint_path(path: str | None, *, label: str) -> str:
    """Return an accessible regular checkpoint file or fail before model setup.

    ``Path.exists`` can hide useful context for dangling/recursive symlinks, so
    the file is also opened read-only. This catches the historical NARR_PRISM
    self-referential ``experiments`` symlink with a clear artifact error.
    """
    if not path:
        raise FileNotFoundError(
            f"{label} checkpoint is required. Set it in the YAML or pass the "
            f"corresponding CLI checkpoint option."
        )
    candidate = Path(path).expanduser()
    try:
        accessible = candidate.is_file()
    except OSError as exc:
        raise FileNotFoundError(
            f"{label} checkpoint is not accessible: {candidate} ({exc})"
        ) from exc
    if not accessible:
        if candidate.is_symlink():
            try:
                candidate.resolve(strict=True)
            except OSError as exc:
                raise FileNotFoundError(
                    f"{label} checkpoint is not accessible: {candidate} ({exc})"
                ) from exc
        raise FileNotFoundError(
            f"{label} checkpoint is not a regular file: {candidate}"
        )
    try:
        with candidate.open("rb"):
            pass
    except OSError as exc:
        raise FileNotFoundError(
            f"{label} checkpoint is not accessible: {candidate} ({exc})"
        ) from exc
    return str(candidate.resolve())


def build_model(
    config,
    config_path: str,
    phase1_checkpoint: str | None,
    device: torch.device,
    *,
    verbose: bool = True,
):
    """Build the two-phase model and load the deterministic Phase-1 weights."""
    if verbose:
        log_case_context(config, "refinement")
    assert_scalars_available(config, role="refinement")
    apply_scalar_paths(config)

    refinement = resolve_refinement_config(config)
    performance = resolve_performance_config(config)
    phase1 = create_finetune_model(config, verbose=verbose)
    model = TwoPhaseDownscalingModel(phase1, refinement=refinement, performance=performance)

    if refinement.is_active:
        phase1_checkpoint = _require_checkpoint_path(
            phase1_checkpoint, label="Phase-1 deterministic"
        )

    fingerprint = None
    if phase1_checkpoint:
        if verbose:
            print(f"[refinement] loading Phase-1 checkpoint (read-only): {phase1_checkpoint}")
        checkpoint = torch.load(phase1_checkpoint, map_location="cpu", mmap=True, weights_only=False)
        # Validate channel ordering, dates, grid/scaler hashes, transforms and
        # architecture semantics before a single checkpoint tensor is applied.
        validate_prism_checkpoint_contract(
            config, checkpoint, role="NARR refinement Phase-1 load"
        )
        report = load_phase1_state_dict(model, checkpoint)
        if verbose:
            print(f"[refinement] Phase-1 load: {report.summary()}")
        unexplained = [k for k in report.missing if not k.startswith("refiner.")]
        if unexplained:
            raise RuntimeError(f"Unexplained missing Phase-1 keys: {unexplained[:8]}")
        fingerprint = phase1_state_fingerprint(
            {k[len("phase1.") :]: v for k, v in model.state_dict().items() if k.startswith("phase1.")}
        )
        if verbose:
            print(f"[refinement] Phase-1 fingerprint: {fingerprint}")

    model.to(device)
    return model, fingerprint


def _first_batch(loader):
    for batch in loader:
        return batch
    raise RuntimeError("The dataloader produced no batches.")


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _resume_with_batch_size_migration(
    trainer: RefinementTrainer,
    path: str,
    *,
    previous_per_device_batch_size: int,
    current_per_device_batch_size: int,
    previous_num_epochs: int = 0,
    current_num_epochs: int = 0,
    announce: bool = True,
):
    """Resume a validated batch migration and optional epoch-budget extension."""
    previous_batch = int(previous_per_device_batch_size)
    current_batch = int(current_per_device_batch_size)
    previous_epochs = int(previous_num_epochs or 0)
    current_epochs = int(current_num_epochs or 0)
    if previous_batch < 1 or current_batch < 1:
        raise ValueError("Per-device batch sizes must be positive")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    saved_training = (payload.get("resolved_config") or {}).get("training")
    current_training = trainer.resolved_config.get("training")
    if not isinstance(saved_training, Mapping) or not isinstance(
        current_training, Mapping
    ):
        raise RuntimeError(
            "Batch-size migration requires saved and current training contracts"
        )
    saved_training = dict(saved_training)
    current_training = dict(current_training)
    saved_extension = saved_training.get("schedule_extension")
    if saved_extension is not None:
        current_training["schedule_extension"] = saved_extension
        trainer.resolved_config["training"] = current_training
    recorded_batch = saved_training.get("per_device_batch_size")
    saved_epochs = int(saved_training.get("epochs", current_epochs or 0))
    configured_epochs = int(current_training.get("epochs", current_epochs or 0))
    if (
        recorded_batch is not None
        and int(recorded_batch) == current_batch
        and saved_epochs == configured_epochs
    ):
        return trainer.resume(path)
    if recorded_batch is not None and int(recorded_batch) != previous_batch:
        raise RuntimeError(
            "Batch-size migration source mismatch: checkpoint records "
            f"per_device_batch_size={recorded_batch}, requested={previous_batch}."
        )

    scaled_fields = (
        "gradient_accumulation_steps",
        "limit_steps_train",
        "limit_steps_valid",
        "effective_train_batches_per_epoch",
    )
    for field in scaled_fields:
        if field not in saved_training or field not in current_training:
            raise RuntimeError(f"Batch-size migration contract lacks {field}")
        saved_samples = int(saved_training[field]) * previous_batch
        current_samples = int(current_training[field]) * current_batch
        if saved_samples != current_samples:
            raise RuntimeError(
                f"Batch-size migration changes {field}: "
                f"saved={saved_training[field]}x{previous_batch}, "
                f"current={current_training[field]}x{current_batch}."
            )

    epoch_changed = saved_epochs != configured_epochs
    epoch_fields: set[str] = set()
    if epoch_changed:
        if previous_epochs != saved_epochs or current_epochs != configured_epochs:
            raise RuntimeError(
                "Epoch-budget migration source mismatch: "
                f"checkpoint={saved_epochs}, requested={previous_epochs}, "
                f"configured={configured_epochs}, requested_current={current_epochs}."
            )
        if configured_epochs <= saved_epochs:
            raise RuntimeError("Epoch-budget migration must extend the saved budget")
        saved_updates = int(saved_training["optimizer_steps_per_epoch"])
        current_updates = int(current_training["optimizer_steps_per_epoch"])
        if saved_updates != current_updates:
            raise RuntimeError(
                "Epoch-budget migration must preserve optimizer steps per epoch"
            )
        expected_saved_total = saved_epochs * saved_updates
        expected_current_total = configured_epochs * current_updates
        for field in ("scheduler_t_max", "total_optimizer_steps"):
            if int(saved_training[field]) != expected_saved_total:
                raise RuntimeError(
                    f"Saved {field} is inconsistent with its epoch budget"
                )
            if int(current_training[field]) != expected_current_total:
                raise RuntimeError(
                    f"Current {field} is inconsistent with its epoch budget"
                )
        epoch_fields = {"epochs", "scheduler_t_max", "total_optimizer_steps"}

    mutable_fields = set(scaled_fields) | {
        "per_device_batch_size",
        "schedule_extension",
    } | epoch_fields
    saved_fixed = {
        key: value for key, value in saved_training.items() if key not in mutable_fields
    }
    current_fixed = {
        key: value for key, value in current_training.items() if key not in mutable_fields
    }
    if saved_fixed != current_fixed:
        changed = sorted(
            key
            for key in set(saved_fixed) | set(current_fixed)
            if saved_fixed.get(key, "<missing>")
            != current_fixed.get(key, "<missing>")
        )
        raise RuntimeError(
            "Batch-size migration cannot change non-batch training fields: "
            + ", ".join(changed)
        )

    trainer.resolved_config["training"] = saved_training
    try:
        state = trainer.resume(path)
    finally:
        trainer.resolved_config["training"] = current_training

    if epoch_changed:
        remaining_epochs = configured_epochs - int(state.epoch)
        remaining_steps = remaining_epochs * int(
            current_training["optimizer_steps_per_epoch"]
        )
        if remaining_steps <= 0:
            raise RuntimeError("Extended epoch budget has no remaining optimizer steps")
        start_lrs = [float(group["lr"]) for group in trainer.optimizer.param_groups]
        for group, start_lr in zip(
            trainer.optimizer.param_groups, start_lrs, strict=True
        ):
            group["initial_lr"] = start_lr
        trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer,
            T_max=remaining_steps,
            eta_min=float(current_training["min_learning_rate"]),
        )
        current_training["schedule_extension"] = {
            "from_epochs": saved_epochs,
            "to_epochs": configured_epochs,
            "at_epoch": int(state.epoch),
            "at_global_step": int(state.global_step),
            "remaining_optimizer_steps": remaining_steps,
            "start_lrs": start_lrs,
        }
        trainer.resolved_config["training"] = current_training
    if announce:
        message = (
            "[refinement] validated resume migration: "
            f"per_device={previous_batch}->{current_batch}; "
            "sample budget and optimizer batch unchanged"
        )
        if epoch_changed:
            message += (
                f"; epochs={saved_epochs}->{configured_epochs}; cosine schedule "
                f"continued from lr={start_lrs[0]:.8g} over {remaining_steps} steps"
            )
        print(message)
    return state


def _save_epochs_vs_loss_curve(state: Any, checkpoint_dir: str) -> str | None:
    """Atomically render the complete resumed train/validation loss history."""
    train_loss = [float(value) for value in state.train_loss_history]
    val_loss = [float(value) for value in state.val_loss_history]
    if not train_loss and not val_loss:
        return None

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output = Path(checkpoint_dir) / "epochs_vs_loss.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".png.tmp")
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    if train_loss:
        train_epochs = np.arange(1, len(train_loss) + 1)
        axis.plot(train_epochs, train_loss, label="Training loss", linewidth=1.8)
    if val_loss:
        val_epochs = np.arange(1, len(val_loss) + 1)
        axis.plot(val_epochs, val_loss, label="Validation loss", linewidth=1.8)
        finite = np.asarray(val_loss, dtype=np.float64)
        if np.isfinite(finite).any():
            best_index = int(np.nanargmin(finite))
            best_epoch = best_index + 1
            best_loss = float(finite[best_index])
            axis.scatter(
                [best_epoch], [best_loss], color="black", marker="*", s=110,
                zorder=5, label=f"Best validation: epoch {best_epoch}",
            )
            axis.annotate(
                f"{best_loss:.5g}",
                (best_epoch, best_loss),
                xytext=(7, 7), textcoords="offset points",
            )
    combined = np.asarray(train_loss + val_loss, dtype=np.float64)
    finite_positive = combined[np.isfinite(combined) & (combined > 0)]
    if finite_positive.size and finite_positive.max() / finite_positive.min() >= 20:
        axis.set_yscale("log")
        axis.set_ylabel("Loss (log scale)")
    else:
        axis.set_ylabel("Loss")
    axis.set_xlabel("Epoch")
    axis.set_title("Phase-2 refinement: epochs versus loss")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.savefig(temporary, format="png", dpi=160)
    plt.close(figure)
    os.replace(temporary, output)
    return str(output)


def _seed_training(seed: int) -> None:
    """Seed every RNG used before the Phase-2 model and loaders are built."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_phase1_cache_runtime(
    performance, *, role: str, verbose: bool = True
) -> dict[str, Any] | None:
    """Apply the YAML precision policy before any live Phase-1 CUDA forward."""
    precision = performance.precision
    allow_tf32 = bool(precision.allow_tf32)
    if performance.phase1_cache.enabled and (
        str(precision.mode).lower() != "fp32" or allow_tf32
    ):
        raise RuntimeError(
            "The Phase-1 residual cache was generated in strict FP32 with TF32 "
            "disabled; the active performance.precision contract must use "
            "mode=fp32 and allow_tf32=false."
        )
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
    if torch.backends.cuda.is_built():
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
    policy = {
        "mode": str(precision.mode).lower(),
        "allow_tf32": allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }
    if verbose:
        print(
            f"[refinement] {role} precision policy: "
            + json.dumps(policy, sort_keys=True),
            flush=True,
        )
    return policy


def _load_phase1_cache(
    *,
    config,
    raw_config: Mapping[str, Any],
    phase1_checkpoint: str,
    phase1_fingerprint: str,
    performance,
    validate_inventory: bool = True,
    show_inventory_progress: bool = True,
    verbose: bool = True,
):
    """Open and authenticate the immutable daily Phase-1/residual cache."""
    cache_cfg = performance.phase1_cache
    if not cache_cfg.enabled:
        return None, None
    from narr_prism_phase1_cache import Phase1ResidualCacheReader

    cache_root = _resolve(cache_cfg.path)
    try:
        reader = Phase1ResidualCacheReader.from_path(
            cache_root,
            cfg=raw_config,
            config=config,
            phase1_checkpoint=phase1_checkpoint,
            phase1_fingerprint=phase1_fingerprint,
            validate_inventory=validate_inventory,
            show_inventory_progress=(
                validate_inventory and show_inventory_progress
            ),
        )
    except Exception as exc:
        raise RuntimeError(
            "The configured Phase-1 residual cache is missing, incomplete, or "
            f"incompatible: {cache_root}. Build/repair it with "
            "`mamba run -n Prithvi python "
            "examples/NARR_PRISM/narr_prism_phase1_cache.py build "
            f"--config {raw_config.get('_config_path', '<refinement.yaml>')} "
            f"--checkpoint {phase1_checkpoint}`. Original error: {exc}"
        ) from exc
    # The full inventory and contract were authenticated above. Avoid a second
    # validation/open for every spatial crop inside DataLoader workers.
    reader.validate_daily = False
    manifest = reader.manifest
    if verbose:
        print(
            "[refinement] authenticated Phase-1 residual cache: "
            f"manifest={manifest['_manifest_path']} "
            f"contract_digest={manifest['contract_digest']}"
        )
    return reader, manifest


def _validated_cache_statistics(
    model: TwoPhaseDownscalingModel,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the training-only residual statistics stored in a cache."""
    from narr_prism_phase1_cache import RESIDUAL_DEFINITION, TARGET_SPACE_NAME

    metadata = manifest.get("residual_normalization")
    if not isinstance(metadata, Mapping):
        raise RuntimeError(
            "A complete Phase-1 residual cache must contain training residual "
            "normalization metadata. Re-finalize the cache."
        )
    expected_variables = list(manifest["contract"]["target_variables"])
    if list(metadata.get("target_variables") or []) != expected_variables:
        raise RuntimeError(
            "Cache residual-normalization channel order does not match its "
            "ordered target-variable contract."
        )
    if metadata.get("fit_split") != "training":
        raise RuntimeError("Cache residual statistics were not fitted on training data.")
    if metadata.get("space") != TARGET_SPACE_NAME:
        raise RuntimeError("Cache uses an unsupported residual target space.")
    if metadata.get("residual_definition") != RESIDUAL_DEFINITION:
        raise RuntimeError("Cache uses an unsupported residual sign/definition.")
    if not bool(metadata.get("enabled")) or not bool(metadata.get("fitted")):
        raise RuntimeError("Cache residual-normalization metadata is not fitted.")
    epsilon = float(metadata.get("epsilon", float("nan")))
    configured_epsilon = float(
        model.refinement_config.residual_normalization.epsilon
    )
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise RuntimeError(
            "Cache residual-normalization epsilon must be finite and positive."
        )
    if not np.isfinite(configured_epsilon) or configured_epsilon <= 0:
        raise RuntimeError(
            "Configured residual-normalization epsilon must be finite and positive."
        )
    if epsilon != configured_epsilon:
        raise RuntimeError(
            "Cache residual-normalization epsilon does not match the refinement "
            f"configuration ({epsilon} != {configured_epsilon})."
        )
    channels = model.target_space.num_channels
    arrays = {}
    for name, dtype in (("mean", np.float64), ("std", np.float64), ("count", np.int64)):
        value = np.asarray(metadata.get(name), dtype=dtype)
        if value.shape != (channels,):
            raise RuntimeError(
                f"Cache residual statistic {name!r} has shape {value.shape}; "
                f"expected ({channels},)."
            )
        arrays[name] = value
    if not np.isfinite(arrays["mean"]).all() or not np.isfinite(arrays["std"]).all():
        raise RuntimeError("Cache residual mean/std contains NaN or infinity.")
    if np.any(arrays["std"] <= 0) or np.any(arrays["count"] <= 0):
        raise RuntimeError("Cache residual std/count must be strictly positive.")
    return dict(metadata)


def _install_cache_statistics(
    model: TwoPhaseDownscalingModel,
    metadata: Mapping[str, Any],
) -> None:
    """Install cache statistics in a fresh head; checkpoint buffers own resume."""
    if not model.refinement_config.residual_normalization.enabled:
        return
    if model.refiner is None:
        raise RuntimeError("Refinement head must be initialized before installing statistics.")
    model.refiner.set_residual_normalization(
        torch.as_tensor(metadata["mean"]),
        torch.as_tensor(metadata["std"]),
        torch.as_tensor(metadata["count"], dtype=torch.int64),
    )
    print(
        "[refinement] installed cached training residual normalization; "
        "the live residual-normalization scan is skipped"
    )


def _assert_cache_statistics_match_resume(
    model: TwoPhaseDownscalingModel,
    metadata: Mapping[str, Any],
    *,
    tolerance: float,
) -> None:
    """Fail if resumed checkpoint normalizers differ from the cache contract."""
    if not model.refinement_config.residual_normalization.enabled:
        return
    if model.refiner is None:
        raise RuntimeError("Resumed checkpoint did not initialize a refinement head.")
    observed = model.refiner.residual_normalization_metadata()
    if not bool(observed.get("fitted")):
        raise RuntimeError("Resumed checkpoint has no fitted residual normalizer.")
    for name in ("mean", "std"):
        actual = np.asarray(observed[name], dtype=np.float64)
        expected = np.asarray(metadata[name], dtype=np.float64)
        if not np.allclose(actual, expected, rtol=0.0, atol=tolerance):
            raise RuntimeError(
                f"Resumed checkpoint residual {name} differs from the authenticated "
                f"cache (max_abs={np.max(np.abs(actual - expected)):.6g}, "
                f"tolerance={tolerance:.6g})."
            )
    if not np.array_equal(
        np.asarray(observed["count"], dtype=np.int64),
        np.asarray(metadata["count"], dtype=np.int64),
    ):
        raise RuntimeError("Resumed checkpoint residual counts differ from the cache.")
    print("[refinement] resumed residual-normalization buffers match the cache")


def _single_sample(batch: Mapping[str, Any], index: int, batch_size: int) -> dict[str, Any]:
    result = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim and value.shape[0] == batch_size:
            result[key] = value[index : index + 1]
        elif isinstance(value, (list, tuple)) and len(value) == batch_size:
            result[key] = value[index : index + 1]
        else:
            result[key] = value
    return result


@torch.no_grad()
def _validate_phase1_cache_parity(
    model: TwoPhaseDownscalingModel,
    loader,
    *,
    device: torch.device,
    samples: int,
    tolerance: float,
) -> dict[str, float]:
    """Compare cached fields with live Phase-1 and residual construction."""
    requested = int(samples)
    if requested <= 0:
        return {"samples": 0, "baseline_max_abs": 0.0, "residual_max_abs": 0.0}
    maxima = {"baseline_max_abs": 0.0, "residual_max_abs": 0.0}
    checked = 0
    was_training = model.training
    model.eval()
    try:
        for batch in loader:
            batch_size = int(batch["__phase1_normalized"].shape[0])
            for index in range(batch_size):
                sample = _to_device(_single_sample(batch, index, batch_size), device)
                cached_baseline = sample["__phase1_normalized"]
                cached_residual = sample["__residual_target_normalized"]
                cached_valid = sample["__residual_valid_mask"].bool()
                live_batch = dict(sample)
                live_batch.pop("__phase1_normalized")
                live_batch.pop("__residual_target_normalized")
                live_batch.pop("__residual_valid_mask")
                _, live_baseline, _ = model.run_phase1(live_batch)
                offset = live_batch.get(
                    "__output_scaler_offset", live_batch.get("__scaler_offset")
                )
                live_residual, live_valid = model.target_space.residual_target(
                    live_batch["y"], live_baseline, scaler_offset=offset
                )
                if not torch.equal(cached_valid, live_valid.bool()):
                    raise RuntimeError(
                        "Cached residual validity mask differs from live target validity."
                    )
                baseline_error = float(
                    (cached_baseline - live_baseline).abs().max().detach().cpu()
                )
                residual_error = float(
                    (cached_residual - live_residual).abs().max().detach().cpu()
                )
                maxima["baseline_max_abs"] = max(maxima["baseline_max_abs"], baseline_error)
                maxima["residual_max_abs"] = max(maxima["residual_max_abs"], residual_error)
                checked += 1
                if baseline_error > tolerance or residual_error > tolerance:
                    raise RuntimeError(
                        "Phase-1 cache parity validation failed: "
                        f"baseline_max_abs={baseline_error:.6g}, "
                        f"residual_max_abs={residual_error:.6g}, "
                        f"tolerance={tolerance:.6g}."
                    )
                if checked >= requested:
                    maxima["samples"] = checked
                    print(
                        "[refinement] Phase-1 cache live parity: "
                        f"samples={checked} baseline_max_abs={maxima['baseline_max_abs']:.6g} "
                        f"residual_max_abs={maxima['residual_max_abs']:.6g} "
                        f"tolerance={tolerance:.6g}"
                    )
                    return maxima
    finally:
        model.train(was_training)
    raise RuntimeError(
        f"Phase-1 cache parity requested {requested} samples but loader supplied {checked}."
    )


class _RefinementLossModule(torch.nn.Module):
    """Expose the custom refinement objective through DDP's forward path."""

    def __init__(self, model: TwoPhaseDownscalingModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, batch: Mapping[str, Any], generator: torch.Generator
    ) -> torch.Tensor:
        return self.model.training_step(batch, generator=generator).losses["loss"]


class _SilentProgress:
    def update(self, _amount: int) -> None:
        pass

    def set_postfix(self, **_values: Any) -> None:
        pass

    def close(self) -> None:
        pass


class _DistributedRefinementTrainer(RefinementTrainer):
    """DDP refinement trainer with rank-zero progress and checkpoint output."""

    def __init__(
        self,
        *args,
        rank: int,
        world_size: int,
        global_train_steps: int,
        global_valid_steps: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.global_train_steps = int(global_train_steps)
        self.global_valid_steps = int(global_valid_steps)
        self._distributed_model = DistributedDataParallel(
            _RefinementLossModule(self.model),
            device_ids=[self.rank],
            output_device=self.rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    def _reduce_loss(self, total: float, count: int) -> float:
        statistics = torch.tensor(
            [total, float(count)], dtype=torch.float64, device=self.device
        )
        dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
        return float((statistics[0] / statistics[1].clamp_min(1.0)).item())

    def _optimizer_step(self) -> None:
        if self.max_grad_norm:
            if self.scaler is not None and self.scaler.is_enabled():
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in self.model.parameters() if parameter.requires_grad],
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
            factor = float(self.state.global_step + 1) / float(self.warmup_steps)
            for group, base_lr in zip(
                self.optimizer.param_groups, self._base_lrs, strict=True
            ):
                group["lr"] = base_lr * factor
        elif self.scheduler is not None:
            self.scheduler.step()

    def train_one_epoch(
        self,
        loader,
        limit_steps: int = 0,
        epoch: int = 0,
        progress_bar: Any | None = None,
    ) -> float:
        self.model.train()
        self._distributed_model.train()
        sampler = getattr(loader, "sampler", None)
        if callable(getattr(sampler, "set_epoch", None)):
            sampler.set_epoch(epoch)
        n = min(len(loader), limit_steps) if limit_steps else len(loader)
        bar = progress_bar or _SilentProgress()
        total, count = 0.0, 0
        self.optimizer.zero_grad(set_to_none=True)
        bar.set_postfix(stage="train")
        for step, batch in enumerate(loader):
            if step >= n:
                break
            batch = _to_device(batch, self.device)
            group_start = (step // self.accum) * self.accum
            group_size = min(self.accum, n - group_start)
            boundary = (step + 1) % self.accum == 0 or (step + 1) == n
            sync_context = nullcontext() if boundary else self._distributed_model.no_sync()
            with sync_context:
                with self._autocast():
                    loss = self._distributed_model(batch, self._generator)
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite refinement loss at rank={self.rank} step={step}: "
                        f"{loss.item()}"
                    )
                scaled = loss / max(1, group_size)
                if self.scaler is not None and self.scaler.is_enabled():
                    self.scaler.scale(scaled).backward()
                else:
                    scaled.backward()
            if boundary:
                self._optimizer_step()
            total += float(loss.detach())
            count += 1
            if self.rank == 0:
                completed_before = min(self.global_train_steps, step * self.world_size)
                bar.update(min(self.world_size, self.global_train_steps - completed_before))
                if count % self.log_every == 0:
                    bar.set_postfix(
                        stage="train",
                        train_loss=f"{total / count:.4f}",
                        lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    )
        return self._reduce_loss(total, count)

    @torch.no_grad()
    def validate(
        self,
        loader,
        limit_steps: int = 0,
        epoch: int = 0,
        progress_bar: Any | None = None,
    ) -> float:
        self.model.eval()
        self._distributed_model.eval()
        sampler = getattr(loader, "sampler", None)
        if callable(getattr(sampler, "set_epoch", None)):
            sampler.set_epoch(epoch)
        n = min(len(loader), limit_steps) if limit_steps else len(loader)
        bar = progress_bar or _SilentProgress()
        total, count = 0.0, 0
        bar.set_postfix(stage="validation")
        for step, batch in enumerate(loader):
            if step >= n:
                break
            batch = _to_device(batch, self.device)
            with self._autocast():
                loss = self._distributed_model(batch, self._generator)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite refinement validation loss at rank={self.rank} "
                    f"step={step}: {loss.item()}"
                )
            total += float(loss.detach())
            count += 1
            if self.rank == 0:
                completed_before = min(self.global_valid_steps, step * self.world_size)
                bar.update(min(self.world_size, self.global_valid_steps - completed_before))
        return self._reduce_loss(total, count)

    def fit(
        self,
        train_loader,
        val_loader=None,
        *,
        num_epochs: int = 1,
        limit_steps_train: int = 0,
        limit_steps_valid: int = 0,
        save_every: int = 1,
    ):
        final_epoch = self.state.epoch + num_epochs
        for epoch in range(self.state.epoch, final_epoch):
            if self.rank == 0:
                bar = tqdm(
                    total=self.global_train_steps + self.global_valid_steps,
                    desc=f"Epoch {epoch + 1:03d}/{final_epoch:03d}",
                    unit="batch",
                    ascii=" #",
                    ncols=100,
                    leave=True,
                    mininterval=1.0,
                )
            else:
                bar = _SilentProgress()
            try:
                train_loss = self.train_one_epoch(
                    train_loader,
                    limit_steps=limit_steps_train,
                    epoch=epoch,
                    progress_bar=bar,
                )
                self.state.train_loss_history.append(train_loss)
                val_loss = None
                if val_loader is not None:
                    bar.set_postfix(stage="validation", train_loss=f"{train_loss:.4f}")
                    val_loss = self.validate(
                        val_loader,
                        limit_steps=limit_steps_valid,
                        epoch=epoch,
                        progress_bar=bar,
                    )
                    self.state.val_loss_history.append(val_loss)
                final_postfix = {
                    "stage": "complete",
                    "train_loss": f"{train_loss:.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                }
                if val_loss is not None:
                    final_postfix["val_loss"] = f"{val_loss:.4f}"
                bar.set_postfix(**final_postfix)
            finally:
                bar.close()
            is_best = val_loss is not None and (
                self.state.best_val_loss is None
                or val_loss < self.state.best_val_loss
            )
            if is_best:
                self.state.best_val_loss = val_loss
            if self.rank == 0:
                val_text = "n/a" if val_loss is None else f"{val_loss:.4f}"
                best_text = (
                    "" if self.state.best_val_loss is None
                    else f"  best={self.state.best_val_loss:.4f}"
                )
                marker = "  *** new best ***" if is_best else ""
                print(
                    f"Ep {epoch + 1:03d}  train={train_loss:.4f}  "
                    f"val={val_text}{best_text}{marker}"
                )
            self.state.epoch += 1
            if self.rank == 0:
                epoch_file = save_every > 0 and self.state.epoch % save_every == 0
                self.save(is_best=is_best, epoch_file=epoch_file)
            dist.barrier()
        return self.state


def _available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _rank_limit(global_limit: int, world_size: int) -> int:
    return math.ceil(global_limit / world_size) if global_limit > 0 else 0


def _validation_rank_limit(global_limit: int, world_size: int, rank: int) -> int:
    """Distribute validation batches exactly; validation has no DDP backward."""
    if global_limit <= 0:
        return 0
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid validation rank/world size")
    quotient, remainder = divmod(global_limit, world_size)
    return quotient + int(rank < remainder)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def _run_train(args, *, rank: int = 0, world_size: int = 1) -> int:
    config = get_config(args.config)
    distributed = world_size > 1
    device = torch.device(f"cuda:{rank}" if distributed else args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    refinement = resolve_refinement_config(config)
    performance = resolve_performance_config(config)
    _configure_phase1_cache_runtime(performance, role="training", verbose=rank == 0)
    # This must precede model construction, DataLoader creation/iteration and
    # lazy refiner initialization. The notebook runs this command in a child
    # process, so its parent-kernel RNG state cannot provide reproducibility.
    _seed_training(refinement.seed + rank)
    if not refinement.is_active:
        raise SystemExit(
            "This configuration has no active refinement section. Use the "
            "deterministic trainer (narr_prism_finetune.py) instead."
        )
    phase1_checkpoint = _require_checkpoint_path(
        _phase1_checkpoint(config, args.phase1_checkpoint),
        label="Phase-1 deterministic",
    )
    model, fingerprint = build_model(
        config, args.config, phase1_checkpoint, device, verbose=rank == 0
    )
    raw_config = load_yaml(args.config)
    raw_config["_config_path"] = str(Path(args.config).resolve())

    cache_reader, cache_manifest = _load_phase1_cache(
        config=config,
        raw_config=raw_config,
        phase1_checkpoint=phase1_checkpoint,
        phase1_fingerprint=fingerprint,
        performance=performance,
        validate_inventory=(rank == 0),
        show_inventory_progress=(rank == 0),
        verbose=(rank == 0),
    )
    train_loader, val_loader = get_dataloaders(
        args.config, config, rank=rank, world_size=world_size,
        phase1_cache_reader=cache_reader,
        log_alignment=False,
    )
    probe = _to_device(_first_batch(train_loader), device)
    if distributed:
        dist.barrier()
    if rank == 0:
        base_dataset = getattr(train_loader.dataset, "base", train_loader.dataset)
        source = (
            "strict preprocessed product"
            if base_dataset.use_preprocessed
            else "on-the-fly coarse-to-fine regridding"
        )
        print(
            f"[dataset] predictor->PRISM alignment OK: {source} is on the "
            f"canonical target grid {tuple(base_dataset.fine_shape)} before tiling."
        )
    model.initialize_from_batch(probe)
    model.to(device)
    residual_norm = model.refinement_config.residual_normalization
    cache_statistics = None
    if cache_manifest is not None:
        cache_statistics = _validated_cache_statistics(model, cache_manifest)
        if (
            performance.phase1_cache.validate_cache and rank == 0
        ):
            _validate_phase1_cache_parity(
                model,
                train_loader,
                device=device,
                samples=performance.phase1_cache.validate_samples,
                tolerance=performance.phase1_cache.validate_tolerance,
            )
        if not args.resume:
            _install_cache_statistics(model, cache_statistics)
    elif residual_norm.enabled and not args.resume:
        if distributed:
            raise RuntimeError(
                "Multi-GPU refinement requires the authenticated Phase-1 cache "
                "when residual normalization must be fitted. Build the cache first."
            )
        fit_limit = int(residual_norm.fit_batches)
        print(
            "[refinement] fitting ordered per-variable residual normalization "
            f"on the Phase-2 training loader (max_batches={fit_limit or 'all'})"
        )
        metadata = model.fit_residual_normalizer(
            train_loader,
            device=device,
            max_batches=fit_limit or None,
            show_progress=(rank == 0),
        )
        if rank == 0:
            print(f"[refinement] residual normalization: {json.dumps(metadata, indent=2)}")
    if rank == 0:
        print(f"[refinement] {json.dumps(model.describe()['refinement'], indent=2)}")
        summary = model.describe()
        print(
            "[refinement] checkpoint/model summary: "
            f"deterministic_loaded_fingerprint={fingerprint} "
            f"new_refinement_keys={len(summary['new_refinement_keys'])} "
            f"frozen_parameters={summary['frozen_parameters']:,} "
            f"trainable_parameters={summary['trainable_parameters']:,}"
        )

    trainable = model.trainable_parameters()
    learning_rate = float(getattr(config, "learning_rate", 1e-4))
    min_learning_rate = float(getattr(config, "min_lr", 1e-6))
    warmup_steps = int(getattr(config, "warm_up_steps", 0) or 0)
    max_grad_norm = float(getattr(config, "max_grad_norm", 0.0)) or None
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    epochs = int(args.num_epochs or getattr(config, "num_epochs", 1))
    limit_steps_train = int(
        args.limit_steps or getattr(config, "limit_steps_train", 0) or 0
    )
    limit_steps_valid = int(getattr(config, "limit_steps_valid", 0) or 0)
    available_global_train_steps = len(train_loader) * world_size
    available_global_valid_steps = len(val_loader) * world_size
    steps = max(1, min(
        available_global_train_steps,
        limit_steps_train or available_global_train_steps,
    ))
    valid_steps = max(1, min(
        available_global_valid_steps,
        limit_steps_valid or available_global_valid_steps,
    ))
    accumulation = int(getattr(config, "gradient_accumulation_steps", 1))
    if distributed and accumulation % world_size:
        raise ValueError(
            "gradient_accumulation_steps must be divisible by the selected GPU "
            f"count to preserve the configured global batch ({accumulation} % "
            f"{world_size} != 0)"
        )
    local_accumulation = max(1, accumulation // world_size)
    local_limit_steps_train = _rank_limit(limit_steps_train, world_size)
    local_limit_steps_valid = _validation_rank_limit(
        limit_steps_valid, world_size, rank
    )
    optimizer_steps = max(1, math.ceil(steps / max(1, accumulation)))
    scheduler_t_max = max(1, epochs * optimizer_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=scheduler_t_max,
        eta_min=min_learning_rate,
    )
    scaler = torch.amp.GradScaler(
        device.type, enabled=(device.type == "cuda" and model.performance_config.precision.mode != "fp32")
    )

    checkpoint_dir = args.checkpoint_dir or getattr(config, "checkpoint_dir", None) or os.path.join(
        str(getattr(config, "path_experiment", "./")), "refinement_checkpoints"
    )
    checkpoint_dir = _resolve(checkpoint_dir)

    trainer_class = _DistributedRefinementTrainer if distributed else RefinementTrainer
    distributed_trainer_kwargs = (
        {
            "rank": rank,
            "world_size": world_size,
            "global_train_steps": steps,
            "global_valid_steps": valid_steps,
        }
        if distributed else {}
    )
    trainer = trainer_class(
        model,
        optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        checkpoint_dir=checkpoint_dir,
        phase1_checkpoint=phase1_checkpoint,
        resolved_config={
            "refinement": model.refinement_config.to_dict(),
            "performance": model.performance_config.to_dict(),
            "training": {
                "contract_version": 1,
                "optimizer": "torch.optim.AdamW",
                "base_learning_rate": learning_rate,
                "min_learning_rate": min_learning_rate,
                "scheduler": "torch.optim.lr_scheduler.CosineAnnealingLR",
                "scheduler_t_max": scheduler_t_max,
                "warmup_steps": warmup_steps,
                "per_device_batch_size": int(getattr(config, "batch_size", 1)),
                "gradient_accumulation_steps": accumulation,
                "max_grad_norm": max_grad_norm,
                "epochs": epochs,
                "limit_steps_train": limit_steps_train,
                "limit_steps_valid": limit_steps_valid,
                "effective_train_batches_per_epoch": steps,
                "optimizer_steps_per_epoch": optimizer_steps,
                "total_optimizer_steps": epochs * optimizer_steps,
                "phase1_cache_contract_digest": (
                    cache_manifest["contract_digest"]
                    if cache_manifest is not None
                    else None
                ),
            },
            "case_name": get_case_name(config),
            "config_path": os.path.abspath(args.config),
        },
        case_name=get_case_name(config),
        gradient_accumulation_steps=local_accumulation,
        max_grad_norm=max_grad_norm,
        seed=model.refinement_config.seed,
        warmup_steps=warmup_steps,
        logger=print if rank == 0 else lambda _message: None,
        **distributed_trainer_kwargs,
    )
    if fingerprint:
        trainer._phase1_fingerprint = fingerprint

    if args.resume:
        resume_path = args.resume if isinstance(args.resume, str) else os.path.join(checkpoint_dir, "last.ckpt")
        previous_batch = int(getattr(args, "resume_batch_size_from", 0) or 0)
        previous_epochs = int(getattr(args, "resume_epochs_from", 0) or 0)
        if previous_batch:
            _resume_with_batch_size_migration(
                trainer,
                resume_path,
                previous_per_device_batch_size=previous_batch,
                current_per_device_batch_size=int(getattr(config, "batch_size", 1)),
                announce=rank == 0,
                previous_num_epochs=previous_epochs,
                current_num_epochs=epochs,
            )
        else:
            trainer.resume(resume_path)
        if cache_statistics is not None:
            _assert_cache_statistics_match_resume(
                model,
                cache_statistics,
                tolerance=performance.phase1_cache.validate_tolerance,
            )
    if distributed and rank != 0:
        trainer._generator.manual_seed(
            model.refinement_config.seed + rank + trainer.state.global_step
        )


    # ``epochs`` is the total run budget, not a number of extra epochs.  An
    # interrupted run therefore completes only the remaining epochs after the
    # scheduler/optimizer state has been restored.
    epochs_remaining = max(0, epochs - trainer.state.epoch)
    if args.resume and rank == 0:
        print(
            "[refinement] resume epoch budget: "
            f"completed={trainer.state.epoch} total={epochs} "
            f"remaining={epochs_remaining}"
        )
    state = trainer.fit(
        train_loader,
        val_loader,
        num_epochs=epochs_remaining,
        limit_steps_train=local_limit_steps_train,
        limit_steps_valid=local_limit_steps_valid,
        save_every=int(args.save_every),
    )
    if rank == 0:
        curve_path = _save_epochs_vs_loss_curve(state, checkpoint_dir)
        if curve_path is not None:
            print(f"[refinement] epochs-versus-loss curve -> {curve_path}")
    return 0


def _distributed_train_entry(
    rank: int, world_size: int, args: argparse.Namespace, port: int
) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        _run_train(args, rank=rank, world_size=world_size)
    finally:
        dist.destroy_process_group()


def cmd_train(args) -> int:
    requested = int(getattr(args, "num_gpus", 1) or 1)
    if requested < 1:
        raise ValueError("--num-gpus must be at least 1")
    if requested == 1:
        return _run_train(args)
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("Multi-GPU refinement requires CUDA and --device cuda:0")
    visible = torch.cuda.device_count()
    if requested > visible:
        raise RuntimeError(
            f"Requested {requested} GPUs, but only {visible} are visible. "
            "Check CUDA_VISIBLE_DEVICES."
        )
    print(
        f"[refinement] distributed training: GPUs={requested} "
        f"visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}"
    )
    mp.spawn(
        _distributed_train_entry,
        args=(requested, args, _available_port()),
        nprocs=requested,
        join=True,
    )
    return 0


def _infer_output_root(cfg: Mapping[str, Any], refinement_type: str, override: str | None) -> str:
    if override:
        return str(Path(override).expanduser().resolve())
    base = cfg.get("inference", {}).get(
        "refinement_output_dir",
        cfg.get("inference", {}).get("output_dir", "./refinement_inference_output"),
    )
    return str(Path(_resolve(base)) / f"refinement_{refinement_type}")


def _tail_worker_log(path: Path, lines: int = 30) -> str:
    try:
        return "".join(path.read_text(errors="replace").splitlines(True)[-lines:])
    except OSError as exc:
        return f"<unable to read {path}: {exc}>"


def _run_parallel_infer(args, requested: int) -> int:
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("Multi-GPU refinement inference requires CUDA")
    visible = torch.cuda.device_count()
    if requested > visible:
        raise RuntimeError(
            f"Requested {requested} GPUs, but only {visible} are visible"
        )

    config = get_config(args.config)
    refinement = resolve_refinement_config(config)
    if not refinement.is_active:
        raise SystemExit("infer requires an active model.refinement configuration")
    phase1_checkpoint = _require_checkpoint_path(
        _phase1_checkpoint(config, args.phase1_checkpoint),
        label="Phase-1 deterministic",
    )
    refinement_checkpoint = _require_checkpoint_path(
        _refinement_checkpoint(config, args.refinement_checkpoint),
        label="Phase-2 refinement",
    )
    cfg = load_yaml(args.config)
    output = _infer_output_root(cfg, refinement.type, args.output)
    from narr_prism_inference import _split_output_root
    from narr_prism_refinement_inference import _write_progress_state
    from narr_prism_utils import case_output_dir, parse_date_range_from_config

    output_path = case_output_dir(
        _split_output_root(output, args.split), get_case_name(cfg)
    )
    output_path.mkdir(parents=True, exist_ok=True)
    start, end = parse_date_range_from_config(cfg, args.split)
    total_dates = (end - start).days + 1
    if args.limit_days:
        total_dates = min(total_dates, max(0, int(args.limit_days)))
    if total_dates < requested:
        raise ValueError(
            f"Refinement inference has {total_dates} dates for {requested} GPUs"
        )

    configured_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    physical_ids = (
        [item.strip() for item in configured_visible.split(",") if item.strip()]
        if configured_visible
        else [str(index) for index in range(visible)]
    )[:requested]
    pattern = f"{get_case_name(cfg)}_{refinement.type}_refined_*.nc"
    initial_done = min(len(list(output_path.glob(pattern))), total_dates)
    print(
        f"[refinement] parallel inference: GPUs={requested} "
        f"resume_files={initial_done}/{total_dates} -> {output_path}",
        flush=True,
    )

    processes = []
    logs = []
    handles = []
    progress_paths = []
    progress_token = f"{os.getpid()}_{time.time_ns()}"
    try:
        for shard_index, physical_id in enumerate(physical_ids):
            log_path = output_path / f"parallel_worker_{shard_index}_gpu{physical_id}.log"
            progress_path = output_path / (
                f".parallel_progress_{progress_token}_{shard_index}.json"
            )
            progress_paths.append(progress_path)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "infer",
                "--config", args.config,
                "--phase1-checkpoint", phase1_checkpoint,
                "--refinement-checkpoint", refinement_checkpoint,
                "--device", "cuda:0",
                "--num-gpus", "1",
                "--date-shard-index", str(shard_index),
                "--date-shard-count", str(requested),
                "--batch-size", str(args.batch_size),
                "--output", output,
                "--split", args.split,
                "--no-progress",
                "--progress-file", str(progress_path),
            ]
            if args.resume_existing:
                command.append("--resume-existing")
            if args.ensemble_size is not None:
                command.extend(["--ensemble-size", str(args.ensemble_size)])
            if args.seed is not None:
                command.extend(["--seed", str(args.seed)])
            if args.limit_days:
                command.extend(["--limit-days", str(args.limit_days)])
            if getattr(args, "phase1_cache_dir", None):
                command.extend(["--phase1-cache-dir", args.phase1_cache_dir])
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = physical_id
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            handle = open(log_path, "w", encoding="utf-8")
            handles.append(handle)
            logs.append(log_path)
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=str(REPO_ROOT),
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )

        progress = tqdm(
            total=None,
            desc="Refined inference (starting workers)",
            unit="tile-batch",
            dynamic_ncols=True,
            # Notebook subprocess streams cannot redraw a terminal bar.
            disable=bool(args.no_progress) or not sys.stderr.isatty(),
        )
        try:
            while True:
                states = []
                for progress_path in progress_paths:
                    try:
                        states.append(json.loads(progress_path.read_text()))
                    except (FileNotFoundError, json.JSONDecodeError, OSError):
                        pass
                if len(states) == requested:
                    progress.total = sum(int(state["total"]) for state in states)
                    progress.n = sum(int(state["completed"]) for state in states)
                    _write_progress_state(
                        args.progress_file,
                        completed=progress.n,
                        total=progress.total,
                    )
                    progress.set_description("Refined inference", refresh=False)
                progress.set_postfix(
                    live_workers=sum(proc.poll() is None for proc in processes),
                    resumed_days=initial_done,
                    refresh=True,
                )
                failures = [
                    (index, proc.returncode)
                    for index, proc in enumerate(processes)
                    if proc.poll() not in (None, 0)
                ]
                if failures or all(proc.poll() is not None for proc in processes):
                    break
                time.sleep(2.0)
        finally:
            progress.close()

        if failures:
            for handle in handles:
                handle.flush()
            raise RuntimeError(
                "Parallel refinement inference worker failure(s): "
                + ", ".join(
                    f"worker {index} rc={code} log={logs[index]}"
                    for index, code in failures
                )
                + "\n\n"
                + "\n\n".join(
                    f"--- worker {index} tail ---\n{_tail_worker_log(logs[index])}"
                    for index, _code in failures
                )
            )
    except BaseException:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for proc in processes:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        raise
    finally:
        for handle in handles:
            handle.close()
        for progress_path in progress_paths:
            progress_path.unlink(missing_ok=True)

    print(f"[refinement] all inference workers finished -> {output_path}")
    return 0


def cmd_infer(args) -> int:
    requested = max(1, int(args.num_gpus))
    if requested > 1:
        if int(args.date_shard_count) != 1 or int(args.date_shard_index) != 0:
            raise ValueError("Parent multi-GPU inference cannot also be a date shard")
        return _run_parallel_infer(args, requested)

    config = get_config(args.config)
    device = torch.device(args.device)
    refinement = resolve_refinement_config(config)
    performance = resolve_performance_config(config)
    _configure_phase1_cache_runtime(performance, role="inference")
    if not refinement.is_active:
        raise SystemExit("infer requires an active model.refinement configuration")
    phase1_checkpoint = _require_checkpoint_path(
        _phase1_checkpoint(config, args.phase1_checkpoint),
        label="Phase-1 deterministic",
    )
    refinement_checkpoint = _require_checkpoint_path(
        _refinement_checkpoint(config, args.refinement_checkpoint),
        label="Phase-2 refinement",
    )
    model, phase1_fingerprint = build_model(
        config, args.config, phase1_checkpoint, device
    )

    cfg = load_yaml(args.config)
    output = _infer_output_root(cfg, refinement.type, args.output)

    from narr_prism_refinement_inference import run_refined_inference

    result = run_refined_inference(
        config_path=args.config,
        cfg=cfg,
        config=config,
        model=model,
        phase1_checkpoint=phase1_checkpoint,
        phase1_fingerprint=phase1_fingerprint,
        refinement_checkpoint=refinement_checkpoint,
        output_dir=output,
        device=device,
        ensemble_size=(
            args.ensemble_size
            if args.ensemble_size is not None
            else refinement.ensemble_size
        ),
        base_seed=args.seed if args.seed is not None else refinement.seed,
        batch_size=max(1, int(args.batch_size)),
        limit_days=max(0, int(args.limit_days or 0)),
        split=args.split,
        date_shard_index=int(args.date_shard_index),
        date_shard_count=int(args.date_shard_count),
        resume_existing=bool(args.resume_existing),
        show_progress=not bool(args.no_progress),
        progress_file=args.progress_file,
        phase1_cache_dir=(
            _resolve(args.phase1_cache_dir)
            if getattr(args, "phase1_cache_dir", None)
            else _resolve(cfg.get("inference", {}).get(
                "phase1_cache_dir",
                "./examples/NARR_PRISM/experiments/phase1_inference_cache",
            ))
        ),
    )
    print(f"[refinement] daily refined outputs saved -> {result}")
    return 0


def cmd_describe(args) -> int:
    config = get_config(args.config)
    refinement = resolve_refinement_config(config)
    performance = resolve_performance_config(config)
    print(json.dumps({"refinement": refinement.to_dict(), "performance": performance.to_dict()}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", required=True)
    common.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    common.add_argument("--phase1-checkpoint", default=None)

    train = sub.add_parser("train", parents=[common], help="train the configured Phase-2 refiner")
    train.add_argument("--resume", nargs="?", const=True, default=False)
    train.add_argument(
        "--resume-batch-size-from",
        type=int,
        default=0,
        help="explicit prior per-device batch for a budget-equivalent resume migration",
    )
    train.add_argument(
        "--resume-epochs-from",
        type=int,
        default=0,
        help="explicit prior epoch budget for a validated schedule extension",
    )
    train.add_argument("--num-epochs", type=int, default=None)
    train.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="number of visible CUDA devices to use with DDP",
    )
    train.add_argument("--limit-steps", type=int, default=0)
    train.add_argument("--save-every", type=int, default=1)
    train.add_argument("--checkpoint-dir", default=None, help="override the YAML checkpoint_dir")
    train.set_defaults(func=cmd_train)

    infer = sub.add_parser("infer", parents=[common], help="deterministic + ensemble inference")
    infer.add_argument("--refinement-checkpoint", default=None)
    infer.add_argument("--ensemble-size", type=int, default=None)
    infer.add_argument("--seed", type=int, default=None)
    infer.add_argument(
        "--limit-days",
        "--limit-batches",
        dest="limit_days",
        type=int,
        default=0,
        help="optional number of leading inference dates (legacy alias: --limit-batches)",
    )
    infer.add_argument("--batch-size", type=int, default=1, help="tile batch size")
    infer.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="independent date-shard workers on visible CUDA devices",
    )
    infer.add_argument(
        "--resume-existing",
        action="store_true",
        help="validate and skip existing daily products; larger ensembles are reduced atomically",
    )
    infer.add_argument("--date-shard-index", type=int, default=0, help=argparse.SUPPRESS)
    infer.add_argument("--date-shard-count", type=int, default=1, help=argparse.SUPPRESS)
    infer.add_argument(
        "--no-progress", action="store_true",
        help="disable inference progress bars, including the multi-GPU parent",
    )
    infer.add_argument("--progress-file", default=None, help=argparse.SUPPRESS)
    infer.add_argument("--output", default=None, help="daily NetCDF output root directory")
    infer.add_argument(
        "--phase1-cache-dir", default=None,
        help=(
            "shared FP32 Phase-1 tile cache; missing days are built with "
            "autocast/TF32 disabled, validated and reused across refinement heads"
        ),
    )
    infer.add_argument(
        "--split",
        choices=("validation", "inference"),
        default="inference",
        help=(
            "YAML date split to predict; validation outputs are isolated in a "
            "validation subdirectory"
        ),
    )
    infer.set_defaults(func=cmd_infer)

    describe = sub.add_parser("describe", parents=[common], help="print the resolved configuration")
    describe.set_defaults(func=cmd_describe)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
