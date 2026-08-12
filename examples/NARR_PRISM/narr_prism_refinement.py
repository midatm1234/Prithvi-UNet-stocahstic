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
import json
import math
import os
import sys
from pathlib import Path

import torch

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


def build_model(config, config_path: str, phase1_checkpoint: str | None, device: torch.device):
    """Build the two-phase model and load the deterministic Phase-1 weights."""
    log_case_context(config, "refinement")
    assert_scalars_available(config, role="refinement")
    apply_scalar_paths(config)

    refinement = resolve_refinement_config(config)
    performance = resolve_performance_config(config)
    phase1 = create_finetune_model(config, verbose=True)
    model = TwoPhaseDownscalingModel(phase1, refinement=refinement, performance=performance)

    if refinement.is_active:
        phase1_checkpoint = _require_checkpoint_path(
            phase1_checkpoint, label="Phase-1 deterministic"
        )

    fingerprint = None
    if phase1_checkpoint:
        print(f"[refinement] loading Phase-1 checkpoint (read-only): {phase1_checkpoint}")
        checkpoint = torch.load(phase1_checkpoint, map_location="cpu", mmap=True, weights_only=False)
        # Validate channel ordering, dates, grid/scaler hashes, transforms and
        # architecture semantics before a single checkpoint tensor is applied.
        validate_prism_checkpoint_contract(
            config, checkpoint, role="NARR refinement Phase-1 load"
        )
        report = load_phase1_state_dict(model, checkpoint)
        print(f"[refinement] Phase-1 load: {report.summary()}")
        unexplained = [k for k in report.missing if not k.startswith("refiner.")]
        if unexplained:
            raise RuntimeError(f"Unexplained missing Phase-1 keys: {unexplained[:8]}")
        fingerprint = phase1_state_fingerprint(
            {k[len("phase1.") :]: v for k, v in model.state_dict().items() if k.startswith("phase1.")}
        )
        print(f"[refinement] Phase-1 fingerprint: {fingerprint}")

    model.to(device)
    return model, fingerprint


def _first_batch(loader):
    for batch in loader:
        return batch
    raise RuntimeError("The dataloader produced no batches.")


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_train(args) -> int:
    config = get_config(args.config)
    device = torch.device(args.device)
    if not resolve_refinement_config(config).is_active:
        raise SystemExit(
            "This configuration has no active refinement section. Use the "
            "deterministic trainer (narr_prism_finetune.py) instead."
        )
    phase1_checkpoint = _require_checkpoint_path(
        _phase1_checkpoint(config, args.phase1_checkpoint),
        label="Phase-1 deterministic",
    )
    model, fingerprint = build_model(config, args.config, phase1_checkpoint, device)

    train_loader, val_loader = get_dataloaders(args.config, config)
    probe = _to_device(_first_batch(train_loader), device)
    model.initialize_from_batch(probe)
    model.to(device)
    residual_norm = model.refinement_config.residual_normalization
    if residual_norm.enabled and not args.resume:
        fit_limit = int(residual_norm.fit_batches)
        print(
            "[refinement] fitting ordered per-variable residual normalization "
            f"on the Phase-2 training loader (max_batches={fit_limit or 'all'})"
        )
        metadata = model.fit_residual_normalizer(
            train_loader,
            device=device,
            max_batches=fit_limit or None,
        )
        print(f"[refinement] residual normalization: {json.dumps(metadata, indent=2)}")
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
    steps = max(
        1,
        min(
            len(train_loader),
            limit_steps_train or len(train_loader),
        ),
    )
    accumulation = int(getattr(config, "gradient_accumulation_steps", 1))
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

    trainer = RefinementTrainer(
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
                "gradient_accumulation_steps": accumulation,
                "max_grad_norm": max_grad_norm,
                "epochs": epochs,
                "limit_steps_train": limit_steps_train,
                "limit_steps_valid": limit_steps_valid,
                "effective_train_batches_per_epoch": steps,
                "optimizer_steps_per_epoch": optimizer_steps,
                "total_optimizer_steps": epochs * optimizer_steps,
            },
            "case_name": get_case_name(config),
            "config_path": os.path.abspath(args.config),
        },
        case_name=get_case_name(config),
        gradient_accumulation_steps=accumulation,
        max_grad_norm=max_grad_norm,
        seed=model.refinement_config.seed,
        warmup_steps=warmup_steps,
    )
    if fingerprint:
        trainer._phase1_fingerprint = fingerprint

    if args.resume:
        resume_path = args.resume if isinstance(args.resume, str) else os.path.join(checkpoint_dir, "last.ckpt")
        trainer.resume(resume_path)

    # ``epochs`` is the total run budget, not a number of extra epochs.  An
    # interrupted run therefore completes only the remaining epochs after the
    # scheduler/optimizer state has been restored.
    epochs_remaining = max(0, epochs - trainer.state.epoch)
    if args.resume:
        print(
            "[refinement] resume epoch budget: "
            f"completed={trainer.state.epoch} total={epochs} "
            f"remaining={epochs_remaining}"
        )
    trainer.fit(
        train_loader,
        val_loader,
        num_epochs=epochs_remaining,
        limit_steps_train=limit_steps_train,
        limit_steps_valid=limit_steps_valid,
        save_every=int(args.save_every),
    )
    return 0


def cmd_infer(args) -> int:
    config = get_config(args.config)
    device = torch.device(args.device)
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
    model, phase1_fingerprint = build_model(
        config, args.config, phase1_checkpoint, device
    )

    cfg = load_yaml(args.config)
    output = args.output
    if not output:
        base = cfg.get("inference", {}).get(
            "refinement_output_dir",
            cfg.get("inference", {}).get("output_dir", "./refinement_inference_output"),
        )
        output = str(Path(_resolve(base)) / f"refinement_{refinement.type}")

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
    train.add_argument("--num-epochs", type=int, default=None)
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
    infer.add_argument("--output", default=None, help="daily NetCDF output root directory")
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
