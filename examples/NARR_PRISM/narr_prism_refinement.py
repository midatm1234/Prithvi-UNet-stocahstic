"""Phase-2 (stochastic refinement) entry point for the NARR-to-PRISM workflow.

Subcommands
-----------
``train``
    Load an existing deterministic Phase-1 checkpoint, freeze it, and train the
    configured Phase-2 refiner on residuals. Resumable.

``infer``
    Run deterministic and refined ensemble inference and write a NetCDF file
    containing the deterministic prediction, the predicted residual, every
    ensemble member, the ensemble mean and the ensemble spread.

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
        --ensemble-size 10 --output /tmp/narr_refined.nc
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from granitewxc.refinement import TwoPhaseDownscalingModel  # noqa: E402
from granitewxc.refinement.checkpoint import (  # noqa: E402
    CHECKPOINT_SCHEMA_VERSION,
    REFINEMENT_CONTRACT_VERSION,
    load_phase1_state_dict,
    load_refinement_state_dict,
    phase1_state_fingerprint,
    validate_phase1_reference,
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
from narr_prism_inference import VAR_UNITS  # noqa: E402
from narr_prism_training import (  # noqa: E402
    create_finetune_model,
    get_dataloaders,
    get_inference_dataloader,
)
from narr_prism_utils import get_case_name  # noqa: E402


def _resolve(path: str | os.PathLike) -> str:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    return str((REPO_ROOT / candidate).resolve())


def _sha256_file(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _phase1_checkpoint(config, override: str | None) -> str | None:
    if override:
        return _resolve(override)
    phase1_cfg = getattr(config.model, "phase1", None) or {}
    if isinstance(phase1_cfg, dict):
        path = phase1_cfg.get("checkpoint")
    else:
        path = getattr(phase1_cfg, "checkpoint", None)
    return _resolve(path) if path else None


def _refinement_checkpoint(config, config_path: str, override: str | None) -> str:
    if override:
        candidate = Path(override).expanduser()
    else:
        refinement_cfg = getattr(config.model, "refinement", None) or {}
        explicit = (
            refinement_cfg.get("checkpoint")
            if isinstance(refinement_cfg, dict)
            else getattr(refinement_cfg, "checkpoint", None)
        )
        if explicit:
            candidate = Path(explicit).expanduser()
        else:
            checkpoint_dir = getattr(config, "checkpoint_dir", None)
            if not checkpoint_dir:
                raise RuntimeError(
                    "Active inference requires --refinement-checkpoint, "
                    "model.refinement.checkpoint, or checkpoint_dir."
                )
            candidate = Path(checkpoint_dir).expanduser() / "best.ckpt"
    if candidate.is_absolute():
        candidates = [candidate.resolve()]
    else:
        # NARR YAMLs historically use both repository-relative paths
        # (``./examples/NARR_PRISM/...``) and paths relative to the YAML file.
        # Search both contracts explicitly and reject ambiguity rather than
        # silently choosing a different checkpoint based on the process cwd.
        candidates = []
        for root in (REPO_ROOT, Path(config_path).expanduser().resolve().parent):
            resolved = (root / candidate).resolve()
            if resolved not in candidates:
                candidates.append(resolved)
    existing = [path for path in candidates if path.is_file()]
    if len(existing) > 1:
        raise RuntimeError(
            "Ambiguous relative refinement checkpoint; both repository- and "
            f"YAML-relative candidates exist: {existing}"
        )
    if not existing:
        attempted = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"Refinement checkpoint not found; tried: {attempted}"
        )
    return str(existing[0])


def build_model(config, config_path: str, phase1_checkpoint: str | None, device: torch.device):
    """Build the two-phase model and load the deterministic Phase-1 weights."""
    log_case_context(config, "refinement")
    assert_scalars_available(config, role="refinement")
    apply_scalar_paths(config)

    refinement = resolve_refinement_config(config)
    performance = resolve_performance_config(config)
    phase1 = create_finetune_model(config, verbose=True)
    model = TwoPhaseDownscalingModel(phase1, refinement=refinement, performance=performance)

    fingerprint = None
    if phase1_checkpoint:
        print(f"[refinement] loading Phase-1 checkpoint (read-only): {phase1_checkpoint}")
        checkpoint = torch.load(phase1_checkpoint, map_location="cpu", mmap=True, weights_only=False)
        report = load_phase1_state_dict(model, checkpoint)
        print(f"[refinement] Phase-1 load: {report.summary()}")
        unexplained = [
            key
            for key in report.missing
            if not key.startswith(("refiner.", "residual_normalizer."))
        ]
        if unexplained:
            raise RuntimeError(f"Unexplained missing Phase-1 keys: {unexplained[:8]}")
        fingerprint = phase1_state_fingerprint(
            {k[len("phase1.") :]: v for k, v in model.state_dict().items() if k.startswith("phase1.")}
        )
        print(f"[refinement] Phase-1 fingerprint: {fingerprint}")
    else:
        raise RuntimeError(
            "Phase-1 checkpoint is required for refinement training and inference; "
            "randomly initialized deterministic conditioning is forbidden."
        )

    model.to(device)
    return model, fingerprint


def _first_batch(loader):
    for batch in loader:
        return batch
    raise RuntimeError("The dataloader produced no batches.")


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _canonical_inference_dataset(loader):
    """Return and validate the full-domain dataset wrapped by a loader."""
    dataset = loader.dataset
    while hasattr(dataset, "base"):
        dataset = dataset.base
    required = ("fine_lat", "fine_lon", "fine_shape", "crop_size", "mode")
    missing = [name for name in required if not hasattr(dataset, name)]
    if missing:
        raise RuntimeError(
            f"Inference loader does not expose canonical dataset metadata: {missing}"
        )
    if str(dataset.mode) != "inference":
        raise RuntimeError(
            f"Refinement inference requires dates.inference, got mode={dataset.mode!r}."
        )
    if getattr(dataset, "_tile_slices", None) is not None:
        raise RuntimeError(
            "Refinement inference received a tiled dataset; tiles must not be "
            "serialized as independent time records."
        )
    fine_shape = tuple(int(value) for value in dataset.fine_shape)
    crop_size = tuple(int(value) for value in dataset.crop_size)
    if crop_size != fine_shape or not bool(getattr(dataset, "full_domain", False)):
        raise RuntimeError(
            "Refinement inference dataset is not full-domain: "
            f"crop={crop_size}, canonical={fine_shape}."
        )
    return dataset


def _batch_date_strings(batch, expected_batch_size: int) -> list[str]:
    raw = batch.get("date")
    if raw is None:
        raise RuntimeError(
            "Inference batch is missing source dates; refusing to fabricate a time axis."
        )
    values = [raw] if isinstance(raw, str) else list(raw)
    dates = [str(value) for value in values]
    if len(dates) != int(expected_batch_size):
        raise RuntimeError(
            f"Inference batch has {len(dates)} dates for batch size {expected_batch_size}."
        )
    return dates


def _physical_output_units(variables) -> dict[str, str]:
    """Return the established physical-unit contract for NARR predictands."""
    missing = [
        str(name)
        for name in variables
        if not str(VAR_UNITS.get(str(name), "")).strip()
    ]
    if missing:
        raise RuntimeError(
            "No physical output-unit contract is defined for NARR variable(s): "
            f"{missing}. Add explicit units before writing refinement output."
        )
    return {str(name): str(VAR_UNITS[str(name)]) for name in variables}


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_train(args) -> int:
    config = get_config(args.config)
    device = torch.device(args.device)
    phase1_checkpoint = _phase1_checkpoint(config, args.phase1_checkpoint)
    model, fingerprint = build_model(config, args.config, phase1_checkpoint, device)

    if not model.refinement_config.is_active:
        raise SystemExit(
            "This configuration has no active refinement section. Use the "
            "deterministic trainer (narr_prism_finetune.py) instead."
        )

    train_loader, val_loader = get_dataloaders(args.config, config)
    probe = _to_device(_first_batch(train_loader), device)
    model.initialize_from_batch(probe)
    model.to(device)
    print(f"[refinement] {json.dumps(model.describe()['refinement'], indent=2)}")
    print(f"[refinement] refiner parameters: {model.describe()['refiner_parameters']:,}")

    trainable = model.trainable_parameters()
    optimizer = torch.optim.AdamW(trainable, lr=float(getattr(config, "learning_rate", 1e-4)))
    epochs = int(args.num_epochs or getattr(config, "num_epochs", 1))
    steps = max(1, min(len(train_loader), int(getattr(config, "limit_steps_train", 0) or len(train_loader))))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs * steps), eta_min=float(getattr(config, "min_lr", 1e-6))
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
            "case_name": get_case_name(config),
            "config_path": os.path.abspath(args.config),
        },
        case_name=get_case_name(config),
        gradient_accumulation_steps=int(getattr(config, "gradient_accumulation_steps", 1)),
        max_grad_norm=float(getattr(config, "max_grad_norm", 0.0)) or None,
        seed=model.refinement_config.seed,
    )
    if fingerprint:
        trainer._phase1_fingerprint = fingerprint

    if args.resume:
        resume_path = args.resume if isinstance(args.resume, str) else os.path.join(checkpoint_dir, "last.ckpt")
        trainer.resume(resume_path)

    trainer.fit(
        train_loader,
        val_loader,
        num_epochs=epochs,
        limit_steps_train=int(args.limit_steps or getattr(config, "limit_steps_train", 0) or 0),
        limit_steps_valid=int(getattr(config, "limit_steps_valid", 0) or 0),
        save_every=int(args.save_every),
    )
    return 0


def cmd_infer(args) -> int:
    config = get_config(args.config)
    device = torch.device(args.device)
    phase1_checkpoint = _phase1_checkpoint(config, args.phase1_checkpoint)
    model, loaded_phase1_fingerprint = build_model(
        config, args.config, phase1_checkpoint, device
    )

    inference_loader = get_inference_dataloader(args.config, config)
    inference_dataset = _canonical_inference_dataset(inference_loader)
    probe = _to_device(_first_batch(inference_loader), device)
    model.initialize_from_batch(probe)
    model.to(device).eval()

    refinement_checkpoint = _refinement_checkpoint(
        config, args.config, args.refinement_checkpoint
    )
    payload = torch.load(refinement_checkpoint, map_location="cpu", weights_only=False)
    phase1_state = {
        k[len("phase1.") :]: v for k, v in model.state_dict().items() if k.startswith("phase1.")
    }
    validate_phase1_reference(payload, phase1_state, strict=True)
    load_refinement_state_dict(model, payload)
    model.eval()
    if any(module.training for module in model.refiner.modules()):
        raise RuntimeError("Refinement modules must be in eval mode during inference.")
    print(f"[refinement] loaded Phase-2 weights from {refinement_checkpoint}")

    ensemble_size = int(
        args.ensemble_size if args.ensemble_size is not None else model.refinement_config.ensemble_size
    )
    seed = args.seed if args.seed is not None else model.refinement_config.seed
    inference_generator = torch.Generator(device=device)
    if seed is not None:
        inference_generator.manual_seed(int(seed))

    deterministic, residual, members, mean, spread, truth = [], [], [], [], [], []
    inference_dates: list[str] = []
    limit = int(args.limit_batches or 0)
    for index, batch in enumerate(inference_loader):
        if limit and index >= limit:
            break
        if "tile_index" in batch:
            raise RuntimeError(
                "Refinement inference received tile_index metadata; tiles must be "
                "stitched before they can become one time record."
            )
        batch_size = int(batch["x"].shape[0])
        inference_dates.extend(_batch_date_strings(batch, batch_size))
        batch = _to_device(batch, device)
        out = model.predict(
            batch,
            ensemble_size=ensemble_size,
            generator=inference_generator,
        )
        deterministic.append(out.deterministic.cpu())
        if out.residual_physical is not None:
            residual.append(out.residual_physical.cpu())
        if out.members is not None:
            members.append(out.members.cpu())
            mean.append(out.ensemble_mean.cpu())
            if out.ensemble_spread is not None:
                spread.append(out.ensemble_spread.cpu())
        if "y" in batch:
            truth.append(batch["y"].cpu())

    if not args.output:
        print(f"[refinement] processed {len(deterministic)} batches (no --output given)")
        return 0

    from granitewxc.refinement.io import build_refined_dataset, write_refined_netcdf

    variables = list(config.data.output_vars)
    contract = payload.get("refinement_contract")
    if not isinstance(contract, dict):
        raise RuntimeError(
            "Refinement checkpoint has no serialized scientific contract."
        )
    normalization_metadata = model.residual_normalization_metadata()
    deterministic_t = torch.cat(deterministic, dim=0)
    n_time, _, n_lat, n_lon = deterministic_t.shape
    canonical_shape = tuple(int(value) for value in inference_dataset.fine_shape)
    if (n_lat, n_lon) != canonical_shape:
        raise RuntimeError(
            "Refinement output is not on the full canonical PRISM grid: "
            f"output={(n_lat, n_lon)}, canonical={canonical_shape}."
        )
    if len(inference_dates) != n_time:
        raise RuntimeError(
            f"Collected {len(inference_dates)} source dates for {n_time} predictions."
        )
    if len(set(inference_dates)) != len(inference_dates):
        raise RuntimeError(
            "Duplicate inference dates detected; refusing to serialize spatial "
            "tiles as independent time records."
        )
    try:
        time_values = np.asarray(inference_dates, dtype="datetime64[ns]")
    except ValueError as exc:
        raise RuntimeError(
            f"Inference dates are not valid ISO timestamps: {inference_dates[:3]}"
        ) from exc
    if np.isnat(time_values).any():
        raise RuntimeError("Inference dates contain NaT values.")
    fine_lat = np.asarray(inference_dataset.fine_lat)
    fine_lon = np.asarray(inference_dataset.fine_lon)
    if fine_lat.ndim != 1 or fine_lon.ndim != 1:
        raise RuntimeError(
            "Canonical NARR/PRISM latitude and longitude coordinates must be one-dimensional."
        )
    if (fine_lat.size, fine_lon.size) != canonical_shape:
        raise RuntimeError(
            "Canonical coordinate lengths do not match the dataset grid: "
            f"coords={(fine_lat.size, fine_lon.size)}, grid={canonical_shape}."
        )
    coords = {
        "time": time_values,
        "lat": fine_lat,
        "lon": fine_lon,
    }
    dataset = build_refined_dataset(
        variables=variables,
        coords=coords,
        deterministic=deterministic_t,
        residual=torch.cat(residual, dim=0) if residual else None,
        members=torch.cat(members, dim=0) if members else None,
        ensemble_mean=torch.cat(mean, dim=0) if mean else None,
        ensemble_spread=torch.cat(spread, dim=0) if spread else None,
        refined=torch.cat(mean, dim=0) if mean else None,
        truth=torch.cat(truth, dim=0) if truth else None,
        units=_physical_output_units(variables),
        attrs={
            "case_name": get_case_name(config),
            "refinement_type": model.refinement_config.type,
            "ensemble_size": ensemble_size,
            "seed": -1 if seed is None else int(seed),
            "phase1_checkpoint": phase1_checkpoint or "",
            "phase1_fingerprint": loaded_phase1_fingerprint,
            "refinement_checkpoint": refinement_checkpoint,
            "refinement_checkpoint_sha256": _sha256_file(
                refinement_checkpoint
            ),
            "checkpoint_schema_version": int(
                payload["checkpoint_schema_version"]
            ),
            "refinement_contract_version": int(contract["contract_version"]),
            "residual_contract": str(contract["residual_contract"]),
            "refinement_contract_fingerprint": str(
                payload["refinement_contract_fingerprint"]
            ),
            "residual_normalization": json.dumps(
                normalization_metadata, sort_keys=True, allow_nan=False
            ),
            "residual_units": "physical",
            "prediction_kind": "two_phase_refinement_products",
        },
    )
    if (
        int(payload["checkpoint_schema_version"]) != CHECKPOINT_SCHEMA_VERSION
        or int(contract["contract_version"]) != REFINEMENT_CONTRACT_VERSION
    ):
        raise RuntimeError("Refusing to write an obsolete refinement contract.")
    io_cfg = model.performance_config.io
    path = write_refined_netcdf(
        dataset,
        args.output,
        compression=io_cfg.netcdf_compression,
        compression_level=io_cfg.netcdf_compression_level,
        chunk_sizes={"time": 1, "member": 1},
    )
    print(f"[refinement] wrote {path}")
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
    infer.add_argument("--limit-batches", type=int, default=0)
    infer.add_argument("--output", default=None)
    infer.set_defaults(func=cmd_infer)

    describe = sub.add_parser("describe", parents=[common], help="print the resolved configuration")
    describe.set_defaults(func=cmd_describe)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
