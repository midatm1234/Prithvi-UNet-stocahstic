"""Shared CLI for the temporal workflow.

Subcommands, all driven by a case YAML so the two cases and the two backends run
through the *same* code path:

``describe``
    Print what the config resolves to: time axis audit, run/window counts,
    checkpoint contract, temporal parameter count. Loads no weights beyond the
    contract check, so it is the cheapest way to validate a config.
``check``
    Engineering checks on real data: legacy parity, gradient flow, causality,
    history sensitivity, chunked-vs-single-pass agreement.
``train``
    Fine-tune from a spatial checkpoint (or resume a temporal one).
``infer``
    Chunked inference over a split, writing NetCDF per contiguous run.
``evaluate``
    Score existing predictions, per variable, with the event-alignment guard.

The thin per-case wrappers in ``examples/CORDEX_ML`` and ``examples/NARR_PRISM``
exist so each workflow keeps its familiar entry point.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from granitewxc.temporal.config import TemporalConfigError, parse_temporal_config

__all__ = ["main", "build_parser"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _load(config_path: str):
    from granitewxc.utils.config import get_config

    config = get_config(config_path)
    raw = getattr(config, "temporal", None)
    cfg = parse_temporal_config(raw)
    if cfg is None:
        raise SystemExit(
            f"{config_path} has no enabled 'temporal:' block. This entry point is for the "
            "temporal workflow; use the existing spatial entry points for spatial-only "
            "configs, or set temporal.enabled: true."
        )
    return config, cfg


def _device(requested: str | None, config: Any) -> torch.device:
    name = requested or getattr(config, "device_target", None) or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        print("[device] CUDA requested but unavailable; falling back to CPU.")
        name = "cpu"
    return torch.device(str(name))


def _emit(payload: Any, out_path: str | None) -> None:
    text = json.dumps(payload, indent=2, default=str)
    print(text)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(text, encoding="utf-8")
        print(f"[write] {out_path}")


# ---------------------------------------------------------------------------
# describe
# ---------------------------------------------------------------------------
def cmd_describe(args: argparse.Namespace) -> int:
    from granitewxc.temporal.checkpoint import describe_spatial_checkpoint
    from granitewxc.temporal.sources import build_frame_source
    from granitewxc.temporal.sequence_dataset import TemporalSequenceDataset
    from granitewxc.temporal.training import (
        _split_dates,
        resolve_crop_size,
        resolve_static_channels,
    )

    config, cfg = _load(args.config)
    report: dict[str, Any] = {
        "config": args.config,
        "case_name": getattr(config, "case_name", None),
        "data_type": getattr(config.data, "type", None),
        "output_vars": list(config.data.output_vars),
        "temporal": cfg.to_dict(),
    }

    static_channels = resolve_static_channels(config)

    splits: dict[str, Any] = {}
    for split in args.splits:
        try:
            source = build_frame_source(config, split)
            crop = resolve_crop_size(config, source)
            start, end = _split_dates(config, split)
            ds = TemporalSequenceDataset(
                source,
                window_length=cfg.context_length,
                stride=cfg.sequence_stride if split == "train" else cfg.output_length,
                cadence_days=cfg.cadence_days,
                crop_size=crop,
                random_crop=False,
                static_channels=static_channels,
                date_start=start,
                date_end=end,
                include_hour_of_day=cfg.include_hour_of_day,
                include_lead_time=cfg.include_lead_time,
                lead_time_days=cfg.lead_time_days,
            )
            info = ds.describe()
            info["unique_emitted_dates"] = len(ds.emitted_dates(cfg.output_length))
            info["windows_in_unique_plan"] = len(ds.unique_emission_plan(cfg.output_length))
            splits[split] = info
        except Exception as exc:  # noqa: BLE001 - report, do not abort the whole audit
            splits[split] = {"error": f"{type(exc).__name__}: {exc}"}
    report["splits"] = splits

    for label, path in (
        ("init_from_spatial_checkpoint", cfg.init_from_spatial_checkpoint),
        ("resume_from_temporal_checkpoint", cfg.resume_from_temporal_checkpoint),
    ):
        if not path:
            continue
        try:
            report[label] = describe_spatial_checkpoint(path)
        except Exception as exc:  # noqa: BLE001
            report[label] = {"path": path, "error": f"{type(exc).__name__}: {exc}"}

    _emit(report, args.output)
    return 0


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------
def cmd_check(args: argparse.Namespace) -> int:
    import copy

    from granitewxc.temporal.model import temporal_parameter_names
    from granitewxc.temporal.training import (
        build_sequence_dataloaders,
        build_temporal_model,
        resolve_time_feature_dim,
    )

    config, cfg = _load(args.config)
    device = _device(args.device, config)
    loaders = build_sequence_dataloaders(
        config, cfg, splits=(args.split,), batch_size=1, verbose=True
    )
    time_dim, time_names = resolve_time_feature_dim(loaders[args.split])
    runner, report, _ = build_temporal_model(
        config, cfg, time_feature_dim=time_dim, time_feature_names=time_names,
        device=device, verbose=True,
    )
    model = runner.base

    batch = next(iter(loaders[args.split]))
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    results: dict[str, Any] = {
        "config": args.config,
        "device": str(device),
        "time_features": {"dim": time_dim, "names": time_names},
        "migration": report.to_dict(),
        "temporal_parameters": int(
            sum(p.numel() for n, p in model.named_parameters() if n.startswith("temporal_adapter."))
        ),
    }

    # 1. legacy parity with the gate forced to zero
    adapter = model.temporal_adapter
    saved_gate = adapter.gate.detach().clone()
    with torch.no_grad():
        adapter.gate.zero_()
    runner.eval()
    with torch.no_grad():
        zero_gate = runner(batch).predictions.clone()
    frames = batch["x"].shape[1]
    emit = runner.emitted_indices(frames)
    with torch.no_grad():
        spatial = []
        for t in emit:
            frame = {"x": batch["x"][:, t], "y": batch["y"][:, t]}
            for key in ("static_x", "static_y"):
                if key in batch:
                    frame[key] = batch[key]
            model._temporal_ctx = None
            spatial.append(model(frame))
        spatial = torch.stack(spatial, dim=1)
    results["legacy_parity_gate_zero"] = {
        "bitwise_identical": bool(torch.equal(zero_gate, spatial)),
        "max_abs_diff": float((zero_gate - spatial).abs().max()),
    }
    with torch.no_grad():
        adapter.gate.copy_(saved_gate)

    # 2. near-identity at the configured gate
    with torch.no_grad():
        default_out = runner(batch).predictions.clone()
    scale = float(spatial.abs().mean())
    results["near_identity_default_gate"] = {
        "adapter_init_gate": cfg.latent.adapter_init_gate,
        "mean_relative_deviation": float((default_out - spatial).abs().mean() / max(scale, 1e-12)),
    }

    # 3. gradient flow
    runner.train()
    out = runner(batch)
    out.predictions.pow(2).mean().backward()
    params = dict(model.named_parameters())
    names = temporal_parameter_names(model)
    dead = [
        n for n in names
        if params[n].grad is None or float(params[n].grad.abs().sum()) == 0.0
    ]
    results["gradient_flow"] = {
        "n_temporal_tensors": len(names),
        "n_zero_gradient": len(dead),
        "zero_gradient_examples": dead[:5],
    }
    model.zero_grad(set_to_none=True)
    runner.eval()

    # 4. causality
    perturbed = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    torch.manual_seed(1234)
    perturbed["x"][:, -1] = torch.randn_like(perturbed["x"][:, -1])
    with torch.no_grad():
        after = runner(perturbed).predictions
    n_emit = default_out.shape[1]
    results["causality"] = {
        "earlier_frames_bitwise_unchanged": bool(
            torch.equal(default_out[:, : n_emit - 1], after[:, : n_emit - 1])
        ),
        "last_frame_changed": bool(not torch.equal(default_out[:, -1], after[:, -1])),
    }

    # 5. history sensitivity with the gate temporarily raised so an untrained
    #    adapter produces a measurable effect
    with torch.no_grad():
        adapter.gate.fill_(1.0)
        base = runner(batch).predictions.clone()
        hist = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        torch.manual_seed(99)
        hist["x"][:, :-1] = torch.randn_like(hist["x"][:, :-1])
        alt = runner(hist).predictions
        results["history_sensitivity"] = {
            "note": "gate temporarily set to 1.0 to measure the pathway, not accuracy",
            "max_abs_change_final_frame": float((alt[:, -1] - base[:, -1]).abs().max()),
            "field_scale": float(base[:, -1].abs().mean()),
            "time_features_identical": True,
        }
        adapter.gate.copy_(saved_gate)

    # 6. chunked vs single pass on real frames
    from granitewxc.temporal.inference import verify_chunk_consistency

    try:
        results["chunk_consistency"] = verify_chunk_consistency(
            runner, config, cfg,
            split=args.split,
            date_start=None, date_end=None,
            n_frames=args.chunk_frames,
            chunk_lengths=(args.chunk_frames, max(args.chunk_frames // 3, 2)),
            device=device,
        )
    except Exception as exc:  # noqa: BLE001
        results["chunk_consistency"] = {"error": f"{type(exc).__name__}: {exc}"}

    _emit(results, args.output)
    ok = (
        results["legacy_parity_gate_zero"]["bitwise_identical"]
        and results["gradient_flow"]["n_zero_gradient"] == 0
        and results["causality"]["earlier_frames_bitwise_unchanged"]
        and results["history_sensitivity"]["max_abs_change_final_frame"] > 0.0
    )
    print(f"\n[check] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
def cmd_train(args: argparse.Namespace) -> int:
    from granitewxc.temporal.training import train_temporal_model

    config, cfg = _load(args.config)
    device = _device(args.device, config)
    summary = train_temporal_model(
        config,
        cfg,
        device=device,
        output_dir=args.output_dir,
        max_steps_per_epoch=args.max_steps,
        max_val_steps=args.max_val_steps,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    _emit(summary, args.output)
    return 0


# ---------------------------------------------------------------------------
# infer
# ---------------------------------------------------------------------------
def cmd_infer(args: argparse.Namespace) -> int:
    from granitewxc.temporal.inference import run_sequence_inference, write_netcdf
    from granitewxc.temporal.training import (
        build_sequence_dataloaders,
        build_temporal_model,
        resolve_time_feature_dim,
        _split_dates,
    )

    config, cfg = _load(args.config)
    device = _device(args.device, config)

    if args.checkpoint:
        cfg = type(cfg)(**{**cfg.__dict__, "resume_from_temporal_checkpoint": args.checkpoint,
                           "init_from_spatial_checkpoint": None})

    probe = build_sequence_dataloaders(config, cfg, splits=("train",), batch_size=1, verbose=False)
    time_dim, time_names = resolve_time_feature_dim(probe["train"])
    runner, report, _ = build_temporal_model(
        config, cfg, time_feature_dim=time_dim, time_feature_names=time_names,
        device=device, verbose=True,
    )

    start, end = _split_dates(config, args.split)
    results = run_sequence_inference(
        runner, config, cfg,
        split=args.split,
        date_start=args.date_start or start,
        date_end=args.date_end or end,
        device=device,
        chunk_length=args.chunk_length,
    )

    out_dir = Path(
        args.output_dir
        or cfg.inference.output_dir
        or Path(getattr(config, "path_experiment", "./experiments")) / "temporal_inference"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    if cfg.inference.write_netcdf:
        from granitewxc.temporal.sources import build_frame_source

        source = build_frame_source(config, args.split)
        units = list(getattr(source, "target_units", []) or [])
        for result in results:
            first = "".join(f"{v:02d}" for v in result.dates[0][:3])
            last = "".join(f"{v:02d}" for v in result.dates[-1][:3])
            name = f"{getattr(config,'case_name','case')}_temporal_run{result.run_id}_{first}_{last}.nc"
            written.append(
                write_netcdf(
                    result,
                    out_dir / name,
                    output_vars=list(config.data.output_vars),
                    units=units,
                    calendar=source.calendar().name,
                    attrs={
                        "temporal_config": cfg.to_dict(),
                        "migration_report": report.to_dict(),
                        "split": args.split,
                    },
                )
            )

    npz = out_dir / f"{getattr(config,'case_name','case')}_temporal_{args.split}.npz"
    np.savez_compressed(
        npz,
        **{
            f"run{r.run_id}_pred": r.predictions for r in results
        },
        **{
            f"run{r.run_id}_target": r.targets
            for r in results
            if r.targets is not None
        },
        **{
            f"run{r.run_id}_mask": r.valid_mask
            for r in results
            if r.valid_mask is not None
        },
        **{f"run{r.run_id}_dates": np.array(r.dates) for r in results},
        **{f"run{r.run_id}_seams": np.array(r.seam_indices, dtype=np.int64) for r in results},
        output_vars=np.array([str(v) for v in config.data.output_vars]),
    )
    _emit(
        {
            "output_dir": str(out_dir),
            "npz": str(npz),
            "netcdf": written,
            "runs": [r.describe() for r in results],
        },
        args.output,
    )
    return 0


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------
def cmd_evaluate(args: argparse.Namespace) -> int:
    from granitewxc.temporal.metrics import evaluate_predictions

    config, cfg = _load(args.config)
    data = np.load(args.predictions, allow_pickle=False)
    output_vars = [str(v) for v in data["output_vars"]]
    run_ids = sorted(
        {int(k.split("_")[0][3:]) for k in data.files if k.startswith("run") and k.endswith("_pred")}
    )

    report: dict[str, Any] = {
        "predictions": args.predictions,
        "event_aligned": bool(args.event_aligned),
        "output_vars": output_vars,
        "runs": {},
    }
    for rid in run_ids:
        pred = data[f"run{rid}_pred"]
        target = data.get(f"run{rid}_target") if hasattr(data, "get") else (
            data[f"run{rid}_target"] if f"run{rid}_target" in data.files else None
        )
        mask = data[f"run{rid}_mask"] if f"run{rid}_mask" in data.files else None
        dates = data[f"run{rid}_dates"] if f"run{rid}_dates" in data.files else None
        seams = data[f"run{rid}_seams"] if f"run{rid}_seams" in data.files else []
        report["runs"][str(rid)] = evaluate_predictions(
            pred,
            target,
            output_vars=output_vars,
            dates=[tuple(int(v) for v in d) for d in dates] if dates is not None else None,
            mask=mask.astype(bool) if mask is not None else None,
            event_aligned=bool(args.event_aligned),
            wet_threshold=cfg.evaluation.wet_threshold,
            lags=cfg.evaluation.autocorr_lags,
            accumulation_windows=cfg.evaluation.accumulation_windows,
            boundary_width=cfg.evaluation.boundary_width,
            seam_indices=[int(s) for s in np.atleast_1d(seams)],
        )
    _emit(report, args.output)
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
def build_parser(prog: str = "temporal") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", required=True, help="Case YAML with a temporal: block")
        p.add_argument("--device", default=None, help="cuda | cpu (default: config/auto)")
        p.add_argument("--output", default=None, help="Write the JSON report here as well")

    p = sub.add_parser("describe", help="Audit the config, splits and checkpoints")
    common(p)
    p.add_argument("--splits", nargs="+", default=["train", "validation"])
    p.set_defaults(func=cmd_describe)

    p = sub.add_parser("check", help="Engineering checks on real data")
    common(p)
    p.add_argument("--split", default="validation")
    p.add_argument("--chunk-frames", type=int, default=12)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("train", help="Fine-tune the temporal model")
    common(p)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None, help="Cap training steps per epoch")
    p.add_argument("--max-val-steps", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("infer", help="Chunked inference over a split")
    common(p)
    p.add_argument("--split", default="test")
    p.add_argument("--checkpoint", default=None, help="Override the temporal checkpoint")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--chunk-length", type=int, default=None)
    p.add_argument("--date-start", default=None)
    p.add_argument("--date-end", default=None)
    p.set_defaults(func=cmd_infer)

    p = sub.add_parser("evaluate", help="Score saved predictions")
    common(p)
    p.add_argument("--predictions", required=True, help=".npz written by 'infer'")
    p.add_argument(
        "--event-aligned",
        dest="event_aligned",
        action="store_true",
        default=True,
        help="Predictors and targets describe the same weather (default for the "
        "perfect-predictor SA and NARR/PRISM cases)",
    )
    p.add_argument(
        "--not-event-aligned",
        dest="event_aligned",
        action="store_false",
        help="Free-running application: report distributional metrics only",
    )
    p.set_defaults(func=cmd_evaluate)
    return parser


def main(argv: Sequence[str] | None = None, *, prog: str = "temporal") -> int:
    args = build_parser(prog).parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.func(args))
    except TemporalConfigError as exc:
        print(f"[config error] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
