#!/usr/bin/env python
"""Bounded real-data comparison of temporal backends against matched controls.

Five variants, all sharing the same data, splits, seed, optimizer step count,
learning rates and per-frame loss. They differ only in how time is handled, so a
difference between them is attributable to that and not to anything else:

``baseline``
    The existing frame-independent Phase-1 checkpoint, untrained here. The
    adapter is attached with ``adapter_init_gate = 0``, which makes the emitted
    field bit-for-bit the frame-independent prediction.
``spatial_ft``
    Matched spatial fine-tuning control. Same steps and same decoder learning
    rate, but the temporal pathway is inert (gate 0, temporal parameters frozen),
    so this isolates "does *any* further fine-tuning on these windows help".
``time_only``
    Date/time-conditioning ablation. The full temporal module trains, receives
    the calendar features, and has the same parameter count -- but its hidden
    state is zeroed before every frame, so it has no memory. This separates
    "knows the date" from "remembers yesterday".
``convgru``
    Full recurrent temporal model.
``mamba``
    Full temporal Mamba (SSD) model.

ACCEPTANCE TOLERANCES ARE DEFINED IN THIS FILE, BEFORE ANY TEST RESULT IS READ
(see :data:`ACCEPTANCE`). They are evaluated mechanically at the end. A temporal
backend passes only if it improves temporal/event representation *and* beats both
controls *and* does not materially degrade spatial accuracy or extremes.

Usage::

    mamba run -n Prithvi python examples/CORDEX_ML/cordex_temporal_experiment.py \
        --steps 400 --val-steps 60 --test-years 3
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import hashlib
from pathlib import Path
from typing import Any

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from granitewxc.temporal.config import parse_temporal_config  # noqa: E402
from granitewxc.temporal.inference import run_sequence_inference  # noqa: E402
from granitewxc.temporal.metrics import evaluate_predictions  # noqa: E402
from granitewxc.temporal.training import (  # noqa: E402
    build_sequence_dataloaders,
    build_temporal_model,
    resolve_time_feature_dim,
    train_temporal_model,
)
from granitewxc.utils.config import ExperimentConfig  # noqa: E402

SA_RECURRENT = "examples/CORDEX_ML/SA_downscaling_refinement_T2_ACCESS-CM2_static_temporal_recurrent.yaml"
SA_MAMBA = "examples/CORDEX_ML/SA_downscaling_refinement_T2_ACCESS-CM2_static_temporal_mamba.yaml"
SA_NATIVE_PAIR = (
    "examples/CORDEX_ML/"
    "SA_downscaling_refinement_T2_ACCESS-CM2_static_temporal_prithvi_native_pair.yaml"
)
SA_NATIVE_PAIR_PRETEXT = (
    "examples/CORDEX_ML/"
    "SA_downscaling_refinement_T2_ACCESS-CM2_static_temporal_prithvi_native_pair_pretext.yaml"
)


# ---------------------------------------------------------------------------
# Pre-registered acceptance tolerances
# ---------------------------------------------------------------------------
#: Each entry: (label, metric path, comparison, threshold).
#: ``rel_improve`` = (|baseline| - |variant|) / |baseline| must be >= threshold.
#: ``rel_degrade``  = (variant - baseline) / |baseline| must be <= threshold.
#: ``abs_degrade``  = (baseline - variant) must be <= threshold.
ACCEPTANCE: dict[str, list[tuple[str, str, str, float]]] = {
    "primary_temporal": [
        # The headline claim: better agreement with observed temporal dependence.
        ("lag-1 autocorr error |.| reduced >=10%", "paired.autocorr_lag1_error", "rel_improve", 0.10),
        ("lag-2 autocorr error |.| reduced >=5%", "paired.autocorr_lag2_error", "rel_improve", 0.05),
        ("day-to-day tendency RMSE reduced >=2%", "paired.tendency_rmse", "rel_improve", 0.02),
        ("3-day accumulation RMSE reduced >=1%", "paired.acc3d_rmse", "rel_improve", 0.01),
    ],
    "guardrail_spatial": [
        # Must not buy temporal skill by degrading what already works.
        ("per-frame RMSE not worse by >1%", "paired.rmse", "rel_degrade", 0.01),
        ("MAE not worse by >1%", "paired.mae", "rel_degrade", 0.01),
        ("daily spatial correlation not worse by >0.005", "paired.spatial_r_daily_mean", "abs_degrade", 0.005),
        ("99th-pct bias |.| not worse by >5%", "paired.q0.99_bias", "rel_degrade", 0.05),
        ("std ratio not moved toward 0 by >2%", "paired.std_ratio", "abs_degrade", 0.02),
        # A model that under-varies day to day is smoothing, not learning.
        ("tendency std ratio not reduced by >2%", "paired.tendency_std_ratio", "abs_degrade", 0.02),
    ],
}

#: A temporal backend must beat these variants on every primary metric, otherwise
#: the improvement is attributable to fine-tuning or date conditioning instead of
#: temporal memory.
MUST_BEAT = ("spatial_ft", "time_only")

#: Variants that are candidates rather than controls. Everything else is scored
#: against the baseline for information but is never "accepted".
CANDIDATES = ("convgru", "mamba", "native_pair", "native_pair_pretext")

#: Per-candidate controls required IN ADDITION to ``MUST_BEAT``. These only ever
#: add hurdles; ``ACCEPTANCE`` and ``MUST_BEAT`` are pre-registered and unchanged.
#:
#: The paired-state pathway widens the patch embedding, so it has slightly more
#: parameters than the frame-independent model. ``native_pair_nohistory`` holds
#: that capacity fixed and removes only the information, which is the control that
#: makes "history helped" separable from "capacity helped".
EXTRA_MUST_BEAT: dict[str, tuple[str, ...]] = {
    "native_pair": ("native_pair_nohistory",),
    "native_pair_pretext": ("native_pair_nohistory", "native_pair"),
}


# ---------------------------------------------------------------------------
# variants
# ---------------------------------------------------------------------------
SCORER_VERSION = "2.0-missing-metrics-and-dotted-keys"


def variant_overrides() -> dict[str, dict[str, Any]]:
    """Config patches defining each variant, relative to the case YAML."""
    return {
        "baseline": {
            "_source": SA_RECURRENT,
            "_train": False,
            "temporal": {"latent": {"adapter_init_gate": 0.0}},
        },
        "spatial_ft": {
            "_source": SA_RECURRENT,
            "_train": True,
            "temporal": {
                "latent": {"adapter_init_gate": 0.0},
                "freeze": {"temporal": True},
            },
        },
        "time_only": {
            "_source": SA_RECURRENT,
            "_train": True,
            "temporal": {"state": {"reset_every_frame": True}},
        },
        "convgru": {"_source": SA_RECURRENT, "_train": True, "temporal": {}},
        "mamba": {"_source": SA_MAMBA, "_train": True, "temporal": {}},
        # --- Prithvi-native paired state -----------------------------------
        # Two dates enter the shared transformer together via the backbone's own
        # multi-timestamp input axis, instead of being encoded separately and
        # mixed afterwards by a new module. The loss configuration is byte
        # identical to the four variants above, so this differs from them in the
        # temporal pathway alone.
        "native_pair": {"_source": SA_NATIVE_PAIR, "_train": True, "temporal": {}},
        # Capacity control for the above: same architecture, same parameter
        # count, same time metadata, but the history slot is a copy of date t. It
        # separates "useful history" from "30,720 extra patch-embedding
        # parameters", which the previous experiment could not do for ConvGRU's
        # 7.6 M new parameters.
        "native_pair_nohistory": {
            "_source": SA_NATIVE_PAIR,
            "_train": True,
            "temporal": {"native_pair": {"history_mode": "duplicate_current"}},
        },
        # Pretraining-aligned auxiliary objectives on top of the paired pathway:
        # masked atmospheric reconstruction and atmospheric transition
        # prediction, both training the shared trunk. Separate variant so the
        # objective change is never confounded with the architectural change.
        "native_pair_pretext": {
            "_source": SA_NATIVE_PAIR_PRETEXT,
            "_train": True,
            "temporal": {},
        },
    }


def _deep_merge(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def build_variant_config(name: str, patch: dict[str, Any], out_root: Path) -> ExperimentConfig:
    raw = yaml.safe_load((Path(_REPO_ROOT) / patch["_source"]).read_text(encoding="utf-8"))
    applied = {k: v for k, v in patch.items() if not k.startswith("_")}
    raw = _deep_merge(raw, applied)
    run_dir = out_root / name
    raw["path_experiment"] = str(run_dir)
    raw["checkpoint_dir"] = str(run_dir / "checkpoints")
    raw["run_dir"] = str(run_dir / "runs")
    raw["case_name"] = f"sa_temporal_experiment_{name}"
    raw["job_id"] = f"sa_temporal_experiment_{name}"
    raw["temporal"]["inference"]["output_dir"] = str(run_dir / "inference")
    # One optimizer step per window so a bounded run still takes many steps.
    raw["gradient_accumulation_steps"] = 1
    raw.setdefault("training", {})["gradient_accumulation_steps"] = 1
    (out_root / name).mkdir(parents=True, exist_ok=True)
    (out_root / name / "resolved.yaml").write_text(
        yaml.safe_dump(raw, sort_keys=False), encoding="utf-8"
    )
    return ExperimentConfig.from_dict(raw)


# ---------------------------------------------------------------------------
# metric helpers
# ---------------------------------------------------------------------------
def _dig(report: dict, variable: str, path: str) -> float | None:
    node: Any = report.get("variables", {}).get(variable)
    if node is None:
        return None
    parts = path.split(".")
    while parts:
        if not isinstance(node, dict):
            return None
        # Metric names themselves contain dots, e.g. q0.99_bias.
        remainder = ".".join(parts)
        if remainder in node:
            node = node[remainder]
            break
        part = parts.pop(0)
        if part not in node:
            return None
        node = node[part]
    try:
        value = float(node)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def score_acceptance(
    baseline: dict, variant: dict, variables: list[str]
) -> dict[str, Any]:
    """Evaluate the pre-registered tolerances for one variant."""
    result: dict[str, Any] = {"variables": {}, "pass_primary": bool(variables), "pass_guardrail": bool(variables)}
    for var in variables:
        entry: dict[str, Any] = {"primary": {}, "guardrail": {}}
        for group, checks in ACCEPTANCE.items():
            key = "primary" if group == "primary_temporal" else "guardrail"
            for label, path, mode, threshold in checks:
                b = _dig(baseline, var, path)
                v = _dig(variant, var, path)
                if b is None or v is None:
                    entry[key][label] = {"status": "unavailable", "baseline": b, "variant": v}
                    result["pass_primary" if key == "primary" else "pass_guardrail"] = False
                    continue
                if mode == "rel_improve":
                    denom = abs(b)
                    value = (denom - abs(v)) / denom if denom > 1e-12 else float("nan")
                    ok = bool(np.isfinite(value) and value >= threshold)
                elif mode == "rel_degrade":
                    denom = abs(b)
                    value = (abs(v) - denom) / denom if denom > 1e-12 else float("nan")
                    ok = bool(np.isfinite(value) and value <= threshold)
                else:  # abs_degrade
                    value = b - v
                    ok = bool(np.isfinite(value) and value <= threshold)
                entry[key][label] = {
                    "status": "pass" if ok else "FAIL",
                    "baseline": b,
                    "variant": v,
                    "measure": None if not np.isfinite(value) else round(float(value), 6),
                    "threshold": threshold,
                }
                if not ok:
                    if key == "primary":
                        result["pass_primary"] = False
                    else:
                        result["pass_guardrail"] = False
        result["variables"][var] = entry
    result["accepted"] = bool(result["pass_primary"] and result["pass_guardrail"])
    return result


def beats_controls(
    reports: dict[str, dict], variant: str, controls: tuple[str, ...], variables: list[str]
) -> dict[str, Any]:
    """Check the temporal variant beats each control on every primary metric."""
    out: dict[str, Any] = {}
    for control in controls:
        if control not in reports:
            out[control] = {"status": "unavailable"}
            continue
        per_var: dict[str, Any] = {}
        all_ok = bool(variables)
        for var in variables:
            checks: dict[str, Any] = {}
            for label, path, mode, _ in ACCEPTANCE["primary_temporal"]:
                v = _dig(reports[variant], var, path)
                c = _dig(reports[control], var, path)
                if v is None or c is None:
                    checks[label] = "unavailable"
                    all_ok = False
                    continue
                ok = abs(v) < abs(c)
                checks[label] = {
                    "status": "pass" if ok else "FAIL",
                    variant: v,
                    control: c,
                }
                all_ok = all_ok and ok
            per_var[var] = checks
        out[control] = {"beats": all_ok, "detail": per_var}
    return out


def assert_optimizer_budget(summary: dict, updates: int, epochs: int) -> None:
    """A spatial control updates its decoder while temporal-update count stays zero."""
    actual = summary.get("global_step")
    if actual != updates * epochs:
        raise RuntimeError(f"Actual optimizer updates {actual} differ from requested {updates * epochs}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="examples/CORDEX_ML/runs_temporal/experiment")
    ap.add_argument("--steps", type=int, default=400, help="Actual optimizer updates per trained variant per epoch; accumulation fixed to 1")
    ap.add_argument("--val-steps", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--test-years", type=int, default=3, help="Years of the held-out test period")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--variants",
        nargs="+",
        default=[
            "baseline",
            "spatial_ft",
            "time_only",
            "native_pair",
            "native_pair_nohistory",
            "native_pair_pretext",
        ],
        help=(
            "Variants to run, in order. The default is the paired-state experiment with "
            "its matched controls; pass 'convgru mamba' to re-run the archived "
            "bottleneck-adapter backends in the same directory."
        ),
    )
    args = ap.parse_args()
    if min(args.steps, args.val_steps, args.epochs, args.test_years) < 1:
        ap.error("steps, val-steps, epochs and test-years must be positive")
    unknown = set(args.variants) - set(variant_overrides())
    if unknown or len(set(args.variants)) != len(args.variants):
        ap.error(f"unknown or duplicate variants: {unknown or args.variants}")

    out_root = Path(args.out)
    if (out_root / "manifest.json").exists():
        raise FileExistsError(f"Preserving existing experiment {out_root}; use a new versioned directory")
    out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    test_start = "1981-01-01"
    test_end = f"{1980 + int(args.test_years)}-12-31"

    manifest: dict[str, Any] = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "training_windows": args.steps,
        "optimizer_updates_per_epoch": args.steps,
        "scorer_version": SCORER_VERSION,
        "source_root": _REPO_ROOT,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "validation_windows": args.val_steps,
        "epochs": args.epochs,
        "test_period": [test_start, test_end],
        "acceptance_tolerances": {
            group: [
                {"label": lab, "metric": path, "mode": mode, "threshold": thr}
                for lab, path, mode, thr in checks
            ]
            for group, checks in ACCEPTANCE.items()
        },
        "must_beat": list(MUST_BEAT),
        # Recorded here, before any variant runs, for the same reason the
        # tolerances are: these are additional controls a candidate must beat, and
        # writing them after seeing results would make them worthless.
        "candidates": list(CANDIDATES),
        "extra_must_beat": {k: list(v) for k, v in EXTRA_MUST_BEAT.items()},
        "planned_variants": list(args.variants),
        "variants": {name: {"status": "not started"} for name in args.variants},
    }
    print("=" * 78)
    print("PRE-REGISTERED ACCEPTANCE TOLERANCES (fixed before any test result is read)")
    print(json.dumps(manifest["acceptance_tolerances"], indent=2))
    print("=" * 78)
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    overrides = variant_overrides()
    reports: dict[str, dict] = {}
    variables: list[str] = []

    for name in args.variants:
        patch = overrides[name]
        print(f"\n{'=' * 78}\nVARIANT: {name}\n{'=' * 78}")
        config = build_variant_config(name, patch, out_root)
        cfg = parse_temporal_config(config.temporal)
        run_dir = out_root / name
        info: dict[str, Any] = {"trained": bool(patch["_train"]), "source": patch["_source"]}

        info.update(status="running", phase="training" if patch["_train"] else "inference",
                    started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    pid=os.getpid(), configuration=str(run_dir / "resolved.yaml"))
        manifest["variants"][name] = info
        (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        t0 = time.time()
        if patch["_train"]:
            summary = train_temporal_model(
                config,
                cfg,
                device=device,
                output_dir=str(run_dir / "checkpoints"),
                max_steps_per_epoch=args.steps,
                max_val_steps=args.val_steps,
                num_epochs=args.epochs,
                batch_size=1,
                num_workers=0,
                verbose=True,
            )
            info["train_seconds"] = time.time() - t0
            info["history"] = summary["history"]
            info["variable_scales"] = summary["variable_scales"]
            info["trained_temporal_steps"] = summary["trained_temporal_steps"]
            info["training_summary"] = summary
            info["actual_optimizer_steps"] = summary["global_step"]
            assert_optimizer_budget(summary, args.steps, args.epochs)
            ckpt = run_dir / "checkpoints" / "best.ckpt"
            if not ckpt.is_file():
                ckpt = run_dir / "checkpoints" / "last.ckpt"
            # Resume from the trained temporal checkpoint for inference.
            infer_cfg_raw = yaml.safe_load(
                (run_dir / "resolved.yaml").read_text(encoding="utf-8")
            )
            infer_cfg_raw["temporal"]["resume_from_temporal_checkpoint"] = str(ckpt)
            infer_cfg_raw["temporal"]["init_from_spatial_checkpoint"] = None
            infer_config = ExperimentConfig.from_dict(infer_cfg_raw)
            infer_temporal = parse_temporal_config(infer_config.temporal)
            info["checkpoint"] = str(ckpt)
        else:
            infer_config, infer_temporal = config, cfg
            info["checkpoint"] = cfg.init_from_spatial_checkpoint
            info["trained_temporal_steps"] = 0

        info["phase"] = "inference"
        (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

        # ---- inference on the held-out test period ----
        probe = build_sequence_dataloaders(
            infer_config, infer_temporal, splits=("train",), batch_size=1, verbose=False
        )
        td, tn = resolve_time_feature_dim(probe["train"])
        runner, migration, _ = build_temporal_model(
            infer_config,
            infer_temporal,
            time_feature_dim=td,
            time_feature_names=tn,
            device=device,
            verbose=True,
        )
        info["migration"] = migration.to_dict()

        t1 = time.time()
        results = run_sequence_inference(
            runner,
            infer_config,
            infer_temporal,
            split="test",
            date_start=test_start,
            date_end=test_end,
            device=device,
            verbose=True,
        )
        info["inference_seconds"] = time.time() - t1

        # ---- evaluate, concatenating runs but never double counting a date ----
        preds = np.concatenate([r.predictions for r in results], axis=0)
        targs = np.concatenate([r.targets for r in results], axis=0)
        masks = np.concatenate([r.valid_mask for r in results], axis=0)
        dates = [d for r in results for d in r.dates]
        seams: list[int] = []
        offset = 0
        for r in results:
            seams.extend(int(s) + offset for s in r.seam_indices)
            offset += r.predictions.shape[0]

        variables = [str(v) for v in infer_config.data.output_vars]
        report = evaluate_predictions(
            preds,
            targs,
            output_vars=variables,
            dates=[tuple(int(x) for x in d) for d in dates],
            mask=masks.astype(bool),
            event_aligned=True,   # verified for the SA perfect-predictor case
            wet_threshold=infer_temporal.evaluation.wet_threshold,
            lags=infer_temporal.evaluation.autocorr_lags,
            accumulation_windows=infer_temporal.evaluation.accumulation_windows,
            boundary_width=infer_temporal.evaluation.boundary_width,
            seam_indices=seams,
        )
        reports[name] = report
        info["n_dates"] = int(preds.shape[0])
        (run_dir / "evaluation.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        np.savez_compressed(
            run_dir / "predictions.npz",
            pred=preds.astype(np.float32),
            target=targs.astype(np.float32),
            mask=masks.astype(np.int8),
            dates=np.array(dates),
            seams=np.array(seams, dtype=np.int64),
            output_vars=np.array(variables),
        )
        info.update(status="completed", phase="completed", predictions=str(run_dir / "predictions.npz"),
                    evaluation=str(run_dir / "evaluation.json"),
                    completed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        manifest["variants"][name] = info
        (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

        for var in variables:
            p = report["variables"][var].get("paired", {})
            print(
                f"  [{name}/{var}] rmse={p.get('rmse', float('nan')):.4f} "
                f"bias={p.get('bias', float('nan')):+.4f} "
                f"tend_rmse={p.get('tendency_rmse', float('nan')):.4f} "
                f"ac1_err={p.get('autocorr_lag1_error', float('nan')):+.4f}"
            )

        del runner
        torch.cuda.empty_cache()

    # ---- scorecard ----
    scorecard: dict[str, Any] = {"scorer_version": SCORER_VERSION, "variables": variables, "variants": {}}
    if "baseline" in reports:
        for name in args.variants:
            if name == "baseline":
                continue
            entry = score_acceptance(reports["baseline"], reports[name], variables)
            if name in CANDIDATES:
                # MUST_BEAT is pre-registered and is applied unchanged. Candidates
                # that add parameters to the *input* pathway must additionally beat
                # their own same-architecture no-history control, which is an
                # ADDITIONAL hurdle, not a relaxation of any existing one: without
                # it, a gain could be bought with the extra parameters rather than
                # with the history they carry.
                required = tuple(MUST_BEAT) + tuple(EXTRA_MUST_BEAT.get(name, ()))
                entry["versus_controls"] = beats_controls(reports, name, required, variables)
                entry["must_beat"] = list(required)
                missing = [
                    c
                    for c in tuple(MUST_BEAT) + tuple(EXTRA_MUST_BEAT.get(name, ()))
                    if c not in reports
                ]
                entry["must_beat_missing"] = missing
                beat_all = all(
                    v.get("beats") is True for v in entry["versus_controls"].values()
                )
                entry["beats_all_controls"] = beat_all
                # A control that was never run cannot be counted as beaten.
                entry["scientific_acceptance"] = bool(
                    entry["accepted"] and beat_all and not missing
                )
            else:
                entry["scientific_acceptance"] = None  # controls are not candidates
            scorecard["variants"][name] = entry

    (out_root / "scorecard.json").write_text(
        json.dumps(scorecard, indent=2, default=str), encoding="utf-8"
    )
    manifest["scorecard_path"] = str(out_root / "scorecard.json")
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print(f"\n{'=' * 78}\nSCORECARD\n{'=' * 78}")
    for name, entry in scorecard["variants"].items():
        verdict = entry.get("scientific_acceptance")
        label = {True: "ACCEPTED", False: "NOT ACCEPTED", None: "control"}[verdict]
        print(f"\n{name}: {label}  (primary={entry['pass_primary']} guardrail={entry['pass_guardrail']})")
        if "beats_all_controls" in entry:
            print(f"  beats {list(MUST_BEAT)}: {entry['beats_all_controls']}")
        for var, detail in entry["variables"].items():
            fails = [
                lab
                for group in ("primary", "guardrail")
                for lab, v in detail[group].items()
                if isinstance(v, dict) and v.get("status") == "FAIL"
            ]
            print(f"  {var}: {len(fails)} failing check(s)" + (f" -> {fails}" if fails else ""))
    print(f"\nWrote {out_root / 'scorecard.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
