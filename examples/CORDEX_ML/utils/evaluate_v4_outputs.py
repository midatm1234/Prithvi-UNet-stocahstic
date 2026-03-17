#!/usr/bin/env python
"""Compare baseline vs v4 CORDEX inference behavior on the same samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
PROJECT_DIR = REPO_ROOT / "examples" / "CORDEX_ML"
for path in (REPO_ROOT, PROJECT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cordex_inference import CordexWrappedDataset, build_inference_dataset  # noqa: E402
from granitewxc.models.model import get_finetune_model_UNET  # noqa: E402
from granitewxc.utils.config import get_config  # noqa: E402
from utils.inference_blending import (  # noqa: E402
    infer_batch_with_boundary_mitigation,
    resolve_boundary_mitigation_settings,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline vs v4 CORDEX inference settings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baseline-config", required=True)
    parser.add_argument("--v4-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--predictor-file", default=None)
    parser.add_argument("--target-file", default=None)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--baseline-use-mitigation",
        action="store_true",
        help="Enable boundary mitigation for baseline run (disabled by default).",
    )
    parser.add_argument(
        "--output-json",
        default=str(PROJECT_DIR / "evaluations" / "v4_summary.json"),
    )
    return parser.parse_args()


def _resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA but no CUDA device is available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_dataset_paths(config: Any, predictor_file: str | None, target_file: str | None) -> tuple[str, str]:
    predictor = predictor_file
    target = target_file
    if predictor is None:
        predictor = list(getattr(config.data, "training_predictor_paths", []))[0]
    if target is None:
        target = list(getattr(config.data, "training_target_paths", []))[0]
    return str(Path(predictor).resolve()), str(Path(target).resolve())


def _load_model(config: Any, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = get_finetune_model_UNET(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model_state = model.state_dict()

    weights_have_module_prefix = all(key.startswith("module.") for key in state_dict.keys())
    model_expects_module_prefix = all(key.startswith("module.") for key in model_state.keys())
    if model_expects_module_prefix and not weights_have_module_prefix:
        state_dict = state_dict.__class__((f"module.{key}", value) for key, value in state_dict.items())
    elif weights_have_module_prefix and not model_expects_module_prefix:
        prefix_len = len("module.")
        state_dict = state_dict.__class__((key[prefix_len:], value) for key, value in state_dict.items())

    scaler_key_parts = ("input_scalers_", "output_scalers_", "static_input_scalers_", "static_output_scalers_")
    state_dict = state_dict.__class__(
        (key, value)
        for key, value in state_dict.items()
        if not any(part in key for part in scaler_key_parts)
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def _run_inference(
    *,
    config: Any,
    model: torch.nn.Module,
    device: torch.device,
    predictor_file: str,
    target_file: str,
    num_samples: int,
    batch_size: int,
    use_mitigation_override: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = build_inference_dataset(
        config,
        predictor_paths=[predictor_file],
        target_paths=[target_file],
        allow_time_mismatch=True,
    )
    wrapped = CordexWrappedDataset(dataset)
    dl = DataLoader(wrapped, batch_size=batch_size, shuffle=False, num_workers=0)

    boundary_cfg = resolve_boundary_mitigation_settings(config)
    if use_mitigation_override is not None:
        boundary_cfg.enabled = bool(use_mitigation_override)

    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    seen = 0
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device=device, dtype=torch.float32) for k, v in batch.items()}
            out, _, _ = infer_batch_with_boundary_mitigation(model=model, batch=batch, cfg=boundary_cfg)
            preds.append(out.detach().cpu().numpy().astype(np.float32, copy=False))
            targets.append(batch["y"].detach().cpu().numpy().astype(np.float32, copy=False))
            seen += out.shape[0]
            if seen >= num_samples:
                break

    if not preds:
        raise RuntimeError("No samples produced during evaluation.")

    pred_arr = np.concatenate(preds, axis=0)[:num_samples]
    tgt_arr = np.concatenate(targets, axis=0)[:num_samples]
    return pred_arr, tgt_arr


def _rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def _seam_metric(values: np.ndarray, step_h: int, step_w: int) -> dict[str, float]:
    # values shape: [N, H, W]
    n, h, w = values.shape
    del n
    seam_chunks: list[np.ndarray] = []
    interior_chunks: list[np.ndarray] = []

    if step_h > 0 and h > 1:
        gy = np.abs(values[:, 1:, :] - values[:, :-1, :])
        mask = np.ones(h - 1, dtype=bool)
        for y in range(step_h, h, step_h):
            seam_chunks.append(gy[:, y - 1 : y, :].reshape(-1))
            mask[y - 1] = False
        if mask.any():
            interior_chunks.append(gy[:, mask, :].reshape(-1))

    if step_w > 0 and w > 1:
        gx = np.abs(values[:, :, 1:] - values[:, :, :-1])
        mask = np.ones(w - 1, dtype=bool)
        for x in range(step_w, w, step_w):
            seam_chunks.append(gx[:, :, x - 1 : x].reshape(-1))
            mask[x - 1] = False
        if mask.any():
            interior_chunks.append(gx[:, :, mask].reshape(-1))

    seam = float(np.mean(np.concatenate(seam_chunks))) if seam_chunks else float("nan")
    interior = float(np.mean(np.concatenate(interior_chunks))) if interior_chunks else float("nan")
    ratio = float(seam / (interior + 1e-12)) if np.isfinite(seam) and np.isfinite(interior) else float("nan")
    return {
        "seam_abs_mean": seam,
        "interior_abs_mean": interior,
        "seam_to_interior_ratio": ratio,
    }


def _precip_distribution_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    p = pred[np.isfinite(pred)].reshape(-1)
    t = target[np.isfinite(target)].reshape(-1)
    if p.size == 0 or t.size == 0:
        return {}

    qs = np.array([0.5, 0.75, 0.9, 0.95, 0.99], dtype=np.float64)
    q_pred = np.quantile(p, qs)
    q_tgt = np.quantile(t, qs)
    zero_thr = 1e-8
    near_zero_thr = 1e-4

    return {
        "zero_fraction_pred": float(np.mean(p <= zero_thr)),
        "zero_fraction_target": float(np.mean(t <= zero_thr)),
        "near_zero_fraction_pred": float(np.mean(p <= near_zero_thr)),
        "near_zero_fraction_target": float(np.mean(t <= near_zero_thr)),
        "p95_pred": float(np.quantile(p, 0.95)),
        "p95_target": float(np.quantile(t, 0.95)),
        "p99_pred": float(np.quantile(p, 0.99)),
        "p99_target": float(np.quantile(t, 0.99)),
        "quantile_mae": float(np.mean(np.abs(q_pred - q_tgt))),
    }


def _to_markdown(summary: dict[str, Any]) -> str:
    lines = ["# CORDEX Baseline vs V4 Evaluation", ""]
    lines.append(f"- Samples: {summary['sample_count']}")
    lines.append(f"- Predictor file: `{summary['predictor_file']}`")
    lines.append(f"- Target file: `{summary['target_file']}`")
    lines.append("")
    lines.append("## Metrics")
    lines.append("| Variable | RMSE Baseline | RMSE V4 | Delta (V4-Baseline) | Seam Ratio Baseline | Seam Ratio V4 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for var_name, var_metrics in summary["variables"].items():
        lines.append(
            f"| {var_name} | {var_metrics['rmse_baseline']:.6g} | {var_metrics['rmse_v4']:.6g} | "
            f"{var_metrics['rmse_delta']:.6g} | {var_metrics['seam_ratio_baseline']:.6g} | "
            f"{var_metrics['seam_ratio_v4']:.6g} |"
        )
    lines.append("")
    if "pr" in summary["variables"] and "precip_distribution" in summary:
        pd = summary["precip_distribution"]
        lines.append("## Precipitation Distribution")
        lines.append(f"- Zero fraction baseline: {pd['zero_fraction_baseline']:.6g}")
        lines.append(f"- Zero fraction v4: {pd['zero_fraction_v4']:.6g}")
        lines.append(f"- Zero fraction target: {pd['zero_fraction_target']:.6g}")
        lines.append(f"- Near-zero fraction baseline: {pd['near_zero_fraction_baseline']:.6g}")
        lines.append(f"- Near-zero fraction v4: {pd['near_zero_fraction_v4']:.6g}")
        lines.append(f"- P99 baseline: {pd['p99_baseline']:.6g}")
        lines.append(f"- P99 v4: {pd['p99_v4']:.6g}")
        lines.append(f"- P99 target: {pd['p99_target']:.6g}")
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)

    baseline_config = get_config(args.baseline_config)
    v4_config = get_config(args.v4_config)
    predictor_file, target_file = _resolve_dataset_paths(baseline_config, args.predictor_file, args.target_file)

    baseline_model = _load_model(baseline_config, args.checkpoint, device)
    v4_model = _load_model(v4_config, args.checkpoint, device)

    baseline_pred, targets = _run_inference(
        config=baseline_config,
        model=baseline_model,
        device=device,
        predictor_file=predictor_file,
        target_file=target_file,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        use_mitigation_override=(True if args.baseline_use_mitigation else False),
    )
    v4_pred, v4_targets = _run_inference(
        config=v4_config,
        model=v4_model,
        device=device,
        predictor_file=predictor_file,
        target_file=target_file,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        use_mitigation_override=None,
    )

    if baseline_pred.shape != v4_pred.shape:
        raise RuntimeError(
            f"Baseline and v4 predictions shape mismatch: {baseline_pred.shape} vs {v4_pred.shape}"
        )
    if targets.shape != v4_targets.shape:
        raise RuntimeError(
            f"Target shape mismatch between runs: {targets.shape} vs {v4_targets.shape}"
        )

    output_vars = list(getattr(baseline_config.data, "output_vars", []))
    if baseline_pred.shape[1] != len(output_vars):
        output_vars = [f"var_{idx}" for idx in range(baseline_pred.shape[1])]

    mask_unit = tuple(getattr(baseline_config, "mask_unit_size", [16, 16]))
    step_h = int(mask_unit[0]) if len(mask_unit) > 0 else 16
    step_w = int(mask_unit[1]) if len(mask_unit) > 1 else 16

    summary: dict[str, Any] = {
        "sample_count": int(baseline_pred.shape[0]),
        "predictor_file": predictor_file,
        "target_file": target_file,
        "baseline_config": str(Path(args.baseline_config).resolve()),
        "v4_config": str(Path(args.v4_config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "variables": {},
    }

    for idx, var_name in enumerate(output_vars):
        p_base = baseline_pred[:, idx]
        p_v4 = v4_pred[:, idx]
        t = targets[:, idx]
        seam_base = _seam_metric(p_base, step_h, step_w)
        seam_v4 = _seam_metric(p_v4, step_h, step_w)
        summary["variables"][var_name] = {
            "rmse_baseline": _rmse(p_base, t),
            "rmse_v4": _rmse(p_v4, t),
            "rmse_delta": _rmse(p_v4, t) - _rmse(p_base, t),
            "seam_ratio_baseline": seam_base["seam_to_interior_ratio"],
            "seam_ratio_v4": seam_v4["seam_to_interior_ratio"],
            "seam_abs_baseline": seam_base["seam_abs_mean"],
            "seam_abs_v4": seam_v4["seam_abs_mean"],
            "min_pred_baseline": float(np.nanmin(p_base)),
            "min_pred_v4": float(np.nanmin(p_v4)),
        }

    if "pr" in output_vars:
        pr_idx = output_vars.index("pr")
        base_pr = _precip_distribution_metrics(baseline_pred[:, pr_idx], targets[:, pr_idx])
        v4_pr = _precip_distribution_metrics(v4_pred[:, pr_idx], targets[:, pr_idx])
        summary["precip_distribution"] = {
            "zero_fraction_baseline": base_pr.get("zero_fraction_pred"),
            "zero_fraction_v4": v4_pr.get("zero_fraction_pred"),
            "zero_fraction_target": base_pr.get("zero_fraction_target"),
            "near_zero_fraction_baseline": base_pr.get("near_zero_fraction_pred"),
            "near_zero_fraction_v4": v4_pr.get("near_zero_fraction_pred"),
            "near_zero_fraction_target": base_pr.get("near_zero_fraction_target"),
            "p95_baseline": base_pr.get("p95_pred"),
            "p95_v4": v4_pr.get("p95_pred"),
            "p95_target": base_pr.get("p95_target"),
            "p99_baseline": base_pr.get("p99_pred"),
            "p99_v4": v4_pr.get("p99_pred"),
            "p99_target": base_pr.get("p99_target"),
            "quantile_mae_baseline": base_pr.get("quantile_mae"),
            "quantile_mae_v4": v4_pr.get("quantile_mae"),
        }

    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    output_md = output_json.with_suffix(".md")
    output_md.write_text(_to_markdown(summary), encoding="utf-8")

    print(f"Saved evaluation summary JSON: {output_json}")
    print(f"Saved evaluation summary Markdown: {output_md}")
    for var_name, metrics in summary["variables"].items():
        print(
            f"{var_name}: rmse_baseline={metrics['rmse_baseline']:.6g}, "
            f"rmse_v4={metrics['rmse_v4']:.6g}, "
            f"seam_ratio_baseline={metrics['seam_ratio_baseline']:.6g}, "
            f"seam_ratio_v4={metrics['seam_ratio_v4']:.6g}"
        )


if __name__ == "__main__":
    main()
