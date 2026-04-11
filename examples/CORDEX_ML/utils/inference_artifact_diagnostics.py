#!/usr/bin/env python
"""Diagnose CORDEX block artifacts by comparing reconstruction modes.

This utility runs inference on the same samples using:
1) full-frame / no blending
2) overlap-window blending

It then reports seam-aligned discontinuity metrics and writes difference maps.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_DIR = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]
for path in (REPO_ROOT, PROJECT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.chdir(PROJECT_DIR)

from cordex_inference import CordexWrappedDataset, build_inference_dataset  # noqa: E402
from utils.inference_blending import (  # noqa: E402
    BoundaryMitigationConfig,
    DeblockConfig,
    infer_batch_with_boundary_mitigation,
    resolve_boundary_mitigation_settings,
)
from granitewxc.models.model import get_finetune_model_UNET  # noqa: E402
from granitewxc.utils.config import get_config  # noqa: E402


def _resolve_globbed(paths: Sequence[str]) -> list[str]:
    resolved: list[str] = []
    for item in paths:
        path = Path(item)
        if any(ch in str(path) for ch in "*?[]"):
            resolved.extend(str(p.resolve()) for p in sorted(path.parent.glob(path.name)))
        else:
            resolved.append(str(path.resolve()))
    return resolved


def _default_eval_paths(cfg) -> tuple[list[str], list[str]]:
    data_cfg = getattr(cfg, "data", object())
    candidates = [
        ("test_predictor_paths", "test_target_paths"),
        ("validation_predictor_paths", "validation_target_paths"),
        ("training_predictor_paths", "training_target_paths"),
    ]
    for pred_key, tgt_key in candidates:
        pred = list(getattr(data_cfg, pred_key, []) or [])
        tgt = list(getattr(data_cfg, tgt_key, []) or [])
        if pred and tgt:
            return pred, tgt
    raise ValueError(
        "No predictor/target paths found in config. Provide --predictors/--targets explicitly."
    )


def _load_model(config_path: Path, checkpoint_path: Path, device: torch.device) -> tuple[object, torch.nn.Module]:
    cfg = get_config(str(config_path))
    model = get_finetune_model_UNET(cfg)

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    model_state = model.state_dict()
    weights_have_module_prefix = all(key.startswith("module.") for key in state_dict.keys())
    model_expects_module_prefix = all(key.startswith("module.") for key in model_state.keys())
    if model_expects_module_prefix and not weights_have_module_prefix:
        state_dict = state_dict.__class__(("module." + key, value) for key, value in state_dict.items())
    elif weights_have_module_prefix and not model_expects_module_prefix:
        prefix_len = len("module.")
        state_dict = state_dict.__class__((key[prefix_len:], value) for key, value in state_dict.items())

    scaler_key_parts = (
        "input_scalers_",
        "output_scalers_",
        "static_input_scalers_",
        "static_output_scalers_",
    )
    state_dict = state_dict.__class__(
        (key, value)
        for key, value in state_dict.items()
        if not any(part in key for part in scaler_key_parts)
    )
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = [
        key
        for key in incompatible.missing_keys
        if not any(part in key for part in scaler_key_parts)
    ]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint/model mismatch after filtering scaler tensors. "
            f"missing={missing[:8]}, unexpected={incompatible.unexpected_keys[:8]}"
        )

    model.to(device)
    model.eval()
    return cfg, model


def _build_loader(cfg, predictor_paths: list[str], target_paths: list[str], batch_size: int, device: torch.device):
    dataset = build_inference_dataset(cfg, predictor_paths, target_paths)
    wrapped = CordexWrappedDataset(dataset)
    return DataLoader(
        wrapped,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    ), dataset


def _compute_starts(size: int, tile: int, overlap: int) -> list[int]:
    if tile >= size:
        return [0]
    stride = max(1, tile - overlap)
    starts = list(range(0, max(size - tile + 1, 1), stride))
    last = size - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def _seam_metrics(pred: np.ndarray, tile_h: int, tile_w: int, overlap_h: int, overlap_w: int) -> dict[str, float]:
    # pred shape: [N, C, H, W]
    h = pred.shape[-2]
    w = pred.shape[-1]
    y_starts = _compute_starts(h, min(tile_h, h), min(overlap_h, max(0, tile_h - 1)))
    x_starts = _compute_starts(w, min(tile_w, w), min(overlap_w, max(0, tile_w - 1)))

    y_bounds = [idx for idx in y_starts[1:] if idx > 0]
    x_bounds = [idx for idx in x_starts[1:] if idx > 0]

    dx = np.abs(pred[..., :, 1:] - pred[..., :, :-1])
    dy = np.abs(pred[..., 1:, :] - pred[..., :-1, :])

    seam_x_vals = np.concatenate([dx[..., idx - 1 : idx] for idx in x_bounds], axis=-1) if x_bounds else np.array([])
    seam_y_vals = np.concatenate([dy[..., idx - 1 : idx, :] for idx in y_bounds], axis=-2) if y_bounds else np.array([])

    non_seam_x_mask = np.ones(dx.shape[-1], dtype=bool)
    for idx in x_bounds:
        non_seam_x_mask[idx - 1] = False
    non_seam_y_mask = np.ones(dy.shape[-2], dtype=bool)
    for idx in y_bounds:
        non_seam_y_mask[idx - 1] = False

    non_seam_x = dx[..., non_seam_x_mask] if non_seam_x_mask.any() else np.array([])
    non_seam_y = dy[..., non_seam_y_mask, :] if non_seam_y_mask.any() else np.array([])

    seam_mean = float(np.mean(np.concatenate([seam_x_vals.reshape(-1), seam_y_vals.reshape(-1)]))) if (seam_x_vals.size or seam_y_vals.size) else 0.0
    non_seam_mean = float(np.mean(np.concatenate([non_seam_x.reshape(-1), non_seam_y.reshape(-1)]))) if (non_seam_x.size or non_seam_y.size) else 0.0

    return {
        "num_x_boundaries": len(x_bounds),
        "num_y_boundaries": len(y_bounds),
        "seam_gradient_mean": seam_mean,
        "non_seam_gradient_mean": non_seam_mean,
        "seam_to_non_seam_ratio": float(seam_mean / max(non_seam_mean, 1e-8)),
    }


def _run_inference(model: torch.nn.Module, loader: DataLoader, cfg: BoundaryMitigationConfig, device: torch.device, max_samples: int):
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device=device, dtype=torch.float32, non_blocking=True) for k, v in batch.items()}
            out, _, _ = infer_batch_with_boundary_mitigation(model=model, batch=batch, cfg=cfg)
            preds.append(out.cpu())
            targets.append(batch["y"].cpu())
            if sum(t.shape[0] for t in preds) >= max_samples:
                break

    pred = torch.cat(preds, dim=0)[:max_samples].numpy()
    target = torch.cat(targets, dim=0)[:max_samples].numpy()
    return pred, target


def _plot_comparison(
    out_dir: Path,
    target_vars: list[str],
    pred_no_overlap: np.ndarray,
    pred_overlap: np.ndarray,
    target: np.ndarray,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_idx = 0
    for var_idx, var_name in enumerate(target_vars):
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        vmin = min(
            float(target[sample_idx, var_idx].min()),
            float(pred_no_overlap[sample_idx, var_idx].min()),
            float(pred_overlap[sample_idx, var_idx].min()),
        )
        vmax = max(
            float(target[sample_idx, var_idx].max()),
            float(pred_no_overlap[sample_idx, var_idx].max()),
            float(pred_overlap[sample_idx, var_idx].max()),
        )

        im0 = axes[0].imshow(target[sample_idx, var_idx], cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0].set_title(f"target ({var_name})")
        axes[1].imshow(pred_no_overlap[sample_idx, var_idx], cmap="viridis", vmin=vmin, vmax=vmax)
        axes[1].set_title("no-overlap/full")
        axes[2].imshow(pred_overlap[sample_idx, var_idx], cmap="viridis", vmin=vmin, vmax=vmax)
        axes[2].set_title("overlap+blend")
        diff = pred_overlap[sample_idx, var_idx] - pred_no_overlap[sample_idx, var_idx]
        vmax_diff = max(1e-6, float(np.abs(diff).max()))
        axes[3].imshow(diff, cmap="coolwarm", vmin=-vmax_diff, vmax=vmax_diff)
        axes[3].set_title("overlap - no-overlap")
        for ax in axes:
            ax.axis("off")
        fig.colorbar(im0, ax=axes[:3], fraction=0.025, pad=0.02)
        fig.tight_layout()
        fig.savefig(out_dir / f"artifact_compare_{var_name}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose CORDEX stitching/block artifacts.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--predictors", nargs="+", default=None)
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument("--out-dir", "--output-dir", dest="out_dir", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tile-size", nargs=2, type=int, default=None)
    parser.add_argument("--overlap", nargs=2, type=int, default=None)
    parser.add_argument("--blend-window", type=str, default=None)
    parser.add_argument("--blend-sigma", type=float, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, model = _load_model(args.config.resolve(), args.checkpoint.resolve(), device)
    if args.predictors is None or args.targets is None:
        default_predictors, default_targets = _default_eval_paths(cfg)
        predictors = _resolve_globbed(default_predictors)
        targets = _resolve_globbed(default_targets)
    else:
        predictors = _resolve_globbed(args.predictors)
        targets = _resolve_globbed(args.targets)
    if len(predictors) != len(targets):
        raise ValueError("predictors and targets lists must have equal length")
    loader, dataset = _build_loader(cfg, predictors, targets, batch_size=args.batch_size, device=device)

    target_vars = list(dataset.target_vars)

    base_cfg = resolve_boundary_mitigation_settings(cfg)
    no_overlap_cfg = BoundaryMitigationConfig(
        enabled=False,
        force_full_frame=True,
        tile_size=base_cfg.tile_size,
        overlap=(0, 0),
        blend_window=base_cfg.blend_window,
        blend_sigma=base_cfg.blend_sigma,
        deblock=DeblockConfig(enabled=False),
    )

    overlap_cfg = replace(base_cfg)
    overlap_cfg.enabled = True
    overlap_cfg.force_full_frame = False
    overlap_cfg.deblock = DeblockConfig(enabled=False)

    if args.tile_size is not None:
        overlap_cfg.tile_size = (int(args.tile_size[0]), int(args.tile_size[1]))
    if args.overlap is not None:
        overlap_cfg.overlap = (int(args.overlap[0]), int(args.overlap[1]))
    if args.blend_window is not None:
        overlap_cfg.blend_window = str(args.blend_window).lower()
    if args.blend_sigma is not None:
        overlap_cfg.blend_sigma = float(args.blend_sigma)

    if overlap_cfg.tile_size is None:
        overlap_cfg.tile_size = dataset.fine_shape

    tile_h, tile_w = overlap_cfg.tile_size
    if tile_h >= dataset.fine_shape[0] and tile_w >= dataset.fine_shape[1]:
        tile_h = max(8, dataset.fine_shape[0] // 2)
        tile_w = max(8, dataset.fine_shape[1] // 2)
        overlap_cfg.tile_size = (tile_h, tile_w)

    if overlap_cfg.overlap == (0, 0):
        overlap_cfg.overlap = (max(1, tile_h // 4), max(1, tile_w // 4))

    pred_no_overlap, target = _run_inference(
        model=model,
        loader=loader,
        cfg=no_overlap_cfg,
        device=device,
        max_samples=args.max_samples,
    )
    pred_overlap, _ = _run_inference(
        model=model,
        loader=loader,
        cfg=overlap_cfg,
        device=device,
        max_samples=args.max_samples,
    )

    seam_no_overlap = _seam_metrics(
        pred_no_overlap,
        tile_h=overlap_cfg.tile_size[0],
        tile_w=overlap_cfg.tile_size[1],
        overlap_h=0,
        overlap_w=0,
    )
    seam_overlap = _seam_metrics(
        pred_overlap,
        tile_h=overlap_cfg.tile_size[0],
        tile_w=overlap_cfg.tile_size[1],
        overlap_h=overlap_cfg.overlap[0],
        overlap_w=overlap_cfg.overlap[1],
    )

    rmse_no_overlap = float(np.sqrt(np.mean((pred_no_overlap - target) ** 2)))
    rmse_overlap = float(np.sqrt(np.mean((pred_overlap - target) ** 2)))

    payload = {
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "predictors": predictors,
        "targets": targets,
        "target_vars": target_vars,
        "num_samples": int(pred_no_overlap.shape[0]),
        "comparison": {
            "rmse_no_overlap": rmse_no_overlap,
            "rmse_overlap": rmse_overlap,
            "seam_no_overlap": seam_no_overlap,
            "seam_overlap": seam_overlap,
        },
        "overlap_cfg": {
            "enabled": overlap_cfg.enabled,
            "tile_size": list(overlap_cfg.tile_size) if overlap_cfg.tile_size else None,
            "overlap": list(overlap_cfg.overlap),
            "blend_window": overlap_cfg.blend_window,
            "blend_sigma": overlap_cfg.blend_sigma,
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "artifact_diagnostics.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    _plot_comparison(
        out_dir=args.out_dir,
        target_vars=target_vars,
        pred_no_overlap=pred_no_overlap,
        pred_overlap=pred_overlap,
        target=target,
    )

    print(json.dumps(payload["comparison"], indent=2))
    print(f"Saved diagnostics under: {args.out_dir}")


if __name__ == "__main__":
    main()
