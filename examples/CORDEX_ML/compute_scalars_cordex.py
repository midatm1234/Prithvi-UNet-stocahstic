"""Compute channel-wise scalars for CORDEX downscaling datasets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Sequence, Tuple

import numpy as np
import torch

from cordex_dataset import CordexDownscaleDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from granitewxc.utils.config import get_config
from granitewxc.utils.predictands import PredictandSpec, build_predictand_specs


EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-channel target/input scalars for CORDEX training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config. When provided, predictand scaling is read from predictands: block.",
    )
    parser.add_argument(
        "--predictor-files",
        nargs="+",
        required=True,
        help="List of coarse-resolution predictor NetCDF files",
    )
    parser.add_argument(
        "--target-files",
        nargs="+",
        required=True,
        help="List of high-resolution target NetCDF files",
    )
    parser.add_argument(
        "--orography-file",
        required=False,
        help="Static orography NetCDF file on the fine grid or coarse grid",
    )
    parser.add_argument(
        "--predictor-vars",
        nargs="+",
        default=None,
        help="Subset of predictor variables to read (defaults to canonical set)",
    )
    parser.add_argument(
        "--target-vars",
        nargs="+",
        default=None,
        help="Target variable ordering. Defaults to config.data.output_vars or ['pr', 'tasmax'].",
    )
    parser.add_argument(
        "--orography-var",
        default="orog",
        help="Variable name that stores the static orography field",
    )
    parser.add_argument(
        "--use-static",
        action="store_true",
        help="Include static orography as an input channel",
    )
    parser.add_argument(
        "--no-static",
        action="store_true",
        help="Disable static orography inputs (overrides --use-static)",
    )
    parser.add_argument(
        "--time-dim",
        default=None,
        help="Optional override for the time dimension name",
    )
    parser.add_argument(
        "--regrid-method",
        default="bilinear",
        help="xESMF regridding method to use for coarse predictors",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float64"],
        help="Torch dtype used for dataset tensors",
    )
    parser.add_argument(
        "--output-dir",
        default="./experiments/cordex_scalars",
        help="Directory where .npy scalars and metadata.json are saved",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=50,
        help="Print progress every N samples (<=0 disables logging)",
    )
    parser.add_argument(
        "--target-scale-sample-size",
        type=int,
        default=2048,
        help="Per-sample random values collected for quantile-based divide_only scaling.",
    )
    return parser.parse_args()


def _init_accumulator(num_channels: int) -> Tuple[torch.Tensor, torch.Tensor]:
    zeros = torch.zeros(num_channels, dtype=torch.float64)
    return zeros.clone(), zeros


def _finalize(
    sum_tensor: torch.Tensor, sumsq_tensor: torch.Tensor, count: int
) -> Tuple[np.ndarray, np.ndarray]:
    if count == 0:
        raise RuntimeError("No pixels observed when computing scalars")

    mean = sum_tensor / count
    variance = sumsq_tensor / count - mean.pow(2)
    variance = torch.clamp(variance, min=0.0)
    std = torch.sqrt(variance)

    return mean.cpu().numpy().astype(np.float32), std.cpu().numpy().astype(np.float32)


def _resolve_scalars_from_specs(
    *,
    specs: Sequence[PredictandSpec],
    target_vars: Sequence[str],
    targets_mean_raw: np.ndarray,
    targets_std_raw: np.ndarray,
    targets_log_mean: np.ndarray,
    targets_log_std: np.ndarray,
    divide_only_samples: dict[int, list[np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    target_mu = targets_mean_raw.astype(np.float32, copy=True)
    target_sigma = targets_std_raw.astype(np.float32, copy=True)

    quantile_map = {"p90": 0.90, "p95": 0.95, "p99": 0.99}
    for ch_idx, spec in enumerate(specs):
        scaling_method = spec.scaling.method
        if scaling_method == "zscore":
            sigma = float(target_sigma[ch_idx])
            target_sigma[ch_idx] = max(sigma, EPS)
            continue

        if scaling_method == "log1p_zscore":
            target_mu[ch_idx] = float(targets_log_mean[ch_idx])
            sigma = float(targets_log_std[ch_idx])
            target_sigma[ch_idx] = max(sigma, EPS)
            continue

        if scaling_method != "divide_only":
            raise ValueError(
                f"Unsupported scaling method '{scaling_method}' for {target_vars[ch_idx]}"
            )

        target_mu[ch_idx] = 0.0
        stat = spec.scaling.scale_stat
        if stat == "fixed":
            if spec.scaling.fixed_scale is None or spec.scaling.fixed_scale <= 0:
                raise ValueError(
                    f"predictands.{spec.name}.scaling.scale_stat=fixed requires positive fixed_scale."
                )
            scale = float(spec.scaling.fixed_scale)
        elif stat == "mean":
            scale = float(targets_mean_raw[ch_idx])
        elif stat in quantile_map:
            channel_samples = divide_only_samples.get(ch_idx, [])
            if channel_samples:
                concat = np.concatenate(channel_samples).astype(np.float64, copy=False)
                concat = concat[np.isfinite(concat)]
                if concat.size > 0:
                    scale = float(np.quantile(concat, quantile_map[stat]))
                else:
                    scale = float(targets_mean_raw[ch_idx])
            else:
                scale = float(targets_mean_raw[ch_idx])
        else:
            raise ValueError(
                f"Unsupported divide_only scale_stat '{stat}' for {target_vars[ch_idx]}"
            )

        target_sigma[ch_idx] = max(scale, EPS)

    return target_mu.astype(np.float32), target_sigma.astype(np.float32)


def compute_scalars(
    dataset: CordexDownscaleDataset,
    progress_interval: int,
    target_specs: Sequence[PredictandSpec],
    target_vars: Sequence[str],
    target_scale_sample_size: int,
) -> Dict[str, np.ndarray]:
    x_sum = x_sumsq = None
    y_sum = y_sumsq = None
    y_log_sum = y_log_sumsq = None
    x_count = 0
    y_count = 0

    rng = np.random.default_rng(42)
    divide_only_quantiles = {"p90", "p95", "p99"}
    divide_only_sample_channels = {
        idx: []
        for idx, spec in enumerate(target_specs)
        if spec.scaling.method == "divide_only"
        and spec.scaling.scale_stat in divide_only_quantiles
    }

    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            x = sample["x"].to(dtype=torch.float64)
            y = sample["y"].to(dtype=torch.float64)

            x_flat = x.view(x.shape[0], -1)
            y_flat = y.view(y.shape[0], -1)
            y_nonnegative = torch.clamp(y_flat, min=0.0)
            y_log = torch.log1p(y_nonnegative)

            if x_sum is None:
                x_sum, x_sumsq = _init_accumulator(x.shape[0])
            if y_sum is None:
                y_sum, y_sumsq = _init_accumulator(y.shape[0])
                y_log_sum, y_log_sumsq = _init_accumulator(y.shape[0])

            x_sum += x_flat.sum(dim=1)
            x_sumsq += (x_flat.pow(2)).sum(dim=1)
            y_sum += y_flat.sum(dim=1)
            y_sumsq += (y_flat.pow(2)).sum(dim=1)
            y_log_sum += y_log.sum(dim=1)
            y_log_sumsq += (y_log.pow(2)).sum(dim=1)

            x_count += x_flat.shape[1]
            y_count += y_flat.shape[1]

            for ch_idx in divide_only_sample_channels:
                values = y_nonnegative[ch_idx].cpu().numpy().astype(np.float32, copy=False)
                if values.size == 0:
                    continue
                sample_count = min(target_scale_sample_size, values.size)
                if sample_count <= 0:
                    continue
                if sample_count == values.size:
                    picked = values
                else:
                    selection = rng.choice(values.size, size=sample_count, replace=False)
                    picked = values[selection]
                divide_only_sample_channels[ch_idx].append(picked)

            if progress_interval > 0 and (idx + 1) % progress_interval == 0:
                print(f"Processed {idx + 1}/{len(dataset)} samples")

    inputs_mean, inputs_std = _finalize(x_sum, x_sumsq, x_count)
    targets_mean_raw, targets_std_raw = _finalize(y_sum, y_sumsq, y_count)
    targets_log_mean, targets_log_std = _finalize(y_log_sum, y_log_sumsq, y_count)

    targets_mean, targets_std = _resolve_scalars_from_specs(
        specs=target_specs,
        target_vars=target_vars,
        targets_mean_raw=targets_mean_raw,
        targets_std_raw=targets_std_raw,
        targets_log_mean=targets_log_mean,
        targets_log_std=targets_log_std,
        divide_only_samples=divide_only_sample_channels,
    )

    return {
        "inputs_mean": inputs_mean,
        "inputs_std": inputs_std,
        "targets_mean": targets_mean,
        "targets_std": targets_std,
        "targets_mean_raw": targets_mean_raw,
        "targets_std_raw": targets_std_raw,
        "targets_log_mean": targets_log_mean,
        "targets_log_std": targets_log_std,
        "input_pixel_count": x_count,
        "target_pixel_count": y_count,
    }


def _resolve_config_and_predictands(
    args: argparse.Namespace,
) -> tuple[object | None, list[str], list[PredictandSpec]]:
    config = get_config(args.config) if args.config else None

    if args.target_vars:
        target_vars = list(args.target_vars)
    elif config is not None:
        target_vars = list(getattr(config.data, "output_vars", []))
    else:
        target_vars = ["pr", "tasmax"]

    if config is None:
        config = SimpleNamespace()
        config.data = SimpleNamespace(output_vars=target_vars)
        config.predictands = {}

    target_specs = build_predictand_specs(config, output_vars=target_vars)
    return config, target_vars, target_specs


def main() -> None:
    args = parse_args()

    config, target_vars, target_specs = _resolve_config_and_predictands(args)

    dtype = getattr(torch, args.dtype)
    if args.no_static:
        use_static = False
    elif args.use_static:
        use_static = True
    else:
        use_static = bool(getattr(getattr(config, "data", None), "use_static", False))

    orography_file = args.orography_file
    if use_static and not orography_file:
        orography_file = getattr(getattr(config, "data", None), "static_path", None)
    if use_static and not orography_file:
        raise SystemExit("--orography-file is required when static predictors are enabled")

    dataset = CordexDownscaleDataset(
        predictor_files=args.predictor_files,
        target_files=args.target_files,
        orography_file=orography_file if use_static else None,
        predictor_variables=args.predictor_vars,
        target_variables=target_vars,
        orography_variable=args.orography_var,
        time_dim=args.time_dim,
        regrid_method=args.regrid_method,
        dtype=dtype,
        use_static=use_static,
    )

    print(f"Loaded {len(dataset)} samples from {len(args.predictor_files)} predictor files")
    print("[predictands] resolved scaling:")
    for spec in target_specs:
        print(
            f"  - {spec.name}: allow_negative_value={spec.allow_negative_value}, "
            f"nonnegativity=({spec.nonnegativity.enabled}, {spec.nonnegativity.method}), "
            f"scaling=({spec.scaling.method}, {spec.scaling.scale_stat})"
        )

    stats = compute_scalars(
        dataset,
        args.progress_interval,
        target_specs=target_specs,
        target_vars=target_vars,
        target_scale_sample_size=max(1, int(args.target_scale_sample_size)),
    )

    os.makedirs(args.output_dir, exist_ok=True)

    inputs_mean_path = os.path.join(args.output_dir, "inputs_mean.npy")
    inputs_std_path = os.path.join(args.output_dir, "inputs_std.npy")
    targets_mean_path = os.path.join(args.output_dir, "targets_mean.npy")
    targets_std_path = os.path.join(args.output_dir, "targets_std.npy")

    np.save(inputs_mean_path, stats["inputs_mean"])
    np.save(inputs_std_path, stats["inputs_std"])
    np.save(targets_mean_path, stats["targets_mean"])
    np.save(targets_std_path, stats["targets_std"])

    metadata = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "args": {
            "config": args.config,
            "predictor_files": args.predictor_files,
            "target_files": args.target_files,
            "orography_file": orography_file,
            "predictor_vars": args.predictor_vars,
            "target_vars": target_vars,
            "orography_var": args.orography_var,
            "time_dim": args.time_dim,
            "regrid_method": args.regrid_method,
            "dtype": args.dtype,
            "progress_interval": args.progress_interval,
            "target_scale_sample_size": args.target_scale_sample_size,
        },
        "predictands": {spec.name: spec.to_dict() for spec in target_specs},
        "num_samples": len(dataset),
        "input_channels": int(stats["inputs_mean"].shape[0]),
        "target_channels": int(stats["targets_mean"].shape[0]),
        "input_pixel_count": stats["input_pixel_count"],
        "target_pixel_count": stats["target_pixel_count"],
        "summary": {
            "targets_mean_raw": stats["targets_mean_raw"].tolist(),
            "targets_std_raw": stats["targets_std_raw"].tolist(),
            "targets_mean_saved": stats["targets_mean"].tolist(),
            "targets_std_saved": stats["targets_std"].tolist(),
        },
        "files": {
            "inputs_mean": inputs_mean_path,
            "inputs_std": inputs_std_path,
            "targets_mean": targets_mean_path,
            "targets_std": targets_std_path,
        },
    }

    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

    print("Saved scalars to:")
    print(f"  {inputs_mean_path}")
    print(f"  {inputs_std_path}")
    print(f"  {targets_mean_path}")
    print(f"  {targets_std_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
