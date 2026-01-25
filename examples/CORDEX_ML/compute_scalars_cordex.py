"""Compute channel-wise scalars for CORDEX downscaling datasets."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import torch

from cordex_dataset import CordexDownscaleDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-channel mean/std for CORDEX predictors and targets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        default=["pr", "tasmax"],
        help="Target variables to use when computing stats",
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
    return parser.parse_args()


def _init_accumulator(num_channels: int) -> Tuple[torch.Tensor, torch.Tensor]:
    zeros = torch.zeros(num_channels, dtype=torch.float64)
    return zeros.clone(), zeros


def _finalize(sum_tensor: torch.Tensor, sumsq_tensor: torch.Tensor, count: int) -> Tuple[np.ndarray, np.ndarray]:
    if count == 0:
        raise RuntimeError("No pixels observed when computing scalars")

    mean = sum_tensor / count
    variance = sumsq_tensor / count - mean.pow(2)
    variance = torch.clamp(variance, min=0.0)
    std = torch.sqrt(variance)

    return mean.cpu().numpy().astype(np.float32), std.cpu().numpy().astype(np.float32)


def compute_scalars(dataset: CordexDownscaleDataset, progress_interval: int) -> Dict[str, np.ndarray]:
    x_sum = x_sumsq = None
    y_sum = y_sumsq = None
    x_count = 0
    y_count = 0

    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            x = sample["x"].to(dtype=torch.float64)
            y = sample["y"].to(dtype=torch.float64)

            x_flat = x.view(x.shape[0], -1)
            y_flat = y.view(y.shape[0], -1)

            if x_sum is None:
                x_sum, x_sumsq = _init_accumulator(x.shape[0])
            if y_sum is None:
                y_sum, y_sumsq = _init_accumulator(y.shape[0])

            x_sum += x_flat.sum(dim=1)
            x_sumsq += (x_flat.pow(2)).sum(dim=1)
            y_sum += y_flat.sum(dim=1)
            y_sumsq += (y_flat.pow(2)).sum(dim=1)

            x_count += x_flat.shape[1]
            y_count += y_flat.shape[1]

            if progress_interval > 0 and (idx + 1) % progress_interval == 0:
                print(f"Processed {idx + 1}/{len(dataset)} samples")

    inputs_mean, inputs_std = _finalize(x_sum, x_sumsq, x_count)
    targets_mean, targets_std = _finalize(y_sum, y_sumsq, y_count)

    return {
        "inputs_mean": inputs_mean,
        "inputs_std": inputs_std,
        "targets_mean": targets_mean,
        "targets_std": targets_std,
        "input_pixel_count": x_count,
        "target_pixel_count": y_count,
    }


def main() -> None:
    args = parse_args()

    dtype = getattr(torch, args.dtype)
    use_static = bool(args.use_static) and not bool(args.no_static)
    if use_static and not args.orography_file:
        raise SystemExit("--orography-file is required when --use-static is set")
    dataset = CordexDownscaleDataset(
        predictor_files=args.predictor_files,
        target_files=args.target_files,
        orography_file=args.orography_file if use_static else None,
        predictor_variables=args.predictor_vars,
        target_variables=args.target_vars,
        orography_variable=args.orography_var,
        time_dim=args.time_dim,
        regrid_method=args.regrid_method,
        dtype=dtype,
        use_static=use_static,
    )

    print(f"Loaded {len(dataset)} samples from {len(args.predictor_files)} predictor files")

    stats = compute_scalars(dataset, args.progress_interval)

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
            "predictor_files": args.predictor_files,
            "target_files": args.target_files,
            "orography_file": args.orography_file,
            "predictor_vars": args.predictor_vars,
            "target_vars": args.target_vars,
            "orography_var": args.orography_var,
            "time_dim": args.time_dim,
            "regrid_method": args.regrid_method,
            "dtype": args.dtype,
            "progress_interval": args.progress_interval,
        },
        "num_samples": len(dataset),
        "input_channels": int(stats["inputs_mean"].shape[0]),
        "target_channels": int(stats["targets_mean"].shape[0]),
        "input_pixel_count": stats["input_pixel_count"],
        "target_pixel_count": stats["target_pixel_count"],
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
