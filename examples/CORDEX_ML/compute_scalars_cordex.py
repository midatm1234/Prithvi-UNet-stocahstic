"""Compute channel-wise scalars for CORDEX downscaling datasets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Sequence, Tuple

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
        default=None,
        help=(
            "Directory where .npy scalars and metadata.json are saved. "
            "Defaults to the config's case-scoped scalars directory."
        ),
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
    targets_grid_mean_raw: np.ndarray,
    targets_grid_std_raw: np.ndarray,
    targets_grid_log_mean: np.ndarray,
    targets_grid_log_std: np.ndarray,
    divide_only_samples: dict[int, list[np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    use_grid = any(spec.scaling.mode == "gridpoint" for spec in specs)
    if use_grid:
        target_mu = np.zeros_like(targets_grid_mean_raw, dtype=np.float32)
        target_sigma = np.zeros_like(targets_grid_std_raw, dtype=np.float32)
    else:
        target_mu = targets_mean_raw.astype(np.float32, copy=True)
        target_sigma = targets_std_raw.astype(np.float32, copy=True)

    quantile_map = {"p90": 0.90, "p95": 0.95, "p99": 0.99}
    for ch_idx, spec in enumerate(specs):
        scaling_method = spec.scaling.method
        mode = spec.scaling.mode
        eps_std = max(float(spec.scaling.eps_std), EPS)

        if mode == "gridpoint":
            raw_mean = targets_grid_mean_raw[ch_idx]
            raw_std = targets_grid_std_raw[ch_idx]
            log_mean = targets_grid_log_mean[ch_idx]
            log_std = targets_grid_log_std[ch_idx]
        else:
            raw_mean = float(targets_mean_raw[ch_idx])
            raw_std = float(targets_std_raw[ch_idx])
            log_mean = float(targets_log_mean[ch_idx])
            log_std = float(targets_log_std[ch_idx])

        if scaling_method == "zscore":
            if mode == "gridpoint":
                target_mu[ch_idx] = raw_mean.astype(np.float32, copy=False)
                target_sigma[ch_idx] = np.maximum(raw_std, eps_std).astype(np.float32, copy=False)
            else:
                target_mu[ch_idx] = np.float32(raw_mean)
                target_sigma[ch_idx] = np.float32(max(float(raw_std), eps_std))
                if use_grid:
                    target_mu[ch_idx, ...] = np.float32(raw_mean)
                    target_sigma[ch_idx, ...] = np.float32(max(float(raw_std), eps_std))
            continue

        if scaling_method in {"log1p_standardize", "log1p_zscore"}:
            if mode == "gridpoint":
                target_mu[ch_idx] = log_mean.astype(np.float32, copy=False)
                target_sigma[ch_idx] = np.maximum(log_std, eps_std).astype(np.float32, copy=False)
            else:
                target_mu[ch_idx] = np.float32(log_mean)
                target_sigma[ch_idx] = np.float32(max(float(log_std), eps_std))
                if use_grid:
                    target_mu[ch_idx, ...] = np.float32(log_mean)
                    target_sigma[ch_idx, ...] = np.float32(max(float(log_std), eps_std))
            continue

        if scaling_method != "divide_only":
            raise ValueError(
                f"Unsupported scaling method '{scaling_method}' for {target_vars[ch_idx]}"
            )

        if mode == "gridpoint":
            target_mu[ch_idx] = 0.0
        else:
            target_mu[ch_idx] = np.float32(0.0)
            if use_grid:
                target_mu[ch_idx, ...] = np.float32(0.0)

        stat = spec.scaling.scale_stat
        if mode == "gridpoint" and stat in quantile_map:
            # Quantile-based divide_only per-gridpoint is noisy with finite time samples;
            # use local mean as the conservative fallback.
            print(
                f"[predictands] {target_vars[ch_idx]}: divide_only+{stat} with mode=gridpoint "
                "falls back to gridpoint mean scaling."
            )
            stat = "mean"

        if stat == "fixed":
            if spec.scaling.fixed_scale is None or spec.scaling.fixed_scale <= 0:
                raise ValueError(
                    f"predictands.{spec.name}.scaling.scale_stat=fixed requires positive fixed_scale."
                )
            scale = float(spec.scaling.fixed_scale)
            if mode == "gridpoint":
                scale_field = np.full_like(targets_grid_std_raw[ch_idx], fill_value=scale, dtype=np.float32)
        elif stat == "mean":
            if mode == "gridpoint":
                scale_field = np.asarray(raw_mean, dtype=np.float32)
            else:
                scale = float(raw_mean)
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

        if mode == "gridpoint":
            target_sigma[ch_idx] = np.maximum(scale_field, eps_std).astype(np.float32, copy=False)
        else:
            target_sigma[ch_idx] = np.float32(max(scale, eps_std))
            if use_grid:
                target_sigma[ch_idx, ...] = np.float32(max(scale, eps_std))

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
    y_grid_sum = y_grid_sumsq = None
    y_grid_log_sum = y_grid_log_sumsq = None
    x_count = 0
    y_count = 0
    y_grid_count = 0

    rng = np.random.default_rng(42)
    divide_only_quantiles = {"p90", "p95", "p99"}
    log1p_channels = {
        idx
        for idx, spec in enumerate(target_specs)
        if spec.scaling.method in {"log1p_standardize", "log1p_zscore"}
    }
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
            y_log = torch.zeros_like(y_flat)
            y_log_grid = torch.zeros_like(y)
            if log1p_channels:
                log_indices = sorted(log1p_channels)
                y_log[log_indices, :] = torch.log1p(y_nonnegative[log_indices, :])
                y_log_grid[log_indices, ...] = y_log[log_indices, :].view_as(y[log_indices, ...])

            if x_sum is None:
                x_sum, x_sumsq = _init_accumulator(x.shape[0])
            if y_sum is None:
                y_sum, y_sumsq = _init_accumulator(y.shape[0])
                y_log_sum, y_log_sumsq = _init_accumulator(y.shape[0])
            if y_grid_sum is None:
                y_grid_sum = torch.zeros_like(y, dtype=torch.float64)
                y_grid_sumsq = torch.zeros_like(y, dtype=torch.float64)
                y_grid_log_sum = torch.zeros_like(y, dtype=torch.float64)
                y_grid_log_sumsq = torch.zeros_like(y, dtype=torch.float64)

            x_sum += x_flat.sum(dim=1)
            x_sumsq += (x_flat.pow(2)).sum(dim=1)
            y_sum += y_flat.sum(dim=1)
            y_sumsq += (y_flat.pow(2)).sum(dim=1)
            y_log_sum += y_log.sum(dim=1)
            y_log_sumsq += (y_log.pow(2)).sum(dim=1)
            y_grid_sum += y
            y_grid_sumsq += y.pow(2)
            y_grid_log_sum += y_log_grid
            y_grid_log_sumsq += y_log_grid.pow(2)

            x_count += x_flat.shape[1]
            y_count += y_flat.shape[1]
            y_grid_count += 1

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
    if y_grid_sum is None or y_grid_sumsq is None or y_grid_log_sum is None or y_grid_log_sumsq is None:
        raise RuntimeError("No target grids observed when computing per-gridpoint scalars")
    if y_grid_count <= 0:
        raise RuntimeError("Invalid target grid sample count")

    targets_grid_mean_raw = (y_grid_sum / y_grid_count).cpu().numpy().astype(np.float32)
    targets_grid_var_raw = (y_grid_sumsq / y_grid_count) - (y_grid_sum / y_grid_count).pow(2)
    targets_grid_std_raw = torch.sqrt(torch.clamp(targets_grid_var_raw, min=0.0)).cpu().numpy().astype(np.float32)

    targets_grid_log_mean = (y_grid_log_sum / y_grid_count).cpu().numpy().astype(np.float32)
    targets_grid_log_var = (y_grid_log_sumsq / y_grid_count) - (y_grid_log_sum / y_grid_count).pow(2)
    targets_grid_log_std = torch.sqrt(torch.clamp(targets_grid_log_var, min=0.0)).cpu().numpy().astype(np.float32)

    targets_mean, targets_std = _resolve_scalars_from_specs(
        specs=target_specs,
        target_vars=target_vars,
        targets_mean_raw=targets_mean_raw,
        targets_std_raw=targets_std_raw,
        targets_log_mean=targets_log_mean,
        targets_log_std=targets_log_std,
        targets_grid_mean_raw=targets_grid_mean_raw,
        targets_grid_std_raw=targets_grid_std_raw,
        targets_grid_log_mean=targets_grid_log_mean,
        targets_grid_log_std=targets_grid_log_std,
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
        "targets_grid_mean_raw": targets_grid_mean_raw,
        "targets_grid_std_raw": targets_grid_std_raw,
        "targets_grid_log_mean": targets_grid_log_mean,
        "targets_grid_log_std": targets_grid_log_std,
        "input_pixel_count": x_count,
        "target_pixel_count": y_count,
        "target_grid_sample_count": y_grid_count,
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


def _canonical_path_list(values: Any) -> list[str]:
    """Resolve config and command-line paths for exact source comparisons."""

    if values in (None, ""):
        return []
    if isinstance(values, (str, Path)):
        values = [values]

    resolved: list[str] = []
    for value in values:
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        resolved.append(os.path.normcase(str(path.resolve(strict=False))))
    return resolved


def _select_scalar_training_partition(
    dataset: torch.utils.data.Dataset,
    *,
    config: object,
    predictor_files: Sequence[str],
    target_files: Sequence[str],
) -> tuple[torch.utils.data.Dataset, dict[str, Any]]:
    """Exclude the configured contiguous validation tail from scalar fitting.

    When training and validation intentionally share files, the training loader
    reserves the final fraction for validation.  Scalar estimation must use the
    same leading partition so validation targets cannot leak into normalization.
    """

    source_count = len(dataset)
    data_cfg = getattr(config, "data", None)
    train_predictors = _canonical_path_list(
        getattr(data_cfg, "training_predictor_paths", None)
        if data_cfg is not None
        else None
    )
    train_targets = _canonical_path_list(
        getattr(data_cfg, "training_target_paths", None)
        if data_cfg is not None
        else None
    )
    validation_predictors = _canonical_path_list(
        getattr(data_cfg, "validation_predictor_paths", None)
        if data_cfg is not None
        else None
    )
    validation_targets = _canonical_path_list(
        getattr(data_cfg, "validation_target_paths", None)
        if data_cfg is not None
        else None
    )
    sources_identical = bool(
        train_predictors
        and train_targets
        and train_predictors == validation_predictors
        and train_targets == validation_targets
    )

    selection: dict[str, Any] = {
        "policy": "all_supplied_samples",
        "source_sample_count": int(source_count),
        "used_sample_count": int(source_count),
        "used_index_start": 0,
        "used_index_stop_exclusive": int(source_count),
        "excluded_index_start": None,
        "excluded_index_stop_exclusive": None,
        "config_training_validation_sources_identical": sources_identical,
        "validation_holdout_fraction": None,
        "validation_holdout_strategy": None,
    }

    raw_fraction = (
        getattr(data_cfg, "validation_holdout_fraction", None)
        if data_cfg is not None
        else None
    )
    if not sources_identical or raw_fraction in (None, "", 0, 0.0):
        return dataset, selection

    fraction = float(raw_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError(
            "data.validation_holdout_fraction must lie in (0, 1) when "
            f"training and validation sources are identical, got {fraction}."
        )
    strategy = str(
        getattr(data_cfg, "validation_holdout_strategy", "contiguous_tail")
    ).strip().lower()
    if strategy != "contiguous_tail":
        raise ValueError(
            "Scalar fitting only supports "
            "data.validation_holdout_strategy='contiguous_tail', got "
            f"{strategy!r}."
        )

    cli_predictors = _canonical_path_list(predictor_files)
    cli_targets = _canonical_path_list(target_files)
    if cli_predictors != train_predictors or cli_targets != train_targets:
        raise ValueError(
            "The config defines a train/validation holdout on identical files, "
            "but the --predictor-files/--target-files supplied to scalar fitting "
            "do not match config.data.training_*_paths. Refusing to apply the "
            "holdout indices to a different dataset."
        )
    if source_count < 2:
        raise ValueError("A train/validation scalar holdout requires at least two samples.")

    validation_count = max(1, int(round(source_count * fraction)))
    validation_count = min(validation_count, source_count - 1)
    split = source_count - validation_count
    subset = torch.utils.data.Subset(dataset, range(0, split))
    selection.update(
        {
            "policy": "leading_training_partition_excluding_contiguous_validation_tail",
            "used_sample_count": int(split),
            "used_index_stop_exclusive": int(split),
            "excluded_index_start": int(split),
            "excluded_index_stop_exclusive": int(source_count),
            "validation_holdout_fraction": fraction,
            "validation_holdout_strategy": strategy,
        }
    )
    return subset, selection


def _compact_array_summary(values: np.ndarray) -> dict | list:
    arr = np.asarray(values)
    if arr.ndim <= 1 and arr.size <= 64:
        return arr.tolist()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"shape": list(arr.shape), "finite_count": 0}
    return {
        "shape": list(arr.shape),
        "finite_count": int(finite.size),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
    }


def _resolve_output_dir(args: argparse.Namespace, config: object) -> str:
    """Resolve explicit, case-scoped, then legacy scalar output locations."""

    if getattr(args, "output_dir", None):
        return str(args.output_dir)
    if config is not None and not bool(
        getattr(config, "derive_output_paths", False)
    ):
        data_scalers = getattr(getattr(config, "data", None), "scalers", {})
        if hasattr(data_scalers, "__dict__"):
            data_scalers = data_scalers.__dict__
        if not isinstance(data_scalers, dict):
            data_scalers = {}
        model = getattr(config, "model", None)
        model_names = {
            "inputs_mean": "input_mu",
            "inputs_std": "input_sigma",
            "targets_mean": "target_mu",
            "targets_std": "target_sigma",
        }
        configured: list[str] = []
        for key, model_name in model_names.items():
            data_path = data_scalers.get(key)
            model_path = getattr(model, model_name, None) if model is not None else None
            if data_path and model_path and os.path.normpath(str(data_path)) != os.path.normpath(str(model_path)):
                raise ValueError(
                    f"Inconsistent configured scalar path for {key}: "
                    f"data.scalers={data_path!r}, model.{model_name}={model_path!r}."
                )
            selected = data_path or model_path
            if selected:
                configured.append(str(selected))
        if configured:
            parents = {os.path.normpath(os.path.dirname(path)) for path in configured}
            if len(parents) != 1:
                raise ValueError(
                    "Configured scalar files do not share one output directory: "
                    f"{sorted(parents)}."
                )
            return parents.pop()
    try:
        path_scalars = getattr(config, "path_scalars", None)
    except (AttributeError, ValueError):
        path_scalars = None
    if path_scalars:
        return str(path_scalars)
    return "./experiments/cordex_scalars"


def main() -> None:
    args = parse_args()

    config, target_vars, target_specs = _resolve_config_and_predictands(args)
    if args.config:
        print(f"[predictands] using config: {os.path.abspath(args.config)}")
    else:
        print(
            "[predictands] no --config provided; using defaults "
            "(pr -> divide_only + p95, others -> zscore)."
        )

    args.output_dir = _resolve_output_dir(args, config)

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

    source_dataset = CordexDownscaleDataset(
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

    dataset, sample_selection = _select_scalar_training_partition(
        source_dataset,
        config=config,
        predictor_files=args.predictor_files,
        target_files=args.target_files,
    )
    print(
        f"Loaded {len(source_dataset)} source samples from "
        f"{len(args.predictor_files)} predictor files"
    )
    if len(dataset) != len(source_dataset):
        print(
            "[holdout] fitting scalars only on training indices "
            f"[{sample_selection['used_index_start']}, "
            f"{sample_selection['used_index_stop_exclusive']}); excluded "
            f"[{sample_selection['excluded_index_start']}, "
            f"{sample_selection['excluded_index_stop_exclusive']}) for validation."
        )
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
        "source_num_samples": len(source_dataset),
        "used_num_samples": len(dataset),
        "sample_selection": sample_selection,
        "input_channels": int(stats["inputs_mean"].shape[0]),
        "target_channels": int(stats["targets_mean"].shape[0]),
        "input_pixel_count": stats["input_pixel_count"],
        "target_pixel_count": stats["target_pixel_count"],
        "target_grid_sample_count": stats["target_grid_sample_count"],
        "summary": {
            "targets_mean_raw": _compact_array_summary(stats["targets_mean_raw"]),
            "targets_std_raw": _compact_array_summary(stats["targets_std_raw"]),
            "targets_log_mean": _compact_array_summary(stats["targets_log_mean"]),
            "targets_log_std": _compact_array_summary(stats["targets_log_std"]),
            "targets_grid_mean_raw": _compact_array_summary(stats["targets_grid_mean_raw"]),
            "targets_grid_std_raw": _compact_array_summary(stats["targets_grid_std_raw"]),
            "targets_grid_log_mean": _compact_array_summary(stats["targets_grid_log_mean"]),
            "targets_grid_log_std": _compact_array_summary(stats["targets_grid_log_std"]),
            "targets_mean_saved": _compact_array_summary(stats["targets_mean"]),
            "targets_std_saved": _compact_array_summary(stats["targets_std"]),
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
