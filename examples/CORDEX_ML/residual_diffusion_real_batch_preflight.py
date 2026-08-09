"""One-real-sample preflight for the SA residual-diffusion training path.

This is a fail-fast integration check, not a trainer.  It constructs the same
CORDEX dataloader and fine-tune model as ``cordex_training.py``, consumes one
training sample, and executes one joint deterministic/diffusion forward pass.
The score-matching call is instrumented so the assertions inspect the tensors
actually used by production code rather than a parallel implementation.

Example (Prithvi environment)::

    python residual_diffusion_real_batch_preflight.py \
        --config SA_T2_ACCESS-CM2_static_residual_diffusion.yaml

The YAML contains repository-relative placeholder data paths used by portable
configs.  If those paths do not exist, the script resolves the same SA dataset
root used by the training notebook (``D:/CORDEX/SA_domain``).  Explicit
``--predictor``, ``--target``, and ``--static`` arguments always take priority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import xarray as xr


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from cordex_training import (  # noqa: E402
    build_dataloader,
    build_predictor_names,
    create_finetune_model,
)
from cordex_inference import (  # noqa: E402
    CordexWrappedDataset as InferenceWrappedDataset,
    build_inference_dataset,
)
from granitewxc.models.loss import build_loss_fn  # noqa: E402
from granitewxc.utils.config import get_config  # noqa: E402
import granitewxc.decoders.diffusion_head as diffusion_head_module  # noqa: E402


DEFAULT_CONFIG = PROJECT_DIR / "SA_T2_ACCESS-CM2_static_residual_diffusion.yaml"
DEFAULT_SA_ROOT = Path(r"D:\CORDEX\SA_domain")
DEFAULT_PREDICTOR_NAME = "ACCESS-CM2_1961-1980_2080-2099_regridded.nc"
DEFAULT_TARGET_NAME = "pr_tasmax_ACCESS-CM2_1961-1980_2080-2099.nc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--predictor", type=Path, default=None)
    parser.add_argument("--target", type=Path, default=None)
    parser.add_argument("--static", type=Path, default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument(
        "--device",
        default="cuda",
        choices=("cuda", "cpu"),
        help="CUDA is the production path and the practical default for this model.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Default: <case directory>/diagnostics/residual_real_batch_preflight.json",
    )
    return parser.parse_args()


def _repo_resolve(path_like: str | os.PathLike[str]) -> Path:
    path = Path(path_like).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _resolve_data_path(
    explicit: Path | None,
    configured: str,
    notebook_fallback: Path,
    label: str,
) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit.expanduser().resolve())
    else:
        candidates.extend((_repo_resolve(configured), notebook_fallback.resolve()))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve {label}. Checked: " + ", ".join(str(p) for p in candidates)
    )


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _summary_1d(values: torch.Tensor) -> dict[str, float | int]:
    values = values.detach().float().reshape(-1).cpu()
    if values.numel() == 0:
        raise AssertionError("Cannot summarize an empty tensor.")
    if not bool(torch.isfinite(values).all().item()):
        raise AssertionError("Stage tensor contains NaN or infinity.")
    quantiles = torch.quantile(
        values, torch.tensor([0.01, 0.05, 0.50, 0.95, 0.99])
    )
    return {
        "count": int(values.numel()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "p01": float(quantiles[0].item()),
        "p05": float(quantiles[1].item()),
        "p50": float(quantiles[2].item()),
        "p95": float(quantiles[3].item()),
        "p99": float(quantiles[4].item()),
    }


def _stage_summary(
    name: str,
    tensor: torch.Tensor,
    variable_names: list[str] | None,
) -> dict[str, Any]:
    detached = tensor.detach()
    result: dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "overall": _summary_1d(detached),
    }
    if variable_names is not None:
        if detached.ndim < 2 or detached.shape[1] != len(variable_names):
            raise AssertionError(
                f"Stage {name!r} cannot be labeled {variable_names}: "
                f"shape is {tuple(detached.shape)}."
            )
        result["by_variable"] = {
            variable: _summary_1d(detached[:, index])
            for index, variable in enumerate(variable_names)
        }

    stats = result["overall"]
    print(
        f"[stage] {name:<36} shape={tuple(detached.shape)!s:<20} "
        f"min={stats['min']:.7g} max={stats['max']:.7g} "
        f"mean={stats['mean']:.7g} std={stats['std']:.7g} "
        f"p01={stats['p01']:.7g} p05={stats['p05']:.7g} "
        f"p50={stats['p50']:.7g} p95={stats['p95']:.7g} p99={stats['p99']:.7g}"
    )
    return result


def _assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    message: str,
    *,
    rtol: float = 1e-5,
    atol: float = 2e-6,
) -> None:
    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as exc:
        raise AssertionError(message) from exc


def _find_time_name(dataset: xr.Dataset) -> str:
    names = [name for name in dataset.dims if "time" in name.lower()]
    if len(names) != 1:
        raise AssertionError(f"Expected exactly one time dimension, found {names}.")
    return names[0]


def _assert_source_alignment(
    predictor_path: Path,
    target_path: Path,
    dataset: Any,
    sample_index: int,
) -> dict[str, Any]:
    file_index, time_index = dataset._locate_index(sample_index)
    if file_index != 0:
        raise AssertionError(
            "This preflight accepts one predictor/target pair and expected file index 0."
        )

    with xr.open_dataset(predictor_path, decode_times=False) as predictor_ds, xr.open_dataset(
        target_path, decode_times=False
    ) as target_ds:
        predictor_time_name = _find_time_name(predictor_ds)
        target_time_name = _find_time_name(target_ds)
        predictor_time = predictor_ds[predictor_time_name]
        target_time = target_ds[target_time_name]
        if int(predictor_ds.sizes[predictor_time_name]) != int(
            target_ds.sizes[target_time_name]
        ):
            raise AssertionError("Predictor and target time lengths differ.")
        for attr in ("units", "calendar"):
            predictor_value = predictor_time.attrs.get(attr, "standard" if attr == "calendar" else None)
            target_value = target_time.attrs.get(attr, "standard" if attr == "calendar" else None)
            if predictor_value != target_value:
                raise AssertionError(
                    f"Predictor/target time {attr} differs: "
                    f"{predictor_value!r} != {target_value!r}."
                )
        predictor_value = np.asarray(predictor_time.isel({predictor_time_name: time_index})).item()
        target_value = np.asarray(target_time.isel({target_time_name: time_index})).item()
        if not bool(np.isclose(predictor_value, target_value, rtol=0.0, atol=0.0)):
            raise AssertionError(
                f"Predictor/target dates differ at index {time_index}: "
                f"{predictor_value!r} != {target_value!r}."
            )

        predictor_lat = np.asarray(predictor_ds[dataset.coarse_lat_name])
        predictor_lon = np.asarray(predictor_ds[dataset.coarse_lon_name])
        target_lat = np.asarray(target_ds[dataset.fine_lat_name])
        target_lon = np.asarray(target_ds[dataset.fine_lon_name])
        if predictor_lat.shape != target_lat.shape or predictor_lon.shape != target_lon.shape:
            raise AssertionError(
                "Regridded predictor and target coordinate shapes are not aligned."
            )
        if not np.allclose(predictor_lat, target_lat, rtol=0.0, atol=1e-6):
            raise AssertionError("Regridded predictor and target latitude coordinates differ.")
        if not np.allclose(predictor_lon, target_lon, rtol=0.0, atol=1e-6):
            raise AssertionError("Regridded predictor and target longitude coordinates differ.")

        return {
            "file_index": file_index,
            "time_index": time_index,
            "time_numeric": float(predictor_value),
            "time_units": predictor_time.attrs.get("units"),
            "time_calendar": predictor_time.attrs.get("calendar", "standard"),
            "time_length": int(predictor_ds.sizes[predictor_time_name]),
            "latitude_shape": list(predictor_lat.shape),
            "longitude_shape": list(predictor_lon.shape),
            "coordinates_equal": True,
        }


def _assert_dataset_preprocessing_parity(
    *,
    config: Any,
    predictor_path: Path,
    target_path: Path,
    crop_size: tuple[int, int],
) -> dict[str, Any]:
    """Compare controlled training and inference dataset preprocessing.

    The production training loader may apply an explicitly configured spatial
    offset augmentation. Exact parity is evaluated with augmentation disabled
    on both sides; the actual augmented production batch is still exercised by
    the remainder of this preflight. Both paths must otherwise return identical
    dynamic/static predictors, targets, dimensions, ordering, and dtype.
    """
    training_loader = build_dataloader(
        config,
        [str(predictor_path)],
        [str(target_path)],
        shuffle=False,
        use_gpu=False,
        distributed=False,
        rank=0,
        world_size=1,
        crop_size=crop_size,
        random_crop=False,
        random_crop_offset=(0, 0),
    )
    training_batch = next(iter(training_loader))

    inference_base = build_inference_dataset(
        config,
        [str(predictor_path)],
        [str(target_path)],
        crop_size=crop_size,
        allow_time_mismatch=True,
    )
    inference_sample = InferenceWrappedDataset(inference_base)[0]
    inference_batch = {
        key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else value
        for key, value in inference_sample.items()
    }

    if set(training_batch) != set(inference_batch):
        raise AssertionError(
            "Training/inference dataset keys differ: "
            f"{sorted(training_batch)} != {sorted(inference_batch)}."
        )
    max_errors: dict[str, float] = {}
    for key in sorted(training_batch):
        training_value = training_batch[key]
        inference_value = inference_batch[key]
        if not isinstance(training_value, torch.Tensor) or not isinstance(
            inference_value, torch.Tensor
        ):
            if training_value != inference_value:
                raise AssertionError(
                    f"Training/inference metadata {key!r} differs."
                )
            continue
        if training_value.shape != inference_value.shape:
            raise AssertionError(
                f"Training/inference tensor {key!r} shapes differ: "
                f"{training_value.shape} != {inference_value.shape}."
            )
        if training_value.dtype != inference_value.dtype:
            raise AssertionError(
                f"Training/inference tensor {key!r} dtypes differ: "
                f"{training_value.dtype} != {inference_value.dtype}."
            )
        max_error = float((training_value - inference_value).abs().max().item())
        max_errors[key] = max_error
        if max_error != 0.0:
            raise AssertionError(
                f"Training/inference tensor {key!r} differs (max error={max_error:.6g})."
            )

    training_base = training_loader.dataset.base
    coordinate_contract = {
        "coarse_lat_name": training_base.coarse_lat_name,
        "coarse_lon_name": training_base.coarse_lon_name,
        "fine_lat_name": training_base.fine_lat_name,
        "fine_lon_name": training_base.fine_lon_name,
        "output_spatial_dims": list(training_base.output_spatial_dims),
        "fine_shape": list(training_base.fine_shape),
    }
    inference_coordinate_contract = {
        "coarse_lat_name": inference_base.coarse_lat_name,
        "coarse_lon_name": inference_base.coarse_lon_name,
        "fine_lat_name": inference_base.fine_lat_name,
        "fine_lon_name": inference_base.fine_lon_name,
        "output_spatial_dims": list(inference_base.output_spatial_dims),
        "fine_shape": list(inference_base.fine_shape),
    }
    if coordinate_contract != inference_coordinate_contract:
        raise AssertionError(
            "Training/inference coordinate or dimension contracts differ: "
            f"{coordinate_contract!r} != {inference_coordinate_contract!r}."
        )

    return {
        "augmentation_disabled_for_exact_comparison": True,
        "production_training_augmentation": {
            "random_crop_offset": list(config.data.train_random_crop_offset),
        },
        "tensor_keys": sorted(training_batch),
        "max_abs_error": max_errors,
        "coordinate_contract": coordinate_contract,
        "exact": True,
    }


def _correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().reshape(-1).cpu()
    b = b.detach().float().reshape(-1).cpu()
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.sqrt(a.square().sum() * b.square().sum())
    if float(denom.item()) == 0.0:
        return math.nan
    return float((a * b).sum().div(denom).item())


def _assert_independent_noise(noise: torch.Tensor) -> dict[str, Any]:
    if noise.ndim != 4 or noise.shape[0] != 1:
        raise AssertionError(f"Expected one BCHW noise sample, got {tuple(noise.shape)}.")
    flat = noise.detach().float()
    mean = float(flat.mean().item())
    std = float(flat.std(unbiased=False).item())
    if abs(mean) >= 0.05 or not 0.95 <= std <= 1.05:
        raise AssertionError(
            f"Forward noise does not look standard-normal: mean={mean:.6g}, std={std:.6g}."
        )

    channel_correlations: list[float] = []
    for left in range(flat.shape[1]):
        for right in range(left + 1, flat.shape[1]):
            if torch.equal(flat[:, left], flat[:, right]):
                raise AssertionError(
                    f"Noise was exactly shared by channels {left} and {right}."
                )
            corr = _correlation(flat[:, left], flat[:, right])
            channel_correlations.append(corr)
            if not math.isfinite(corr) or abs(corr) >= 0.10:
                raise AssertionError(
                    f"Noise channels {left}/{right} are unexpectedly correlated: {corr:.6g}."
                )

    horizontal = [
        _correlation(flat[:, channel, :, :-1], flat[:, channel, :, 1:])
        for channel in range(flat.shape[1])
    ]
    vertical = [
        _correlation(flat[:, channel, :-1, :], flat[:, channel, 1:, :])
        for channel in range(flat.shape[1])
    ]
    for direction, correlations in (("horizontal", horizontal), ("vertical", vertical)):
        for channel, corr in enumerate(correlations):
            if not math.isfinite(corr) or abs(corr) >= 0.10:
                raise AssertionError(
                    f"Channel {channel} has unexpected {direction} neighbor noise "
                    f"correlation: {corr:.6g}."
                )

    return {
        "mean": mean,
        "std": std,
        "channel_correlations": channel_correlations,
        "horizontal_neighbor_correlations": horizontal,
        "vertical_neighbor_correlations": vertical,
        "independent_channel_pixel_draws": True,
    }


def _save_forward_noising_artifacts(
    *,
    sde: Any,
    residual: torch.Tensor,
    noise: torch.Tensor,
    variable_names: list[str],
    report_path: Path,
    sampling_eps: float,
) -> dict[str, Any]:
    """Save exact forward marginals and signal/noise amplitude diagnostics.

    The same independently sampled ``noise`` field is used at each displayed
    timestep so the visual change is attributable only to the scheduler
    coefficients. This is an audit artifact and is never fed back into
    training or inference.
    """
    import matplotlib.pyplot as plt

    if residual.ndim != 4 or residual.shape[0] != 1:
        raise AssertionError(
            f"Forward artifact expects one BCHW residual, got {tuple(residual.shape)}."
        )
    if noise.shape != residual.shape:
        raise AssertionError(
            f"Forward residual/noise shapes differ: {residual.shape} != {noise.shape}."
        )
    if residual.shape[1] != len(variable_names):
        raise AssertionError(
            f"Forward artifact variables {variable_names} do not label shape "
            f"{tuple(residual.shape)}."
        )

    eps = max(float(sampling_eps), 1e-5)
    requested_times = [eps, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    times_list = sorted({min(1.0, max(eps, value)) for value in requested_times})
    times = torch.tensor(
        times_list,
        device=residual.device,
        dtype=residual.dtype,
    )
    residual_at_times = residual.expand(times.numel(), -1, -1, -1)
    noise_at_times = noise.expand_as(residual_at_times)
    signal_component, sigma = sde.marginal_prob(residual_at_times, times)
    alpha_field, _ = sde.marginal_prob(
        torch.ones_like(residual_at_times), times
    )
    noise_component = sigma[:, None, None, None] * noise_at_times
    x_t = signal_component + noise_component
    reconstructed = (
        alpha_field * residual_at_times
        + sigma[:, None, None, None] * noise_at_times
    )
    max_equation_error = float((x_t - reconstructed).abs().max().item())
    if max_equation_error > 3e-6:
        raise AssertionError(
            "Forward artifact failed x_t = alpha*x0 + sigma*z; "
            f"max error={max_equation_error:.6g}."
        )

    alpha = alpha_field[:, 0, 0, 0]
    signal_rms = signal_component.square().mean(dim=(-2, -1)).sqrt()
    noise_rms = noise_component.square().mean(dim=(-2, -1)).sqrt()

    prefix = report_path.with_suffix("")
    npz_path = prefix.with_name(prefix.name + ".forward_noising.npz")
    amplitude_plot_path = prefix.with_name(
        prefix.name + ".forward_noising_amplitudes.png"
    )
    fields_plot_path = prefix.with_name(prefix.name + ".forward_noising_fields.png")
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {
        "timesteps": times.detach().float().cpu().numpy(),
        "alpha": alpha.detach().float().cpu().numpy(),
        "sigma": sigma.detach().float().cpu().numpy(),
        "residual_x0": residual.detach().float().cpu().numpy(),
        "noise_z": noise.detach().float().cpu().numpy(),
        "signal_component": signal_component.detach().float().cpu().numpy(),
        "noise_component": noise_component.detach().float().cpu().numpy(),
        "x_t": x_t.detach().float().cpu().numpy(),
        "signal_rms_by_variable": signal_rms.detach().float().cpu().numpy(),
        "noise_rms_by_variable": noise_rms.detach().float().cpu().numpy(),
    }
    np.savez_compressed(npz_path, **arrays)

    times_np = arrays["timesteps"]
    signal_rms_np = arrays["signal_rms_by_variable"]
    noise_rms_np = arrays["noise_rms_by_variable"]
    figure, axes = plt.subplots(
        1,
        len(variable_names),
        figsize=(5.2 * len(variable_names), 4.0),
        squeeze=False,
    )
    for variable_index, variable_name in enumerate(variable_names):
        axis = axes[0, variable_index]
        axis.plot(
            times_np,
            signal_rms_np[:, variable_index],
            marker="o",
            label="RMS(alpha(t) residual)",
        )
        axis.plot(
            times_np,
            noise_rms_np[:, variable_index],
            marker="o",
            label="RMS(sigma(t) noise)",
        )
        axis.set_title(variable_name)
        axis.set_xlabel("normalized diffusion time t")
        axis.set_ylabel("standardized amplitude")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Forward diffusion residual signal and noise amplitude")
    figure.tight_layout()
    figure.savefig(amplitude_plot_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    x_t_np = arrays["x_t"]
    figure, axes = plt.subplots(
        len(variable_names),
        len(times_np),
        figsize=(2.5 * len(times_np), 2.5 * len(variable_names)),
        squeeze=False,
    )
    for variable_index, variable_name in enumerate(variable_names):
        values = x_t_np[:, variable_index]
        color_limit = float(np.percentile(np.abs(values), 99.0))
        color_limit = max(color_limit, 1e-6)
        image = None
        for time_index, time_value in enumerate(times_np):
            axis = axes[variable_index, time_index]
            image = axis.imshow(
                values[time_index],
                cmap="RdBu_r",
                vmin=-color_limit,
                vmax=color_limit,
                origin="lower",
            )
            axis.set_title(f"{variable_name}, t={time_value:.3g}", fontsize=8)
            axis.set_xticks([])
            axis.set_yticks([])
        if image is not None:
            figure.colorbar(
                image,
                ax=axes[variable_index].tolist(),
                shrink=0.72,
                label="standardized residual x_t",
            )
    figure.suptitle("Actual residual under the configured forward SDE")
    figure.savefig(fields_plot_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    by_variable = {
        name: {
            "signal_rms": signal_rms_np[:, index].tolist(),
            "noise_rms": noise_rms_np[:, index].tolist(),
        }
        for index, name in enumerate(variable_names)
    }
    return {
        "npz": {"path": str(npz_path.resolve()), "sha256": _sha256(npz_path)},
        "amplitude_plot": {
            "path": str(amplitude_plot_path.resolve()),
            "sha256": _sha256(amplitude_plot_path),
        },
        "fields_plot": {
            "path": str(fields_plot_path.resolve()),
            "sha256": _sha256(fields_plot_path),
        },
        "timesteps": times_np.tolist(),
        "alpha": arrays["alpha"].tolist(),
        "sigma": arrays["sigma"].tolist(),
        "amplitudes_by_variable": by_variable,
        "max_abs_equation_error": max_equation_error,
        "same_noise_field_across_displayed_timesteps": True,
    }


def _load_scalar_metadata(scalar_dir: Path) -> dict[str, Any]:
    metadata_path = scalar_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Fresh scalar metadata is required for the preflight: {metadata_path}"
        )
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return metadata


def _assert_scalers_and_order(
    config: Any,
    model: torch.nn.Module,
    base_dataset: Any,
    predictor_path: Path,
    target_path: Path,
    static_path: Path,
) -> dict[str, Any]:
    predictor_names = build_predictor_names(config)
    output_names = list(config.data.output_vars)
    if list(base_dataset.predictor_vars) != predictor_names:
        raise AssertionError(
            f"Predictor variable order differs: {base_dataset.predictor_vars} != {predictor_names}."
        )
    if list(base_dataset.target_vars) != output_names:
        raise AssertionError(
            f"Target variable order differs: {base_dataset.target_vars} != {output_names}."
        )
    if list(model.output_var_names) != output_names:
        raise AssertionError(
            f"Model output order differs: {model.output_var_names} != {output_names}."
        )

    scalar_paths = {
        "inputs_mean": _repo_resolve(config.model.input_mu),
        "inputs_std": _repo_resolve(config.model.input_sigma),
        "targets_mean": _repo_resolve(config.model.target_mu),
        "targets_std": _repo_resolve(config.model.target_sigma),
    }
    for name, path in scalar_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Fresh scalar file is missing: {name}={path}")
    arrays = {name: np.load(path, mmap_mode="r") for name, path in scalar_paths.items()}

    expected_dynamic = len(predictor_names)
    expected_static = 1 if bool(config.data.use_static) else 0
    expected_input_channels = expected_dynamic + expected_static
    expected_target_shape = (
        len(output_names),
        int(config.data.target_size_lat),
        int(config.data.target_size_lon),
    )
    if tuple(arrays["inputs_mean"].shape) != (expected_input_channels,):
        raise AssertionError(
            f"inputs_mean shape {arrays['inputs_mean'].shape} != {(expected_input_channels,)}."
        )
    if tuple(arrays["inputs_std"].shape) != (expected_input_channels,):
        raise AssertionError(
            f"inputs_std shape {arrays['inputs_std'].shape} != {(expected_input_channels,)}."
        )
    for name in ("targets_mean", "targets_std"):
        if tuple(arrays[name].shape) != expected_target_shape:
            raise AssertionError(
                f"{name} shape {arrays[name].shape} != {expected_target_shape}; "
                "the tasmax gridpoint normalization is not represented."
            )
    if not np.allclose(np.asarray(arrays["targets_mean"])[0], 0.0, rtol=0.0, atol=1e-6):
        raise AssertionError("divide_only precipitation requires a zero target-mean field.")
    for name, array in arrays.items():
        values = np.asarray(array)
        if not np.isfinite(values).all():
            raise AssertionError(f"{name} contains NaN or infinity.")
        if name.endswith("std") and not bool((values > 0.0).all()):
            raise AssertionError(f"{name} contains non-positive scales.")

    if tuple(model.input_scalers_mu.shape) != (1, expected_dynamic, 1, 1):
        raise AssertionError(f"Unexpected model input mean shape {model.input_scalers_mu.shape}.")
    if tuple(model.static_input_scalers_mu.shape) != (1, expected_static, 1, 1):
        raise AssertionError(f"Unexpected model static mean shape {model.static_input_scalers_mu.shape}.")
    if tuple(model.output_scalers_mu.shape) != (1, *expected_target_shape):
        raise AssertionError(f"Unexpected model target mean shape {model.output_scalers_mu.shape}.")
    if tuple(model.output_scalers_sigma.shape) != (1, *expected_target_shape):
        raise AssertionError(f"Unexpected model target std shape {model.output_scalers_sigma.shape}.")

    metadata = _load_scalar_metadata(scalar_paths["inputs_mean"].parent)
    metadata_args = metadata.get("args", {})
    recorded_predictor_vars = metadata_args.get("predictor_vars")
    if recorded_predictor_vars is None:
        # ``compute_scalars_cordex.py`` passes None through to the dataset when
        # --predictor-vars is omitted; CordexDownscaleDataset then uses this
        # canonical tuple.  Verify that implicit order instead of treating the
        # metadata's null as an unknown ordering.
        scalar_predictor_order = list(base_dataset.DEFAULT_PREDICTORS)
        scalar_predictor_order_source = "CordexDownscaleDataset.DEFAULT_PREDICTORS"
    else:
        scalar_predictor_order = list(recorded_predictor_vars)
        scalar_predictor_order_source = "metadata.args.predictor_vars"
    if scalar_predictor_order != predictor_names:
        raise AssertionError("Scalar metadata predictor order differs from the training config.")
    if list(metadata_args.get("target_vars", [])) != output_names:
        raise AssertionError("Scalar metadata target order differs from the training config.")
    if int(metadata.get("input_channels", -1)) != expected_input_channels:
        raise AssertionError("Scalar metadata input-channel count is incompatible.")
    if int(metadata.get("target_channels", -1)) != len(output_names):
        raise AssertionError("Scalar metadata target-channel count is incompatible.")

    configured_sources = {
        "predictor_files": [str(predictor_path)],
        "target_files": [str(target_path)],
        "orography_file": str(static_path),
    }
    for key, expected in configured_sources.items():
        recorded = metadata_args.get(key)
        recorded_values = recorded if isinstance(recorded, list) else [recorded]
        expected_values = expected if isinstance(expected, list) else [expected]
        if [str(Path(value).resolve()) for value in recorded_values] != [
            str(Path(value).resolve()) for value in expected_values
        ]:
            raise AssertionError(
                f"Scalar metadata {key} was computed from {recorded!r}, expected {expected!r}."
            )

    return {
        "predictor_order": predictor_names,
        "scalar_predictor_order_source": scalar_predictor_order_source,
        "target_order": output_names,
        "static_order": ["orog"] if expected_static else [],
        "array_shapes": {name: list(array.shape) for name, array in arrays.items()},
        "model_scaler_shapes": {
            "dynamic_input_mean": list(model.input_scalers_mu.shape),
            "static_input_mean": list(model.static_input_scalers_mu.shape),
            "target_mean": list(model.output_scalers_mu.shape),
            "target_std": list(model.output_scalers_sigma.shape),
        },
        "files": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in scalar_paths.items()
        },
        "metadata": {
            "path": str(scalar_paths["inputs_mean"].parent / "metadata.json"),
            "sha256": _sha256(scalar_paths["inputs_mean"].parent / "metadata.json"),
            "created_at": metadata.get("created_at"),
            "num_samples": metadata.get("num_samples"),
        },
        "order_and_shapes_valid": True,
    }


def _instrument_one_forward(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    head = model.diffusion_head
    captures: dict[str, torch.Tensor] = {}
    original_training_loss = head.training_loss
    original_score_fn = head.score_fn
    original_score_matching_loss = diffusion_head_module.score_matching_loss

    def captured_training_loss(
        self: Any,
        cond: torch.Tensor,
        target: torch.Tensor,
        baseline_std: torch.Tensor | None = None,
    ) -> torch.Tensor:
        captures["raw_conditioning"] = cond.detach().clone()
        captures["ground_truth_std_argument"] = target.detach().clone()
        if baseline_std is None:
            raise AssertionError("Residual training did not provide baseline_std.")
        captures["baseline_std_argument"] = baseline_std.detach().clone()
        return original_training_loss(cond, target, baseline_std=baseline_std)

    def captured_score_fn(
        self: Any,
        x: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        captures["score_input_xt"] = x.detach().clone()
        captures["score_conditioning"] = cond.detach().clone()
        captures["score_time"] = t.detach().clone()
        return original_score_fn(x, cond, t)

    def captured_score_matching_loss(
        sde: Any,
        score_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        target: torch.Tensor,
        cond: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        captures["score_matching_target"] = target.detach().clone()
        captures["score_matching_conditioning"] = cond.detach().clone()
        return original_score_matching_loss(sde, score_fn, target, cond, **kwargs)

    head.training_loss = types.MethodType(captured_training_loss, head)
    head.score_fn = types.MethodType(captured_score_fn, head)
    diffusion_head_module.score_matching_loss = captured_score_matching_loss
    try:
        prediction = model(batch)
    finally:
        head.training_loss = original_training_loss
        head.score_fn = original_score_fn
        diffusion_head_module.score_matching_loss = original_score_matching_loss

    if not isinstance(prediction, dict):
        raise AssertionError(
            f"Joint residual model must return a mapping, got {type(prediction)!r}."
        )
    for key in ("baseline_prediction", "diffusion_loss"):
        if key not in prediction or not isinstance(prediction[key], torch.Tensor):
            raise AssertionError(f"Joint residual output is missing tensor {key!r}.")
    expected_captures = {
        "raw_conditioning",
        "ground_truth_std_argument",
        "baseline_std_argument",
        "score_input_xt",
        "score_conditioning",
        "score_time",
        "score_matching_target",
        "score_matching_conditioning",
    }
    missing = expected_captures.difference(captures)
    if missing:
        raise AssertionError(f"Instrumentation did not observe tensors: {sorted(missing)}")
    return prediction, captures


def run(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    config_path = args.config.expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config does not exist: {config_path}")
    config = get_config(str(config_path))

    predictor_path = _resolve_data_path(
        args.predictor,
        config.data.training_predictor_paths[0],
        DEFAULT_SA_ROOT
        / "train"
        / "Emulator_hist_future"
        / "predictors"
        / DEFAULT_PREDICTOR_NAME,
        "training predictor",
    )
    target_path = _resolve_data_path(
        args.target,
        config.data.training_target_paths[0],
        DEFAULT_SA_ROOT
        / "train"
        / "Emulator_hist_future"
        / "target"
        / DEFAULT_TARGET_NAME,
        "training target",
    )
    static_path = _resolve_data_path(
        args.static,
        config.data.static_path,
        DEFAULT_SA_ROOT / "train" / "Emulator_hist_future" / "predictors" / "Static_fields.nc",
        "static field",
    )
    config.data.training_predictor_paths = [str(predictor_path)]
    config.data.training_target_paths = [str(target_path)]
    config.data.static_path = str(static_path)
    config.batch_size = 1
    config.dl_num_workers = 0
    config.dl_prefetch_size = 0
    config.dl_pin_memory = False

    case_dir = _repo_resolve(config.case_dir)
    output_path = (
        args.output_json.expanduser().resolve()
        if args.output_json is not None
        else case_dir / "diagnostics" / "residual_real_batch_preflight.json"
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    device = torch.device(args.device)

    # Config paths are repository-relative in production, matching the training
    # notebook's model-construction working directory.
    original_cwd = Path.cwd()
    os.chdir(REPO_ROOT)
    try:
        train_loader = build_dataloader(
            config,
            [str(predictor_path)],
            [str(target_path)],
            shuffle=False,
            use_gpu=device.type == "cuda",
            distributed=False,
            rank=0,
            world_size=1,
            crop_size=(
                int(config.data.train_crop_size_lat),
                int(config.data.train_crop_size_lon),
            ),
            random_crop=True,
            random_crop_offset=tuple(config.data.train_random_crop_offset),
        )
        if args.sample_index < 0 or args.sample_index >= len(train_loader.dataset):
            raise IndexError(
                f"sample-index {args.sample_index} is outside [0, {len(train_loader.dataset)})."
            )
        if args.sample_index != 0:
            raise ValueError(
                "--sample-index must be 0 so the production DataLoader can be consumed "
                "exactly once without reading and discarding earlier samples."
            )
        base_dataset = train_loader.dataset.base
        # Preserve the real random-offset augmentation while making the selected
        # tile reproducible for this diagnostic artifact.
        base_dataset._rng = np.random.default_rng(args.seed)
        # This is the sole augmented production DataLoader iteration in the
        # script. batch_size=1, workers=0, shuffle=False, and no prefetch ensure
        # exactly one augmented training __getitem__ call. A separate controlled
        # no-augmentation loader is constructed below only for parity auditing.
        batch = next(iter(train_loader))
        alignment = _assert_source_alignment(
            predictor_path, target_path, base_dataset, args.sample_index
        )

        expected_dynamic = len(build_predictor_names(config))
        expected_outputs = len(config.data.output_vars)
        expected_hw = (
            int(config.data.train_crop_size_lat),
            int(config.data.train_crop_size_lon),
        )
        expected_shapes = {
            "x": (1, expected_dynamic, *expected_hw),
            "y": (1, expected_outputs, *expected_hw),
            "static_x": (1, 1, *expected_hw),
            "static_y": (1, 1, *expected_hw),
        }
        for key, expected_shape in expected_shapes.items():
            if key not in batch or tuple(batch[key].shape) != expected_shape:
                actual = None if key not in batch else tuple(batch[key].shape)
                raise AssertionError(f"Batch {key} shape {actual} != {expected_shape}.")
        if int(batch["x"].shape[0]) != 1:
            raise AssertionError("Preflight must load exactly one training sample.")

        preprocessing_parity = _assert_dataset_preprocessing_parity(
            config=config,
            predictor_path=predictor_path,
            target_path=target_path,
            crop_size=expected_hw,
        )

        print(f"[data] sample_index={args.sample_index} date={alignment['time_numeric']} "
              f"{alignment['time_units']} tile={expected_hw} variables={config.data.output_vars}")
        print(f"[model] constructing production fine-tune model on {device}")
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        model = create_finetune_model(config, verbose=True).to(device)
        model.train()

        if not bool(model.diffusion_head.cfg.residual_diffusion):
            raise AssertionError("Config/model did not enable residual_diffusion.")
        if not hasattr(model, "output_conv_block") or model.output_conv_block is None:
            raise AssertionError("Joint residual model has no deterministic U-Net output head.")
        if not any(parameter.requires_grad for parameter in model.output_conv_block.parameters()):
            raise AssertionError("Deterministic U-Net output head is not trainable.")

        scaler_report = _assert_scalers_and_order(
            config, model, base_dataset, predictor_path, target_path, static_path
        )

        batch = {
            key: value.to(device, non_blocking=False) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

        prediction, captures = _instrument_one_forward(model, batch)
        baseline_physical = prediction["baseline_prediction"]
        if not baseline_physical.requires_grad:
            raise AssertionError(
                "Deterministic baseline is detached; supervised U-Net/backbone training cannot work."
            )
        if not prediction["diffusion_loss"].requires_grad:
            raise AssertionError("Diffusion score loss has no gradient graph.")
        if tuple(baseline_physical.shape) != tuple(batch["y"].shape):
            raise AssertionError("Baseline and ground-truth tile shapes differ.")

        loss_fn = build_loss_fn(config, list(config.data.output_vars))
        if type(loss_fn).__name__ != "JointResidualDiffusionLoss":
            raise AssertionError(
                f"Expected JointResidualDiffusionLoss, got {type(loss_fn).__name__}."
            )
        joint_loss = loss_fn(prediction, batch)
        if joint_loss.numel() != 1 or not bool(torch.isfinite(joint_loss).item()):
            raise AssertionError("Joint deterministic + diffusion loss is not finite scalar.")
        if not joint_loss.requires_grad:
            raise AssertionError("Joint loss has no gradient graph.")

        truth_std = captures["ground_truth_std_argument"]
        baseline_std = captures["baseline_std_argument"]
        true_residual_std = truth_std - baseline_std
        score_target = captures["score_matching_target"]
        _assert_close(
            score_target,
            true_residual_std,
            "Score matching did not receive the raw signed target-minus-baseline residual.",
        )
        _assert_close(
            truth_std,
            baseline_std + true_residual_std,
            "Residual identity ground_truth_std = baseline_std + residual_std failed.",
        )
        residual_energy = float(true_residual_std.square().sum().item())
        if residual_energy <= 0.0:
            raise AssertionError("Residual is identically zero, so its sign cannot be checked.")
        sign_dot = float((score_target * true_residual_std).sum().item())
        reversed_dot = float((score_target * (-true_residual_std)).sum().item())
        if sign_dot <= 0.0 or reversed_dot >= 0.0:
            raise AssertionError("Residual sign is reversed; expected target - baseline.")
        if not bool((true_residual_std < 0.0).any().item()):
            raise AssertionError(
                "Real residual has no negative values; signed-residual activation check is inconclusive."
            )
        negative_expected = true_residual_std < 0.0
        if not torch.equal(score_target < 0.0, negative_expected):
            raise AssertionError(
                "A nonnegative activation or sign-changing transform was applied to the residual."
            )

        scaler_offset = batch.get("__scaler_offset")
        truth_std_direct = model._encode_targets_std(batch["y"], scaler_offset=scaler_offset)
        baseline_std_direct = model._encode_targets_std(
            baseline_physical, scaler_offset=scaler_offset
        )
        _assert_close(
            truth_std,
            truth_std_direct,
            "Training target normalization differs from _encode_targets_std.",
        )
        _assert_close(
            baseline_std,
            baseline_std_direct,
            "Baseline was not encoded exactly once from physical space.",
        )
        decoded_truth = model._decode_targets_std(truth_std, scaler_offset=scaler_offset)
        decoded_baseline = model._decode_targets_std(baseline_std, scaler_offset=scaler_offset)
        _assert_close(
            decoded_truth,
            batch["y"],
            "Physical/std ground-truth encode-decode roundtrip failed.",
            rtol=2e-5,
            atol=2e-4,
        )
        _assert_close(
            decoded_baseline,
            baseline_physical,
            "Physical/std baseline encode-decode roundtrip failed.",
            rtol=2e-5,
            atol=2e-4,
        )

        # Inspect the noising realization that the real score loss supplied to
        # score_fn.  For VPSDE, marginal_prob directly gives alpha*r and sigma.
        t = captures["score_time"]
        x_t = captures["score_input_xt"]
        marginal_mean, marginal_std = model.diffusion_head.sde.marginal_prob(
            true_residual_std, t
        )
        alpha = model.diffusion_head.sde.marginal_prob(
            torch.ones_like(true_residual_std), t
        )[0]
        noise = (x_t - marginal_mean) / marginal_std[:, None, None, None]
        reconstructed_xt = alpha * true_residual_std + marginal_std[:, None, None, None] * noise
        _assert_close(
            marginal_mean,
            alpha * true_residual_std,
            "SDE marginal mean is not alpha(t) * residual.",
            rtol=2e-5,
            atol=2e-6,
        )
        _assert_close(
            x_t,
            reconstructed_xt,
            "Forward noising does not satisfy x_t = alpha*r + sigma*z.",
            rtol=2e-5,
            atol=3e-6,
        )
        noise_report = _assert_independent_noise(noise)
        forward_artifacts = _save_forward_noising_artifacts(
            sde=model.diffusion_head.sde,
            residual=true_residual_std,
            noise=noise,
            variable_names=list(config.data.output_vars),
            report_path=output_path,
            sampling_eps=float(model.diffusion_head.cfg.sampling_eps),
        )

        raw_cond = captures["raw_conditioning"]
        actual_training_cond = captures["score_matching_conditioning"]
        expected_training_cond = model.diffusion_head._build_cond(raw_cond, baseline_std)
        _assert_close(
            actual_training_cond,
            expected_training_cond,
            "Actual training conditioning differs from residual conditioning helper.",
            rtol=0.0,
            atol=0.0,
        )
        _assert_close(
            captures["score_conditioning"],
            actual_training_cond,
            "score_fn did not receive the conditioning built by training_loss.",
            rtol=0.0,
            atol=0.0,
        )

        # Traverse the real sample_components inference entry point without a
        # reverse chain.  The stand-in sampler uses the exact training x_t/t
        # (and therefore the exact same z) for one score evaluation.
        model.eval()
        with torch.no_grad():
            training_score_eval = model.diffusion_head.score_fn(
                x_t, expected_training_cond, t
            )
        inference_capture: dict[str, torch.Tensor] = {}
        original_build_sampler = diffusion_head_module.build_sampler

        def fake_build_sampler(*_args: Any, **_kwargs: Any):
            def fake_sampler(
                score_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
                cond: torch.Tensor,
                generator: torch.Generator | None = None,
            ) -> torch.Tensor:
                del generator
                inference_capture["conditioning"] = cond.detach().clone()
                inference_capture["score"] = score_fn(x_t, cond, t).detach().clone()
                return torch.zeros_like(true_residual_std)

            return fake_sampler

        diffusion_head_module.build_sampler = fake_build_sampler
        try:
            with torch.no_grad():
                fake_full_std, fake_residual_std = model.diffusion_head.sample_components(
                    raw_cond,
                    int(true_residual_std.shape[-2]),
                    int(true_residual_std.shape[-1]),
                    baseline_std=baseline_std,
                )
        finally:
            diffusion_head_module.build_sampler = original_build_sampler
        if set(inference_capture) != {"conditioning", "score"}:
            raise AssertionError("Inference conditioning instrumentation was not exercised.")
        _assert_close(
            inference_capture["conditioning"],
            actual_training_cond,
            "Training and inference conditioning construction differ.",
            rtol=0.0,
            atol=0.0,
        )
        _assert_close(
            inference_capture["score"],
            training_score_eval,
            "Training/inference score calls differ for identical t, z, x_t, and conditioning.",
            rtol=1e-6,
            atol=2e-6,
        )
        _assert_close(
            fake_residual_std,
            torch.zeros_like(fake_residual_std),
            "Fake inference sampler did not return its unactivated signed residual.",
            rtol=0.0,
            atol=0.0,
        )
        _assert_close(
            fake_full_std,
            baseline_std,
            "Inference did not add residual to baseline with the expected sign.",
        )

        stages: dict[str, Any] = {}
        output_names = list(config.data.output_vars)
        stage_tensors = {
            "ground_truth_physical": batch["y"],
            "baseline_physical": baseline_physical,
            "ground_truth_std": truth_std,
            "baseline_std": baseline_std,
            "true_signed_residual_std": true_residual_std,
            "score_matching_target_std": score_target,
            "forward_noise_z": noise,
            "forward_noised_residual_xt": x_t,
            "decoded_ground_truth_physical": decoded_truth,
            "decoded_baseline_physical": decoded_baseline,
            "inference_zero_residual_std": fake_residual_std,
            "inference_full_std_for_zero_residual": fake_full_std,
        }
        for stage_name, tensor in stage_tensors.items():
            stages[stage_name] = _stage_summary(stage_name, tensor, output_names)

        report = {
            "status": "passed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "purpose": "one-real-sample residual diffusion integration preflight; no training",
            "config": {
                "path": str(config_path),
                "sha256": _sha256(config_path),
                "case_name": config.case_name,
                "job_id": config.job_id,
            },
            "environment": {
                "python": sys.version,
                "torch": torch.__version__,
                "device": str(device),
                "cuda_available": torch.cuda.is_available(),
                "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "seed": args.seed,
            },
            "data": {
                "predictor": str(predictor_path),
                "target": str(target_path),
                "static": str(static_path),
                "sample_count_loaded": 1,
                "sample_index": args.sample_index,
                "batch_shapes": {key: list(value.shape) for key, value in batch.items() if isinstance(value, torch.Tensor) and not key.startswith("__")},
                "alignment": alignment,
                "training_inference_preprocessing_parity": preprocessing_parity,
                "validity_policy": {
                    "mask_keys": sorted(key for key in batch if "mask" in key.lower()),
                    "targets": "all target values must be finite; non-finite targets hard-fail in CordexDownscaleDataset",
                },
            },
            "scalers_and_variable_order": scaler_report,
            "joint_objective": {
                "adapter": type(loss_fn).__name__,
                "total": float(joint_loss.detach().cpu().item()),
                "diffusion": float(prediction["diffusion_loss"].detach().cpu().item()),
                "terms": loss_fn.get_last_terms() if hasattr(loss_fn, "get_last_terms") else {},
                "baseline_requires_grad": baseline_physical.requires_grad,
                "diffusion_loss_requires_grad": prediction["diffusion_loss"].requires_grad,
                "joint_loss_requires_grad": joint_loss.requires_grad,
            },
            "residual_contract": {
                "definition": "ground_truth_std - stop_gradient(baseline_std)",
                "identity_max_abs_error": float(
                    (truth_std - (baseline_std + true_residual_std)).abs().max().item()
                ),
                "score_target_max_abs_error": float(
                    (score_target - true_residual_std).abs().max().item()
                ),
                "sign_dot_product": sign_dot,
                "reversed_sign_dot_product": reversed_dot,
                "negative_count": int((true_residual_std < 0.0).sum().item()),
                "positive_count": int((true_residual_std > 0.0).sum().item()),
                "signed_no_activation": True,
            },
            "normalization_contract": {
                "truth_physical_roundtrip_max_abs_error": float(
                    (decoded_truth - batch["y"]).abs().max().item()
                ),
                "baseline_physical_roundtrip_max_abs_error": float(
                    (decoded_baseline - baseline_physical).abs().max().item()
                ),
                "baseline_encoded_once": True,
            },
            "forward_noising": {
                "equation": "x_t = alpha(t) * residual + sigma(t) * z",
                "t": t.detach().float().cpu().tolist(),
                "alpha": alpha[:, :, :1, :1].detach().float().cpu().reshape(-1).tolist(),
                "sigma": marginal_std.detach().float().cpu().tolist(),
                "max_abs_equation_error": float((x_t - reconstructed_xt).abs().max().item()),
                "noise_checks": noise_report,
                "artifacts": forward_artifacts,
            },
            "conditioning_contract": {
                "raw_shape": list(raw_cond.shape),
                "full_shape": list(actual_training_cond.shape),
                "training_inference_max_abs_error": float(
                    (actual_training_cond - inference_capture["conditioning"]).abs().max().item()
                ),
                "same_t_z_xt_score_max_abs_error": float(
                    (training_score_eval - inference_capture["score"]).abs().max().item()
                ),
                "parity": True,
            },
            "stages": stages,
            "assertions": {
                "aligned_variables_dates_tiles": True,
                "residual_identity": True,
                "residual_sign_target_minus_baseline": True,
                "physical_standardized_roundtrip": True,
                "signed_residual_unactivated": True,
                "forward_noising_equation": True,
                "independent_channel_pixel_noise": True,
                "training_inference_conditioning_parity": True,
                "training_inference_dataset_preprocessing_parity": True,
                "scaler_shapes_and_variable_order": True,
            },
        }
    finally:
        os.chdir(original_cwd)

    return report, output_path


def main() -> None:
    args = parse_args()
    report, output_path = run(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(report), handle, indent=2, sort_keys=True)
    print(f"[PASS] all real-batch residual-diffusion preflight assertions passed")
    print(f"[PASS] JSON report: {output_path}")


if __name__ == "__main__":
    main()
