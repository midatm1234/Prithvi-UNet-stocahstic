"""Artifact-free smoke harness for the NARR--PRISM Phase-2 refiners.

This module is intentionally separate from the production NARR--PRISM data and
checkpoint path.  It uses a tiny deterministic Phase-1 fixture and synthetic,
same-timestamp fields to exercise one refinement objective, backward pass, and
stochastic sampling pass.  It is suitable for CI and notebook smoke execution;
its metrics are implementation diagnostics and must never be reported as
scientific NARR--PRISM skill.

The selected production YAML is still parsed, and its refinement type is used.
Only model width, process-step count, batch size, and spatial extent are
reduced to keep the check fast and independent of large checkpoints and
datasets.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from granitewxc.refinement import build_two_phase_model  # noqa: E402
from granitewxc.refinement.config import (  # noqa: E402
    REFINEMENT_TYPES,
    resolve_refinement_config,
)
from granitewxc.refinement.io import (  # noqa: E402
    build_refined_dataset,
    write_refined_netcdf,
)
from granitewxc.utils.config import get_config  # noqa: E402

VARIABLES = ("ppt", "tmax", "tmin")
UNITS = {"ppt": "mm/day", "tmax": "degC", "tmin": "degC"}
DEFAULT_CONFIGS = (
    "NARR_PRISM_diffusion_unet.yaml",
    "NARR_PRISM_diffusion_transformer.yaml",
    "NARR_PRISM_flow_matching_unet.yaml",
    "NARR_PRISM_flow_matching_transformer.yaml",
)


class TinyPhase1(nn.Module):
    """Small deterministic model implementing the real two-phase interface."""

    SUPPORTED_FEATURE_CAPTURES = ("prithvi", "unet")

    def __init__(self, in_channels: int = 4, out_channels: int = 3) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.head = nn.Conv2d(8, out_channels, kernel_size=1)
        self.output_scalers_mu = nn.Parameter(
            torch.tensor([0.0, 18.0, 8.0]).reshape(1, out_channels, 1, 1),
            requires_grad=False,
        )
        self.output_scalers_sigma = nn.Parameter(
            torch.tensor([4.0, 6.0, 6.0]).reshape(1, out_channels, 1, 1),
            requires_grad=False,
        )
        self.register_buffer(
            "predictand_scaling_method_codes",
            torch.tensor([1, 0, 0], dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "predictand_nonneg_enabled_mask",
            torch.tensor([True, False, False], dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "predictand_nonneg_method_codes",
            torch.tensor([1, 0, 0], dtype=torch.int64),
            persistent=False,
        )
        self._capture_features: tuple[str, ...] = ()
        self._last_phase1_features: dict[str, torch.Tensor] = {}
        self.n_input_timestamps = 1
        self.input_scalers_epsilon = 1.0e-6

    def _resolve_output_scalers(self, reference, scaler_offset=None):
        return (
            self.output_scalers_mu.to(
                device=reference.device, dtype=reference.dtype
            ),
            self.output_scalers_sigma.to(
                device=reference.device, dtype=reference.dtype
            ),
        )

    def _resolve_input_scalers(self, reference, scaler_offset=None):
        channels = reference.shape[1]
        mu = torch.zeros(
            1, channels, 1, 1, device=reference.device, dtype=reference.dtype
        )
        sigma = torch.ones_like(mu)
        return mu, sigma

    def set_feature_capture(self, names) -> None:
        self._capture_features = tuple(names or ())
        self._last_phase1_features = {}

    def get_last_phase1_features(self) -> dict[str, torch.Tensor]:
        return dict(self._last_phase1_features)

    def clear_last_phase1_features(self) -> None:
        self._last_phase1_features = {}

    def _decode(self, normalized: torch.Tensor) -> torch.Tensor:
        mu, sigma = self._resolve_output_scalers(normalized)
        decoded = normalized * sigma + mu
        divide = self.predictand_scaling_method_codes.view(1, -1, 1, 1) == 1
        return torch.where(divide, normalized * sigma, decoded)

    def forward(
        self,
        batch,
        return_pre_inverse: bool = False,
        return_raw_output: bool = False,
    ):
        features = self.body(batch["x"])
        if "prithvi" in self._capture_features:
            self._last_phase1_features["prithvi"] = features
        normalized = self.head(features)
        target_hw = batch.get("y", normalized).shape[-2:]
        if normalized.shape[-2:] != target_hw:
            normalized = torch.nn.functional.interpolate(
                normalized,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
        if "unet" in self._capture_features:
            self._last_phase1_features["unet"] = (
                torch.nn.functional.interpolate(
                    features,
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                )
            )
        nonnegative = self.predictand_nonneg_enabled_mask.view(1, -1, 1, 1)
        normalized = torch.where(
            nonnegative, torch.nn.functional.softplus(normalized), normalized
        )
        physical = self._decode(normalized)
        if return_pre_inverse and return_raw_output:
            return physical, normalized, normalized
        if return_pre_inverse:
            return physical, normalized
        if return_raw_output:
            return physical, normalized
        return physical


def set_reproducible_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch without changing global package state."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _repo_path(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def _config_refiner_type(config_path: Path) -> str:
    config = get_config(str(config_path))
    refiner_type = resolve_refinement_config(config).type
    if refiner_type not in REFINEMENT_TYPES or refiner_type == "none":
        raise ValueError(
            f"{config_path} does not select an active refinement head"
        )
    return refiner_type


def _tiny_config(
    refiner_type: str, ensemble_size: int, seed: int
) -> dict[str, Any]:
    refinement: dict[str, Any] = {
        "enabled": True,
        "type": refiner_type,
        "freeze_phase1": True,
        "joint_finetuning": False,
        "train_on_residual": True,
        "ensemble_size": int(ensemble_size),
        "loss": "mse",
        "seed": int(seed),
        "conditioning": {
            "deterministic_output": True,
            "input_predictors": True,
            "prithvi_features": False,
            "unet_features": False,
            "static_fields": True,
            "masks": True,
            "coordinates": False,
        },
        "unet": {
            "hidden_channels": 8,
            "num_levels": 2,
            "time_embedding_dim": 32,
            "bottleneck_attention": False,
            "attention_heads": 2,
            "zero_init_output": False,
        },
        "transformer": {
            "patch_size": [4, 4],
            "embedding_dim": 32,
            "num_heads": 4,
            "num_blocks": 1,
            "mlp_ratio": 2.0,
            "dropout": 0.0,
            "positional_encoding": "sincos_2d",
            "max_tokens_lat": 16,
            "max_tokens_lon": 16,
            "gradient_checkpointing": False,
            "optimized_attention": "math",
            "zero_init_output": False,
        },
    }
    if refiner_type.startswith("diffusion"):
        refinement["diffusion"] = {
            "training_timesteps": 12,
            "inference_steps": 2,
            "prediction_type": "epsilon",
            "schedule": "cosine",
            "eta": 0.0,
        }
    else:
        refinement["flow_matching"] = {
            "integration_steps": 2,
            "solver": "euler",
            "source_distribution": "gaussian",
            "stochastic_initialization": True,
            "time_sampling": "uniform",
        }
    return {"refinement": refinement}


def make_synthetic_batch(
    *, seed: int, batch_size: int = 2, height: int = 13, width: int = 17
) -> dict[str, torch.Tensor]:
    """Create correlated predictors and targets on a rectangular grid."""
    generator = torch.Generator().manual_seed(seed)
    ycoord = torch.linspace(-1.0, 1.0, height).reshape(1, 1, height, 1)
    xcoord = torch.linspace(-1.0, 1.0, width).reshape(1, 1, 1, width)
    terrain = 1.25 * torch.exp(
        -((xcoord + 0.25) ** 2 + (ycoord - 0.1) ** 2) / 0.16
    ) + 0.45 * torch.sin(3.0 * xcoord) * torch.cos(2.0 * ycoord)
    terrain = terrain.expand(batch_size, -1, -1, -1)
    noise = (
        torch.randn(batch_size, 3, height, width, generator=generator) * 0.12
    )
    dynamic = torch.randn(batch_size, 1, height, width, generator=generator)
    x = torch.cat(
        [
            dynamic,
            terrain,
            xcoord.expand(batch_size, -1, height, -1),
            ycoord.expand(batch_size, -1, -1, width),
        ],
        dim=1,
    )
    ppt = torch.relu(2.0 * terrain + 0.5 * dynamic + noise[:, 0:1])
    tmax = 24.0 - 5.0 * terrain + 1.4 * xcoord + noise[:, 1:2]
    tmin = 12.0 - 3.5 * terrain + 0.8 * ycoord + noise[:, 2:3]
    target = torch.cat([ppt, tmax, tmin], dim=1)
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[..., :2, :2] = False
    target = target.masked_fill(~valid, float("nan"))
    return {"x": x, "y": target, "static_x": terrain, "static_y": terrain}


def _masked_rmse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    valid = torch.isfinite(prediction) & torch.isfinite(target)
    if not bool(valid.any()):
        return float("nan")
    return float(
        torch.sqrt(
            torch.mean((prediction[valid] - target[valid]).square())
        ).cpu()
    )


def _masked_bias(prediction: torch.Tensor, target: torch.Tensor) -> float:
    valid = torch.isfinite(prediction) & torch.isfinite(target)
    if not bool(valid.any()):
        return float("nan")
    return float(torch.mean(prediction[valid] - target[valid]).cpu())


def _metrics(
    out, target: torch.Tensor, loss: torch.Tensor, gradient_norm: float
) -> dict[str, Any]:
    fields = (
        out.ensemble_mean if out.ensemble_mean is not None else out.refined
    )
    result: dict[str, Any] = {
        "loss": float(loss.detach().cpu()),
        "gradient_norm": float(gradient_norm),
        "finite_loss": bool(torch.isfinite(loss)),
        "finite_valid_output": bool(
            torch.isfinite(fields[torch.isfinite(target)]).all()
        ),
        "phase1_rmse": _masked_rmse(out.deterministic, target),
        "refined_rmse": _masked_rmse(fields, target),
        "phase1_bias": _masked_bias(out.deterministic, target),
        "refined_bias": _masked_bias(fields, target),
        "ensemble_spread_mean": (
            float(torch.nanmean(out.ensemble_spread).cpu())
            if out.ensemble_spread is not None
            else 0.0
        ),
    }
    for index, name in enumerate(VARIABLES):
        result[name] = {
            "phase1_rmse": _masked_rmse(
                out.deterministic[:, index], target[:, index]
            ),
            "refined_rmse": _masked_rmse(fields[:, index], target[:, index]),
            "phase1_bias": _masked_bias(
                out.deterministic[:, index], target[:, index]
            ),
            "refined_bias": _masked_bias(fields[:, index], target[:, index]),
        }
    return result


def _diffusion_process_diagnostics(
    model, target, conditioning, valid, generator
):
    refiner = model.refiner
    steps = torch.tensor(
        [0, refiner.num_train_timesteps // 2, refiner.num_train_timesteps - 1],
        device=target.device,
        dtype=torch.long,
    )
    clean = target[:1].expand(3, -1, -1, -1)
    noise = torch.randn(
        clean.shape,
        generator=generator,
        device=clean.device,
        dtype=clean.dtype,
    )
    noised = refiner.schedule.add_noise(clean, noise, steps)
    conditioning3 = conditioning[:1].expand(3, -1, -1, -1)
    with torch.no_grad():
        estimated = refiner.net(
            noised, conditioning3, refiner._embed_time(steps)
        )
        clean_estimate = refiner.schedule.to_clean(
            refiner.prediction_type, estimated, noised, steps
        )
    return noised.detach().cpu(), clean_estimate.detach().cpu()


def _flow_process_diagnostics(model, target, conditioning, generator):
    refiner = model.refiner
    x0 = torch.randn(
        target[:1].shape,
        generator=generator,
        device=target.device,
        dtype=target.dtype,
    )
    times = torch.tensor(
        [0.0, 0.5, 1.0], device=target.device, dtype=target.dtype
    )
    path = torch.cat([(1.0 - t) * x0 + t * target[:1] for t in times], dim=0)
    with torch.no_grad():
        velocity = refiner.net(
            path,
            conditioning[:1].expand(3, -1, -1, -1),
            refiner._embed_time(times),
        )
    clean_estimate = path + (1.0 - times).reshape(-1, 1, 1, 1) * velocity
    return path.detach().cpu(), clean_estimate.detach().cpu()


def _write_figure(
    path: Path,
    *,
    clean_residual: torch.Tensor,
    process_states: torch.Tensor,
    clean_estimate: torch.Tensor,
    sampled_residual: torch.Tensor,
    deterministic: torch.Tensor,
    refined: torch.Tensor,
    truth: torch.Tensor,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    panels = (
        (clean_residual, "Clean normalized residual"),
        (process_states[0], "Process state: start"),
        (process_states[1], "Process state: middle"),
        (clean_estimate[-1], "Model clean-residual estimate"),
        (sampled_residual, "Sampled residual"),
        (deterministic, "Deterministic baseline"),
        (refined, "Refined ensemble mean"),
        (truth, "Synthetic target"),
    )
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    for axis, (field, title) in zip(axes.flat, panels, strict=True):
        image = axis.imshow(
            field.detach().float().cpu().numpy(),
            origin="lower",
            cmap="viridis",
        )
        axis.set_title(f"ppt: {title}", fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(image, ax=axis, shrink=0.72)
    fig.suptitle("Synthetic smoke diagnostic — not scientific validation")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_smoke(
    config_path: str | os.PathLike[str],
    *,
    output_dir: str | os.PathLike[str],
    ensemble_size: int = 2,
    seed: int = 1234,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run an artifact-free train/backward/sample check for one YAML."""
    config_path = _repo_path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Refinement YAML not found: {config_path}")
    output_dir = _repo_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    refiner_type = _config_refiner_type(config_path)
    set_reproducible_seed(seed)
    torch_device = torch.device(device)

    batch = {
        key: value.to(torch_device)
        for key, value in make_synthetic_batch(seed=seed).items()
    }
    model = build_two_phase_model(
        TinyPhase1(),
        _tiny_config(refiner_type, ensemble_size=ensemble_size, seed=seed),
    ).to(torch_device)
    model.initialize_from_batch(batch)
    model.to(torch_device).train()
    train_generator = torch.Generator(device=torch_device).manual_seed(
        seed + 1
    )
    training = model.training_step(batch, generator=train_generator)
    loss = training.losses["loss"]
    if not bool(torch.isfinite(loss)):
        raise RuntimeError(
            f"{refiner_type} produced non-finite smoke loss: {loss}"
        )
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.refiner.parameters()
        if parameter.grad is not None
    ]
    if not gradients or not all(
        bool(torch.isfinite(grad).all()) for grad in gradients
    ):
        raise RuntimeError(
            f"{refiner_type} did not produce finite refiner gradients"
        )
    if any(
        parameter.grad is not None for parameter in model.phase1.parameters()
    ):
        raise RuntimeError(
            "Frozen TinyPhase1 unexpectedly received a gradient"
        )
    gradient_norm = math.sqrt(
        sum(
            float(gradient.detach().float().square().sum().cpu())
            for gradient in gradients
        )
    )

    model.eval()
    with torch.no_grad():
        physical, normalized, features = model.run_phase1(batch)
        conditioning = model.build_conditioning(batch, normalized, features)
        clean_residual, valid = model.target_space.residual_target(
            batch["y"], normalized
        )
        diagnostic_generator = torch.Generator(
            device=torch_device
        ).manual_seed(seed + 2)
        if model.refinement_config.is_diffusion:
            process_states, clean_estimate = _diffusion_process_diagnostics(
                model,
                clean_residual,
                conditioning,
                valid,
                diagnostic_generator,
            )
        else:
            process_states, clean_estimate = _flow_process_diagnostics(
                model, clean_residual, conditioning, diagnostic_generator
            )
        prediction = model.predict(
            batch, ensemble_size=ensemble_size, seed=seed + 3
        )

    metrics = _metrics(prediction, batch["y"], loss, gradient_norm)
    metrics.update(
        {
            "schema_version": 1,
            "mode": "synthetic_smoke",
            "scientific_validation": False,
            "config_path": str(config_path),
            "refinement_type": refiner_type,
            "variables": list(VARIABLES),
            "ensemble_size": int(ensemble_size),
            "seed": int(seed),
            "device": str(torch_device),
            "phase1_frozen": bool(model.phase1_frozen),
            "trainable_refiner_parameters": sum(
                p.numel()
                for p in model.refiner.parameters()
                if p.requires_grad
            ),
        }
    )

    stem = refiner_type
    metrics_path = output_dir / f"{stem}_smoke_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )

    target = batch["y"].detach().cpu()
    output_dataset = build_refined_dataset(
        variables=VARIABLES,
        coords={
            "time": np.array(
                ["2000-01-01", "2000-01-02"], dtype="datetime64[ns]"
            ),
            "lat": np.linspace(32.55, 41.075, target.shape[-2]),
            "lon": np.linspace(-124.45, -115.925, target.shape[-1]),
        },
        deterministic=prediction.deterministic,
        residual=prediction.residual,
        members=prediction.members,
        ensemble_mean=prediction.ensemble_mean,
        ensemble_spread=prediction.ensemble_spread,
        refined=prediction.refined,
        truth=target,
        units=UNITS,
        attrs={
            "mode": "synthetic_smoke",
            "scientific_validation": "false",
            "refinement_type": refiner_type,
            "source_config": str(config_path),
        },
    )
    netcdf_path = Path(
        write_refined_netcdf(
            output_dataset,
            output_dir / f"{stem}_smoke.nc",
            compression=False,
            chunk_sizes={"time": 1, "member": 1},
        )
    )
    figure_path = output_dir / f"{stem}_smoke_diagnostics.png"
    _write_figure(
        figure_path,
        clean_residual=clean_residual[0, 0],
        process_states=process_states[:, 0],
        clean_estimate=clean_estimate[:, 0],
        sampled_residual=prediction.residual[0, 0],
        deterministic=prediction.deterministic[0, 0],
        refined=prediction.refined[0, 0],
        truth=target[0, 0],
    )

    result = dict(metrics)
    result["artifacts"] = {
        "metrics": str(metrics_path),
        "netcdf": str(netcdf_path),
        "diagnostics": str(figure_path),
    }
    return result


def run_smoke_suite(
    config_paths: Sequence[str | os.PathLike[str]] | None = None,
    *,
    output_dir: str | os.PathLike[str],
    ensemble_size: int = 2,
    seed: int = 1234,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run the smoke harness for all four configs, or an explicit subset."""
    paths = list(
        config_paths
        or [Path(__file__).parent / name for name in DEFAULT_CONFIGS]
    )
    results = [
        run_smoke(
            path,
            output_dir=output_dir,
            ensemble_size=ensemble_size,
            seed=seed + index,
            device=device,
        )
        for index, path in enumerate(paths)
    ]
    suite = {
        "schema_version": 1,
        "mode": "synthetic_smoke",
        "scientific_validation": False,
        "results": results,
    }
    destination = _repo_path(output_dir) / "smoke_suite_summary.json"
    destination.write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n")
    suite["summary_path"] = str(destination)
    return suite


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help=(
            "Refinement YAML to smoke-test; repeat for a subset. "
            "Default: all four."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ensemble-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    suite = run_smoke_suite(
        args.config,
        output_dir=args.output_dir,
        ensemble_size=args.ensemble_size,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(suite, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
