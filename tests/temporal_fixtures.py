"""Shared fixtures for the temporal-extension tests.

The model built here is a *structurally real* ClimateDownscaleFinetuneUNETModel
-- same classes, same forward path, same hook -- just small enough to run many
times on CPU. Tests that must exercise the actual SA checkpoint geometry use
:func:`sa_config_dict` instead and are marked ``slow``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from granitewxc.utils.config import ExperimentConfig

#: Small grid that still satisfies every divisibility assertion in the model:
#: H, W divisible by mask_unit_size (8) and by 2**num_upsample (8).
TINY_H = 32
TINY_W = 32
TINY_VARS = ("u", "v")
TINY_LEVELS = (850.0, 700.0)
TINY_N_DYNAMIC = len(TINY_VARS) * len(TINY_LEVELS)  # 4
TINY_OUTPUTS = ("pr", "tasmax")


def write_scalers(tmp_path: Path, *, n_dynamic: int, n_static: int, outputs: tuple[str, ...],
                  height: int, width: int) -> dict[str, str]:
    """Write scaler ``.npy`` files matching the model's expectations.

    ``pr`` uses ``divide_only`` scaling, which the model validates requires a
    zero target mean; the temperature channel uses gridpoint standardization, so
    ``targets_mean`` is a full field exactly as in the real SA checkpoint.
    """
    d = tmp_path / "scalars"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    inputs_mean = rng.normal(0.0, 1.0, size=n_dynamic + n_static).astype(np.float32)
    inputs_std = (rng.uniform(0.5, 2.0, size=n_dynamic + n_static)).astype(np.float32)

    targets_mean = np.zeros((len(outputs), height, width), dtype=np.float32)
    targets_std = np.ones((len(outputs), height, width), dtype=np.float32)
    for idx, name in enumerate(outputs):
        if name in {"pr", "ppt"}:
            targets_mean[idx] = 0.0            # divide_only requires mu == 0
            targets_std[idx] = 5.0
        else:
            targets_mean[idx] = 285.0 + rng.normal(0.0, 1.0, size=(height, width))
            targets_std[idx] = 3.0

    paths = {}
    for name, arr in (
        ("inputs_mean", inputs_mean),
        ("inputs_std", inputs_std),
        ("targets_mean", targets_mean),
        ("targets_std", targets_std),
    ):
        p = d / f"{name}.npy"
        np.save(p, arr)
        paths[name] = str(p)
    return paths


def tiny_config_dict(tmp_path: Path, *, use_static: bool = True) -> dict[str, Any]:
    """A minimal but structurally faithful CORDEX-style config."""
    n_static = 1 if use_static else 0
    scalers = write_scalers(
        tmp_path,
        n_dynamic=TINY_N_DYNAMIC,
        n_static=n_static,
        outputs=TINY_OUTPUTS,
        height=TINY_H,
        width=TINY_W,
    )
    return {
        "case_name": "temporal_tiny",
        "job_id": "temporal_tiny",
        "data": {
            "type": "cordex",
            "input_vars": list(TINY_VARS),
            "input_levels": list(TINY_LEVELS),
            "input_surface_vars": [],
            "input_static_surface_vars": ["orog"] if use_static else [],
            "vertical_pres_vars": list(TINY_VARS),
            "input_level_pres": list(TINY_LEVELS),
            "vertical_level1_vars": [],
            "input_level1": [],
            "vertical_level2_vars": [],
            "input_level2": [],
            "other": [],
            "output_vars": list(TINY_OUTPUTS),
            "use_static": use_static,
            "n_input_timestamps": 1,
            "input_size_lat": TINY_H,
            "input_size_lon": TINY_W,
            "target_size_lat": TINY_H,
            "target_size_lon": TINY_W,
            "train_crop_size_lat": TINY_H,
            "train_crop_size_lon": TINY_W,
            "static_path": "unused-in-model-construction.nc",
            "training_predictor_paths": [],
            "training_target_paths": [],
            "validation_predictor_paths": [],
            "validation_target_paths": [],
            "test_predictor_paths": [],
            "test_target_paths": [],
            "scalers": {
                "inputs_mean": scalers["inputs_mean"],
                "inputs_std": scalers["inputs_std"],
                "targets_mean": scalers["targets_mean"],
                "targets_std": scalers["targets_std"],
            },
        },
        "predictands": {
            "pr": {
                "name": "pr",
                "allow_negative_value": False,
                "nonnegativity": {"enabled": True, "method": "softplus"},
                "scaling": {"method": "divide_only", "scale_stat": "p95", "fixed_scale": None},
                "normalization": {
                    "method": "divide_only",
                    "mode": "global",
                    "eps_std": 1.0e-6,
                    "allow_negative_value": False,
                },
            },
            "tasmax": {
                "name": "tasmax",
                "allow_negative_value": False,
                "nonnegativity": {"enabled": False, "method": "none"},
                "scaling": {"method": "zscore", "scale_stat": "mean", "fixed_scale": None},
                "normalization": {"method": "standardize", "mode": "gridpoint", "eps_std": 1.0e-6},
            },
        },
        "model": {
            "input_mu": scalers["inputs_mean"],
            "input_sigma": scalers["inputs_std"],
            "target_mu": scalers["targets_mean"],
            "target_sigma": scalers["targets_std"],
            "embed_dim": 32,
            "n_blocks_encoder": 1,
            "mlp_multiplier": 1,
            "n_heads": 2,
            "dropout_rate": 0.0,
            "drop_path": 0.0,
            "residual": "none",
            "num_static_channels": n_static,
            "token_size": [1, 1],
            "downscaling_patch_size": [2, 2],
            "downscaling_embed_dim": 16,
            "encoder_decoder_type": "conv",
            "encoder_decoder_upsampling_mode": "pixel_shuffle",
            "encoder_decoder_kernel_size_per_stage": [[3], [3]],
            "encoder_decoder_scale_per_stage": [[2], [3]],
            "encoder_decoder_conv_channels": 16,
            "residual_connection": True,
            "encoder_shift": False,
            "unet": True,
            "decoder_upsampling_mode": "bilinear",
            "output_scaler_resize_mode": "bilinear",
            "output_scaler_align_corners": False,
            "unet_upsample_scales": [2, 2, 2],
            "unet_decoder_kernel_size": [3, 3, 3],
        },
        "mask_unit_size": [8, 8],
        "batch_size": 1,
        "num_epochs": 1,
        "learning_rate": 1.0e-4,
        "min_lr": 1.0e-6,
        "max_lr": 1.0e-4,
        "warm_up_steps": 0,
        "limit_steps_train": 1,
        "limit_steps_valid": 1,
        "dl_num_workers": 0,
        "dl_prefetch_size": 0,
        "mask_ratio_inputs": 0.0,
        "mask_ratio_targets": 0.0,
        "path_experiment": str(tmp_path / "experiments"),
        "backbone_use": True,
        "backbone_freeze": False,
        "finetune_w_static": use_static,
        "precip_model": "single_head",
        "precip_wet_threshold": 0.01,
        "device_target": "cpu",
        "backbone_gradient_checkpointing": False,
        "skip_activation_offload": True,
    }


def build_tiny_model(tmp_path: Path, *, use_static: bool = True, seed: int = 0):
    """Construct the tiny model deterministically."""
    from granitewxc.models.model import get_finetune_model_UNET

    cfg = ExperimentConfig.from_dict(tiny_config_dict(tmp_path, use_static=use_static))
    torch.manual_seed(seed)
    model = get_finetune_model_UNET(cfg)
    model.eval()
    return model, cfg


def make_sequence_batch(
    *,
    batch: int = 2,
    frames: int = 5,
    n_dynamic: int = TINY_N_DYNAMIC,
    n_out: int = len(TINY_OUTPUTS),
    height: int = TINY_H,
    width: int = TINY_W,
    time_dim: int = 5,
    use_static: bool = True,
    seed: int = 7,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    out = {
        "x": torch.randn(batch, frames, n_dynamic, height, width, generator=g),
        "y": torch.randn(batch, frames, n_out, height, width, generator=g),
        "time_features": torch.randn(batch, frames, time_dim, generator=g),
        "interval_ratio": torch.ones(batch, frames),
        "reset": torch.zeros(batch, frames, dtype=torch.bool),
        "__target_valid_mask": torch.ones(batch, frames, n_out, height, width, dtype=torch.bool),
    }
    out["reset"][:, 0] = True
    if use_static:
        out["static_x"] = torch.randn(batch, 1, height, width, generator=g)
        out["static_y"] = out["static_x"].clone()
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in out.items()}


def frame_by_frame_reference(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Run the model one frame at a time with no temporal adapter active."""
    frames = batch["x"].shape[1]
    outs = []
    with torch.no_grad():
        for t in range(frames):
            frame = {"x": batch["x"][:, t], "y": batch["y"][:, t]}
            for key in ("static_x", "static_y"):
                if key in batch:
                    frame[key] = batch[key]
            outs.append(model(frame))
    return torch.stack(outs, dim=1)


def temporal_config(**overrides) -> Any:
    """Build a valid TemporalConfig with test-friendly defaults."""
    from granitewxc.temporal.config import parse_temporal_config

    base: dict[str, Any] = {
        "enabled": True,
        "backend": "recurrent",
        "context_length": 5,
        "output_length": 5,
        "cadence_days": 1.0,
        "latent": {"hidden_channels": 16, "groups": 4},
        "init_from_spatial_checkpoint": "unused.ckpt",
    }
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return parse_temporal_config(merged)
