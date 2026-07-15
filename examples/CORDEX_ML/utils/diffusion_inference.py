"""Helpers for diffusion-head checkpoint detection and ensemble inference."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable

import numpy as np
import torch
import xarray as xr


DIFFUSION_HEAD_ALIASES = {"diffusion", "diffusion_head", "sde", "score", "score_sde"}


def _get_nested(obj: Any, *keys: str) -> Any:
    current = obj
    for key in keys:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
    return current


def canonical_head_type(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in DIFFUSION_HEAD_ALIASES:
        return "diffusion"
    if text in {"deterministic", "conv", "convolutional", "unet", "standard"}:
        return "deterministic"
    return text or None


def checkpoint_head_type(checkpoint: Any) -> str | None:
    """Return a checkpoint-declared head type when present."""
    if not isinstance(checkpoint, dict):
        return None

    candidates = [
        checkpoint.get("head_type"),
        checkpoint.get("decoder_type"),
        checkpoint.get("diffusion_head"),
        _get_nested(checkpoint, "model_config", "head_type"),
        _get_nested(checkpoint, "model_config", "decoder_type"),
        _get_nested(checkpoint, "model", "head_type"),
        _get_nested(checkpoint, "config", "model", "head_type"),
        _get_nested(checkpoint, "config", "model", "decoder_type"),
        _get_nested(checkpoint, "metadata", "head_type"),
        _get_nested(checkpoint, "metadata", "decoder_type"),
        _get_nested(checkpoint, "metadata", "diffusion_head"),
    ]
    for value in candidates:
        if isinstance(value, bool):
            if value:
                return "diffusion"
            continue
        head_type = canonical_head_type(value)
        if head_type:
            return head_type
    state_dict = checkpoint.get("model") or checkpoint.get("state_dict") or checkpoint
    if isinstance(state_dict, dict) and any("diffusion_head" in str(key) for key in state_dict.keys()):
        return "diffusion"
    return None


def config_head_type(config: Any) -> str:
    head_type = canonical_head_type(_get_nested(config, "model", "head_type"))
    if head_type:
        return head_type
    head_type = canonical_head_type(_get_nested(config, "model", "decoder_type"))
    if head_type:
        return head_type
    return "deterministic"


def model_head_type(model: Any) -> str | None:
    diffusion_enabled = getattr(model, "diffusion_enabled", None)
    if diffusion_enabled is not None:
        return "diffusion" if bool(diffusion_enabled) else "deterministic"
    return canonical_head_type(getattr(model, "head_type", None))


def infer_head_type(checkpoint: Any = None, config: Any = None, model: Any = None) -> str:
    """Detect model head type, preferring explicit checkpoint metadata."""
    return checkpoint_head_type(checkpoint) or config_head_type(config) or model_head_type(model) or "deterministic"


def resolve_ensemble_size(
    config: Any = None,
    *,
    requested: int | None = None,
    head_type: str = "deterministic",
) -> int:
    """Resolve ensemble size from an explicit request or the loaded config.

    Both deterministic and diffusion paths default to one member when the YAML
    omits ``inference.ensemble_size``. No diffusion-specific size is imposed.
    """
    value = requested
    if value is None:
        value = _get_nested(config, "inference", "ensemble_size")
    if value is None:
        value = 1
    value = int(value)
    if value < 1:
        raise ValueError(f"ensemble_size must be at least 1, got {value}")
    return value


def resolve_base_seed(config: Any = None, *, default: int = 42) -> int:
    value = _get_nested(config, "inference", "base_seed")
    if value is None:
        value = _get_nested(config, "inference", "ensemble_seed")
    return int(default if value is None else value)


def seed_everything(seed: int, device: torch.device | None = None) -> None:
    torch.manual_seed(int(seed))
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))


def call_model_with_optional_raw(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model_out = model(batch, return_pre_inverse=True, return_raw_output=True)
    if isinstance(model_out, tuple):
        out = model_out[0]
        if len(model_out) >= 3:
            pre_inverse = model_out[1]
            raw = model_out[2]
        elif len(model_out) == 2:
            pre_inverse = model_out[1]
            raw = model_out[1]
        else:
            pre_inverse = out
            raw = out
    else:
        out = model_out
        pre_inverse = model_out
        raw = model_out
    return out, pre_inverse, raw


def infer_batch_ensemble(
    *,
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    infer_batch: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    boundary_cfg: Any,
    head_type: str,
    ensemble_size: int,
    base_seed: int,
    device: torch.device,
    autocast_context: Callable[[], Any] | None = None,
    force_float32: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one deterministic pass or multiple independent diffusion samples.

    Deterministic outputs keep shape ``[B, V, H, W]``. Diffusion ensemble outputs
    use shape ``[B, E, V, H, W]``.
    """
    is_diffusion = canonical_head_type(head_type) == "diffusion"
    n_members = resolve_ensemble_size(head_type=head_type, requested=ensemble_size) if is_diffusion else 1
    autocast_context = autocast_context or nullcontext

    outs: list[torch.Tensor] = []
    pre_inverse_outs: list[torch.Tensor] = []
    raw_outs: list[torch.Tensor] = []

    for member_idx in range(n_members):
        if is_diffusion:
            seed_everything(base_seed + member_idx, device=device)
        with autocast_context():
            out, pre_inverse, raw = infer_batch(model=model, batch=batch, cfg=boundary_cfg)
        if force_float32:
            out = out.float()
            pre_inverse = pre_inverse.float()
            raw = raw.float()
        outs.append(out)
        pre_inverse_outs.append(pre_inverse)
        raw_outs.append(raw)
        if not is_diffusion:
            break

    if is_diffusion:
        return (
            torch.stack(outs, dim=1),
            torch.stack(pre_inverse_outs, dim=1),
            torch.stack(raw_outs, dim=1),
        )
    return outs[0], pre_inverse_outs[0], raw_outs[0]


def variable_array_map(var_names: list[str], values: np.ndarray) -> dict[str, np.ndarray]:
    """Map ``[T,V,H,W]`` or ``[T,E,V,H,W]`` arrays to per-variable arrays."""
    arr = np.asarray(values, dtype=np.float32)
    out: dict[str, np.ndarray] = {}
    for var_idx, var_name in enumerate(var_names):
        if arr.ndim == 5:
            out[var_name] = arr[:, :, var_idx]
        else:
            out[var_name] = arr[:, var_idx]
    return out


def add_predictions_to_dataset(
    *,
    prediction_ds: xr.Dataset,
    target_vars: list[str],
    outputs_np: np.ndarray,
    coords: dict[str, Any],
    time_dim: str,
    lat_dim: str,
    lon_dim: str,
    target_attrs: dict[str, dict[str, Any]],
    head_type: str,
    ensemble_size: int,
    base_seed: int,
) -> xr.Dataset:
    """Populate prediction variables, adding ensemble dim only for diffusion."""
    is_diffusion = canonical_head_type(head_type) == "diffusion"
    if is_diffusion:
        prediction_ds = prediction_ds.assign_coords(ensemble=np.arange(int(ensemble_size), dtype=np.int32))
        dims = (time_dim, "ensemble", lat_dim, lon_dim)
    else:
        dims = (time_dim, lat_dim, lon_dim)

    for var_idx, name in enumerate(target_vars):
        values = outputs_np[:, :, var_idx] if is_diffusion else outputs_np[:, var_idx]
        prediction_ds[name] = xr.DataArray(
            values,
            dims=dims,
            coords={dim: prediction_ds.coords[dim] for dim in dims},
            attrs=target_attrs.get(name, {}),
        ).astype(np.float32)

    if is_diffusion:
        prediction_ds.attrs.update(
            {
                "head_type": "diffusion",
                "ensemble_size": int(ensemble_size),
                "ensemble_generation": "diffusion_sampling",
                "base_seed": int(base_seed),
            }
        )
    else:
        prediction_ds.attrs.setdefault("head_type", "deterministic")
    return prediction_ds


def as_ensemble_mean(values: np.ndarray, *, ensemble_axis: int = 1) -> np.ndarray:
    """Default evaluation policy for diffusion outputs: evaluate ensemble mean."""
    arr = np.asarray(values)
    if arr.ndim > ensemble_axis and arr.shape[ensemble_axis] > 1:
        return arr.mean(axis=ensemble_axis)
    return arr
