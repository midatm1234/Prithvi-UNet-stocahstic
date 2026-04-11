"""Inference-time boundary mitigation utilities for CORDEX reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class DeblockConfig:
    enabled: bool = False
    boundary_width: int = 2
    strength: float = 0.15
    kernel_size: int = 3


@dataclass
class BoundaryMitigationConfig:
    enabled: bool = False
    force_full_frame: bool = False
    tile_size: tuple[int, int] | None = None
    overlap: tuple[int, int] = (0, 0)
    blend_window: str = "hann"
    blend_sigma: float = 0.35
    deblock: DeblockConfig = field(default_factory=DeblockConfig)

    @property
    def blend_mode(self) -> str:
        # Backward-compatible alias used by existing inference scripts.
        return self.blend_window


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {}


def _to_tuple2(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        v = int(value)
        return v, v
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    return default


def _resolve_model_stride(model: torch.nn.Module) -> tuple[int, int]:
    raw = getattr(model, "mask_unit_size_px_backbone", None)
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return max(1, int(raw[0])), max(1, int(raw[1]))
    return 1, 1


def _auto_tile_dim(size: int, stride: int) -> int:
    """Pick a smaller tile so overlap/blending can run, aligned to model stride."""
    if size <= stride:
        return size

    target = int(round(size * 0.75))
    target = min(target, size - 1)
    target = max(stride, target)

    snapped = max(stride, (target // stride) * stride)
    if snapped >= size:
        fallback = size - stride
        snapped = max(stride, (fallback // stride) * stride)
    if snapped <= 0:
        return size
    return min(snapped, size)


def resolve_boundary_mitigation_settings(config: Any) -> BoundaryMitigationConfig:
    root = _as_mapping(getattr(config, "inference", None))
    raw = _as_mapping(root.get("boundary_mitigation", None))
    if not raw:
        raw = _as_mapping(getattr(config, "boundary_mitigation", None))

    target_h = int(getattr(getattr(config, "data", object()), "target_size_lat", 0) or 0)
    target_w = int(getattr(getattr(config, "data", object()), "target_size_lon", 0) or 0)

    force_full_frame = bool(
        raw.get(
            "force_full_frame",
            root.get("force_full_frame", False),
        )
    )

    tile_size = _to_tuple2(
        raw.get("tile_size", root.get("inference_tile_size")),
        (target_h, target_w),
    )
    if tile_size[0] <= 0 or tile_size[1] <= 0:
        tile_size = None

    overlap = _to_tuple2(
        raw.get("overlap", root.get("inference_overlap")),
        (0, 0),
    )
    overlap = (max(0, overlap[0]), max(0, overlap[1]))

    blend_window = str(
        raw.get(
            "blend_window",
            raw.get("blend_mode", root.get("inference_blend_window", "hann")),
        )
    ).lower()
    if blend_window not in {"hann", "cosine", "linear", "uniform", "gaussian"}:
        blend_window = "hann"
    blend_sigma = float(raw.get("blend_sigma", root.get("inference_blend_sigma", 0.35)))
    blend_sigma = max(1e-3, blend_sigma)

    deblock_raw = _as_mapping(raw.get("deblock", None))
    use_seam_deblocking = bool(
        raw.get(
            "use_seam_deblocking",
            root.get("use_seam_deblocking", deblock_raw.get("enabled", False)),
        )
    )
    seam_deblocking_strength = float(
        raw.get(
            "seam_deblocking_strength",
            root.get("seam_deblocking_strength", deblock_raw.get("strength", 0.15)),
        )
    )
    deblock = DeblockConfig(
        enabled=use_seam_deblocking,
        boundary_width=max(1, int(deblock_raw.get("boundary_width", 2))),
        strength=seam_deblocking_strength,
        kernel_size=max(3, int(deblock_raw.get("kernel_size", 3))),
    )
    if deblock.kernel_size % 2 == 0:
        deblock.kernel_size += 1
    deblock.strength = min(max(deblock.strength, 0.0), 1.0)

    enabled = bool(raw.get("enabled", root.get("enable_overlap_blending", False)))
    return BoundaryMitigationConfig(
        enabled=enabled,
        force_full_frame=force_full_frame,
        tile_size=tile_size,
        overlap=overlap,
        blend_window=blend_window,
        blend_sigma=blend_sigma,
        deblock=deblock,
    )


def _compute_starts(size: int, tile: int, overlap: int) -> list[int]:
    if tile >= size:
        return [0]
    stride = max(1, tile - overlap)
    starts = list(range(0, max(size - tile + 1, 1), stride))
    last = size - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def _axis_weight(
    length: int,
    overlap: int,
    *,
    touch_start: bool,
    touch_end: bool,
    blend_window: str,
    blend_sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    w = torch.ones(length, device=device, dtype=dtype)
    if overlap <= 0 or blend_window == "uniform":
        return w

    ramp_lin = torch.linspace(0.0, 1.0, steps=overlap + 2, device=device, dtype=dtype)[1:-1]
    if blend_window in {"cosine", "hann"}:
        ramp = 0.5 - 0.5 * torch.cos(torch.pi * ramp_lin)
    elif blend_window == "linear":
        ramp = ramp_lin
    elif blend_window == "gaussian":
        # Smooth center-weighted transition from edge (0) to interior (1).
        x = 1.0 - ramp_lin
        ramp = torch.exp(-0.5 * (x / blend_sigma) ** 2)
        ramp = (ramp - ramp.min()) / (ramp.max() - ramp.min() + 1e-6)
    else:
        ramp = ramp_lin

    if not touch_start:
        w[:overlap] = ramp
    if not touch_end:
        w[-overlap:] = torch.flip(ramp, dims=[0])
    return w.clamp_min(1e-6)


def _tile_weight(
    tile_h: int,
    tile_w: int,
    overlap: tuple[int, int],
    *,
    touch_top: bool,
    touch_bottom: bool,
    touch_left: bool,
    touch_right: bool,
    blend_window: str,
    blend_sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    wy = _axis_weight(
        tile_h,
        min(overlap[0], max(0, tile_h - 1)),
        touch_start=touch_top,
        touch_end=touch_bottom,
        blend_window=blend_window,
        blend_sigma=blend_sigma,
        device=device,
        dtype=dtype,
    ).view(1, 1, tile_h, 1)
    wx = _axis_weight(
        tile_w,
        min(overlap[1], max(0, tile_w - 1)),
        touch_start=touch_left,
        touch_end=touch_right,
        blend_window=blend_window,
        blend_sigma=blend_sigma,
        device=device,
        dtype=dtype,
    ).view(1, 1, 1, tile_w)
    return wy * wx


def _apply_boundary_deblock(
    tensor: torch.Tensor,
    *,
    y_starts: list[int],
    x_starts: list[int],
    cfg: DeblockConfig,
) -> torch.Tensor:
    if not cfg.enabled or cfg.strength <= 0.0:
        return tensor

    _, _, h, w = tensor.shape
    mask = torch.zeros((1, 1, h, w), device=tensor.device, dtype=tensor.dtype)
    band = int(cfg.boundary_width)

    for y0 in y_starts[1:]:
        y_min = max(0, y0 - band)
        y_max = min(h, y0 + band)
        if y_min < y_max:
            mask[:, :, y_min:y_max, :] = 1.0
    for x0 in x_starts[1:]:
        x_min = max(0, x0 - band)
        x_max = min(w, x0 + band)
        if x_min < x_max:
            mask[:, :, :, x_min:x_max] = 1.0

    if not bool(mask.any().item()):
        return tensor

    smoothed = F.avg_pool2d(
        tensor,
        kernel_size=cfg.kernel_size,
        stride=1,
        padding=cfg.kernel_size // 2,
    )
    alpha = (mask * cfg.strength).clamp(0.0, 1.0)
    return tensor * (1.0 - alpha) + smoothed * alpha


def _call_model_with_optional_raw(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def infer_batch_with_boundary_mitigation(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    cfg: BoundaryMitigationConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not cfg.enabled or cfg.force_full_frame:
        return _call_model_with_optional_raw(model, batch)

    x = batch["x"]
    _, _, h, w = x.shape

    stride_h, stride_w = _resolve_model_stride(model)

    def _snap_tile(value: int, *, stride: int, limit: int) -> int:
        if limit <= stride:
            return limit
        value = min(max(stride, int(value)), limit)
        snapped = (value // stride) * stride
        if snapped <= 0:
            snapped = stride
        if snapped > limit:
            snapped = (limit // stride) * stride
            if snapped <= 0:
                snapped = limit
        return min(limit, snapped)

    tile_h, tile_w = cfg.tile_size or (h, w)
    tile_h = _snap_tile(int(tile_h), stride=stride_h, limit=h)
    tile_w = _snap_tile(int(tile_w), stride=stride_w, limit=w)

    overlap_h = min(max(0, int(cfg.overlap[0])), max(0, tile_h - stride_h))
    overlap_w = min(max(0, int(cfg.overlap[1])), max(0, tile_w - stride_w))
    if stride_h > 1 and overlap_h > 0:
        overlap_h = max(0, (overlap_h // stride_h) * stride_h)
    if stride_w > 1 and overlap_w > 0:
        overlap_w = max(0, (overlap_w // stride_w) * stride_w)

    # Full-frame request means no spatial stitching.
    if tile_h >= h and tile_w >= w:
        return _call_model_with_optional_raw(model, batch)

    y_starts = _compute_starts(h, tile_h, overlap_h)
    x_starts = _compute_starts(w, tile_w, overlap_w)
    base_offset = batch.get("__scaler_offset", (0, 0))
    base_y, base_x = int(base_offset[0]), int(base_offset[1])

    out_sum = pre_sum = raw_sum = weight_sum = None

    for y0 in y_starts:
        y1 = min(y0 + tile_h, h)
        for x0 in x_starts:
            x1 = min(x0 + tile_w, w)
            tile_batch = {
                key: value[..., y0:y1, x0:x1] if torch.is_tensor(value) and value.ndim >= 4 else value
                for key, value in batch.items()
            }
            tile_batch["__scaler_offset"] = (base_y + y0, base_x + x0)
            tile_out, tile_pre, tile_raw = _call_model_with_optional_raw(model, tile_batch)

            weight = _tile_weight(
                y1 - y0,
                x1 - x0,
                (overlap_h, overlap_w),
                touch_top=(y0 == 0),
                touch_bottom=(y1 == h),
                touch_left=(x0 == 0),
                touch_right=(x1 == w),
                blend_window=cfg.blend_window,
                blend_sigma=cfg.blend_sigma,
                device=tile_out.device,
                dtype=tile_out.dtype,
            )

            if out_sum is None:
                out_sum = torch.zeros(
                    (tile_out.shape[0], tile_out.shape[1], h, w),
                    device=tile_out.device,
                    dtype=tile_out.dtype,
                )
                pre_sum = torch.zeros_like(out_sum)
                raw_sum = torch.zeros_like(out_sum)
                weight_sum = torch.zeros((1, 1, h, w), device=tile_out.device, dtype=tile_out.dtype)

            out_sum[..., y0:y1, x0:x1] += tile_out * weight
            pre_sum[..., y0:y1, x0:x1] += tile_pre * weight
            raw_sum[..., y0:y1, x0:x1] += tile_raw * weight
            weight_sum[..., y0:y1, x0:x1] += weight

    assert out_sum is not None and pre_sum is not None and raw_sum is not None and weight_sum is not None
    denom = weight_sum.clamp_min(1e-6)
    out = out_sum / denom
    pre = pre_sum / denom
    raw = raw_sum / denom

    if cfg.deblock.enabled:
        out = _apply_boundary_deblock(
            out,
            y_starts=y_starts,
            x_starts=x_starts,
            cfg=cfg.deblock,
        )
        pre = _apply_boundary_deblock(
            pre,
            y_starts=y_starts,
            x_starts=x_starts,
            cfg=cfg.deblock,
        )
        raw = _apply_boundary_deblock(
            raw,
            y_starts=y_starts,
            x_starts=x_starts,
            cfg=cfg.deblock,
        )

    return out, pre, raw
