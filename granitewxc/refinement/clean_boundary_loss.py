"""Smooth regional-boundary supervision on reconstructed physical residuals."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def physical_clean_boundary_loss(prediction, target, valid_mask, channels, *, alpha, length, delta):
    """Weighted Huber loss after inverse residual normalization and gating.

    Subtracting physical residuals is exactly the reconstructed field error:
    (baseline + predicted_residual) - (baseline + target_residual). No clipping,
    velocity/noise interpretation, or weighting of inference fields occurs here.
    Distances refer to the rectangular regional frame, not missing observations.
    The physical Huber delta has the selected variable's units (mm/day for pr).
    """
    if prediction.ndim != 4 or prediction.shape != target.shape:
        raise ValueError("Physical clean loss requires matching BCHW tensors")
    if not channels or len(set(channels)) != len(channels) or any(type(c) is not int or c < 0 or c >= prediction.shape[1] for c in channels):
        raise ValueError("Clean-loss channels must be unique valid integer indices")
    if not all(math.isfinite(v) for v in (alpha, length, delta)) or alpha < 0 or length <= 0 or delta <= 0:
        raise ValueError("Clean-loss alpha must be nonnegative; length and delta positive and finite")
    selected = list(channels)
    pred, truth = prediction[:, selected].float(), target[:, selected].float()
    mask = torch.ones_like(pred, dtype=torch.bool) if valid_mask is None else torch.broadcast_to(valid_mask.to(device=prediction.device, dtype=torch.bool), prediction.shape)[:, selected]
    if not bool(torch.isfinite(pred[mask]).all()) or not bool(torch.isfinite(truth[mask]).all()):
        raise FloatingPointError("Nonfinite physical clean prediction or target at a valid cell")
    # Mask before arithmetic: invalid NaNs must not contaminate gradients.
    pred = torch.where(mask, pred, 0.)
    truth = torch.where(mask, truth, 0.)
    h, w = pred.shape[-2:]
    yy = torch.arange(h, device=pred.device)[:, None]
    xx = torch.arange(w, device=pred.device)[None, :]
    distance = torch.minimum(torch.minimum(yy, h - 1 - yy), torch.minimum(xx, w - 1 - xx))
    weights = mask * (1. + float(alpha) * torch.exp(-distance.float() / float(length)))
    errors = F.huber_loss(pred, truth, reduction="none", delta=float(delta))
    counts = weights.sum(dim=(0, 2, 3))
    means = (errors * weights).sum(dim=(0, 2, 3)) / counts.clamp_min(1.)
    return means.sum() / (counts > 0).sum().clamp_min(1)
