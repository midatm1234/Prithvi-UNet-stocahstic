"""Per-variable native diffusion objectives with explicit time weighting."""
from __future__ import annotations

import torch

from granitewxc.refinement.base import masked_loss


def clean_prediction_process_loss(prediction, target, valid_mask, kind, sigma, velocity_channels):
    """Use velocity-equivalent MSE for selected clean-prediction channels.

    x_t=alpha*x0+sigma*epsilon and v=alpha*epsilon-sigma*x0 imply
    x0=alpha*x_t-sigma*v. Therefore delta_v=-delta_x0/sigma. This
    changes the selected variables' noise-time weighting; unselected variables
    keep their original clean-prediction MSE and equal-channel coefficient.
    No physical residual or inference sample is rescaled by this function.
    """
    channels=tuple(velocity_channels)
    if not channels:
        return masked_loss(prediction,target,valid_mask,kind)
    if kind != 'mse':
        raise ValueError('Velocity-equivalent clean-prediction loss requires MSE')
    if len(set(channels)) != len(channels) or any(c<0 or c>=prediction.shape[1] for c in channels):
        raise ValueError('Velocity-loss channel indices must be unique and present')
    sigma=sigma.to(device=prediction.device,dtype=torch.float32).reshape(-1,1,1,1)
    if sigma.shape[0] not in (1,prediction.shape[0]) or not bool(torch.isfinite(sigma).all()) or bool((sigma<=0).any()):
        raise ValueError('Velocity loss needs finite positive noise scales per sample')
    mask=torch.ones_like(prediction,dtype=torch.bool) if valid_mask is None else torch.broadcast_to(valid_mask.to(device=prediction.device)>0,prediction.shape)
    error=torch.where(mask,prediction.float(),0.)-torch.where(mask,target.float(),0.)
    scale=torch.ones((prediction.shape[0],prediction.shape[1],1,1),device=prediction.device)
    scale[:,list(channels)]=sigma
    adjusted=error/scale
    return masked_loss(adjusted,torch.zeros_like(adjusted),valid_mask,'mse')


def boundary_balanced_process_weights(reference, valid_mask, channels):
    """Half global-cell and half equal-band native loss, selected channels only.

    Bands measure distance from the rectangular physical tensor perimeter:
    0, 1, 2, 3, 4--7, 8--15, >=16. Missing observations have zero weight;
    they do not create artificial geographic boundaries. For each channel,
    occupied bands receive equal weight in the balanced half. The other half
    retains the ordinary valid-cell objective. Returned weights sum to the
    original valid weight sum, and unselected channels are unchanged.
    This acts only during training; no inference sample is weighted or blended.
    """
    if not channels:
        return valid_mask
    if reference.ndim != 4:
        raise ValueError("Boundary balance requires BCHW fields")
    if len(set(channels)) != len(channels) or any(type(c) is not int or c < 0 or c >= reference.shape[1] for c in channels):
        raise ValueError("Boundary-balance channels must be unique valid integer indices")
    weights = torch.ones_like(reference,dtype=torch.float32) if valid_mask is None else torch.broadcast_to(valid_mask.to(device=reference.device,dtype=torch.float32),reference.shape).clone()
    if not bool(torch.isfinite(weights).all()) or bool((weights<0).any()):
        raise ValueError("Boundary loss weights must be finite and non-negative")
    h,w=reference.shape[-2:]
    yy=torch.arange(h,device=reference.device)[:,None]
    xx=torch.arange(w,device=reference.device)[None,:]
    distance=torch.minimum(torch.minimum(yy,h-1-yy),torch.minimum(xx,w-1-xx))
    bands=[distance==n for n in range(4)]+[(distance>=4)&(distance<8),(distance>=8)&(distance<16),distance>=16]
    for channel in channels:
        original=weights[:,channel].clone()
        counts=torch.stack([(original*band).sum() for band in bands])
        occupied=(counts>0).sum().clamp_min(1)
        total=original.sum()
        balanced=torch.zeros_like(original)
        for band,count in zip(bands,counts):
            balanced=balanced+original*band*(total/(occupied*torch.where(count>0,count,torch.ones_like(count))))
        weights[:,channel]=.5*original+.5*balanced
    return weights
