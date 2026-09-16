"""Explicit affine transfer from raw to standardized predictor conditioning."""
from __future__ import annotations

import torch
from torch import nn


@torch.no_grad()
def fold_conditioning_affine(refiner, mean: torch.Tensor, scale: torch.Tensor) -> None:
    """Preserve the initial function when raw = mean + scale * normalized.

    Statistics must be spatially constant and come from the frozen training
    scalers. This is a weight transfer for retraining, not an inference override
    of a checkpoint's conditioning contract. Replication padding commutes with
    the affine map, including corners; zero padding does not.
    """
    mean, scale = mean.reshape(-1), scale.reshape(-1)
    if mean.numel() != refiner.cond_channels or scale.shape != mean.shape:
        raise ValueError("Affine statistics must match all conditioning channels")
    if not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Affine statistics must be finite with positive scales")
    if hasattr(refiner.net, "down_blocks"):
        convolution = refiner.net.down_blocks[0].conv1
        mean = torch.cat((mean.new_zeros(refiner.residual_channels), mean))
        scale = torch.cat((scale.new_ones(refiner.residual_channels), scale))
    elif hasattr(refiner.net, "condition_stem"):
        convolution = refiner.net.condition_stem[0]
    else:
        raise ValueError("No supported direct conditioning convolution")
    if not isinstance(convolution, nn.Conv2d) or convolution.groups != 1 or convolution.bias is None:
        raise ValueError("Affine transfer requires a biased, ungrouped input convolution")
    if convolution.padding != (0, 0) and convolution.padding_mode != "replicate":
        raise ValueError("Affine transfer requires replication padding")
    if convolution.in_channels != mean.numel():
        raise ValueError("Unexpected input convolution channel layout")
    # Accumulate the constant offset accurately before returning to weight dtype.
    weight = convolution.weight.double()
    offset = (weight * mean.to(weight)[None, :, None, None]).sum((1, 2, 3))
    convolution.bias.add_(offset.to(convolution.bias))
    convolution.weight.mul_(scale.to(convolution.weight)[None, :, None, None])


def frozen_conditioning_affine(wrapper, batch):
    """Build a global affine map in the wrapper's actual conditioning order."""
    config = wrapper.refinement_config.conditioning
    if config.prithvi_features or config.unet_features:
        raise ValueError("Feature conditioning transfer is not defined")
    means, scales = [], []

    def identity(channels):
        means.append(torch.zeros(channels))
        scales.append(torch.ones(channels))

    def append_statistics(mu, sigma, epsilon, repeats=1):
        if mu.ndim != 4 or sigma.ndim != 4 or mu.shape[-2:] != (1, 1) or sigma.shape[-2:] != (1, 1):
            raise ValueError("Affine weight folding requires global, not gridpoint, predictor statistics")
        means.append(mu.detach().cpu().float().reshape(-1).repeat(repeats))
        scales.append((sigma.detach().cpu().float().reshape(-1) + epsilon).repeat(repeats))

    if config.deterministic_output:
        identity(batch['__phase1_normalized'].shape[1])
    if config.input_predictors:
        values = batch['x']
        mu, sigma = wrapper.phase1._resolve_input_scalers(
            values, scaler_offset=batch.get('__input_scaler_offset', batch.get('__scaler_offset')))
        append_statistics(mu, sigma, float(wrapper.phase1.input_scalers_epsilon),
                          int(wrapper.phase1.n_input_timestamps))
    if config.static_fields:
        for key, prefix in (('static_x', 'static_input'), ('static_y', 'static_output')):
            if batch.get(key) is not None:
                append_statistics(getattr(wrapper.phase1, prefix+'_scalers_mu'),
                                  getattr(wrapper.phase1, prefix+'_scalers_sigma'),
                                  float(wrapper.phase1.static_input_scalers_epsilon))
    if config.masks:
        identity(1)
    return torch.cat(means), torch.cat(scales)
