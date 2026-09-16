"""Pointwise Gaussian process skips with a noise-independent source endpoint.

These are training parameterizations, never postprocessing of physical fields.
They preserve the native flow velocity / DDPM epsilon, velocity or sample
objective. Checkpoints trained without them must not be reinterpreted with them.
"""
from __future__ import annotations

import torch


def flow_coefficients(time, sigma_min, state):
    t = time.float().reshape(-1, 1, 1, 1)
    a = 1 - (1 - sigma_min) * t
    variance = a.square() + t.square()
    # Cov(u_t, x_t)/Var(x_t), and conditional standard deviation of u_t,
    # for independent unit Gaussian source and standardized clean residual.
    skip = (t - (1 - sigma_min) * a) / variance
    scale = variance.rsqrt()
    input_scale = t * variance.rsqrt()
    return skip.to(state), scale.to(state), input_scale.to(state)


def diffusion_coefficients(alpha, sigma, prediction_type, state):
    alpha, sigma = alpha.reshape(-1,1,1,1).to(state), sigma.reshape(-1,1,1,1).to(state)
    if prediction_type == "epsilon":
        skip, scale = sigma, alpha
    elif prediction_type == "sample":
        skip, scale = alpha, sigma
    elif prediction_type == "velocity":
        skip, scale = torch.zeros_like(alpha), torch.ones_like(alpha)
    else:
        raise ValueError(f"Unsupported diffusion parameterization {prediction_type!r}")
    return skip, scale, alpha


def predict_process(net, state, conditioning, embedded_time, coefficients, channels):
    """Predict selected channels with analytic source response at every cell.

    At a pure-noise endpoint the selected branch sees zero stochastic state.
    Thus source removal cannot depend on convolution padding, patch position,
    or a learned approximation to the pixelwise identity. Other channels use
    the existing network input and output parameterization unchanged.
    """
    channels = tuple(channels)
    if not channels:
        return net(state.to(conditioning.dtype), conditioning, embedded_time)
    if len(set(channels)) != len(channels) or any(i < 0 or i >= state.shape[1] for i in channels):
        raise ValueError("Preconditioned channel indices must be unique and present in the residual field")
    skip, scale, input_scale = coefficients
    learned = net((state * input_scale).to(conditioning.dtype), conditioning, embedded_time)
    corrected = skip * state.float() + scale * learned.float()
    if len(channels) == state.shape[1]:
        return corrected
    legacy = net(state.to(conditioning.dtype), conditioning, embedded_time)
    mask = torch.zeros((1,state.shape[1],1,1),device=state.device,dtype=torch.bool)
    mask[:,list(channels)] = True
    return torch.where(mask, corrected, legacy)
