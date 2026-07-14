"""Reverse-diffusion samplers for the conditional diffusion head.

Adapted from ``mlde``'s ``sampling.py`` (predictor/corrector framework). Only
the pieces required for conditional CORDEX sampling are kept:

    * Euler-Maruyama and reverse-diffusion predictors
    * Langevin and (no-op) correctors
    * a predictor-corrector (PC) sampler and a probability-flow ODE sampler

Samplers take a conditional ``score_fn(x, cond, t)`` and a conditioning tensor,
and return a sample in standardized target space of shape
``[B, output_channels, H, W]`` (batch size taken from ``cond``).
"""

from __future__ import annotations

import abc
from typing import Callable

import numpy as np
import torch

from granitewxc.models.diffusion_sde import SDE, VPSDE, VESDE, subVPSDE

__all__ = ["build_sampler", "get_pc_sampler", "get_ode_sampler"]

ScoreFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


# ----------------------------------------------------------------------------
# Predictors
# ----------------------------------------------------------------------------
class Predictor(abc.ABC):
    def __init__(self, sde: SDE, score_fn: ScoreFn, probability_flow: bool = False):
        self.sde = sde
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, cond, t):
        ...


class EulerMaruyamaPredictor(Predictor):
    def update_fn(self, x, cond, t):
        dt = -1.0 / self.rsde.N
        z = torch.randn_like(x)
        drift, diffusion = self.rsde.sde(x, cond, t)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
        return x, x_mean


class ReverseDiffusionPredictor(Predictor):
    def update_fn(self, x, cond, t):
        f, G = self.rsde.discretize(x, cond, t)
        z = torch.randn_like(x)
        x_mean = x - f
        x = x_mean + G[:, None, None, None] * z
        return x, x_mean


class NonePredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        self.sde = sde
        self.score_fn = score_fn

    def update_fn(self, x, cond, t):
        return x, x


_PREDICTORS = {
    "euler_maruyama": EulerMaruyamaPredictor,
    "reverse_diffusion": ReverseDiffusionPredictor,
    "none": NonePredictor,
}


# ----------------------------------------------------------------------------
# Correctors
# ----------------------------------------------------------------------------
class Corrector(abc.ABC):
    def __init__(self, sde: SDE, score_fn: ScoreFn, snr: float, n_steps: int):
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, cond, t):
        ...


class LangevinCorrector(Corrector):
    def update_fn(self, x, cond, t):
        sde = self.sde
        if isinstance(sde, (VPSDE, subVPSDE)):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep] if isinstance(sde, VPSDE) else torch.ones_like(t)
        else:
            alpha = torch.ones_like(t)

        x_mean = x
        for _ in range(self.n_steps):
            grad = self.score_fn(x, cond, t)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (self.snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise
        return x, x_mean


class NoneCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        pass

    def update_fn(self, x, cond, t):
        return x, x


_CORRECTORS = {
    "langevin": LangevinCorrector,
    "none": NoneCorrector,
}


def _get_predictor(name: str):
    key = str(name or "none").lower()
    if key not in _PREDICTORS:
        raise ValueError(f"Unknown diffusion predictor '{name}'. Expected {sorted(_PREDICTORS)}.")
    return _PREDICTORS[key]


def _get_corrector(name: str):
    key = str(name or "none").lower()
    if key not in _CORRECTORS:
        raise ValueError(f"Unknown diffusion corrector '{name}'. Expected {sorted(_CORRECTORS)}.")
    return _CORRECTORS[key]


# ----------------------------------------------------------------------------
# Samplers
# ----------------------------------------------------------------------------
def get_pc_sampler(
    sde: SDE,
    shape,
    predictor,
    corrector,
    snr: float,
    n_steps: int = 1,
    probability_flow: bool = False,
    denoise: bool = True,
    eps: float = 1e-3,
    device: torch.device | str = "cpu",
):
    """Create a Predictor-Corrector sampler.

    Returns a callable ``sampler(score_fn, cond, generator=None)`` producing a
    sample of shape ``[cond.shape[0], *shape]``.
    """

    def pc_sampler(score_fn: ScoreFn, cond: torch.Tensor, generator=None) -> torch.Tensor:
        with torch.no_grad():
            output_shape = (cond.shape[0], *shape)
            x = sde.prior_sampling(output_shape).to(device=device, dtype=cond.dtype)
            timesteps = torch.linspace(sde.T, eps, sde.N, device=device, dtype=cond.dtype)

            predictor_obj = (
                NonePredictor(sde, score_fn, probability_flow)
                if predictor is None
                else predictor(sde, score_fn, probability_flow)
            )
            corrector_obj = (
                NoneCorrector(sde, score_fn, snr, n_steps)
                if corrector is None
                else corrector(sde, score_fn, snr, n_steps)
            )

            x_mean = x
            for i in range(sde.N):
                vec_t = torch.ones(output_shape[0], device=device, dtype=cond.dtype) * timesteps[i]
                x, x_mean = corrector_obj.update_fn(x, cond, vec_t)
                x, x_mean = predictor_obj.update_fn(x, cond, vec_t)

            return x_mean if denoise else x

    return pc_sampler


def get_ode_sampler(
    sde: SDE,
    shape,
    denoise: bool = False,
    eps: float = 1e-3,
    device: torch.device | str = "cpu",
):
    """Create a black-box probability-flow ODE sampler (fixed-step RK)."""

    def ode_sampler(score_fn: ScoreFn, cond: torch.Tensor, generator=None) -> torch.Tensor:
        with torch.no_grad():
            output_shape = (cond.shape[0], *shape)
            x = sde.prior_sampling(output_shape).to(device=device, dtype=cond.dtype)
            rsde = sde.reverse(score_fn, probability_flow=True)
            n_steps = sde.N
            timesteps = torch.linspace(sde.T, eps, n_steps, device=device, dtype=cond.dtype)
            dt = -(sde.T - eps) / (n_steps - 1) if n_steps > 1 else -(sde.T - eps)
            for i in range(n_steps):
                vec_t = torch.ones(output_shape[0], device=device, dtype=cond.dtype) * timesteps[i]
                drift, _ = rsde.sde(x, cond, vec_t)
                x = x + drift * dt
            return x

    return ode_sampler


def build_sampler(head_config, sde: SDE, shape, device: torch.device | str = "cpu"):
    """Build a sampler callable from a :class:`DiffusionHeadConfig`.

    Args:
        head_config: resolved diffusion config (``sampling_method``, ``predictor`` ...).
        sde: the forward SDE. A shallow copy is used with ``num_sampling_steps``
            discretization so training and sampling step counts can differ.
        shape: per-sample output shape ``(C, H, W)``.
        device: sampling device.
    """
    import copy

    sample_sde = copy.copy(sde)
    sample_sde.N = int(head_config.num_sampling_steps)

    method = str(head_config.sampling_method).lower()
    if method == "ode":
        return get_ode_sampler(
            sample_sde, shape, denoise=head_config.denoise, eps=head_config.sampling_eps, device=device
        )
    if method != "pc":
        raise ValueError(
            f"Unknown diffusion.sampling_method '{head_config.sampling_method}'. Expected 'pc' or 'ode'."
        )

    predictor = _get_predictor(head_config.predictor)
    corrector = _get_corrector(head_config.corrector)
    return get_pc_sampler(
        sample_sde,
        shape,
        predictor,
        corrector,
        snr=head_config.snr,
        n_steps=head_config.n_corrector_steps,
        probability_flow=head_config.probability_flow,
        denoise=head_config.denoise,
        eps=head_config.sampling_eps,
        device=device,
    )
