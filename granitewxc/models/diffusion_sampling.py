"""Reverse-diffusion samplers for the conditional diffusion head.

Adapted from ``mlde``'s ``sampling.py`` (predictor/corrector framework). Only
the pieces required for conditional CORDEX sampling are kept:

    * Euler-Maruyama and reverse-diffusion predictors
    * Langevin and (no-op) correctors
    * a predictor-corrector (PC) sampler and a probability-flow ODE sampler
    * a DDIM (Denoising Diffusion Implicit Model) sampler

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

__all__ = ["build_sampler", "get_pc_sampler", "get_ode_sampler", "get_ddim_sampler"]

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


def get_ddim_sampler(
    sde: SDE,
    shape,
    eta: float = 0.0,
    denoise: bool = True,
    eps: float = 1e-3,
    device: torch.device | str = "cpu",
):
    """DDIM (Denoising Diffusion Implicit Model) sampler.

    Implements the deterministic DDIM update rule for VP/subVP SDEs with an
    optional stochasticity parameter ``eta``:

    * ``eta=0`` is fully deterministic (DDIM proper) and gives the same
      result for every call with the same conditioning — useful when
      ensemble spread is desired only from different input conditions.
    * ``eta=1`` recovers DDPM-like stochastic sampling.
    * Intermediate values blend deterministic and stochastic trajectories
      and are the main lever for tuning ensemble spread.

    For VESDE the sampler falls back to Euler-Maruyama (PC with
    ``corrector=none``).

    Returns a callable ``sampler(score_fn, cond, generator=None)``.
    """
    if isinstance(sde, VESDE):
        # DDIM is defined for VP-family SDEs; fall back to ODE for VESDE
        return get_ode_sampler(sde, shape, denoise=denoise, eps=eps, device=device)

    def _alpha_bar(t: torch.Tensor) -> torch.Tensor:
        """Cumulative signal retention: alpha_bar(t) = exp(2 * log_mean_coeff)."""
        log_mean_coeff = (
            -0.25 * t ** 2 * (sde.beta_1 - sde.beta_0) - 0.5 * t * sde.beta_0
        )
        return torch.exp(2.0 * log_mean_coeff)

    def ddim_sampler(score_fn: ScoreFn, cond: torch.Tensor, generator=None) -> torch.Tensor:
        with torch.no_grad():
            output_shape = (cond.shape[0], *shape)
            x = sde.prior_sampling(output_shape).to(device=device, dtype=cond.dtype)
            timesteps = torch.linspace(sde.T, eps, sde.N, device=device, dtype=cond.dtype)

            for i in range(sde.N):
                t_cur = timesteps[i]
                t_next = timesteps[i + 1] if i + 1 < sde.N else torch.zeros_like(t_cur)

                vec_t = torch.full((output_shape[0],), t_cur, device=device, dtype=cond.dtype)

                # Get noise prediction from score function.  For VP/subVP,
                # score = -z / std, so z = -score * std.
                score = score_fn(x, cond, vec_t)
                std_cur = sde.marginal_prob(torch.zeros_like(x[:, :1, :1, :1]), vec_t)[1]
                std_cur = std_cur.clamp(min=1e-6)
                # noise pred (denoising direction)
                z_pred = -score * std_cur[:, None, None, None]

                ab_cur = _alpha_bar(vec_t).clamp(min=1e-8, max=1.0)
                ab_next = _alpha_bar(
                    torch.full_like(vec_t, t_next)
                ).clamp(min=1e-8, max=1.0)

                # Predicted x_0
                sqrt_ab_cur = ab_cur.sqrt()[:, None, None, None]
                std_cur_ddim = (1.0 - ab_cur).clamp(min=1e-8).sqrt()[:, None, None, None]
                x0_pred = (x - std_cur_ddim * z_pred) / sqrt_ab_cur.clamp(min=1e-8)

                # DDIM variance
                sigma_t = (
                    eta
                    * torch.sqrt(
                        (1.0 - ab_next).clamp(min=0.0)
                        / (1.0 - ab_cur).clamp(min=1e-8)
                    )
                    * torch.sqrt(1.0 - ab_cur / ab_next.clamp(min=1e-8))
                )[:, None, None, None]

                # Direction pointing to x_t
                sqrt_ab_next = ab_next.sqrt()[:, None, None, None]
                mean_next_coeff = torch.sqrt(
                    (1.0 - ab_next - sigma_t ** 2).clamp(min=0.0)
                )

                x = (
                    sqrt_ab_next * x0_pred
                    + mean_next_coeff * z_pred
                )
                if float(eta) > 0.0:
                    noise = (
                        torch.randn_like(x)
                        if generator is None
                        else torch.empty_like(x).normal_(generator=generator)
                    )
                    x = x + sigma_t * noise

            return x

    return ddim_sampler


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

    if method == "ddim":
        eta = float(getattr(head_config, "eta", 0.0))
        return get_ddim_sampler(
            sample_sde,
            shape,
            eta=eta,
            denoise=head_config.denoise,
            eps=head_config.sampling_eps,
            device=device,
        )

    if method == "ode":
        return get_ode_sampler(
            sample_sde, shape, denoise=head_config.denoise, eps=head_config.sampling_eps, device=device
        )
    if method != "pc":
        raise ValueError(
            f"Unknown diffusion.sampling_method '{head_config.sampling_method}'. "
            "Expected 'pc', 'ode', or 'ddim'."
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
