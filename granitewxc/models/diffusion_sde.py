"""Stochastic differential equations for the diffusion downscaling head.

Adapted (minimally) from the score-based generative modelling reference
implementation used by `mlde` (https://github.com/midatm1234/mlde), which in
turn forks Song et al., "Score-Based Generative Modeling through Stochastic
Differential Equations". Only the pieces required for conditional CORDEX
downscaling are kept: the ``VPSDE``/``subVPSDE``/``VESDE`` forward processes and
their reverse-time counterpart.

The classes are self-contained ``nn.Module``-free helpers so that they can be
used both for training-loss computation and for reverse-diffusion sampling.
"""

from __future__ import annotations

import abc
from typing import Any, Mapping

import numpy as np
import torch

__all__ = ["SDE", "VPSDE", "subVPSDE", "VESDE", "build_sde"]


class SDE(abc.ABC):
    """Abstract SDE. All methods operate on a mini-batch of inputs."""

    def __init__(self, N: int):
        super().__init__()
        self.N = int(N)

    @property
    @abc.abstractmethod
    def T(self) -> float:
        """End time of the SDE."""

    @abc.abstractmethod
    def sde(self, x: torch.Tensor, t: torch.Tensor):
        """Return the drift and diffusion coefficients of the forward SDE."""

    @abc.abstractmethod
    def marginal_prob(self, x: torch.Tensor, t: torch.Tensor):
        """Mean and std of the perturbation kernel ``p_t(x)``."""

    @abc.abstractmethod
    def prior_sampling(self, shape) -> torch.Tensor:
        """Draw a sample from the prior ``p_T``."""

    def discretize(self, x: torch.Tensor, t: torch.Tensor):
        """Euler-Maruyama discretization ``x_{i+1} = x_i + f + G z``."""
        dt = 1.0 / self.N
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device, dtype=x.dtype))
        return f, G

    def reverse(self, score_fn, probability_flow: bool = False):
        """Build the reverse-time SDE/ODE from a conditional ``score_fn``.

        ``score_fn`` has signature ``score_fn(x, cond, t) -> score``.
        """
        N = self.N
        T = self.T
        forward_sde = self.sde
        forward_discretize = self.discretize

        class ReverseSDE:
            def __init__(self) -> None:
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self) -> float:
                return T

            def sde(self, x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor):
                drift, diffusion = forward_sde(x, t)
                score = score_fn(x, cond, t)
                drift = drift - diffusion[:, None, None, None] ** 2 * score * (
                    0.5 if self.probability_flow else 1.0
                )
                diffusion = torch.zeros_like(diffusion) if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(self, x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor):
                f, G = forward_discretize(x, t)
                rev_f = f - G[:, None, None, None] ** 2 * score_fn(x, cond, t) * (
                    0.5 if self.probability_flow else 1.0
                )
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return ReverseSDE()


class VPSDE(SDE):
    """Variance-preserving SDE (DDPM-style)."""

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0, N: int = 1000):
        super().__init__(N)
        self.beta_0 = float(beta_min)
        self.beta_1 = float(beta_max)
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1.0 - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    @property
    def T(self) -> float:
        return 1.0

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def discretize(self, x, t):
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        sqrt_beta = torch.sqrt(beta)
        f = torch.sqrt(alpha)[:, None, None, None] * x - x
        G = sqrt_beta
        return f, G


class subVPSDE(SDE):
    """Sub-variance-preserving SDE (default; excels at likelihoods)."""

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0, N: int = 1000):
        super().__init__(N)
        self.beta_0 = float(beta_min)
        self.beta_1 = float(beta_max)

    @property
    def T(self) -> float:
        return 1.0

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x
        discount = 1.0 - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t**2)
        diffusion = torch.sqrt(beta_t * discount)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff)[:, None, None, None] * x
        std = 1 - torch.exp(2.0 * log_mean_coeff)
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)


class VESDE(SDE):
    """Variance-exploding SDE (SMLD/NCSN-style)."""

    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 50.0, N: int = 1000):
        super().__init__(N)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.discrete_sigmas = torch.exp(
            torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N)
        )

    @property
    def T(self) -> float:
        return 1.0

    def sde(self, x, t):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(
            torch.tensor(
                2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                device=t.device,
                dtype=x.dtype,
            )
        )
        return drift, diffusion

    def marginal_prob(self, x, t):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape) * self.sigma_max

    def discretize(self, x, t):
        timestep = (t * (self.N - 1) / self.T).long()
        sigma = self.discrete_sigmas.to(t.device)[timestep]
        adjacent_sigma = torch.where(
            timestep == 0,
            torch.zeros_like(t),
            self.discrete_sigmas.to(t.device)[timestep - 1],
        )
        f = torch.zeros_like(x)
        G = torch.sqrt(sigma**2 - adjacent_sigma**2)
        return f, G


_SDE_ALIASES = {
    "vpsde": "vpsde",
    "vp": "vpsde",
    "ddpm": "vpsde",
    "subvpsde": "subvpsde",
    "subvp": "subvpsde",
    "sub-vp": "subvpsde",
    "vesde": "vesde",
    "ve": "vesde",
    "smld": "vesde",
    "ncsn": "vesde",
}


def build_sde(params: Mapping[str, Any], *, num_scales: int | None = None) -> SDE:
    """Instantiate an SDE from a diffusion-config mapping.

    Args:
        params: mapping with keys such as ``sde``, ``beta_min``, ``beta_max``,
            ``sigma_min``, ``sigma_max`` and ``num_scales``.
        num_scales: optional override for the number of discretization steps.
    """
    name = str(params.get("sde", "subvpsde")).strip().lower()
    canonical = _SDE_ALIASES.get(name)
    if canonical is None:
        raise ValueError(
            f"Unsupported diffusion.sde '{name}'. Expected one of {sorted(set(_SDE_ALIASES))}."
        )

    n = int(
        num_scales
        if num_scales is not None
        else params.get("num_scales", params.get("num_diffusion_steps", 1000))
    )
    if canonical == "vpsde":
        return VPSDE(
            beta_min=float(params.get("beta_min", 0.1)),
            beta_max=float(params.get("beta_max", 20.0)),
            N=n,
        )
    if canonical == "subvpsde":
        return subVPSDE(
            beta_min=float(params.get("beta_min", 0.1)),
            beta_max=float(params.get("beta_max", 20.0)),
            N=n,
        )
    return VESDE(
        sigma_min=float(params.get("sigma_min", 0.01)),
        sigma_max=float(params.get("sigma_max", 50.0)),
        N=n,
    )
