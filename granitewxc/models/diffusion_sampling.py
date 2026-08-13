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
import copy
from typing import Callable

import numpy as np
import torch

from granitewxc.models.diffusion_sde import SDE, VPSDE, VESDE, subVPSDE

__all__ = ["build_sampler", "get_pc_sampler", "get_ode_sampler", "get_ddim_sampler"]

ScoreFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def _randn_like(
    value: torch.Tensor,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if generator is None:
        return torch.randn_like(value)
    return torch.empty_like(value).normal_(generator=generator)


def _draw_prior(
    sde: SDE,
    output_shape,
    *,
    generator: torch.Generator | None,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    # Preserve the historical/global-RNG path for custom SDEs and tests when no
    # explicit generator is requested. Qualification always supplies one.
    if generator is None:
        return sde.prior_sampling(output_shape).to(device=device, dtype=dtype)
    try:
        return sde.prior_sampling(
            output_shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )
    except TypeError:
        # Backward compatibility for tests/custom SDEs that override
        # ``prior_sampling(shape)`` with a fixed or legacy callable. Built-in
        # SDEs accept the explicit generator and are used for qualification.
        return sde.prior_sampling(output_shape).to(device=device, dtype=dtype)


# ----------------------------------------------------------------------------
# Predictors
# ----------------------------------------------------------------------------
class Predictor(abc.ABC):
    def __init__(self, sde: SDE, score_fn: ScoreFn, probability_flow: bool = False):
        self.sde = sde
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, cond, t, generator=None):
        ...


class EulerMaruyamaPredictor(Predictor):
    def update_fn(self, x, cond, t, generator=None):
        dt = -1.0 / self.rsde.N
        z = _randn_like(x, generator)
        drift, diffusion = self.rsde.sde(x, cond, t)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
        return x, x_mean


class ReverseDiffusionPredictor(Predictor):
    def update_fn(self, x, cond, t, generator=None):
        f, G = self.rsde.discretize(x, cond, t)
        z = _randn_like(x, generator)
        x_mean = x - f
        x = x_mean + G[:, None, None, None] * z
        return x, x_mean


class NonePredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        self.sde = sde
        self.score_fn = score_fn

    def update_fn(self, x, cond, t, generator=None):
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
    def update_fn(self, x, cond, t, generator=None):
        ...


class LangevinCorrector(Corrector):
    def update_fn(self, x, cond, t, generator=None):
        sde = self.sde
        if isinstance(sde, (VPSDE, subVPSDE)):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep] if isinstance(sde, VPSDE) else torch.ones_like(t)
        else:
            alpha = torch.ones_like(t)

        x_mean = x
        for _ in range(self.n_steps):
            grad = self.score_fn(x, cond, t)
            noise = _randn_like(x, generator)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (self.snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise
        return x, x_mean


class NoneCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        pass

    def update_fn(self, x, cond, t, generator=None):
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
            x = _draw_prior(
                sde,
                output_shape,
                generator=generator,
                device=device,
                dtype=cond.dtype,
            )
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
                if generator is None:
                    x, x_mean = corrector_obj.update_fn(x, cond, vec_t)
                    x, x_mean = predictor_obj.update_fn(x, cond, vec_t)
                else:
                    x, x_mean = corrector_obj.update_fn(
                        x, cond, vec_t, generator=generator
                    )
                    x, x_mean = predictor_obj.update_fn(
                        x, cond, vec_t, generator=generator
                    )

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
            x = _draw_prior(
                sde,
                output_shape,
                generator=generator,
                device=device,
                dtype=cond.dtype,
            )
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
    optional stochasticity parameter ``eta`` for VPSDE:

    * ``eta=0`` makes each reverse trajectory deterministic conditional on its
      initial prior sample. Independent prior samples still produce an ensemble.
    * For VPSDE, ``eta=1`` recovers the standard DDPM-like variance.
    * Intermediate values blend deterministic and stochastic trajectories
      and are the main lever for tuning ensemble spread.

    A stochastic DDIM transition is not enabled for ``subVPSDE``. Its marginal
    noise standard deviation is ``1 - alpha_bar`` rather than
    ``sqrt(1 - alpha_bar)``, so the standard VP/DDPM posterior-variance formula
    does not apply. Select ``eta=0`` for subVP instead of silently applying VP
    coefficients to a different forward process.

    For VESDE the sampler falls back to Euler-Maruyama (PC with
    ``corrector=none``).

    The x0 reconstruction and variance coefficients are computed from the
    SDE's own ``marginal_prob`` so that the formulas are correct for both
    VPSDE and subVPSDE.  (For VPSDE the marginal std is
    ``sqrt(1 - exp(2*lambda))``; for subVPSDE it is ``1 - exp(2*lambda)``
    — using the VP formula for subVP gives wrong x0 predictions and DDIM
    update directions, producing large spatially-coherent output biases.)

    Returns a callable ``sampler(score_fn, cond, generator=None)``.
    """
    if isinstance(sde, VESDE):
        # DDIM is defined for VP-family SDEs; fall back to ODE for VESDE
        return get_ode_sampler(sde, shape, denoise=denoise, eps=eps, device=device)

    eta = float(eta)
    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"DDIM eta must lie in [0, 1], got {eta}.")
    if isinstance(sde, subVPSDE) and eta != 0.0:
        raise ValueError(
            "Stochastic DDIM (eta != 0) is only implemented for VPSDE. "
            "subVPSDE uses a different marginal variance; set diffusion.eta=0 "
            "or select sde='vpsde'."
        )

    def _marginal(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mean_coeff, std) for the SDE's marginal p_t(x).

        mean_coeff : sqrt(alpha_bar(t)),  shape [B]
        std        : noise std,           shape [B]

        Using sde.marginal_prob directly ensures correctness for both VPSDE
        (std = sqrt(1-alpha_bar)) and subVPSDE (std = 1-alpha_bar).
        """
        # marginal_prob returns ``mean_coeff * x``, not ``mean_coeff`` itself.
        # A zero probe therefore destroys the coefficient (it returns all
        # zeros). A unit probe recovers the multiplicative signal coefficient.
        unit_probe = torch.ones(t.shape[0], 1, 1, 1, device=t.device, dtype=t.dtype)
        mean_vec, std_vec = sde.marginal_prob(unit_probe, t)
        mean_coeff = mean_vec[:, 0, 0, 0]
        return mean_coeff, std_vec  # std_vec shape [B]

    def ddim_sampler(score_fn: ScoreFn, cond: torch.Tensor, generator=None) -> torch.Tensor:
        with torch.no_grad():
            output_shape = (cond.shape[0], *shape)
            x = _draw_prior(
                sde,
                output_shape,
                generator=generator,
                device=device,
                dtype=cond.dtype,
            )
            # Compute exp/subtraction-based SDE coefficients in at least fp32.
            # In bf16/fp16, subVP's ``1 - exp(...)`` rounds to zero at the final
            # low-noise timesteps and corrupts x0 reconstruction.
            time_dtype = torch.float64 if cond.dtype == torch.float64 else torch.float32
            timesteps = torch.linspace(sde.T, eps, sde.N, device=device, dtype=time_dtype)

            for i in range(sde.N):
                t_cur = timesteps[i]
                t_next = (
                    timesteps[i + 1]
                    if i + 1 < sde.N
                    else (torch.zeros_like(t_cur) if denoise else t_cur)
                )

                vec_t_cur = t_cur.expand(output_shape[0])
                vec_t_next = t_next.expand(output_shape[0])

                # Get marginal statistics from the SDE (correct for VP and subVP).
                mean_cur, std_cur = _marginal(vec_t_cur)
                mean_next, std_next = _marginal(vec_t_next)

                std_cur_s = std_cur.clamp(min=1e-12)
                mean_cur_s = mean_cur.clamp(min=1e-12)
                std_next_s = std_next.clamp(min=0.0)
                mean_next_s = mean_next.clamp(min=1e-12)

                # Get score; convert to noise prediction.
                # score = -z / std  =>  z = -score * std
                score = score_fn(x, cond, vec_t_cur)
                z_pred = -score * std_cur_s[:, None, None, None]

                # Predicted x_0 using the SDE's own mean/std coefficients.
                # x_t = mean_coeff * x0 + std * z  =>  x0 = (x_t - std*z) / mean_coeff
                x0_pred = (
                    x - std_cur_s[:, None, None, None] * z_pred
                ) / mean_cur_s[:, None, None, None]

                if eta == 0.0:
                    sigma_t = torch.zeros_like(std_next_s)[:, None, None, None]
                else:
                    # Standard VPSDE/DDIM posterior variance:
                    # sigma^2 = eta^2 * (1-a_next^2)/(1-a_cur^2)
                    #                    * (1-a_cur^2/a_next^2)
                    # where a=sqrt(alpha_bar). For VP, std^2=1-a^2.
                    signal_ratio_sq = torch.square(mean_cur_s / mean_next_s)
                    sigma_sq = (
                        eta**2
                        * torch.square(std_next_s / std_cur_s)
                        * (1.0 - signal_ratio_sq).clamp(min=0.0)
                    )
                    sigma_t = torch.sqrt(sigma_sq.clamp(min=0.0))[:, None, None, None]

                # Coefficient for the noise direction term.
                # Deterministic direction: sqrt(std_next^2 - sigma_t^2)
                noise_dir_coeff = torch.sqrt(
                    (std_next_s[:, None, None, None] ** 2 - sigma_t ** 2).clamp(min=0.0)
                )

                x = (
                    mean_next_s[:, None, None, None] * x0_pred
                    + noise_dir_coeff * z_pred
                )
                if float(eta) > 0.0:
                    noise = _randn_like(x, generator)
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
    num_sampling_steps = int(head_config.num_sampling_steps)
    if type(sde) is VPSDE:
        # Rebuild discrete schedules for the requested reverse-step count.
        # A shallow copy with only N changed leaves beta/alpha arrays at the
        # training length and makes PC discretization internally inconsistent.
        sample_sde = VPSDE(
            beta_min=sde.beta_0,
            beta_max=sde.beta_1,
            N=num_sampling_steps,
        )
    elif type(sde) is subVPSDE:
        sample_sde = subVPSDE(
            beta_min=sde.beta_0,
            beta_max=sde.beta_1,
            N=num_sampling_steps,
        )
    elif type(sde) is VESDE:
        sample_sde = VESDE(
            sigma_min=sde.sigma_min,
            sigma_max=sde.sigma_max,
            N=num_sampling_steps,
        )
    else:
        # Preserve custom SDE/subclass behavior. Such implementations are
        # responsible for rebuilding any N-dependent arrays themselves.
        sample_sde = copy.copy(sde)
        sample_sde.N = num_sampling_steps

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
