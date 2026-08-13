"""Bernoulli-Gamma zero-inflated precipitation head for CORDEX-ML.

Formulation
-----------
The precipitation distribution is modelled as a zero-inflated mixture::

    pr ~ (1 - p_wet) * delta(0) + p_wet * Gamma(mu_pos, phi)

where

* ``p_wet``  – probability of a wet day (Bernoulli occurrence parameter).
* ``mu_pos`` – conditional mean precipitation on wet days (Gamma mean).
* ``phi``    – Gamma dispersion (inverse of concentration; phi = 1/k where k
  is the Gamma shape parameter).  A numerically stable *mean–dispersion*
  parameterisation is used instead of raw shape/scale to avoid pathological
  values.

The unconditional expected precipitation is::

    E[pr] = p_wet * mu_pos

Parameterisation outputs from the network:
* logit_wet  → p_wet = sigmoid(logit_wet)
* log_mu_pos → mu_pos = softplus(log_mu_pos) + min_mu
* log_phi    → phi    = softplus(log_phi)    + min_phi

Training uses::

    loss = lambda_occ  * BCE(logit_wet, wet_target)
         + lambda_pos  * Gamma_NLL(pr | mu_pos, phi)   [on wet pixels only]

Inference supports deterministic and stochastic modes:
* deterministic: E[pr] = p_wet * mu_pos
* stochastic:    pr ~ Bernoulli(p_wet) * Gamma(mu_pos, phi)  per pixel

Spatial coherence
-----------------
Independently sampling Bernoulli and Gamma at every pixel produces
salt-and-pepper patterns.  The spatial coherence of sampled fields is
improved by using the network's convolutional structure (spatial context is
already built in) and by optionally applying a spatially-correlated noise
prior via a diffusion step (``spatial_diffusion_steps > 0``).

When ``spatial_diffusion_steps > 0`` the stochastic sample is run through
``spatial_diffusion_steps`` rounds of Gaussian blurring and re-quantisation
to smooth out isolated pixels while preserving wet-day spatial patterns.

Public surface
--------------
* ``BernoulliGammaHead``   – nn.Module decoder head.
* ``BernoulliGammaConfig`` – configuration dataclass.
* ``bernoulli_gamma_nll``  – standalone loss function.
* ``build_bernoulli_gamma_head`` – factory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "BernoulliGammaConfig",
    "BernoulliGammaHead",
    "BernoulliGammaLoss",
    "bernoulli_gamma_nll",
    "build_bernoulli_gamma_head",
]


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------
_LOG_2PI = float(torch.tensor(2.0 * 3.141592653589793).log().item())


def _softplus_eps(x: torch.Tensor, threshold: float = 20.0, min_val: float = 1e-5) -> torch.Tensor:
    """Numerically stable softplus with a minimum value floor."""
    return F.softplus(x, beta=1.0, threshold=threshold).clamp(min=min_val)


def gamma_nll(
    y: torch.Tensor,
    mu: torch.Tensor,
    phi: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Negative log-likelihood of a Gamma(mu, phi) distribution.

    Parameterisation: mean = mu, variance = mu^2 * phi  (phi = 1/k, k = shape).

    NLL = log(phi) + (1/phi - 1)*log(1/phi) - lgamma(1/phi)
          + (1/phi)*log(mu*phi)
          + (1/phi)*y/(mu*phi)   ← Gamma NLL in mean-dispersion form
        = -lgamma(k) + k*log(k/mu) + (k-1)*log(y) - k*y/mu   with k=1/phi
    """
    k = 1.0 / phi.clamp(min=eps)
    y_safe = y.clamp(min=eps)
    nll = (
        torch.lgamma(k)
        - k * torch.log(k / mu.clamp(min=eps))
        - (k - 1.0) * torch.log(y_safe)
        + k * y_safe / mu.clamp(min=eps)
    )
    return nll


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class BernoulliGammaConfig:
    """Resolved Bernoulli-Gamma head configuration."""

    # wet-day threshold (mm/day) below which a sample is considered dry
    wet_threshold: float = 0.1

    # loss weights
    lambda_occurrence: float = 1.0
    lambda_positive_amount: float = 1.0

    # numerical stability floors for Gamma parameters
    min_mu: float = 1e-4   # minimum conditional mean (mm/day)
    min_phi: float = 1e-4  # minimum dispersion

    # output constraint for occurrence probability decision at inference
    occurrence_prob_threshold: float = 0.5

    # number of output parameters per pixel (3: logit_wet, log_mu, log_phi)
    n_params_per_output: int = 3

    # spatial coherence via repeated Gaussian smoothing of stochastic samples
    spatial_diffusion_steps: int = 0
    spatial_diffusion_sigma: float = 1.0

    # optional per-variable gamma NLL weighting
    variable_weights: tuple[float, ...] = field(default_factory=lambda: ())

    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: Any) -> "BernoulliGammaConfig":
        """Build from an ExperimentConfig."""
        raw = {}
        model_cfg = getattr(config, "model", None)
        if model_cfg is not None:
            bg_cfg = getattr(model_cfg, "bernoulli_gamma", None)
            if bg_cfg is not None:
                if hasattr(bg_cfg, "__dict__"):
                    raw = dict(bg_cfg.__dict__)
                elif hasattr(bg_cfg, "items"):
                    raw = dict(bg_cfg)
        # Also accept top-level keys
        for key in (
            "wet_threshold",
            "lambda_occurrence",
            "lambda_positive_amount",
            "occurrence_prob_threshold",
        ):
            if key in raw:
                continue
            val = getattr(config, key, None)
            if val is not None:
                raw[key] = val
        return cls(
            wet_threshold=float(raw.get("wet_threshold", 0.1)),
            lambda_occurrence=float(raw.get("lambda_occurrence", 1.0)),
            lambda_positive_amount=float(raw.get("lambda_positive_amount", 1.0)),
            min_mu=float(raw.get("min_mu", 1e-4)),
            min_phi=float(raw.get("min_phi", 1e-4)),
            occurrence_prob_threshold=float(raw.get("occurrence_prob_threshold", 0.5)),
            spatial_diffusion_steps=int(raw.get("spatial_diffusion_steps", 0)),
            spatial_diffusion_sigma=float(raw.get("spatial_diffusion_sigma", 1.0)),
        )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def bernoulli_gamma_nll(
    logit_wet: torch.Tensor,
    log_mu: torch.Tensor,
    log_phi: torch.Tensor,
    target: torch.Tensor,
    *,
    wet_threshold: float = 0.1,
    lambda_occurrence: float = 1.0,
    lambda_positive_amount: float = 1.0,
    min_mu: float = 1e-4,
    min_phi: float = 1e-4,
    validity_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined zero-inflated Bernoulli-Gamma negative log-likelihood loss.

    Args:
        logit_wet: raw logit for precipitation occurrence ``[*, 1, H, W]``
            or ``[*, H, W]``.
        log_mu: raw log-scale conditional mean ``[*, 1, H, W]`` or ``[*, H, W]``.
        log_phi: raw log-scale dispersion ``[*, 1, H, W]`` or ``[*, H, W]``.
        target: physical precipitation ``[B, H, W]`` or ``[B, 1, H, W]`` in
            the same units as ``mu_pos``.
        wet_threshold: mm/day threshold for wet/dry classification.
        lambda_occurrence: weight on the Bernoulli BCE term.
        lambda_positive_amount: weight on the Gamma NLL term.
        min_mu: minimum allowed conditional mean.
        min_phi: minimum allowed dispersion.
        validity_mask: optional boolean mask ``[B, H, W]``; True = valid pixel.

    Returns:
        Tuple of (total_loss, terms_dict).
    """
    # Flatten optional channel dim
    def _sq(t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 4 and t.shape[1] == 1:
            return t[:, 0]
        return t

    logit_wet = _sq(logit_wet)
    log_mu = _sq(log_mu)
    log_phi = _sq(log_phi)
    target = _sq(target)

    wet_target = (target > wet_threshold).to(dtype=logit_wet.dtype)

    if validity_mask is not None:
        mask = validity_mask.to(device=logit_wet.device, dtype=torch.bool)
        logit_wet = logit_wet[mask]
        log_mu = log_mu[mask]
        log_phi = log_phi[mask]
        target = target[mask]
        wet_target = wet_target[mask]

    # --- occurrence loss ------------------------------------------------
    occ_loss = F.binary_cross_entropy_with_logits(
        logit_wet, wet_target, reduction="mean"
    )

    # --- positive amount loss -------------------------------------------
    mu_pos = _softplus_eps(log_mu, min_val=min_mu)
    phi = _softplus_eps(log_phi, min_val=min_phi)
    wet_mask = wet_target > 0.5
    amount_loss = torch.zeros((), device=logit_wet.device, dtype=logit_wet.dtype)
    if bool(wet_mask.any().item()):
        amount_loss = gamma_nll(
            target[wet_mask],
            mu_pos[wet_mask],
            phi[wet_mask],
        ).mean()

    total = lambda_occurrence * occ_loss + lambda_positive_amount * amount_loss
    terms = {
        "bg.occurrence_bce": float(occ_loss.detach().item()),
        "bg.positive_gamma_nll": float(amount_loss.detach().item()),
        "bg.total": float(total.detach().item()),
    }
    return total, terms


class BernoulliGammaLoss:
    """Callable wrapper around :func:`bernoulli_gamma_nll` for trainer integration."""

    def __init__(self, config: BernoulliGammaConfig, output_vars: list[str] | None = None):
        self.cfg = config
        self.output_vars = list(output_vars or [])
        self._last_terms: dict[str, float] = {}

    def __call__(
        self,
        logit_wet: torch.Tensor,
        log_mu: torch.Tensor,
        log_phi: torch.Tensor,
        target: torch.Tensor,
        validity_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss, terms = bernoulli_gamma_nll(
            logit_wet,
            log_mu,
            log_phi,
            target,
            wet_threshold=self.cfg.wet_threshold,
            lambda_occurrence=self.cfg.lambda_occurrence,
            lambda_positive_amount=self.cfg.lambda_positive_amount,
            min_mu=self.cfg.min_mu,
            min_phi=self.cfg.min_phi,
            validity_mask=validity_mask,
        )
        self._last_terms = terms
        return loss

    def get_last_terms(self) -> dict[str, float]:
        return dict(self._last_terms)


# ---------------------------------------------------------------------------
# Head module
# ---------------------------------------------------------------------------
class BernoulliGammaHead(nn.Module):
    """Bernoulli-Gamma decoder head for zero-inflated precipitation prediction.

    Produces three spatial maps per output variable: ``logit_wet``,
    ``log_mu_pos``, ``log_phi``.  All outputs are in physical units (mm/day
    for the conditional mean); no target normalization is applied because the
    likelihood is defined directly in physical space.

    Args:
        in_channels: number of input feature channels (from upstream decoder).
        n_output_vars: number of precipitation output variables (normally 1).
        config: :class:`BernoulliGammaConfig` with hyper-parameters.
        mid_channels: intermediate channel count; defaults to ``in_channels``.
    """

    N_PARAMS = 3  # logit_wet, log_mu_pos, log_phi

    def __init__(
        self,
        in_channels: int,
        n_output_vars: int = 1,
        config: BernoulliGammaConfig | None = None,
        mid_channels: int = 0,
    ):
        super().__init__()
        self.n_output_vars = int(n_output_vars)
        self.cfg = config or BernoulliGammaConfig()
        mid = mid_channels if mid_channels > 0 else in_channels

        self.conv1 = nn.Conv2d(in_channels, mid, 3, padding=1, padding_mode="replicate")
        self.conv2 = nn.Conv2d(mid, self.N_PARAMS * self.n_output_vars, 3, padding=1)

        nn.init.trunc_normal_(self.conv1.weight, std=0.02)
        nn.init.constant_(self.conv1.bias, 0.0)
        nn.init.trunc_normal_(self.conv2.weight, std=0.02)
        nn.init.constant_(self.conv2.bias, 0.0)

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(logit_wet, log_mu_pos, log_phi)`` each ``[B, V, H, W]``."""
        h = F.leaky_relu(self.conv1(features))
        out = self.conv2(h)  # [B, 3*V, H, W]
        # Split into 3 parameter tensors, each [B, V, H, W]
        logit_wet, log_mu, log_phi = out.chunk(3, dim=1)
        return logit_wet, log_mu, log_phi

    # -- inference helpers --------------------------------------------------
    def decode_deterministic(
        self,
        logit_wet: torch.Tensor,
        log_mu: torch.Tensor,
        log_phi: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return deterministic expected precipitation and auxiliary maps.

        Returns:
            Dictionary with keys:
            * ``pr_expected``   – E[pr] = p_wet * mu_pos  ``[B, V, H, W]``
            * ``p_wet``         – occurrence probability  ``[B, V, H, W]``
            * ``mu_pos``        – conditional mean        ``[B, V, H, W]``
            * ``phi``           – dispersion              ``[B, V, H, W]``
            * ``variance``      – unconditional variance  ``[B, V, H, W]``
        """
        p_wet = torch.sigmoid(logit_wet)
        mu_pos = _softplus_eps(log_mu, min_val=self.cfg.min_mu)
        phi = _softplus_eps(log_phi, min_val=self.cfg.min_phi)

        pr_expected = p_wet * mu_pos
        # Var[pr] = p*(1-p)*mu^2 + p*Var[Gamma] = p*(1-p)*mu^2 + p*mu^2*phi
        variance = p_wet * (1.0 - p_wet) * mu_pos ** 2 + p_wet * mu_pos ** 2 * phi

        return {
            "pr_expected": pr_expected,
            "p_wet": p_wet,
            "mu_pos": mu_pos,
            "phi": phi,
            "variance": variance,
        }

    def sample_stochastic(
        self,
        logit_wet: torch.Tensor,
        log_mu: torch.Tensor,
        log_phi: torch.Tensor,
        generator: torch.Generator | None = None,
        n_members: int = 1,
    ) -> torch.Tensor:
        """Draw stochastic ensemble samples.

        Returns ``[B, n_members, V, H, W]`` precipitation fields in physical
        units (mm/day).

        Spatial coherence is improved compared to independent pixel sampling
        because the network itself uses spatial convolutions (neighbours share
        information), so the predicted parameter maps are already spatially
        smooth.  Additional coherence smoothing can be applied with
        ``cfg.spatial_diffusion_steps > 0``.
        """
        p_wet = torch.sigmoid(logit_wet)           # [B, V, H, W]
        mu_pos = _softplus_eps(log_mu, min_val=self.cfg.min_mu)
        phi = _softplus_eps(log_phi, min_val=self.cfg.min_phi)

        B, V, H, W = p_wet.shape
        members: list[torch.Tensor] = []

        for _ in range(n_members):
            # Occurrence
            if generator is not None:
                u = torch.empty_like(p_wet).uniform_(generator=generator)
            else:
                u = torch.rand_like(p_wet)
            wet = (u < p_wet).to(dtype=p_wet.dtype)

            # Positive amounts from Gamma(mu, phi) via shape-rate
            k = 1.0 / phi.clamp(min=1e-8)     # shape
            rate = k / mu_pos.clamp(min=1e-8)  # rate = k / mu
            # Sample using PyTorch Gamma distribution
            try:
                dist = torch.distributions.Gamma(
                    concentration=k.detach(),
                    rate=rate.detach(),
                )
                amounts = dist.sample().to(device=p_wet.device, dtype=p_wet.dtype)
            except Exception:
                # Fallback: method of moments reparameterisation via log-normal
                log_var = torch.log(1.0 + phi)
                log_mean = torch.log(mu_pos) - 0.5 * log_var
                if generator is not None:
                    z = torch.empty_like(p_wet).normal_(generator=generator)
                else:
                    z = torch.randn_like(p_wet)
                amounts = torch.exp(log_mean + log_var.sqrt() * z)

            sample = wet * amounts  # [B, V, H, W]

            if self.cfg.spatial_diffusion_steps > 0:
                sample = self._apply_spatial_smoothing(sample)

            members.append(sample)

        return torch.stack(members, dim=1)  # [B, n_members, V, H, W]

    def _apply_spatial_smoothing(self, x: torch.Tensor) -> torch.Tensor:
        """Apply light Gaussian smoothing to reduce salt-and-pepper artifacts."""
        sigma = self.cfg.spatial_diffusion_sigma
        if sigma <= 0.0:
            return x
        B, V, H, W = x.shape
        # Build 2D Gaussian kernel
        ks = max(3, int(2 * round(2 * sigma) + 1))  # kernel size, always odd
        ks = ks if ks % 2 == 1 else ks + 1
        half = ks // 2
        g1d = torch.exp(-0.5 * (torch.arange(ks, device=x.device, dtype=x.dtype) - half) ** 2 / sigma ** 2)
        g1d = g1d / g1d.sum()
        kernel = g1d.unsqueeze(0) * g1d.unsqueeze(1)  # [ks, ks]
        kernel = kernel.unsqueeze(0).unsqueeze(0)      # [1, 1, ks, ks]
        for _ in range(self.cfg.spatial_diffusion_steps):
            x_reshaped = x.view(B * V, 1, H, W)
            x_smoothed = F.conv2d(x_reshaped, kernel, padding=half)
            x = x_smoothed.view(B, V, H, W)
        return x

    # -- combined forward for inference -----------------------------------
    def predict(
        self,
        features: torch.Tensor,
        stochastic: bool = False,
        n_members: int = 1,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Full prediction pipeline.

        Returns a dict with all relevant outputs.
        """
        logit_wet, log_mu, log_phi = self.forward(features)
        result = self.decode_deterministic(logit_wet, log_mu, log_phi)
        result["logit_wet"] = logit_wet
        result["log_mu"] = log_mu
        result["log_phi"] = log_phi
        if stochastic:
            samples = self.sample_stochastic(
                logit_wet, log_mu, log_phi,
                generator=generator,
                n_members=n_members,
            )
            result["pr_samples"] = samples
        return result


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_bernoulli_gamma_head(
    config: Any,
    in_channels: int,
    n_output_vars: int = 1,
) -> BernoulliGammaHead:
    """Factory used by CORDEX model classes."""
    bg_config = BernoulliGammaConfig.from_config(config)
    return BernoulliGammaHead(
        in_channels=in_channels,
        n_output_vars=n_output_vars,
        config=bg_config,
    )
