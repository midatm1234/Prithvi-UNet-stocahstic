"""Stochastic-process time embeddings and diffusion noise schedules.

Everything in this module is a *mathematical* process variable:

* ``t`` in the diffusion refiners is a discrete noise-process index in
  ``[0, training_timesteps)``.
* ``t`` in the flow-matching refiners is a continuous interpolation coordinate
  in ``[0, 1]``.

Neither is a physical timestamp, a forecast lead time or a sequence position.
Schedules are constructed once per configuration and cached as buffers on the
owning module so no host/device synchronisation happens inside sampling loops.
"""

from __future__ import annotations

import math

import torch
from torch import nn

__all__ = [
    "sinusoidal_embedding",
    "GaussianFourierProjection",
    "ProcessTimeEmbedding",
    "DiffusionSchedule",
]


def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    """Standard transformer sinusoidal embedding of a scalar process time.

    Args:
        t: ``[N]`` tensor of process-time values (any real scale).
        dim: even embedding width.
        max_period: largest sinusoid period.

    Returns:
        ``[N, dim]`` embedding.
    """
    if dim % 2 != 0:
        raise ValueError(f"sinusoidal_embedding requires an even dim, got {dim}")
    t = t.reshape(-1).float()
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class GaussianFourierProjection(nn.Module):
    """Random Fourier features of the process time.

    Ported from ``CORDEX_ML_diffusion_head``'s score-network time embedding and
    kept as an alternative to the deterministic sinusoidal embedding.
    """

    def __init__(self, embed_dim: int, scale: float = 16.0) -> None:
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(f"GaussianFourierProjection requires an even embed_dim, got {embed_dim}")
        self.register_buffer("W", torch.randn(embed_dim // 2) * scale, persistent=True)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        proj = t.reshape(-1)[:, None].float() * self.W[None, :] * 2 * math.pi
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class ProcessTimeEmbedding(nn.Module):
    """MLP-projected embedding of the diffusion timestep / flow interpolation time."""

    def __init__(self, dim: int, kind: str = "sinusoidal", fourier_scale: float = 16.0) -> None:
        super().__init__()
        kind = str(kind).lower()
        if kind not in {"sinusoidal", "fourier"}:
            raise ValueError(f"Unsupported process-time embedding {kind!r}")
        self.kind = kind
        self.dim = int(dim)
        if self.dim % 2 != 0:
            raise ValueError(f"ProcessTimeEmbedding dim must be even, got {dim}")
        self.fourier = GaussianFourierProjection(self.dim, scale=fourier_scale) if kind == "fourier" else None
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        base = self.fourier(t) if self.fourier is not None else sinusoidal_embedding(t, self.dim)
        return self.mlp(base.to(self.mlp[0].weight.dtype))


class DiffusionSchedule(nn.Module):
    """Discrete DDPM noise schedule with DDIM-compatible reverse sampling.

    All coefficient tensors are registered as non-persistent buffers so they
    live on the module's device and are never rebuilt inside a sampling loop.
    They are deterministic functions of the configuration, so excluding them
    from the state dict keeps refinement checkpoints small and portable.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        cosine_s: float = 8e-3,
    ) -> None:
        super().__init__()
        self.num_train_timesteps = int(num_train_timesteps)
        self.schedule = str(schedule).lower()

        betas = self._build_betas(
            self.num_train_timesteps, self.schedule, beta_start, beta_end, cosine_s
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt(), persistent=False)
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt(), persistent=False
        )

    @staticmethod
    def _build_betas(
        num_steps: int, schedule: str, beta_start: float, beta_end: float, cosine_s: float
    ) -> torch.Tensor:
        if schedule == "linear":
            return torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float64).float()
        if schedule == "scaled_linear":
            return (
                torch.linspace(beta_start**0.5, beta_end**0.5, num_steps, dtype=torch.float64) ** 2
            ).float()
        if schedule == "cosine":
            steps = torch.arange(num_steps + 1, dtype=torch.float64) / num_steps
            f = torch.cos((steps + cosine_s) / (1.0 + cosine_s) * math.pi * 0.5) ** 2
            alphas_cumprod = f / f[0]
            betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            return betas.clamp(1e-8, 0.999).float()
        raise ValueError(f"Unsupported diffusion schedule {schedule!r}")

    # -- forward (noising) process --------------------------------------
    def add_noise(
        self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """``q(x_t | x_0)``: ``sqrt(a_bar) * x_0 + sqrt(1 - a_bar) * eps``."""
        shape = (-1,) + (1,) * (clean.ndim - 1)
        s_a = self.sqrt_alphas_cumprod.to(clean.device)[timesteps].reshape(shape).to(clean.dtype)
        s_1ma = (
            self.sqrt_one_minus_alphas_cumprod.to(clean.device)[timesteps]
            .reshape(shape)
            .to(clean.dtype)
        )
        return s_a * clean + s_1ma * noise

    def velocity_target(
        self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """``v = sqrt(a_bar) * eps - sqrt(1 - a_bar) * x_0`` (Salimans & Ho)."""
        shape = (-1,) + (1,) * (clean.ndim - 1)
        s_a = self.sqrt_alphas_cumprod.to(clean.device)[timesteps].reshape(shape).to(clean.dtype)
        s_1ma = (
            self.sqrt_one_minus_alphas_cumprod.to(clean.device)[timesteps]
            .reshape(shape)
            .to(clean.dtype)
        )
        return s_a * noise - s_1ma * clean

    def training_target(
        self,
        prediction_type: str,
        clean: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if prediction_type == "epsilon":
            return noise
        if prediction_type == "sample":
            return clean
        if prediction_type == "velocity":
            return self.velocity_target(clean, noise, timesteps)
        raise ValueError(f"Unsupported prediction_type {prediction_type!r}")

    # -- reverse (denoising) process ------------------------------------
    def to_clean(
        self,
        prediction_type: str,
        model_output: torch.Tensor,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Convert any supported network parameterisation to the clean ``x_0``."""
        shape = (-1,) + (1,) * (noisy.ndim - 1)
        s_a = self.sqrt_alphas_cumprod.to(noisy.device)[timesteps].reshape(shape).to(noisy.dtype)
        s_1ma = (
            self.sqrt_one_minus_alphas_cumprod.to(noisy.device)[timesteps]
            .reshape(shape)
            .to(noisy.dtype)
        )
        if prediction_type == "sample":
            return model_output
        if prediction_type == "epsilon":
            return (noisy - s_1ma * model_output) / s_a.clamp(min=1e-12)
        if prediction_type == "velocity":
            return s_a * noisy - s_1ma * model_output
        raise ValueError(f"Unsupported prediction_type {prediction_type!r}")

    def to_epsilon(
        self,
        prediction_type: str,
        model_output: torch.Tensor,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        shape = (-1,) + (1,) * (noisy.ndim - 1)
        s_a = self.sqrt_alphas_cumprod.to(noisy.device)[timesteps].reshape(shape).to(noisy.dtype)
        s_1ma = (
            self.sqrt_one_minus_alphas_cumprod.to(noisy.device)[timesteps]
            .reshape(shape)
            .to(noisy.dtype)
        )
        if prediction_type == "epsilon":
            return model_output
        if prediction_type == "sample":
            return (noisy - s_a * model_output) / s_1ma.clamp(min=1e-12)
        if prediction_type == "velocity":
            return s_a * model_output + s_1ma * noisy
        raise ValueError(f"Unsupported prediction_type {prediction_type!r}")

    def inference_timesteps(self, num_inference_steps: int, device: torch.device) -> torch.Tensor:
        """Descending DDIM timestep grid, built once per sampling call."""
        num_inference_steps = int(num_inference_steps)
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be >= 1")
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"num_inference_steps ({num_inference_steps}) exceeds "
                f"num_train_timesteps ({self.num_train_timesteps})"
            )
        stride = self.num_train_timesteps / num_inference_steps
        steps = (torch.arange(num_inference_steps, dtype=torch.float64) * stride).round().long()
        return steps.flip(0).to(device)
