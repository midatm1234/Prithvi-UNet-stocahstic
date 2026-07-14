"""Conditional diffusion (score-based) head for CORDEX-ML downscaling.

This head is an optional, config-selectable alternative to the deterministic
convolutional / pixel-shuffle head. It learns the distribution of the
high-resolution target field conditioned on features produced by the Prithvi
WxC backbone/encoder, following the score-based SDE formulation used by
``mlde`` (https://github.com/midatm1234/mlde).

The head operates purely in *standardized* target space: the enclosing model is
responsible for encoding physical targets into standardized space (training)
and decoding generated samples back to physical units (inference). Keeping the
head scaler-agnostic makes it reusable across the CORDEX models and keeps the
precipitation non-negativity logic in one place (the model decoder).

Public surface:
    * ``DiffusionHeadConfig`` - resolves diffusion parameters from a config.
    * ``build_diffusion_head`` - factory used by the CORDEX models.
    * ``DiffusionHead`` - the ``nn.Module`` with ``training_loss`` / ``sample``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from granitewxc.models.diffusion_sde import SDE, build_sde
from granitewxc.models.diffusion_loss import score_matching_loss
from granitewxc.models.diffusion_sampling import build_sampler

__all__ = [
    "DiffusionHeadConfig",
    "DiffusionHead",
    "build_diffusion_head",
    "GaussianFourierProjection",
    "ConditionalScoreUNet",
]


def _num_groups(channels: int, max_groups: int = 8) -> int:
    """Largest group count <= ``max_groups`` that divides ``channels``."""
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class GaussianFourierProjection(nn.Module):
    """Gaussian random features encoding a scalar noise level / timestep."""

    def __init__(self, embed_dim: int, scale: float = 16.0):
        super().__init__()
        # Fixed (non-trainable) random projection, as in the reference impl.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class _ResBlock(nn.Module):
    """Time-conditioned residual block (GroupNorm + SiLU + Conv)."""

    def __init__(self, in_ch: int, out_ch: int, temb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb_proj = nn.Linear(temb_dim, out_ch)
        self.norm2 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class ConditionalScoreUNet(nn.Module):
    """Resolution-flexible conditional score/noise network.

    The noised target and the conditioning features are concatenated along the
    channel dimension and processed by a small time-conditioned U-Net. Skip
    connections are resized so that arbitrary (non power-of-two) spatial sizes
    are supported; the output is returned at the input spatial resolution.
    """

    def __init__(
        self,
        target_channels: int,
        cond_channels: int,
        base_channels: int = 64,
        channel_multipliers: tuple[int, ...] = (1, 2, 2),
        num_res_blocks: int = 1,
        time_embed_dim: int = 128,
        dropout: float = 0.0,
        fourier_scale: float = 16.0,
    ):
        super().__init__()
        self.target_channels = int(target_channels)
        self.cond_channels = int(cond_channels)

        self.time_embed = nn.Sequential(
            GaussianFourierProjection(time_embed_dim, scale=fourier_scale),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        in_channels = self.target_channels + self.cond_channels
        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        skip_channels = [base_channels]
        ch = base_channels
        for level, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(_ResBlock(ch, out_ch, time_embed_dim, dropout))
                ch = out_ch
                skip_channels.append(ch)
            if level != len(channel_multipliers) - 1:
                self.downsamplers.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                skip_channels.append(ch)
            else:
                self.downsamplers.append(None)

        # Bottleneck
        self.mid_block1 = _ResBlock(ch, ch, time_embed_dim, dropout)
        self.mid_block2 = _ResBlock(ch, ch, time_embed_dim, dropout)

        # Decoder
        self.up_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_multipliers))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(
                    _ResBlock(ch + skip_channels.pop(), out_ch, time_embed_dim, dropout)
                )
                ch = out_ch
            if level != 0:
                self.upsamplers.append(nn.Conv2d(ch, ch, 3, padding=1))
            else:
                self.upsamplers.append(None)

        self.out_norm = nn.GroupNorm(_num_groups(ch), ch)
        self.out_conv = nn.Conv2d(ch, self.target_channels, 3, padding=1)
        self.num_res_blocks = num_res_blocks
        self.channel_multipliers = tuple(channel_multipliers)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        out_size = x.shape[-2:]
        if cond.shape[-2:] != out_size:
            cond = F.interpolate(cond, size=out_size, mode="bilinear", align_corners=False)

        temb = self.time_embed(t)
        h = self.in_conv(torch.cat([x, cond], dim=1))

        skips = [h]
        block_idx = 0
        for level in range(len(self.channel_multipliers)):
            for _ in range(self.num_res_blocks):
                h = self.down_blocks[block_idx](h, temb)
                skips.append(h)
                block_idx += 1
            downsampler = self.downsamplers[level]
            if downsampler is not None:
                h = downsampler(h)
                skips.append(h)

        h = self.mid_block1(h, temb)
        h = self.mid_block2(h, temb)

        block_idx = 0
        for pos, level in enumerate(reversed(range(len(self.channel_multipliers)))):
            for _ in range(self.num_res_blocks + 1):
                skip = skips.pop()
                if skip.shape[-2:] != h.shape[-2:]:
                    h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
                h = self.up_blocks[block_idx](torch.cat([h, skip], dim=1), temb)
                block_idx += 1
            upsampler = self.upsamplers[pos]
            if upsampler is not None:
                h = F.interpolate(h, scale_factor=2, mode="nearest")
                h = upsampler(h)

        h = self.out_conv(F.silu(self.out_norm(h)))
        if h.shape[-2:] != out_size:
            h = F.interpolate(h, size=out_size, mode="bilinear", align_corners=False)
        return h


@dataclass
class DiffusionHeadConfig:
    """Resolved diffusion hyper-parameters."""

    sde: str = "subvpsde"
    beta_min: float = 0.1
    beta_max: float = 20.0
    sigma_min: float = 0.01
    sigma_max: float = 50.0
    num_scales: int = 1000
    continuous: bool = True
    likelihood_weighting: bool = False
    reduce_mean: bool = True
    eps: float = 1e-5
    noise_conditioning_scale: float = 999.0
    # sampling
    sampling_method: str = "pc"
    predictor: str = "euler_maruyama"
    corrector: str = "none"
    snr: float = 0.16
    n_corrector_steps: int = 1
    num_sampling_steps: int = 128
    probability_flow: bool = False
    denoise: bool = True
    sampling_eps: float = 1e-3
    # network
    base_channels: int = 64
    channel_multipliers: tuple[int, ...] = (1, 2, 2)
    num_res_blocks: int = 1
    time_embed_dim: int = 128
    dropout: float = 0.0
    fourier_scale: float = 16.0
    extra: dict[str, Any] = field(default_factory=dict)

    def sde_params(self) -> dict[str, Any]:
        return {
            "sde": self.sde,
            "beta_min": self.beta_min,
            "beta_max": self.beta_max,
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
            "num_scales": self.num_scales,
        }

    @classmethod
    def from_config(cls, config: Any) -> "DiffusionHeadConfig":
        """Build from an ``ExperimentConfig`` (reads ``config.model.diffusion``).

        Both a nested ``model.diffusion`` mapping and flat ``model.*`` keys are
        supported so example YAMLs can use whichever is clearer.
        """
        model_cfg = getattr(config, "model", None)
        block = _coerce_mapping(getattr(model_cfg, "diffusion", None)) if model_cfg else {}
        flat = _coerce_mapping(getattr(model_cfg, "__dict__", None)) if model_cfg else {}

        def pick(key: str, default):
            if key in block:
                return block[key]
            if key in flat:
                return flat[key]
            return default

        defaults = cls()
        num_scales = int(pick("num_scales", pick("num_diffusion_steps", defaults.num_scales)))
        multipliers = pick("channel_multipliers", pick("channel_mults", defaults.channel_multipliers))
        multipliers = tuple(int(m) for m in multipliers)

        known = {
            "sde",
            "beta_min",
            "beta_max",
            "sigma_min",
            "sigma_max",
            "num_scales",
            "num_diffusion_steps",
            "continuous",
            "likelihood_weighting",
            "reduce_mean",
            "eps",
            "noise_conditioning_scale",
            "sampling_method",
            "predictor",
            "corrector",
            "snr",
            "n_corrector_steps",
            "num_sampling_steps",
            "probability_flow",
            "denoise",
            "sampling_eps",
            "base_channels",
            "channel_multipliers",
            "channel_mults",
            "num_res_blocks",
            "time_embed_dim",
            "dropout",
            "fourier_scale",
            "output_channels",
            "conditioning_channels",
        }
        extra = {k: v for k, v in block.items() if k not in known}

        return cls(
            sde=str(pick("sde", defaults.sde)).lower(),
            beta_min=float(pick("beta_min", defaults.beta_min)),
            beta_max=float(pick("beta_max", defaults.beta_max)),
            sigma_min=float(pick("sigma_min", defaults.sigma_min)),
            sigma_max=float(pick("sigma_max", defaults.sigma_max)),
            num_scales=num_scales,
            continuous=bool(pick("continuous", defaults.continuous)),
            likelihood_weighting=bool(pick("likelihood_weighting", defaults.likelihood_weighting)),
            reduce_mean=bool(pick("reduce_mean", defaults.reduce_mean)),
            eps=float(pick("eps", defaults.eps)),
            noise_conditioning_scale=float(
                pick("noise_conditioning_scale", defaults.noise_conditioning_scale)
            ),
            sampling_method=str(pick("sampling_method", defaults.sampling_method)).lower(),
            predictor=str(pick("predictor", defaults.predictor)).lower(),
            corrector=str(pick("corrector", defaults.corrector)).lower(),
            snr=float(pick("snr", defaults.snr)),
            n_corrector_steps=int(pick("n_corrector_steps", defaults.n_corrector_steps)),
            num_sampling_steps=int(pick("num_sampling_steps", defaults.num_sampling_steps)),
            probability_flow=bool(pick("probability_flow", defaults.probability_flow)),
            denoise=bool(pick("denoise", defaults.denoise)),
            sampling_eps=float(pick("sampling_eps", defaults.sampling_eps)),
            base_channels=int(pick("base_channels", defaults.base_channels)),
            channel_multipliers=multipliers,
            num_res_blocks=int(pick("num_res_blocks", defaults.num_res_blocks)),
            time_embed_dim=int(pick("time_embed_dim", defaults.time_embed_dim)),
            dropout=float(pick("dropout", defaults.dropout)),
            fourier_scale=float(pick("fourier_scale", defaults.fourier_scale)),
            extra=extra,
        )


class DiffusionHead(nn.Module):
    """Conditional score-based diffusion head.

    Args:
        cond_channels: number of channels in the conditioning feature map.
        output_channels: number of target variables (e.g. 2 for ``pr``/``tasmax``).
        head_config: resolved :class:`DiffusionHeadConfig`.
    """

    def __init__(
        self,
        cond_channels: int,
        output_channels: int,
        head_config: DiffusionHeadConfig,
    ):
        super().__init__()
        self.cfg = head_config
        self.cond_channels = int(cond_channels)
        self.output_channels = int(output_channels)

        self.score_model = ConditionalScoreUNet(
            target_channels=self.output_channels,
            cond_channels=self.cond_channels,
            base_channels=head_config.base_channels,
            channel_multipliers=head_config.channel_multipliers,
            num_res_blocks=head_config.num_res_blocks,
            time_embed_dim=head_config.time_embed_dim,
            dropout=head_config.dropout,
            fourier_scale=head_config.fourier_scale,
        )
        self.sde: SDE = build_sde(head_config.sde_params(), num_scales=head_config.num_scales)

    # -- score function -----------------------------------------------------
    def score_fn(self, x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return the (real) score ``\\nabla_x log p_t(x | cond)``."""
        from granitewxc.models.diffusion_sde import VESDE

        if isinstance(self.sde, VESDE):
            labels = self.sde.marginal_prob(torch.zeros_like(x), t)[1]
            return self.score_model(x, cond, labels)

        labels = t * self.cfg.noise_conditioning_scale
        model_out = self.score_model(x, cond, labels)
        std = self.sde.marginal_prob(torch.zeros_like(x), t)[1]
        return -model_out / std[:, None, None, None]

    # -- training -----------------------------------------------------------
    def training_loss(self, cond: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Score-matching loss for a standardized ``target`` given ``cond``."""
        return score_matching_loss(
            self.sde,
            self.score_fn,
            target,
            cond,
            reduce_mean=self.cfg.reduce_mean,
            likelihood_weighting=self.cfg.likelihood_weighting,
            eps=self.cfg.eps,
        )

    def forward(self, cond: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`training_loss` (keeps ``nn.Module`` semantics)."""
        return self.training_loss(cond, target)

    # -- sampling -----------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        height: int,
        width: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Reverse-diffusion sample in standardized target space.

        Returns a tensor of shape ``[B, output_channels, height, width]``.
        """
        sampler = build_sampler(
            self.cfg,
            self.sde,
            (self.output_channels, int(height), int(width)),
            device=cond.device,
        )
        return sampler(self.score_fn, cond, generator=generator)


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {}


def build_diffusion_head(
    config: Any,
    cond_channels: int,
    output_channels: int,
) -> DiffusionHead:
    """Factory used by the CORDEX models to build a :class:`DiffusionHead`."""
    head_config = DiffusionHeadConfig.from_config(config)
    return DiffusionHead(
        cond_channels=cond_channels,
        output_channels=output_channels,
        head_config=head_config,
    )
