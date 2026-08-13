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

**Residual diffusion mode** (``residual_diffusion=True``):
When enabled the head generates a *residual* correction relative to a provided
deterministic baseline rather than the full target from noise.  The baseline
(in standardized space) is concatenated to the conditioning features. This
decomposes the learning problem, but does not by itself guarantee improvement;
the decoded ensemble mean must pass the configured held-out loss gate.

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
    "ResidualMeanPredictor",
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

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        temb_dim: int,
        dropout: float = 0.0,
        padding_mode: str = "replicate",
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, 3, padding=1, padding_mode=padding_mode
        )
        self.temb_proj = nn.Linear(temb_dim, out_ch)
        self.norm2 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(
            out_ch, out_ch, 3, padding=1, padding_mode=padding_mode
        )
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

    When ``cond_channels`` is large (e.g. 640 from the CORDEX UNet decoder) a
    lightweight ``cond_proj`` layer first projects conditioning to
    ``projected_cond_channels`` (default = 2 × ``base_channels``) before
    concatenation, preventing the first conv from being overwhelmed by an
    excessively wide input.
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
        projected_cond_channels: int = 0,
        padding_mode: str = "replicate",
        zero_init_output: bool = True,
    ):
        super().__init__()
        self.target_channels = int(target_channels)
        self.cond_channels = int(cond_channels)

        # Conditioning projection: if cond_channels is large, project down first
        # so the score network isn't dominated by conditioning at the first layer.
        if projected_cond_channels > 0 and cond_channels > projected_cond_channels:
            self.cond_proj: nn.Module = nn.Sequential(
                nn.Conv2d(cond_channels, projected_cond_channels, 1),
                nn.SiLU(),
            )
            effective_cond = projected_cond_channels
        else:
            self.cond_proj = nn.Identity()
            effective_cond = cond_channels

        self.time_embed = nn.Sequential(
            GaussianFourierProjection(time_embed_dim, scale=fourier_scale),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        in_channels = self.target_channels + effective_cond
        self.in_conv = nn.Conv2d(
            in_channels,
            base_channels,
            3,
            padding=1,
            padding_mode=padding_mode,
        )

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        skip_channels = [base_channels]
        ch = base_channels
        for level, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    _ResBlock(
                        ch,
                        out_ch,
                        time_embed_dim,
                        dropout,
                        padding_mode=padding_mode,
                    )
                )
                ch = out_ch
                skip_channels.append(ch)
            if level != len(channel_multipliers) - 1:
                self.downsamplers.append(
                    nn.Conv2d(
                        ch,
                        ch,
                        3,
                        stride=2,
                        padding=1,
                        padding_mode=padding_mode,
                    )
                )
                skip_channels.append(ch)
            else:
                self.downsamplers.append(None)

        # Bottleneck
        self.mid_block1 = _ResBlock(
            ch, ch, time_embed_dim, dropout, padding_mode=padding_mode
        )
        self.mid_block2 = _ResBlock(
            ch, ch, time_embed_dim, dropout, padding_mode=padding_mode
        )

        # Decoder
        self.up_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_multipliers))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(
                    _ResBlock(
                        ch + skip_channels.pop(),
                        out_ch,
                        time_embed_dim,
                        dropout,
                        padding_mode=padding_mode,
                    )
                )
                ch = out_ch
            if level != 0:
                self.upsamplers.append(
                    nn.Conv2d(
                        ch,
                        ch,
                        3,
                        padding=1,
                        padding_mode=padding_mode,
                    )
                )
            else:
                self.upsamplers.append(None)

        self.out_norm = nn.GroupNorm(_num_groups(ch), ch)
        self.out_conv = nn.Conv2d(
            ch,
            self.target_channels,
            3,
            padding=1,
            padding_mode=padding_mode,
        )
        if zero_init_output:
            nn.init.zeros_(self.out_conv.weight)
            if self.out_conv.bias is not None:
                nn.init.zeros_(self.out_conv.bias)
        self.num_res_blocks = num_res_blocks
        self.channel_multipliers = tuple(channel_multipliers)
        self.padding_mode = padding_mode
        self.zero_init_output = bool(zero_init_output)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        out_size = x.shape[-2:]
        if cond.shape[-2:] != out_size:
            cond = F.interpolate(cond, size=out_size, mode="bilinear", align_corners=False)

        cond = self.cond_proj(cond)

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


class ResidualMeanPredictor(nn.Module):
    """Directly supervised conditional mean of the U-Net residual.

    Score matching learns stochastic innovations around this field. Keeping
    the correction mean in a separate, zero-initialized network gives residual
    diffusion an explicit deterministic path that can be compared with the
    paired U-Net using the configured task loss.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        *,
        padding_mode: str,
    ) -> None:
        super().__init__()
        hidden_channels = int(hidden_channels)
        if hidden_channels < 1:
            raise ValueError("residual_mean_channels must be positive")
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.SiLU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                3,
                padding=1,
                padding_mode=padding_mode,
            ),
            nn.GroupNorm(_num_groups(hidden_channels), hidden_channels),
            nn.SiLU(),
            nn.Conv2d(
                hidden_channels,
                out_channels,
                3,
                padding=1,
                padding_mode=padding_mode,
            ),
        )
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        if final.bias is not None:
            nn.init.zeros_(final.bias)

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        return self.net(conditioning)


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
    # Preserve the historical default for legacy full-field diffusion configs.
    # New residual configs should set 1.0 explicitly when continuous SDE time
    # t in [0, 1] is intended (as the active CORDEX residual YAML does).
    noise_conditioning_scale: float = 999.0
    # sampling
    sampling_method: str = "pc"
    predictor: str = "euler_maruyama"
    corrector: str = "none"
    snr: float = 0.16
    n_corrector_steps: int = 1
    num_sampling_steps: int = 256
    probability_flow: bool = False
    denoise: bool = True
    sampling_eps: float = 1e-3
    # DDIM stochasticity: 0 = fully deterministic, 1 = DDPM-equivalent
    eta: float = 0.0
    # conditioning projection (0 = no projection)
    projected_cond_channels: int = 128
    # residual diffusion: generate residual around deterministic baseline
    residual_diffusion: bool = False
    # Decompose the residual into a directly supervised conditional mean and a
    # stochastic innovation relative to that mean. Disabled by default for checkpoint
    # compatibility; production residual configs must opt in explicitly.
    residual_mean_enabled: bool = False
    residual_mean_channels: int = 64
    # Safe deployment gate for sampled residuals. Zero exactly recovers the
    # deterministic baseline; one applies the full learned correction.
    residual_application_scale: float = 0.0
    # The score network predicts forward-process Gaussian noise (epsilon).
    # Other parameterizations are rejected until both loss and sampler support them.
    prediction_type: str = "epsilon"
    # Optional clean-residual reconstruction derived from epsilon_theta. The
    # inverse-SNR cap bounds training gradients only; samples are never clipped.
    clean_x0_reconstruction_weight: float = 0.0
    clean_x0_inverse_snr_cap: float = 100.0
    # Hard inference guard against corrections far outside the observed
    # training-residual scale. A non-positive value disables the guard.
    residual_magnitude_guard_multiple: float = 5.0
    residual_guard_min_count: int = 1024
    # How to respond when a generated correction exceeds the guard threshold:
    #   'reject' – raise RuntimeError (original behavior; hard gate at deployment)
    #   'clip'   – clamp per-channel RMS to the guard limit, preserve direction
    #   'warn'   – log a warning and pass the correction unchanged
    # Any mode that does NOT raise still records the event in the returned
    # metadata via _last_residual_clamp_applied.
    residual_guard_mode: str = "reject"
    # network
    base_channels: int = 64
    channel_multipliers: tuple[int, ...] = (1, 2, 2)
    num_res_blocks: int = 2
    time_embed_dim: int = 128
    dropout: float = 0.1
    fourier_scale: float = 16.0
    # Preserve the historical zero-padding default for legacy full-field
    # checkpoints whose YAML predates this option. Regional residual configs
    # must opt into ``replicate`` explicitly and record it in checkpoint metadata.
    padding_mode: str = "zeros"
    # A zero initial epsilon prediction is conservative. Loading a checkpoint
    # overwrites these parameters without changing state-dict compatibility.
    zero_init_output: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.prediction_type = str(self.prediction_type).strip().lower()
        if self.prediction_type != "epsilon":
            raise ValueError(
                "Only diffusion.prediction_type='epsilon' is implemented; "
                f"got {self.prediction_type!r}. The training target and reverse "
                "samplers must use the same parameterization."
            )
        self.padding_mode = str(self.padding_mode).strip().lower()
        valid_padding_modes = {"zeros", "reflect", "replicate", "circular"}
        if self.padding_mode not in valid_padding_modes:
            raise ValueError(
                "diffusion.padding_mode must be one of "
                f"{sorted(valid_padding_modes)}, got {self.padding_mode!r}."
            )
        try:
            self.clean_x0_reconstruction_weight = float(
                self.clean_x0_reconstruction_weight
            )
            self.clean_x0_inverse_snr_cap = float(self.clean_x0_inverse_snr_cap)
            self.residual_application_scale = float(
                self.residual_application_scale
            )
            self.residual_magnitude_guard_multiple = float(
                self.residual_magnitude_guard_multiple
            )
            self.residual_guard_min_count = int(self.residual_guard_min_count)
            self.residual_mean_channels = int(self.residual_mean_channels)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "diffusion clean-x0 and residual-application options must be numeric: "
                "clean_x0_reconstruction_weight="
                f"{self.clean_x0_reconstruction_weight!r}, "
                "clean_x0_inverse_snr_cap="
                f"{self.clean_x0_inverse_snr_cap!r}, "
                "residual_application_scale="
                f"{self.residual_application_scale!r}."
            ) from exc
        if not np.isfinite(self.clean_x0_reconstruction_weight) or (
            self.clean_x0_reconstruction_weight < 0.0
        ):
            raise ValueError(
                "diffusion.clean_x0_reconstruction_weight must be finite and "
                f"non-negative, got {self.clean_x0_reconstruction_weight!r}."
            )
        if not np.isfinite(self.clean_x0_inverse_snr_cap) or (
            self.clean_x0_inverse_snr_cap <= 0.0
        ):
            raise ValueError(
                "diffusion.clean_x0_inverse_snr_cap must be finite and positive, "
                f"got {self.clean_x0_inverse_snr_cap!r}."
            )
        if not np.isfinite(self.residual_application_scale) or not (
            0.0 <= self.residual_application_scale <= 1.0
        ):
            raise ValueError(
                "diffusion.residual_application_scale must be finite and lie "
                f"in [0, 1], got {self.residual_application_scale!r}."
            )
        if not np.isfinite(self.residual_magnitude_guard_multiple):
            raise ValueError(
                "diffusion.residual_magnitude_guard_multiple must be finite, got "
                f"{self.residual_magnitude_guard_multiple!r}."
            )
        if self.residual_guard_min_count < 0:
            raise ValueError(
                "diffusion.residual_guard_min_count must be non-negative, got "
                f"{self.residual_guard_min_count!r}."
            )
        self.residual_guard_mode = str(self.residual_guard_mode).strip().lower()
        _valid_guard_modes = {"reject", "clip", "warn"}
        if self.residual_guard_mode not in _valid_guard_modes:
            raise ValueError(
                f"diffusion.residual_guard_mode must be one of {sorted(_valid_guard_modes)}, "
                f"got {self.residual_guard_mode!r}."
            )
        if self.residual_mean_channels < 1:
            raise ValueError(
                "diffusion.residual_mean_channels must be positive, got "
                f"{self.residual_mean_channels!r}."
            )
        if self.residual_mean_enabled and not self.residual_diffusion:
            raise ValueError(
                "diffusion.residual_mean_enabled requires residual_diffusion=true."
            )
        if not isinstance(self.zero_init_output, (bool, np.bool_)):
            raise ValueError(
                "diffusion.zero_init_output must be a boolean, got "
                f"{self.zero_init_output!r}."
            )
        self.zero_init_output = bool(self.zero_init_output)

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
            "eta",
            "projected_cond_channels",
            "residual_diffusion",
            "residual_mean_enabled",
            "residual_mean_channels",
            "residual_application_scale",
            "prediction_type",
            "clean_x0_reconstruction_weight",
            "clean_x0_inverse_snr_cap",
            "residual_magnitude_guard_multiple",
            "residual_guard_min_count",
            "residual_guard_mode",
            "base_channels",
            "channel_multipliers",
            "channel_mults",
            "num_res_blocks",
            "time_embed_dim",
            "dropout",
            "fourier_scale",
            "padding_mode",
            "zero_init_output",
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
            eta=float(pick("eta", defaults.eta)),
            projected_cond_channels=int(
                pick("projected_cond_channels", defaults.projected_cond_channels)
            ),
            residual_diffusion=bool(pick("residual_diffusion", defaults.residual_diffusion)),
            residual_mean_enabled=bool(
                pick("residual_mean_enabled", defaults.residual_mean_enabled)
            ),
            residual_mean_channels=int(
                pick("residual_mean_channels", defaults.residual_mean_channels)
            ),
            residual_application_scale=float(
                pick(
                    "residual_application_scale",
                    defaults.residual_application_scale,
                )
            ),
            prediction_type=str(pick("prediction_type", defaults.prediction_type)).lower(),
            clean_x0_reconstruction_weight=float(
                pick(
                    "clean_x0_reconstruction_weight",
                    defaults.clean_x0_reconstruction_weight,
                )
            ),
            clean_x0_inverse_snr_cap=float(
                pick(
                    "clean_x0_inverse_snr_cap",
                    defaults.clean_x0_inverse_snr_cap,
                )
            ),
            residual_magnitude_guard_multiple=float(
                pick(
                    "residual_magnitude_guard_multiple",
                    defaults.residual_magnitude_guard_multiple,
                )
            ),
            residual_guard_min_count=int(
                pick("residual_guard_min_count", defaults.residual_guard_min_count)
            ),
            residual_guard_mode=str(
                pick("residual_guard_mode", defaults.residual_guard_mode)
            ),
            base_channels=int(pick("base_channels", defaults.base_channels)),
            channel_multipliers=multipliers,
            num_res_blocks=int(pick("num_res_blocks", defaults.num_res_blocks)),
            time_embed_dim=int(pick("time_embed_dim", defaults.time_embed_dim)),
            dropout=float(pick("dropout", defaults.dropout)),
            fourier_scale=float(pick("fourier_scale", defaults.fourier_scale)),
            padding_mode=str(pick("padding_mode", defaults.padding_mode)),
            zero_init_output=pick("zero_init_output", defaults.zero_init_output),
            extra=extra,
        )


class DiffusionHead(nn.Module):
    """Conditional score-based diffusion head.

    Args:
        cond_channels: number of channels in the conditioning feature map.
        output_channels: number of target variables (e.g. 2 for ``pr``/``tasmax``).
        head_config: resolved :class:`DiffusionHeadConfig`.

    Residual diffusion (``head_config.residual_diffusion=True``):
        In this mode the head generates a *residual* around a deterministic
        baseline supplied in the conditioning tensor.  The score network input
        is the noised residual (not the noised full target) and the conditioning
        features include both the UNet decoder activations and the deterministic
        baseline (projected to a small number of channels).  During training,
        call ``training_loss(cond, target_std, baseline_std=...)``.  During
        sampling, call ``sample(cond, H, W, baseline_std=...)``.
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
        if self.cfg.prediction_type != "epsilon":
            raise ValueError(
                "The implemented score-matching objective predicts Gaussian noise "
                f"(prediction_type='epsilon'), got {self.cfg.prediction_type!r}."
            )

        # In residual mode both correction networks see the decoder features
        # and the deterministic baseline.  When the supervised residual-mean
        # branch is enabled, the score network additionally conditions on that
        # mean and only models the remaining stochastic innovation.
        base_cond_channels = self.cond_channels
        if head_config.residual_diffusion:
            base_cond_channels += self.output_channels

        if head_config.residual_mean_enabled:
            self.residual_mean_model: ResidualMeanPredictor | None = (
                ResidualMeanPredictor(
                    in_channels=base_cond_channels,
                    out_channels=self.output_channels,
                    hidden_channels=head_config.residual_mean_channels,
                    padding_mode=head_config.padding_mode,
                )
            )
            effective_cond = base_cond_channels + self.output_channels
        else:
            self.residual_mean_model = None
            effective_cond = base_cond_channels

        self.score_model = ConditionalScoreUNet(
            target_channels=self.output_channels,
            cond_channels=effective_cond,
            base_channels=head_config.base_channels,
            channel_multipliers=head_config.channel_multipliers,
            num_res_blocks=head_config.num_res_blocks,
            time_embed_dim=head_config.time_embed_dim,
            dropout=head_config.dropout,
            fourier_scale=head_config.fourier_scale,
            projected_cond_channels=head_config.projected_cond_channels,
            padding_mode=head_config.padding_mode,
            zero_init_output=head_config.zero_init_output,
        )
        self.sde: SDE = build_sde(head_config.sde_params(), num_scales=head_config.num_scales)
        # Online moments of the exact normalized residual tensors supplied to
        # forward noising. They are checkpointed and used only as a hard safety
        # check; they never rescale, clip, or damp generated corrections.
        self.register_buffer("residual_stats_count", torch.zeros((), dtype=torch.float64))
        self.register_buffer(
            "residual_stats_sum", torch.zeros(self.output_channels, dtype=torch.float64)
        )
        self.register_buffer(
            "residual_stats_sumsq", torch.zeros(self.output_channels, dtype=torch.float64)
        )
        self.register_buffer(
            "residual_stats_min", torch.full((self.output_channels,), float("inf"))
        )
        self.register_buffer(
            "residual_stats_max", torch.full((self.output_channels,), float("-inf"))
        )
        self._last_training_loss_terms: dict[str, torch.Tensor] = {}
        self._last_residual_mean: torch.Tensor | None = None

    @torch.no_grad()
    def _update_residual_stats(
        self,
        residual: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        if not self.cfg.residual_diffusion or not self.training:
            return
        values = residual.detach().to(dtype=torch.float64)
        if valid_mask is not None:
            valid = torch.broadcast_to(
                valid_mask.to(device=values.device, dtype=torch.bool), values.shape
            )
            # The checkpoint schema stores a shared scalar count, so statistics
            # use only locations valid for every modeled output channel.
            valid = valid.all(dim=1, keepdim=True).expand_as(values)
            if not bool(valid.any().item()):
                return
            channel_first = values.movedim(1, 0)
            valid_channel_first = valid.movedim(1, 0)
            values = channel_first.masked_select(valid_channel_first).reshape(
                channel_first.shape[0], -1
            )
            per_channel_count = values.shape[1]
            reduce_dims = (1,)
        else:
            reduce_dims = (0, *range(2, values.ndim))
            per_channel_count = values.shape[0]
            for size in values.shape[2:]:
                per_channel_count *= size
        self.residual_stats_count.add_(float(per_channel_count))
        self.residual_stats_sum.add_(values.sum(dim=reduce_dims))
        self.residual_stats_sumsq.add_((values * values).sum(dim=reduce_dims))
        self.residual_stats_min.copy_(
            torch.minimum(
                self.residual_stats_min,
                values.amin(dim=reduce_dims).to(dtype=self.residual_stats_min.dtype),
            )
        )
        self.residual_stats_max.copy_(
            torch.maximum(
                self.residual_stats_max,
                values.amax(dim=reduce_dims).to(dtype=self.residual_stats_max.dtype),
            )
        )

    @torch.no_grad()
    def residual_training_stats(self) -> dict[str, torch.Tensor]:
        """Return checkpointed per-channel moments of normalized true residuals."""
        count = self.residual_stats_count.clamp(min=1.0)
        mean = self.residual_stats_sum / count
        second = self.residual_stats_sumsq / count
        std = torch.sqrt((second - mean.square()).clamp(min=0.0))
        rms = torch.sqrt(second.clamp(min=0.0))
        return {
            "count": self.residual_stats_count.detach().clone(),
            "mean": mean,
            "std": std,
            "rms": rms,
            "min": self.residual_stats_min.detach().clone(),
            "max": self.residual_stats_max.detach().clone(),
        }

    @torch.no_grad()
    def _validate_generated_residual(self, residual: torch.Tensor) -> torch.Tensor:
        """Guard the generated residual against catastrophic magnitudes.

        Returns a (possibly clipped) residual tensor. The returned tensor is
        identical to the input unless ``residual_guard_mode='clip'`` and the
        magnitude check trips, in which case a per-channel-rescaled version is
        returned that preserves the spatial pattern but limits RMS to the
        configured multiple of the training RMS.

        Raises RuntimeError for NaN/inf unconditionally and for over-large
        residuals when ``residual_guard_mode='reject'``.
        """
        if not bool(torch.isfinite(residual).all().item()):
            raise RuntimeError(
                "Generated diffusion residual contains NaN or infinity; "
                "rejecting the sample before residual application."
            )
        multiple = float(self.cfg.residual_magnitude_guard_multiple)
        if multiple <= 0.0:
            return residual
        if float(self.residual_stats_count.item()) < float(self.cfg.residual_guard_min_count):
            return residual
        observed_rms = self.residual_training_stats()["rms"].to(
            device=residual.device, dtype=residual.dtype
        )
        reduce_dims = (0, *range(2, residual.ndim))
        generated_rms = torch.sqrt(residual.square().mean(dim=reduce_dims))
        limit = observed_rms.clamp(min=1e-6) * multiple
        bad = generated_rms > limit

        if not bool(bad.any().item()):
            return residual

        channels = torch.nonzero(bad, as_tuple=False).flatten().tolist()
        mode = self.cfg.residual_guard_mode

        if mode == "reject":
            raise RuntimeError(
                "Generated diffusion residual failed the training-scale guard: "
                f"channels={channels}, generated_rms={generated_rms.tolist()}, "
                f"training_rms={observed_rms.tolist()}, multiple={multiple}. "
                "Rejecting this checkpoint/sample instead of damping or clipping it. "
                "Set diffusion.residual_guard_mode=clip or warn to allow clipped output."
            )

        msg = (
            "WARNING: generated diffusion residual exceeds the training-scale guard "
            f"(channels={channels}, generated_rms={generated_rms.tolist()}, "
            f"training_rms={observed_rms.tolist()}, multiple={multiple}). "
        )

        if mode == "warn":
            import warnings
            warnings.warn(msg + "Passing through unchanged.", stacklevel=2)
            return residual

        # mode == "clip": rescale each over-limit channel so its RMS equals the limit.
        import warnings
        scale = torch.where(bad, limit / generated_rms.clamp(min=1e-12), torch.ones_like(generated_rms))
        # Broadcast scale to [1, C, 1, 1] so it applies per channel across the batch.
        scale = scale[None, :, None, None]
        clipped = residual * scale
        actual_rms = torch.sqrt(clipped.square().mean(dim=reduce_dims))
        warnings.warn(
            msg + f"Clipped to rms={actual_rms.tolist()}.", stacklevel=2
        )
        return clipped

    # -- conditioning helper ------------------------------------------------
    def _build_cond(
        self,
        cond: torch.Tensor,
        baseline_std: torch.Tensor | None,
    ) -> torch.Tensor:
        """Concatenate baseline to conditioning in residual mode."""
        if not self.cfg.residual_diffusion:
            return cond
        if baseline_std is None:
            raise ValueError(
                "residual_diffusion=True requires baseline_std to be provided "
                "to training_loss() and sample()."
            )
        if baseline_std.shape[-2:] != cond.shape[-2:]:
            baseline_std = F.interpolate(
                baseline_std, size=cond.shape[-2:], mode="bilinear", align_corners=False
            )
        return torch.cat([cond, baseline_std], dim=1)

    def _predict_residual_mean(
        self,
        base_cond: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        """Predict the conditional residual mean at the target resolution."""
        if self.residual_mean_model is None:
            raise RuntimeError(
                "Residual-mean prediction requested while "
                "diffusion.residual_mean_enabled is false."
            )
        mean_cond = base_cond
        if mean_cond.shape[-2:] != output_size:
            # Predict at target resolution so the YAML-configured spatial
            # gradient loss can supervise target-grid structure directly.
            mean_cond = F.interpolate(
                mean_cond,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        return self.residual_mean_model(mean_cond)

    def _build_score_cond(
        self,
        base_cond: torch.Tensor,
        residual_mean: torch.Tensor | None,
    ) -> torch.Tensor:
        """Append the predicted residual mean to score conditioning."""
        if self.residual_mean_model is None:
            return base_cond
        if residual_mean is None:
            raise ValueError(
                "residual_mean must be supplied when residual_mean_enabled=true."
            )
        if residual_mean.shape[-2:] != base_cond.shape[-2:]:
            residual_mean = F.interpolate(
                residual_mean,
                size=base_cond.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return torch.cat([base_cond, residual_mean], dim=1)

    def get_last_residual_mean(self) -> torch.Tensor | None:
        """Return the differentiable residual mean from the latest train call."""
        return self._last_residual_mean

    # -- score function -----------------------------------------------------
    def score_fn(self, x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return the score ``\\nabla_x log p_t(x | cond)``.

        The score network is explicitly epsilon-parameterized. Its raw output
        is ``epsilon_theta`` and is converted via
        ``score_theta = -epsilon_theta / marginal_std``. Consequently the
        unweighted score-matching objective reduces exactly to
        ``||epsilon_theta - epsilon||^2`` (apart from the small-std numerical
        clamp).
        """
        labels = t * self.cfg.noise_conditioning_scale
        model_out = self.score_model(x, cond, labels)
        std = self.sde.marginal_prob(torch.zeros_like(x[:, :1, :1, :1]), t)[1]
        # Clamp std away from zero to prevent numerical explosion at small t.
        std = std.clamp(min=1e-5)
        return -model_out / std[:, None, None, None]

    # -- training -----------------------------------------------------------
    def training_loss(
        self,
        cond: torch.Tensor,
        target: torch.Tensor,
        baseline_std: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score-matching loss.

        Args:
            cond: conditioning feature map ``[B, C_cond, h, w]``.
            target: full standardized target, shape ``[B, C_out, H, W]``.
                Residual mode subtracts ``baseline_std`` internally.
            baseline_std: deterministic baseline in standardized space, required
                when ``residual_diffusion=True``.
        """
        if self.cfg.residual_diffusion:
            if baseline_std is None:
                raise ValueError(
                    "residual_diffusion=True requires baseline_std to be "
                    "provided to training_loss()."
                )
            if target.ndim != 4 or baseline_std.ndim != 4:
                raise ValueError(
                    "target and baseline_std must both have shape [B,C,H,W], got "
                    f"{tuple(target.shape)} and {tuple(baseline_std.shape)}."
                )
            if target.shape[:2] != baseline_std.shape[:2]:
                raise ValueError(
                    "target and baseline_std batch/channel dimensions differ: "
                    f"{tuple(target.shape[:2])} vs {tuple(baseline_std.shape[:2])}."
                )
            if target.shape[1] != self.output_channels:
                raise ValueError(
                    "target channel count does not match diffusion outputs: "
                    f"{target.shape[1]} vs {self.output_channels}."
                )
            if target.device != baseline_std.device:
                raise ValueError(
                    "target and baseline_std must be on the same device: "
                    f"{target.device} vs {baseline_std.device}."
                )
            # The score objective must update only the diffusion head. The
            # deterministic branch is trained separately by its supervised
            # loss, so stop gradients through both shared conditioning and the
            # baseline used for residual construction/conditioning here.
            detached_baseline = baseline_std.detach()
            if detached_baseline.shape[-2:] != target.shape[-2:]:
                detached_baseline = F.interpolate(
                    detached_baseline,
                    size=target.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            valid = torch.isfinite(target) & torch.isfinite(detached_baseline)
            if valid_mask is not None:
                try:
                    configured_valid = torch.broadcast_to(
                        valid_mask.to(device=target.device, dtype=torch.bool),
                        target.shape,
                    )
                except RuntimeError as exc:
                    raise ValueError(
                        f"valid_mask shape {tuple(valid_mask.shape)} cannot "
                        f"broadcast to target shape {tuple(target.shape)}."
                    ) from exc
                valid = valid & configured_valid
            full_residual = torch.where(
                valid,
                target - detached_baseline,
                torch.zeros_like(target),
            )
            base_cond = self._build_cond(cond.detach(), detached_baseline)
            if self.residual_mean_model is not None:
                residual_mean = self._predict_residual_mean(
                    base_cond, tuple(target.shape[-2:])
                )
                self._last_residual_mean = residual_mean
                # The supervised configured task loss trains the mean branch.
                # Score matching learns only the remaining stochastic
                # innovation and must not backpropagate into that branch.
                target_input = full_residual - residual_mean.detach()
                full_cond = self._build_score_cond(
                    base_cond, residual_mean.detach()
                )
            else:
                self._last_residual_mean = None
                target_input = full_residual
                full_cond = base_cond
        else:
            self._last_residual_mean = None
            valid = torch.isfinite(target)
            if valid_mask is not None:
                try:
                    configured_valid = torch.broadcast_to(
                        valid_mask.to(device=target.device, dtype=torch.bool),
                        target.shape,
                    )
                except RuntimeError as exc:
                    raise ValueError(
                        f"valid_mask shape {tuple(valid_mask.shape)} cannot "
                        f"broadcast to target shape {tuple(target.shape)}."
                    ) from exc
                valid = valid & configured_valid
            target_input = torch.where(valid, target, torch.zeros_like(target))
            full_cond = self._build_cond(cond, baseline_std)

        if self.cfg.residual_diffusion:
            self._update_residual_stats(full_residual, valid_mask=valid)

        loss, terms = score_matching_loss(
            self.sde,
            self.score_fn,
            target_input,
            full_cond,
            reduce_mean=self.cfg.reduce_mean,
            likelihood_weighting=self.cfg.likelihood_weighting,
            eps=self.cfg.eps,
            clean_x0_reconstruction_weight=(
                self.cfg.clean_x0_reconstruction_weight
            ),
            clean_x0_inverse_snr_cap=self.cfg.clean_x0_inverse_snr_cap,
            valid_mask=valid,
            return_terms=True,
        )
        self._last_training_loss_terms = terms
        return loss

    def get_last_training_loss_terms(self) -> dict[str, float]:
        """Return scalar components from the most recent training loss call."""
        return {
            name: float(value.detach().cpu().item())
            for name, value in self._last_training_loss_terms.items()
        }

    def get_last_training_loss_term_tensors(self) -> dict[str, torch.Tensor]:
        """Return detached device tensors for trainer-side term reporting."""
        return dict(self._last_training_loss_terms)

    def forward(
        self,
        cond: torch.Tensor,
        target: torch.Tensor,
        baseline_std: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Alias for :meth:`training_loss` (keeps ``nn.Module`` semantics)."""
        return self.training_loss(
            cond,
            target,
            baseline_std=baseline_std,
            valid_mask=valid_mask,
        )

    # -- sampling -----------------------------------------------------------
    @torch.no_grad()
    def sample_components(
        self,
        cond: torch.Tensor,
        height: int,
        width: int,
        generator: torch.Generator | None = None,
        baseline_std: torch.Tensor | None = None,
        application_scale: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(full_prediction, residual)`` in standardized target space.

        Returns a tensor of shape ``[B, output_channels, height, width]``.

        In residual diffusion mode the first tensor is the FULL prediction
        (baseline + ``residual_application_scale`` * generated residual). At
        alpha zero the sampler is bypassed and the second tensor is exactly
        zero; at nonzero alpha it preserves the unscaled generated residual.
        """
        # Never expose the mean tensor from a prior batch if fail-closed alpha=0
        # bypasses the sampler below.
        self._last_residual_mean = None
        resolved_application_scale = (
            self.cfg.residual_application_scale
            if application_scale is None
            else float(application_scale)
        )
        if not np.isfinite(resolved_application_scale) or not (
            0.0 <= resolved_application_scale <= 1.0
        ):
            raise ValueError(
                "application_scale must be finite and lie in [0, 1], got "
                f"{resolved_application_scale!r}."
            )
        baseline_for_output = baseline_std
        if self.cfg.residual_diffusion:
            if baseline_std is None:
                raise ValueError(
                    "residual_diffusion=True requires baseline_std to be provided "
                    "to sample_components()."
                )
            if baseline_std.ndim != 4:
                raise ValueError(
                    "baseline_std must have shape [B,C,H,W], got "
                    f"{tuple(baseline_std.shape)}."
                )
            if baseline_std.shape[0] != cond.shape[0]:
                raise ValueError(
                    "baseline_std and conditioning batch sizes differ: "
                    f"{baseline_std.shape[0]} vs {cond.shape[0]}."
                )
            if baseline_std.shape[1] != self.output_channels:
                raise ValueError(
                    "baseline_std channel count does not match diffusion outputs: "
                    f"{baseline_std.shape[1]} vs {self.output_channels}."
                )
            if baseline_std.device != cond.device:
                raise ValueError(
                    "baseline_std and conditioning must be on the same device: "
                    f"{baseline_std.device} vs {cond.device}."
                )
            if baseline_std.shape[-2:] != (height, width):
                baseline_for_output = F.interpolate(
                    baseline_std,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
            # Fail-closed deployment means no reverse-process work at all. This
            # guarantees an exact deterministic baseline even when an untrusted
            # score model would diverge or trip the magnitude guard.
            if resolved_application_scale == 0.0:
                return baseline_for_output, torch.zeros_like(baseline_for_output)

        base_cond = self._build_cond(cond, baseline_std)
        residual_mean: torch.Tensor | None = None
        if self.residual_mean_model is not None:
            residual_mean = self._predict_residual_mean(
                base_cond, (int(height), int(width))
            )
            self._last_residual_mean = residual_mean.detach()
        else:
            self._last_residual_mean = None
        full_cond = self._build_score_cond(base_cond, residual_mean)
        sampler = build_sampler(
            self.cfg,
            self.sde,
            (self.output_channels, int(height), int(width)),
            device=cond.device,
        )
        residual_or_full = sampler(self.score_fn, full_cond, generator=generator)

        if self.cfg.residual_diffusion and baseline_for_output is not None:
            generated_residual = residual_or_full
            if residual_mean is not None:
                generated_residual = residual_mean + residual_or_full
            generated_residual = self._validate_generated_residual(generated_residual)
            full_prediction = (
                baseline_for_output
                + resolved_application_scale * generated_residual
            )
            return full_prediction, generated_residual
        return residual_or_full, residual_or_full

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        height: int,
        width: int,
        generator: torch.Generator | None = None,
        baseline_std: torch.Tensor | None = None,
        application_scale: float | None = None,
    ) -> torch.Tensor:
        """Reverse-diffusion sample in standardized target space."""
        full, _ = self.sample_components(
            cond,
            height,
            width,
            generator=generator,
            baseline_std=baseline_std,
            application_scale=application_scale,
        )
        return full


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
