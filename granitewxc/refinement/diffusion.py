"""Diffusion-based Phase-2 residual refiners.

Formulation
-----------
Let ``r`` be the residual between the ground truth and the deterministic
Phase-1 prediction, both expressed in the Phase-1 *normalized target space*::

    r = encode(y) - deterministic_normalized

Training draws a noise-process index ``k ~ U{0, ..., K-1}`` and Gaussian noise
``eps ~ N(0, I)``, forms the forward-noised residual

    r_k = sqrt(abar_k) * r + sqrt(1 - abar_k) * eps

and regresses the configured parameterisation (``epsilon``, ``velocity`` or
``sample``) with a masked loss. ``k`` indexes the *noise* process only: it is
not a timestamp, not a lead time and not a sequence position.

Inference runs a deterministic DDIM reverse pass (``eta = 0``, the default) or a
stochastic DDPM-like pass (``eta > 0``) over ``inference_steps`` timesteps taken
from the same schedule that was used in training.

Reused from ``CORDEX_ML_diffusion_head``: the conditional score-network layout
(FiLM-conditioned residual blocks, Fourier/sinusoidal process-time embedding)
and the conditioning-by-concatenation design. Re-implemented here as a
*residual* model with configurable prediction type and schedule so that the same
code serves CORDEX_ML, MERRA_PRISM and NARR_PRISM.
"""

from __future__ import annotations

import torch

from granitewxc.refinement.backbones import ConditionalResidualUNet, SpatialResidualTransformer
from granitewxc.refinement.base import ResidualRefiner, masked_loss, register_refiner
from granitewxc.refinement.config import RefinementConfig
from granitewxc.refinement.schedules import DiffusionSchedule

__all__ = ["DiffusionUNetRefiner", "DiffusionTransformerRefiner"]


class _BaseDiffusionRefiner(ResidualRefiner):
    def __init__(self, config: RefinementConfig, *, residual_channels: int, cond_channels: int) -> None:
        super().__init__(config, residual_channels=residual_channels, cond_channels=cond_channels)
        d = config.diffusion
        self.prediction_type = d.prediction_type
        self.num_train_timesteps = d.training_timesteps
        self.num_inference_steps = d.inference_steps
        self.eta = d.eta
        self.clip_sample = d.clip_sample
        self.clip_sample_range = d.clip_sample_range
        self.schedule = DiffusionSchedule(
            num_train_timesteps=d.training_timesteps,
            schedule=d.schedule,
            beta_start=d.beta_start,
            beta_end=d.beta_end,
            cosine_s=d.cosine_s,
        )
        self.net = self._build_net(config)

    def _build_net(self, config: RefinementConfig) -> torch.nn.Module:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- process-time scaling -------------------------------------------
    @staticmethod
    def _embed_time(timesteps: torch.Tensor) -> torch.Tensor:
        """Feed the raw integer noise index to the embedding (float cast only)."""
        return timesteps.float()

    # -- training --------------------------------------------------------
    def training_loss(
        self,
        residual_target: torch.Tensor,
        conditioning: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        self._require_residual_normalization()
        residual_target = self.apply_valid_mask(residual_target, valid_mask)
        batch = residual_target.shape[0]
        device = residual_target.device
        if generator is not None and generator.device != torch.device(device):
            timesteps = torch.randint(
                0, self.num_train_timesteps, (batch,), generator=generator, device=generator.device
            ).to(device)
        else:
            timesteps = torch.randint(
                0, self.num_train_timesteps, (batch,), generator=generator, device=device
            )
        noise = self._randn(
            tuple(residual_target.shape),
            device,
            residual_target.dtype,
            generator,
        )
        noise = self.apply_valid_mask(noise, valid_mask)

        noisy = self.schedule.add_noise(residual_target, noise, timesteps)
        target = self.schedule.training_target(self.prediction_type, residual_target, noise, timesteps)
        prediction = self.net(noisy, conditioning, self._embed_time(timesteps))
        variable_weights = self.config.auxiliary_loss.variable_weights or None
        objective = masked_loss(
            prediction, target, valid_mask, self.loss_kind, variable_weights
        )
        clean_estimate = self.schedule.to_clean(
            self.prediction_type, prediction, noisy, timesteps
        )
        auxiliary = self.auxiliary_losses(clean_estimate, residual_target, valid_mask)
        loss = objective + auxiliary["auxiliary_loss"]
        return {
            "loss": loss,
            "stochastic_objective": objective,
            **auxiliary,
            "timesteps": timesteps,
            "clean_residual": residual_target,
            "noised_residual": noisy,
            "estimated_clean_residual": clean_estimate,
            "prediction": prediction,
        }

    # -- sampling --------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        conditioning: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        num_steps: int | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._require_residual_normalization()
        steps = int(num_steps if num_steps is not None else self.num_inference_steps)
        shape = self.residual_shape(conditioning)
        device, dtype = conditioning.device, conditioning.dtype

        x = self.apply_valid_mask(
            self._randn(shape, device, dtype, generator), valid_mask
        )
        timesteps = self.schedule.inference_timesteps(steps, device)
        alphas_cumprod = self.schedule.alphas_cumprod.to(device=device, dtype=torch.float32)

        for idx in range(steps):
            t = timesteps[idx]
            t_batch = t.expand(shape[0])
            model_out = self.net(x, conditioning, self._embed_time(t_batch))

            x0 = self.schedule.to_clean(self.prediction_type, model_out, x, t_batch)
            if self.clip_sample:
                x0 = x0.clamp(-self.clip_sample_range, self.clip_sample_range)
            eps = self.schedule.to_epsilon(self.prediction_type, model_out, x, t_batch)

            a_t = alphas_cumprod[t]
            prev_index = timesteps[idx + 1] if idx + 1 < steps else None
            a_prev = alphas_cumprod[prev_index] if prev_index is not None else torch.ones((), device=device)

            # Generalized DDIM update: eta=0 is deterministic.  With a full
            # consecutive timestep grid eta=1 agrees with the DDPM posterior
            # variance; on a strided grid it is a stochastic DDIM update, not
            # an exact ancestral DDPM transition.
            sigma = self.eta * torch.sqrt(
                ((1 - a_prev) / (1 - a_t).clamp(min=1e-12)) * (1 - a_t / a_prev.clamp(min=1e-12))
            )
            sigma = torch.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
            dir_coeff = torch.sqrt((1 - a_prev - sigma.pow(2)).clamp(min=0.0))
            x = a_prev.sqrt().to(dtype) * x0 + dir_coeff.to(dtype) * eps
            if float(sigma) > 0 and prev_index is not None:
                x = x + sigma.to(dtype) * self._randn(shape, device, dtype, generator)
            x = self.apply_valid_mask(x, valid_mask)
        return x

    @torch.no_grad()
    def deterministic_residual(
        self,
        conditioning: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One-step posterior mean from pure noise at the largest timestep.

        This is a cheap diagnostic, not a substitute for the configured
        multi-step sampler.
        """
        self._require_residual_normalization()
        shape = self.residual_shape(conditioning)
        device, dtype = conditioning.device, conditioning.dtype
        x = torch.zeros(shape, device=device, dtype=dtype)
        t = torch.full((shape[0],), self.num_train_timesteps - 1, device=device, dtype=torch.long)
        model_out = self.net(x, conditioning, self._embed_time(t))
        clean = self.schedule.to_clean(
            self.prediction_type, model_out, x, t
        )
        return self.apply_valid_mask(clean, valid_mask)


@register_refiner("diffusion_unet")
class DiffusionUNetRefiner(_BaseDiffusionRefiner):
    """Convolutional UNet DDPM/DDIM residual refiner."""

    def _build_net(self, config: RefinementConfig) -> torch.nn.Module:
        u = config.unet
        return ConditionalResidualUNet(
            in_channels=self.residual_channels,
            cond_channels=self.cond_channels,
            out_channels=self.residual_channels,
            hidden_channels=u.hidden_channels,
            num_levels=u.num_levels,
            time_embedding_dim=u.time_embedding_dim,
            dropout=u.dropout,
            bottleneck_attention=u.bottleneck_attention,
            attention_heads=u.attention_heads,
            zero_init_output=u.zero_init_output,
        )


@register_refiner("diffusion_transformer")
class DiffusionTransformerRefiner(_BaseDiffusionRefiner):
    """Spatial-token Transformer DDPM/DDIM residual refiner."""

    def _build_net(self, config: RefinementConfig) -> torch.nn.Module:
        t = config.transformer
        return SpatialResidualTransformer(
            in_channels=self.residual_channels,
            cond_channels=self.cond_channels,
            out_channels=self.residual_channels,
            patch_size=t.patch_size,
            embedding_dim=t.embedding_dim,
            num_heads=t.num_heads,
            num_blocks=t.num_blocks,
            mlp_ratio=t.mlp_ratio,
            dropout=t.dropout,
            positional_encoding=t.positional_encoding,
            max_tokens_lat=t.max_tokens_lat,
            max_tokens_lon=t.max_tokens_lon,
            gradient_checkpointing=t.gradient_checkpointing,
            optimized_attention=t.optimized_attention,
            zero_init_output=t.zero_init_output,
        )

    def set_attention_implementation(self, implementation: str) -> None:
        self.net.set_attention_implementation(implementation)
