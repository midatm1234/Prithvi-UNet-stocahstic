"""Diffusion-based Phase-2 residual refiners.

Formulation
-----------
Let ``r_phys`` be the physical residual between ground truth and the frozen
Phase-1 prediction and ``N_r`` the persisted training-residual transform::

    r_phys = y_phys - phase1_phys
    r = N_r(r_phys)

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

from collections.abc import Callable, Sequence

import torch

from granitewxc.refinement.backbones import ConditionalResidualUNet, SpatialResidualTransformer
from granitewxc.refinement.base import (
    ResidualRefiner,
    masked_loss,
    register_refiner,
    residual_reconstruction_terms,
)
from granitewxc.refinement.config import RefinementConfig
from granitewxc.refinement.schedules import DiffusionSchedule

__all__ = [
    "DiffusionUNetRefiner",
    "DiffusionTransformerRefiner",
    "ddim_reverse_step",
]


def _broadcast_coefficient(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Broadcast a scalar/per-sample schedule coefficient over a BCHW tensor."""
    if value.ndim == 0:
        return value.to(device=reference.device, dtype=reference.dtype)
    return value.to(device=reference.device, dtype=reference.dtype).reshape(
        -1, *([1] * (reference.ndim - 1))
    )


def ddim_reverse_step(
    schedule: DiffusionSchedule,
    prediction_type: str,
    model_output: torch.Tensor,
    sample: torch.Tensor,
    timesteps: torch.Tensor,
    previous_timestep: int | torch.Tensor | None,
    *,
    eta: float = 0.0,
    noise: torch.Tensor | None = None,
    clip_sample: bool = False,
    clip_sample_range: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perform one explicit DDIM reverse step.

    Returns ``(previous_sample, predicted_clean, predicted_epsilon)``.  Keeping
    this algebra outside the model loop makes the forward/reverse oracle test
    exact and ensures convolutional and Transformer refiners use one formula.
    No random noise is drawn here; callers must provide it for a non-terminal
    stochastic step (``eta > 0``).
    """
    if model_output.shape != sample.shape:
        raise ValueError(
            f"DDIM model output shape {tuple(model_output.shape)} does not match "
            f"sample shape {tuple(sample.shape)}."
        )
    if not 0.0 <= float(eta) <= 1.0:
        raise ValueError(f"DDIM eta must be in [0, 1], got {eta}.")
    if not bool(torch.isfinite(sample).all()) or not bool(torch.isfinite(model_output).all()):
        raise FloatingPointError("DDIM sample and model output must be finite.")

    # alpha_bar[0] rounds to one in fp16 and 1e-12 rounds to zero. Reverse
    # variances and the evolving state therefore use at least float32.
    if sample.dtype in {torch.float16, torch.bfloat16}:
        sample = sample.float()
    model_output = model_output.to(sample.dtype)
    alphas = schedule.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
    t = timesteps.to(device=sample.device, dtype=torch.long).reshape(-1)
    if t.numel() not in {1, sample.shape[0]}:
        raise ValueError(
            f"timesteps must be scalar or have batch size {sample.shape[0]}, got {t.numel()}."
        )
    if t.numel() == 1 and sample.shape[0] > 1:
        t = t.expand(sample.shape[0])
    if bool(((t < 0) | (t >= schedule.num_train_timesteps)).any()):
        raise ValueError(
            f"DDIM timesteps must be in [0, {schedule.num_train_timesteps}), "
            f"got {t.tolist()}."
        )

    clean = schedule.to_clean(prediction_type, model_output, sample, t)
    epsilon = schedule.to_epsilon(prediction_type, model_output, sample, t)
    a_t_raw = alphas[t]

    if clip_sample:
        clean = clean.clamp(-float(clip_sample_range), float(clip_sample_range))
        # Clipping changes x0; recompute epsilon so the two terms in the DDIM
        # update remain a self-consistent decomposition of the current sample.
        a_t = _broadcast_coefficient(a_t_raw, sample)
        epsilon = (sample - a_t.sqrt() * clean) / (1.0 - a_t).sqrt().clamp(min=1e-12)

    terminal = previous_timestep is None
    if terminal:
        a_prev_raw = torch.ones_like(a_t_raw)
    else:
        prev = torch.as_tensor(previous_timestep, device=sample.device, dtype=torch.long).reshape(-1)
        if prev.numel() == 1 and sample.shape[0] > 1:
            prev = prev.expand(sample.shape[0])
        if prev.numel() != sample.shape[0]:
            raise ValueError(
                "previous_timestep must be scalar or have one value per sample; "
                f"got {prev.numel()} for batch {sample.shape[0]}."
            )
        if bool(((prev < 0) | (prev >= schedule.num_train_timesteps)).any()):
            raise ValueError(
                f"DDIM previous timesteps must be in [0, {schedule.num_train_timesteps}), "
                f"got {prev.tolist()}."
            )
        if bool((prev >= t).any()):
            raise ValueError(
                "DDIM reverse sampling requires previous_timestep < timestep; "
                f"got current={t.tolist()} previous={prev.tolist()}."
            )
        a_prev_raw = alphas[prev]

    a_t = _broadcast_coefficient(a_t_raw, sample)
    a_prev = _broadcast_coefficient(a_prev_raw, sample)
    sigma = float(eta) * torch.sqrt(
        ((1.0 - a_prev) / (1.0 - a_t).clamp(min=1e-12))
        * (1.0 - a_t / a_prev.clamp(min=1e-12))
    )
    if not bool(torch.isfinite(sigma).all()):
        raise FloatingPointError("DDIM variance became non-finite.")
    direction = torch.sqrt((1.0 - a_prev - sigma.square()).clamp(min=0.0))
    previous = a_prev.sqrt() * clean + direction * epsilon
    if not terminal and float(eta) > 0.0:
        if noise is None:
            raise ValueError("A DDIM noise tensor is required when eta > 0.")
        if noise.shape != sample.shape:
            raise ValueError(
                f"DDIM noise shape {tuple(noise.shape)} does not match sample {tuple(sample.shape)}."
            )
        previous = previous + sigma * noise.to(previous.dtype)
    if not bool(torch.isfinite(previous).all()):
        raise FloatingPointError("DDIM reverse step produced a non-finite sample.")
    return previous, clean, epsilon


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
        self.reconstruction_loss_weight = float(
            getattr(config, "reconstruction_loss_weight", 0.25)
        )
        self.multiscale_loss_weight = float(
            getattr(config, "multiscale_loss_weight", 0.10)
        )
        self.gradient_loss_weight = float(
            getattr(config, "gradient_loss_weight", 0.05)
        )
        self.mean_bias_loss_weight = float(
            getattr(config, "mean_bias_loss_weight", 0.01)
        )
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

    def apply_correction_gate(self, residual: torch.Tensor) -> torch.Tensor:
        """Apply the optional correction gate in the tensor's current space.

        U-Net refiners have no gate.  Transformer subclasses expose a
        per-channel gate; the two-phase wrapper also uses this method *after*
        inverse residual normalization in physical space.  Samplers themselves
        deliberately remain raw so diagnostics cannot hide a broken process.
        """
        return residual

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
        zero_residual: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
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
        noise = self._randn(tuple(residual_target.shape), device, residual_target.dtype, generator)

        noisy = self.schedule.add_noise(residual_target, noise, timesteps)
        target = self.schedule.training_target(self.prediction_type, residual_target, noise, timesteps)
        prediction = self.net(noisy.to(conditioning.dtype), conditioning, self._embed_time(timesteps))
        process_loss = masked_loss(prediction, target, valid_mask, self.loss_kind)
        clean_prediction = self.schedule.to_clean(
            self.prediction_type, prediction, noisy, timesteps
        )
        zero_anchor = (
            torch.zeros_like(clean_prediction)
            if zero_residual is None
            else zero_residual.to(clean_prediction)
        )
        terms = residual_reconstruction_terms(
            clean_prediction,
            residual_target,
            valid_mask,
            zero_anchor,
            self.apply_correction_gate if hasattr(self, "correction_gate") else None,
        )
        loss = (
            process_loss
            + self.reconstruction_loss_weight
            * (terms["reconstruction_loss"] + terms["gate_calibration_loss"])
            + self.multiscale_loss_weight
            * terms["multiscale_loss"]
            + self.gradient_loss_weight * terms["gradient_loss"]
            + self.mean_bias_loss_weight * terms["mean_bias_loss"]
        )
        return {
            "loss": loss,
            "process_loss": process_loss,
            **terms,
            "timesteps": timesteps,
            "source_noise": noise,
            "intermediate_state": noisy,
            "process_target": target,
            "prediction": prediction,
            "clean_residual_prediction": clean_prediction,
        }

    # -- sampling --------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        conditioning: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        num_steps: int | None = None,
        initial_state: torch.Tensor | None = None,
        step_noises: Sequence[torch.Tensor] | torch.Tensor | None = None,
        trajectory_callback: Callable[[str, int, torch.Tensor, torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        """Sample with optional shared random fields and detached snapshots.

        step_noises[i] is the domain-wide innovation at reverse step i. Supply
        exactly steps - 1 fields for eta > 0. A callback receives (stage, index,
        process_time, state); copies keep diagnostics from mutating sampling.
        """
        steps = int(num_steps if num_steps is not None else self.num_inference_steps)
        shape = self.residual_shape(conditioning)
        device, dtype = conditioning.device, conditioning.dtype

        terminal_alpha = float(self.schedule.alphas_cumprod[-1])
        if terminal_alpha > 1.0e-3:
            raise RuntimeError(
                "Diffusion sampling starts from N(0, I), but the configured "
                f"schedule has terminal alpha_bar={terminal_alpha:.6g}. Use a "
                "longer schedule or a near-zero-terminal-SNR schedule."
            )
        if initial_state is None:
            x = self._randn(shape, device, dtype, generator)
        else:
            if tuple(initial_state.shape) != tuple(shape):
                raise ValueError(f"Diffusion initial_state must have shape {shape}")
            x = initial_state.detach().clone().to(device=device, dtype=dtype)
        if x.dtype in {torch.float16, torch.bfloat16}:
            x = x.float()
        if step_noises is not None:
            if self.eta == 0.0:
                raise ValueError("step_noises are unused when DDIM eta is zero")
            if len(step_noises) != steps - 1:
                raise ValueError(f"Expected {steps - 1} DDIM step_noises, got {len(step_noises)}")
            if any(tuple(value.shape) != tuple(shape) for value in step_noises):
                raise ValueError(f"Every DDIM step noise must have shape {shape}")
        timesteps = self.schedule.inference_timesteps(steps, device)
        if trajectory_callback is not None:
            trajectory_callback("initial", 0, timesteps[0].detach().clone(), x.detach().clone())
        for idx in range(steps):
            t = timesteps[idx]
            t_batch = t.expand(shape[0])
            model_out = self.net(x.to(dtype), conditioning, self._embed_time(t_batch))
            prev_index = timesteps[idx + 1] if idx + 1 < steps else None
            step_noise = None
            if self.eta > 0.0 and prev_index is not None:
                step_noise = (
                    self._randn(shape, device, dtype, generator)
                    if step_noises is None else step_noises[idx].to(device=device)
                )
            x, _, _ = ddim_reverse_step(
                self.schedule,
                self.prediction_type,
                model_out,
                x,
                t_batch,
                prev_index,
                eta=self.eta,
                noise=step_noise,
                clip_sample=self.clip_sample,
                clip_sample_range=self.clip_sample_range,
            )
            if trajectory_callback is not None:
                stage = "final_normalized_residual" if prev_index is None else "intermediate"
                state_time = torch.tensor(-1, device=device) if prev_index is None else prev_index
                trajectory_callback(stage, idx + 1, state_time.detach().clone(), x.detach().clone())
        return x

    @torch.no_grad()
    def deterministic_residual(self, conditioning: torch.Tensor) -> torch.Tensor:
        """One-step clean estimate from a zero state at the largest timestep.

        A nonlinear network evaluated at zero is not generally the stochastic
        ensemble mean or a posterior mean. This is a cheap diagnostic, not a
        substitute for the configured multi-step sampler.
        """
        shape = self.residual_shape(conditioning)
        device, dtype = conditioning.device, conditioning.dtype
        x = torch.zeros(shape, device=device, dtype=dtype)
        t = torch.full((shape[0],), self.num_train_timesteps - 1, device=device, dtype=torch.long)
        model_out = self.net(x, conditioning, self._embed_time(t))
        return self.schedule.to_clean(self.prediction_type, model_out, x, t)


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
            spatial_alignment=u.spatial_alignment,
        )


@register_refiner("diffusion_transformer")
class DiffusionTransformerRefiner(_BaseDiffusionRefiner):
    """Spatial-token Transformer DDPM/DDIM residual refiner."""

    def __init__(self, config: RefinementConfig, *, residual_channels: int, cond_channels: int) -> None:
        super().__init__(
            config,
            residual_channels=residual_channels,
            cond_channels=cond_channels,
        )
        # This gate is applied by the two-phase wrapper after inverse residual
        # normalization.  Keeping raw sampler outputs ungated is intentional:
        # diagnostics must still expose scale/sign failures in the sampler.
        self.correction_gate = torch.nn.Parameter(
            torch.zeros(1, self.residual_channels, 1, 1)
        )

    def apply_correction_gate(self, residual: torch.Tensor) -> torch.Tensor:
        if residual.ndim < 3 or residual.shape[-3] != self.residual_channels:
            raise ValueError(
                "Correction gate expected residual channels at dimension -3; "
                f"got shape {tuple(residual.shape)} for {self.residual_channels} channels."
            )
        shape = [1] * residual.ndim
        shape[-3] = self.residual_channels
        return residual * self.correction_gate.reshape(shape).to(residual.dtype)

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
            spatial_alignment=t.spatial_alignment,
        )

    def set_attention_implementation(self, implementation: str) -> None:
        self.net.set_attention_implementation(implementation)
