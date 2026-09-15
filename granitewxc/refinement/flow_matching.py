"""Conditional flow-matching Phase-2 residual refiners.

Formulation
-----------
Following the rectified-flow / conditional-flow-matching construction used in
the Aurora ``aurora_finetune_flow_matching`` branch, adapted to spatial
downscaling residuals.

* **Source distribution** ``p_0``: isotropic Gaussian ``N(0, I)`` on the residual
  grid (``flow_matching.source_distribution: gaussian``).
* **Target distribution** ``p_1``: the distribution of the dedicated
  residual-normalized physical correction
  ``r = N_r(y_phys - phase1_phys)``, conditioned on the Phase-1 output and the
  other configured spatial conditioning fields.
* **Probability path**: the regularized straight interpolation
  ``x_t = [1 - (1-sigma_min)t] * x_0 + t * r`` for ``t`` in ``[0, 1]``.
* **Velocity target**:
  ``u_t = r - (1-sigma_min) * x_0``, constant along each conditional path.
* **Model**: ``v_theta(x_t, t, cond)`` regressed on ``u_t`` with a masked MSE.
* **Inference**: integrate ``dx/dt = v_theta(x, t, cond)`` from ``t = 0`` to
  ``t = 1`` with ``integration_steps`` uniform steps using the configured solver
  (``euler``, ``midpoint`` or ``heun``). The final state ``x_1`` is converted
  to the clean predicted residual with the same path algebra.

``t`` is the integration coordinate of a probability path. It is **not** a
physical timestamp, a forecast lead time, or a temporal positional index, and it
never touches dataset time handling.

Reproducibility
---------------
The initial state ``x_0`` is drawn from a caller-supplied
:class:`torch.Generator`, so ensembles are reproducible and individual members
can be compared one-to-one across implementations. Setting
``stochastic_initialization: false`` starts the integration from ``x_0 = 0``,
which makes the refiner a deterministic zero-source-path corrector (not, in
general, the nonlinear ensemble mean).
"""

from __future__ import annotations

from typing import Callable

import torch

from granitewxc.refinement.backbones import ConditionalResidualUNet, SpatialResidualTransformer
from granitewxc.refinement.base import (
    ResidualRefiner,
    masked_loss,
    register_refiner,
    residual_reconstruction_terms,
)
from granitewxc.refinement.config import RefinementConfig

__all__ = [
    "FlowMatchingUNetRefiner",
    "FlowMatchingTransformerRefiner",
    "integrate_flow",
    "flow_interpolate",
    "flow_to_clean",
]

#: The flow-time is fed to the network scaled to the same numeric range as the
#: diffusion timestep index so a single embedding implementation serves both.
_TIME_SCALE = 1000.0


def _flow_time_like(time: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    time = torch.as_tensor(time, device=reference.device, dtype=reference.dtype).reshape(-1)
    if time.numel() not in {1, reference.shape[0]}:
        raise ValueError(
            f"Flow time must be scalar or have batch size {reference.shape[0]}, "
            f"got {time.numel()}."
        )
    if time.numel() == 1 and reference.shape[0] > 1:
        time = time.expand(reference.shape[0])
    if not bool(torch.isfinite(time).all()) or bool(((time < 0.0) | (time > 1.0)).any()):
        raise ValueError(f"Flow time must be in [0, 1], got {time.tolist()}.")
    return time.reshape(-1, *([1] * (reference.ndim - 1)))


def flow_interpolate(
    source: torch.Tensor,
    target: torch.Tensor,
    time: torch.Tensor,
    sigma_min: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the regularized conditional-flow state and oracle velocity."""
    if source.shape != target.shape:
        raise ValueError(
            f"Flow source/target shapes differ: {tuple(source.shape)} and {tuple(target.shape)}."
        )
    sigma = float(sigma_min)
    if not 0.0 <= sigma <= 0.5:
        raise ValueError(f"sigma_min must be in [0, 0.5], got {sigma_min}.")
    if source.dtype in {torch.float16, torch.bfloat16}:
        source, target = source.float(), target.float()
    t = _flow_time_like(time, source)
    source_rate = 1.0 - sigma
    source_coefficient = 1.0 - source_rate * t
    state = source_coefficient * source + t * target
    velocity = target - source_rate * source
    return state, velocity


def flow_to_clean(
    state: torch.Tensor,
    velocity: torch.Tensor,
    time: torch.Tensor,
    sigma_min: float,
) -> torch.Tensor:
    """Recover the clean target implied by a regularized flow state/velocity."""
    if state.shape != velocity.shape:
        raise ValueError(
            f"Flow state/velocity shapes differ: {tuple(state.shape)} and {tuple(velocity.shape)}."
        )
    sigma = float(sigma_min)
    if not 0.0 <= sigma <= 0.5:
        raise ValueError(f"sigma_min must be in [0, 0.5], got {sigma_min}.")
    if state.dtype in {torch.float16, torch.bfloat16}:
        state = state.float()
    velocity = velocity.to(state.dtype)
    t = _flow_time_like(time, state)
    source_rate = 1.0 - sigma
    source_coefficient = 1.0 - source_rate * t
    return source_rate * state + source_coefficient * velocity


def integrate_flow(
    initial_state: torch.Tensor,
    velocity: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    steps: int,
    solver: str,
    t_start: float = 0.0,
    t_end: float = 1.0,
    trajectory_callback: Callable[[str, int, torch.Tensor, torch.Tensor], None] | None = None,
) -> torch.Tensor:
    """Integrate ``dx/dt = velocity(x, t)`` on an explicit increasing grid.

    The helper is shared by both backbones and independently testable with a
    known constant field, preventing a sign or interval reversal from hiding in
    model-specific sampling code.
    """
    steps = int(steps)
    if steps < 1:
        raise ValueError("integration_steps must be >= 1")
    solver = str(solver).lower()
    if solver not in {"euler", "midpoint", "heun"}:
        raise ValueError(f"Unsupported flow solver {solver!r}")
    if not float(t_end) > float(t_start):
        raise ValueError(
            f"Flow integration must proceed forward in time, got [{t_start}, {t_end}]."
        )

    # fp16 rounds 100 + 0.01 to 100 at every ODE step. Retain fp32 state.
    x = initial_state.float() if initial_state.dtype in {torch.float16, torch.bfloat16} else initial_state
    if not bool(torch.isfinite(x).all()):
        raise FloatingPointError("Flow initial state must be finite")
    grid = torch.linspace(
        float(t_start),
        float(t_end),
        steps + 1,
        device=x.device,
        dtype=torch.float32,
    )

    def checked_velocity(state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        value = velocity(state, time)
        if value.shape != state.shape:
            raise ValueError(
                f"Velocity shape {tuple(value.shape)} does not match state {tuple(state.shape)}."
            )
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"Flow velocity is non-finite at process time {float(time)}")
        return value.to(state.dtype)

    if trajectory_callback is not None:
        trajectory_callback("initial", 0, grid[0].detach().clone(), x.detach().clone())
    for idx in range(steps):
        t0 = grid[idx]
        t1 = grid[idx + 1]
        dt = (t1 - t0).to(x.dtype)
        if solver == "euler":
            x = x + dt * checked_velocity(x, t0)
        elif solver == "midpoint":
            k1 = checked_velocity(x, t0)
            midpoint = x + 0.5 * dt * k1
            x = x + dt * checked_velocity(midpoint, 0.5 * (t0 + t1))
        else:  # heun
            k1 = checked_velocity(x, t0)
            euler = x + dt * k1
            k2 = checked_velocity(euler, t1)
            x = x + 0.5 * dt * (k1 + k2)
        if trajectory_callback is not None:
            trajectory_callback("intermediate", idx + 1, t1.detach().clone(), x.detach().clone())
    if not bool(torch.isfinite(x).all()):
        raise FloatingPointError("Flow integration produced a non-finite endpoint.")
    return x


class _BaseFlowMatchingRefiner(ResidualRefiner):
    def __init__(self, config: RefinementConfig, *, residual_channels: int, cond_channels: int) -> None:
        super().__init__(config, residual_channels=residual_channels, cond_channels=cond_channels)
        f = config.flow_matching
        self.integration_steps = f.integration_steps
        self.solver = f.solver
        self.sigma_min = f.sigma_min
        self.stochastic_initialization = f.stochastic_initialization
        self.time_sampling = f.time_sampling
        self.logit_normal_mean = f.logit_normal_mean
        self.logit_normal_std = f.logit_normal_std
        self.mean_path_loss_weight = float(f.mean_path_loss_weight)
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
        self.net = self._build_net(config)

    def _build_net(self, config: RefinementConfig) -> torch.nn.Module:  # pragma: no cover - abstract
        raise NotImplementedError

    def apply_correction_gate(self, residual: torch.Tensor) -> torch.Tensor:
        """Apply an optional gate without changing raw flow samples."""
        return residual

    @staticmethod
    def _embed_time(t: torch.Tensor) -> torch.Tensor:
        return t.float() * _TIME_SCALE

    def _sample_flow_time(
        self, batch: int, device: torch.device, generator: torch.Generator | None
    ) -> torch.Tensor:
        gen_device = generator.device if generator is not None else device
        if self.time_sampling == "uniform":
            t = torch.rand(batch, generator=generator, device=gen_device, dtype=torch.float32)
        else:  # logit_normal
            z = torch.randn(batch, generator=generator, device=gen_device, dtype=torch.float32)
            t = torch.sigmoid(z * self.logit_normal_std + self.logit_normal_mean)
        return t.to(device).clamp(0.0, 1.0)

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
        device, dtype = residual_target.device, residual_target.dtype

        x0 = self._randn(tuple(residual_target.shape), device, dtype, generator)
        t = self._sample_flow_time(batch, device, generator)
        x_t, velocity_target = flow_interpolate(
            x0, residual_target, t, self.sigma_min
        )

        prediction = self.net(x_t.to(conditioning.dtype), conditioning, self._embed_time(t))
        process_loss = masked_loss(prediction, velocity_target, valid_mask, self.loss_kind)
        clean_prediction = flow_to_clean(
            x_t, prediction, t, self.sigma_min
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

        # The stochastic Gaussian branch above is the actual flow-matching
        # objective and remains unchanged.  Optionally supervise the same vector
        # field on the deterministic x_0=0 trajectory, x_t=t*r and u_t=r.  This
        # adds supervision on a different source path. Its zero-source endpoint
        # is not generally the conditional ensemble mean of the Gaussian-source
        # sampler; this auxiliary changes the training objective.
        mean_path_state = torch.zeros_like(x_t)
        mean_path_velocity_prediction = torch.zeros_like(prediction)
        mean_path_loss = prediction.float().sum() * 0.0
        if self.mean_path_loss_weight > 0.0:
            t_view = _flow_time_like(t, residual_target)
            mean_path_state = t_view * residual_target
            mean_path_velocity_prediction = self.net(
                mean_path_state, conditioning, self._embed_time(t)
            )
            mean_path_loss = masked_loss(
                mean_path_velocity_prediction,
                residual_target,
                valid_mask,
                "huber",
            )
        loss = (
            process_loss
            + self.reconstruction_loss_weight
            * (terms["reconstruction_loss"] + terms["gate_calibration_loss"])
            + self.multiscale_loss_weight
            * terms["multiscale_loss"]
            + self.gradient_loss_weight * terms["gradient_loss"]
            + self.mean_bias_loss_weight * terms["mean_bias_loss"]
            + self.mean_path_loss_weight * mean_path_loss
        )
        result = {
            "loss": loss,
            "process_loss": process_loss,
            **terms,
            "flow_time": t,
            "prediction": prediction,
            "clean_residual_prediction": clean_prediction,
            "mean_path_loss": mean_path_loss,
            # Detached snapshots make the sampled source/path/target explicit
            # to diagnostics without retaining or mutating the training graph.
            "source_state": x0.detach(),
            "interpolated_state": x_t.detach(),
            "target_velocity": velocity_target.detach(),
        }
        if self.mean_path_loss_weight > 0.0:
            result.update(
                {
                    "zero_source_state": mean_path_state.detach(),
                    "zero_source_velocity_prediction": mean_path_velocity_prediction.detach(),
                    "zero_source_target_velocity": residual_target.detach(),
                }
            )
        return result

    # -- integration -----------------------------------------------------
    @torch.no_grad()
    def _integrate_from(
        self,
        conditioning: torch.Tensor,
        initial_state: torch.Tensor,
        steps: int,
        trajectory_callback: Callable[[str, int, torch.Tensor, torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        if steps < 1:
            raise ValueError("integration_steps must be >= 1")
        shape = self.residual_shape(conditioning)
        device, dtype = conditioning.device, conditioning.dtype
        if initial_state.shape != shape:
            raise ValueError(
                f"Flow initial state shape {tuple(initial_state.shape)} does not "
                f"match residual shape {shape}."
            )
        x = initial_state.to(device=device, dtype=dtype)

        batch = shape[0]

        def velocity(state: torch.Tensor, t_scalar: torch.Tensor) -> torch.Tensor:
            t_batch = t_scalar.expand(batch)
            return self.net(state.to(dtype), conditioning, self._embed_time(t_batch))

        endpoint = integrate_flow(
            x, velocity, steps=steps, solver=self.solver,
            trajectory_callback=trajectory_callback,
        )
        if self.sigma_min == 0.0:
            clean = endpoint
        else:
            terminal_time = torch.ones(batch, device=device, dtype=torch.float32)
            terminal_velocity = self.net(
                endpoint.to(dtype), conditioning, self._embed_time(terminal_time)
            )
            clean = flow_to_clean(endpoint, terminal_velocity, terminal_time, self.sigma_min)
        if trajectory_callback is not None:
            trajectory_callback(
                "final_normalized_residual", steps,
                torch.tensor(1.0, device=device), clean.detach().clone(),
            )
        return clean

    @torch.no_grad()
    def sample(
        self,
        conditioning: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        num_steps: int | None = None,
        initial_state: torch.Tensor | None = None,
        trajectory_callback: Callable[[str, int, torch.Tensor, torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        """Sample using optional shared source noise and detached snapshots."""
        steps = int(num_steps if num_steps is not None else self.integration_steps)
        shape = self.residual_shape(conditioning)
        device, dtype = conditioning.device, conditioning.dtype
        if initial_state is not None:
            initial = initial_state.detach().clone().to(device=device, dtype=dtype)
        elif self.stochastic_initialization:
            initial = self._randn(shape, device, dtype, generator)
        else:
            initial = torch.zeros(shape, device=device, dtype=dtype)
        return self._integrate_from(conditioning, initial, steps, trajectory_callback)

    @torch.no_grad()
    def deterministic_residual(self, conditioning: torch.Tensor) -> torch.Tensor:
        """Integrate the explicit zero-source path without mutating configuration."""
        shape = self.residual_shape(conditioning)
        initial = torch.zeros(
            shape, device=conditioning.device, dtype=conditioning.dtype
        )
        return self._integrate_from(
            conditioning, initial, int(self.integration_steps)
        )


@register_refiner("flow_matching_unet")
class FlowMatchingUNetRefiner(_BaseFlowMatchingRefiner):
    """Convolutional UNet conditional flow-matching residual refiner."""

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


@register_refiner("flow_matching_transformer")
class FlowMatchingTransformerRefiner(_BaseFlowMatchingRefiner):
    """Spatial-token Transformer conditional flow-matching residual refiner."""

    def __init__(self, config: RefinementConfig, *, residual_channels: int, cond_channels: int) -> None:
        super().__init__(
            config,
            residual_channels=residual_channels,
            cond_channels=cond_channels,
        )
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
