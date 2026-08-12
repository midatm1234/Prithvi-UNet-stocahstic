"""Conditional flow-matching Phase-2 residual refiners.

Formulation
-----------
Following the rectified-flow / conditional-flow-matching construction used in
the Aurora ``aurora_finetune_flow_matching`` branch, adapted to spatial
downscaling residuals.

* **Source distribution** ``p_0``: isotropic Gaussian ``N(0, I)`` on the residual
  grid (``flow_matching.source_distribution: gaussian``).
* **Target distribution** ``p_1``: the distribution of the normalized residual
  ``r = encode(y) - deterministic_normalized`` conditioned on the Phase-1 output
  and the other configured spatial conditioning fields.
* **Probability path**: the straight (optimal-transport) interpolation
  ``x_t = (1 - t) * x_0 + t * r`` for ``t`` in ``[0, 1]``.
* **Velocity target**: ``u_t = d x_t / d t = r - x_0``, which is constant along
  each conditional path.
* **Model**: ``v_theta(x_t, t, cond)`` regressed on ``u_t`` with a masked MSE.
* **Inference**: integrate ``dx/dt = v_theta(x, t, cond)`` from ``t = 0`` to
  ``t = 1`` with ``integration_steps`` uniform steps using the configured solver
  (``euler``, ``midpoint`` or ``heun``). The final state ``x_1`` *is* the
  predicted residual.

``t`` is the integration coordinate of a probability path. It is **not** a
physical timestamp, a forecast lead time, or a temporal positional index, and it
never touches dataset time handling.

Reproducibility
---------------
The initial state ``x_0`` is drawn from a caller-supplied
:class:`torch.Generator`, so ensembles are reproducible and individual members
can be compared one-to-one across implementations. Setting
``stochastic_initialization: false`` starts the integration from ``x_0 = 0``,
which makes the refiner a deterministic (mean-path) corrector.
"""

from __future__ import annotations

import torch

from granitewxc.refinement.backbones import ConditionalResidualUNet, SpatialResidualTransformer
from granitewxc.refinement.base import ResidualRefiner, masked_loss, register_refiner
from granitewxc.refinement.config import RefinementConfig

__all__ = ["FlowMatchingUNetRefiner", "FlowMatchingTransformerRefiner"]

#: The flow-time is fed to the network scaled to the same numeric range as the
#: diffusion timestep index so a single embedding implementation serves both.
_TIME_SCALE = 1000.0


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
        self.net = self._build_net(config)

    def _build_net(self, config: RefinementConfig) -> torch.nn.Module:  # pragma: no cover - abstract
        raise NotImplementedError

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
        t = t.to(device)
        return t.clamp(self.sigma_min, 1.0 - self.sigma_min)

    # -- training --------------------------------------------------------
    def training_loss(
        self,
        residual_target: torch.Tensor,
        conditioning: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        batch = residual_target.shape[0]
        device, dtype = residual_target.device, residual_target.dtype

        x0 = self._randn(tuple(residual_target.shape), device, dtype, generator)
        t = self._sample_flow_time(batch, device, generator)
        t_b = t.reshape(-1, *([1] * (residual_target.ndim - 1))).to(dtype)

        x_t = (1.0 - t_b) * x0 + t_b * residual_target
        velocity_target = residual_target - x0

        prediction = self.net(x_t, conditioning, self._embed_time(t))
        loss = masked_loss(prediction, velocity_target, valid_mask, self.loss_kind)
        return {"loss": loss, "flow_time": t, "prediction": prediction}

    # -- integration -----------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        conditioning: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        steps = int(num_steps if num_steps is not None else self.integration_steps)
        if steps < 1:
            raise ValueError("integration_steps must be >= 1")
        shape = self.residual_shape(conditioning)
        device, dtype = conditioning.device, conditioning.dtype

        if self.stochastic_initialization:
            x = self._randn(shape, device, dtype, generator)
        else:
            x = torch.zeros(shape, device=device, dtype=dtype)

        # The integration grid is a fixed function of ``steps`` and is built once
        # per call, on the target device, so the loop performs no host sync.
        grid = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=torch.float32)
        batch = shape[0]

        def velocity(state: torch.Tensor, t_scalar: torch.Tensor) -> torch.Tensor:
            t_batch = t_scalar.expand(batch)
            return self.net(state, conditioning, self._embed_time(t_batch))

        for idx in range(steps):
            t0 = grid[idx]
            t1 = grid[idx + 1]
            dt = (t1 - t0).to(dtype)
            if self.solver == "euler":
                x = x + dt * velocity(x, t0)
            elif self.solver == "midpoint":
                k1 = velocity(x, t0)
                mid = x + 0.5 * dt * k1
                x = x + dt * velocity(mid, (t0 + t1) * 0.5)
            elif self.solver == "heun":
                k1 = velocity(x, t0)
                x_euler = x + dt * k1
                k2 = velocity(x_euler, t1)
                x = x + 0.5 * dt * (k1 + k2)
            else:  # pragma: no cover - guarded by config validation
                raise ValueError(f"Unsupported flow solver {self.solver!r}")
        return x

    @torch.no_grad()
    def deterministic_residual(self, conditioning: torch.Tensor) -> torch.Tensor:
        """Integrate the mean path from ``x_0 = 0`` with the configured solver."""
        saved = self.stochastic_initialization
        try:
            self.stochastic_initialization = False
            return self.sample(conditioning, generator=None)
        finally:
            self.stochastic_initialization = saved


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
        )


@register_refiner("flow_matching_transformer")
class FlowMatchingTransformerRefiner(_BaseFlowMatchingRefiner):
    """Spatial-token Transformer conditional flow-matching residual refiner."""

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
