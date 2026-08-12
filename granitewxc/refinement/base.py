"""Common Phase-2 residual-refiner interface and registry.

Every refiner implements the same three operations so that training and
inference never need a per-type ``if/elif`` chain:

``training_loss(residual_target, conditioning, valid_mask)``
    One stochastic training objective evaluation.

``sample(conditioning, generator=None)``
    One reverse/integration pass producing a predicted residual.

``deterministic_residual(conditioning)``
    The cheapest deterministic (mean-like) residual estimate, used for
    diagnostics and for single-member "deterministic stochastic" inference.

All tensors are in the **normalized target space** (see
:mod:`granitewxc.refinement.target_space`).
"""

from __future__ import annotations

import abc
import contextlib
from typing import Callable, Sequence

import torch
from torch import nn

from granitewxc.refinement.config import ConfigValidationError, RefinementConfig

__all__ = [
    "ChunkNoiseSource",
    "ResidualRefiner",
    "register_refiner",
    "build_refiner",
    "available_refiners",
    "masked_loss",
    "clean_residual_auxiliary_loss",
]


_REGISTRY: dict[str, Callable[..., "ResidualRefiner"]] = {}


class ChunkNoiseSource:
    """Per-ensemble-member noise source.

    Batched ensemble generation replicates the (identical) conditioning across
    members with ``repeat_interleave``, giving the flat layout
    ``[b0m0, b0m1, ..., b0m(M-1), b1m0, ...]``.  Drawing one big ``randn`` for
    that flat batch would make the result depend on how many members happen to
    share a chunk.  Instead, every member owns a dedicated
    :class:`torch.Generator`, so member ``m`` receives exactly the same noise
    sequence whether it was evaluated alone or inside a chunk of any size.

    This is what makes "serial versus batched ensemble generation" a bitwise
    parity test rather than a statistical one.
    """

    def __init__(self, generators: Sequence[torch.Generator], batch_size: int) -> None:
        self.generators = list(generators)
        self.batch_size = int(batch_size)

    def randn(self, shape: tuple[int, ...], device, dtype) -> torch.Tensor:
        members = len(self.generators)
        total, *rest = shape
        if total != self.batch_size * members:
            raise RuntimeError(
                f"ChunkNoiseSource expected a leading dimension of "
                f"{self.batch_size * members}, got {total}."
            )
        per_member = [
            torch.randn(
                (self.batch_size, *rest), generator=g, device=g.device, dtype=torch.float32
            ).to(device=device, dtype=dtype)
            for g in self.generators
        ]
        return torch.stack(per_member, dim=1).reshape(total, *rest)


def register_refiner(name: str):
    """Class decorator registering a refiner under ``name``."""

    def _decorator(cls):
        key = str(name).lower()
        if key in _REGISTRY:
            raise ValueError(f"Refiner {key!r} is already registered.")
        _REGISTRY[key] = cls
        cls.refiner_type = key
        return cls

    return _decorator


def available_refiners() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def build_refiner(
    config: RefinementConfig,
    *,
    residual_channels: int,
    cond_channels: int,
) -> "ResidualRefiner | None":
    """Factory for Phase-2 refiners.

    Returns ``None`` when refinement is disabled so callers can keep a single
    code path for the deterministic configuration.
    """
    if not config.is_active:
        return None
    cls = _REGISTRY.get(config.type)
    if cls is None:
        raise ConfigValidationError(
            f"Unknown refinement.type {config.type!r}. Registered refiners: "
            f"{list(available_refiners())}."
        )
    return cls(config, residual_channels=residual_channels, cond_channels=cond_channels)


def masked_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None,
    kind: str = "mse",
    channel_weights: torch.Tensor | Sequence[float] | None = None,
) -> torch.Tensor:
    """Masked reduction shared by every refiner.

    Invalid cells (missing target values) contribute exactly zero and are
    excluded from the denominator, so masks never bias the loss magnitude.
    The reduction is always performed in float32 for numerical stability under
    automatic mixed precision.
    """
    diff = (prediction - target).float()
    if kind == "mse":
        elementwise = diff.pow(2)
    elif kind == "l1":
        elementwise = diff.abs()
    elif kind == "huber":
        elementwise = torch.nn.functional.huber_loss(
            prediction.float(), target.float(), reduction="none", delta=1.0
        )
    else:
        raise ConfigValidationError(f"Unsupported refinement.loss {kind!r}")

    if valid_mask is None:
        mask = torch.ones_like(elementwise)
    else:
        mask = valid_mask.to(elementwise.dtype)
    if channel_weights is not None:
        weights = torch.as_tensor(
            channel_weights, device=elementwise.device, dtype=elementwise.dtype
        )
        if weights.numel() != elementwise.shape[1]:
            raise ValueError(
                f"Expected {elementwise.shape[1]} variable weights in output-channel "
                f"order, got {weights.numel()}."
            )
        mask = mask * weights.reshape(1, -1, 1, 1)
    denom = mask.sum().clamp(min=1.0)
    return (elementwise * mask).sum() / denom


def _masked_channel_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    axes = tuple(index for index in range(value.ndim) if index != 1)
    denominator = mask.sum(dim=axes).clamp(min=1.0)
    return (value * mask).sum(dim=axes) / denominator


def clean_residual_auxiliary_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None,
    config,
) -> dict[str, torch.Tensor]:
    """Losses on a valid estimate of the clean normalized residual.

    Diffusion calls this after converting its configured parameterization to
    ``x0``; flow matching calls it after extrapolating the predicted velocity to
    the path endpoint.  This keeps auxiliary losses mathematically consistent
    with the stochastic objective instead of comparing epsilon/velocity with a
    clean field.
    """

    zero = prediction.float().sum() * 0.0
    mask = (
        torch.ones_like(target, dtype=torch.bool)
        if valid_mask is None
        else valid_mask.to(device=target.device, dtype=torch.bool)
    )
    variable_weights = config.variable_weights or None
    terms: dict[str, torch.Tensor] = {}

    if config.reconstruction_weight > 0:
        terms["clean_reconstruction_loss"] = masked_loss(
            prediction,
            target,
            mask,
            config.reconstruction_loss,
            variable_weights,
        )

    if config.bias_weight > 0:
        diff = (prediction - target).float()
        bias = _masked_channel_mean(diff, mask.float())
        if variable_weights:
            weights = torch.as_tensor(variable_weights, device=bias.device, dtype=bias.dtype)
            terms["clean_bias_loss"] = (bias.square() * weights).sum() / weights.sum()
        else:
            terms["clean_bias_loss"] = bias.square().mean()

    if config.gradient_weight > 0:
        horizontal_mask = mask[..., :, 1:] & mask[..., :, :-1]
        vertical_mask = mask[..., 1:, :] & mask[..., :-1, :]
        horizontal_prediction = prediction[..., :, 1:] - prediction[..., :, :-1]
        horizontal_target = target[..., :, 1:] - target[..., :, :-1]
        vertical_prediction = prediction[..., 1:, :] - prediction[..., :-1, :]
        vertical_target = target[..., 1:, :] - target[..., :-1, :]
        h_loss = masked_loss(
            horizontal_prediction,
            horizontal_target,
            horizontal_mask,
            "l1",
            variable_weights,
        )
        v_loss = masked_loss(
            vertical_prediction,
            vertical_target,
            vertical_mask,
            "l1",
            variable_weights,
        )
        terms["clean_gradient_loss"] = 0.5 * (h_loss + v_loss)

    if config.laplacian_weight > 0:
        if prediction.shape[-2] >= 3 and prediction.shape[-1] >= 3:
            pred_lap = (
                prediction[..., 1:-1, :-2]
                + prediction[..., 1:-1, 2:]
                + prediction[..., :-2, 1:-1]
                + prediction[..., 2:, 1:-1]
                - 4.0 * prediction[..., 1:-1, 1:-1]
            )
            target_lap = (
                target[..., 1:-1, :-2]
                + target[..., 1:-1, 2:]
                + target[..., :-2, 1:-1]
                + target[..., 2:, 1:-1]
                - 4.0 * target[..., 1:-1, 1:-1]
            )
            lap_mask = (
                mask[..., 1:-1, 1:-1]
                & mask[..., 1:-1, :-2]
                & mask[..., 1:-1, 2:]
                & mask[..., :-2, 1:-1]
                & mask[..., 2:, 1:-1]
            )
            terms["clean_laplacian_loss"] = masked_loss(
                pred_lap, target_lap, lap_mask, "l1", variable_weights
            )
        else:
            terms["clean_laplacian_loss"] = zero

    if config.multiscale_weight > 0:
        scale_losses = []
        for kernel in (2, 4):
            if prediction.shape[-2] < kernel or prediction.shape[-1] < kernel:
                continue
            mask_f = mask.float()
            pooled_count = torch.nn.functional.avg_pool2d(mask_f, kernel, kernel)
            pred_pool = torch.nn.functional.avg_pool2d(
                prediction * mask_f, kernel, kernel
            ) / pooled_count.clamp(min=1.0 / (kernel * kernel))
            target_pool = torch.nn.functional.avg_pool2d(
                target * mask_f, kernel, kernel
            ) / pooled_count.clamp(min=1.0 / (kernel * kernel))
            scale_losses.append(
                masked_loss(
                    pred_pool,
                    target_pool,
                    pooled_count > 0,
                    "l1",
                    variable_weights,
                )
            )
        terms["clean_multiscale_loss"] = (
            torch.stack(scale_losses).mean() if scale_losses else zero
        )

    if config.tail_weight > 0:
        tail_mask = mask & (target.abs() >= float(config.tail_threshold))
        terms["clean_tail_loss"] = (
            masked_loss(prediction, target, tail_mask, "huber", variable_weights)
            if bool(tail_mask.any())
            else zero
        )

    weighted = zero
    weights_by_name = {
        "clean_reconstruction_loss": config.reconstruction_weight,
        "clean_bias_loss": config.bias_weight,
        "clean_gradient_loss": config.gradient_weight,
        "clean_laplacian_loss": config.laplacian_weight,
        "clean_multiscale_loss": config.multiscale_weight,
        "clean_tail_loss": config.tail_weight,
    }
    for name, value in terms.items():
        weighted = weighted + float(weights_by_name[name]) * value
    terms["auxiliary_loss"] = weighted
    return terms


class ResidualRefiner(nn.Module, abc.ABC):
    """Base class for all Phase-2 stochastic residual refiners."""

    #: set by :func:`register_refiner`
    refiner_type: str = "abstract"

    def __init__(
        self,
        config: RefinementConfig,
        *,
        residual_channels: int,
        cond_channels: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.residual_channels = int(residual_channels)
        self.cond_channels = int(cond_channels)
        self.loss_kind = config.loss
        self._noise_source: ChunkNoiseSource | None = None
        normalization = config.residual_normalization
        self.residual_normalization_enabled = bool(normalization.enabled)
        self.residual_normalization_epsilon = float(normalization.epsilon)
        self.register_buffer(
            "residual_mean",
            torch.zeros(1, self.residual_channels, 1, 1),
            persistent=self.residual_normalization_enabled,
        )
        self.register_buffer(
            "residual_std",
            torch.ones(1, self.residual_channels, 1, 1),
            persistent=self.residual_normalization_enabled,
        )
        self.register_buffer(
            "residual_count",
            torch.zeros(self.residual_channels, dtype=torch.int64),
            persistent=self.residual_normalization_enabled,
        )
        self.register_buffer(
            "residual_normalization_fitted",
            torch.tensor(not self.residual_normalization_enabled, dtype=torch.bool),
            persistent=self.residual_normalization_enabled,
        )
        variable_weights = config.auxiliary_loss.variable_weights
        if variable_weights and len(variable_weights) != self.residual_channels:
            raise ConfigValidationError(
                "refinement.auxiliary_loss.variable_weights must match the explicit "
                f"output-channel count ({self.residual_channels}), got {len(variable_weights)}."
            )

    @contextlib.contextmanager
    def use_noise_source(self, source: "ChunkNoiseSource | None"):
        """Temporarily route every ``randn`` draw through ``source``."""
        previous = self._noise_source
        self._noise_source = source
        try:
            yield
        finally:
            self._noise_source = previous

    # -- required API ----------------------------------------------------
    @abc.abstractmethod
    def training_loss(
        self,
        residual_target: torch.Tensor,
        conditioning: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return ``{"loss": scalar, ...}`` for one stochastic training step."""

    @abc.abstractmethod
    def sample(
        self,
        conditioning: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        num_steps: int | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Low-level draw in residual-refiner training space.

        When residual standardization is enabled this returns standardized
        values. Prefer :meth:`sample_target_space` outside the two-phase
        wrapper so the fitted transform cannot be accidentally skipped.
        """

    @abc.abstractmethod
    def deterministic_residual(
        self,
        conditioning: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Low-level deterministic estimate in refiner training space."""

    # -- shared helpers --------------------------------------------------
    def residual_shape(self, conditioning: torch.Tensor) -> tuple[int, ...]:
        return (conditioning.shape[0], self.residual_channels, *conditioning.shape[-2:])

    @staticmethod
    def apply_valid_mask(
        value: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Zero invalid state cells so they cannot feed neighboring pixels."""
        if valid_mask is None:
            return value
        mask = valid_mask.to(device=value.device, dtype=torch.bool)
        try:
            mask = torch.broadcast_to(mask, value.shape)
        except RuntimeError as exc:
            raise ValueError(
                f"Residual validity mask {tuple(mask.shape)} cannot broadcast "
                f"to state {tuple(value.shape)}."
            ) from exc
        return torch.where(mask, value, torch.zeros_like(value))

    def _randn(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        if self._noise_source is not None:
            return self._noise_source.randn(shape, device, dtype)
        if generator is not None and generator.device != torch.device(device):
            # ``torch.randn`` requires the generator and the output tensor to be
            # on the same device. Draw on the generator's device and move.
            drawn = torch.randn(shape, generator=generator, device=generator.device, dtype=torch.float32)
            return drawn.to(device=device, dtype=dtype)
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)

    def set_residual_normalization(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        count: torch.Tensor | Sequence[int],
    ) -> None:
        """Install training-only per-variable statistics as checkpoint buffers."""
        mean = torch.as_tensor(mean, dtype=self.residual_mean.dtype, device=self.residual_mean.device)
        std = torch.as_tensor(std, dtype=self.residual_std.dtype, device=self.residual_std.device)
        count = torch.as_tensor(count, dtype=torch.int64, device=self.residual_count.device)
        if mean.numel() != self.residual_channels or std.numel() != self.residual_channels:
            raise ValueError("Residual mean/std must contain one value per ordered output channel.")
        if count.numel() != self.residual_channels or bool((count <= 0).any()):
            raise ValueError("Residual normalization requires at least one valid training value per channel.")
        std = std.clamp(min=self.residual_normalization_epsilon)
        self.residual_mean.copy_(mean.reshape_as(self.residual_mean))
        self.residual_std.copy_(std.reshape_as(self.residual_std))
        self.residual_count.copy_(count.reshape_as(self.residual_count))
        self.residual_normalization_fitted.fill_(True)

    def _require_residual_normalization(self) -> None:
        if self.residual_normalization_enabled and not bool(self.residual_normalization_fitted):
            raise RuntimeError(
                "Residual normalization is enabled but has not been fitted on the Phase-2 "
                "training split. Call model.fit_residual_normalizer(train_loader) before "
                "training, or load a Phase-2 checkpoint containing the fitted buffers."
            )

    def normalize_residual(self, residual: torch.Tensor) -> torch.Tensor:
        self._require_residual_normalization()
        if not self.residual_normalization_enabled:
            return residual
        return (residual - self.residual_mean.to(residual)) / self.residual_std.to(residual)

    def denormalize_residual(self, residual: torch.Tensor) -> torch.Tensor:
        self._require_residual_normalization()
        if not self.residual_normalization_enabled:
            return residual
        return residual * self.residual_std.to(residual) + self.residual_mean.to(residual)

    @torch.no_grad()
    def sample_target_space(
        self,
        conditioning: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        num_steps: int | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Draw one residual in Phase-1 normalized target space."""
        self._require_residual_normalization()
        sample = self.sample(
            conditioning,
            generator=generator,
            num_steps=num_steps,
            valid_mask=valid_mask,
        )
        sample = self.denormalize_residual(sample)
        return self.apply_valid_mask(sample, valid_mask)

    def residual_normalization_metadata(self) -> dict[str, object]:
        return {
            "enabled": self.residual_normalization_enabled,
            "fitted": bool(self.residual_normalization_fitted),
            "mean": self.residual_mean.detach().cpu().reshape(-1).tolist(),
            "std": self.residual_std.detach().cpu().reshape(-1).tolist(),
            "count": self.residual_count.detach().cpu().reshape(-1).tolist(),
            "space": "phase1_normalized_residual",
            "fit_split": "training",
        }

    def auxiliary_losses(
        self,
        clean_prediction: torch.Tensor,
        clean_target: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        return clean_residual_auxiliary_loss(
            clean_prediction, clean_target, valid_mask, self.config.auxiliary_loss
        )

    def extra_repr(self) -> str:
        return (
            f"type={self.refiner_type}, residual_channels={self.residual_channels}, "
            f"cond_channels={self.cond_channels}"
        )
