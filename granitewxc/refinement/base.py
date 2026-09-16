"""Common Phase-2 residual-refiner interface and registry.

Every refiner implements the same three operations so that training and
inference never need a per-type ``if/elif`` chain:

``training_loss(residual_target, conditioning, valid_mask)``
    One stochastic training objective evaluation.

``sample(conditioning, generator=None)``
    One reverse/integration pass producing a predicted residual.

``deterministic_residual(conditioning)``
    A zero-source diagnostic estimate. For nonlinear stochastic refiners this
    is not the expectation of the sampled ensemble.

Residual targets and samples are in the dedicated residual-model space fitted
from physical training residuals. Conditioning may contain the Phase-1 field
in its own normalized target space; final corrections are inverse-transformed
to physical units exactly once by the two-phase wrapper.
"""

from __future__ import annotations

import abc
import contextlib
from typing import Callable, Sequence

import torch
from torch import nn

from granitewxc.refinement.config import ConfigValidationError, RefinementConfig
from granitewxc.refinement.randomness import NoiseSource

__all__ = [
    "ChunkNoiseSource",
    "ResidualRefiner",
    "register_refiner",
    "build_refiner",
    "available_refiners",
    "masked_loss",
    "residual_structure_losses",
    "residual_reconstruction_terms",
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

    This preserves random draws across ensemble chunking, but neural arithmetic
    can still vary with batch shape. It does not preserve sample/date streams
    when DataLoader batching or ordering changes. Use StableNoiseSource and
    canonical sample/member evaluation for that explicit inference contract.
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
) -> torch.Tensor:
    """Masked reduction shared by every refiner.

    Invalid cells (missing target values) contribute exactly zero and are
    excluded from the denominator. Each valid output channel is reduced first
    and receives equal weight, so unequal precipitation/temperature masks do
    not silently change the predictand weighting.
    The reduction is always performed in float32 for numerical stability under
    automatic mixed precision.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction {tuple(prediction.shape)} and target {tuple(target.shape)} "
            "must have identical shapes."
        )
    prediction32, target32 = prediction.float(), target.float()
    mask = None
    if valid_mask is not None:
        try:
            mask = torch.broadcast_to(
                valid_mask.to(device=prediction.device, dtype=torch.float32),
                prediction.shape,
            )
        except RuntimeError as exc:
            raise ValueError(
                f"valid_mask {tuple(valid_mask.shape)} is not broadcastable to loss "
                f"tensor {tuple(prediction.shape)}."
            ) from exc
        if not bool(torch.isfinite(mask).all()) or bool((mask < 0).any()):
            raise ValueError("Loss masks/weights must be finite and non-negative.")
        # Mask operands before arithmetic: NaN * 0 remains NaN, including in
        # backward. Cast before subtracting to avoid half-precision overflow.
        included = mask > 0
        prediction32 = torch.where(included, prediction32, torch.zeros_like(prediction32))
        target32 = torch.where(included, target32, torch.zeros_like(target32))
    diff = prediction32 - target32
    if kind == "mse":
        elementwise = diff.pow(2)
    elif kind == "l1":
        elementwise = diff.abs()
    elif kind == "huber":
        elementwise = torch.nn.functional.huber_loss(
            prediction32, target32, reduction="none", delta=1.0
        )
    else:
        raise ConfigValidationError(f"Unsupported refinement.loss {kind!r}")

    if mask is None:
        return elementwise.mean()

    # Residual channels have separate training-fitted scales and represent
    # different physical variables.  Reduce each channel over its own valid
    # cells first, then average the valid channels.  A precipitation mask with
    # fewer cells must not silently reduce that predictand's training weight
    # relative to (for example) an everywhere-valid temperature channel.
    if elementwise.ndim >= 2:
        reduce_dims = (0, *range(2, elementwise.ndim))
        counts = mask.sum(dim=reduce_dims)
        sums = (elementwise * mask).sum(dim=reduce_dims)
        channel_valid = counts > 0
        channel_means = sums / torch.where(channel_valid, counts, torch.ones_like(counts))
        if bool(channel_valid.any()):
            return channel_means[channel_valid].mean()
        return elementwise.sum() * 0.0

    denom = mask.sum()
    denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    return (elementwise * mask).sum() / denom


def residual_structure_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """Losses that match true residual structure without smoothing outputs.

    Each term compares prediction against the target residual. Nothing
    penalizes high frequencies merely for existing: genuine terrain-related
    gradients and extremes are rewarded when they agree with the truth.
    """
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError(
            "Residual structure losses require equal BCHW tensors, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}."
        )
    mask = (
        torch.ones_like(target, dtype=torch.bool)
        if valid_mask is None
        else torch.broadcast_to(
            valid_mask.to(device=target.device, dtype=torch.bool), target.shape
        )
    )
    prediction = torch.where(mask, prediction.float(), torch.zeros_like(prediction, dtype=torch.float32))
    target = torch.where(mask, target.float(), torch.zeros_like(target, dtype=torch.float32))

    multiscale_terms: list[torch.Tensor] = []
    for factor in (2, 4):
        if min(prediction.shape[-2:]) < factor:
            continue
        mask_f = mask.to(prediction.dtype)
        counts = torch.nn.functional.avg_pool2d(mask_f, factor, factor) * (factor**2)

        def pool_valid(values: torch.Tensor) -> torch.Tensor:
            summed = torch.nn.functional.avg_pool2d(
                torch.where(mask, values, torch.zeros_like(values)), factor, factor
            ) * (factor**2)
            return summed / counts.clamp(min=1.0)

        pooled_mask = counts > 0
        multiscale_terms.append(
            masked_loss(pool_valid(prediction), pool_valid(target), pooled_mask, "huber")
        )
    zero = prediction.float().sum() * 0.0
    multiscale = (
        torch.stack(multiscale_terms).mean() if multiscale_terms else zero
    )

    gradient_terms: list[torch.Tensor] = []
    if prediction.shape[-1] > 1:
        mask_x = mask[..., :, 1:] & mask[..., :, :-1]
        gradient_terms.append(
            masked_loss(
                prediction[..., :, 1:] - prediction[..., :, :-1],
                target[..., :, 1:] - target[..., :, :-1],
                mask_x,
                "huber",
            )
        )
    if prediction.shape[-2] > 1:
        mask_y = mask[..., 1:, :] & mask[..., :-1, :]
        gradient_terms.append(
            masked_loss(
                prediction[..., 1:, :] - prediction[..., :-1, :],
                target[..., 1:, :] - target[..., :-1, :],
                mask_y,
                "huber",
            )
        )
    gradient = torch.stack(gradient_terms).mean() if gradient_terms else zero

    mask_f = mask.to(prediction.dtype)
    counts = mask_f.sum(dim=(-2, -1)).clamp(min=1.0)
    pred_mean = torch.where(mask, prediction, torch.zeros_like(prediction)).sum(
        dim=(-2, -1)
    ) / counts
    target_mean = torch.where(mask, target, torch.zeros_like(target)).sum(
        dim=(-2, -1)
    ) / counts
    channel_valid = mask.any(dim=(-2, -1))
    mean_bias = masked_loss(pred_mean, target_mean, channel_valid, "huber")
    return {
        "multiscale_loss": multiscale,
        "gradient_loss": gradient,
        "mean_bias_loss": mean_bias,
    }


def residual_reconstruction_terms(
    clean_prediction: torch.Tensor,
    residual_target: torch.Tensor,
    valid_mask: torch.Tensor | None,
    zero_anchor: torch.Tensor,
    correction_gate: Callable[[torch.Tensor], torch.Tensor] | None = None,
    *, excluded_channels: tuple[int, ...] = (),
) -> dict[str, torch.Tensor]:
    """Shared deterministic residual losses and detached gate calibration.

    The stochastic network is always trained against the *raw* clean residual.
    A Transformer correction gate is calibrated on a detached raw prediction,
    so it learns to open without asking the network to inflate its output by
    ``1 / gate``.  That separation prevents a small identity gate from both
    concealing and actively encouraging an unstable raw sampler.
    """
    if clean_prediction.shape != residual_target.shape:
        raise ValueError(
            "Clean prediction and residual target must have the same shape, got "
            f"{tuple(clean_prediction.shape)} and {tuple(residual_target.shape)}."
        )
    if zero_anchor.shape != clean_prediction.shape:
        raise ValueError(
            "zero_residual must match the clean residual shape, got "
            f"{tuple(zero_anchor.shape)} and {tuple(clean_prediction.shape)}."
        )

    original_clean = clean_prediction
    excluded = None
    if excluded_channels:
        if len(set(excluded_channels)) != len(excluded_channels) or any(type(c) is not int or c < 0 or c >= clean_prediction.shape[1] for c in excluded_channels):
            raise ValueError("Excluded auxiliary channels must be unique valid indices")
        excluded = torch.zeros((1, clean_prediction.shape[1], 1, 1), device=clean_prediction.device, dtype=torch.bool)
        excluded[:, list(excluded_channels)] = True
        # Retain original masks/denominators, including active-channel counts.
        # Equal anchors give exactly zero selected-channel errors for every
        # reconstruction/structure/gate term without changing Tmax's coefficient.
        clean_prediction = torch.where(excluded, zero_anchor.detach(), clean_prediction)
        residual_target = torch.where(excluded, zero_anchor.detach(), residual_target)

    reconstruction = masked_loss(
        clean_prediction, residual_target, valid_mask, "huber"
    )
    structure = residual_structure_losses(
        clean_prediction, residual_target, valid_mask
    )
    zero = reconstruction * 0.0
    if correction_gate is None:
        effective = clean_prediction
        gate_calibration = zero
    else:
        effective = zero_anchor + correction_gate(
            (clean_prediction - zero_anchor).detach()
        )
        gate_calibration = masked_loss(
            effective, residual_target, valid_mask, "huber"
        )
    if excluded is not None:
        # Diagnostic predictions remain the real model estimates; only the
        # auxiliary loss evaluation above uses matched zero-error anchors.
        original_effective = original_clean if correction_gate is None else zero_anchor + correction_gate((original_clean - zero_anchor).detach())
        effective = torch.where(excluded, original_effective, effective)
    return {
        "reconstruction_loss": reconstruction,
        "gate_calibration_loss": gate_calibration,
        **structure,
        "effective_clean_residual_prediction": effective,
    }


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
        if any(c >= self.residual_channels for c in config.native_only_channels):
            raise ValueError("Native-only channel index exceeds residual channels")
        if any(c >= self.residual_channels for c in config.process_boundary_balance_channels):
            raise ValueError("Boundary-balance channel index exceeds residual channels")
        self._noise_source: NoiseSource | None = None

    @contextlib.contextmanager
    def use_noise_source(self, source: "NoiseSource | None"):
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
        zero_residual: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return ``{"loss": scalar, ...}`` for one stochastic training step.

        ``zero_residual`` is physical zero expressed in model space. It matters
        for affine residual standardization: a zero-initialized correction gate
        must interpolate around that value rather than around numeric zero.
        """

    @abc.abstractmethod
    def sample(
        self,
        conditioning: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """Draw one residual sample per batch element."""

    @abc.abstractmethod
    def deterministic_residual(self, conditioning: torch.Tensor) -> torch.Tensor:
        """Cheapest deterministic residual estimate (no random draw)."""

    # -- shared helpers --------------------------------------------------
    def residual_shape(self, conditioning: torch.Tensor) -> tuple[int, ...]:
        return (conditioning.shape[0], self.residual_channels, *conditioning.shape[-2:])

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

    def extra_repr(self) -> str:
        return (
            f"type={self.refiner_type}, residual_channels={self.residual_channels}, "
            f"cond_channels={self.cond_channels}"
        )
