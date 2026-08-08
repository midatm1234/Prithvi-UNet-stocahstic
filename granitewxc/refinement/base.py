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
        return elementwise.mean()
    mask = valid_mask.to(elementwise.dtype)
    denom = mask.sum().clamp(min=1.0)
    return (elementwise * mask).sum() / denom


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
