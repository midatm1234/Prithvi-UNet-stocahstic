"""Sequence-aware conditioning and noise for the existing residual refiners.

All four existing Phase-2 refiners (``diffusion_unet``, ``diffusion_transformer``,
``flow_matching_unet``, ``flow_matching_transformer``) are reused unchanged. This
module supplies the two things a *sequence* of refined frames needs that a single
frame does not:

1. **Temporal conditioning channels** -- so the refiner knows where in a sequence
   it is and what the Phase-1 temporal state was, instead of treating each date
   as an isolated problem.
2. **Temporally correlated noise** -- so an ensemble member's trajectory is
   coherent across dates.

On the noise question, the literature bounds the answer from both sides and the
default here sits deliberately between them:

* **i.i.d. per-frame noise** (the existing behaviour, and what most weather
  diffusion work does) destroys temporal autocorrelation in the refined
  increment. Schillinger et al. (2026, arXiv:2509.26258) measure this: their
  per-day-independent variant "removes temporal autocorrelation".
* **Identical noise at every date** (a single shared draw) manufactures
  persistence rather than modelling it, and produces over-correlated,
  under-dispersed ensembles -- again measured in the same study, where the
  shared-initialization variant *over*-estimates lag-1 autocorrelation
  (+0.22 for precipitation).

:class:`AR1NoiseSource` interpolates between the two with an explicit lag-1
coefficient, ``rho = 0`` reproducing the legacy i.i.d. behaviour exactly and
``rho -> 1`` approaching the shared-noise failure mode (which the config layer
rejects). The construction is the AR(1) stationary form

    ``e_t = rho * e_{t-1} + sqrt(1 - rho^2) * z_t``,

so every frame's marginal remains exactly standard normal -- the refiner still
sees the noise distribution it was trained against, and only the *correlation
between* frames changes. This matters because train/test noise-structure
mismatch is a documented failure mode (Deng et al., 2026, arXiv:2608.02575).

Per-member isolation is preserved: each ensemble member owns its own
:class:`torch.Generator` *and* its own AR(1) state, following the same contract
as :class:`granitewxc.refinement.base.ChunkNoiseSource`, so member ``m``'s
trajectory is identical whether it was drawn alone or inside a batch of any size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from granitewxc.temporal.config import TemporalConfig, TemporalRefinementConfig

__all__ = [
    "AR1NoiseSource",
    "TemporalConditioningAdapter",
    "temporal_conditioning_channels",
    "append_temporal_conditioning",
    "check_refiner_temporal_compatibility",
    "RefinerTemporalCompatibilityError",
]


class RefinerTemporalCompatibilityError(RuntimeError):
    """Raised when temporal conditioning invalidates a Phase-2 checkpoint."""


# ---------------------------------------------------------------------------
# noise
# ---------------------------------------------------------------------------
class AR1NoiseSource:
    """Per-member AR(1)-correlated standard normal noise.

    Layout matches :class:`~granitewxc.refinement.base.ChunkNoiseSource`: the
    flat leading dimension is ``batch_size * members`` ordered
    ``[b0m0, b0m1, ..., b1m0, ...]``.

    Call :meth:`advance` once per output date. Within a date, repeated
    :meth:`randn` calls (the many denoising steps of one frame) draw *fresh*
    independent noise -- only the per-frame "seed" noise that determines the
    trajectory is correlated in time. Correlating the internal denoising steps
    too would couple the solver's integration error across dates, which is not
    what temporal coherence means.
    """

    def __init__(
        self,
        generators: Sequence[torch.Generator],
        batch_size: int,
        *,
        rho: float = 0.0,
    ) -> None:
        if not 0.0 <= rho < 1.0:
            raise ValueError(f"rho must satisfy 0 <= rho < 1, got {rho}")
        self.generators = list(generators)
        self.batch_size = int(batch_size)
        self.rho = float(rho)
        self._state: torch.Tensor | None = None
        self._frame = 0

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> None:
        """Start a fresh sequence (drops the carried AR(1) state)."""
        self._state = None
        self._frame = 0

    def advance(self) -> None:
        """Move to the next output date."""
        self._frame += 1

    # -- draws -------------------------------------------------------------
    def _base_draw(self, shape: tuple[int, ...], device, dtype) -> torch.Tensor:
        members = len(self.generators)
        total, *rest = shape
        if total != self.batch_size * members:
            raise RuntimeError(
                f"AR1NoiseSource expected a leading dimension of "
                f"{self.batch_size * members}, got {total}."
            )
        per_member = [
            torch.randn(
                (self.batch_size, *rest), generator=g, device=g.device, dtype=torch.float32
            ).to(device=device, dtype=dtype)
            for g in self.generators
        ]
        return torch.stack(per_member, dim=1).reshape(total, *rest)

    def randn(self, shape: tuple[int, ...], device, dtype) -> torch.Tensor:
        """Independent noise, used for the solver's internal steps."""
        return self._base_draw(shape, device, dtype)

    def frame_noise(self, shape: tuple[int, ...], device, dtype) -> torch.Tensor:
        """AR(1)-correlated noise for the current output date.

        With ``rho == 0`` this is exactly :meth:`randn`, so the legacy path is
        bit-for-bit preserved when correlation is switched off.
        """
        innovation = self._base_draw(shape, device, dtype)
        if self.rho == 0.0:
            self._state = innovation
            return innovation
        if self._state is None or self._state.shape != innovation.shape:
            # First frame of a sequence: draw from the stationary distribution,
            # which for a standard-normal AR(1) marginal is just N(0, 1).
            self._state = innovation
            return innovation
        rho = self.rho
        correlated = rho * self._state + math.sqrt(1.0 - rho * rho) * innovation
        self._state = correlated
        return correlated

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "ar1" if self.rho > 0 else "iid_per_frame",
            "rho": self.rho,
            "members": len(self.generators),
            "batch_size": self.batch_size,
            "frame": self._frame,
        }


# ---------------------------------------------------------------------------
# conditioning
# ---------------------------------------------------------------------------
def temporal_conditioning_channels(
    cfg: TemporalRefinementConfig,
    *,
    time_feature_dim: int,
    latent_projection_channels: int = 16,
) -> int:
    """Extra conditioning channels contributed by temporal information.

    ``none`` returns 0, which is what keeps existing Phase-2 checkpoints valid:
    the refiner's ``cond_channels`` is unchanged, so its first convolution still
    matches.
    """
    if cfg.temporal_conditioning == "none":
        return 0
    if cfg.temporal_conditioning == "time_features":
        return int(time_feature_dim)
    if cfg.temporal_conditioning == "latent_state":
        return int(time_feature_dim) + int(latent_projection_channels)
    raise ValueError(f"Unknown temporal_conditioning {cfg.temporal_conditioning!r}")


class TemporalConditioningAdapter(nn.Module):
    """Project Phase-1 temporal information into refiner conditioning channels.

    The Phase-1 temporal latent is ``[B, embed_dim, h, w]`` at bottleneck
    resolution (1024 x 16 x 16 for the SA case). Concatenating that directly
    would add 1024 conditioning channels and dominate the refiner's input, so it
    is first projected to a small number of channels and then bilinearly resized
    to the target grid.

    Time features are broadcast as constant spatial planes: they carry no spatial
    structure, and pretending otherwise would be misleading.
    """

    def __init__(
        self,
        *,
        latent_channels: int,
        time_feature_dim: int,
        projection_channels: int = 16,
    ):
        super().__init__()
        self.time_feature_dim = int(time_feature_dim)
        self.projection_channels = int(projection_channels)
        self.project = nn.Conv2d(int(latent_channels), self.projection_channels, kernel_size=1)
        nn.init.zeros_(self.project.bias)

    def forward(
        self,
        *,
        time_features: torch.Tensor | None,
        latent: torch.Tensor | None,
        size: tuple[int, int],
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        parts: list[torch.Tensor] = []
        if time_features is not None:
            b, f = time_features.shape
            planes = time_features.to(dtype).view(b, f, 1, 1).expand(b, f, *size)
            parts.append(planes)
        if latent is not None:
            projected = self.project(latent.to(dtype))
            if projected.shape[-2:] != size:
                projected = F.interpolate(
                    projected, size=size, mode="bilinear", align_corners=False
                )
            parts.append(projected)
        if not parts:
            return None
        return torch.cat(parts, dim=1)


def append_temporal_conditioning(
    conditioning: torch.Tensor,
    extra: torch.Tensor | None,
) -> torch.Tensor:
    """Concatenate temporal channels onto an existing conditioning tensor."""
    if extra is None:
        return conditioning
    if extra.shape[0] != conditioning.shape[0]:
        raise ValueError(
            f"Temporal conditioning batch {extra.shape[0]} does not match "
            f"conditioning batch {conditioning.shape[0]}."
        )
    if extra.shape[-2:] != conditioning.shape[-2:]:
        extra = F.interpolate(
            extra, size=conditioning.shape[-2:], mode="bilinear", align_corners=False
        )
    return torch.cat([conditioning, extra.to(conditioning.dtype)], dim=1)


# ---------------------------------------------------------------------------
# checkpoint compatibility
# ---------------------------------------------------------------------------
def check_refiner_temporal_compatibility(
    *,
    checkpoint_cond_channels: int | None,
    required_cond_channels: int,
    temporal_conditioning: str,
    checkpoint_path: str | None = None,
) -> None:
    """Accept or reject a Phase-2 checkpoint under the requested conditioning.

    Enabling temporal conditioning widens the refiner's first convolution, so a
    checkpoint trained without it is genuinely incompatible -- not "close enough
    to load with ``strict=False``". This raises with the exact channel counts and
    tells the caller which of the two valid actions to take.
    """
    if checkpoint_cond_channels is None:
        return
    if int(checkpoint_cond_channels) == int(required_cond_channels):
        return
    where = f" ({checkpoint_path})" if checkpoint_path else ""
    raise RefinerTemporalCompatibilityError(
        f"Phase-2 refinement checkpoint{where} was trained with "
        f"cond_channels={checkpoint_cond_channels}, but "
        f"temporal.refinement.temporal_conditioning={temporal_conditioning!r} requires "
        f"cond_channels={required_cond_channels}. The refiner's input projection has a "
        "different shape, so these weights do not describe the same model. Either set "
        "temporal.refinement.temporal_conditioning='none' to reuse the existing "
        "checkpoint unchanged, or retrain Phase 2 with temporal conditioning enabled and "
        "save it under a new checkpoint directory."
    )


@dataclass
class SequenceRefinementPlan:
    """Resolved decisions for refining a sequence.

    ``refine_whole_sequence`` records the semantics explicitly:

    * ``False`` (default) -- each date is refined causally, conditioned on the
      temporal metadata and (optionally) the Phase-1 temporal latent for that
      date, with the AR(1) noise state carried forward. This preserves the
      existing per-frame refiner architecture and its tiling behaviour.
    * ``True`` -- reserved for a future refiner that consumes a whole sequence at
      once. Not implemented; selecting it raises rather than silently doing the
      causal thing.
    """

    refine_whole_sequence: bool
    temporal_conditioning: str
    noise_kind: str
    noise_rho: float
    ensemble_size: int
    per_member_state: bool

    @staticmethod
    def from_config(cfg: TemporalConfig) -> "SequenceRefinementPlan":
        r = cfg.refinement
        return SequenceRefinementPlan(
            refine_whole_sequence=False,
            temporal_conditioning=r.temporal_conditioning,
            noise_kind=r.noise,
            noise_rho=r.noise_rho,
            ensemble_size=r.ensemble_size,
            per_member_state=r.per_member_state,
        )

    def build_noise_source(
        self,
        *,
        batch_size: int,
        device: torch.device | str,
        seed: int,
    ) -> AR1NoiseSource:
        """One generator per member, so member streams never interleave."""
        generators = []
        for member in range(self.ensemble_size):
            g = torch.Generator(device="cpu")
            g.manual_seed(int(seed) + 1000003 * member)
            generators.append(g)
        return AR1NoiseSource(
            generators,
            batch_size,
            rho=self.noise_rho if self.noise_kind == "ar1_correlated" else 0.0,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "refine_whole_sequence": self.refine_whole_sequence,
            "temporal_conditioning": self.temporal_conditioning,
            "noise_kind": self.noise_kind,
            "noise_rho": self.noise_rho,
            "ensemble_size": self.ensemble_size,
            "per_member_state": self.per_member_state,
        }
