"""Unified two-phase Prithvi-UNet model wrapper.

Phase 1 (deterministic)
    Prithvi-WxC encoder/backbone + deterministic UNet decoder. Unchanged for
    CORDEX_ML, MERRA_PRISM and NARR_PRISM.

Phase 2 (optional, stochastic)
    One of ``diffusion_unet``, ``flow_matching_unet``, ``diffusion_transformer``
    or ``flow_matching_transformer``, predicting a **residual** in the Phase-1
    normalized target space. The four options are mutually exclusive
    alternatives; flow matching never requires the diffusion head to run first.

Timestamp contract
------------------
Predictors and targets in every batch refer to the same date/time. This wrapper
does not shift, offset, roll or re-index the time axis, does not build temporal
windows, and does not add any lead-time input. The only scalar "time" seen by
Phase 2 is the diffusion timestep or the flow interpolation coordinate, both of
which are generated internally by the refiner.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from granitewxc.refinement.base import ChunkNoiseSource, ResidualRefiner, build_refiner
from granitewxc.refinement.config import (
    ConfigValidationError,
    PerformanceConfig,
    RefinementConfig,
    resolve_performance_config,
    resolve_refinement_config,
)
from granitewxc.refinement.target_space import NormalizedTargetSpace

__all__ = ["TwoPhaseOutput", "TwoPhaseDownscalingModel", "build_two_phase_model"]


@dataclass
class TwoPhaseOutput:
    """Everything the training loop, the evaluator and the writer may need."""

    deterministic: torch.Tensor
    """Phase-1 prediction in physical units, ``[B, C, H, W]``."""

    deterministic_normalized: torch.Tensor
    """Phase-1 prediction in normalized target space, ``[B, C, H, W]``."""

    refined: torch.Tensor | None = None
    """Refined prediction in physical units (ensemble mean when ``n > 1``)."""

    refined_normalized: torch.Tensor | None = None

    residual: torch.Tensor | None = None
    """Predicted residual in normalized target space (ensemble mean when ``n > 1``)."""

    members: torch.Tensor | None = None
    """Per-member refined physical fields, ``[B, N, C, H, W]`` in draw order."""

    member_residuals: torch.Tensor | None = None
    """Per-member normalized residuals, ``[B, N, C, H, W]`` in draw order."""

    ensemble_mean: torch.Tensor | None = None
    ensemble_spread: torch.Tensor | None = None
    """Unbiased per-cell standard deviation across members (``None`` if ``N < 2``)."""

    residual_target: torch.Tensor | None = None
    """Training-only normalized residual target."""

    valid_mask: torch.Tensor | None = None
    losses: dict[str, torch.Tensor] = field(default_factory=dict)
    diagnostics: dict[str, torch.Tensor] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def _crop_rows(output_crop: object | None, batch_size: int) -> list[tuple[int, int, int, int]]:
    if output_crop is None:
        return []
    if torch.is_tensor(output_crop):
        rows = output_crop.detach().cpu().reshape(-1, 4).tolist()
    elif isinstance(output_crop, (list, tuple)):
        if len(output_crop) == 4 and not isinstance(output_crop[0], (list, tuple, torch.Tensor)):
            rows = [output_crop]
        else:
            rows = [
                row.detach().cpu().reshape(-1).tolist() if torch.is_tensor(row) else row
                for row in output_crop
            ]
    else:
        raise ValueError(f"Unsupported __output_crop type {type(output_crop).__name__}.")
    rows = [tuple(int(value) for value in row) for row in rows]
    if len(rows) == 1 and batch_size > 1:
        rows *= batch_size
    if len(rows) != batch_size or any(len(row) != 4 for row in rows):
        raise ValueError(
            "__output_crop must provide (top,left,height,width) once or per batch item."
        )
    return rows


def _align_to(
    tensor: torch.Tensor,
    size: tuple[int, int],
    output_crop: object | None = None,
) -> torch.Tensor:
    """Resample a conditioning field onto the target grid when shapes differ.

    PRISM predictors are already co-registered to the target grid but padded to
    a multiple of the backbone mask-unit size, so this is normally a pure crop.
    A genuine resolution mismatch (CORDEX-style coarse predictors) falls back to
    bilinear interpolation.
    """
    if tuple(tensor.shape[-2:]) == tuple(size):
        return tensor
    h, w = tensor.shape[-2], tensor.shape[-1]
    rows = _crop_rows(output_crop, int(tensor.shape[0]))
    if rows:
        parts = []
        usable = True
        for sample, (top, left, height, width) in enumerate(rows):
            if (
                height != size[0]
                or width != size[1]
                or top < 0
                or left < 0
                or top + height > h
                or left + width > w
            ):
                usable = False
                break
            parts.append(
                tensor[sample : sample + 1, ..., top : top + height, left : left + width]
            )
        if usable:
            return torch.cat(parts, dim=0)
    if h >= size[0] and w >= size[1]:
        # Symmetric halo/context without explicit metadata is centered. NARR
        # always supplies __output_crop; this is the safe fallback for legacy
        # datasets and avoids the historical top-left spatial shift.
        top = (h - size[0]) // 2
        left = (w - size[1]) // 2
        return tensor[..., top : top + size[0], left : left + size[1]]
    return F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)


class TwoPhaseDownscalingModel(nn.Module):
    """Deterministic Prithvi-UNet plus an optional stochastic residual refiner.

    Args:
        phase1: the deterministic ``ClimateDownscaleFinetuneUNETModel``.
        refinement: resolved :class:`RefinementConfig`.
        performance: resolved :class:`PerformanceConfig`.

    Module names are chosen so that Phase-1 state-dict keys are simply prefixed
    with ``phase1.`` and Phase-2 keys with ``refiner.``. See
    :mod:`granitewxc.refinement.checkpoint` for the migration helpers that load
    a legacy (unprefixed) deterministic checkpoint into this wrapper.
    """

    def __init__(
        self,
        phase1: nn.Module,
        refinement: RefinementConfig | None = None,
        performance: PerformanceConfig | None = None,
    ) -> None:
        super().__init__()
        self.phase1 = phase1
        self.refinement_config = refinement or RefinementConfig()
        self.performance_config = performance or PerformanceConfig()
        self.target_space = NormalizedTargetSpace(phase1)

        self._requested_features = self._resolve_requested_features()
        #: Built lazily: the conditioning width depends on the dataset (number of
        #: predictor channels, static fields, feature maps), so the refiner net is
        #: constructed the first time a batch is seen. Call
        #: :meth:`initialize_refiner` (directly or via ``initialize_from_batch``)
        #: before creating an optimizer.
        self.refiner: ResidualRefiner | None = None
        self._refiner_built = not self.refinement_config.is_active

        self._apply_phase1_freeze()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _resolve_requested_features(self) -> tuple[str, ...]:
        cond = self.refinement_config.conditioning
        if not self.refinement_config.is_active:
            return ()
        names: list[str] = []
        if cond.prithvi_features:
            names.append("prithvi")
        if cond.unet_features:
            names.append("unet")
        return tuple(names)

    def _apply_phase1_freeze(self) -> None:
        """Freeze Phase 1 when Phase 2 is trained on top of it."""
        self.phase1_partially_trainable: tuple[str, ...] = ()
        if not self.refinement_config.is_active:
            self.phase1_frozen = False
            self.phase1_inference_only = False
            return
        if self.refinement_config.joint_finetuning:
            for param in self.phase1.parameters():
                param.requires_grad_(True)
            self.phase1_frozen = False
            self.phase1_inference_only = False
            return

        patterns = self.refinement_config.trainable_phase1_patterns
        matched: list[str] = []
        for name, param in self.phase1.named_parameters():
            selected = any(fnmatch(name, pattern) for pattern in patterns)
            param.requires_grad_(selected)
            if selected:
                matched.append(name)
        if patterns and not matched:
            raise ConfigValidationError(
                "No Phase-1 parameter matched refinement.trainable_phase1_patterns "
                f"{list(patterns)}."
            )
        self.phase1_partially_trainable = tuple(matched)
        self.phase1_frozen = not bool(matched)
        self.phase1_inference_only = self.phase1_frozen
        # Frozen and selected-component modes keep Phase 1 in eval mode. Selected
        # parameters still receive gradients, while running statistics/dropout do
        # not drift in all the other frozen components.
        if self.refinement_config.freeze_phase1:
            for param in self.phase1.parameters():
                if not matched:
                    param.requires_grad_(False)
            self.phase1.eval()

    def train(self, mode: bool = True):  # noqa: D102 - torch API
        super().train(mode)
        if self.phase1_frozen or self.phase1_partially_trainable:
            # A frozen Phase 1 must stay in eval() so dropout / normalization
            # layers cannot perturb the deterministic conditioning.
            self.phase1.eval()
        return self

    def initialize_refiner(self, cond_channels: int) -> None:
        """Construct the Phase-2 network for a known conditioning width.

        Idempotent for a matching width; raises if called again with a different
        width after parameters have been created, because that would silently
        discard trained weights.
        """
        if not self.refinement_config.is_active:
            return
        if self.refiner is not None:
            if self.refiner.cond_channels == int(cond_channels):
                return
            raise RuntimeError(
                f"Refiner was built for {self.refiner.cond_channels} conditioning "
                f"channels but the batch provides {cond_channels}. The dataset or "
                "the conditioning configuration changed after construction."
            )
        self.refiner = build_refiner(
            self.refinement_config,
            residual_channels=self.target_space.num_channels,
            cond_channels=int(cond_channels),
        )
        try:
            reference = next(self.phase1.parameters())
        except StopIteration:  # pragma: no cover - Phase 1 always has parameters
            reference = None
        if reference is not None:
            self.refiner.to(device=reference.device)
        self._refiner_built = True

    def initialize_from_batch(self, batch: Mapping[str, torch.Tensor]) -> "TwoPhaseDownscalingModel":
        """Run one Phase-1 pass to discover the conditioning width and build Phase 2."""
        if not self.refinement_config.is_active:
            return self
        with torch.no_grad():
            _, normalized, features = self.run_phase1(batch)
            cond = self.build_conditioning(batch, normalized, features)
        self.initialize_refiner(cond.shape[1])
        return self

    # ------------------------------------------------------------------
    # Phase 1
    # ------------------------------------------------------------------
    def run_phase1(
        self, batch: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Run the deterministic model.

        Returns ``(physical, normalized, features)``. When Phase 1 is frozen the
        pass runs under :func:`torch.no_grad` so no autograd graph is retained.
        """
        setter = getattr(self.phase1, "set_feature_capture", None)
        if callable(setter):
            setter(self._requested_features)

        work_batch = dict(batch)
        if self.phase1_inference_only:
            with torch.no_grad():
                physical, normalized = self.phase1(work_batch, return_pre_inverse=True)
            physical = physical.detach()
            normalized = normalized.detach()
        else:
            physical, normalized = self.phase1(work_batch, return_pre_inverse=True)

        # The Phase-1 pre-inverse tensor is not always the inverse image of its
        # final physical prediction. In particular, the precipitation hurdle
        # returns an ungated positive-amount latent but a wet/dry-gated physical
        # field. Canonicalize the baseline in target data space so a zero
        # residual reconstructs Phase 1 exactly for every output head.
        offset = batch.get("__output_scaler_offset", batch.get("__scaler_offset"))
        normalized = self.target_space.encode(physical, scaler_offset=offset)
        if self.phase1_inference_only:
            normalized = normalized.detach()

        features: dict[str, torch.Tensor] = {}
        getter = getattr(self.phase1, "get_last_phase1_features", None)
        if self._requested_features and callable(getter):
            features = {
                name: (tensor.detach() if self.phase1_inference_only else tensor)
                for name, tensor in getter().items()
            }
        clearer = getattr(self.phase1, "clear_last_phase1_features", None)
        if callable(clearer):
            clearer()
        return physical, normalized, features

    def _normalized_predictors(
        self, batch: Mapping[str, torch.Tensor], dtype: torch.dtype
    ) -> torch.Tensor:
        """Reuse the exact Phase-1 predictor standardization contract."""
        x = batch["x"].to(dtype)
        resolver = getattr(self.phase1, "_resolve_input_scalers", None)
        timestamps = int(getattr(self.phase1, "n_input_timestamps", 1))
        if not callable(resolver) or x.shape[1] % timestamps:
            return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        batch_size, _, height, width = x.shape
        legacy_offset = batch.get("__scaler_offset")
        input_offset = batch.get("__input_scaler_offset", legacy_offset)
        mu, sigma = resolver(x, scaler_offset=input_offset)
        epsilon = getattr(self.phase1, "input_scalers_epsilon", 1.0e-6)
        x_time = x.view(batch_size, timestamps, -1, height, width)
        scaled = (x_time - mu.unsqueeze(1)) / (sigma.unsqueeze(1) + epsilon)
        scaled = scaled.reshape_as(x)
        return torch.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

    def _coordinate_conditioning(
        self,
        batch: Mapping[str, torch.Tensor],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Absolute output-grid coordinates, stable across overlapping tiles."""
        batch_size, _, height, width = reference.shape
        offset = batch.get("__output_scaler_offset", batch.get("__scaler_offset"))
        if offset is None:
            offsets = torch.zeros(batch_size, 2, device=reference.device, dtype=torch.long)
        else:
            offsets = torch.as_tensor(offset, device=reference.device, dtype=torch.long)
            if offsets.ndim == 1:
                offsets = offsets.unsqueeze(0).expand(batch_size, -1)
        scaler_shape = tuple(getattr(self.phase1, "output_scalers_sigma").shape[-2:])
        full_height = scaler_shape[0] if scaler_shape[0] > 1 else height
        full_width = scaler_shape[1] if scaler_shape[1] > 1 else width
        y_local = torch.arange(height, device=reference.device, dtype=reference.dtype)
        x_local = torch.arange(width, device=reference.device, dtype=reference.dtype)
        y = offsets[:, 0, None, None].to(reference.dtype) + y_local[None, :, None]
        x = offsets[:, 1, None, None].to(reference.dtype) + x_local[None, None, :]
        y = y.expand(batch_size, height, width) / max(full_height - 1, 1)
        x = x.expand(batch_size, height, width) / max(full_width - 1, 1)
        return torch.stack((2.0 * y - 1.0, 2.0 * x - 1.0), dim=1)

    # ------------------------------------------------------------------
    # Conditioning
    # ------------------------------------------------------------------
    def build_conditioning(
        self,
        batch: Mapping[str, torch.Tensor],
        deterministic_normalized: torch.Tensor,
        features: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Concatenate the configured spatial conditioning fields.

        Every component is resampled/cropped to the *target* grid so the refiner
        always sees ``[B, C_cond, H_target, W_target]``. Only fields that the
        configuration actually enables are computed.
        """
        cond_cfg = self.refinement_config.conditioning
        size = tuple(deterministic_normalized.shape[-2:])
        parts: list[torch.Tensor] = []
        dtype = deterministic_normalized.dtype
        output_crop = batch.get("__output_crop")

        if cond_cfg.deterministic_output:
            parts.append(deterministic_normalized)

        if cond_cfg.input_predictors and "x" in batch:
            x = self._normalized_predictors(batch, dtype)
            parts.append(_align_to(x, size, output_crop))

        if cond_cfg.static_fields:
            for key in ("static_x", "static_y"):
                if key in batch and batch[key] is not None:
                    static = torch.nan_to_num(batch[key].to(dtype), nan=0.0)
                    parts.append(_align_to(static, size, output_crop))

        if cond_cfg.masks:
            # Predictor validity only. The target mask is never used as
            # conditioning: that would leak ground-truth information.
            if "x" in batch:
                mask = torch.isfinite(batch["x"]).all(dim=1, keepdim=True).to(dtype)
            else:
                mask = torch.ones(
                    (deterministic_normalized.shape[0], 1, *size),
                    dtype=dtype,
                    device=deterministic_normalized.device,
                )
            parts.append(_align_to(mask, size, output_crop))

        if cond_cfg.coordinates:
            parts.append(self._coordinate_conditioning(batch, deterministic_normalized))

        features = features or {}
        if cond_cfg.prithvi_features:
            feat = features.get("prithvi")
            if feat is None:
                raise RuntimeError(
                    "refinement.conditioning.prithvi_features is enabled but Phase 1 "
                    "did not return a 'prithvi' feature map."
                )
            parts.append(_align_to(feat.to(dtype), size, output_crop))
        if cond_cfg.unet_features:
            feat = features.get("unet")
            if feat is None:
                raise RuntimeError(
                    "refinement.conditioning.unet_features is enabled but Phase 1 "
                    "did not return a 'unet' feature map."
                )
            parts.append(_align_to(feat.to(dtype), size, output_crop))

        if not parts:  # pragma: no cover - guarded by ConditioningConfig
            raise ConfigValidationError("No conditioning inputs were enabled.")
        return torch.cat(parts, dim=1)

    # ------------------------------------------------------------------
    # Forward / training / inference
    # ------------------------------------------------------------------
    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Deterministic forward pass (drop-in replacement for Phase 1).

        Kept signature-compatible with the existing trainer so deterministic
        training and evaluation are unaffected by the wrapper.
        """
        return self.phase1(batch)

    def _prepare(self, batch: Mapping[str, torch.Tensor]):
        physical, normalized, features = self.run_phase1(batch)
        if not self.refinement_config.is_active:
            return physical, normalized, None
        cond = self.build_conditioning(batch, normalized, features)
        self.initialize_refiner(cond.shape[1])
        return physical, normalized, cond.to(next(self.refiner.parameters()).dtype)

    def training_step(
        self,
        batch: Mapping[str, torch.Tensor],
        generator: torch.Generator | None = None,
    ) -> TwoPhaseOutput:
        """One Phase-2 training objective evaluation.

        The Phase-1 conditioning can optionally be supplied through
        ``batch['__phase1_normalized']`` (precomputed cache); when present, the
        Phase-1 forward pass is skipped entirely.
        """
        if not self.refinement_config.is_active:
            raise RuntimeError(
                "training_step() requires an active refinement configuration; "
                "use the deterministic trainer for Phase-1-only runs."
            )
        if "y" not in batch:
            raise KeyError("training_step() requires the ground truth under batch['y'].")

        cached = batch.get("__phase1_normalized")
        if cached is not None:
            if not self.phase1_frozen:
                raise RuntimeError(
                    "Cached Phase-1 conditioning cannot be used while Phase 1 is trainable."
                )
            normalized = cached
            physical = self.target_space.decode(
                normalized, scaler_offset=batch.get("__output_scaler_offset")
            )
            features = {
                name: batch[f"__phase1_feature_{name}"]
                for name in self._requested_features
                if f"__phase1_feature_{name}" in batch
            }
            cond = self.build_conditioning(batch, normalized, features)
            self.initialize_refiner(cond.shape[1])
            cond = cond.to(next(self.refiner.parameters()).dtype)
        else:
            physical, normalized, cond = self._prepare(batch)

        offset = batch.get("__output_scaler_offset", batch.get("__scaler_offset"))
        target, valid = self.target_space.residual_target(
            batch["y"], normalized, scaler_offset=offset
        )

        training_target = self.refiner.normalize_residual(target.to(cond.dtype))
        # Standardization maps the explicit zero fill at invalid cells to
        # ``-mean/std``. Reapply the validity mask before forward noising/path
        # construction so oceans/missing pixels cannot leak into neighboring
        # valid predictions through convolutions or shared Transformer patches.
        training_target = torch.where(
            valid.to(device=training_target.device, dtype=torch.bool),
            training_target,
            torch.zeros_like(training_target),
        )
        losses = self.refiner.training_loss(
            training_target, cond, valid.to(cond.device), generator=generator
        )
        scalar_losses = {
            key: value
            for key, value in losses.items()
            if torch.is_tensor(value) and value.ndim == 0
        }
        diagnostics = {
            key: value.detach()
            for key, value in losses.items()
            if torch.is_tensor(value) and value.ndim > 0
        }
        return TwoPhaseOutput(
            deterministic=physical,
            deterministic_normalized=normalized,
            residual_target=target,
            valid_mask=valid,
            losses=scalar_losses,
            diagnostics=diagnostics,
        )

    @torch.no_grad()
    def predict(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        ensemble_size: int | None = None,
        generator: torch.Generator | None = None,
        seed: int | None = None,
        return_members: bool = True,
        chunk_size: int | None = None,
    ) -> TwoPhaseOutput:
        """Deterministic and (optionally) refined ensemble inference.

        Args:
            ensemble_size: number of stochastic members. ``0`` or ``None`` with a
                disabled refiner returns the deterministic result only.
            generator / seed: reproducible member draws. ``seed`` builds a fresh
                generator on the batch device; ``generator`` is used verbatim.
            chunk_size: how many members to evaluate per network call. Members
                are always returned in draw order regardless of chunking.
        """
        physical, normalized, cond = self._prepare(batch)
        offset = batch.get("__output_scaler_offset", batch.get("__scaler_offset"))
        target_mask = batch.get("__target_valid_mask")
        if target_mask is None and "y" in batch:
            target_mask = torch.isfinite(batch["y"])

        out = TwoPhaseOutput(
            deterministic=physical,
            deterministic_normalized=normalized,
        )
        if self.refiner is None:
            return out

        n = int(ensemble_size if ensemble_size is not None else self.refinement_config.ensemble_size)
        if n < 1:
            return out

        if generator is None and seed is not None:
            generator = torch.Generator(device=cond.device)
            generator.manual_seed(int(seed))

        perf = self.performance_config.ensemble
        if chunk_size is None:
            chunk_size = perf.chunk_size if perf.chunk_size is not None else (n if perf.batch_members else 1)
        chunk_size = max(1, min(int(chunk_size), n))

        # Each ensemble member gets its own generator, seeded from the caller's
        # generator (or from fresh entropy). Member m therefore always sees the
        # same noise sequence regardless of the ensemble chunking, which makes
        # serial and batched generation bitwise identical.
        gen_device = generator.device if generator is not None else cond.device
        if generator is not None:
            member_seeds = torch.randint(
                0, 2**62, (n,), generator=generator, device=gen_device, dtype=torch.int64
            ).tolist()
        else:
            member_seeds = torch.randint(0, 2**62, (n,), dtype=torch.int64).tolist()
        member_generators = []
        for member_seed in member_seeds:
            g = torch.Generator(device=gen_device)
            g.manual_seed(int(member_seed))
            member_generators.append(g)

        batch_size = cond.shape[0]
        residuals: list[torch.Tensor] = []
        drawn = 0
        while drawn < n:
            members = min(chunk_size, n - drawn)
            chunk_generators = member_generators[drawn : drawn + members]
            source = ChunkNoiseSource(chunk_generators, batch_size)
            if members == 1:
                with self.refiner.use_noise_source(source):
                    res = self.refiner.sample_target_space(
                        cond, valid_mask=target_mask
                    )
                res = res * float(self.refinement_config.correction_scale)
                residuals.append(res.unsqueeze(1))
            else:
                # Replicate the (identical) conditioning across members so a
                # single batched network call produces `members` independent
                # draws. Member order is preserved by the reshape below.
                cond_rep = cond.repeat_interleave(members, dim=0)
                mask_rep = (
                    target_mask.repeat_interleave(members, dim=0)
                    if target_mask is not None
                    else None
                )
                with self.refiner.use_noise_source(source):
                    res = self.refiner.sample_target_space(
                        cond_rep, valid_mask=mask_rep
                    )
                res = res * float(self.refinement_config.correction_scale)
                res = res.reshape(batch_size, members, *res.shape[1:])
                residuals.append(res)
            drawn += members

        member_residuals = torch.cat(residuals, dim=1)  # [B, N, C, H, W]
        member_fields = []
        for idx in range(n):
            _, refined_physical = self.target_space.reconstruct(
                normalized,
                member_residuals[:, idx].to(normalized.dtype),
                scaler_offset=offset,
                target_mask=target_mask,
            )
            member_fields.append(refined_physical.unsqueeze(1))
        members_physical = torch.cat(member_fields, dim=1)

        # Ensemble statistics are accumulated in float32 regardless of the
        # compute dtype so mixed precision cannot bias the mean or the spread.
        # NaN cells are masked-out/invalid points: they are excluded from the
        # statistics rather than poisoning every member's contribution.
        stack32 = members_physical.float()
        valid = torch.isfinite(stack32)
        counts = valid.sum(dim=1)
        filled = torch.where(valid, stack32, torch.zeros_like(stack32))
        mean32 = filled.sum(dim=1) / counts.clamp(min=1)
        mean32 = torch.where(counts > 0, mean32, torch.full_like(mean32, float("nan")))
        ensemble_mean = mean32.to(members_physical.dtype)
        spread = None
        if n > 1:
            deviations = torch.where(valid, stack32 - mean32.unsqueeze(1), torch.zeros_like(stack32))
            var = deviations.pow(2).sum(dim=1) / (counts - 1).clamp(min=1)
            var = torch.where(counts > 1, var, torch.full_like(var, float("nan")))
            spread = var.sqrt().to(members_physical.dtype)

        mean_residual = member_residuals.float().mean(dim=1).to(normalized.dtype)
        refined_norm, refined_physical = self.target_space.reconstruct(
            normalized, mean_residual, scaler_offset=offset, target_mask=target_mask
        )

        out.members = members_physical if return_members else None
        out.member_residuals = member_residuals if return_members else None
        out.ensemble_mean = ensemble_mean
        out.ensemble_spread = spread
        out.residual = mean_residual
        out.refined_normalized = refined_norm
        out.refined = refined_physical if n > 1 else members_physical[:, 0]
        if n > 1:
            out.refined = ensemble_mean
        return out

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    @torch.no_grad()
    def fit_residual_normalizer(
        self,
        loader,
        *,
        device: torch.device | str | None = None,
        max_batches: int | None = None,
    ) -> dict[str, Any]:
        """Fit per-variable residual statistics on the supplied training loader.

        This operation is deliberately unavailable when Phase 1 is trainable:
        changing the deterministic baseline would immediately stale the fitted
        residual distribution.
        """
        if self.refiner is not None and not self.refiner.residual_normalization_enabled:
            return self.refiner.residual_normalization_metadata()
        if not self.phase1_inference_only:
            raise RuntimeError(
                "Residual normalization can only be fitted with a frozen Phase 1."
            )
        if device is None:
            try:
                device = next(self.phase1.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        device = torch.device(device)
        sums = torch.zeros(self.target_space.num_channels, dtype=torch.float64, device=device)
        squares = torch.zeros_like(sums)
        counts = torch.zeros(self.target_space.num_channels, dtype=torch.int64, device=device)
        seen = 0
        for batch in loader:
            if max_batches is not None and max_batches > 0 and seen >= max_batches:
                break
            moved = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            _, baseline, features = self.run_phase1(moved)
            cond = self.build_conditioning(moved, baseline, features)
            self.initialize_refiner(cond.shape[1])
            offset = moved.get("__output_scaler_offset", moved.get("__scaler_offset"))
            residual, valid = self.target_space.residual_target(
                moved["y"], baseline, scaler_offset=offset
            )
            values = residual.double()
            valid_d = valid.double()
            axes = (0, 2, 3)
            sums += (values * valid_d).sum(dim=axes)
            squares += (values.square() * valid_d).sum(dim=axes)
            counts += valid.sum(dim=axes)
            seen += 1
        if self.refiner is None or seen == 0:
            raise RuntimeError("Cannot fit residual normalization: training loader was empty.")
        if bool((counts <= 0).any()):
            raise RuntimeError(
                "Cannot fit residual normalization: at least one output channel has no valid training cells."
            )
        means = sums / counts.double()
        variances = (squares / counts.double() - means.square()).clamp(min=0.0)
        std = variances.sqrt().clamp(min=self.refiner.residual_normalization_epsilon)
        self.refiner.set_residual_normalization(means.float(), std.float(), counts)
        return self.refiner.residual_normalization_metadata()

    def describe(self) -> dict[str, Any]:
        return {
            "refinement": self.refinement_config.to_dict(),
            "performance": self.performance_config.to_dict(),
            "phase1_frozen": self.phase1_frozen,
            "phase1_partially_trainable": list(self.phase1_partially_trainable),
            "phase1_features": list(self._requested_features),
            "phase1_parameters": sum(p.numel() for p in self.phase1.parameters()),
            "refiner_parameters": (
                sum(p.numel() for p in self.refiner.parameters()) if self.refiner is not None else 0
            ),
            "frozen_parameters": sum(
                p.numel() for p in self.parameters() if not p.requires_grad
            ),
            "trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
            "new_refinement_keys": (
                sorted(f"refiner.{key}" for key in self.refiner.state_dict())
                if self.refiner is not None
                else []
            ),
            "residual_normalization": (
                self.refiner.residual_normalization_metadata()
                if self.refiner is not None
                else None
            ),
        }


def build_two_phase_model(
    phase1: nn.Module,
    config: Any,
) -> TwoPhaseDownscalingModel:
    """Build the wrapper from a raw experiment configuration."""
    refinement = resolve_refinement_config(config)
    performance = resolve_performance_config(config)
    if performance.precision.allow_tf32:
        warnings.warn(
            "performance.precision.allow_tf32 is enabled; validate numerical "
            "parity against the fp32 baseline before trusting the results.",
            RuntimeWarning,
            stacklevel=2,
        )
    return TwoPhaseDownscalingModel(phase1, refinement=refinement, performance=performance)
