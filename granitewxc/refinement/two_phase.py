"""Unified two-phase Prithvi-UNet model wrapper.

Phase 1 (deterministic)
    Prithvi-WxC encoder/backbone + deterministic UNet decoder. Unchanged for
    CORDEX_ML, MERRA_PRISM and NARR_PRISM.

Phase 2 (optional, stochastic)
    One of ``diffusion_unet``, ``flow_matching_unet``, ``diffusion_transformer``
    or ``flow_matching_transformer``, modeling a separately standardized
    **physical residual** ``ground_truth - frozen_phase1``. The four options are
    mutually exclusive alternatives; flow matching never requires the
    diffusion head to run first.

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
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Mapping, Sequence

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
from granitewxc.refinement.normalization import ResidualNormalizer
from granitewxc.refinement.randomness import NoiseSource, StableNoiseSource
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
    """Raw predicted residual in residual-normalized model space."""

    residual_physical: torch.Tensor | None = None
    """Effective physical correction, equal to ``refined - deterministic``."""

    members: torch.Tensor | None = None
    """Per-member refined physical fields, ``[B, N, C, H, W]`` in draw order."""

    members_unbounded: torch.Tensor | None = None
    """Physical ``base + correction`` members before the one final constraint."""

    member_residuals: torch.Tensor | None = None
    """Raw model-space residuals, ``[B, N, C, H, W]`` in draw order."""

    member_residuals_physical: torch.Tensor | None = None
    """Effective per-member physical corrections in draw order."""

    ensemble_mean: torch.Tensor | None = None
    ensemble_spread: torch.Tensor | None = None
    """Unbiased per-cell standard deviation across members (``None`` if ``N < 2``)."""

    unbounded_ensemble_mean: torch.Tensor | None = None
    """Mean of unbounded physical members, before precipitation constraints."""

    memberwise_clipped_mean: torch.Tensor | None = None
    """Diagnostic mean from the historical independent memberwise clamp."""

    memberwise_clipping_mean_shift: torch.Tensor | None = None
    """``memberwise_clipped_mean - unbounded_ensemble_mean`` in physical units."""

    residual_target: torch.Tensor | None = None
    """Training-only residual target in residual-normalized model space."""

    residual_target_physical: torch.Tensor | None = None
    """Training-only target ``ground_truth - deterministic`` in physical units."""

    valid_mask: torch.Tensor | None = None
    losses: dict[str, torch.Tensor] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def _normalize_output_crops(output_crop: object | None, batch_size: int):
    if output_crop is None:
        return []
    if torch.is_tensor(output_crop):
        rows = output_crop.detach().cpu()
        if rows.ndim == 1 and rows.numel() == 4:
            rows = rows.reshape(1, 4)
        if rows.ndim != 2 or rows.shape[1] != 4:
            raise ValueError(
                "__output_crop must have shape [4] or [B,4], got "
                f"{tuple(rows.shape)}."
            )
        rows = rows.tolist()
    elif isinstance(output_crop, (list, tuple)):
        if len(output_crop) == 4 and not isinstance(
            output_crop[0], (list, tuple, torch.Tensor)
        ):
            rows = [output_crop]
        else:
            rows = [
                row.detach().cpu().reshape(-1).tolist() if torch.is_tensor(row) else row
                for row in output_crop
            ]
    else:
        raise ValueError(f"Unsupported __output_crop type {type(output_crop).__name__}.")
    crops = [tuple(int(v) for v in row) for row in rows]
    if any(len(row) != 4 for row in crops):
        raise ValueError("Every __output_crop row must be (top,left,height,width).")
    if len(crops) == 1 and batch_size > 1:
        crops *= batch_size
    if len(crops) != batch_size:
        raise ValueError(
            f"__output_crop count {len(crops)} does not match batch size {batch_size}."
        )
    return crops


def _slice_prediction_batch(batch: Mapping[str, Any], index: int, batch_size: int):
    """Slice sample-leading tensors while preserving shared crop/offset metadata."""
    selected = {}
    for key, value in batch.items():
        if key == "__output_crop":
            crops = _normalize_output_crops(value, batch_size)
            selected[key] = [crops[index]] if crops else None
        elif key.endswith("scaler_offset") and (
            (torch.is_tensor(value) and value.ndim == 1)
            or (isinstance(value, (tuple, list)) and len(value) == 2
                and not isinstance(value[0], (tuple, list, torch.Tensor)))
        ):
            selected[key] = value
        elif torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch_size:
            selected[key] = value[index:index + 1]
        elif isinstance(value, (list, tuple)) and len(value) == batch_size:
            selected[key] = value[index:index + 1]
        elif isinstance(value, Mapping):
            selected[key] = _slice_prediction_batch(value, index, batch_size)
        else:
            selected[key] = value
    return selected


def _align_to(
    tensor: torch.Tensor,
    size: tuple[int, int],
    *,
    output_crop: object | None = None,
    crop_reference_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Resample a conditioning field onto the target grid when shapes differ.

    PRISM predictors are already co-registered to the target grid but padded to
    a multiple of the backbone mask-unit size, so this is normally a pure crop.
    A genuine resolution mismatch (CORDEX-style coarse predictors) falls back to
    bilinear interpolation.
    """
    crops = _normalize_output_crops(output_crop, int(tensor.shape[0]))
    if crops:
        reference_h, reference_w = crop_reference_size or tuple(tensor.shape[-2:])
        source_h, source_w = tensor.shape[-2:]
        pieces = []
        for sample_idx, (top, left, height, width) in enumerate(crops):
            if min(top, left) < 0 or min(height, width) <= 0:
                raise ValueError(f"Invalid __output_crop[{sample_idx}]={crops[sample_idx]}.")
            if top + height > reference_h or left + width > reference_w:
                raise ValueError(
                    f"__output_crop[{sample_idx}]={crops[sample_idx]} lies outside "
                    f"the reference grid {(reference_h, reference_w)}."
                )
            y0 = int(round(top * source_h / reference_h))
            y1 = int(round((top + height) * source_h / reference_h))
            x0 = int(round(left * source_w / reference_w))
            x1 = int(round((left + width) * source_w / reference_w))
            y0, x0 = max(0, y0), max(0, x0)
            y1, x1 = min(source_h, max(y0 + 1, y1)), min(source_w, max(x0 + 1, x1))
            pieces.append(tensor[sample_idx : sample_idx + 1, ..., y0:y1, x0:x1])
        shapes = {tuple(piece.shape[-2:]) for piece in pieces}
        if len(shapes) != 1:
            pieces = [
                F.interpolate(piece, size=size, mode="bilinear", align_corners=False)
                for piece in pieces
            ]
        tensor = torch.cat(pieces, dim=0)
    if tuple(tensor.shape[-2:]) == tuple(size):
        return tensor
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
        self.residual_normalizer: ResidualNormalizer | None = None
        if self.refinement_config.is_active:
            self.residual_normalizer = ResidualNormalizer(
                self.target_space.num_channels,
                self.refinement_config.residual_normalization,
                nonnegative_mask=getattr(phase1, "predictand_nonneg_enabled_mask", None),
            )

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
        self.phase1_frozen = bool(
            self.refinement_config.is_active and self.refinement_config.freeze_phase1
        )
        if self.phase1_frozen:
            for param in self.phase1.parameters():
                param.requires_grad_(False)
            self.phase1.eval()

    def train(self, mode: bool = True):  # noqa: D102 - torch API
        super().train(mode)
        if self.phase1_frozen:
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
        # Modules attached after ``model.eval()`` otherwise remain in training
        # mode. Mirror the parent immediately so lazy inference cannot leave
        # dropout or stochastic depth active.
        self.refiner.train(self.training)
        self._refiner_built = True

    def initialize_from_batch(self, batch: Mapping[str, torch.Tensor]) -> "TwoPhaseDownscalingModel":
        """Run one Phase-1 pass to discover the conditioning width and build Phase 2."""
        if not self.refinement_config.is_active:
            return self
        with torch.no_grad():
            _, normalized, features = self._phase1_from_batch(batch)
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
        if self.phase1_frozen:
            self.phase1.eval()
            with torch.no_grad():
                physical, normalized = self.phase1(work_batch, return_pre_inverse=True)
            physical = physical.detach()
            normalized = normalized.detach()
        else:
            physical, normalized = self.phase1(work_batch, return_pre_inverse=True)

        features: dict[str, torch.Tensor] = {}
        getter = getattr(self.phase1, "get_last_phase1_features", None)
        if self._requested_features and callable(getter):
            features = {
                name: (tensor.detach() if self.phase1_frozen else tensor)
                for name, tensor in getter().items()
            }
        clearer = getattr(self.phase1, "clear_last_phase1_features", None)
        if callable(clearer):
            clearer()
        return physical, normalized, features

    def _phase1_from_batch(self, batch: Mapping[str, torch.Tensor]):
        """Reuse exact cached baseline fields for training and inference alike."""
        normalized = batch.get("__phase1_normalized")
        if normalized is None:
            return self.run_phase1(batch)
        if not self.phase1_frozen:
            raise RuntimeError("Cached Phase-1 conditioning cannot be used while Phase 1 is trainable.")
        self.phase1.eval()
        physical = batch.get("__phase1_physical")
        if physical is None:
            if bool(getattr(self.phase1, "precip_hurdle_enabled", False)):
                raise RuntimeError(
                    "Hurdle precipitation caches must include __phase1_physical: "
                    "normalized amount alone omits the wet-occurrence decision. "
                    "Regenerate this cache from the exact frozen Phase-1 output."
                )
            physical = self.target_space.decode(
                normalized,
                scaler_offset=batch.get("__output_scaler_offset", batch.get("__scaler_offset")),
            )
        if physical.shape != normalized.shape:
            raise ValueError("Cached physical and normalized Phase-1 fields must have identical shapes.")
        features = {
            name: batch[f"__phase1_feature_{name}"].detach()
            for name in self._requested_features
            if f"__phase1_feature_{name}" in batch
        }
        return physical.detach(), normalized.detach(), features

    # ------------------------------------------------------------------
    # Conditioning
    # ------------------------------------------------------------------
    def _normalize_predictor_conditioning(self, values: torch.Tensor, batch) -> torch.Tensor:
        """Use frozen Phase-1 input statistics before target-grid alignment.

        The input channel layout may contain repeated timestamps. Phase 1
        applies the same per-variable scalers to each timestamp independently.
        This opt-in changes learned conditioning and requires retraining.
        """
        resolver = getattr(self.phase1, "_resolve_input_scalers", None)
        if not callable(resolver):
            raise RuntimeError(
                "conditioning.normalize_predictors requires Phase-1 _resolve_input_scalers; "
                "statistics cannot be fitted or substituted from this batch."
            )
        values = self.target_space._transform_values(values)
        mu, sigma = resolver(
            values,
            scaler_offset=batch.get("__input_scaler_offset", batch.get("__scaler_offset")),
        )
        mu, sigma = mu.to(values), sigma.to(values)
        epsilon = float(getattr(self.phase1, "input_scalers_epsilon", 0.0))
        denominator = sigma + epsilon
        if not bool(torch.isfinite(mu).all()) or not bool(torch.isfinite(denominator).all()) or bool((denominator <= 0).any()):
            raise RuntimeError("Frozen Phase-1 predictor statistics must be finite with positive scale plus epsilon.")
        timestamps = int(getattr(self.phase1, "n_input_timestamps", 1))
        if timestamps < 1 or values.shape[1] % timestamps:
            raise ValueError("Predictor channels must be divisible by Phase-1 n_input_timestamps.")
        batch_size, channels, height, width = values.shape
        separated = values.reshape(batch_size, timestamps, channels // timestamps, height, width)
        if mu.shape[1] != channels // timestamps or sigma.shape[1] != channels // timestamps:
            raise ValueError("Frozen Phase-1 predictor scaler channels do not match each input timestamp.")
        normalized = (separated - mu.unsqueeze(1)) / denominator.unsqueeze(1)
        return normalized.reshape_as(values)

    def _normalize_static_conditioning(self, values: torch.Tensor, key: str) -> torch.Tensor:
        prefix = "static_input" if key == "static_x" else "static_output"
        mu = getattr(self.phase1, f"{prefix}_scalers_mu", None)
        sigma = getattr(self.phase1, f"{prefix}_scalers_sigma", None)
        if not torch.is_tensor(mu) or not torch.is_tensor(sigma):
            raise RuntimeError(
                f"conditioning.normalize_predictors requires frozen Phase-1 {prefix} scalers."
            )
        values = self.target_space._transform_values(values)
        mu, sigma = mu.to(values), sigma.to(values)
        # This is the exact epsilon used for both static fields in Phase 1.
        denominator = sigma + float(getattr(self.phase1, "static_input_scalers_epsilon", 0.0))
        if not bool(torch.isfinite(mu).all()) or not bool(torch.isfinite(denominator).all()) or bool((denominator <= 0).any()):
            raise RuntimeError("Frozen Phase-1 static statistics must be finite with positive scale plus epsilon.")
        if mu.shape[1] != values.shape[1] or sigma.shape[1] != values.shape[1]:
            raise ValueError(f"Frozen Phase-1 {prefix} scaler channels do not match {key}.")
        normalized = (values - mu) / denominator
        if normalized.shape != values.shape:
            raise ValueError(f"Frozen Phase-1 {prefix} scaler layout does not match {key}.")
        return normalized

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
        output_crop = batch.get("__output_crop")
        reference_size = tuple(batch["x"].shape[-2:]) if "x" in batch else None
        parts: list[torch.Tensor] = []
        dtype = deterministic_normalized.dtype

        if cond_cfg.deterministic_output:
            parts.append(torch.nan_to_num(deterministic_normalized, nan=0.0, posinf=0.0, neginf=0.0))

        if cond_cfg.input_predictors and "x" in batch:
            x = batch["x"]
            if cond_cfg.normalize_predictors:
                x = self._normalize_predictor_conditioning(x, batch)
            x = x.to(dtype)
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            parts.append(
                _align_to(
                    x,
                    size,
                    output_crop=output_crop,
                    crop_reference_size=reference_size,
                )
            )

        if cond_cfg.static_fields:
            for key in ("static_x", "static_y"):
                if key in batch and batch[key] is not None:
                    static = batch[key]
                    if cond_cfg.normalize_predictors:
                        static = self._normalize_static_conditioning(static, key)
                    static = torch.nan_to_num(static.to(dtype), nan=0.0, posinf=0.0, neginf=0.0)
                    parts.append(
                        _align_to(
                            static,
                            size,
                            output_crop=output_crop,
                            crop_reference_size=reference_size,
                        )
                    )

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
            parts.append(
                _align_to(
                    mask,
                    size,
                    output_crop=output_crop,
                    crop_reference_size=reference_size,
                )
            )

        features = features or {}
        if cond_cfg.prithvi_features:
            feat = features.get("prithvi")
            if feat is None:
                raise RuntimeError(
                    "refinement.conditioning.prithvi_features is enabled but Phase 1 "
                    "did not return a 'prithvi' feature map."
                )
            parts.append(
                _align_to(
                    torch.nan_to_num(feat.to(dtype), nan=0.0, posinf=0.0, neginf=0.0),
                    size,
                    output_crop=output_crop,
                    crop_reference_size=reference_size,
                )
            )
        if cond_cfg.unet_features:
            feat = features.get("unet")
            if feat is None:
                raise RuntimeError(
                    "refinement.conditioning.unet_features is enabled but Phase 1 "
                    "did not return a 'unet' feature map."
                )
            parts.append(
                _align_to(
                    torch.nan_to_num(feat.to(dtype), nan=0.0, posinf=0.0, neginf=0.0),
                    size,
                    output_crop=output_crop,
                    crop_reference_size=reference_size,
                )
            )

        if not parts:  # pragma: no cover - guarded by ConditioningConfig
            raise ConfigValidationError("No conditioning inputs were enabled.")
        return torch.cat(parts, dim=1)

    # ------------------------------------------------------------------
    # Training-only residual statistics
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_explicit_target_mask(
        batch: Mapping[str, torch.Tensor],
        residual: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Combine dataset validity with finite residual validity.

        Some legacy deterministic datasets finite-fill missing targets so their
        Phase-1 loss remains backward compatible.  ``__target_valid_mask``
        preserves the pre-fill mask for Phase 2; without it, a missing target
        would be misread as a real dry zero and train the refiner toward
        ``0 - base``.  This mask is training-only and is never conditioning or
        prediction-domain input.
        """
        explicit = batch.get("__target_valid_mask")
        if explicit is None:
            return residual, valid
        if tuple(explicit.shape) != tuple(valid.shape):
            raise ValueError(
                "__target_valid_mask shape "
                f"{tuple(explicit.shape)} does not match target/residual shape "
                f"{tuple(valid.shape)}."
            )
        valid = valid & explicit.to(device=valid.device, dtype=torch.bool)
        residual = torch.where(valid, residual, torch.zeros_like(residual))
        return residual, valid

    @property
    def residual_statistics_ready(self) -> bool:
        return self.residual_normalizer is None or self.residual_normalizer.is_fitted

    @torch.no_grad()
    def reset_residual_statistics(self) -> None:
        if self.residual_normalizer is not None:
            self.residual_normalizer.reset()

    @torch.no_grad()
    def update_residual_statistics(self, batch: Mapping[str, torch.Tensor]) -> None:
        """Accumulate ``ground_truth - frozen_phase1`` from a training batch."""
        if self.residual_normalizer is None or self.residual_normalizer.is_identity:
            return
        if "y" not in batch:
            raise KeyError("Residual-statistics fitting requires batch['y'].")
        base, _, _ = self._phase1_from_batch(batch)
        residual, valid = self.target_space.physical_residual_target(batch["y"], base)
        residual, valid = self._apply_explicit_target_mask(
            batch, residual, valid
        )
        self.residual_normalizer.update(residual, valid)

    @torch.no_grad()
    def finalize_residual_statistics(self) -> None:
        if self.residual_normalizer is not None:
            self.residual_normalizer.finalize()

    def residual_normalization_metadata(self) -> dict[str, Any]:
        if self.residual_normalizer is None:
            return {"method": "disabled", "fitted": True}
        metadata = self.residual_normalizer.metadata()
        names = getattr(self.phase1, "output_var_names", None) or getattr(
            self.phase1, "predictands", None
        )
        if names is not None:
            metadata["channel_names"] = list(names)
        return metadata

    def _effective_physical_residual(self, residual_physical: torch.Tensor) -> torch.Tensor:
        """Apply the Transformer identity gate after inverse normalization."""
        if self.refiner is None:
            return residual_physical
        apply_gate = getattr(self.refiner, "apply_correction_gate", None)
        if callable(apply_gate):
            return apply_gate(residual_physical)
        return residual_physical

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
        physical, normalized, features = self._phase1_from_batch(batch)
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

        physical, normalized, cond = self._prepare(batch)

        target_physical, valid = self.target_space.physical_residual_target(
            batch["y"], physical
        )
        target_physical, valid = self._apply_explicit_target_mask(
            batch, target_physical, valid
        )
        if self.residual_normalizer is None:
            raise RuntimeError("Active refinement has no residual normalizer.")
        target = self.residual_normalizer.normalize(target_physical, valid)
        zero_residual = self.residual_normalizer.normalize(
            torch.zeros_like(target_physical), valid
        )

        losses = self.refiner.training_loss(
            target.to(cond.dtype),
            cond,
            valid.to(cond.device),
            generator=generator,
            zero_residual=zero_residual.to(cond.dtype),
        )
        return TwoPhaseOutput(
            deterministic=physical,
            deterministic_normalized=normalized,
            residual_target=target,
            residual_target_physical=target_physical,
            valid_mask=valid,
            losses=losses,
        )

    @torch.no_grad()
    def predict(
        self,
        batch: Mapping[str, Any],
        *,
        ensemble_size: int | None = None,
        generator: torch.Generator | None = None,
        seed: int | None = None,
        return_members: bool = True,
        chunk_size: int | None = None,
        trajectory_callback: Callable | None = None,
        sample_ids: Sequence[str | int] | None = None,
        sampling_mode: str | None = None,
        sample_trajectory_callback: Callable | None = None,
    ) -> TwoPhaseOutput:
        """Predict with an explicit legacy or sample-identified random stream.

        The default is the unchanged legacy seeded path. An explicit sample_ids
        argument selects stable mode unless sampling_mode='legacy' is specified.
        sampling_mode='stable' also resolves identifiers from batch metadata;
        metadata alone never changes legacy inference behavior. Stable
        mode requires seed (the experiment seed), rejects a stateful generator,
        and evaluates each sample/member separately, including Phase 1 and
        reconstruction. chunk_size is ignored in this mode. Thus random fields
        AND neural arithmetic are invariant to input batch size/order on the
        same deterministic software/device settings; cross-device/version
        bitwise equality is not promised. IDs must identify the actual sample,
        including its domain/crop when a timestamp alone is not unique.

        Stable diagnostics use sample_trajectory_callback(sample_id, stage,
        step, process_time, state, member_index), with state [1,1,C,H,W].
        The historical trajectory_callback API remains available in legacy
        mode. Identifiers are metadata only and never conditioning features.
        """
        mode = sampling_mode or ("stable" if sample_ids is not None else "legacy")
        if sample_ids is None:
            sample_ids = batch.get(
                "__sample_ids", batch.get("__sample_id", batch.get(
                    "__timestamps", batch.get("__sample_timestamp")
                ))
            )
        options = dict(
            ensemble_size=ensemble_size, generator=generator, seed=seed,
            return_members=return_members, chunk_size=chunk_size,
            trajectory_callback=trajectory_callback,
        )
        if mode == "legacy":
            if sample_trajectory_callback is not None:
                raise ValueError("sample_trajectory_callback requires stable sampling.")
            return self._predict_impl(batch, **options)
        if mode != "stable":
            raise ValueError("sampling_mode must be 'legacy' or 'stable'.")
        if generator is not None or seed is None:
            raise ValueError("Stable sampling requires an experiment seed and no stateful generator.")
        if sample_ids is None or isinstance(sample_ids, (str, bytes)):
            raise ValueError("Stable sampling requires a sequence of sample identifiers.")
        ids = list(sample_ids)
        anchor = next(
            (batch[key] for key in ("x", "__phase1_normalized", "__phase1_physical", "y")
             if key in batch), None,
        )
        if not torch.is_tensor(anchor) or anchor.ndim < 1 or len(ids) != anchor.shape[0] or not ids:
            raise ValueError("sample_ids must have one identifier per batch sample.")
        if len({str(value) for value in ids}) != len(ids):
            raise ValueError("sample_ids must be unique within a batch; include domain/crop identity.")
        if trajectory_callback is not None:
            raise ValueError("Use sample_trajectory_callback for identified stable diagnostics.")
        # Validate identifiers before any model computation or RNG use.
        for sample_id in ids:
            StableNoiseSource(seed, [sample_id], [0])
        outputs = []
        for sample_index, sample_id in enumerate(ids):
            single = _slice_prediction_batch(batch, sample_index, len(ids))
            callback = None
            if sample_trajectory_callback is not None:
                def callback(stage, step, process_time, state, member_index):
                    sample_trajectory_callback(
                        sample_id, stage, step, process_time, state, member_index
                    )
            outputs.append(self._predict_impl(
                single, ensemble_size=ensemble_size, return_members=return_members,
                chunk_size=1, trajectory_callback=callback,
                noise_source_factory=lambda member: StableNoiseSource(seed, [sample_id], [member]),
            ))
        result = {}
        for descriptor in fields(TwoPhaseOutput):
            values = [getattr(output, descriptor.name) for output in outputs]
            if all(torch.is_tensor(value) for value in values):
                result[descriptor.name] = torch.cat(values, dim=0)
            elif all(value is None for value in values):
                result[descriptor.name] = None
            elif descriptor.name == "losses" and all(not value for value in values):
                result[descriptor.name] = {}
            else:
                raise RuntimeError(f"Inconsistent stable inference field {descriptor.name}.")
        return TwoPhaseOutput(**result)

    @torch.no_grad()
    def _predict_impl(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        ensemble_size: int | None = None,
        generator: torch.Generator | None = None,
        seed: int | None = None,
        return_members: bool = True,
        chunk_size: int | None = None,
        trajectory_callback: Callable[[str, int, torch.Tensor, torch.Tensor, int], None] | None = None,
        noise_source_factory: Callable[[int], NoiseSource] | None = None,
    ) -> TwoPhaseOutput:
        """Deterministic and (optionally) refined ensemble inference.

        Args:
            ensemble_size: number of stochastic members. ``0`` or ``None`` with a
                disabled refiner returns the deterministic result only.
            generator / seed: reproducible member draws. ``seed`` builds a fresh
                generator on the batch device; ``generator`` is used verbatim.
            chunk_size: how many members to evaluate per network call. Members
                are always returned in draw order regardless of chunking.
            trajectory_callback: optional read-only sampler diagnostics. Receives
                ``(stage, step, process_time, state, member_start)`` with a
                detached state shaped ``[B,chunk_members,C,H,W]``. Stages and
                times retain the algorithm-specific sampler meaning; process
                time is not a physical timestamp.
        """
        physical, normalized, cond = self._prepare(batch)
        offset = batch.get("__output_scaler_offset", batch.get("__scaler_offset"))
        prediction_mask = batch.get("__prediction_mask")
        if prediction_mask is not None:
            prediction_mask = prediction_mask.to(device=physical.device, dtype=torch.bool)
            if prediction_mask.ndim == 3:
                prediction_mask = prediction_mask.unsqueeze(1)
            if prediction_mask.shape[1] == 1:
                prediction_mask = prediction_mask.expand(-1, physical.shape[1], -1, -1)
            prediction_mask = _align_to(
                prediction_mask.float(),
                tuple(physical.shape[-2:]),
                output_crop=batch.get("__output_crop"),
                crop_reference_size=(
                    tuple(batch["x"].shape[-2:]) if "x" in batch else None
                ),
            ).bool()
        elif "x" in batch:
            predictor_mask = torch.isfinite(batch["x"]).all(dim=1, keepdim=True).float()
            prediction_mask = _align_to(
                predictor_mask,
                tuple(physical.shape[-2:]),
                output_crop=batch.get("__output_crop"),
                crop_reference_size=tuple(batch["x"].shape[-2:]),
            ).bool().expand(-1, physical.shape[1], -1, -1)

        out = TwoPhaseOutput(
            deterministic=physical,
            deterministic_normalized=normalized,
        )
        if self.refiner is None:
            return out
        if self.residual_normalizer is None:
            raise RuntimeError("Active refinement has no residual normalizer.")
        # A refiner attached lazily during ``_prepare`` must mirror the already
        # evaluated parent. This assertion catches any future construction path
        # that could reactivate dropout or stochastic depth.
        if not self.training:
            self.refiner.eval()
        if self.training or self.refiner.training:
            raise RuntimeError("predict() requires model.eval(); stochasticity comes only from the sampler.")

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

        member_generators = []
        if noise_source_factory is None:
            # Each ensemble member gets its own generator, seeded from the caller's
            # generator (or from fresh entropy). Member m therefore always sees the
            # same noise sequence regardless of the ensemble chunking, which preserves
            # random draws; neural arithmetic may still depend on batch shape.
            gen_device = generator.device if generator is not None else cond.device
            if generator is not None:
                member_seeds = torch.randint(
                    0, 2**62, (n,), generator=generator, device=gen_device, dtype=torch.int64
                ).tolist()
            else:
                member_seeds = torch.randint(0, 2**62, (n,), dtype=torch.int64).tolist()
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
            source = (
                noise_source_factory(drawn) if noise_source_factory is not None
                else ChunkNoiseSource(chunk_generators, batch_size)
            )
            sample_options = {}
            if trajectory_callback is not None:
                def capture(stage, step, process_time, state):
                    shaped = state.reshape(batch_size, members, *state.shape[1:])
                    trajectory_callback(stage, step, process_time, shaped, drawn)
                sample_options["trajectory_callback"] = capture
            if members == 1:
                with self.refiner.use_noise_source(source):
                    res = self.refiner.sample(cond, **sample_options)
                residuals.append(res.unsqueeze(1))
            else:
                # Replicate the (identical) conditioning across members so a
                # single batched network call produces `members` independent
                # draws. Member order is preserved by the reshape below.
                cond_rep = cond.repeat_interleave(members, dim=0)
                with self.refiner.use_noise_source(source):
                    res = self.refiner.sample(cond_rep, **sample_options)
                res = res.reshape(batch_size, members, *res.shape[1:])
                residuals.append(res)
            drawn += members

        member_residuals = torch.cat(residuals, dim=1)  # raw residual-model space
        member_unbounded_fields = []
        for idx in range(n):
            raw_physical = self.residual_normalizer.denormalize(
                member_residuals[:, idx].to(physical.dtype)
            )
            correction = self._effective_physical_residual(raw_physical)
            # Build the physical field exactly once without clipping. The
            # selected ensemble strategy below performs the one final physical
            # constraint and exposes the pre-constraint tensor for diagnostics.
            member_unbounded_fields.append((physical + correction).unsqueeze(1))
        members_unbounded = torch.cat(member_unbounded_fields, dim=1)
        constrained = self.target_space.constrain_physical_ensemble(
            members_unbounded,
            strategy=self.refinement_config.nonnegative_ensemble_strategy,
            target_mask=prediction_mask,
        )
        members_physical = constrained["members"]
        members_unbounded = constrained["members_unbounded"]
        members_correction_physical = members_physical - physical.unsqueeze(1)

        # Ensemble statistics are accumulated in float32 regardless of the
        # compute dtype so mixed precision cannot bias the mean or the spread.
        # NaN cells are masked-out/invalid points: they are excluded from the
        # statistics rather than poisoning every member's contribution.
        # Average constrained physical members directly. Subtracting Phase 1
        # before the reduction and adding it back can turn exact dry zeros
        # into negative precipitation through float32 cancellation.
        stack32 = members_physical.float()
        valid = torch.isfinite(stack32)
        counts = valid.sum(dim=1)
        filled = torch.where(valid, stack32, torch.zeros_like(stack32))
        mean32 = filled.sum(dim=1) / counts.clamp(min=1)
        # Preserve exact Phase-1 identity when every valid member is unchanged,
        # avoiding summation drift without reconstructing the other means.
        base32 = physical.float()
        unchanged = ((stack32 == base32.unsqueeze(1)) | ~valid).all(dim=1)
        mean32 = torch.where(unchanged, base32, mean32)
        mean32 = torch.where(counts > 0, mean32, torch.full_like(mean32, float("nan")))
        ensemble_mean = mean32.to(members_physical.dtype)
        spread = None
        if n > 1:
            deviations = torch.where(
                valid,
                stack32 - mean32.unsqueeze(1),
                torch.zeros_like(stack32),
            )
            var = deviations.pow(2).sum(dim=1) / (counts - 1).clamp(min=1)
            var = torch.where(counts > 1, var, torch.full_like(var, float("nan")))
            spread = var.sqrt().to(members_physical.dtype)

        mean_residual = member_residuals.float().mean(dim=1).to(normalized.dtype)
        refined_physical = ensemble_mean if n > 1 else members_physical[:, 0]
        refined_norm = self.target_space.encode(refined_physical, scaler_offset=offset)
        effective_correction = refined_physical - physical

        out.members = members_physical if return_members else None
        out.members_unbounded = members_unbounded if return_members else None
        out.member_residuals = member_residuals if return_members else None
        out.member_residuals_physical = (
            members_correction_physical if return_members else None
        )
        out.ensemble_mean = ensemble_mean
        out.ensemble_spread = spread
        out.unbounded_ensemble_mean = constrained["unbounded_mean"]
        out.memberwise_clipped_mean = constrained["memberwise_clipped_mean"]
        out.memberwise_clipping_mean_shift = constrained[
            "memberwise_clipping_mean_shift"
        ]
        out.residual = mean_residual
        out.residual_physical = effective_correction
        out.refined_normalized = refined_norm
        out.refined = refined_physical
        return out

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def trainable_parameters(self) -> list[nn.Parameter]:
        if self.refiner is None or not self.refinement_config.freeze_phase1:
            return [p for p in self.parameters() if p.requires_grad]
        return [p for p in self.refiner.parameters() if p.requires_grad]

    def describe(self) -> dict[str, Any]:
        return {
            "refinement": self.refinement_config.to_dict(),
            "performance": self.performance_config.to_dict(),
            "phase1_frozen": self.phase1_frozen,
            "phase1_features": list(self._requested_features),
            "phase1_parameters": sum(p.numel() for p in self.phase1.parameters()),
            "refiner_parameters": (
                sum(p.numel() for p in self.refiner.parameters()) if self.refiner is not None else 0
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
