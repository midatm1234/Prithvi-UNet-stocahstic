"""Normalized target space used for residual refinement.

Phase 2 always operates in the *normalized target space* of the Phase-1 model,
never in physical units.  This module provides the exact, invertible mapping
between the two, derived from the Phase-1 model's own output scalers and
per-predictand scaling methods so that CORDEX, MERRA-PRISM and NARR-PRISM all
work without dataset-specific code.

The forward/backward pair per channel is

======================  ===========================================  ==========================================
``scaling.method``      ``encode`` (physical -> normalized)           ``decode`` (normalized -> physical)
======================  ===========================================  ==========================================
``zscore``              ``(y - mu) / sigma``                          ``n * sigma + mu``
``divide_only``         ``y / sigma``                                 ``n * sigma``
``log1p_zscore``        ``(log1p(y) - mu) / sigma``                   ``expm1(n * sigma + mu)``
======================  ===========================================  ==========================================

Residual reconstruction therefore is::

    residual_target   = encode(y)                     - deterministic_normalized
    refined_normalized = deterministic_normalized     + predicted_residual
    refined_physical   = decode(refined_normalized)          # exactly once
    refined_physical   = apply_physical_constraints(...)     # exactly once

``deterministic_normalized`` is the ``x_pre_inverse`` tensor returned by the
Phase-1 model, i.e. the constrained normalized field that Phase 1 itself
inverts.  Using it (rather than re-encoding the physical output) keeps the
round trip exact and avoids inverting the network's softplus/exp output link.

Non-negativity, masks and other physical constraints are applied *after*
``decode`` and are applied exactly once; they are never applied to a normalized
tensor and a normalized residual is never added to a physical field.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["NormalizedTargetSpace"]

_SCALING_ZSCORE = 0
_SCALING_DIVIDE_ONLY = 1
_SCALING_LOG1P_ZSCORE = 2

# Matches ``cordex_finetune_model._decode_outputs``: values in this band are the
# numerical noise of ``expm1`` around zero and are snapped to exactly zero.
_TINY_NEGATIVE_TOL = 1e-7


@dataclass
class _ResolvedScalers:
    mu: torch.Tensor  # [1, C, H, W] or [1, C, 1, 1]
    sigma: torch.Tensor


class NormalizedTargetSpace:
    """Encode/decode between physical target units and Phase-1 normalized space.

    Parameters
    ----------
    phase1:
        A :class:`~granitewxc.models.cordex_finetune_model.ClimateDownscaleFinetuneUNETModel`
        (or any module exposing ``output_scalers_sigma``,
        ``predictand_scaling_method_codes`` and ``predictand_nonneg_*`` buffers).
    """

    def __init__(self, phase1: torch.nn.Module) -> None:
        self._phase1 = phase1

    # -- scaler plumbing -------------------------------------------------
    def _scalers(self, reference: torch.Tensor, scaler_offset=None) -> _ResolvedScalers:
        mu, sigma = self._phase1._resolve_output_scalers(reference, scaler_offset=scaler_offset)
        return _ResolvedScalers(mu=mu, sigma=sigma)

    @property
    def scaling_codes(self) -> torch.Tensor:
        return self._phase1.predictand_scaling_method_codes

    @property
    def num_channels(self) -> int:
        return int(self._phase1.output_scalers_sigma.shape[1])

    # -- transforms ------------------------------------------------------
    def encode(self, physical: torch.Tensor, scaler_offset=None) -> torch.Tensor:
        """Map a physical-unit target field into Phase-1 normalized space.

        NaNs are preserved (they mark invalid/missing cells) and never
        propagate into the scaler arithmetic in a way that changes the mask.
        """
        scalers = self._scalers(physical, scaler_offset=scaler_offset)
        codes = self.scaling_codes.to(physical.device)
        mu = scalers.mu
        sigma = torch.clamp(scalers.sigma, min=1e-12)

        code_map = codes.view(1, -1, 1, 1)
        # zscore
        out = (physical - mu) / sigma
        # divide_only
        out = torch.where(code_map == _SCALING_DIVIDE_ONLY, physical / sigma, out)
        # log1p_zscore
        if bool((codes == _SCALING_LOG1P_ZSCORE).any()):
            safe = torch.clamp(physical, min=-1.0 + 1e-6)
            log_encoded = (torch.log1p(safe) - mu) / sigma
            out = torch.where(code_map == _SCALING_LOG1P_ZSCORE, log_encoded, out)
        return out

    def decode(self, normalized: torch.Tensor, scaler_offset=None) -> torch.Tensor:
        """Map a normalized field back to physical units (inverse of :meth:`encode`)."""
        scalers = self._scalers(normalized, scaler_offset=scaler_offset)
        codes = self.scaling_codes.to(normalized.device)
        mu = scalers.mu
        sigma = torch.clamp(scalers.sigma, min=1e-12)
        code_map = codes.view(1, -1, 1, 1)

        out = normalized * sigma + mu
        out = torch.where(code_map == _SCALING_DIVIDE_ONLY, normalized * sigma, out)
        if bool((codes == _SCALING_LOG1P_ZSCORE).any()):
            log_decoded = torch.expm1(normalized * sigma + mu)
            log_decoded = torch.where(
                (log_decoded < 0.0) & (log_decoded > -_TINY_NEGATIVE_TOL),
                torch.zeros_like(log_decoded),
                log_decoded,
            )
            out = torch.where(code_map == _SCALING_LOG1P_ZSCORE, log_decoded, out)
        return out

    # -- physical constraints -------------------------------------------
    def apply_physical_constraints(self, physical: torch.Tensor) -> torch.Tensor:
        """Clamp non-negative predictands (e.g. precipitation) to ``>= 0``.

        This mirrors the physical meaning of ``predictands.<name>.nonnegativity``
        without re-applying Phase 1's *network output link* (softplus/exp), which
        is an activation on the raw head output and not a data transform.  It is
        idempotent, so calling it once after residual reconstruction is exact.
        """
        mask = self._phase1.predictand_nonneg_enabled_mask
        if not bool(mask.any()):
            return physical
        enabled = mask.to(physical.device).view(1, -1, 1, 1)
        return torch.where(enabled, torch.clamp(physical, min=0.0), physical)

    def reconstruct(
        self,
        deterministic_normalized: torch.Tensor,
        predicted_residual: torch.Tensor,
        *,
        scaler_offset=None,
        target_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add a normalized residual and return ``(refined_norm, refined_physical)``.

        The residual is added in normalized space, inverse-normalized exactly
        once, and physical constraints/masks are then applied exactly once.
        """
        if deterministic_normalized.shape != predicted_residual.shape:
            raise ValueError(
                "Residual shape "
                f"{tuple(predicted_residual.shape)} does not match the deterministic "
                f"normalized prediction {tuple(deterministic_normalized.shape)}."
            )
        refined_norm = deterministic_normalized + predicted_residual
        refined_physical = self.decode(refined_norm, scaler_offset=scaler_offset)
        refined_physical = self.apply_physical_constraints(refined_physical)
        if target_mask is not None:
            refined_physical = torch.where(
                target_mask.to(refined_physical.device),
                refined_physical,
                torch.full_like(refined_physical, float("nan")),
            )
        return refined_norm, refined_physical

    def residual_target(
        self,
        target_physical: torch.Tensor,
        deterministic_normalized: torch.Tensor,
        *,
        scaler_offset=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build ``(residual_target, valid_mask)`` in normalized space.

        ``valid_mask`` is ``True`` where the target is finite. Invalid cells get a
        zero residual so they contribute nothing to a masked loss and never leak
        NaNs into the network.
        """
        valid = torch.isfinite(target_physical)
        filled = torch.where(valid, target_physical, torch.zeros_like(target_physical))
        target_norm = self.encode(filled, scaler_offset=scaler_offset)
        residual = target_norm - deterministic_normalized
        residual = torch.where(valid, residual, torch.zeros_like(residual))
        return residual, valid
