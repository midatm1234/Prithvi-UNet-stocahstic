"""Target transforms and the explicit physical residual-refinement contract.

This module provides the exact, invertible mapping between the Phase-1 model's
normalized target space and physical units. Modern Phase 2 first constructs a
physical residual and then uses its own training-only normalizer; it does not
reuse the full-field target scaler for stochastic residuals.

The forward/backward pair per channel is

======================  ===========================================  ==========================================
``scaling.method``      ``encode`` (physical -> normalized)           ``decode`` (normalized -> physical)
======================  ===========================================  ==========================================
``zscore``              ``(y - mu) / sigma``                          ``n * sigma + mu``
``divide_only``         ``y / sigma``                                 ``n * sigma``
``log1p_zscore``        ``(log1p(y) - mu) / sigma``                   ``expm1(n * sigma + mu)``
======================  ===========================================  ==========================================

Modern Phase-2 refinement uses physical residuals with a separately persisted
training-residual normalizer::

    residual_physical = target_physical - deterministic_physical
    residual_model    = residual_normalizer.normalize(residual_physical)
    predicted_physical = residual_normalizer.denormalize(predicted_model)
    refined_physical  = deterministic_physical + predicted_physical
    refined_physical  = apply_physical_constraints(refined_physical)

``deterministic_normalized`` is the ``x_pre_inverse`` tensor returned by the
Phase-1 model, i.e. the constrained normalized field that Phase 1 itself
inverts.  Using it (rather than re-encoding the physical output) keeps the
round trip exact and avoids inverting the network's softplus/exp output link.

Non-negativity and masks are applied only after the final physical addition.
The older normalized-target helpers remain for deterministic scaler round-trip
tests and controlled legacy migration, but the two-phase wrapper does not use
them as its training or inference residual contract.
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

    @staticmethod
    def _transform_values(values: torch.Tensor) -> torch.Tensor:
        # Half-precision expm1 overflows well inside finite float32 rainfall
        # ranges, and low-precision scaler subtraction loses small corrections.
        return values.float() if values.dtype in {torch.float16, torch.bfloat16} else values

    # -- transforms ------------------------------------------------------
    def encode(self, physical: torch.Tensor, scaler_offset=None) -> torch.Tensor:
        """Map a physical-unit target field into Phase-1 normalized space.

        NaNs are preserved (they mark invalid/missing cells) and never
        propagate into the scaler arithmetic in a way that changes the mask.
        """
        physical = self._transform_values(physical)
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
        normalized = self._transform_values(normalized)
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

    def constrain_physical_ensemble(
        self,
        unbounded_members: torch.Tensor,
        *,
        strategy: str = "memberwise",
        target_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Apply the selected final physical constraint to ``[B,N,C,H,W]``.

        ``memberwise`` reproduces the historical independent precipitation
        clamp. ``mean_preserving`` removes its Jensen shift: at each grid cell,
        non-negative members are lowered by one common offset (with a zero
        floor) until their mean equals the non-negative projection of the
        *unbounded* ensemble mean. Member ordering and nonzero spread are
        retained wherever a positive mean permits them. Channels without a
        non-negativity constraint are returned bit-for-bit unchanged.

        The returned unbounded members and ordinary memberwise-clipped mean are
        diagnostics, not hidden intermediate state.  The selected members are
        computed from the unbounded fields and receive one final constraint;
        callers must not pre-clamp them.
        """
        if unbounded_members.ndim != 5:
            raise ValueError(
                "Physical ensemble must have [B,N,C,H,W] layout, got "
                f"{tuple(unbounded_members.shape)}."
            )
        if unbounded_members.shape[2] != self.num_channels:
            raise ValueError(
                f"Physical ensemble has {unbounded_members.shape[2]} channels, "
                f"expected {self.num_channels}."
            )
        strategy = str(strategy).lower()
        if strategy not in {"memberwise", "mean_preserving"}:
            raise ValueError(
                "nonnegative ensemble strategy must be 'memberwise' or "
                f"'mean_preserving', got {strategy!r}."
            )

        unbounded = unbounded_members
        if target_mask is not None:
            expected = (unbounded.shape[0], unbounded.shape[2], *unbounded.shape[-2:])
            if tuple(target_mask.shape) != expected:
                raise ValueError(
                    f"Prediction mask {tuple(target_mask.shape)} does not match "
                    f"ensemble field shape {expected}."
                )
            mask = target_mask.to(device=unbounded.device, dtype=torch.bool).unsqueeze(1)
            unbounded = torch.where(mask, unbounded, torch.full_like(unbounded, float("nan")))

        enabled = self._phase1.predictand_nonneg_enabled_mask.to(
            unbounded.device
        ).view(1, 1, -1, 1, 1)
        memberwise = torch.where(enabled, torch.clamp(unbounded, min=0.0), unbounded)

        def finite_mean(values: torch.Tensor) -> torch.Tensor:
            finite = torch.isfinite(values)
            counts = finite.sum(dim=1)
            total = torch.where(finite, values, torch.zeros_like(values)).sum(dim=1)
            mean = total / counts.clamp(min=1)
            return torch.where(counts > 0, mean, torch.full_like(mean, float("nan")))

        unbounded_mean = finite_mean(unbounded)
        memberwise_mean = finite_mean(memberwise)
        clipping_shift = memberwise_mean - unbounded_mean

        if strategy == "memberwise" or not bool(enabled.any()):
            selected = memberwise
        else:
            # Desired constrained mean. For a non-negative channel this is the
            # projection of the affine/unbounded ensemble mean, not the mean of
            # independently rectified members.
            desired = torch.where(
                enabled[:, 0], torch.clamp(unbounded_mean, min=0.0), unbounded_mean
            ).unsqueeze(1)
            finite = torch.isfinite(unbounded)
            counts = finite.sum(dim=1, keepdim=True).clamp(min=1)
            finite_memberwise = torch.where(finite, memberwise, torch.zeros_like(memberwise))
            low = torch.zeros_like(desired)
            high = finite_memberwise.amax(dim=1, keepdim=True)
            # Monotone water-filling solve for mean(max(x-delta, 0))=desired.
            # 32 iterations reaches float32 precision for physical climate
            # ranges while avoiding a member-count-dependent branchy sort.
            for _ in range(32):
                midpoint = 0.5 * (low + high)
                candidate = torch.clamp(unbounded - midpoint, min=0.0)
                candidate_mean = torch.where(
                    finite, candidate, torch.zeros_like(candidate)
                ).sum(dim=1, keepdim=True) / counts
                too_high = candidate_mean > desired
                low = torch.where(too_high, midpoint, low)
                high = torch.where(too_high, high, midpoint)
            projected = torch.clamp(unbounded - high, min=0.0)
            # A non-negative ensemble with exactly zero target mean has only
            # one feasible solution. Snap it to exact zero rather than leaving
            # a sub-ulp bisection remnant that accumulates into a dry-day bias.
            projected = torch.where(desired <= 0.0, torch.zeros_like(projected), projected)
            selected = torch.where(enabled, projected, unbounded)
            selected = torch.where(finite, selected, torch.full_like(selected, float("nan")))

        return {
            "members": selected,
            "members_unbounded": unbounded,
            "unbounded_mean": unbounded_mean,
            "memberwise_clipped_mean": memberwise_mean,
            "memberwise_clipping_mean_shift": clipping_shift,
        }

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

    def reconstruct_physical(
        self,
        deterministic_physical: torch.Tensor,
        predicted_residual_physical: torch.Tensor,
        *,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Perform the one allowed final reconstruction in physical units.

        This function deliberately has no scaler arguments: the caller must
        inverse-normalize the residual exactly once before entering it.  A zero
        residual is therefore an exact identity before optional domain masking.
        """
        if deterministic_physical.shape != predicted_residual_physical.shape:
            raise ValueError(
                "Physical residual shape "
                f"{tuple(predicted_residual_physical.shape)} does not match the "
                f"Phase-1 prediction {tuple(deterministic_physical.shape)}."
            )
        refined = self._transform_values(deterministic_physical) + self._transform_values(predicted_residual_physical)
        refined = self.apply_physical_constraints(refined)
        if target_mask is not None:
            if target_mask.shape != refined.shape:
                raise ValueError(
                    f"Prediction mask {tuple(target_mask.shape)} does not match "
                    f"the reconstructed field {tuple(refined.shape)}."
                )
            refined = torch.where(
                target_mask.to(device=refined.device, dtype=torch.bool),
                refined,
                torch.full_like(refined, float("nan")),
            )
        return refined

    def physical_residual_target(
        self,
        target_physical: torch.Tensor,
        deterministic_physical: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``ground_truth - Phase1`` in physical units and its mask."""
        if target_physical.shape != deterministic_physical.shape:
            raise ValueError(
                f"Target {tuple(target_physical.shape)} and Phase-1 prediction "
                f"{tuple(deterministic_physical.shape)} must have identical BCHW layout."
            )
        target_physical = self._transform_values(target_physical)
        deterministic_physical = self._transform_values(deterministic_physical)
        valid = torch.isfinite(target_physical) & torch.isfinite(deterministic_physical)
        residual = torch.where(
            valid,
            target_physical - deterministic_physical,
            torch.zeros_like(target_physical),
        )
        return residual, valid

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
        valid = torch.isfinite(target_physical) & torch.isfinite(deterministic_normalized)
        filled = torch.where(valid, target_physical, torch.zeros_like(target_physical))
        target_norm = self.encode(filled, scaler_offset=scaler_offset)
        residual = target_norm - deterministic_normalized
        residual = torch.where(valid, residual, torch.zeros_like(residual))
        return residual, valid
