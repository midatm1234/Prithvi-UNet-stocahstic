"""Score-matching loss for the conditional diffusion downscaling head.

Adapted from ``mlde``'s ``losses.py`` (``get_sde_loss_fn``). Only the continuous
score-matching objective needed for CORDEX training is kept. The companion
:class:`DiffusionLossPassthrough` lets the diffusion head slot into the existing
CORDEX trainer, whose loop is ``prediction = model(batch); loss =
loss_func(prediction, batch)``. Full-field diffusion forwards its scalar loss
through :class:`DiffusionLossPassthrough`. Residual diffusion uses
:class:`JointResidualDiffusionLoss` to combine score matching with the existing
supervised deterministic loss.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import torch

__all__ = [
    "score_matching_loss",
    "DiffusionLossPassthrough",
    "JointResidualDiffusionLoss",
]


def score_matching_loss(
    sde,
    score_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    target: torch.Tensor,
    cond: torch.Tensor,
    *,
    reduce_mean: bool = True,
    likelihood_weighting: bool = False,
    eps: float = 1e-5,
    generator: torch.Generator | None = None,
    clean_x0_reconstruction_weight: float = 0.0,
    clean_x0_inverse_snr_cap: float = 100.0,
    return_terms: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Continuous score-matching loss for an arbitrary SDE.

    Args:
        sde: forward SDE (see :mod:`granitewxc.models.diffusion_sde`).
        score_fn: conditional score function ``score_fn(x, cond, t)``.
        target: standardized target field, shape ``[B, C, H, W]``.
        cond: conditioning features, shape ``[B, C_cond, h, w]``.
        reduce_mean: average (vs. half-sum) across data dimensions.
        likelihood_weighting: weight the per-sample loss by ``g(t)**2``.
        eps: smallest diffusion time to sample.
        generator: optional RNG for reproducible noise/timesteps.
        clean_x0_reconstruction_weight: coefficient for a clean-target
            reconstruction term derived from the epsilon prediction. Zero
            preserves the original score-matching objective exactly.
        clean_x0_inverse_snr_cap: finite upper bound on the inverse-SNR
            amplification in the clean-target term. This stabilizes training
            loss weights only; it never clips or rescales generated samples.
        return_terms: return ``(total, detached_terms)`` for diagnostics.
    """
    reduce_op = (
        torch.mean
        if reduce_mean
        else (lambda tensor, dim: 0.5 * torch.sum(tensor, dim=dim))
    )

    batch = target
    if generator is None:
        t = torch.rand(batch.shape[0], device=batch.device, dtype=batch.dtype)
        z = torch.randn_like(batch)
    else:
        t = torch.rand(
            batch.shape[0], device=batch.device, dtype=batch.dtype, generator=generator
        )
        z = torch.empty_like(batch).normal_(generator=generator)
    t = t * (sde.T - eps) + eps

    mean, std = sde.marginal_prob(batch, t)
    perturbed_data = mean + std[:, None, None, None] * z
    score = score_fn(perturbed_data, cond, t)

    epsilon_error = score * std[:, None, None, None] + z
    epsilon_mse = torch.mean(
        torch.mean(
            torch.square(epsilon_error).reshape(epsilon_error.shape[0], -1),
            dim=-1,
        )
    )

    if not likelihood_weighting:
        losses = torch.square(epsilon_error)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    else:
        g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
        losses = torch.square(score + z / std[:, None, None, None])
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

    score_loss = torch.mean(losses)

    clean_weight = _validate_loss_weight(
        "clean_x0_reconstruction_weight", clean_x0_reconstruction_weight
    )
    inverse_snr_cap = _validate_positive_finite(
        "clean_x0_inverse_snr_cap", clean_x0_inverse_snr_cap
    )
    clean_x0_loss = score_loss.new_zeros(())
    if clean_weight > 0.0:
        # All supported forward SDEs have a spatially uniform scalar signal
        # coefficient: x_t = alpha_t*x0 + sigma_t*epsilon. Recover alpha with
        # a unit probe (a zero probe would erase it), then derive x0 from the
        # same epsilon/score prediction used by the primary objective.
        unit_probe = torch.ones(
            batch.shape[0], 1, 1, 1, device=batch.device, dtype=batch.dtype
        )
        mean_probe, _ = sde.marginal_prob(unit_probe, t)
        alpha = mean_probe[:, 0, 0, 0].clamp(min=1e-6)
        epsilon_prediction = -score * std[:, None, None, None]
        x0_prediction = (
            perturbed_data - std[:, None, None, None] * epsilon_prediction
        ) / alpha[:, None, None, None]

        # ||x0_hat-x0||^2 = inverse_SNR*||epsilon_hat-epsilon||^2.
        # min(1, cap*SNR) therefore caps only the inverse-SNR loss weight:
        #   min(1, cap*SNR) * x0_error^2
        #     = min(inverse_SNR, cap) * epsilon_error^2.
        # This supplies the clean reconstruction pressure needed at the
        # high-noise end without unbounded gradients and without touching
        # inference outputs.
        snr = alpha.square() / std.clamp(min=1e-6).square()
        stabilizer = torch.minimum(torch.ones_like(snr), inverse_snr_cap * snr)
        clean_per_example = reduce_op(
            torch.square(x0_prediction - batch).reshape(batch.shape[0], -1),
            dim=-1,
        )
        clean_x0_loss = torch.mean(stabilizer * clean_per_example)

    total = score_loss + clean_weight * clean_x0_loss
    if not return_terms:
        return total
    return total, {
        "score_matching": score_loss.detach(),
        "epsilon_mse": epsilon_mse.detach(),
        "clean_x0_reconstruction": clean_x0_loss.detach(),
        "clean_x0_weighted": (clean_weight * clean_x0_loss).detach(),
        "total": total.detach(),
    }


class DiffusionLossPassthrough:
    """Loss adapter for the diffusion head used by the CORDEX trainer.

    The diffusion model returns the scalar score-matching loss from its forward
    pass, so this callable simply forwards it. It mirrors the small interface
    (``describe`` / ``get_last_terms``) expected by the training loop.
    """

    def __init__(self, output_vars: list[str] | None = None):
        self.output_vars = list(output_vars or [])
        self._last_terms: dict[str, float] = {}

    def __call__(self, prediction, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        loss = _extract_loss(prediction)
        if loss is None:
            raise TypeError(
                "DiffusionLossPassthrough expected the diffusion model forward to "
                "return a scalar loss (or a mapping with a 'diffusion_loss'/'loss' "
                f"key), got type {type(prediction)!r}."
            )
        self._last_terms = {"diffusion_score_matching": float(loss.detach().cpu().item())}
        return loss

    def get_last_terms(self) -> dict[str, float]:
        return dict(self._last_terms)

    def describe(self) -> dict[str, object]:
        return {"type": "diffusion_score_matching", "output_vars": list(self.output_vars)}


class JointResidualDiffusionLoss:
    """Combine YAML-configured field losses with score matching.

    Residual-diffusion model forwards return a mapping containing a
    ``baseline_prediction``, an optional ``corrected_mean_prediction``, and a
    scalar ``diffusion_loss``.  Both physical prediction fields are evaluated by
    the exact same deterministic loss object built from the active YAML.  This
    adapter therefore never selects or hard-codes RMSE, gradient, distribution,
    or any other task-loss component.
    """

    def __init__(
        self,
        deterministic_loss: Callable[
            [torch.Tensor, dict[str, torch.Tensor]], torch.Tensor
        ],
        output_vars: list[str] | None = None,
        *,
        deterministic_weight: float = 1.0,
        diffusion_weight: float = 1.0,
        corrected_mean_weight: float = 0.0,
        improvement_penalty_weight: float = 0.0,
        minimum_relative_improvement: float = 0.0,
    ):
        self.deterministic_loss = deterministic_loss
        self.output_vars = list(output_vars or [])
        self.deterministic_weight = _validate_loss_weight(
            "loss.deterministic_weight", deterministic_weight
        )
        self.diffusion_weight = _validate_loss_weight(
            "loss.diffusion.weight", diffusion_weight
        )
        self.corrected_mean_weight = _validate_loss_weight(
            "loss.diffusion.corrected_mean_weight", corrected_mean_weight
        )
        self.improvement_penalty_weight = _validate_loss_weight(
            "loss.diffusion.improvement_penalty_weight",
            improvement_penalty_weight,
        )
        self.minimum_relative_improvement = _validate_relative_improvement(
            "loss.diffusion.minimum_relative_improvement",
            minimum_relative_improvement,
        )
        self._last_terms: dict[str, float] = {}

    def evaluate_configured_prediction(
        self,
        prediction: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Evaluate a physical field with the loss constructed from YAML.

        This method is also the validation hook for comparing a sampled
        ensemble mean with its paired U-Net baseline.  Term names are whatever
        the configured deterministic loss reports; no list of metrics is
        duplicated here.
        """
        if not isinstance(prediction, torch.Tensor):
            raise TypeError(
                "Configured predictand loss expected a prediction tensor, got "
                f"{type(prediction)!r}."
            )
        value = self.deterministic_loss(prediction, batch)
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            shape = tuple(value.shape) if isinstance(value, torch.Tensor) else None
            raise TypeError(
                "The YAML-configured deterministic loss must return a scalar "
                f"tensor, got {type(value)!r} with shape {shape}."
            )
        value = value.reshape(())
        reported: dict[str, float] = {}
        get_terms = getattr(self.deterministic_loss, "get_last_terms", None)
        if callable(get_terms):
            for name, term in get_terms().items():
                if isinstance(term, torch.Tensor):
                    term = _scalar_value(term)
                reported[str(name)] = float(term)
        return value, reported

    def __call__(
        self,
        prediction: Mapping[str, Any],
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if not isinstance(prediction, Mapping):
            raise TypeError(
                "JointResidualDiffusionLoss expected the residual-diffusion "
                "model forward to return a mapping with "
                "'baseline_prediction' and "
                f"'diffusion_loss', got type {type(prediction)!r}."
            )

        baseline_prediction = prediction.get("baseline_prediction")
        if not isinstance(baseline_prediction, torch.Tensor):
            raise TypeError(
                "JointResidualDiffusionLoss expected "
                "prediction['baseline_prediction'] to be a tensor."
            )
        diffusion_loss = prediction.get("diffusion_loss")
        if not isinstance(diffusion_loss, torch.Tensor):
            raise TypeError(
                "JointResidualDiffusionLoss expected "
                "prediction['diffusion_loss'] to be a tensor."
            )
        if diffusion_loss.numel() != 1:
            raise ValueError(
                "JointResidualDiffusionLoss expected "
                "prediction['diffusion_loss'] to be scalar, got shape "
                f"{tuple(diffusion_loss.shape)}."
            )
        diffusion_loss = diffusion_loss.reshape(())

        baseline_loss, baseline_terms = self.evaluate_configured_prediction(
            baseline_prediction, batch
        )
        correction_objective_enabled = (
            self.corrected_mean_weight > 0.0
            or self.improvement_penalty_weight > 0.0
        )
        corrected_prediction = prediction.get("corrected_mean_prediction")
        corrected_loss: torch.Tensor | None = None
        corrected_terms: dict[str, float] = {}
        if correction_objective_enabled:
            if not isinstance(corrected_prediction, torch.Tensor):
                raise TypeError(
                    "The YAML enables corrected-mean supervision, but the model "
                    "did not return tensor prediction['corrected_mean_prediction']. "
                    "Enable model.diffusion.residual_mean_enabled."
                )
            corrected_loss, corrected_terms = self.evaluate_configured_prediction(
                corrected_prediction, batch
            )

        weighted_deterministic = baseline_loss * self.deterministic_weight
        weighted_diffusion = diffusion_loss * self.diffusion_weight
        weighted_corrected = baseline_loss.new_zeros(())
        improvement_penalty = baseline_loss.new_zeros(())
        weighted_improvement_penalty = baseline_loss.new_zeros(())
        if corrected_loss is not None:
            weighted_corrected = corrected_loss * self.corrected_mean_weight
            # The comparison threshold is detached.  Otherwise the model could
            # reduce the hinge by making the U-Net baseline worse.
            required_loss = baseline_loss.detach() * (
                1.0 - self.minimum_relative_improvement
            )
            improvement_penalty = torch.relu(corrected_loss - required_loss)
            weighted_improvement_penalty = (
                improvement_penalty * self.improvement_penalty_weight
            )
        total = (
            weighted_deterministic
            + weighted_corrected
            + weighted_diffusion
            + weighted_improvement_penalty
        )

        terms: dict[str, float] = {}
        for name, value in baseline_terms.items():
            # ``deterministic.*`` is retained as a backward-compatible alias.
            terms[f"deterministic.{name}"] = value
            terms[f"baseline.{name}"] = value
        for name, value in corrected_terms.items():
            terms[f"corrected_mean.{name}"] = value
        reported_diffusion_terms = prediction.get("diffusion_loss_terms")
        if isinstance(reported_diffusion_terms, Mapping):
            for name, value in reported_diffusion_terms.items():
                if isinstance(value, torch.Tensor):
                    value = _scalar_value(value)
                terms[f"diffusion.{name}"] = float(value)
        else:
            # Backward-compatible label for models that return only the scalar
            # score loss and do not expose clean-x0 components.
            terms["diffusion.score_matching"] = _scalar_value(diffusion_loss)
        terms.update(
            {
                "deterministic.total": _scalar_value(baseline_loss),
                "deterministic.weighted": _scalar_value(
                    weighted_deterministic
                ),
                "baseline.total": _scalar_value(baseline_loss),
                "baseline.weighted": _scalar_value(weighted_deterministic),
                "diffusion.total": _scalar_value(diffusion_loss),
                "diffusion.weighted": _scalar_value(weighted_diffusion),
                "joint.total": _scalar_value(total),
            }
        )
        if corrected_loss is not None:
            baseline_denom = baseline_loss.detach().abs().clamp(min=1e-12)
            relative_improvement = (
                baseline_loss.detach() - corrected_loss.detach()
            ) / baseline_denom
            terms.update(
                {
                    "corrected_mean.total": _scalar_value(corrected_loss),
                    "corrected_mean.weighted": _scalar_value(weighted_corrected),
                    "correction.relative_improvement": _scalar_value(
                        relative_improvement
                    ),
                    "correction.improvement_penalty": _scalar_value(
                        improvement_penalty
                    ),
                    "correction.improvement_penalty_weighted": _scalar_value(
                        weighted_improvement_penalty
                    ),
                }
            )
        self._last_terms = terms
        return total

    def get_last_terms(self) -> dict[str, float]:
        return dict(self._last_terms)

    def describe(self) -> dict[str, object]:
        describe_deterministic = getattr(
            self.deterministic_loss, "describe", None
        )
        deterministic_description: object
        if callable(describe_deterministic):
            deterministic_description = describe_deterministic()
        else:
            deterministic_description = {
                "type": type(self.deterministic_loss).__name__,
            }
        return {
            "type": "joint_residual_diffusion",
            "output_vars": list(self.output_vars),
            "deterministic_weight": self.deterministic_weight,
            "diffusion_weight": self.diffusion_weight,
            "corrected_mean_weight": self.corrected_mean_weight,
            "improvement_penalty_weight": self.improvement_penalty_weight,
            "minimum_relative_improvement": self.minimum_relative_improvement,
            "deterministic_loss": deterministic_description,
        }


def _validate_loss_weight(name: str, value: float) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a finite non-negative number, got {value!r}."
        ) from exc
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError(
            f"{name} must be a finite non-negative number, got {value!r}."
        )
    return weight


def _validate_relative_improvement(name: str, value: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a finite number in [0, 1), got {value!r}."
        ) from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed < 1.0:
        raise ValueError(
            f"{name} must be a finite number in [0, 1), got {value!r}."
        )
    return parsed


def _validate_positive_finite(name: str, value: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a finite positive number, got {value!r}."
        ) from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}.")
    return parsed


def _scalar_value(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _extract_loss(prediction) -> torch.Tensor | None:
    if isinstance(prediction, torch.Tensor):
        return prediction
    if isinstance(prediction, dict):
        for key in ("diffusion_loss", "loss"):
            value = prediction.get(key)
            if isinstance(value, torch.Tensor):
                return value
    if isinstance(prediction, (tuple, list)) and prediction:
        first = prediction[0]
        if isinstance(first, torch.Tensor):
            return first
    return None
