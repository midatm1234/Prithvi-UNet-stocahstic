"""Score-matching loss for the conditional diffusion downscaling head.

Adapted from ``mlde``'s ``losses.py`` (``get_sde_loss_fn``). Only the continuous
score-matching objective needed for CORDEX training is kept. The companion
:class:`DiffusionLossPassthrough` lets the diffusion head slot into the existing
CORDEX trainer, whose loop is ``prediction = model(batch);
loss = loss_func(prediction, batch)``. In diffusion mode the model forward
already returns the scalar diffusion loss, so the loss function simply passes it
through (and exposes ``describe`` / ``get_last_terms`` like the deterministic
:class:`~granitewxc.models.loss.CompositePredictandLoss`).
"""

from __future__ import annotations

from typing import Callable

import torch

__all__ = ["score_matching_loss", "DiffusionLossPassthrough"]


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
) -> torch.Tensor:
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

    if not likelihood_weighting:
        losses = torch.square(score * std[:, None, None, None] + z)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    else:
        g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
        losses = torch.square(score + z / std[:, None, None, None])
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

    return torch.mean(losses)


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
