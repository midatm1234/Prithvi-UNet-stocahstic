"""Target-space transforms used by CORDEX output decoding."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F


class PositivePrecipLink(nn.Module):
    """Smooth non-negative link for precipitation outputs."""

    def __init__(self, beta: float = 1.0, threshold: float = 20.0) -> None:
        super().__init__()
        self.beta = float(beta)
        self.threshold = float(threshold)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.softplus(z, beta=self.beta, threshold=self.threshold)


def _apply_pr_positive_link_to_tensor(
    pred: torch.Tensor,
    var_names: Sequence[str],
    pr_name: str,
    link: nn.Module,
) -> torch.Tensor:
    if pred.ndim < 2:
        raise ValueError("Tensor prediction must have at least 2 dimensions [B, V, ...].")
    if pr_name not in var_names:
        return pred
    pr_idx = list(var_names).index(pr_name)
    out = pred.clone()
    out[:, pr_idx, ...] = link(out[:, pr_idx, ...])
    return out


def apply_pr_positive_link(
    pred: torch.Tensor | Mapping[str, torch.Tensor],
    var_names: Sequence[str],
    pr_name: str = "pr",
    link: nn.Module | None = None,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Apply precipitation positivity link to tensor outputs or dict outputs."""

    link = link or PositivePrecipLink()
    if isinstance(pred, Mapping):
        return {
            key: _apply_pr_positive_link_to_tensor(value, var_names, pr_name, link)
            for key, value in pred.items()
        }
    if torch.is_tensor(pred):
        return _apply_pr_positive_link_to_tensor(pred, var_names, pr_name, link)
    raise TypeError(f"Unsupported prediction type for pr link: {type(pred)}")
