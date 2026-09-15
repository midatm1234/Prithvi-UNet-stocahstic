"""Versioned experimental conditional means and frozen stochastic remainders.

This is an opt-in fitting API, not a replacement for production Phase 1. Network
inputs are an already-audited inference conditioning tensor; observations are
accepted only by training/target-preparation methods. No spatial bias map or
sampled-ensemble centering is used.
"""
from __future__ import annotations

import copy
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn

from .base import masked_loss
from .checkpoint import phase1_state_fingerprint
from .config import ResidualNormalizationConfig, config_fingerprint
from .normalization import ResidualNormalizer

MEAN_CONTRACT_VERSION = "conditional_signed_physical_mean_mse_v1"
REMAINDER_CONTRACT_VERSION = "frozen_mean_signed_physical_remainder_v1"
_REQUIRED_PROVENANCE = ("phase1_fingerprint", "conditioning_fingerprint", "training_selection_fingerprint")


def checked_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(provenance))
    missing = [key for key in _REQUIRED_PROVENANCE if not result.get(key)]
    if missing:
        raise ValueError(f"Missing experiment provenance: {missing}")
    return result


def training_only(split: str) -> None:
    if split != "train":
        raise ValueError("Statistics may be fitted only with split='train'.")


def tensor_fingerprint(module: nn.Module) -> str:
    return phase1_state_fingerprint(module.state_dict())


class SignedResidualMeanCorrector(nn.Module):
    """Small spatial regressor for E[y-b | inference conditioning].

    A fixed positive scale for each variable changes optimization balance, but
    does not change that variable's unconstrained conditional-MSE optimum.
    Scaling is division only: model initialization is exactly zero correction.
    Zero padding is explicit, matches all fitting/inference calls and does not
    claim to create missing regional context. There is no output activation.
    """
    def __init__(self, cond_channels: int, variables: Sequence[str], *,
                 hidden_channels: int = 32, depth: int = 3,
                 variable_weights: Sequence[float] | None = None) -> None:
        super().__init__()
        self.variables = tuple(str(v) for v in variables)
        if not self.variables or len(set(self.variables)) != len(self.variables):
            raise ValueError("Variables must be nonempty, ordered and unique.")
        if min(cond_channels, hidden_channels, depth) < 1:
            raise ValueError("Network dimensions must be positive.")
        self.cond_channels, self.hidden_channels, self.depth = int(cond_channels), int(hidden_channels), int(depth)
        weights = torch.as_tensor(variable_weights if variable_weights is not None else [1.] * len(self.variables), dtype=torch.float64)
        if weights.shape != (len(self.variables),) or not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
            raise ValueError("Mean variable weights must be finite, positive and match variables.")
        self.register_buffer("variable_weights", weights)
        blocks: list[nn.Module] = []
        for layer in range(depth):
            blocks.extend([nn.Conv2d(cond_channels if layer == 0 else hidden_channels, hidden_channels, 3, padding=1), nn.SiLU()])
        blocks.append(nn.Conv2d(hidden_channels, len(self.variables), 1))
        self.net = nn.Sequential(*blocks)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.scaling = ResidualNormalizer(len(self.variables), ResidualNormalizationConfig())
        self.scaling_provenance: dict[str, Any] | None = None

    @torch.no_grad()
    def fit_scaling(self, residual: torch.Tensor, valid: torch.Tensor | None = None, *,
                    split: str, provenance: Mapping[str, Any]) -> None:
        """One explicit fit on all designated fitting residuals; never evaluation."""
        training_only(split)
        if self.scaling.is_fitted:
            raise RuntimeError("Mean scales are already fitted; construct a fresh control to refit.")
        self.scaling_provenance = checked_provenance(provenance)
        self.scaling.update(residual, valid)
        self.scaling.finalize()

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        self.scaling._assert_ready()
        if conditioning.ndim != 4 or conditioning.shape[1] != self.cond_channels:
            raise ValueError("Mean conditioning must be BCHW with the configured channels.")
        if not bool(torch.isfinite(conditioning).all()):
            raise ValueError("Conditioning contains nonfinite cells; use the audited conditioning builder.")
        normalized_correction = self.net(conditioning.detach())
        return normalized_correction.float() * self.scaling.scale.to(normalized_correction.device, torch.float32)

    def training_loss(self, conditioning: torch.Tensor, baseline: torch.Tensor,
                      truth: torch.Tensor, valid: torch.Tensor | None = None) -> dict[str, Any]:
        if baseline.shape != truth.shape or baseline.shape[1] != len(self.variables):
            raise ValueError("Mean truth and unchanged Phase-1 baseline must match the configured variables.")
        valid = torch.isfinite(truth) & torch.isfinite(baseline) if valid is None else valid.bool() & torch.isfinite(truth) & torch.isfinite(baseline)
        correction = self(conditioning)
        target = truth.detach().float() - baseline.detach().float()
        scale = self.scaling.scale.to(target.device, target.dtype)
        losses = torch.stack([masked_loss(correction[:, i:i+1] / scale[:, i:i+1], target[:, i:i+1] / scale[:, i:i+1], valid[:, i:i+1], "mse") for i in range(len(self.variables))])
        active = valid.any(dim=(0, 2, 3))
        weights = self.variable_weights.to(losses) * active
        loss = (losses * weights).sum() / weights.sum().clamp(min=torch.finfo(losses.dtype).tiny)
        return {"loss": loss, "per_variable_loss": losses, "correction_physical": correction,
                "mean_physical": baseline.detach().float() + correction}

    def semantic_contract(self) -> dict[str, Any]:
        return {"version": MEAN_CONTRACT_VERSION, "variables": list(self.variables),
                "cond_channels": self.cond_channels, "hidden_channels": self.hidden_channels,
                "depth": self.depth, "variable_weights": self.variable_weights.cpu().tolist(),
                "target": "truth_physical - exact_frozen_phase1_physical", "loss": "fixed_scale_MSE",
                "scale_application": "division_only_no_centering", "padding": "zero_same",
                "scaling_provenance": self.scaling_provenance}

    def fingerprint(self) -> str:
        return config_fingerprint({"contract": self.semantic_contract(), "state": tensor_fingerprint(self)})

    def checkpoint(self) -> dict[str, Any]:
        self.scaling._assert_ready()
        checked_provenance(self.scaling_provenance or {})
        return {"kind": MEAN_CONTRACT_VERSION, "contract": self.semantic_contract(),
                "fingerprint": self.fingerprint(),
                "state_dict": {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}}

    @classmethod
    def from_checkpoint(cls, checkpoint: Mapping[str, Any], *,
                        expected_provenance: Mapping[str, Any] | None = None) -> "SignedResidualMeanCorrector":
        if checkpoint.get("kind") != MEAN_CONTRACT_VERSION:
            raise ValueError("Incompatible mean checkpoint; production residual weights cannot be reinterpreted.")
        c = checkpoint["contract"]
        model = cls(c["cond_channels"], c["variables"], hidden_channels=c["hidden_channels"], depth=c["depth"], variable_weights=c["variable_weights"])
        model.scaling_provenance = checked_provenance(c["scaling_provenance"])
        if expected_provenance is not None and model.scaling_provenance != checked_provenance(expected_provenance):
            raise ValueError("Mean checkpoint provenance does not match this Phase-1/conditioning/fitting selection.")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        if model.semantic_contract() != c or model.fingerprint() != checkpoint.get("fingerprint"):
            raise ValueError("Mean checkpoint semantic/state fingerprint mismatch.")
        model.scaling._assert_ready()
        return model


class FrozenMeanComposition(nn.Module):
    """Shared optional y-m remainder API for every native stochastic head.

    Freeze a selected mean model first, then fit entirely new remainder stats.
    The caller must build fresh experimental refinement weights whose checkpoint
    target contract is this object's contract. No old direct-residual state loads.
    """
    def __init__(self, mean_corrector: SignedResidualMeanCorrector) -> None:
        super().__init__()
        mean_corrector.scaling._assert_ready()
        self.mean_corrector = mean_corrector.requires_grad_(False).eval()
        self.mean_fingerprint = mean_corrector.fingerprint()
        self.normalizer = ResidualNormalizer(len(mean_corrector.variables), ResidualNormalizationConfig())
        self.remainder_provenance: dict[str, Any] | None = None

    def train(self, mode: bool = True):
        super().train(mode)
        self.mean_corrector.eval()
        return self

    @torch.no_grad()
    def fit_remainder_statistics(self, batches: Iterable[Mapping[str, torch.Tensor]], *,
                                 split: str, provenance: Mapping[str, Any]) -> None:
        training_only(split)
        if self.normalizer.is_fitted:
            raise RuntimeError("Remainder statistics already fitted; never reuse original y-b statistics.")
        if self.mean_corrector.fingerprint() != self.mean_fingerprint:
            raise RuntimeError("Mean corrector changed after freezing.")
        self.remainder_provenance = checked_provenance(provenance)
        for batch in batches:
            baseline, truth = batch["baseline"], batch["truth"]
            mean = baseline.detach().float() + self.mean_corrector(batch["conditioning"])
            valid = torch.isfinite(truth) & torch.isfinite(mean)
            if "valid" in batch:
                valid &= batch["valid"].bool()
            self.normalizer.update(truth.detach().float() - mean, valid)
        self.normalizer.finalize()

    def prepare(self, baseline: torch.Tensor, conditioning: torch.Tensor,
                truth: torch.Tensor, valid: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            mean = baseline.detach().float() + self.mean_corrector(conditioning)
            mask = torch.isfinite(truth) & torch.isfinite(mean)
            if valid is not None:
                mask &= valid.bool()
            remainder = truth.detach().float() - mean
            return {"baseline_physical": baseline.detach(), "mean_physical": mean,
                    "remainder_physical": remainder,
                    "remainder_normalized": self.normalizer.normalize(remainder, mask),
                    "zero_anchor": self.normalizer.normalize(torch.zeros_like(remainder)), "valid": mask}

    def reconstruct(self, baseline: torch.Tensor, conditioning: torch.Tensor,
                    normalized_members: torch.Tensor, refiner: nn.Module) -> dict[str, torch.Tensor]:
        if normalized_members.ndim != 5:
            raise ValueError("Members must be [B,M,C,H,W].")
        b, m, c, h, w = normalized_members.shape
        if baseline.shape != (b, c, h, w):
            raise ValueError("Member and Phase-1 domain/channel alignment mismatch.")
        with torch.no_grad():
            mean = baseline.detach().float() + self.mean_corrector(conditioning)
        remainder = self.normalizer.denormalize(normalized_members.reshape(b*m, c, h, w)).reshape(b, m, c, h, w)
        gated = refiner.apply_correction_gate(remainder)
        raw = mean[:, None] + gated
        return {"baseline_physical": baseline.detach(), "mean_physical": mean,
                "ungated_remainder_physical": remainder, "gated_remainder_physical": gated,
                "raw_members_physical": raw, "raw_ensemble_mean_physical": raw.mean(dim=1)}

    def semantic_contract(self) -> dict[str, Any]:
        self.normalizer._assert_ready()
        if self.mean_corrector.fingerprint() != self.mean_fingerprint:
            raise RuntimeError("Frozen mean fingerprint changed; cached remainders/checkpoint are incompatible.")
        return {"version": REMAINDER_CONTRACT_VERSION, "variables": list(self.mean_corrector.variables),
                "mean_fingerprint": self.mean_fingerprint, "target": "truth_physical - (exact_phase1 + frozen_mean_correction)",
                "normalizer_fingerprint": tensor_fingerprint(self.normalizer),
                "normalizer": self.normalizer.metadata(), "provenance": self.remainder_provenance}

    def checkpoint(self) -> dict[str, Any]:
        contract = self.semantic_contract()
        return {"kind": REMAINDER_CONTRACT_VERSION, "contract": contract,
                "mean_checkpoint": self.mean_corrector.checkpoint(),
                "normalizer_state": {k: v.detach().cpu().clone() for k, v in self.normalizer.state_dict().items()}}

    @classmethod
    def from_checkpoint(cls, checkpoint: Mapping[str, Any], *,
                        expected_provenance: Mapping[str, Any]) -> "FrozenMeanComposition":
        if checkpoint.get("kind") != REMAINDER_CONTRACT_VERSION:
            raise ValueError("Direct-residual weights/statistics cannot be loaded as a mean remainder.")
        contract = checkpoint["contract"]
        if contract.get("provenance") != checked_provenance(expected_provenance):
            raise ValueError("Remainder fitting provenance mismatch.")
        mean = SignedResidualMeanCorrector.from_checkpoint(checkpoint["mean_checkpoint"])
        model = cls(mean)
        model.normalizer.load_state_dict(checkpoint["normalizer_state"], strict=True)
        model.remainder_provenance = checked_provenance(expected_provenance)
        if model.semantic_contract() != contract:
            raise ValueError("Remainder mean/statistics semantic fingerprint mismatch.")
        return model
