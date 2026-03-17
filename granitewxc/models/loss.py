from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


def rmse_loss(y_hat: torch.Tensor, y: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.sqrt(torch.mean((y_hat - y["y"]) ** 2))


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {}


@dataclass
class DistributionLossSpec:
    enabled: bool
    method: str
    weight: float
    use_mean: bool = True
    use_std: bool = True
    use_skewness: bool = False
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    bins: int = 100
    cdf_temperature: float = 0.05


class CompositePredictandLoss:
    """RMSE plus optional per-predictand distribution-aware penalties."""

    def __init__(self, output_vars: list[str], loss_cfg: Mapping[str, Any] | None = None):
        self.output_vars = list(output_vars)
        cfg = _coerce_mapping(loss_cfg)
        self.base = str(cfg.get("base", "rmse")).lower()
        if self.base != "rmse":
            raise ValueError(f"Unsupported base loss '{self.base}'. Only 'rmse' is supported.")

        predictand_cfg = _coerce_mapping(cfg.get("predictands"))
        self._dist_specs: dict[int, DistributionLossSpec] = {}
        for idx, name in enumerate(self.output_vars):
            var_cfg = _coerce_mapping(predictand_cfg.get(name))
            dist_cfg = _coerce_mapping(var_cfg.get("distribution_loss"))
            enabled = bool(dist_cfg.get("enabled", False))
            if not enabled:
                continue
            method = str(dist_cfg.get("method", "moment")).lower()
            if method not in {"moment", "quantile", "cdf"}:
                raise ValueError(
                    f"loss.predictands.{name}.distribution_loss.method='{method}' is invalid; "
                    "expected one of ['moment', 'quantile', 'cdf']"
                )
            weight = float(dist_cfg.get("weight", 0.0))
            quantiles = tuple(float(q) for q in dist_cfg.get("quantiles", (0.1, 0.5, 0.9)))
            bins = int(dist_cfg.get("bins", 100))
            cdf_temperature = float(dist_cfg.get("cdf_temperature", 0.05))
            self._dist_specs[idx] = DistributionLossSpec(
                enabled=enabled,
                method=method,
                weight=weight,
                use_mean=bool(dist_cfg.get("use_mean", True)),
                use_std=bool(dist_cfg.get("use_std", True)),
                use_skewness=bool(dist_cfg.get("use_skewness", False)),
                quantiles=quantiles,
                bins=max(8, bins),
                cdf_temperature=max(1e-4, cdf_temperature),
            )
        self._last_terms: dict[str, float] = {}

    @staticmethod
    def _flatten(values: torch.Tensor) -> torch.Tensor:
        return values.reshape(-1)

    def _moment_loss(self, pred: torch.Tensor, target: torch.Tensor, spec: DistributionLossSpec) -> torch.Tensor:
        pred_flat = self._flatten(pred)
        target_flat = self._flatten(target)
        pieces: list[torch.Tensor] = []

        if spec.use_mean:
            pieces.append(torch.abs(pred_flat.mean() - target_flat.mean()))

        if spec.use_std:
            pred_std = torch.sqrt(torch.clamp(pred_flat.var(unbiased=False), min=1e-12))
            target_std = torch.sqrt(torch.clamp(target_flat.var(unbiased=False), min=1e-12))
            pieces.append(torch.abs(pred_std - target_std))

        if spec.use_skewness:
            pred_center = pred_flat - pred_flat.mean()
            target_center = target_flat - target_flat.mean()
            pred_std = torch.sqrt(torch.clamp(pred_center.pow(2).mean(), min=1e-12))
            target_std = torch.sqrt(torch.clamp(target_center.pow(2).mean(), min=1e-12))
            pred_skew = (pred_center.pow(3).mean()) / (pred_std.pow(3) + 1e-12)
            target_skew = (target_center.pow(3).mean()) / (target_std.pow(3) + 1e-12)
            pieces.append(torch.abs(pred_skew - target_skew))

        if not pieces:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        return torch.stack(pieces).mean()

    def _quantile_loss(self, pred: torch.Tensor, target: torch.Tensor, spec: DistributionLossSpec) -> torch.Tensor:
        pred_flat = self._flatten(pred)
        target_flat = self._flatten(target)
        q = torch.tensor(spec.quantiles, device=pred.device, dtype=pred.dtype).clamp(0.0, 1.0)
        q_pred = torch.quantile(pred_flat, q)
        q_target = torch.quantile(target_flat, q)
        return torch.mean(torch.abs(q_pred - q_target))

    def _cdf_loss(self, pred: torch.Tensor, target: torch.Tensor, spec: DistributionLossSpec) -> torch.Tensor:
        pred_flat = self._flatten(pred)
        target_flat = self._flatten(target)

        lo = torch.minimum(pred_flat.min(), target_flat.min()).detach()
        hi = torch.maximum(pred_flat.max(), target_flat.max()).detach()
        span = torch.clamp(hi - lo, min=1e-6)
        edges = torch.linspace(lo, hi, steps=spec.bins, device=pred.device, dtype=pred.dtype)
        temp = span * spec.cdf_temperature + 1e-8

        pred_cdf = torch.sigmoid((edges.unsqueeze(0) - pred_flat.unsqueeze(1)) / temp).mean(dim=0)
        target_cdf = torch.sigmoid((edges.unsqueeze(0) - target_flat.unsqueeze(1)) / temp).mean(dim=0)
        return torch.mean((pred_cdf - target_cdf) ** 2)

    def _compute_distribution_loss(
        self, pred: torch.Tensor, target: torch.Tensor, spec: DistributionLossSpec
    ) -> torch.Tensor:
        if spec.method == "moment":
            return self._moment_loss(pred, target, spec)
        if spec.method == "quantile":
            return self._quantile_loss(pred, target, spec)
        if spec.method == "cdf":
            return self._cdf_loss(pred, target, spec)
        raise ValueError(f"Unsupported distribution loss method: {spec.method}")

    def __call__(self, y_hat: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        target = batch["y"]
        total = rmse_loss(y_hat, batch)
        terms: dict[str, float] = {"base.rmse": float(total.detach().item())}

        for ch_idx, spec in self._dist_specs.items():
            if not spec.enabled or spec.weight <= 0.0:
                continue
            dist_term = self._compute_distribution_loss(y_hat[:, ch_idx, ...], target[:, ch_idx, ...], spec)
            weighted = dist_term * spec.weight
            total = total + weighted
            name = self.output_vars[ch_idx]
            terms[f"{name}.distribution.{spec.method}"] = float(dist_term.detach().item())
            terms[f"{name}.distribution.{spec.method}.weighted"] = float(weighted.detach().item())

        self._last_terms = terms
        return total

    def get_last_terms(self) -> dict[str, float]:
        return dict(self._last_terms)

    def describe(self) -> dict[str, Any]:
        return {
            "base": self.base,
            "predictands": {
                self.output_vars[idx]: {
                    "enabled": spec.enabled,
                    "method": spec.method,
                    "weight": spec.weight,
                    "use_mean": spec.use_mean,
                    "use_std": spec.use_std,
                    "use_skewness": spec.use_skewness,
                    "quantiles": list(spec.quantiles),
                    "bins": spec.bins,
                    "cdf_temperature": spec.cdf_temperature,
                }
                for idx, spec in self._dist_specs.items()
            },
        }


def build_loss_fn(config: Any, output_vars: list[str]):
    loss_cfg = _coerce_mapping(getattr(config, "loss", {}))
    if not loss_cfg:
        return rmse_loss
    return CompositePredictandLoss(output_vars=output_vars, loss_cfg=loss_cfg)
