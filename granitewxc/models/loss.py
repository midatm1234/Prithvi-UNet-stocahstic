from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn


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


@dataclass
class PrecipHurdleLossSpec:
    enabled: bool
    precip_index: int
    wet_threshold: float
    lambda_occurrence: float
    lambda_amount: float
    amount_loss_type: str


PRECIP_VAR_NAMES = {"pr", "precip", "precipitation"}


def _canonicalize_precip_model(value: Any) -> str:
    text = str(value or "single_head").strip().lower()
    aliases = {
        "single": "single_head",
        "single_head": "single_head",
        "default": "single_head",
        "legacy": "single_head",
        "hurdle": "hurdle",
        "bernoulli_positive": "hurdle",
        "bernoulli_plus_positive": "hurdle",
        "bernoulli_positive_amount": "hurdle",
        "bernoulli-plus-positive-amount": "hurdle",
    }
    if text not in aliases:
        raise ValueError(
            f"Unsupported precip_model '{value}'. Expected one of {sorted(aliases)}."
        )
    return aliases[text]


def _resolve_precip_hurdle_spec(output_vars: list[str], cfg: Mapping[str, Any]) -> PrecipHurdleLossSpec | None:
    precip_model = _canonicalize_precip_model(
        cfg.get("precip_model", cfg.get("precip_head_type", "single_head"))
    )
    if precip_model != "hurdle":
        return None

    precip_index = next(
        (idx for idx, name in enumerate(output_vars) if str(name).lower() in PRECIP_VAR_NAMES),
        -1,
    )
    if precip_index < 0:
        raise ValueError("precip_model='hurdle' requires a precipitation output variable (e.g. 'pr').")

    wet_threshold = float(cfg.get("precip_wet_threshold", 0.5))
    lambda_occurrence = float(cfg.get("precip_lambda_occurrence", 1.0))
    lambda_amount = float(cfg.get("precip_lambda_amount", 1.0))
    amount_loss_type = str(cfg.get("precip_amount_loss_type", "smoothl1")).lower()
    if amount_loss_type not in {"smoothl1", "mse"}:
        raise ValueError(
            f"precip_amount_loss_type='{amount_loss_type}' is invalid; expected 'smoothl1' or 'mse'."
        )

    return PrecipHurdleLossSpec(
        enabled=True,
        precip_index=precip_index,
        wet_threshold=wet_threshold,
        lambda_occurrence=lambda_occurrence,
        lambda_amount=lambda_amount,
        amount_loss_type=amount_loss_type,
    )


class CompositePredictandLoss:
    """RMSE plus optional per-predictand distribution-aware penalties."""

    def __init__(self, output_vars: list[str], loss_cfg: Mapping[str, Any] | None = None):
        self.output_vars = list(output_vars)
        cfg = _coerce_mapping(loss_cfg)
        self._cfg = cfg
        self.base = str(cfg.get("base", "rmse")).lower()
        if self.base != "rmse":
            raise ValueError(f"Unsupported base loss '{self.base}'. Only 'rmse' is supported.")

        self._precip_hurdle = _resolve_precip_hurdle_spec(self.output_vars, cfg)
        self._occ_loss_fn: nn.Module | None = None
        self._amount_loss_fn: nn.Module | None = None
        if self._precip_hurdle is not None:
            self._occ_loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
            if self._precip_hurdle.amount_loss_type == "mse":
                self._amount_loss_fn = nn.MSELoss(reduction="mean")
            else:
                self._amount_loss_fn = nn.SmoothL1Loss(reduction="mean")

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

    def _base_rmse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self._precip_hurdle is None:
            return torch.sqrt(torch.mean((pred - target) ** 2))

        precip_idx = self._precip_hurdle.precip_index
        keep_indices = [idx for idx in range(pred.shape[1]) if idx != precip_idx]
        if not keep_indices:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        pred_non_precip = pred[:, keep_indices, ...]
        target_non_precip = target[:, keep_indices, ...]
        return torch.sqrt(torch.mean((pred_non_precip - target_non_precip) ** 2))

    @staticmethod
    def _squeeze_single_channel(value: torch.Tensor) -> torch.Tensor:
        if value.ndim >= 2 and value.shape[1] == 1:
            return value[:, 0, ...]
        return value

    def _compute_precip_hurdle_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        assert self._precip_hurdle is not None
        assert self._occ_loss_fn is not None
        assert self._amount_loss_fn is not None

        aux = batch.get("__precip_hurdle_aux")
        if not isinstance(aux, Mapping):
            raise ValueError(
                "precip_model='hurdle' requires model auxiliary outputs in batch['__precip_hurdle_aux']."
            )

        wet_logits = aux.get("wet_logits")
        amount_pred_norm = aux.get("amount_pred_norm")
        q95 = aux.get("q95")
        if not torch.is_tensor(wet_logits) or not torch.is_tensor(amount_pred_norm) or not torch.is_tensor(q95):
            raise ValueError(
                "batch['__precip_hurdle_aux'] must contain tensors: wet_logits, amount_pred_norm, q95."
            )

        precip_idx = self._precip_hurdle.precip_index
        target = batch["y"][:, precip_idx, ...]

        wet_logits_use = self._squeeze_single_channel(wet_logits)
        amount_pred_norm_use = self._squeeze_single_channel(amount_pred_norm)
        q95_use = torch.clamp(self._squeeze_single_channel(q95), min=1e-12)

        if wet_logits_use.shape != target.shape:
            raise ValueError(
                f"wet_logits shape {tuple(wet_logits_use.shape)} does not match precip target shape {tuple(target.shape)}."
            )
        if amount_pred_norm_use.shape != target.shape:
            raise ValueError(
                f"amount_pred_norm shape {tuple(amount_pred_norm_use.shape)} does not match precip target shape {tuple(target.shape)}."
            )

        wet_target = (target > 0.0).to(dtype=wet_logits_use.dtype)
        occ_loss = self._occ_loss_fn(wet_logits_use, wet_target)

        amount_target_norm = target.to(dtype=amount_pred_norm_use.dtype) / q95_use.to(
            dtype=amount_pred_norm_use.dtype
        )
        wet_mask = wet_target > 0.5
        if bool(wet_mask.any().item()):
            amount_loss = self._amount_loss_fn(
                amount_pred_norm_use[wet_mask],
                amount_target_norm[wet_mask],
            )
        else:
            amount_loss = torch.zeros((), device=target.device, dtype=target.dtype)

        weighted_occ = occ_loss * self._precip_hurdle.lambda_occurrence
        weighted_amount = amount_loss * self._precip_hurdle.lambda_amount
        total = weighted_occ + weighted_amount
        terms = {
            "pr.hurdle.occurrence": float(occ_loss.detach().item()),
            "pr.hurdle.amount": float(amount_loss.detach().item()),
            "pr.hurdle.occurrence.weighted": float(weighted_occ.detach().item()),
            "pr.hurdle.amount.weighted": float(weighted_amount.detach().item()),
            "pr.hurdle.total": float(total.detach().item()),
        }
        return total, terms

    def __call__(self, y_hat: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        target = batch["y"]
        total = self._base_rmse(y_hat, target)
        terms: dict[str, float] = {"base.rmse": float(total.detach().item())}

        if self._precip_hurdle is not None:
            precip_loss, precip_terms = self._compute_precip_hurdle_loss(batch)
            total = total + precip_loss
            terms.update(precip_terms)

        for ch_idx, spec in self._dist_specs.items():
            if not spec.enabled or spec.weight <= 0.0:
                continue
            if self._precip_hurdle is not None and ch_idx == self._precip_hurdle.precip_index:
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
        payload = {
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
        if self._precip_hurdle is not None:
            payload["precip_hurdle"] = {
                "enabled": True,
                "precip_index": self._precip_hurdle.precip_index,
                "wet_threshold": self._precip_hurdle.wet_threshold,
                "lambda_occurrence": self._precip_hurdle.lambda_occurrence,
                "lambda_amount": self._precip_hurdle.lambda_amount,
                "amount_loss_type": self._precip_hurdle.amount_loss_type,
            }
        else:
            payload["precip_hurdle"] = {"enabled": False}
        return payload


def build_loss_fn(config: Any, output_vars: list[str]):
    loss_cfg = _coerce_mapping(getattr(config, "loss", {}))
    merged_cfg = dict(loss_cfg)
    for key in (
        "precip_model",
        "precip_head_type",
        "precip_wet_threshold",
        "precip_lambda_occurrence",
        "precip_lambda_amount",
        "precip_amount_loss_type",
    ):
        if key not in merged_cfg and hasattr(config, key):
            merged_cfg[key] = getattr(config, key)

    precip_model = _canonicalize_precip_model(
        merged_cfg.get("precip_model", merged_cfg.get("precip_head_type", "single_head"))
    )
    if not loss_cfg and precip_model != "hurdle":
        return rmse_loss
    return CompositePredictandLoss(output_vars=output_vars, loss_cfg=merged_cfg)
