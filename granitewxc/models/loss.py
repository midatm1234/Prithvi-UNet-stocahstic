from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


def rmse_loss(y_hat: torch.Tensor, y: dict[str, torch.Tensor]) -> torch.Tensor:
    # NaN-safe RMSE over valid target pixels only (PRISM is NaN over ocean).
    target = y["y"]
    valid = torch.isfinite(target)
    if not bool(valid.any()):
        return y_hat.sum() * 0.0
    diff = (y_hat - torch.where(valid, target, y_hat.detach())) ** 2
    valid_f = valid.to(y_hat.dtype)
    return torch.sqrt((diff * valid_f).sum() / valid_f.sum().clamp(min=1.0) + 1e-12)


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


@dataclass
class SpatialGradientLossSpec:
    enabled: bool
    weight: float
    channel_weights: tuple[float, ...]
    precipitation_wet_only: bool
    precipitation_wet_threshold: float


@dataclass
class TotalVariationLossSpec:
    enabled: bool
    weight: float
    channel_weights: tuple[float, ...]


@dataclass
class MultiScaleLossSpec:
    enabled: bool
    weight: float
    scales: tuple[int, ...]
    loss_type: str


@dataclass
class BoundaryContinuityLossSpec:
    enabled: bool
    weight: float
    channel_weights: tuple[float, ...]
    stride_lat: int
    stride_lon: int
    match_target: bool


PRECIP_VAR_NAMES = {"pr", "precip", "precipitation", "ppt"}


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
        "bernoulli_gamma": "bernoulli_gamma",
        "bernoulli-gamma": "bernoulli_gamma",
        "bg": "bernoulli_gamma",
        "zero_inflated_gamma": "bernoulli_gamma",
    }
    if text not in aliases:
        raise ValueError(
            f"Unsupported precip_model '{value}'. Expected one of {sorted(aliases)}."
        )
    return aliases[text]


def _resolve_bernoulli_gamma_index(
    output_vars: list[str], cfg: Mapping[str, Any]
) -> int:
    """Return the physical precipitation channel modeled by the BG head."""

    precip_model = _canonicalize_precip_model(
        cfg.get("precip_model", cfg.get("precip_head_type", "single_head"))
    )
    if precip_model != "bernoulli_gamma":
        return -1
    precip_index = next(
        (
            idx
            for idx, name in enumerate(output_vars)
            if str(name).lower() in PRECIP_VAR_NAMES
        ),
        -1,
    )
    if precip_index < 0:
        raise ValueError(
            "precip_model='bernoulli_gamma' requires a precipitation output "
            "variable (e.g. 'pr')."
        )
    return precip_index


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

        self._bg_precip_index = _resolve_bernoulli_gamma_index(
            self.output_vars, cfg
        )
        self._bg_loss = None
        if self._bg_precip_index >= 0:
            from granitewxc.models.bernoulli_gamma_head import (
                BernoulliGammaConfig,
                BernoulliGammaLoss,
            )

            bg_config = BernoulliGammaConfig(
                wet_threshold=float(cfg.get("precip_wet_threshold", 0.1)),
                lambda_occurrence=float(
                    cfg.get("precip_lambda_occurrence", 1.0)
                ),
                lambda_positive_amount=float(
                    cfg.get("precip_lambda_amount", 1.0)
                ),
                min_mu=float(cfg.get("precip_bg_min_mu", 1e-4)),
                min_phi=float(cfg.get("precip_bg_min_phi", 1e-4)),
            )
            self._bg_loss = BernoulliGammaLoss(
                config=bg_config, output_vars=self.output_vars
            )

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

        gradient_cfg = _coerce_mapping(cfg.get("spatial_gradient"))
        if "use_gradient_loss" in cfg:
            gradient_cfg["enabled"] = bool(cfg.get("use_gradient_loss"))
        if "gradient_loss_weight" in cfg:
            gradient_cfg["weight"] = float(cfg.get("gradient_loss_weight", 0.0))
        gradient_enabled = bool(gradient_cfg.get("enabled", False))
        gradient_weight = float(gradient_cfg.get("weight", 0.0))
        channel_weights = [1.0 for _ in self.output_vars]
        gradient_predictands = _coerce_mapping(gradient_cfg.get("predictands"))
        for idx, name in enumerate(self.output_vars):
            if name in gradient_predictands:
                channel_weights[idx] = max(0.0, float(gradient_predictands[name]))
        precip_idx = next(
            (idx for idx, name in enumerate(self.output_vars) if str(name).lower() in PRECIP_VAR_NAMES),
            -1,
        )
        self._spatial_gradient = SpatialGradientLossSpec(
            enabled=gradient_enabled and gradient_weight > 0.0,
            weight=max(0.0, gradient_weight),
            channel_weights=tuple(channel_weights),
            precipitation_wet_only=bool(gradient_cfg.get("precipitation_wet_only", True)),
            precipitation_wet_threshold=float(
                gradient_cfg.get(
                    "precipitation_wet_threshold",
                    cfg.get("precip_wet_threshold", 0.01),
                )
            ),
        )

        tv_cfg = _coerce_mapping(cfg.get("tv"))
        if "use_tv_loss" in cfg:
            tv_cfg["enabled"] = bool(cfg.get("use_tv_loss"))
        if "tv_loss_weight" in cfg:
            tv_cfg["weight"] = float(cfg.get("tv_loss_weight", 0.0))
        tv_channel_weights = [1.0 for _ in self.output_vars]
        tv_predictands = _coerce_mapping(tv_cfg.get("predictands"))
        for idx, name in enumerate(self.output_vars):
            if name in tv_predictands:
                tv_channel_weights[idx] = max(0.0, float(tv_predictands[name]))
        self._tv = TotalVariationLossSpec(
            enabled=bool(tv_cfg.get("enabled", False)) and float(tv_cfg.get("weight", 0.0)) > 0.0,
            weight=max(0.0, float(tv_cfg.get("weight", 0.0))),
            channel_weights=tuple(tv_channel_weights),
        )

        multiscale_cfg = _coerce_mapping(cfg.get("multiscale"))
        if "use_multiscale_loss" in cfg:
            multiscale_cfg["enabled"] = bool(cfg.get("use_multiscale_loss"))
        if "multiscale_loss_weight" in cfg:
            multiscale_cfg["weight"] = float(cfg.get("multiscale_loss_weight", 0.0))
        scales_raw = multiscale_cfg.get("scales", (2, 4))
        scales: list[int] = []
        if isinstance(scales_raw, (int, float)):
            scales = [int(scales_raw)]
        elif isinstance(scales_raw, (list, tuple)):
            scales = [int(item) for item in scales_raw]
        scales = [value for value in scales if value > 1]
        if not scales:
            scales = [2]
        multiscale_loss_type = str(multiscale_cfg.get("loss_type", "l1")).lower()
        if multiscale_loss_type not in {"l1", "l2", "rmse"}:
            multiscale_loss_type = "l1"
        self._multiscale = MultiScaleLossSpec(
            enabled=bool(multiscale_cfg.get("enabled", False))
            and float(multiscale_cfg.get("weight", 0.0)) > 0.0,
            weight=max(0.0, float(multiscale_cfg.get("weight", 0.0))),
            scales=tuple(scales),
            loss_type=multiscale_loss_type,
        )

        boundary_cfg = _coerce_mapping(cfg.get("boundary_continuity"))
        boundary_channel_weights = [1.0 for _ in self.output_vars]
        boundary_predictands = _coerce_mapping(boundary_cfg.get("predictands"))
        for idx, name in enumerate(self.output_vars):
            if name in boundary_predictands:
                boundary_channel_weights[idx] = max(0.0, float(boundary_predictands[name]))
        mask_size = cfg.get("mask_unit_size", (16, 16))
        if isinstance(mask_size, (list, tuple)) and len(mask_size) >= 2:
            default_stride_lat = int(mask_size[0])
            default_stride_lon = int(mask_size[1])
        else:
            default_stride_lat = default_stride_lon = 16
        self._boundary_continuity = BoundaryContinuityLossSpec(
            enabled=bool(boundary_cfg.get("enabled", False))
            and float(boundary_cfg.get("weight", 0.0)) > 0.0,
            weight=max(0.0, float(boundary_cfg.get("weight", 0.0))),
            channel_weights=tuple(boundary_channel_weights),
            stride_lat=max(2, int(boundary_cfg.get("stride_lat", default_stride_lat))),
            stride_lon=max(2, int(boundary_cfg.get("stride_lon", default_stride_lon))),
            match_target=bool(boundary_cfg.get("match_target", True)),
        )
        self._precip_index = precip_idx
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

    @staticmethod
    def _masked_mean_abs(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.to(dtype=value.dtype)
        if not bool(mask.any().item()):
            return torch.zeros((), device=value.device, dtype=value.dtype)
        return (torch.abs(value) * mask_f).sum() / mask_f.sum().clamp_min(1.0)

    def _compute_spatial_gradient_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        weights = torch.tensor(
            self._spatial_gradient.channel_weights,
            device=pred.device,
            dtype=pred.dtype,
        )
        total = torch.zeros((), device=pred.device, dtype=pred.dtype)
        total_weight = torch.zeros((), device=pred.device, dtype=pred.dtype)

        for ch_idx in range(pred.shape[1]):
            weight = weights[ch_idx]
            if float(weight.item()) <= 0.0:
                continue

            pred_ch = pred[:, ch_idx : ch_idx + 1, ...]
            target_ch = target[:, ch_idx : ch_idx + 1, ...]
            valid_ch = valid[:, ch_idx : ch_idx + 1, ...]

            pred_dx = pred_ch[..., :, 1:] - pred_ch[..., :, :-1]
            pred_dy = pred_ch[..., 1:, :] - pred_ch[..., :-1, :]
            target_dx = target_ch[..., :, 1:] - target_ch[..., :, :-1]
            target_dy = target_ch[..., 1:, :] - target_ch[..., :-1, :]
            valid_x = valid_ch[..., :, 1:] & valid_ch[..., :, :-1]
            valid_y = valid_ch[..., 1:, :] & valid_ch[..., :-1, :]

            if (
                self._spatial_gradient.precipitation_wet_only
                and ch_idx == self._precip_index
            ):
                wet_threshold = self._spatial_gradient.precipitation_wet_threshold
                wet_x = (
                    (target_ch[..., :, 1:] > wet_threshold)
                    | (target_ch[..., :, :-1] > wet_threshold)
                ) & valid_x
                wet_y = (
                    (target_ch[..., 1:, :] > wet_threshold)
                    | (target_ch[..., :-1, :] > wet_threshold)
                ) & valid_y

                loss_x = self._masked_mean_abs(pred_dx - target_dx, wet_x)
                loss_y = self._masked_mean_abs(pred_dy - target_dy, wet_y)
            else:
                loss_x = self._masked_mean_abs(pred_dx - target_dx, valid_x)
                loss_y = self._masked_mean_abs(pred_dy - target_dy, valid_y)

            channel_loss = 0.5 * (loss_x + loss_y)
            total = total + weight * channel_loss
            total_weight = total_weight + weight

        if float(total_weight.item()) <= 0.0:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        return total / total_weight

    def _compute_total_variation_loss(self, pred: torch.Tensor) -> torch.Tensor:
        weights = torch.tensor(
            self._tv.channel_weights,
            device=pred.device,
            dtype=pred.dtype,
        )
        total = torch.zeros((), device=pred.device, dtype=pred.dtype)
        total_weight = torch.zeros((), device=pred.device, dtype=pred.dtype)

        for ch_idx in range(pred.shape[1]):
            weight = weights[ch_idx]
            if float(weight.item()) <= 0.0:
                continue
            pred_ch = pred[:, ch_idx : ch_idx + 1, ...]
            dx = pred_ch[..., :, 1:] - pred_ch[..., :, :-1]
            dy = pred_ch[..., 1:, :] - pred_ch[..., :-1, :]
            channel_tv = 0.5 * (torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy)))
            total = total + weight * channel_tv
            total_weight = total_weight + weight

        if float(total_weight.item()) <= 0.0:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        return total / total_weight

    def _compute_multiscale_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        terms: list[torch.Tensor] = []
        valid_f = valid.to(dtype=pred.dtype)
        for scale in self._multiscale.scales:
            if pred.shape[-2] < scale or pred.shape[-1] < scale:
                continue
            count_ds = F.avg_pool2d(valid_f, kernel_size=scale, stride=scale)
            block_valid = count_ds > 0.0
            if not bool(block_valid.any().item()):
                continue
            denom = count_ds.clamp_min(1.0 / float(scale * scale))
            pred_ds = F.avg_pool2d(pred * valid_f, kernel_size=scale, stride=scale) / denom
            target_ds = F.avg_pool2d(target * valid_f, kernel_size=scale, stride=scale) / denom
            diff = pred_ds[block_valid] - target_ds[block_valid]
            if self._multiscale.loss_type == "l2":
                term = torch.mean(diff ** 2)
            elif self._multiscale.loss_type == "rmse":
                term = torch.sqrt(torch.mean(diff ** 2) + 1e-12)
            else:
                term = torch.mean(torch.abs(diff))
            terms.append(term)
        if not terms:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        return torch.stack(terms).mean()

    def _compute_boundary_continuity_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weights = torch.tensor(
            self._boundary_continuity.channel_weights,
            device=pred.device,
            dtype=pred.dtype,
        )
        total = torch.zeros((), device=pred.device, dtype=pred.dtype)
        total_weight = torch.zeros((), device=pred.device, dtype=pred.dtype)
        stride_lat = self._boundary_continuity.stride_lat
        stride_lon = self._boundary_continuity.stride_lon
        h, w = pred.shape[-2:]
        y_idx = [idx for idx in range(stride_lat, h, stride_lat)]
        x_idx = [idx for idx in range(stride_lon, w, stride_lon)]
        if not y_idx and not x_idx:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)

        for ch_idx in range(pred.shape[1]):
            weight = weights[ch_idx]
            if float(weight.item()) <= 0.0:
                continue
            pred_ch = pred[:, ch_idx : ch_idx + 1, ...]
            target_ch = target[:, ch_idx : ch_idx + 1, ...]
            channel_terms: list[torch.Tensor] = []

            if x_idx:
                pred_jump_x = torch.cat(
                    [pred_ch[..., :, idx : idx + 1] - pred_ch[..., :, idx - 1 : idx] for idx in x_idx],
                    dim=-1,
                )
                if self._boundary_continuity.match_target:
                    target_jump_x = torch.cat(
                        [target_ch[..., :, idx : idx + 1] - target_ch[..., :, idx - 1 : idx] for idx in x_idx],
                        dim=-1,
                    )
                    channel_terms.append(torch.mean(torch.abs(pred_jump_x - target_jump_x)))
                else:
                    channel_terms.append(torch.mean(torch.abs(pred_jump_x)))

            if y_idx:
                pred_jump_y = torch.cat(
                    [pred_ch[..., idx : idx + 1, :] - pred_ch[..., idx - 1 : idx, :] for idx in y_idx],
                    dim=-2,
                )
                if self._boundary_continuity.match_target:
                    target_jump_y = torch.cat(
                        [target_ch[..., idx : idx + 1, :] - target_ch[..., idx - 1 : idx, :] for idx in y_idx],
                        dim=-2,
                    )
                    channel_terms.append(torch.mean(torch.abs(pred_jump_y - target_jump_y)))
                else:
                    channel_terms.append(torch.mean(torch.abs(pred_jump_y)))

            if not channel_terms:
                continue
            channel_term = torch.stack(channel_terms).mean()
            total = total + weight * channel_term
            total_weight = total_weight + weight

        if float(total_weight.item()) <= 0.0:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        return total / total_weight

    def _base_rmse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        excluded: set[int] = set()
        if self._precip_hurdle is not None:
            excluded.add(self._precip_hurdle.precip_index)
        if self._bg_precip_index >= 0:
            excluded.add(self._bg_precip_index)
        if not excluded:
            return torch.sqrt(torch.mean((pred - target) ** 2))

        keep_indices = [idx for idx in range(pred.shape[1]) if idx not in excluded]
        if not keep_indices:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        pred_non_precip = pred[:, keep_indices, ...]
        target_non_precip = target[:, keep_indices, ...]
        return torch.sqrt(torch.mean((pred_non_precip - target_non_precip) ** 2))

    def _base_rmse_masked(
        self, pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        """RMSE over VALID target pixels only (invalid/NaN cells are ignored).

        ``valid`` is a boolean tensor with the same shape as ``target``. Invalid
        target cells (PRISM ocean / outside-CONUS, ~44%) must not contribute to
        the loss; otherwise the model is trained to predict zeros there.
        """
        excluded: set[int] = set()
        if self._precip_hurdle is not None:
            excluded.add(self._precip_hurdle.precip_index)
        if self._bg_precip_index >= 0:
            excluded.add(self._bg_precip_index)
        if excluded:
            keep_indices = [idx for idx in range(pred.shape[1]) if idx not in excluded]
            if not keep_indices:
                return torch.zeros((), device=pred.device, dtype=pred.dtype)
            pred = pred[:, keep_indices, ...]
            target = target[:, keep_indices, ...]
            valid = valid[:, keep_indices, ...]
        valid_f = valid.to(pred.dtype)
        sq_err = (pred - target) ** 2 * valid_f
        denom = valid_f.sum().clamp(min=1.0)
        return torch.sqrt(sq_err.sum() / denom + 1e-12)

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
        # Validity mask for the precip channel (NaN over ocean / outside CONUS).
        valid_mask = batch.get("__valid_mask")
        if isinstance(valid_mask, torch.Tensor):
            vp = valid_mask[:, precip_idx, ...]
        else:
            vp = torch.isfinite(target)

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

        wet_target = (target > self._precip_hurdle.wet_threshold).to(dtype=wet_logits_use.dtype)
        # Occurrence loss over VALID precip pixels only (ignore missing cells).
        if bool(vp.any().item()):
            occ_loss = self._occ_loss_fn(wet_logits_use[vp], wet_target[vp])
        else:
            occ_loss = torch.zeros((), device=target.device, dtype=wet_logits_use.dtype)

        amount_target_norm = target.to(dtype=amount_pred_norm_use.dtype) / q95_use.to(
            dtype=amount_pred_norm_use.dtype
        )
        # Amount loss over wet AND valid pixels only.
        wet_mask = (wet_target > 0.5) & vp
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

    def _compute_bernoulli_gamma_loss(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        assert self._bg_precip_index >= 0
        assert self._bg_loss is not None
        aux = batch.get("__bernoulli_gamma_aux")
        if not isinstance(aux, Mapping):
            raise ValueError(
                "precip_model='bernoulli_gamma' requires model auxiliary "
                "outputs in batch['__bernoulli_gamma_aux']."
            )
        required = ("logit_wet", "log_mu", "log_phi")
        if any(not torch.is_tensor(aux.get(name)) for name in required):
            raise ValueError(
                "batch['__bernoulli_gamma_aux'] must contain tensors: "
                "logit_wet, log_mu, log_phi."
            )

        target = batch["y"][:, self._bg_precip_index, ...]
        valid_mask = batch.get("__valid_mask")
        if torch.is_tensor(valid_mask):
            valid_mask = valid_mask[:, self._bg_precip_index, ...]
        else:
            valid_mask = torch.isfinite(target)
        if not bool(valid_mask.any().item()):
            zero = aux["logit_wet"].sum() * 0.0
            return zero, {
                "bg.occurrence_bce": 0.0,
                "bg.positive_gamma_nll": 0.0,
                "bg.total": 0.0,
            }
        loss = self._bg_loss(
            aux["logit_wet"],
            aux["log_mu"],
            aux["log_phi"],
            target,
            validity_mask=valid_mask,
        )
        return loss, self._bg_loss.get_last_terms()

    def __call__(self, y_hat: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        raw_target = batch["y"]

        # ------------------------------------------------------------------
        # Missing-target handling. PRISM targets are NaN over ocean / outside
        # CONUS (~44% of the grid). Those pixels must be IGNORED by the loss, not
        # trained toward zero. We:
        #   * build a validity mask (finite cells);
        #   * sanitize the target by replacing invalid cells with the model's own
        #     prediction (detached) so every difference-based auxiliary term
        #     contributes exactly 0 there (and never produces NaN gradients);
        #   * use an explicitly masked base RMSE over valid cells only;
        #   * report the valid-pixel fraction.
        # ------------------------------------------------------------------
        valid = torch.isfinite(raw_target)
        target = torch.where(valid, raw_target, y_hat.detach().to(raw_target.dtype))
        batch = dict(batch)
        batch["y"] = target
        batch["__valid_mask"] = valid

        n_valid = float(valid.sum().item())
        n_total = float(valid.numel())
        valid_fraction = n_valid / max(n_total, 1.0)
        if n_valid <= 0.0:
            total = y_hat.sum() * 0.0
            self._last_terms = {
                "base.rmse": 0.0,
                "valid_fraction": 0.0,
                "valid_pixels": 0.0,
            }
            return total

        total = self._base_rmse_masked(y_hat, target, valid)
        terms: dict[str, float] = {
            "base.rmse": float(total.detach().item()),
            "valid_fraction": valid_fraction,
            "valid_pixels": n_valid,
        }

        if self._precip_hurdle is not None:
            precip_loss, precip_terms = self._compute_precip_hurdle_loss(batch)
            total = total + precip_loss
            terms.update(precip_terms)

        if self._bg_precip_index >= 0:
            bg_loss, bg_terms = self._compute_bernoulli_gamma_loss(batch)
            total = total + bg_loss
            terms.update(bg_terms)

        for ch_idx, spec in self._dist_specs.items():
            if not spec.enabled or spec.weight <= 0.0:
                continue
            if self._precip_hurdle is not None and ch_idx == self._precip_hurdle.precip_index:
                continue
            if self._bg_precip_index >= 0 and ch_idx == self._bg_precip_index:
                continue
            ch_valid = valid[:, ch_idx, ...]
            if not bool(ch_valid.any().item()):
                continue
            dist_term = self._compute_distribution_loss(
                y_hat[:, ch_idx, ...][ch_valid],
                target[:, ch_idx, ...][ch_valid],
                spec,
            )
            weighted = dist_term * spec.weight
            total = total + weighted
            name = self.output_vars[ch_idx]
            terms[f"{name}.distribution.{spec.method}"] = float(dist_term.detach().item())
            terms[f"{name}.distribution.{spec.method}.weighted"] = float(weighted.detach().item())

        if self._spatial_gradient.enabled:
            grad_term = self._compute_spatial_gradient_loss(y_hat, target, valid)
            weighted_grad = grad_term * self._spatial_gradient.weight
            total = total + weighted_grad
            terms["spatial_gradient"] = float(grad_term.detach().item())
            terms["spatial_gradient.weighted"] = float(weighted_grad.detach().item())

        if self._tv.enabled:
            tv_term = self._compute_total_variation_loss(y_hat)
            weighted_tv = tv_term * self._tv.weight
            total = total + weighted_tv
            terms["tv"] = float(tv_term.detach().item())
            terms["tv.weighted"] = float(weighted_tv.detach().item())

        if self._multiscale.enabled:
            multiscale_term = self._compute_multiscale_loss(y_hat, target, valid)
            weighted_multiscale = multiscale_term * self._multiscale.weight
            total = total + weighted_multiscale
            terms["multiscale"] = float(multiscale_term.detach().item())
            terms["multiscale.weighted"] = float(weighted_multiscale.detach().item())

        if self._boundary_continuity.enabled:
            boundary_term = self._compute_boundary_continuity_loss(y_hat, target)
            weighted_boundary = boundary_term * self._boundary_continuity.weight
            total = total + weighted_boundary
            terms["boundary_continuity"] = float(boundary_term.detach().item())
            terms["boundary_continuity.weighted"] = float(weighted_boundary.detach().item())

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
            "spatial_gradient": {
                "enabled": self._spatial_gradient.enabled,
                "weight": self._spatial_gradient.weight,
                "predictands": {
                    name: float(self._spatial_gradient.channel_weights[idx])
                    for idx, name in enumerate(self.output_vars)
                },
                "precipitation_wet_only": self._spatial_gradient.precipitation_wet_only,
                "precipitation_wet_threshold": self._spatial_gradient.precipitation_wet_threshold,
            },
            "tv": {
                "enabled": self._tv.enabled,
                "weight": self._tv.weight,
                "predictands": {
                    name: float(self._tv.channel_weights[idx])
                    for idx, name in enumerate(self.output_vars)
                },
            },
            "multiscale": {
                "enabled": self._multiscale.enabled,
                "weight": self._multiscale.weight,
                "scales": list(self._multiscale.scales),
                "loss_type": self._multiscale.loss_type,
            },
            "boundary_continuity": {
                "enabled": self._boundary_continuity.enabled,
                "weight": self._boundary_continuity.weight,
                "predictands": {
                    name: float(self._boundary_continuity.channel_weights[idx])
                    for idx, name in enumerate(self.output_vars)
                },
                "stride_lat": self._boundary_continuity.stride_lat,
                "stride_lon": self._boundary_continuity.stride_lon,
                "match_target": self._boundary_continuity.match_target,
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
        if self._bg_loss is not None:
            bg_cfg = self._bg_loss.cfg
            payload["bernoulli_gamma"] = {
                "enabled": True,
                "precip_index": self._bg_precip_index,
                "wet_threshold": bg_cfg.wet_threshold,
                "lambda_occurrence": bg_cfg.lambda_occurrence,
                "lambda_positive_amount": bg_cfg.lambda_positive_amount,
                "min_mu": bg_cfg.min_mu,
                "min_phi": bg_cfg.min_phi,
            }
        else:
            payload["bernoulli_gamma"] = {"enabled": False}
        return payload


def _head_type_is_diffusion(config: Any) -> bool:
    model_cfg = getattr(config, "model", None)
    if model_cfg is None:
        return False
    raw = getattr(model_cfg, "head_type", None)
    if raw is None:
        raw = getattr(model_cfg, "decoder_type", None)
    if raw is None:
        return False
    return str(raw).strip().lower() in {
        "diffusion",
        "diffusion_head",
        "sde",
        "score",
        "score_sde",
    }


def _residual_diffusion_enabled(config: Any) -> bool:
    model_cfg = getattr(config, "model", None)
    if isinstance(model_cfg, Mapping):
        diffusion_cfg = model_cfg.get("diffusion")
    else:
        diffusion_cfg = (
            getattr(model_cfg, "diffusion", None)
            if model_cfg is not None
            else None
        )
    return bool(_coerce_mapping(diffusion_cfg).get("residual_diffusion", False))


def build_loss_fn(config: Any, output_vars: list[str]):
    is_diffusion = _head_type_is_diffusion(config)
    is_residual_diffusion = is_diffusion and _residual_diffusion_enabled(config)
    if is_diffusion and not is_residual_diffusion:
        from granitewxc.models.diffusion_loss import DiffusionLossPassthrough

        return DiffusionLossPassthrough(output_vars=output_vars)

    loss_cfg = _coerce_mapping(getattr(config, "loss", {}))
    merged_cfg = dict(loss_cfg)
    for key in (
        "precip_model",
        "precip_head_type",
        "precip_wet_threshold",
        "precip_lambda_occurrence",
        "precip_lambda_amount",
        "precip_amount_loss_type",
        "precip_bg_min_mu",
        "precip_bg_min_phi",
        "use_gradient_loss",
        "gradient_loss_weight",
        "use_tv_loss",
        "tv_loss_weight",
        "use_multiscale_loss",
        "multiscale_loss_weight",
    ):
        if key not in merged_cfg and hasattr(config, key):
            merged_cfg[key] = getattr(config, key)
    if "mask_unit_size" not in merged_cfg and hasattr(config, "mask_unit_size"):
        merged_cfg["mask_unit_size"] = getattr(config, "mask_unit_size")

    model_cfg = getattr(config, "model", None)
    bg_cfg = _coerce_mapping(
        getattr(model_cfg, "bernoulli_gamma", None)
        if model_cfg is not None
        else None
    )
    for loss_key, bg_key in (
        ("precip_wet_threshold", "wet_threshold"),
        ("precip_lambda_occurrence", "lambda_occurrence"),
        ("precip_lambda_amount", "lambda_positive_amount"),
        ("precip_bg_min_mu", "min_mu"),
        ("precip_bg_min_phi", "min_phi"),
    ):
        if loss_key not in merged_cfg and bg_key in bg_cfg:
            merged_cfg[loss_key] = bg_cfg[bg_key]

    precip_model = _canonicalize_precip_model(
        merged_cfg.get("precip_model", merged_cfg.get("precip_head_type", "single_head"))
    )
    if (
        not is_residual_diffusion
        and not loss_cfg
        and precip_model not in {"hurdle", "bernoulli_gamma"}
    ):
        return rmse_loss
    deterministic_loss = CompositePredictandLoss(
        output_vars=output_vars, loss_cfg=merged_cfg
    )

    if is_residual_diffusion:
        from granitewxc.models.diffusion_loss import JointResidualDiffusionLoss

        diffusion_loss_cfg = _coerce_mapping(loss_cfg.get("diffusion"))
        joint_loss = JointResidualDiffusionLoss(
            deterministic_loss=deterministic_loss,
            output_vars=output_vars,
            deterministic_weight=float(loss_cfg.get("deterministic_weight", 1.0)),
            diffusion_weight=float(diffusion_loss_cfg.get("weight", 1.0)),
            corrected_mean_weight=diffusion_loss_cfg.get(
                "corrected_mean_weight", 0.0
            ),
            improvement_penalty_weight=diffusion_loss_cfg.get(
                "improvement_penalty_weight", 0.0
            ),
            minimum_relative_improvement=diffusion_loss_cfg.get(
                "minimum_relative_improvement", 0.0
            ),
        )
        correction_objective_enabled = (
            joint_loss.corrected_mean_weight > 0.0
            or joint_loss.improvement_penalty_weight > 0.0
        )
        has_precipitation = any(
            str(name).lower() in PRECIP_VAR_NAMES for name in output_vars
        )
        if (
            correction_objective_enabled
            and has_precipitation
            and precip_model != "single_head"
        ):
            raise ValueError(
                "YAML-enabled corrected-mean supervision requires "
                "precip_model='single_head' when precipitation is an output."
            )
        return joint_loss

    return deterministic_loss
