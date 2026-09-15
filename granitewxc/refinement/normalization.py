"""Training-only normalization for physical Phase-2 residuals.

The stochastic source distributions are dimensionless and approximately unit
scale.  Feeding them residuals normalized with the full target-field scaler can
turn one unit of harmless latent noise into tens of physical units.  This
module instead fits per-output-channel residual statistics from the *training
split only*, persists them in the Phase-2 checkpoint, and refuses to infer with
missing statistics.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from granitewxc.refinement.config import ResidualNormalizationConfig

__all__ = ["ResidualNormalizer"]


class ResidualNormalizer(nn.Module):
    """Masked streaming per-channel standardization of physical residuals.

    Statistics use a numerically stable parallel-Welford update.  Buffers are
    part of a modern refinement checkpoint, so validation and test data can
    never silently replace the training statistics.  ``identity`` keeps empty
    persistent state and is an explicit legacy-compatibility mode.
    """

    def __init__(
        self,
        channels: int,
        config: ResidualNormalizationConfig,
        *,
        nonnegative_mask: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.method = str(config.method)
        self.epsilon = float(config.epsilon)
        self.minimum_scale = float(config.minimum_scale)
        self.require_fitted = bool(config.require_fitted)
        self.signed_log_enabled = bool(config.signed_log_nonnegative_channels)
        self.signed_log_scale = float(config.signed_log_scale)
        if self.signed_log_enabled and self.method != "standardize":
            raise RuntimeError(
                "signed_log_nonnegative_channels requires method='standardize'."
            )
        persistent = self.method == "standardize"
        shape = (1, self.channels, 1, 1)
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float64), persistent=persistent)
        self.register_buffer("scale", torch.ones(shape, dtype=torch.float64), persistent=persistent)
        self.register_buffer("count", torch.zeros(shape, dtype=torch.float64), persistent=persistent)
        self.register_buffer("_m2", torch.zeros(shape, dtype=torch.float64), persistent=persistent)
        self.register_buffer("fitted", torch.tensor(self.method == "identity"), persistent=persistent)

        if self.signed_log_enabled:
            if nonnegative_mask is None:
                raise RuntimeError(
                    "signed_log_nonnegative_channels is enabled but no "
                    "nonnegative_mask was supplied to ResidualNormalizer."
                )
            mask = nonnegative_mask.detach().to(dtype=torch.bool).reshape(-1)
            if mask.numel() != self.channels:
                raise ValueError(
                    f"nonnegative_mask has {mask.numel()} entries, expected "
                    f"{self.channels}."
                )
            mask = mask.reshape(shape)
        else:
            mask = torch.zeros(shape, dtype=torch.bool)
        self.register_buffer("signed_log_mask", mask, persistent=self.signed_log_enabled)

    # -- signed-log residual transform ------------------------------------
    def _forward_transform(self, values: torch.Tensor) -> torch.Tensor:
        """``sign(r) * log1p(|r| / scale)`` on the configured nonnegative channels.

        Compresses the dynamic range of heavy-tailed precipitation residuals
        (a small fraction of extreme wet-cell errors otherwise dominates a
        per-channel MSE/Huber loss) while remaining exactly invertible and
        leaving unmasked channels (e.g. temperature) bit-for-bit unchanged.
        """
        if not self.signed_log_enabled:
            return values
        mask = self.signed_log_mask.to(device=values.device)
        transformed = torch.sign(values) * torch.log1p(torch.abs(values) / self.signed_log_scale)
        return torch.where(mask, transformed, values)

    def _inverse_transform(self, values: torch.Tensor) -> torch.Tensor:
        """Exact inverse of :meth:`_forward_transform`."""
        if not self.signed_log_enabled:
            return values
        mask = self.signed_log_mask.to(device=values.device)
        reconstructed = torch.sign(values) * torch.expm1(torch.abs(values)) * self.signed_log_scale
        return torch.where(mask, reconstructed, values)

    @property
    def is_identity(self) -> bool:
        return self.method == "identity"

    @property
    def is_fitted(self) -> bool:
        return self.is_identity or bool(self.fitted.item())

    def reset(self) -> None:
        if self.is_identity:
            return
        self.mean.zero_()
        self.scale.fill_(1.0)
        self.count.zero_()
        self._m2.zero_()
        self.fitted.zero_()

    @torch.no_grad()
    def update(self, residual_physical: torch.Tensor, valid_mask: torch.Tensor | None = None) -> None:
        """Accumulate a physical residual batch without retaining its graph."""
        if self.is_identity:
            return
        if residual_physical.ndim != 4 or residual_physical.shape[1] != self.channels:
            raise ValueError(
                "ResidualNormalizer expects [B,C,H,W] with "
                f"C={self.channels}, got {tuple(residual_physical.shape)}."
            )
        valid = torch.isfinite(residual_physical)
        if valid_mask is not None:
            if valid_mask.shape != residual_physical.shape:
                raise ValueError(
                    f"valid_mask {tuple(valid_mask.shape)} does not match residual "
                    f"{tuple(residual_physical.shape)}."
                )
            valid &= valid_mask.to(device=valid.device, dtype=torch.bool)

        values = residual_physical.detach().to(device=self.mean.device, dtype=torch.float64)
        values = self._forward_transform(values)
        valid = valid.to(self.mean.device)
        reduce_dims = (0, 2, 3)
        batch_count = valid.sum(dim=reduce_dims, keepdim=True).to(torch.float64)
        safe = torch.where(valid, values, torch.zeros_like(values))
        batch_mean = safe.sum(dim=reduce_dims, keepdim=True) / batch_count.clamp(min=1.0)
        centered = torch.where(valid, values - batch_mean, torch.zeros_like(values))
        batch_m2 = centered.square().sum(dim=reduce_dims, keepdim=True)

        total = self.count + batch_count
        delta = batch_mean - self.mean
        merge = delta.square() * self.count * batch_count / total.clamp(min=1.0)
        has_values = batch_count > 0
        self.mean.copy_(torch.where(has_values, self.mean + delta * batch_count / total.clamp(min=1.0), self.mean))
        self._m2.copy_(torch.where(has_values, self._m2 + batch_m2 + merge, self._m2))
        self.count.copy_(total)
        self.fitted.zero_()

    @torch.no_grad()
    def finalize(self) -> None:
        if self.is_identity:
            return
        missing = (self.count <= 0).reshape(-1)
        if bool(missing.any()):
            indices = torch.nonzero(missing, as_tuple=False).reshape(-1).tolist()
            raise RuntimeError(
                "Cannot fit residual normalization: no finite training residuals "
                f"were observed for output channel(s) {indices}."
            )
        variance = self._m2 / (self.count - 1.0).clamp(min=1.0)
        scale = torch.sqrt(torch.clamp(variance, min=0.0) + self.epsilon)
        self.scale.copy_(scale.clamp(min=self.minimum_scale))
        self.fitted.fill_(True)

    def _assert_ready(self) -> None:
        if self.method not in {"standardize", "identity"}:
            raise RuntimeError(f"Unsupported residual normalization method {self.method!r}.")
        if not self.is_fitted and self.require_fitted:
            raise RuntimeError(
                "Residual normalization statistics are not fitted. Fit them on the "
                "training split or load a compatible refinement checkpoint; validation "
                "and inference must never recompute these statistics."
            )

    def _arithmetic_values(self, residual: torch.Tensor) -> torch.Tensor:
        if residual.ndim != 4 or residual.shape[1] != self.channels:
            raise ValueError(
                f"ResidualNormalizer expects [B,{self.channels},H,W], got "
                f"{tuple(residual.shape)}."
            )
        # Statistics are accumulated in float64; residual arithmetic must be at
        # least float32 even when the neural network uses mixed precision.
        return residual.float() if residual.dtype in {torch.float16, torch.bfloat16} else residual

    def normalize(
        self, residual_physical: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        self._assert_ready()
        values = self._arithmetic_values(residual_physical)
        values = self._forward_transform(values)
        if valid_mask is not None:
            if valid_mask.shape != values.shape:
                raise ValueError(
                    f"valid_mask {tuple(valid_mask.shape)} does not match residual "
                    f"{tuple(values.shape)}."
                )
            valid_mask = valid_mask.to(device=values.device, dtype=torch.bool)
            values = torch.where(valid_mask, values, torch.zeros_like(values))
        if self.is_identity:
            out = values
        else:
            mean = self.mean.to(device=values.device, dtype=values.dtype)
            scale = self.scale.to(device=values.device, dtype=values.dtype)
            out = (values - mean) / scale.clamp(min=self.minimum_scale)
        if valid_mask is not None:
            out = torch.where(valid_mask, out, torch.zeros_like(out))
        return out

    def denormalize(self, residual_normalized: torch.Tensor) -> torch.Tensor:
        self._assert_ready()
        values = self._arithmetic_values(residual_normalized)
        if self.is_identity:
            out = values
        else:
            mean = self.mean.to(device=values.device, dtype=values.dtype)
            scale = self.scale.to(device=values.device, dtype=values.dtype)
            out = values * scale + mean
        return self._inverse_transform(out)

    def metadata(self) -> dict[str, Any]:
        """JSON-safe provenance and summary statistics.

        ``mean``/``scale`` are reported in the space actually standardized:
        the signed-log-transformed residual for channels where
        ``signed_log_nonnegative_channels`` is enabled, physical units
        otherwise.
        """
        meta = {
            "method": self.method,
            "epsilon": self.epsilon,
            "minimum_scale": self.minimum_scale,
            "require_fitted": self.require_fitted,
            "fitted": self.is_fitted,
            "count": self.count.detach().cpu().reshape(-1).tolist(),
            "mean": self.mean.detach().cpu().reshape(-1).tolist(),
            "scale": self.scale.detach().cpu().reshape(-1).tolist(),
        }
        if self.signed_log_enabled:
            meta["signed_log_nonnegative_channels"] = True
            meta["signed_log_scale"] = self.signed_log_scale
            meta["signed_log_mask"] = self.signed_log_mask.detach().cpu().reshape(-1).tolist()
        return meta
