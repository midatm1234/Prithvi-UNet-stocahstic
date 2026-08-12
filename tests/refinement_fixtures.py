"""Shared fixtures for the Phase-2 stochastic-refinement tests.

``TinyPhase1`` is a minimal stand-in for
``ClimateDownscaleFinetuneUNETModel``. It implements exactly the interface that
:class:`~granitewxc.refinement.two_phase.TwoPhaseDownscalingModel` relies on, so
the wrapper can be unit tested without loading a 3 GB Prithvi backbone.

It deliberately mirrors the real model's contract:

* ``forward(batch, return_pre_inverse=True)`` -> ``(physical, normalized)``
* ``_resolve_output_scalers(reference, scaler_offset=None)`` -> ``(mu, sigma)``
* ``predictand_scaling_method_codes`` / ``predictand_nonneg_enabled_mask`` buffers
* opt-in ``set_feature_capture`` / ``get_last_phase1_features``
"""

from __future__ import annotations

import torch
from torch import nn

SCALING_ZSCORE = 0
SCALING_DIVIDE_ONLY = 1
SCALING_LOG1P_ZSCORE = 2


class TinyPhase1(nn.Module):
    """Deterministic conv 'downscaler' with the real Phase-1 output contract."""

    SUPPORTED_FEATURE_CAPTURES = ("prithvi", "unet")

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        scaling_codes: tuple[int, ...] = (SCALING_DIVIDE_ONLY, SCALING_ZSCORE, SCALING_ZSCORE),
        nonneg: tuple[bool, ...] = (True, False, False),
        hidden: int = 8,
    ) -> None:
        super().__init__()
        if len(scaling_codes) != out_channels or len(nonneg) != out_channels:
            raise ValueError("scaling_codes / nonneg must match out_channels")
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
        )
        self.head = nn.Conv2d(hidden, out_channels, 1)

        mu = torch.zeros(1, out_channels, 1, 1)
        sigma = torch.ones(1, out_channels, 1, 1)
        for idx, code in enumerate(scaling_codes):
            if code == SCALING_DIVIDE_ONLY:
                mu[0, idx] = 0.0
                sigma[0, idx] = 2.0
            else:
                mu[0, idx] = 0.5 * (idx + 1)
                sigma[0, idx] = 1.5

        self.output_scalers_mu = nn.Parameter(mu, requires_grad=False)
        self.output_scalers_sigma = nn.Parameter(sigma, requires_grad=False)
        self.register_buffer(
            "predictand_scaling_method_codes",
            torch.tensor(scaling_codes, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "predictand_nonneg_enabled_mask",
            torch.tensor(nonneg, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "predictand_nonneg_method_codes",
            torch.tensor([1 if flag else 0 for flag in nonneg], dtype=torch.int64),
            persistent=False,
        )
        self._capture_features: tuple[str, ...] = ()
        self._last_phase1_features: dict[str, torch.Tensor] = {}
        self.n_input_timestamps = 1
        self.input_scalers_epsilon = 1.0e-6

    # -- Phase-1 contract -------------------------------------------------
    def _resolve_output_scalers(self, reference, scaler_offset=None):
        mu = self.output_scalers_mu.to(device=reference.device, dtype=reference.dtype)
        sigma = self.output_scalers_sigma.to(device=reference.device, dtype=reference.dtype)
        return mu, sigma

    def _resolve_input_scalers(self, reference, scaler_offset=None):
        channels = reference.shape[1]
        mu = torch.zeros(1, channels, 1, 1, device=reference.device, dtype=reference.dtype)
        sigma = torch.ones_like(mu)
        return mu, sigma

    def set_feature_capture(self, names):
        self._capture_features = tuple(names or ())
        self._last_phase1_features = {}

    def get_last_phase1_features(self):
        return dict(self._last_phase1_features)

    def clear_last_phase1_features(self):
        self._last_phase1_features = {}

    def _decode(self, normalized):
        mu = self.output_scalers_mu.to(normalized.device)
        sigma = self.output_scalers_sigma.to(normalized.device)
        codes = self.predictand_scaling_method_codes.view(1, -1, 1, 1)
        out = normalized * sigma + mu
        out = torch.where(codes == SCALING_DIVIDE_ONLY, normalized * sigma, out)
        out = torch.where(
            codes == SCALING_LOG1P_ZSCORE, torch.expm1(normalized * sigma + mu), out
        )
        return out

    def forward(self, batch, return_pre_inverse: bool = False, return_raw_output: bool = False):
        x = batch["x"]
        target_hw = batch["y"].shape[-2:] if "y" in batch else x.shape[-2:]
        feats = self.body(x)
        if self._capture_features and "prithvi" in self._capture_features:
            self._last_phase1_features["prithvi"] = feats
        raw = self.head(feats)
        if raw.shape[-2:] != tuple(target_hw):
            raw = torch.nn.functional.interpolate(
                raw, size=tuple(target_hw), mode="bilinear", align_corners=False
            )
        if self._capture_features and "unet" in self._capture_features:
            self._last_phase1_features["unet"] = torch.nn.functional.interpolate(
                feats, size=tuple(target_hw), mode="bilinear", align_corners=False
            )
        # Emulate the real non-negativity output link on constrained channels.
        constrained = torch.where(
            self.predictand_nonneg_enabled_mask.view(1, -1, 1, 1),
            torch.nn.functional.softplus(raw),
            raw,
        )
        physical = self._decode(constrained)
        if return_pre_inverse and return_raw_output:
            return physical, constrained, raw
        if return_pre_inverse:
            return physical, constrained
        if return_raw_output:
            return physical, raw
        return physical


def make_batch(
    batch_size: int = 2,
    in_channels: int = 4,
    out_channels: int = 3,
    height: int = 24,
    width: int = 32,
    *,
    seed: int = 0,
    with_static: bool = True,
    nan_fraction: float = 0.0,
):
    """Build a synthetic batch whose predictors and targets share one timestamp."""
    g = torch.Generator().manual_seed(seed)
    batch = {
        "x": torch.randn(batch_size, in_channels, height, width, generator=g),
        "y": torch.randn(batch_size, out_channels, height, width, generator=g).abs(),
    }
    if with_static:
        batch["static_x"] = torch.randn(batch_size, 1, height, width, generator=g)
        batch["static_y"] = torch.randn(batch_size, 1, height, width, generator=g)
    if nan_fraction > 0:
        mask = torch.rand(batch["y"].shape, generator=g) < nan_fraction
        batch["y"] = batch["y"].masked_fill(mask, float("nan"))
    return batch
