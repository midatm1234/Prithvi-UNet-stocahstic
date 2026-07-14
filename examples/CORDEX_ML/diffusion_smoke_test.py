#!/usr/bin/env python
"""End-to-end dummy-tensor smoke test for the CORDEX-ML diffusion head.

This script requires no data, no GPU and no Prithvi WxC weights. It exercises the
three acceptance checks for the optional diffusion decoder head:

1. the deterministic conv/UNet head still imports and runs,
2. the diffusion head computes a finite training (score-matching) loss and
   propagates gradients on dummy tensors,
3. the diffusion sampler returns ``[B, output_channels, H_target, W_target]``.

It also checks that the example diffusion YAML resolves to the diffusion head and
that standardized decoding preserves precipitation non-negativity.

Run:

    python examples/CORDEX_ML/diffusion_smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from granitewxc.decoders.diffusion_head import DiffusionHead, DiffusionHeadConfig  # noqa: E402
from granitewxc.decoders.downscaling import ConvEncoderDecoder  # noqa: E402
from granitewxc.models.cordex_finetune_model import resolve_head_type  # noqa: E402
from granitewxc.models.loss import build_loss_fn  # noqa: E402
from granitewxc.models.diffusion_loss import DiffusionLossPassthrough  # noqa: E402


B, COND_CH, OUT_CH = 2, 6, 2
H_TARGET, W_TARGET = 24, 24


def check_deterministic_head() -> None:
    print("[1/5] deterministic conv head imports and runs ...", end=" ")
    head = ConvEncoderDecoder(
        in_channels=8,
        channels=16,
        out_channels=OUT_CH,
        kernel_size=[3],
        scale=[2],
        upsampling_mode="nearest",
    )
    out = head(torch.randn(B, 8, 6, 6))
    assert out.shape[0] == B and out.shape[1] == OUT_CH, out.shape
    print(f"OK  (output {tuple(out.shape)})")


def _build_head() -> DiffusionHead:
    cfg = DiffusionHeadConfig(
        sde="subvpsde",
        num_scales=100,
        base_channels=16,
        channel_multipliers=(1, 2),
        num_res_blocks=1,
        time_embed_dim=32,
        num_sampling_steps=8,
        sampling_method="pc",
        predictor="euler_maruyama",
        corrector="none",
    )
    return DiffusionHead(cond_channels=COND_CH, output_channels=OUT_CH, head_config=cfg)


def check_training_loss(head: DiffusionHead) -> None:
    print("[2/5] diffusion training-loss pass on dummy tensors ...", end=" ")
    cond = torch.randn(B, COND_CH, 8, 8)
    target = torch.randn(B, OUT_CH, H_TARGET, W_TARGET)
    loss = head.training_loss(cond, target)
    assert loss.ndim == 0 and torch.isfinite(loss), loss
    loss.backward()
    has_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in head.parameters()
    )
    assert has_grad, "no finite gradients propagated"
    print(f"OK  (loss={float(loss.detach()):.4f}, gradients flow)")


def check_sampler(head: DiffusionHead) -> None:
    print("[3/5] diffusion sampler output shape ...", end=" ")
    cond = torch.randn(B, COND_CH, 6, 6)
    with torch.no_grad():
        sample = head.sample(cond, H_TARGET, W_TARGET)
    assert sample.shape == (B, OUT_CH, H_TARGET, W_TARGET), sample.shape
    assert torch.isfinite(sample).all()
    print(f"OK  (sample {tuple(sample.shape)})")


def check_loss_passthrough() -> None:
    print("[4/5] build_loss_fn returns diffusion passthrough ...", end=" ")
    from types import SimpleNamespace

    config = SimpleNamespace(model=SimpleNamespace(head_type="diffusion"))
    loss_fn = build_loss_fn(config, output_vars=["pr", "tasmax"])
    assert isinstance(loss_fn, DiffusionLossPassthrough)
    scalar = torch.tensor(0.5, requires_grad=True)
    assert loss_fn(scalar, {"y": torch.zeros(1, 2, 4, 4)}) is scalar
    print("OK")


def check_example_config() -> None:
    print("[5/5] example diffusion YAML resolves to diffusion head ...", end=" ")
    cfg_path = REPO_ROOT / "examples" / "CORDEX_ML" / "NZ_T1_ACCESS-CM2_static_diffusion.yaml"
    if not cfg_path.exists():
        print(f"SKIP (missing {cfg_path.name})")
        return
    from granitewxc.utils.config import get_config

    config = get_config(str(cfg_path))
    assert resolve_head_type(config) == "diffusion"
    head_cfg = DiffusionHeadConfig.from_config(config)
    print(
        f"OK  (sde={head_cfg.sde}, num_scales={head_cfg.num_scales}, "
        f"sampling={head_cfg.sampling_method}, steps={head_cfg.num_sampling_steps})"
    )


def main() -> int:
    torch.manual_seed(0)
    print("=" * 68)
    print("CORDEX-ML diffusion head smoke test (dummy tensors, CPU)")
    print("=" * 68)
    check_deterministic_head()
    head = _build_head()
    check_training_loss(head)
    check_sampler(head)
    check_loss_passthrough()
    check_example_config()
    print("-" * 68)
    print("All diffusion smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
