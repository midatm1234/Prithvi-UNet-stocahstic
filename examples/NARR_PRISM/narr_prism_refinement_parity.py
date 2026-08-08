"""Deterministic Phase-1 parity check against an existing NARR_PRISM checkpoint.

Loads the *unmodified* NARR_PRISM deterministic checkpoint twice:

1. into the original ``ClimateDownscaleFinetuneUNETModel`` (baseline), and
2. into :class:`~granitewxc.refinement.two_phase.TwoPhaseDownscalingModel` via
   :func:`granitewxc.refinement.checkpoint.load_phase1_state_dict`,

then compares the raw normalized deterministic output (before any Phase-2
operation) and the final deterministic physical-space output.

The checkpoint is opened read-only and is never rewritten.

Usage::

    python examples/NARR_PRISM/narr_prism_refinement_parity.py \
        --config examples/NARR_PRISM/NARR_PRISM_subdomain.yaml \
        --checkpoint examples/NARR_PRISM/experiments/checkpoints/narr_prism_California/last.ckpt \
        --device cuda:1 --size 256
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from granitewxc.models.model import get_finetune_model_UNET  # noqa: E402
from granitewxc.refinement import build_two_phase_model  # noqa: E402
from granitewxc.refinement.checkpoint import (  # noqa: E402
    extract_model_state,
    load_phase1_state_dict,
    migrate_phase1_state_dict,
    phase1_state_fingerprint,
)
from granitewxc.utils.config import get_config  # noqa: E402
from granitewxc.utils.normalization import (  # noqa: E402
    apply_scalar_paths,
    assert_scalars_available,
)


def _prepare_config(config):
    """Fill in the derived data fields the model builder expects."""
    for attr in (
        "input_static_surface_vars",
        "input_surface_vars",
        "other",
        "vertical_level1_vars",
        "input_level1",
        "vertical_level2_vars",
        "input_level2",
    ):
        if not hasattr(config.data, attr):
            setattr(config.data, attr, [])
    if not getattr(config.data, "output_vars", None):
        config.data.output_vars = list(getattr(config.data, "target_variables", []))
    # The training entry point rewires the model/dataset onto the case-scoped
    # scalers before building the model. Do the same here so the constructed
    # scaler buffers have exactly the shapes stored in the checkpoint.
    assert_scalars_available(config, role="parity")
    apply_scalar_paths(config)
    return config


def _n_input_channels(config) -> int:
    data = config.data
    return (
        len(data.input_surface_vars)
        + len(data.other)
        + len(data.vertical_pres_vars) * len(data.input_level_pres)
        + len(data.vertical_level1_vars) * len(data.input_level1)
        + len(data.vertical_level2_vars) * len(data.input_level2)
    ) * int(getattr(data, "n_input_timestamps", 1))


def _stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a32, b32 = a.detach().float().cpu(), b.detach().float().cpu()
    diff = (a32 - b32).abs()
    denom = b32.abs().clamp(min=1e-12)
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "max_rel_diff": float((diff / denom).max()),
        "bitwise_identical": bool(torch.equal(a32, b32)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--size", type=int, default=256, help="square tile size for the probe batch")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--tolerance", type=float, default=0.0, help="absolute tolerance (0 = bitwise)")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    config = _prepare_config(get_config(args.config))

    print(f"[parity] config        : {args.config}")
    print(f"[parity] checkpoint    : {args.checkpoint}")
    print(f"[parity] device        : {device}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    raw_state = extract_model_state(checkpoint)
    fingerprint = phase1_state_fingerprint(raw_state)
    migrated, renames = migrate_phase1_state_dict(raw_state)
    print(f"[parity] phase-1 keys  : {len(raw_state)}")
    print(f"[parity] fingerprint   : {fingerprint}")
    print(f"[parity] key renames   : {len(renames)} (all 'X' -> 'phase1.X')")

    # -- baseline: the original deterministic model -----------------------
    baseline = get_finetune_model_UNET(copy.deepcopy(config))
    missing, unexpected = baseline.load_state_dict(raw_state, strict=False)
    unexplained = [k for k in missing if "scaler" not in k]
    if unexplained or unexpected:
        raise RuntimeError(
            f"Baseline load is not clean: missing={unexplained[:5]} unexpected={list(unexpected)[:5]}"
        )
    baseline.to(device).eval()

    # -- new branch: the two-phase wrapper --------------------------------
    phase1 = get_finetune_model_UNET(copy.deepcopy(config))
    wrapper = build_two_phase_model(phase1, config)
    report = load_phase1_state_dict(wrapper, checkpoint)
    print(f"[parity] wrapper load  : {report.summary()}")
    wrapper.to(device).eval()

    torch.manual_seed(args.seed)
    channels = _n_input_channels(config)
    n_out = len(config.data.output_vars)
    size = int(args.size)
    batch = {
        "x": torch.randn(1, channels, size, size, device=device),
        "y": torch.randn(1, n_out, size, size, device=device).abs(),
        # Grid-point target scalers cover the full domain; a tile probe must
        # declare its origin so the matching scaler slice is used.
        "__input_scaler_offset": (0, 0),
        "__output_scaler_offset": (0, 0),
    }

    with torch.no_grad():
        base_physical, base_normalized = baseline(dict(batch), return_pre_inverse=True)
        new_physical, new_normalized, _ = wrapper.run_phase1(dict(batch))

    norm_stats = _stats(new_normalized, base_normalized)
    phys_stats = _stats(new_physical, base_physical)

    result = {
        "config": os.path.abspath(args.config),
        "checkpoint": os.path.abspath(args.checkpoint),
        "phase1_fingerprint": fingerprint,
        "device": str(device),
        "input_shape": list(batch["x"].shape),
        "output_shape": list(base_physical.shape),
        "key_renames": len(renames),
        "loaded_tensors": report.loaded,
        "missing_keys_phase2_only": len(report.missing),
        "normalized_space": norm_stats,
        "physical_space": phys_stats,
        "tolerance": args.tolerance,
    }
    print(json.dumps(result, indent=2))

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(result, indent=2))

    ok = (
        norm_stats["max_abs_diff"] <= args.tolerance
        and phys_stats["max_abs_diff"] <= args.tolerance
    )
    print("[parity] RESULT:", "PASS (bitwise identical)" if norm_stats["bitwise_identical"] and phys_stats["bitwise_identical"] else ("PASS (within tolerance)" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
