"""Non-destructive smoke tests for the two-phase Prithvi-UNet refinement.

For a given case configuration this script builds the real Phase-1
Prithvi-UNet, optionally loads an existing deterministic checkpoint (read-only),
and then, for each requested Phase-2 refinement type, runs:

* one deterministic Phase-1 inference pass,
* one Phase-2 training step (forward + backward + optimizer step), and
* one Phase-2 inference pass (small ensemble).

Nothing is written to disk unless ``--save-dir`` is given, and no existing
checkpoint, dataset or configuration is modified. Stochastic step counts are
overridden here (``--steps``) purely to keep the smoke test fast; the production
example configurations keep their full settings.

Examples::

    # NARR_PRISM against the checkpoint that is currently training
    python examples/refinement_smoke_test.py \
        --config examples/NARR_PRISM/NARR_PRISM_subdomain.yaml \
        --checkpoint examples/NARR_PRISM/experiments/checkpoints/narr_prism_California/last.ckpt \
        --device cuda:0 --size 128

    # CORDEX_ML / MERRA_PRISM (no checkpoint required)
    python examples/refinement_smoke_test.py \
        --config examples/CORDEX_ML/NZ_T1_ACCESS-CM2_static.yaml --device cpu --size 64
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from granitewxc.models.model import get_finetune_model_UNET  # noqa: E402
from granitewxc.refinement import TwoPhaseDownscalingModel  # noqa: E402
from granitewxc.refinement.checkpoint import load_phase1_state_dict  # noqa: E402
from granitewxc.refinement.config import (  # noqa: E402
    resolve_performance_config,
    resolve_refinement_config,
)
from granitewxc.utils.config import get_config  # noqa: E402

REFINERS = (
    "diffusion_unet",
    "flow_matching_unet",
    "diffusion_transformer",
    "flow_matching_transformer",
)


def prepare_config(path: str, synthetic_scalers: str | None = None):
    config = get_config(path)
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
    if not getattr(config.data, "input_levels", None):
        config.data.input_levels = [1]
    if synthetic_scalers:
        _write_synthetic_scalers(config, synthetic_scalers)
        return config
    try:
        from granitewxc.utils.normalization import apply_scalar_paths, assert_scalars_available

        assert_scalars_available(config, role="smoke")
        apply_scalar_paths(config)
    except Exception as exc:  # pragma: no cover - optional for non-PRISM cases
        print(f"[smoke] note: case-scoped scalers unavailable ({exc}); using YAML paths")
    return config


def _write_synthetic_scalers(config, directory: str) -> None:
    """Write placeholder scaler files so a case can be smoke tested without data.

    Used only when the real scaler artefacts are not present on this machine.
    It exercises the model wiring (channel counts, shapes, Phase-2 conditioning)
    and never touches any existing scaler file.
    """
    import numpy as np

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    n_in = n_input_channels(config)
    n_out = len(config.data.output_vars)
    files = {
        "inputs_mean.npy": np.zeros(n_in, dtype="float32"),
        "inputs_std.npy": np.ones(n_in, dtype="float32"),
        "targets_mean.npy": np.zeros(n_out, dtype="float32"),
        "targets_std.npy": np.ones(n_out, dtype="float32"),
    }
    for name, array in files.items():
        np.save(target / name, array)
    config.model.input_mu = str(target / "inputs_mean.npy")
    config.model.input_sigma = str(target / "inputs_std.npy")
    config.model.target_mu = str(target / "targets_mean.npy")
    config.model.target_sigma = str(target / "targets_std.npy")
    config.data.scalar_dir = str(target)
    config.data.scalers = {
        "inputs_mean": config.model.input_mu,
        "inputs_std": config.model.input_sigma,
        "targets_mean": config.model.target_mu,
        "targets_std": config.model.target_sigma,
    }
    print(f"[smoke] using synthetic scalers in {target} (n_in={n_in}, n_out={n_out})")


def n_input_channels(config) -> int:
    data = config.data
    return (
        len(data.input_surface_vars)
        + len(data.other)
        + len(data.vertical_pres_vars) * len(data.input_level_pres)
        + len(data.vertical_level1_vars) * len(data.input_level1)
        + len(data.vertical_level2_vars) * len(data.input_level2)
    ) * int(getattr(data, "n_input_timestamps", 1))


def make_batch(config, size: int, device: torch.device, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    channels = n_input_channels(config)
    n_out = len(config.data.output_vars)
    batch = {
        "x": torch.randn(1, channels, size, size, device=device),
        # A single timestamp for predictors *and* target: this is a spatial
        # downscaling problem, there is no lead time.
        "y": torch.rand(1, n_out, size, size, device=device) * 5.0,
        "__input_scaler_offset": (0, 0),
        "__output_scaler_offset": (0, 0),
    }
    use_static = bool(getattr(config.data, "use_static", False))
    n_static = int(getattr(config.model, "num_static_channels", 0))
    if use_static and n_static > 0:
        batch["static_x"] = torch.randn(1, n_static, size, size, device=device)
        batch["static_y"] = torch.randn(1, n_static, size, size, device=device)
    return batch


def refinement_overrides(refiner_type: str, steps: int, size: int) -> dict:
    patch = 4 if size % 4 == 0 else 1
    return {
        "enabled": True,
        "type": refiner_type,
        "ensemble_size": 2,
        "freeze_phase1": True,
        "train_on_residual": True,
        "seed": 1234,
        "conditioning": {
            "deterministic_output": True,
            "input_predictors": True,
            "prithvi_features": False,
            "unet_features": False,
            "static_fields": True,
            "masks": True,
        },
        # Smoke-test-only step counts. Production configs keep 1000/50/50.
        "diffusion": {"training_timesteps": 100, "inference_steps": steps},
        "flow_matching": {"integration_steps": steps},
        "unet": {"hidden_channels": 16, "num_levels": 2, "attention_heads": 4},
        "transformer": {
            "patch_size": patch,
            "embedding_dim": 64,
            "num_heads": 4,
            "num_blocks": 2,
            "max_tokens_lat": max(64, size),
            "max_tokens_lon": max(64, size),
        },
    }


def peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return float("nan")
    return torch.cuda.max_memory_allocated(device) / (1024**2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="existing deterministic Phase-1 checkpoint (read-only)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=3, help="smoke-test stochastic step count")
    parser.add_argument("--ensemble", type=int, default=2)
    parser.add_argument("--refiners", nargs="*", default=list(REFINERS))
    parser.add_argument("--json-out", default=None)
    parser.add_argument(
        "--synthetic-scalers",
        default=None,
        help=(
            "directory for placeholder scaler files; use only when the real "
            "scaler artefacts for the case are not available on this machine"
        ),
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    config = prepare_config(args.config, synthetic_scalers=args.synthetic_scalers)
    results: list[dict] = []

    print(f"[smoke] config     : {args.config}")
    print(f"[smoke] case_name  : {getattr(config, 'case_name', '?')}")
    print(f"[smoke] device     : {device}  size={args.size}")

    checkpoint = None
    if args.checkpoint:
        print(f"[smoke] checkpoint : {args.checkpoint} (read-only)")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)

    # --- deterministic Phase-1 inference --------------------------------
    phase1 = get_finetune_model_UNET(copy.deepcopy(config))
    deterministic = TwoPhaseDownscalingModel(
        phase1,
        refinement=resolve_refinement_config(config),
        performance=resolve_performance_config(config),
    )
    if checkpoint is not None:
        report = load_phase1_state_dict(deterministic, checkpoint)
        print(f"[smoke] phase-1 load: {report.summary()}")
    deterministic.to(device).eval()
    batch = make_batch(config, args.size, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with torch.no_grad():
        out = deterministic.predict(batch, ensemble_size=0)
    elapsed = time.perf_counter() - start
    print(
        f"[smoke] deterministic inference OK  shape={tuple(out.deterministic.shape)}  "
        f"{elapsed:.3f}s  peak={peak_memory_mb(device):.0f} MiB"
    )
    results.append(
        {
            "refiner": "none",
            "deterministic_inference_s": elapsed,
            "output_shape": list(out.deterministic.shape),
            "peak_mem_mib": peak_memory_mb(device),
            "finite": bool(torch.isfinite(out.deterministic).all()),
        }
    )
    del deterministic, phase1
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- one training + one inference step per refiner -------------------
    for refiner_type in args.refiners:
        cfg = copy.deepcopy(config)
        cfg.model.refinement = refinement_overrides(refiner_type, args.steps, args.size)
        phase1 = get_finetune_model_UNET(copy.deepcopy(config))
        model = TwoPhaseDownscalingModel(
            phase1,
            refinement=resolve_refinement_config(cfg),
            performance=resolve_performance_config(cfg),
        )
        load_report = None
        if checkpoint is not None:
            report = load_phase1_state_dict(model, checkpoint)
            unexplained = [k for k in report.missing if not k.startswith("refiner.")]
            assert not unexplained, unexplained
            load_report = report.summary()
        model.to(device)
        batch = make_batch(cfg, args.size, device)
        model.initialize_from_batch(batch)
        model.to(device)

        optimizer = torch.optim.AdamW(
            [p for p in model.refiner.parameters() if p.requires_grad], lr=1e-4
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        model.train()
        start = time.perf_counter()
        step_out = model.training_step(batch, generator=torch.Generator().manual_seed(0))
        loss = step_out.losses["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        train_time = time.perf_counter() - start

        model.eval()
        start = time.perf_counter()
        pred = model.predict(batch, ensemble_size=args.ensemble, seed=1234)
        infer_time = time.perf_counter() - start

        entry = {
            "refiner": refiner_type,
            "phase1_load": load_report,
            "loss": float(loss.detach()),
            "train_step_s": train_time,
            "ensemble_inference_s": infer_time,
            "ensemble_size": args.ensemble,
            "members_shape": list(pred.members.shape),
            "spread_finite_fraction": float(torch.isfinite(pred.ensemble_spread).float().mean()),
            "refined_finite": bool(torch.isfinite(pred.refined).all()),
            "refiner_parameters": sum(p.numel() for p in model.refiner.parameters()),
            "conditioning_channels": model.refiner.cond_channels,
            "peak_mem_mib": peak_memory_mb(device),
        }
        results.append(entry)
        print(
            f"[smoke] {refiner_type:28s} loss={entry['loss']:.5f} "
            f"train={train_time:.3f}s infer={infer_time:.3f}s "
            f"params={entry['refiner_parameters']:,} cond_ch={entry['conditioning_channels']} "
            f"peak={entry['peak_mem_mib']:.0f} MiB"
        )
        assert entry["refined_finite"], f"{refiner_type} produced non-finite output"
        del model, phase1, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(results, indent=2))
        print(f"[smoke] wrote {args.json_out}")
    print("[smoke] ALL PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
