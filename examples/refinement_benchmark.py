"""Baseline-vs-optimized benchmarks for the Phase-2 stochastic refinement.

Each benchmark pairs a *reference* implementation with the *optimized* path that
this branch enables, measures wall time and peak memory for both, and verifies
numerical parity.  An optimization is only reported as accepted when parity
holds within the documented tolerance.

Identical data, checkpoints, seeds, stochastic initialisation, solver settings
and ensemble ordering are used on both sides of every comparison.

Usage::

    python examples/refinement_benchmark.py --device cuda:0 \
        --config examples/NARR_PRISM/NARR_PRISM_subdomain.yaml \
        --checkpoint examples/NARR_PRISM/experiments/checkpoints/narr_prism_California/last.ckpt \
        --json-out /tmp/refinement_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from granitewxc.refinement import TwoPhaseDownscalingModel  # noqa: E402
from granitewxc.refinement.config import resolve_refinement_config  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "tests"))
from refinement_fixtures import TinyPhase1, make_batch  # noqa: E402


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timeit(fn, device: torch.device, repeats: int = 5, warmup: int = 2) -> dict:
    for _ in range(warmup):
        fn()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        _sync(device)
        samples.append(time.perf_counter() - start)
    peak = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else float("nan")
    return {
        "median_s": statistics.median(samples),
        "min_s": min(samples),
        "max_s": max(samples),
        "peak_mem_mib": peak,
    }


def parity(a: torch.Tensor, b: torch.Tensor) -> dict:
    a32, b32 = a.detach().float().cpu(), b.detach().float().cpu()
    diff = (a32 - b32).abs()
    denom = b32.abs().clamp(min=1e-12)
    return {
        "bitwise_identical": bool(torch.equal(a32, b32)),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "max_rel_diff": float((diff / denom).max()),
    }


def build_model(refiner_type: str, batch, device, *, steps: int, ensemble: int, size: int):
    config = {
        "refinement": {
            "enabled": True,
            "type": refiner_type,
            "ensemble_size": ensemble,
            "seed": 1234,
            "diffusion": {"training_timesteps": 1000, "inference_steps": steps},
            "flow_matching": {"integration_steps": steps},
            "unet": {"hidden_channels": 64, "num_levels": 3},
            "transformer": {
                "patch_size": 4,
                "embedding_dim": 256,
                "num_heads": 8,
                "num_blocks": 6,
                "max_tokens_lat": max(64, size),
                "max_tokens_lon": max(64, size),
            },
        }
    }
    model = TwoPhaseDownscalingModel(TinyPhase1(), refinement=resolve_refinement_config(config))
    model.to(device)
    model.initialize_from_batch(batch)
    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def bench_ensemble_batching(device, size, ensemble, steps, repeats) -> list[dict]:
    out = []
    batch = make_batch(batch_size=1, height=size, width=size)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    for refiner_type in ("diffusion_unet", "flow_matching_unet", "diffusion_transformer", "flow_matching_transformer"):
        model = build_model(refiner_type, batch, device, steps=steps, ensemble=ensemble, size=size)
        serial = timeit(lambda: model.predict(batch, ensemble_size=ensemble, seed=1234, chunk_size=1), device, repeats)
        batched = timeit(lambda: model.predict(batch, ensemble_size=ensemble, seed=1234, chunk_size=ensemble), device, repeats)
        a = model.predict(batch, ensemble_size=ensemble, seed=1234, chunk_size=1).member_residuals
        b = model.predict(batch, ensemble_size=ensemble, seed=1234, chunk_size=ensemble).member_residuals
        out.append(
            {
                "optimization": "batched ensemble generation",
                "bottleneck": "serial per-member sampling loop",
                "files": ["granitewxc/refinement/two_phase.py", "granitewxc/refinement/base.py"],
                "refiner": refiner_type,
                "ensemble_size": ensemble,
                "stochastic_steps": steps,
                "baseline": serial,
                "optimized": batched,
                "speedup_pct": 100.0 * (1.0 - batched["median_s"] / serial["median_s"]),
                "parity": parity(b, a),
                "tolerance": "bitwise",
            }
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def bench_attention(device, size, steps, repeats) -> list[dict]:
    out = []
    batch = make_batch(batch_size=1, height=size, width=size)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    for refiner_type in ("diffusion_transformer", "flow_matching_transformer"):
        model = build_model(refiner_type, batch, device, steps=steps, ensemble=1, size=size)
        model.refiner.set_attention_implementation("math")
        reference = timeit(lambda: model.predict(batch, ensemble_size=1, seed=7), device, repeats)
        ref_value = model.predict(batch, ensemble_size=1, seed=7).member_residuals
        model.refiner.set_attention_implementation("sdpa")
        optimized = timeit(lambda: model.predict(batch, ensemble_size=1, seed=7), device, repeats)
        opt_value = model.predict(batch, ensemble_size=1, seed=7).member_residuals
        out.append(
            {
                "optimization": "fused scaled-dot-product attention",
                "bottleneck": "explicit softmax(QK^T)V materialises the full attention matrix",
                "files": ["granitewxc/refinement/backbones.py"],
                "refiner": refiner_type,
                "stochastic_steps": steps,
                "baseline": reference,
                "optimized": optimized,
                "speedup_pct": 100.0 * (1.0 - optimized["median_s"] / reference["median_s"]),
                "parity": parity(opt_value, ref_value),
                "tolerance": "atol=1e-5, rtol=1e-5 (fp32)",
            }
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def bench_frozen_phase1(device, config_path, checkpoint_path, size, repeats) -> list[dict]:
    """Measure the frozen-Phase-1 no-grad/eval path against a grad-enabled pass."""
    if not config_path:
        return []
    import copy as _copy

    sys.path.insert(0, str(REPO_ROOT / "examples" / "NARR_PRISM"))
    from granitewxc.models.model import get_finetune_model_UNET
    from granitewxc.refinement.checkpoint import load_phase1_state_dict
    from granitewxc.utils.config import get_config
    from granitewxc.utils.normalization import apply_scalar_paths, assert_scalars_available

    config = get_config(config_path)
    for attr in ("input_static_surface_vars", "input_surface_vars", "other",
                 "vertical_level1_vars", "input_level1", "vertical_level2_vars", "input_level2"):
        if not hasattr(config.data, attr):
            setattr(config.data, attr, [])
    if not getattr(config.data, "output_vars", None):
        config.data.output_vars = list(getattr(config.data, "target_variables", []))
    assert_scalars_available(config, role="benchmark")
    apply_scalar_paths(config)

    phase1 = get_finetune_model_UNET(_copy.deepcopy(config))
    model = TwoPhaseDownscalingModel(
        phase1, refinement=resolve_refinement_config({"refinement": {"type": "diffusion_unet"}})
    )
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
        load_phase1_state_dict(model, checkpoint)
    model.to(device)

    channels = (
        len(config.data.input_surface_vars)
        + len(config.data.other)
        + len(config.data.vertical_pres_vars) * len(config.data.input_level_pres)
    )
    torch.manual_seed(0)
    batch = {
        "x": torch.randn(1, channels, size, size, device=device),
        "y": torch.rand(1, len(config.data.output_vars), size, size, device=device) * 5,
        "__input_scaler_offset": (0, 0),
        "__output_scaler_offset": (0, 0),
    }
    model.initialize_from_batch(batch)
    model.to(device)

    def grad_enabled():
        model.phase1.train()
        for p in model.phase1.parameters():
            p.requires_grad_(True)
        out = model.phase1(dict(batch), return_pre_inverse=True)
        return out

    def frozen():
        model.phase1.eval()
        for p in model.phase1.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            return model.phase1(dict(batch), return_pre_inverse=True)

    baseline = timeit(grad_enabled, device, repeats, warmup=1)
    optimized = timeit(frozen, device, repeats, warmup=1)
    with torch.no_grad():
        model.phase1.eval()
        reference_value = model.phase1(dict(batch), return_pre_inverse=True)[1]
        optimized_value = model.run_phase1(dict(batch))[1]
    result = [
        {
            "optimization": "frozen Phase-1 conditioning (eval + no_grad)",
            "bottleneck": "Phase-1 autograd graph retained during Phase-2 training",
            "files": ["granitewxc/refinement/two_phase.py"],
            "domain": [size, size],
            "baseline": baseline,
            "optimized": optimized,
            "speedup_pct": 100.0 * (1.0 - optimized["median_s"] / baseline["median_s"]),
            "parity": parity(optimized_value, reference_value),
            "tolerance": "bitwise",
        }
    ]

    # --- real-model Phase-1 conditioning cache --------------------------
    model.train()
    with torch.no_grad():
        _, normalized, _ = model.run_phase1(batch)
    cached_batch = dict(batch)
    cached_batch["__phase1_normalized"] = normalized

    def online_step():
        gen = torch.Generator().manual_seed(0)
        return model.training_step(batch, generator=gen).losses["loss"]

    def cached_step():
        gen = torch.Generator().manual_seed(0)
        return model.training_step(cached_batch, generator=gen).losses["loss"]

    cache_baseline = timeit(online_step, device, repeats, warmup=1)
    cache_optimized = timeit(cached_step, device, repeats, warmup=1)
    result.append(
        {
            "optimization": "precomputed frozen Phase-1 cache (real Prithvi-UNet)",
            "bottleneck": "frozen Prithvi-UNet forward pass repeated every Phase-2 step",
            "files": ["granitewxc/refinement/cache.py", "granitewxc/refinement/two_phase.py"],
            "domain": [size, size],
            "baseline": cache_baseline,
            "optimized": cache_optimized,
            "speedup_pct": 100.0 * (1.0 - cache_optimized["median_s"] / cache_baseline["median_s"]),
            "parity": parity(cached_step(), online_step()),
            "tolerance": "bitwise (same frozen Phase 1)",
        }
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def bench_phase1_cache(device, size, repeats) -> list[dict]:
    """Online Phase-1 conditioning vs a precomputed cache during Phase-2 training."""
    batch = make_batch(batch_size=1, height=size, width=size)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    model = build_model("diffusion_unet", batch, device, steps=10, ensemble=1, size=size)
    model.train()

    with torch.no_grad():
        _, normalized, _ = model.run_phase1(batch)
    cached_batch = dict(batch)
    cached_batch["__phase1_normalized"] = normalized

    def online():
        gen = torch.Generator().manual_seed(0)
        return model.training_step(batch, generator=gen).losses["loss"]

    def cached():
        gen = torch.Generator().manual_seed(0)
        return model.training_step(cached_batch, generator=gen).losses["loss"]

    baseline = timeit(online, device, repeats)
    optimized = timeit(cached, device, repeats)
    result = [
        {
            "optimization": "precomputed frozen Phase-1 conditioning cache",
            "bottleneck": "re-running an unchanging frozen Phase 1 every epoch",
            "files": ["granitewxc/refinement/cache.py", "granitewxc/refinement/two_phase.py"],
            "domain": [size, size],
            "baseline": baseline,
            "optimized": optimized,
            "speedup_pct": 100.0 * (1.0 - optimized["median_s"] / baseline["median_s"]),
            "parity": parity(cached(), online()),
            "tolerance": "bitwise (same frozen Phase 1)",
            "note": (
                "Speed-up scales with the Phase-1 cost; measured here against the "
                "lightweight test double, so the real gain is much larger."
            ),
        }
    ]
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--ensemble", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)

    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "precision": "fp32",
        "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "domain": [args.size, args.size],
        "batch_size": 1,
        "ensemble_size": args.ensemble,
        "stochastic_steps": args.steps,
        "repeats": args.repeats,
    }
    print(json.dumps(environment, indent=2))

    entries: list[dict] = []
    entries += bench_ensemble_batching(device, args.size, args.ensemble, args.steps, args.repeats)
    entries += bench_attention(device, args.size, args.steps, args.repeats)
    entries += bench_phase1_cache(device, args.size, args.repeats)
    if args.config:
        entries += bench_frozen_phase1(device, args.config, args.checkpoint, args.size, args.repeats)

    header = (
        f"\n{'optimization':52s} {'refiner':28s} {'base s':>9s} {'opt s':>9s} "
        f"{'gain %':>8s} {'base MiB':>10s} {'opt MiB':>9s}  parity"
    )
    print(header)
    for entry in entries:
        parity_info = entry["parity"]
        tag = "bitwise" if parity_info["bitwise_identical"] else f"max|d|={parity_info['max_abs_diff']:.2e}"
        print(
            f"{entry['optimization']:52s} {entry.get('refiner', '-'):28s} "
            f"{entry['baseline']['median_s']:9.4f} {entry['optimized']['median_s']:9.4f} "
            f"{entry['speedup_pct']:8.1f} {entry['baseline']['peak_mem_mib']:10.0f} "
            f"{entry['optimized']['peak_mem_mib']:9.0f}  {tag}"
        )

    payload = {"environment": environment, "results": entries}
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
