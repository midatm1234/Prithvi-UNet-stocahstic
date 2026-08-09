"""Deterministic tiny-data overfit gate for residual diffusion.

This is an intentionally small, fast end-to-end diagnostic.  It trains the
real :class:`DiffusionHead` epsilon parameterization on one example and on a
four-example batch, then reconstructs the residual with the real DDIM reverse
sampler from several independent prior seeds. Training uses the production
``DiffusionHead.training_loss`` path: epsilon score matching plus the configured
clean-residual (x0) reconstruction term whose inverse-SNR amplification is
capped for finite training gradients. The command exits non-zero if the sampler
cannot recover the memorized residuals or if applying the learned correction
fails to improve the deterministic baseline.

The synthetic conditioning deliberately includes the residual pattern.  This
makes the gate an integration/memorization test of the production objective and
reverse sampler, rather than a test of whether a tiny U-Net can discover a
forecast relationship from four synthetic examples.

Run in the requested environment from ``examples/CORDEX_ML``::

    mamba run -n Prithvi python residual_diffusion_overfit_gate.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from granitewxc.decoders.diffusion_head import (  # noqa: E402
    DiffusionHead,
    DiffusionHeadConfig,
)


@dataclass(frozen=True)
class GateMetrics:
    examples: int
    steps: int
    initial_epsilon_mse: float
    final_epsilon_mse: float
    loss_reduction: float
    baseline_mae: float
    baseline_rmse: float
    residual_mae_mean: float
    residual_rmse_mean: float
    residual_rmse_worst_seed: float
    corrected_mae_mean: float
    corrected_rmse_mean: float
    corrected_rmse_worst_seed: float
    correction_improvement_fraction: float
    sample_seeds: tuple[int, ...]
    loss_history: tuple[tuple[int, float, float, float], ...]


def _load_diffusion_contract(config_path: Path) -> dict[str, object]:
    """Load and validate the scheduler/objective contract from the run YAML."""
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    try:
        contract = dict(raw["model"]["diffusion"])
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"{config_path} has no model.diffusion mapping."
        ) from exc

    required = (
        "sde",
        "beta_min",
        "beta_max",
        "num_scales",
        "noise_conditioning_scale",
        "sampling_method",
        "num_sampling_steps",
        "sampling_eps",
        "eta",
        "residual_diffusion",
        "prediction_type",
        "clean_x0_reconstruction_weight",
        "clean_x0_inverse_snr_cap",
        "fourier_scale",
    )
    missing = [name for name in required if name not in contract]
    if missing:
        raise ValueError(
            f"{config_path} is missing required diffusion gate fields: {missing}."
        )
    if not bool(contract["residual_diffusion"]):
        raise ValueError("The overfit gate requires residual_diffusion: true.")
    if str(contract["prediction_type"]).lower() != "epsilon":
        raise ValueError("The overfit gate requires prediction_type: epsilon.")
    if str(contract["sde"]).lower() != "vpsde":
        raise ValueError("The eta=0.2 residual gate requires sde: vpsde.")
    if str(contract["sampling_method"]).lower() != "ddim":
        raise ValueError("The residual gate requires sampling_method: ddim.")
    return contract


def _make_cases(n_examples: int, size: int, device: torch.device):
    """Build deterministic baseline/target pairs with spatial conditioning."""
    axis = torch.linspace(-1.0, 1.0, size, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    conds: list[torch.Tensor] = []
    baselines: list[torch.Tensor] = []
    residuals: list[torch.Tensor] = []

    for index in range(n_examples):
        phase = -0.55 + 1.1 * index / max(n_examples - 1, 1)
        code = -0.75 + 1.5 * index / max(n_examples - 1, 1)
        wave = torch.sin(math.pi * (xx + phase)) * torch.cos(math.pi * yy)
        base_cond = torch.stack(
            (
                xx,
                yy,
                wave,
                torch.full_like(xx, code),
            )
        )
        baseline = torch.stack(
            (
                0.42 * xx + 0.18 * yy + 0.08 * code,
                -0.34 * yy + 0.12 * torch.cos(math.pi * xx) - 0.05 * code,
            )
        )
        residual = torch.stack(
            (
                0.24 * wave + 0.055 * xx * yy + 0.025 * code,
                0.19 * torch.cos(math.pi * (yy - 0.5 * phase))
                * torch.sin(math.pi * xx)
                - 0.045 * yy
                - 0.02 * code,
            )
        )
        # Include the exact residual pattern in conditioning. The full CORDEX
        # model supplies learned decoder features here; for an overfit gate we
        # intentionally remove representation learning as a confounder.
        conds.append(torch.cat((base_cond, residual), dim=0))
        baselines.append(baseline)
        residuals.append(residual)

    cond = torch.stack(conds)
    baseline = torch.stack(baselines)
    residual = torch.stack(residuals)
    target = baseline + residual
    return cond, baseline, residual, target


def _build_head(
    device: torch.device,
    sampling_steps: int,
    x0_weight: float,
    inverse_snr_cap: float,
    contract: dict[str, object],
) -> DiffusionHead:
    # The SDE, epsilon parameterization, time scaling, and DDIM method mirror
    # SA_T2_ACCESS-CM2_static_residual_diffusion.yaml.  The score U-Net is
    # deliberately narrower so this gate remains a sub-three-minute check.
    config = DiffusionHeadConfig(
        sde=str(contract["sde"]),
        beta_min=float(contract["beta_min"]),
        beta_max=float(contract["beta_max"]),
        num_scales=int(contract["num_scales"]),
        eps=float(contract.get("eps", 1e-5)),
        noise_conditioning_scale=float(contract["noise_conditioning_scale"]),
        sampling_method=str(contract["sampling_method"]),
        num_sampling_steps=sampling_steps,
        sampling_eps=float(contract["sampling_eps"]),
        eta=float(contract["eta"]),
        denoise=bool(contract.get("denoise", True)),
        projected_cond_channels=0,
        residual_diffusion=True,
        prediction_type=str(contract["prediction_type"]),
        clean_x0_reconstruction_weight=x0_weight,
        clean_x0_inverse_snr_cap=inverse_snr_cap,
        residual_magnitude_guard_multiple=0.0,
        base_channels=24,
        channel_multipliers=(1, 2),
        num_res_blocks=1,
        time_embed_dim=64,
        dropout=0.0,
        fourier_scale=float(contract["fourier_scale"]),
    )
    return DiffusionHead(cond_channels=6, output_channels=2, head_config=config).to(device)


@torch.no_grad()
def _evaluate(
    head: DiffusionHead,
    cond: torch.Tensor,
    baseline: torch.Tensor,
    residual: torch.Tensor,
    target: torch.Tensor,
    sample_seeds: tuple[int, ...],
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    head.eval()
    residual_maes: list[float] = []
    residual_rmses: list[float] = []
    corrected_maes: list[float] = []
    corrected_rmses: list[float] = []
    predicted_residuals: list[torch.Tensor] = []
    corrected_predictions: list[torch.Tensor] = []

    for seed in sample_seeds:
        # The current SDE prior sampler consumes the device-global RNG; seeding
        # it explicitly makes the complete reverse trajectory reproducible.
        torch.manual_seed(seed)
        if cond.is_cuda:
            torch.cuda.manual_seed_all(seed)
        corrected, predicted_residual = head.sample_components(
            cond,
            residual.shape[-2],
            residual.shape[-1],
            baseline_std=baseline,
        )
        residual_error = predicted_residual - residual
        corrected_error = corrected - target
        residual_maes.append(residual_error.abs().mean().item())
        residual_rmses.append(residual_error.square().mean().sqrt().item())
        corrected_maes.append(corrected_error.abs().mean().item())
        corrected_rmses.append(corrected_error.square().mean().sqrt().item())
        predicted_residuals.append(predicted_residual.detach().cpu())
        corrected_predictions.append(corrected.detach().cpu())

    baseline_error = baseline - target
    baseline_mae = baseline_error.abs().mean().item()
    baseline_rmse = baseline_error.square().mean().sqrt().item()
    corrected_rmse_mean = sum(corrected_rmses) / len(corrected_rmses)
    metrics = {
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse,
        "residual_mae_mean": sum(residual_maes) / len(residual_maes),
        "residual_rmse_mean": sum(residual_rmses) / len(residual_rmses),
        "residual_rmse_worst_seed": max(residual_rmses),
        "corrected_mae_mean": sum(corrected_maes) / len(corrected_maes),
        "corrected_rmse_mean": corrected_rmse_mean,
        "corrected_rmse_worst_seed": max(corrected_rmses),
        "correction_improvement_fraction": 1.0 - corrected_rmse_mean / baseline_rmse,
    }
    predicted_stack = torch.stack(predicted_residuals)
    corrected_stack = torch.stack(corrected_predictions)
    artifact = {
        "true_residual": residual.detach().cpu(),
        "predicted_residual": predicted_stack,
        "residual_error": predicted_stack - residual.detach().cpu().unsqueeze(0),
        "baseline": baseline.detach().cpu(),
        "corrected_prediction": corrected_stack,
        "target": target.detach().cpu(),
        "sample_seeds": torch.tensor(sample_seeds, dtype=torch.int64),
    }
    return metrics, artifact


def _run_gate(
    n_examples: int,
    steps: int,
    device: torch.device,
    sampling_steps: int,
    train_seed: int,
    sample_seeds: tuple[int, ...],
    x0_weight: float,
    inverse_snr_cap: float,
    contract: dict[str, object],
) -> tuple[GateMetrics, dict[str, torch.Tensor]]:
    torch.manual_seed(train_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(train_seed)
    cond, baseline, residual, target = _make_cases(n_examples, size=8, device=device)
    head = _build_head(
        device, sampling_steps, x0_weight, inverse_snr_cap, contract
    )
    head.train()
    optimizer = torch.optim.AdamW(head.score_model.parameters(), lr=2e-3, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=2e-4
    )
    history_steps = {0, steps // 8, steps // 4, steps // 2, 3 * steps // 4, steps - 1}
    history: list[tuple[int, float, float, float]] = []
    replicas = max(8, 32 // n_examples)
    train_cond = cond.repeat_interleave(replicas, dim=0)
    train_baseline = baseline.repeat_interleave(replicas, dim=0)
    train_target = target.repeat_interleave(replicas, dim=0)
    torch.manual_seed(train_seed + 1000)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(train_seed + 1000)

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        # Exercise the exact core path used by CORDEX training. The clean-x0
        # term and its inverse-SNR stabilization are resolved from head.cfg.
        loss = head.training_loss(
            train_cond,
            train_target,
            baseline_std=train_baseline,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.score_model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        if step in history_steps:
            terms = head.get_last_training_loss_terms()
            history.append(
                (
                    step + 1,
                    terms["epsilon_mse"],
                    terms["clean_x0_reconstruction"],
                    terms["total"],
                )
            )

    evaluation, artifact = _evaluate(
        head, cond, baseline, residual, target, sample_seeds
    )
    initial_mse = history[0][1]
    final_mse = history[-1][1]
    metrics = GateMetrics(
        examples=n_examples,
        steps=steps,
        initial_epsilon_mse=initial_mse,
        final_epsilon_mse=final_mse,
        loss_reduction=initial_mse / max(final_mse, 1e-12),
        sample_seeds=sample_seeds,
        loss_history=tuple(history),
        **evaluation,
    )
    artifact["loss_history"] = torch.tensor(history, dtype=torch.float64)
    return metrics, artifact


def _assert_gate(metrics: GateMetrics) -> None:
    failures: list[str] = []
    if metrics.loss_reduction < 8.0:
        failures.append(f"epsilon loss reduction {metrics.loss_reduction:.2f}x < 8x")
    if metrics.final_epsilon_mse > 0.08:
        failures.append(f"final epsilon MSE {metrics.final_epsilon_mse:.4f} > 0.08")
    if metrics.corrected_rmse_mean > 0.055:
        failures.append(
            f"mean corrected RMSE {metrics.corrected_rmse_mean:.4f} > 0.055"
        )
    if metrics.corrected_rmse_worst_seed > 0.09:
        failures.append(
            "worst-seed corrected RMSE "
            f"{metrics.corrected_rmse_worst_seed:.4f} > 0.09"
        )
    if metrics.correction_improvement_fraction < 0.65:
        failures.append(
            "baseline-error improvement "
            f"{100 * metrics.correction_improvement_fraction:.1f}% < 65%"
        )
    if failures:
        raise AssertionError(
            f"{metrics.examples}-example residual-diffusion gate failed: "
            + "; ".join(failures)
        )


def _save_artifacts(
    output_dir: Path,
    report: dict[str, object],
    one_case_artifact: dict[str, torch.Tensor],
) -> tuple[Path, Path]:
    """Persist the acceptance report and normalized-space stage tensors."""
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "one_case_residual_overfit_stages.npz"
    np.savez_compressed(
        artifact_path,
        **{
            name: value.detach().cpu().numpy()
            for name, value in one_case_artifact.items()
        },
    )
    report_path = output_dir / "residual_diffusion_overfit_gate.json"
    report["artifacts"] = {
        "report_json": str(report_path.resolve()),
        "one_case_npz": str(artifact_path.resolve()),
        "space": "normalized_target_space",
        "npz_fields": sorted(one_case_artifact),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report_path, artifact_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--one-steps", type=int, default=1000)
    parser.add_argument("--batch-steps", type=int, default=1400)
    parser.add_argument("--sampling-steps", type=int)
    parser.add_argument("--x0-weight", type=float)
    parser.add_argument("--inverse-snr-cap", type=float)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name(
            "SA_T2_ACCESS-CM2_static_residual_diffusion.yaml"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("experiments")
        / "SA_T2_ACCESS-CM2_static_residual_diffusion"
        / "diagnostics"
        / "tiny_overfit_gate",
    )
    parser.add_argument(
        "--sample-seeds", type=int, nargs="+", default=[101, 202, 303, 404]
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")
    seeds = tuple(args.sample_seeds)
    contract = _load_diffusion_contract(args.config)
    sampling_steps = int(
        args.sampling_steps
        if args.sampling_steps is not None
        else contract["num_sampling_steps"]
    )
    x0_weight = float(
        args.x0_weight
        if args.x0_weight is not None
        else contract["clean_x0_reconstruction_weight"]
    )
    inverse_snr_cap = float(
        args.inverse_snr_cap
        if args.inverse_snr_cap is not None
        else contract["clean_x0_inverse_snr_cap"]
    )
    started = time.perf_counter()

    gate_runs = [
        _run_gate(
            1,
            args.one_steps,
            device,
            sampling_steps,
            17,
            seeds,
            x0_weight,
            inverse_snr_cap,
            contract,
        ),
        _run_gate(
            4,
            args.batch_steps,
            device,
            sampling_steps,
            29,
            seeds,
            x0_weight,
            inverse_snr_cap,
            contract,
        ),
    ]
    results = [metrics for metrics, _ in gate_runs]
    elapsed = time.perf_counter() - started
    failures: list[str] = []
    for result in results:
        try:
            _assert_gate(result)
        except AssertionError as exc:
            failures.append(str(exc))

    report = {
        "status": "FAIL" if failures else "PASS",
        "device": str(device),
        "torch_version": torch.__version__,
        "elapsed_seconds": elapsed,
        "source_config": str(args.config.resolve()),
        "diffusion_config": {
            "sde": str(contract["sde"]),
            "beta_min": float(contract["beta_min"]),
            "beta_max": float(contract["beta_max"]),
            "num_scales": int(contract["num_scales"]),
            "prediction_type": str(contract["prediction_type"]),
            "noise_conditioning_scale": float(
                contract["noise_conditioning_scale"]
            ),
            "fourier_scale": float(contract["fourier_scale"]),
            "sampling_method": str(contract["sampling_method"]),
            "sampling_eps": float(contract["sampling_eps"]),
            "eta": float(contract["eta"]),
            "num_sampling_steps": sampling_steps,
        },
        "objective": {
            "epsilon_weight": 1.0,
            "clean_x0_reconstruction_weight": x0_weight,
            "clean_x0_inverse_snr_cap": inverse_snr_cap,
        },
        "thresholds": {
            "minimum_epsilon_loss_reduction": 8.0,
            "maximum_final_epsilon_mse": 0.08,
            "maximum_mean_corrected_rmse": 0.055,
            "maximum_worst_seed_corrected_rmse": 0.09,
            "minimum_correction_improvement_fraction": 0.65,
        },
        "gates": [asdict(result) for result in results],
        "failures": failures,
    }
    report_path, artifact_path = _save_artifacts(
        args.output_dir, report, gate_runs[0][1]
    )
    print(json.dumps(report, indent=2))
    if failures:
        raise AssertionError("\n".join(failures))
    print(f"RESIDUAL_DIFFUSION_OVERFIT_GATE=PASS report={report_path}")
    print(f"RESIDUAL_DIFFUSION_OVERFIT_ARTIFACT={artifact_path}")


if __name__ == "__main__":
    main()
