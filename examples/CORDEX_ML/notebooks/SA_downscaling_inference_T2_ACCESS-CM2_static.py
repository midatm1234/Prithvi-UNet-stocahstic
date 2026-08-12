#!/usr/bin/env python
"""Python version of SA_downscaling_inference_T2_ACCESS-CM2_static.ipynb.

Fill in the parameter lists below (length NUM_RUNS) to loop over multiple
inference runs without editing the script each time.
"""

from __future__ import annotations

import logging
import os
import sys
import subprocess
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import xarray as xr
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Ensure local modules resolve when running as a script.
SCRIPT_DIR = Path(__file__).resolve()
REPO_ROOT = SCRIPT_DIR.parents[3]
PROJECT_DIR = REPO_ROOT / "examples/CORDEX_ML"
for path in (REPO_ROOT, PROJECT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.chdir(PROJECT_DIR)

from cordex_inference import CordexWrappedDataset, build_inference_dataset, build_predictor_names  # noqa: E402
from utils.nearest_fill import repair_invalid_by_nearest_xr, summarize_invalid_counts_xr  # noqa: E402
from utils.predictand_runtime import assert_nonnegative_outputs, resolve_predictand_specs  # noqa: E402
from utils.inference_blending import infer_batch_with_boundary_mitigation, resolve_boundary_mitigation_settings  # noqa: E402
from utils.quantization_diagnostics import (  # noqa: E402
    RunningStats,
    format_compact_table,
    quantization_detector,
    save_distribution_plot,
    save_json,
)
from granitewxc.models.model import get_finetune_model_UNET  # noqa: E402
from granitewxc.refinement.checkpoint import (  # noqa: E402
    load_phase1_state_dict,
    load_refinement_state_dict,
    validate_phase1_reference,
)
from granitewxc.refinement.two_phase import build_two_phase_model  # noqa: E402
from granitewxc.utils.config import get_config  # noqa: E402
from sa_params import (  # noqa: E402
    UserParams,
    export_params,
    resolve_checkpoint,
    resolve_existing_run_dir,
    validate_paths,
)
from run_utils import assert_no_eccc_reference, load_run_manifest  # noqa: E402


# ===================== USER PARAMETERS (EDIT ME) =====================
NUM_RUNS = 12
ACTIVE_RUN_INDICES = [0]  # run only case 1/12 for now

# Each list must have NUM_RUNS entries.
TEST_SPLITS = ["test/historical/predictors/perfect","test/historical/predictors/perfect",
        "test/historical/predictors/imperfect", "test/historical/predictors/imperfect",
        "test/mid_century/predictors/perfect","test/mid_century/predictors/perfect",
        "test/mid_century/predictors/imperfect", "test/mid_century/predictors/imperfect",
        "test/end_century/predictors/perfect","test/end_century/predictors/perfect",
        "test/end_century/predictors/imperfect", "test/end_century/predictors/imperfect"
]

PREDICTOR_FILES = ["ACCESS-CM2_1981-2000.nc","NorESM2-MM_1981-2000.nc",
        "ACCESS-CM2_1981-2000.nc","NorESM2-MM_1981-2000.nc",
        "ACCESS-CM2_2041-2060.nc","NorESM2-MM_2041-2060.nc",
        "ACCESS-CM2_2041-2060.nc","NorESM2-MM_2041-2060.nc",
        "ACCESS-CM2_2080-2099.nc","NorESM2-MM_2080-2099.nc",
        "ACCESS-CM2_2080-2099.nc","NorESM2-MM_2080-2099.nc"
]

PREDICTION_OUTPUT_NAMES = ["Predictions_pr_tasmax_ACCESS-CM2_1981-2000.nc", "Predictions_pr_tasmax_NorESM2-MM_1981-2000.nc",
        "Predictions_pr_tasmax_ACCESS-CM2_1981-2000.nc", "Predictions_pr_tasmax_NorESM2-MM_1981-2000.nc",
        "Predictions_pr_tasmax_ACCESS-CM2_2041-2060.nc", "Predictions_pr_tasmax_NorESM2-MM_2041-2060.nc",
        "Predictions_pr_tasmax_ACCESS-CM2_2041-2060.nc", "Predictions_pr_tasmax_NorESM2-MM_2041-2060.nc",
        "Predictions_pr_tasmax_ACCESS-CM2_2080-2099.nc", "Predictions_pr_tasmax_NorESM2-MM_2080-2099.nc",
        "Predictions_pr_tasmax_ACCESS-CM2_2080-2099.nc", "Predictions_pr_tasmax_NorESM2-MM_2080-2099.nc"
]

INFERENCE_OUTPUT_ROOTS = [
    str(PROJECT_DIR / f"runs/SA_T2_ACCESS-CM2_static_train/predictions/{period}/{quality}/")
    for period, quality in (
        ("historical", "perfect"), ("historical", "perfect"),
        ("historical", "imperfect"), ("historical", "imperfect"),
        ("mid-century", "perfect"), ("mid-century", "perfect"),
        ("mid-century", "imperfect"), ("mid-century", "imperfect"),
        ("end-century", "perfect"), ("end-century", "perfect"),
        ("end-century", "imperfect"), ("end-century", "imperfect"),
    )
]

# Fixed config for the fine-tuned model
REPO_ROOT = REPO_ROOT.resolve()
PROJECT_DIR = PROJECT_DIR.resolve()
CONFIG_PATH = PROJECT_DIR / "SA_T2_ACCESS-CM2_static_diffusion_transformer.yaml"
DATASET_ROOT: Path | None = None  # derived from YAML by _resolve_dataset_root()
RUNS_ROOT = PROJECT_DIR / f"runs_v6/{CONFIG_PATH.stem.replace('_v6', '')}_train"

FINETUNE_RUN_NAME = getattr(get_config(str(CONFIG_PATH)), "job_id", None)  # set None to auto-pick latest
USE_STATIC = True
STATIC_PATH = None  # e.g., DATASET_ROOT / TRAIN_SPLIT / "predictors" / "Static_fields.nc"

REPAIR_INVALID_INPUTS = True

DEVICE_TARGET = "cuda"
NUM_WORKERS = 2
INFERENCE_BATCH_SIZE = 8  # inference-only; does not affect model weights
PREFERRED_CHECKPOINT = "best"  # Phase-1 fallback only
REFINEMENT_CHECKPOINT_NAME = "last.ckpt"
REFINEMENT_ENSEMBLE_SIZE = 5
MIN_FREE_GB = 8  # minimum free GPU memory to consider "idle"
MAX_GPUS = 1  # set to an int to cap how many GPUs to expose
RESPECT_CUDA_VISIBLE_DEVICES = True  # ignore preset CUDA_VISIBLE_DEVICES when auto-selecting idle GPUs
ENABLE_MIXED_PRECISION = False  # keeps inference in full precision to avoid output quantization
FORCE_OUTPUT_FLOAT32 = True
NETCDF_OUTPUT_DTYPE = "float32"
DIAGNOSTIC_MAX_SAMPLES = 250_000
DIAGNOSTIC_ROUND_DECIMALS = 6
DIAGNOSTIC_TIME_WINDOW = 5
DIAGNOSTIC_SPATIAL_WINDOW = 20
SAVE_DIAGNOSTICS_JSON = True
SAVE_DISTRIBUTION_PLOT = True

DIST_ENABLED = False
DIST_RANK = 0
DIST_WORLD_SIZE = 1
DIST_LOCAL_RANK = 0

# ================================================================


def _check_list_lengths() -> None:
    lists = {
        "TEST_SPLITS": TEST_SPLITS,
        "PREDICTOR_FILES": PREDICTOR_FILES,
        "PREDICTION_OUTPUT_NAMES": PREDICTION_OUTPUT_NAMES,
        "INFERENCE_OUTPUT_ROOTS": INFERENCE_OUTPUT_ROOTS,
    }
    for name, values in lists.items():
        if len(values) != NUM_RUNS:
            raise ValueError(f"{name} must have {NUM_RUNS} entries (got {len(values)})")


def _candidate_runs_roots(preferred_runs_root: Path) -> list[Path]:
    """Return fallback candidates for run roots (runs_v6 -> runs)."""
    try:
        rel = preferred_runs_root.resolve().relative_to(PROJECT_DIR.resolve())
    except Exception:
        return [preferred_runs_root]
    if not rel.parts:
        return [preferred_runs_root]
    tail = Path(*rel.parts[1:]) if len(rel.parts) > 1 else Path()
    preferred_base = rel.parts[0]
    base_order = [name for name in (preferred_base, "runs", "runs_v6") if name in ("runs", "runs_v6")]
    base_order = list(dict.fromkeys(base_order))
    return [PROJECT_DIR / base / tail for base in base_order]


def _select_runs_root(preferred_runs_root: Path, run_name_hint: str | None) -> Path:
    """Choose a run root that actually contains the requested run/manifest."""
    candidates = _candidate_runs_roots(preferred_runs_root)

    if run_name_hint:
        for candidate in candidates:
            manifest = candidate / run_name_hint / "run_manifest.json"
            if manifest.exists():
                return candidate

    for candidate in candidates:
        if not candidate.exists():
            continue
        for child in candidate.iterdir():
            if child.is_dir() and (child / "run_manifest.json").exists():
                return candidate

    return preferred_runs_root


def _remap_runs_base(path: Path, runs_base_name: str) -> Path:
    """Rewrite '/runs_v6/' segments to the selected runs base."""
    text = str(path)
    for name in ("runs_v6", "runs"):
        token = f"/{name}/"
        if token in text:
            return Path(text.replace(token, f"/{runs_base_name}/", 1))
    return path


def _resolve_inputs(values):
    return [str(Path(p).resolve()) for p in values]


def _select_idle_gpus(min_free_gb=8, max_gpus=None, respect_visible_devices=True):
    visible = None
    visible_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_env and respect_visible_devices:
        visible = [int(v) for v in visible_env.split(",") if v.strip() != ""]
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception as exc:
        print(f"GPU auto-selection skipped (nvidia-smi unavailable): {exc}")
        return None
    rows = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        idx_str, free_str = line.split(",")
        idx = int(idx_str.strip())
        free_mb = int(free_str.strip())
        rows.append((idx, free_mb))
    rows.sort(key=lambda x: x[1], reverse=True)
    if visible is not None:
        rows = [row for row in rows if row[0] in visible]
    selected = [idx for idx, free_mb in rows if free_mb >= min_free_gb * 1024]
    if max_gpus is not None:
        selected = selected[:max_gpus]
    if not selected and visible is not None:
        return visible
    return selected


def _configure_idle_gpus() -> None:
    selected = _select_idle_gpus(MIN_FREE_GB, MAX_GPUS, RESPECT_CUDA_VISIBLE_DEVICES)
    if selected:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in selected)
        print(f"Using idle GPUs (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})")
    else:
        print("Using default CUDA_VISIBLE_DEVICES (no idle GPU filter applied).")


def _select_device() -> torch.device:
    target = (DEVICE_TARGET or "auto").lower()
    if target == "cpu":
        return torch.device("cpu")

    if DIST_ENABLED:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed inference requested but CUDA is unavailable.")
        return torch.device(f"cuda:{DIST_LOCAL_RANK}")

    if target == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device_target is 'cuda' but no CUDA device is available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _collect_scaler_dtype_summary(model: torch.nn.Module) -> dict[str, str]:
    model = getattr(model, "phase1", model)
    summary: dict[str, str] = {}
    for name in (
        "input_scalers_mu",
        "input_scalers_sigma",
        "static_input_scalers_mu",
        "static_input_scalers_sigma",
        "output_scalers_mu",
        "output_scalers_sigma",
    ):
        tensor = getattr(model, name, None)
        if torch.is_tensor(tensor):
            summary[name] = str(tensor.dtype)
    return summary


def _normalize_predictors_for_diagnostics(batch: dict[str, torch.Tensor], model: torch.nn.Module) -> torch.Tensor:
    model = getattr(model, "phase1", model)
    x = batch["x"].float()
    batch_size, _, height, width = x.shape
    n_input_timestamps = int(getattr(model, "n_input_timestamps", 1))
    x_sep_time = x.view(batch_size, n_input_timestamps, -1, height, width)
    mu = model.input_scalers_mu.view(1, 1, -1, 1, 1).to(device=x.device, dtype=x.dtype)
    sigma = model.input_scalers_sigma.view(1, 1, -1, 1, 1).to(device=x.device, dtype=x.dtype)
    eps = float(getattr(model, "input_scalers_epsilon", 1e-6))
    x_norm = (x_sep_time - mu) / (sigma + eps)
    return x_norm.reshape(batch_size, -1, height, width)


def _to_pre_inverse_output(pred_inverse: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    model = getattr(model, "phase1", model)
    pred = pred_inverse.float()
    sigma = model.output_scalers_sigma.to(device=pred.device, dtype=pred.dtype)
    mu = getattr(model, "output_scalers_mu", None)
    if mu is None:
        mu = torch.zeros_like(sigma)
    else:
        mu = mu.to(device=pred.device, dtype=pred.dtype)
    return (pred - mu) / (sigma + 1e-12)


def _run_full_inference(dataloader, model, device, target_vars, boundary_cfg):
    predictions = []
    deterministic_predictions = []
    ensemble_member_predictions = []
    pre_inverse_predictions = []
    targets = []

    stage_stats = {
        "raw_predictors": RunningStats(
            max_samples=DIAGNOSTIC_MAX_SAMPLES,
            round_decimals=DIAGNOSTIC_ROUND_DECIMALS,
        ),
        "normalized_predictors": RunningStats(
            max_samples=DIAGNOSTIC_MAX_SAMPLES,
            round_decimals=DIAGNOSTIC_ROUND_DECIMALS,
        ),
        "raw_model_outputs_pre_inverse": RunningStats(
            max_samples=DIAGNOSTIC_MAX_SAMPLES,
            round_decimals=DIAGNOSTIC_ROUND_DECIMALS,
        ),
        "inverse_outputs": RunningStats(
            max_samples=DIAGNOSTIC_MAX_SAMPLES,
            round_decimals=DIAGNOSTIC_ROUND_DECIMALS,
        ),
    }
    stage_stats_by_var = {
        "raw_model_outputs_pre_inverse": {
            name: RunningStats(max_samples=DIAGNOSTIC_MAX_SAMPLES, round_decimals=DIAGNOSTIC_ROUND_DECIMALS)
            for name in target_vars
        },
        "inverse_outputs": {
            name: RunningStats(max_samples=DIAGNOSTIC_MAX_SAMPLES, round_decimals=DIAGNOSTIC_ROUND_DECIMALS)
            for name in target_vars
        },
        "targets": {
            name: RunningStats(max_samples=DIAGNOSTIC_MAX_SAMPLES, round_decimals=DIAGNOSTIC_ROUND_DECIMALS)
            for name in target_vars
        },
    }

    autocast_enabled = bool(ENABLE_MIXED_PRECISION and device.type == "cuda")
    autocast_dtype = (
        torch.bfloat16 if (autocast_enabled and torch.cuda.is_bf16_supported()) else torch.float16
    )

    def autocast_context():
        return (
            torch.cuda.amp.autocast(dtype=autocast_dtype, enabled=autocast_enabled)
            if autocast_enabled
            else nullcontext()
        )

    with torch.no_grad():
        model.eval()
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Running inference", leave=False)):
            if "x" not in batch or "y" not in batch:
                raise KeyError("Inference batch must include 'x' and 'y'")

            stage_stats["raw_predictors"].add(batch["x"].numpy())
            target_cpu = batch["y"].numpy()
            for var_idx, var_name in enumerate(target_vars):
                stage_stats_by_var["targets"][var_name].add(target_cpu[:, var_idx])

            batch = {
                key: value.to(device=device, dtype=torch.float32, non_blocking=True)
                for key, value in batch.items()
            }

            normalized_x = _normalize_predictors_for_diagnostics(batch, model)
            stage_stats["normalized_predictors"].add(normalized_x.detach().cpu().numpy())

            # Initialise the lazy refiner from the first real batch, then load Phase 2.
            if not getattr(model, "_last_checkpoint_loaded", False):
                with torch.no_grad():
                    model._prepare(batch)
                report = load_refinement_state_dict(model, model._last_checkpoint_payload)
                model._last_checkpoint_loaded = True
                print(f"[checkpoint] loaded refinement {REFINEMENT_CHECKPOINT_NAME}: {report.summary()}")
            if boundary_cfg.enabled and not getattr(model, "_boundary_notice_printed", False):
                print("[refinement] full-frame sampling avoids independent tile-noise seams.")
                model._boundary_notice_printed = True
            with autocast_context():
                refined = model.predict(
                    batch,
                    ensemble_size=REFINEMENT_ENSEMBLE_SIZE,
                    seed=int(model.refinement_config.seed) + batch_idx,
                    return_members=True,
                )
                if refined.refined is None or refined.refined_normalized is None or refined.members is None:
                    raise RuntimeError("The active refinement model returned incomplete ensemble output.")
                out = refined.refined
                out_raw_model = refined.refined_normalized

            if FORCE_OUTPUT_FLOAT32:
                out = out.float()
                out_raw_model = out_raw_model.float()

            out_cpu = out.detach().cpu()
            out_pre_inverse_cpu = out_raw_model.detach().cpu()

            predictions.append(out_cpu)
            deterministic_predictions.append(refined.deterministic.detach().cpu().float())
            ensemble_member_predictions.append(refined.members.detach().cpu().float())
            pre_inverse_predictions.append(out_pre_inverse_cpu)
            targets.append(batch["y"].detach().cpu().float())

            stage_stats["raw_model_outputs_pre_inverse"].add(out_pre_inverse_cpu.numpy())
            stage_stats["inverse_outputs"].add(out_cpu.numpy())
            for var_idx, var_name in enumerate(target_vars):
                stage_stats_by_var["raw_model_outputs_pre_inverse"][var_name].add(
                    out_pre_inverse_cpu[:, var_idx].numpy()
                )
                stage_stats_by_var["inverse_outputs"][var_name].add(out_cpu[:, var_idx].numpy())

    if not predictions:
        raise RuntimeError("Inference produced no predictions.")

    return {
        "predictions": torch.cat(predictions, dim=0),
        "deterministic_predictions": torch.cat(deterministic_predictions, dim=0),
        "ensemble_member_predictions": torch.cat(ensemble_member_predictions, dim=0),
        "predictions_pre_inverse": torch.cat(pre_inverse_predictions, dim=0),
        "targets": torch.cat(targets, dim=0),
        "stage_stats": {key: value.finalize() for key, value in stage_stats.items()},
        "stage_stats_by_var": {
            stage: {name: stats.finalize() for name, stats in per_var.items()}
            for stage, per_var in stage_stats_by_var.items()
        },
    }


def _concat_time_coordinate(paths, time_key):
    arrays = []
    attrs = None
    encoding = None
    for path in paths:
        with xr.open_dataset(path) as ds:
            if time_key not in ds.coords and time_key not in ds.data_vars:
                raise KeyError(f"Time dimension '{time_key}' missing in {path}")
            da = ds[time_key]
            arrays.append(da)
            if attrs is None:
                attrs = dict(da.attrs)
            if encoding is None and hasattr(da, "encoding"):
                encoding = dict(da.encoding)
    combined = xr.concat(arrays, dim=time_key)
    if attrs:
        combined.attrs.update(attrs)
    if encoding:
        combined.encoding.update(encoding)
    return combined.load()


def _sanitize_data_attrs(attrs: dict[str, object]) -> dict[str, object]:
    clean_attrs = dict(attrs)
    for key in ("scale_factor", "add_offset", "_FillValue", "missing_value", "dtype"):
        clean_attrs.pop(key, None)
    return clean_attrs


def _build_netcdf_encoding(var_names: list[str]) -> dict[str, dict[str, object]]:
    fill = np.float32(np.nan)
    return {
        name: {
            "dtype": NETCDF_OUTPUT_DTYPE,
            "_FillValue": fill,
        }
        for name in var_names
    }


def _build_var_array_map(var_names: list[str], values: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for var_idx, var_name in enumerate(var_names):
        out[var_name] = np.asarray(values[:, var_idx], dtype=np.float32)
    return out


def _compute_quantization_summary(var_values: dict[str, np.ndarray]) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for var_name, values in var_values.items():
        summary[var_name] = quantization_detector(
            values,
            time_window=DIAGNOSTIC_TIME_WINDOW,
            spatial_window=DIAGNOSTIC_SPATIAL_WINDOW,
            round_decimals=DIAGNOSTIC_ROUND_DECIMALS,
        )
    return summary


def _build_diagnostic_rows(
    stage_stats: dict[str, dict[str, object]],
    stage_stats_by_var: dict[str, dict[str, dict[str, object]]],
    quant_summary: dict[str, dict[str, dict[str, object]]],
) -> list[dict[str, object]]:
    rows = [
        {"stage": "raw_predictors", "var": "all", "stats": stage_stats["raw_predictors"], "quantization": {}},
        {
            "stage": "normalized_predictors",
            "var": "all",
            "stats": stage_stats["normalized_predictors"],
            "quantization": {},
        },
    ]
    for stage_name in ("raw_model_outputs_pre_inverse", "inverse_outputs", "targets"):
        for var_name, stats in stage_stats_by_var.get(stage_name, {}).items():
            rows.append(
                {
                    "stage": stage_name,
                    "var": var_name,
                    "stats": stats,
                    "quantization": quant_summary.get(stage_name, {}).get(var_name, {}),
                }
            )
    return rows



def _repair_predictor_files_if_needed(
    predictor_paths,
    *,
    output_root: Path,
    predictor_var_names,
    run_index: int,
):
    resolved_paths = [str(Path(path).resolve()) for path in predictor_paths]
    if not REPAIR_INVALID_INPUTS:
        print("[repair] Predictor invalid-value repair disabled by REPAIR_INVALID_INPUTS=False")
        return resolved_paths

    repaired_dir = output_root / "_repaired_predictors" / f"run_{run_index:02d}"
    repaired_dir.mkdir(parents=True, exist_ok=True)

    final_paths: list[str] = []
    for predictor_path in resolved_paths:
        source_path = Path(predictor_path)
        with xr.open_dataset(source_path) as ds:
            predictor_ds = ds.load()

        before_counts = summarize_invalid_counts_xr(predictor_ds, var_names=predictor_var_names)
        invalid_before_total = sum(before_counts.values())

        if invalid_before_total > 0:
            print("Invalid predictor values detected (NaN/inf/fill). Auto-repair enabled.")
            predictor_ds.encoding["source"] = str(source_path)
            repaired_ds = repair_invalid_by_nearest_xr(
                predictor_ds,
                var_names=predictor_var_names,
            )
            after_counts = summarize_invalid_counts_xr(repaired_ds, var_names=predictor_var_names)

            repaired_path = repaired_dir / source_path.name
            repaired_ds.to_netcdf(repaired_path, engine="h5netcdf")
            final_paths.append(str(repaired_path.resolve()))
        else:
            after_counts = before_counts
            final_paths.append(str(source_path))

        keys = sorted(set(before_counts) | set(after_counts))
        for key in keys:
            print(
                f"[repair] {source_path.name} {key}: "
                f"invalid_count_before={before_counts.get(key, 0)} "
                f"invalid_count_after={after_counts.get(key, 0)}"
            )

    return final_paths

def _build_dataloader(config, predictor_paths, target_paths, device):
    base_dataset = build_inference_dataset(config, predictor_paths, target_paths)
    dataset = CordexWrappedDataset(base_dataset)
    batch_size = getattr(config, "batch_size", 1)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.dl_num_workers,
        pin_memory=(device.type == "cuda"),
    )


def _resolve_repo_path(value: str | os.PathLike) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _resolve_dataset_root(config) -> Path:
    """Derive SA_domain from YAML-owned data paths; never guess it silently."""
    configured = []
    for field in ("training_predictor_paths", "training_target_paths", "static_path"):
        value = getattr(config.data, field, None)
        values = value if isinstance(value, (list, tuple)) else [value]
        configured.extend(Path(item).expanduser().resolve() for item in values if item)
    roots = []
    for path in configured:
        for parent in (path, *path.parents):
            if parent.name.lower() == "sa_domain":
                roots.append(parent)
                break
    roots = list(dict.fromkeys(roots))
    if len(roots) != 1:
        raise RuntimeError(f"YAML data paths must resolve to one SA_domain root; found {roots}")
    root = roots[0]
    if not root.is_dir():
        raise FileNotFoundError(f"YAML-derived dataset root does not exist: {root}")
    return root


def _resolve_test_predictor(dataset_root: Path, test_split: str, filename: str) -> Path:
    directory = dataset_root / test_split
    requested = directory / filename
    alternatives = [requested]
    if filename.endswith("_regridded.nc"):
        alternatives.append(directory / filename.replace("_regridded.nc", ".nc"))
    elif filename.endswith(".nc"):
        alternatives.append(directory / filename.replace(".nc", "_regridded.nc"))
    existing = list(dict.fromkeys(path.resolve() for path in alternatives if path.is_file()))
    if len(existing) == 1:
        if existing[0] != requested.resolve():
            print(f"[data] predictor filename compatibility fallback: {requested.name} -> {existing[0].name}")
        return existing[0]
    if len(existing) > 1:
        raise RuntimeError(f"Ambiguous predictor files for {filename}: {existing}")
    attempted = "\n  ".join(str(path) for path in alternatives)
    raise FileNotFoundError(f"Test predictor not found; tried:\n  {attempted}")


def _resolve_refinement_checkpoint(config) -> Path:
    configured_dir = Path(config.checkpoint_dir).expanduser()
    if configured_dir.is_absolute():
        candidates = [configured_dir / REFINEMENT_CHECKPOINT_NAME]
    else:
        # The refinement notebook runs with cwd=PROJECT_DIR, while YAML paths
        # are commonly authored relative to REPO_ROOT. Support both meanings.
        candidates = [
            (REPO_ROOT / configured_dir / REFINEMENT_CHECKPOINT_NAME).resolve(),
            (PROJECT_DIR / configured_dir / REFINEMENT_CHECKPOINT_NAME).resolve(),
        ]
    candidates = list(dict.fromkeys(candidates))
    existing = [path for path in candidates if path.is_file()]
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        print(f"[checkpoint] multiple last.ckpt candidates found; using {existing[0]}")
        return existing[0]
    attempted = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"refinement last.ckpt not found; tried:\n  {attempted}")


def _load_model_and_config() -> tuple[str, Path, Path, object, torch.nn.Module, Path]:
    global DATASET_ROOT
    config = get_config(str(CONFIG_PATH))
    assert_no_eccc_reference(CONFIG_PATH)
    DATASET_ROOT = _resolve_dataset_root(config)
    print(f"[config] source={CONFIG_PATH}")
    print(f"[data] YAML-derived dataset root={DATASET_ROOT}")
    config.dl_num_workers = int(NUM_WORKERS) if NUM_WORKERS is not None else config.dl_num_workers
    config.batch_size = int(INFERENCE_BATCH_SIZE) if INFERENCE_BATCH_SIZE is not None else config.batch_size
    config.device_target = DEVICE_TARGET
    if USE_STATIC is not None:
        config.data.use_static = bool(USE_STATIC)
    if STATIC_PATH:
        config.data.static_path = str(Path(STATIC_PATH).resolve())

    phase1_path = _resolve_repo_path(config.model.phase1["checkpoint"])
    refinement_path = _resolve_refinement_checkpoint(config)
    if not phase1_path.is_file():
        raise FileNotFoundError(f"Phase-1 checkpoint not found: {phase1_path}")
    assert_no_eccc_reference(phase1_path)
    assert_no_eccc_reference(refinement_path)

    model = build_two_phase_model(get_finetune_model_UNET(config), config)
    phase1_payload = torch.load(str(phase1_path), map_location="cpu", weights_only=True)
    print(f"[checkpoint] loaded Phase 1: {load_phase1_state_dict(model, phase1_payload).summary()}")
    refinement_payload = torch.load(str(refinement_path), map_location="cpu", weights_only=True)
    validate_phase1_reference(refinement_payload, model.phase1.state_dict(), strict=True)
    model._last_checkpoint_payload = refinement_payload
    model._last_checkpoint_loaded = False

    run_name = str(getattr(config, "job_id", CONFIG_PATH.stem))
    run_dir = phase1_path.parents[2]
    print(f"[checkpoint] staged refinement checkpoint: {refinement_path}")
    return run_name, run_dir, refinement_path, config, model, run_dir.parent


def main() -> None:
    global DIST_ENABLED, DIST_RANK, DIST_WORLD_SIZE, DIST_LOCAL_RANK

    logging.disable(logging.CRITICAL)
    warnings.simplefilter(action="ignore", category=FutureWarning)

    _check_list_lengths()

    DIST_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
    DIST_RANK = int(os.environ.get("RANK", "0"))
    DIST_LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
    DIST_ENABLED = DIST_WORLD_SIZE > 1

    if DIST_ENABLED and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    if DIST_ENABLED:
        print(
            f"[dist] inference enabled: rank={DIST_RANK}/{DIST_WORLD_SIZE}, "
            f"local_rank={DIST_LOCAL_RANK}"
        )
    else:
        _configure_idle_gpus()

    torch.jit.enable_onednn_fusion(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    device = _select_device()
    run_name, run_dir, checkpoint_path, config, model, runs_root_used = _load_model_and_config()
    model.to(device)
    scaler_dtype_summary = _collect_scaler_dtype_summary(model)
    print(f"[diag] scaler dtypes: {scaler_dtype_summary}")
    print(
        f"[diag] mixed_precision_enabled={ENABLE_MIXED_PRECISION}, "
        f"force_output_float32={FORCE_OUTPUT_FLOAT32}"
    )

    invalid_indices = [idx for idx in ACTIVE_RUN_INDICES if idx < 0 or idx >= NUM_RUNS]
    if invalid_indices:
        raise ValueError(f"ACTIVE_RUN_INDICES contains invalid entries: {invalid_indices}")
    run_indices = ACTIVE_RUN_INDICES[DIST_RANK::DIST_WORLD_SIZE] if DIST_ENABLED else ACTIVE_RUN_INDICES
    active_run_count = len(ACTIVE_RUN_INDICES)
    for active_position, idx in enumerate(run_indices, start=1):
        test_split = TEST_SPLITS[idx]
        predictor_file = PREDICTOR_FILES[idx]
        prediction_output_name = PREDICTION_OUTPUT_NAMES[idx]
        inference_output_root = _remap_runs_base(
            Path(INFERENCE_OUTPUT_ROOTS[idx]),
            runs_root_used.parent.name,
        )

        if DATASET_ROOT is None:
            raise RuntimeError("Dataset root was not initialized from the YAML.")
        if predictor_file:
            test_predictors = [_resolve_test_predictor(DATASET_ROOT, test_split, predictor_file)]
        else:
            test_predictors = sorted((DATASET_ROOT / test_split).glob("*.nc"))

        params = UserParams(
            repo_root=REPO_ROOT,
            project_dir=PROJECT_DIR,
            runs_root=runs_root_used,
            config_path=CONFIG_PATH,
            inference_run_name=run_name,
            inference_output_root=inference_output_root,
            inference_predictor_root=DATASET_ROOT / test_split,
            test_predictor_paths=test_predictors,
            test_target_paths=[_resolve_repo_path(path) for path in config.data.training_target_paths],
            use_static=USE_STATIC,
            checkpoint_path=None,
            preferred_checkpoint=PREFERRED_CHECKPOINT,
            device_target=DEVICE_TARGET,
            num_workers=NUM_WORKERS,
            batch_size=INFERENCE_BATCH_SIZE,
        )

        validate_paths(params, require_inference=True)

        test_predictor_paths = _resolve_inputs(params.test_predictor_paths) if params.test_predictor_paths else []
        if not test_predictor_paths and params.inference_predictor_root:
            test_predictor_paths = [
                str(path.resolve()) for path in Path(params.inference_predictor_root).glob("*.nc")
            ]
        if not test_predictor_paths:
            raise FileNotFoundError("No inference predictor files configured in USER PARAMETERS.")

        output_root = Path(params.inference_output_root or (run_dir / "predictions"))
        output_root.mkdir(parents=True, exist_ok=True)

        predictor_var_names = build_predictor_names(config)
        test_predictor_paths = _repair_predictor_files_if_needed(
            test_predictor_paths,
            output_root=output_root,
            predictor_var_names=predictor_var_names,
            run_index=idx + 1,
        )

        config.data.test_predictor_paths = test_predictor_paths
        config.data.test_target_paths = _resolve_inputs(params.test_target_paths)

        prediction_output_stub = Path(
            prediction_output_name or f"{run_name}_predictions_{idx + 1:02d}.nc"
        )

        params_json = export_params(
            params,
            output_root / prediction_output_stub.with_suffix(".json"),
            extra={"run_name": run_name, "checkpoint": str(checkpoint_path)},
        )

        print(f"[{active_position:02d}/{active_run_count}] Using predictors from {Path(test_predictor_paths[0]).parent}")
        print(f"[{active_position:02d}/{active_run_count}] Output directory: {output_root}")
        print(f"[{active_position:02d}/{active_run_count}] Saved parameter snapshot to {params_json}")

        test_dl = _build_dataloader(config, config.data.test_predictor_paths, config.data.test_target_paths, device)

        base_dataset = test_dl.dataset.base
        target_vars = list(base_dataset.target_vars)
        predictand_specs = resolve_predictand_specs(config, target_vars)
        predictor_paths = list(base_dataset.predictor_paths)
        target_template_paths = list(base_dataset.target_paths)

        boundary_cfg = resolve_boundary_mitigation_settings(config)
        boundary_cfg.force_full_frame = True
        if boundary_cfg.force_full_frame:
            print("[boundary] force_full_frame=True (tiling/blending disabled)")
        else:
            print(
                f"[boundary] enabled={boundary_cfg.enabled}, tile_size={boundary_cfg.tile_size}, "
                f"overlap={boundary_cfg.overlap}, blend_mode={boundary_cfg.blend_mode}, "
                f"deblock_enabled={boundary_cfg.deblock.enabled}"
            )
        inference_result = _run_full_inference(test_dl, model, device, target_vars, boundary_cfg)
        full_outputs = inference_result["predictions"]
        full_deterministic_outputs = inference_result["deterministic_predictions"]
        full_ensemble_member_outputs = inference_result["ensemble_member_predictions"]
        full_outputs_pre_inverse = inference_result["predictions_pre_inverse"]
        full_targets = inference_result["targets"]

        outputs_np = full_outputs.numpy().astype(np.float32, copy=False)
        deterministic_outputs_np = full_deterministic_outputs.numpy().astype(np.float32, copy=False)
        ensemble_member_outputs_np = full_ensemble_member_outputs.numpy().astype(np.float32, copy=False)
        outputs_pre_inverse_np = full_outputs_pre_inverse.numpy().astype(np.float32, copy=False)
        targets_np = full_targets.numpy().astype(np.float32, copy=False)
        if outputs_np.shape[1] != len(target_vars):
            raise ValueError(
                f"Model produced {outputs_np.shape[1]} channels but target_vars expects {len(target_vars)}"
            )
        expected_member_shape = (outputs_np.shape[0], REFINEMENT_ENSEMBLE_SIZE, len(target_vars))
        if ensemble_member_outputs_np.shape[:3] != expected_member_shape:
            raise ValueError(
                f"Unexpected ensemble shape {ensemble_member_outputs_np.shape}; "
                f"expected [time, {REFINEMENT_ENSEMBLE_SIZE}, {len(target_vars)}, lat, lon]"
            )

        predicted_var_values = _build_var_array_map(target_vars, outputs_np)
        predicted_pre_inverse_var_values = _build_var_array_map(target_vars, outputs_pre_inverse_np)
        target_var_values = _build_var_array_map(target_vars, targets_np)

        assert_nonnegative_outputs(predicted_var_values, predictand_specs, eps=1e-8)

        quantization_summary = {
            "raw_model_outputs_pre_inverse": _compute_quantization_summary(predicted_pre_inverse_var_values),
            "inverse_outputs": _compute_quantization_summary(predicted_var_values),
            "targets": _compute_quantization_summary(target_var_values),
        }
        diagnostic_rows = _build_diagnostic_rows(
            inference_result["stage_stats"],
            inference_result["stage_stats_by_var"],
            quantization_summary,
        )
        print(f"[diag] Quantization summary for run {idx + 1:02d}")
        print(format_compact_table(diagnostic_rows))

        diagnostics_payload = {
            "run_index": idx + 1,
            "scenario_count": NUM_RUNS,
            "active_run_position": active_position,
            "active_run_count": active_run_count,
            "predictor_paths": test_predictor_paths,
            "target_template_paths": target_template_paths,
            "stage_stats": inference_result["stage_stats"],
            "stage_stats_by_var": inference_result["stage_stats_by_var"],
            "quantization_detector": quantization_summary,
            "scaler_dtypes": scaler_dtype_summary,
            "mixed_precision_enabled": bool(ENABLE_MIXED_PRECISION),
            "force_output_float32": bool(FORCE_OUTPUT_FLOAT32),
            "refinement_ensemble_size": REFINEMENT_ENSEMBLE_SIZE,
            "predictands": {name: spec.to_dict() for name, spec in predictand_specs.items()},
        }
        if SAVE_DIAGNOSTICS_JSON:
            diagnostics_path = output_root / prediction_output_stub.with_suffix(".diagnostics.json")
            save_json(diagnostics_payload, diagnostics_path)
            print(f"[diag] Saved diagnostics JSON to {diagnostics_path}")
        if SAVE_DISTRIBUTION_PLOT:
            if "tasmax" in predicted_var_values and "pr" in predicted_var_values:
                plot_path = output_root / prediction_output_stub.with_suffix(".distribution.png")
                save_distribution_plot(
                    target_values=target_var_values,
                    predicted_values=predicted_var_values,
                    predicted_raw_values=predicted_pre_inverse_var_values,
                    output_path=plot_path,
                )
                print(f"[diag] Saved distribution comparison plot to {plot_path}")
            else:
                print("[diag] Skipped distribution plot; expected vars 'pr' and 'tasmax' were not both present.")

        time_dim = base_dataset.time_dim or "time"
        lat_dim, lon_dim = base_dataset.output_spatial_dims
        lat_name = base_dataset.fine_lat_name
        lon_name = base_dataset.fine_lon_name

        predictor_time_coord = _concat_time_coordinate(predictor_paths, time_dim)
        if outputs_np.shape[0] != predictor_time_coord.sizes[time_dim]:
            raise ValueError(
                "Prediction time dimension "
                f"{outputs_np.shape[0]} does not match predictor timestamps {predictor_time_coord.sizes[time_dim]}"
            )

        if not target_template_paths:
            raise FileNotFoundError("Target template paths missing; ensure config.data.test_target_paths is set")

        with xr.open_dataset(target_template_paths[0], engine="h5netcdf") as template_ds:
            template_attrs = dict(template_ds.attrs)
            target_attrs = {
                name: _sanitize_data_attrs(dict(template_ds[name].attrs))
                for name in target_vars
                if name in template_ds.data_vars
            }
            lat_coord = template_ds[lat_name].load()
            lon_coord = template_ds[lon_name].load()

        coords = {time_dim: predictor_time_coord, lat_dim: lat_coord, lon_dim: lon_coord}
        ensemble_coords = {
            **coords,
            "ensemble": np.arange(REFINEMENT_ENSEMBLE_SIZE, dtype=np.int32),
        }
        prediction_ds = xr.Dataset(coords=ensemble_coords)
        for var_idx, name in enumerate(target_vars):
            prediction_ds[name] = xr.DataArray(
                ensemble_member_outputs_np[:, :, var_idx],
                dims=(time_dim, "ensemble", lat_dim, lon_dim),
                coords=ensemble_coords,
                attrs=target_attrs.get(name, {}),
            )

        for name in target_vars:
            prediction_ds[name] = prediction_ds[name].astype(np.float32)

        prediction_ds.attrs.update(template_attrs)
        prediction_ds.attrs["prediction_kind"] = "refinement_ensemble_members"
        prediction_ds.attrs["ensemble_size"] = REFINEMENT_ENSEMBLE_SIZE
        prediction_output_path = output_root / prediction_output_stub
        prediction_ds.to_netcdf(
            prediction_output_path,
            engine="h5netcdf",
            encoding=_build_netcdf_encoding(target_vars),
        )
        print(
            f"[{active_position:02d}/{active_run_count}] Saved refined ensemble members to {prediction_output_path} "
            f"({predictor_time_coord.values[0]} -> {predictor_time_coord.values[-1]})"
        )

        deterministic_ds = xr.Dataset(coords=coords)
        for var_idx, name in enumerate(target_vars):
            deterministic_ds[name] = xr.DataArray(
                deterministic_outputs_np[:, var_idx],
                dims=(time_dim, lat_dim, lon_dim),
                coords=coords,
                attrs=target_attrs.get(name, {}),
            ).astype(np.float32)
        deterministic_ds.attrs.update(template_attrs)
        deterministic_ds.attrs["prediction_kind"] = "phase1_unet_deterministic"
        deterministic_stub = prediction_output_stub.with_name(
            f"{prediction_output_stub.stem}.baseline{prediction_output_stub.suffix}"
        )
        deterministic_output_path = output_root / deterministic_stub
        deterministic_ds.to_netcdf(
            deterministic_output_path,
            engine="h5netcdf",
            encoding=_build_netcdf_encoding(target_vars),
        )
        print(f"[{active_position:02d}/{active_run_count}] Saved deterministic U-Net output to {deterministic_output_path}")

        pickle_path = output_root / prediction_output_stub.with_suffix(".pkl")
        with open(pickle_path, "wb") as handle:
            import pickle

            pickle.dump(ensemble_member_outputs_np, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[{active_position:02d}/{active_run_count}] Saved raw ensemble-members array to {pickle_path}")

        pre_inverse_pickle_path = output_root / prediction_output_stub.with_suffix(".pre_inverse.pkl")
        with open(pre_inverse_pickle_path, "wb") as handle:
            import pickle

            pickle.dump(outputs_pre_inverse_np, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(
            f"[{active_position:02d}/{active_run_count}] Saved pre-inverse predictions array to "
            f"{pre_inverse_pickle_path}"
        )

    if DIST_ENABLED and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
