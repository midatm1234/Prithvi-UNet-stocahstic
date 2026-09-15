#!/usr/bin/env python
"""Python version of SA_downscaling_inference_T2_ACCESS-CM2_static.ipynb.

Fill in the parameter lists below (length NUM_RUNS) to loop over multiple
inference runs without editing the script each time.
"""

from __future__ import annotations

import logging
import hashlib
import json
import os
import re
import shutil
import sys
import subprocess
import warnings
from contextlib import nullcontext
from datetime import datetime, timezone
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
# Dispatch before the historical notebook changes directories, so scientific
# command-line paths resolve against the caller's working directory.
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--scientific-experiment":
    from granitewxc.refinement.scientific_entrypoints import dispatch_scientific
    raise SystemExit(dispatch_scientific(sys.argv[1:], expected_operation="predict"))
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
    CHECKPOINT_SCHEMA_VERSION,
    REFINEMENT_CONTRACT_VERSION,
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

# Exact held-out truth paired by timestamp (not by positional index). The
# target calendar includes leap days while predictors are no-leap; the dataset
# joins identical dates and ignores target-only February 29 values.
TARGET_FILES = [
    "test/historical/target/pr_tasmax_ACCESS-CM2_1981-2000.nc",
    "test/historical/target/pr_tasmax_Nor-ESM2-MM_1981-2000.nc",
    "test/historical/target/pr_tasmax_ACCESS-CM2_1981-2000.nc",
    "test/historical/target/pr_tasmax_Nor-ESM2-MM_1981-2000.nc",
    "test/mid_century/target/pr_tasmax_ACCESS-CM2_2041-2060.nc",
    "test/mid_century/target/pr_tasmax_Nor-ESM2-MM_2041-2060.nc",
    "test/mid_century/target/pr_tasmax_ACCESS-CM2_2041-2060.nc",
    "test/mid_century/target/pr_tasmax_Nor-ESM2-MM_2041-2060.nc",
    "test/end_century/target/pr_tasmax_ACCESS-CM2_2080-2099.nc",
    "test/end_century/target/pr_tasmax_Nor-ESM2-MM_2080-2099.nc",
    "test/end_century/target/pr_tasmax_ACCESS-CM2_2080-2099.nc",
    "test/end_century/target/pr_tasmax_Nor-ESM2-MM_2080-2099.nc",
]

PREDICTION_OUTPUT_NAMES = ["Predictions_pr_tasmax_ACCESS-CM2_1981-2000.nc", "Predictions_pr_tasmax_NorESM2-MM_1981-2000.nc",
        "Predictions_pr_tasmax_ACCESS-CM2_1981-2000.nc", "Predictions_pr_tasmax_NorESM2-MM_1981-2000.nc",
        "Predictions_pr_tasmax_ACCESS-CM2_2041-2060.nc", "Predictions_pr_tasmax_NorESM2-MM_2041-2060.nc",
        "Predictions_pr_tasmax_ACCESS-CM2_2041-2060.nc", "Predictions_pr_tasmax_NorESM2-MM_2041-2060.nc",
        "Predictions_pr_tasmax_ACCESS-CM2_2080-2099.nc", "Predictions_pr_tasmax_NorESM2-MM_2080-2099.nc",
        "Predictions_pr_tasmax_ACCESS-CM2_2080-2099.nc", "Predictions_pr_tasmax_NorESM2-MM_2080-2099.nc"
]

INFERENCE_OUTPUT_SUBDIRS = [
    Path(period) / quality
    for period, quality in (
        ("historical", "perfect"), ("historical", "perfect"),
        ("historical", "imperfect"), ("historical", "imperfect"),
        ("mid-century", "perfect"), ("mid-century", "perfect"),
        ("mid-century", "imperfect"), ("mid-century", "imperfect"),
        ("end-century", "perfect"), ("end-century", "perfect"),
        ("end-century", "imperfect"), ("end-century", "imperfect"),
    )
]

# Outputs default to:
#   <model run>/predictions/<refinement case>/<period>/<quality>/
# The schema-2 suffix prevents a repaired run from overwriting legacy NetCDFs.
# Override only with a distinct, filesystem-safe directory name.
REFINEMENT_OUTPUT_HEADER: str | None = os.environ.get(
    "GRANITE_REFINEMENT_OUTPUT_HEADER"
)
# Relocate the complete output tree to a drive with sufficient free space.
# The refinement case / period / quality subdirectories are appended here.
REFINEMENT_OUTPUT_ROOT: str | None = os.environ.get(
    "GRANITE_REFINEMENT_OUTPUT_ROOT"
)

# Fixed config for the fine-tuned model
REPO_ROOT = REPO_ROOT.resolve()
PROJECT_DIR = PROJECT_DIR.resolve()
_CONFIG_OVERRIDE = os.environ.get("GRANITE_REFINEMENT_CONFIG")
CONFIG_PATH = Path(
    _CONFIG_OVERRIDE
    or PROJECT_DIR / "SA_T2_ACCESS-CM2_static_flow_matching_unet.yaml"
)
if not CONFIG_PATH.is_absolute():
    candidates = [PROJECT_DIR / CONFIG_PATH, REPO_ROOT / CONFIG_PATH]
    existing = list(dict.fromkeys(path.resolve() for path in candidates if path.is_file()))
    if len(existing) > 1:
        raise RuntimeError(
            f"Ambiguous relative GRANITE_REFINEMENT_CONFIG={CONFIG_PATH!s}: {existing}"
        )
    CONFIG_PATH = existing[0] if existing else candidates[0]
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
REFINEMENT_CHECKPOINT_NAME = os.environ.get(
    "GRANITE_REFINEMENT_CHECKPOINT", "best.ckpt"
)
_REFINEMENT_ENSEMBLE_OVERRIDE = os.environ.get(
    "GRANITE_REFINEMENT_ENSEMBLE_SIZE"
)
REFINEMENT_ENSEMBLE_SIZE: int | None = (
    int(_REFINEMENT_ENSEMBLE_OVERRIDE)
    if _REFINEMENT_ENSEMBLE_OVERRIDE is not None
    else None
)
MIN_FREE_GB = 8  # minimum free GPU memory to consider "idle"
MAX_GPUS = 1  # set to an int to cap how many GPUs to expose
RESPECT_CUDA_VISIBLE_DEVICES = True  # ignore preset CUDA_VISIBLE_DEVICES when auto-selecting idle GPUs
ENABLE_MIXED_PRECISION = False  # keeps inference in full precision to avoid output quantization
FORCE_OUTPUT_FLOAT32 = True
NETCDF_OUTPUT_DTYPE = "float32"
# Lossless compression preserves all float32 values and enables HDF5 chunking.
NETCDF_COMPRESSION_LEVEL = 4
# Pickles duplicate the ensemble and add a pre-inverse diagnostic array.
SAVE_PICKLE_OUTPUTS = os.environ.get("GRANITE_REFINEMENT_SAVE_PICKLES", "0") == "1"
SAVE_SAMPLING_STATES = os.environ.get("GRANITE_REFINEMENT_SAVE_SAMPLING_STATES", "0") == "1"
OUTPUT_DISK_RESERVE_BYTES = 1024**3
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
        "TARGET_FILES": TARGET_FILES,
        "PREDICTION_OUTPUT_NAMES": PREDICTION_OUTPUT_NAMES,
        "INFERENCE_OUTPUT_SUBDIRS": INFERENCE_OUTPUT_SUBDIRS,
    }
    for name, values in lists.items():
        if len(values) != NUM_RUNS:
            raise ValueError(f"{name} must have {NUM_RUNS} entries (got {len(values)})")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prediction_output_root(run_dir: Path, header: str, subdir: Path) -> Path:
    root = (
        _resolve_repo_path(REFINEMENT_OUTPUT_ROOT)
        if REFINEMENT_OUTPUT_ROOT
        else run_dir / "predictions"
    )
    return root / header / subdir


def _check_output_disk_space(
    output_root: Path, *, sample_count: int, target_shape: tuple[int, int],
    variable_count: int, ensemble_size: int,
) -> int:
    """Budget the full output bundle without assuming a compression ratio."""
    field_bytes = (
        int(sample_count) * int(target_shape[0]) * int(target_shape[1])
        * int(variable_count) * np.dtype(NETCDF_OUTPUT_DTYPE).itemsize
    )
    # Ensemble members, deterministic baseline, and three constraint fields.
    output_bytes = field_bytes * (ensemble_size + 4)
    if SAVE_PICKLE_OUTPUTS:
        output_bytes += field_bytes * (ensemble_size + 1)
    # Allow for coordinates, chunk metadata, plots, and other small artifacts.
    required = int(output_bytes * 1.05) + OUTPUT_DISK_RESERVE_BYTES
    free = shutil.disk_usage(output_root).free
    print(
        f"[storage] {output_root}: free={free / 1024**3:.2f} GiB, "
        f"conservative output budget={required / 1024**3:.2f} GiB "
        f"(pickle copies={'enabled' if SAVE_PICKLE_OUTPUTS else 'disabled'})"
    )
    if free < required:
        raise OSError(
            28,
            f"Insufficient output disk space before inference: "
            f"{free / 1024**3:.2f} GiB free, {required / 1024**3:.2f} GiB "
            "required by the uncompressed estimate plus reserve. Set "
            "GRANITE_REFINEMENT_OUTPUT_ROOT to a drive with more free space "
            "or free space on this drive. Compression is lossless, but its "
            "size reduction is not guaranteed.",
            str(output_root),
        )
    return required


def _planned_output_artifacts(
    output_root: Path,
    prediction_output_stub: Path,
) -> dict[str, Path]:
    """Return every case-scoped artifact written by one inference run."""
    if prediction_output_stub.name != str(prediction_output_stub):
        raise ValueError(
            "PREDICTION_OUTPUT_NAMES entries must be plain filenames, not paths; "
            f"got {prediction_output_stub!s}."
        )
    if prediction_output_stub.suffix.lower() != ".nc":
        raise ValueError(
            "PREDICTION_OUTPUT_NAMES entries must end in '.nc'; "
            f"got {prediction_output_stub!s}."
        )
    deterministic_stub = prediction_output_stub.with_name(
        f"{prediction_output_stub.stem}.baseline{prediction_output_stub.suffix}"
    )
    constraint_stub = prediction_output_stub.with_name(
        f"{prediction_output_stub.stem}.constraint_diagnostics{prediction_output_stub.suffix}"
    )
    artifacts = {
        "prediction_netcdf": output_root / prediction_output_stub,
        "baseline_netcdf": output_root / deterministic_stub,
        "constraint_diagnostics_netcdf": output_root / constraint_stub,
        "parameters_json": output_root / prediction_output_stub.with_suffix(".json"),
        "diagnostics_json": output_root
        / prediction_output_stub.with_suffix(".diagnostics.json"),
        "distribution_plot": output_root
        / prediction_output_stub.with_suffix(".distribution.png"),
    }
    if SAVE_SAMPLING_STATES:
        artifacts["sampling_states"] = output_root / prediction_output_stub.with_suffix(".sampling_states.npz")
    if SAVE_PICKLE_OUTPUTS:
        artifacts.update({
            "ensemble_pickle": output_root / prediction_output_stub.with_suffix(".pkl"),
            "pre_inverse_pickle": output_root
            / prediction_output_stub.with_suffix(".pre_inverse.pkl"),
        })
    return artifacts


def _refuse_existing_output_artifacts(artifacts: dict[str, Path]) -> None:
    """Fail before inference rather than replace any part of an output bundle."""
    existing = [(name, path) for name, path in artifacts.items() if path.exists()]
    if not existing:
        return
    details = "\n".join(f"  - {name}: {path}" for name, path in existing)
    raise FileExistsError(
        "Refusing to overwrite an existing inference artifact. Choose a distinct "
        "GRANITE_REFINEMENT_OUTPUT_HEADER or deliberately remove/move the complete "
        f"old bundle first:\n{details}"
    )


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


def _refinement_output_header(config, run_name: str) -> str:
    """Return a filesystem-safe, refinement-specific output header."""
    def is_schema2_fixed(value: str) -> bool:
        return bool(
            re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9.-]*(?:_[A-Za-z0-9][A-Za-z0-9.-]*)*"
                r"_schema2(?:_[A-Za-z0-9][A-Za-z0-9.-]*)*_fixed",
                value,
            )
        )

    safe_default = (
        run_name
        if is_schema2_fixed(str(run_name))
        else f"{run_name}_schema2_fixed"
    )
    configured = REFINEMENT_OUTPUT_HEADER or safe_default
    header = str(configured).strip()
    if not header:
        refinement = getattr(config.model, "refinement", None) or {}
        header = str(refinement.get("type") or CONFIG_PATH.stem).strip()
    if not header or header in {".", ".."} or Path(header).name != header:
        raise ValueError(
            "GRANITE_REFINEMENT_OUTPUT_HEADER must be one non-empty directory name; "
            f"got {configured!r}"
        )
    if not is_schema2_fixed(header):
        raise ValueError(
            "GRANITE_REFINEMENT_OUTPUT_HEADER must be filesystem-safe and end "
            "in '_schema2_fixed' or '_schema2_<case>_fixed' so schema-2 "
            "inference cannot overwrite a legacy output directory; "
            f"got {header!r}."
        )
    return header


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


def _sampling_mode():
    mode = os.environ.get("GRANITE_REFINEMENT_SAMPLING_MODE", "legacy").lower()
    if mode not in ("legacy", "stable"):
        raise ValueError("GRANITE_REFINEMENT_SAMPLING_MODE must be legacy or stable.")
    return mode


def _run_full_inference(dataloader, model, device, target_vars, boundary_cfg):
    trace_parts = {}
    physical_stages = {}
    sampling_mode = _sampling_mode()
    first_sample_id = None
    def capture_trajectory(stage, step, process_time, state, member_start):
        key = f"{stage}_{step:04d}"
        trace_parts.setdefault(key, []).append((member_start, state[0].detach().float().cpu().numpy()))
    predictions = []
    deterministic_predictions = []
    ensemble_member_predictions = []
    unbounded_ensemble_means = []
    memberwise_clipped_means = []
    memberwise_clipping_mean_shifts = []
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

            stable_ids = None
            if sampling_mode == "stable":
                timestamps = batch.get("__sample_timestamp")
                paths = batch.get("__sample_predictor_path")
                if timestamps is None or paths is None:
                    raise ValueError("Stable inference requires source-file and exact timestamp metadata.")
                stable_ids = [f"{Path(path).name}|{stamp}" for path, stamp in zip(paths, timestamps)]
                if first_sample_id is None:
                    first_sample_id = stable_ids[0]
            batch = {
                key: (value.to(device=device, dtype=torch.float32, non_blocking=True)
                      if torch.is_tensor(value) else value)
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
                model.eval()
                if any(module.training for module in model.refiner.modules()):
                    raise RuntimeError(
                        "Refinement checkpoint was loaded with train-mode modules active."
                    )
                print(f"[checkpoint] loaded refinement {REFINEMENT_CHECKPOINT_NAME}: {report.summary()}")
            if boundary_cfg.enabled and not getattr(model, "_boundary_notice_printed", False):
                print("[refinement] full-frame sampling avoids independent tile-noise seams.")
                model._boundary_notice_printed = True
            with autocast_context():
                sampling_options = ({"sampling_mode": "stable"} if sampling_mode == "stable" else {})
                if sampling_mode == "stable":
                    sampling_options["sample_ids"] = stable_ids
                if SAVE_SAMPLING_STATES and batch_idx == 0:
                    if sampling_mode == "stable":
                        def identified_trace(sample_id, stage, step, process_time, state, member):
                            if sample_id == first_sample_id:
                                capture_trajectory(stage, step, process_time, state, member)
                        sampling_options["sample_trajectory_callback"] = identified_trace
                    else:
                        sampling_options["trajectory_callback"] = capture_trajectory
                refined = model.predict(
                    batch,
                    **sampling_options,
                    ensemble_size=REFINEMENT_ENSEMBLE_SIZE,
                    seed=int(model.refinement_config.seed) + (batch_idx if sampling_mode == "legacy" else 0),
                    return_members=True,
                )
                if refined.refined is None or refined.refined_normalized is None or refined.members is None:
                    raise RuntimeError("The active refinement model returned incomplete ensemble output.")
                if (
                    refined.unbounded_ensemble_mean is None
                    or refined.memberwise_clipped_mean is None
                    or refined.memberwise_clipping_mean_shift is None
                ):
                    raise RuntimeError(
                        "The active refinement model did not expose physical-constraint "
                        "diagnostics. Refusing an untracked precipitation clamp."
                    )
                out = refined.refined
                out_raw_model = refined.refined_normalized

            if FORCE_OUTPUT_FLOAT32:
                out = out.float()
                out_raw_model = out_raw_model.float()

            out_cpu = out.detach().cpu()
            out_pre_inverse_cpu = out_raw_model.detach().cpu()

            if SAVE_SAMPLING_STATES and batch_idx == 0:
                physical_stages = {
                    "deterministic_physical": refined.deterministic[0].detach().float().cpu().numpy(),
                    "reconstructed_unbounded_physical": refined.members_unbounded[0].detach().float().cpu().numpy(),
                    "postprocessed_physical": refined.members[0].detach().float().cpu().numpy(),
                    "effective_physical_residual": refined.member_residuals_physical[0].detach().float().cpu().numpy(),
                }
            predictions.append(out_cpu)
            deterministic_predictions.append(refined.deterministic.detach().cpu().float())
            ensemble_member_predictions.append(refined.members.detach().cpu().float())
            unbounded_ensemble_means.append(
                refined.unbounded_ensemble_mean.detach().cpu().float()
            )
            memberwise_clipped_means.append(
                refined.memberwise_clipped_mean.detach().cpu().float()
            )
            memberwise_clipping_mean_shifts.append(
                refined.memberwise_clipping_mean_shift.detach().cpu().float()
            )
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
        "sampling_states": {
            **{key: np.concatenate([value for _, value in sorted(parts)], axis=0)
               for key, parts in trace_parts.items()},
            **physical_stages,
        },
        "predictions": torch.cat(predictions, dim=0),
        "deterministic_predictions": torch.cat(deterministic_predictions, dim=0),
        "ensemble_member_predictions": torch.cat(ensemble_member_predictions, dim=0),
        "unbounded_ensemble_means": torch.cat(unbounded_ensemble_means, dim=0),
        "memberwise_clipped_means": torch.cat(memberwise_clipped_means, dim=0),
        "memberwise_clipping_mean_shifts": torch.cat(
            memberwise_clipping_mean_shifts, dim=0
        ),
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


def _concat_time_auxiliary(
    paths: list[str],
    time_key: str,
    variable_name: str | None,
) -> xr.DataArray | None:
    """Concatenate a predictor-time auxiliary, or return ``None`` if incomplete."""
    if not variable_name:
        return None
    arrays: list[xr.DataArray] = []
    for path in paths:
        with xr.open_dataset(path) as ds:
            if variable_name not in ds.variables:
                return None
            variable = ds[variable_name]
            if time_key not in variable.dims:
                return None
            arrays.append(variable.load())
    if not arrays:
        return None
    return xr.concat(arrays, dim=time_key).load()


def _sanitize_data_attrs(attrs: dict[str, object]) -> dict[str, object]:
    clean_attrs = dict(attrs)
    for key in ("scale_factor", "add_offset", "_FillValue", "missing_value", "dtype"):
        clean_attrs.pop(key, None)
    return clean_attrs


def _physical_output_variable_attrs(
    template_attrs: dict[str, dict[str, object]],
    target_vars: list[str],
    physical_units: list[str],
) -> dict[str, dict[str, object]]:
    """Return channel-aligned metadata for already-decoded physical outputs.

    The target template can describe precipitation as a per-second flux even
    though CordexDataset has converted it to mm/day before the scalers and
    models see it. Use the dataset's public, channel-aligned unit contract
    instead of guessing output units from variable names.
    """
    if len(physical_units) != len(target_vars):
        raise ValueError(
            "Physical output units must align one-to-one with target_vars: "
            f"got {len(physical_units)} units for {len(target_vars)} variables."
        )

    output: dict[str, dict[str, object]] = {}
    for name, units_value in zip(target_vars, physical_units):
        units = str(units_value).strip()
        if not units:
            raise ValueError(
                f"Physical output units are missing for target variable {name!r}."
            )
        attrs = _sanitize_data_attrs(dict(template_attrs.get(name, {})))
        source_units = str(attrs.pop("source_units", attrs.get("units", ""))).strip()
        attrs.pop("unit_conversion", None)
        attrs["units"] = units
        if source_units and source_units != units:
            attrs["source_units"] = source_units
            attrs["unit_conversion"] = (
                "target values converted exactly once by CordexDataset before "
                "training/inference; model products are decoded directly in the "
                "declared physical output units"
            )
        output[name] = attrs
    return output


def _cf_reference_names(attribute: str, value: object) -> list[str]:
    """Extract auxiliary-variable names from CF ``bounds``/``grid_mapping``."""
    text = str(value).strip()
    if not text:
        return []
    if attribute == "bounds":
        return text.split()
    if attribute != "grid_mapping":
        return []
    # CF's extended syntax is ``mapping_name: coordinate ...`` and may list
    # multiple mappings. The common syntax is simply ``mapping_name``.
    extended = re.findall(
        r"(?:^|\s)([A-Za-z_][A-Za-z0-9_.-]*)\s*:",
        text,
    )
    return extended or [text.split()[0]]


def _load_template_cf_auxiliaries(
    template_ds: xr.Dataset,
    data_attrs: dict[str, dict[str, object]],
    coordinates: dict[str, xr.DataArray],
) -> dict[str, xr.DataArray]:
    """Load CF auxiliaries referenced by output fields/coordinates."""
    referenced: set[str] = set()
    for attrs in data_attrs.values():
        for attribute in ("grid_mapping", "bounds"):
            referenced.update(_cf_reference_names(attribute, attrs.get(attribute, "")))
    for coordinate in coordinates.values():
        referenced.update(
            _cf_reference_names("bounds", coordinate.attrs.get("bounds", ""))
        )
    return {
        name: template_ds[name].load()
        for name in sorted(referenced)
        if name in template_ds.variables
    }


def _attach_cf_auxiliaries(
    dataset: xr.Dataset,
    auxiliaries: dict[str, xr.DataArray],
) -> tuple[list[str], list[str]]:
    """Attach shape-compatible auxiliaries and remove every dangling CF ref."""
    copied: list[str] = []
    for name, variable in auxiliaries.items():
        if name in dataset.variables:
            continue
        incompatible = any(
            dim in dataset.sizes and int(dataset.sizes[dim]) != int(size)
            for dim, size in variable.sizes.items()
        )
        if incompatible:
            continue
        dataset[name] = variable.copy(deep=True)
        copied.append(name)

    removed: list[str] = []
    for variable_name in list(dataset.variables):
        attrs = dataset[variable_name].attrs
        for attribute in ("grid_mapping", "bounds"):
            if attribute not in attrs:
                continue
            references = _cf_reference_names(attribute, attrs[attribute])
            if not references or any(
                reference not in dataset.variables for reference in references
            ):
                attrs.pop(attribute, None)
                removed.append(f"{variable_name}:{attribute}")
    return copied, removed


def _model_product_global_attrs(
    template_attrs: dict[str, object],
    *,
    product_kind: str,
    target_template_path: Path,
    refinement_type: str,
    refinement_case: str,
) -> dict[str, object]:
    """Build truthful model-product metadata without inheriting truth history."""
    if product_kind == "refinement_ensemble_members":
        title = "Granite-WxC stochastic residual-refinement ensemble prediction"
        summary = (
            "Physical ensemble members reconstructed from a frozen Phase-1 U-Net "
            f"and a {refinement_type} residual-refinement head."
        )
        source = f"Granite-WxC Phase-1 U-Net plus {refinement_type} inference"
    elif product_kind == "phase1_unet_deterministic":
        title = "Granite-WxC deterministic Phase-1 U-Net prediction"
        summary = (
            "Physical deterministic prediction from the frozen Phase-1 U-Net used "
            "as the paired residual-refinement baseline."
        )
        source = "Granite-WxC Phase-1 U-Net inference"
    elif product_kind == "refinement_constraint_diagnostics":
        title = "Granite-WxC residual-refinement physical-constraint diagnostics"
        summary = (
            "Physical ensemble means before and under the configured final "
            "nonnegativity constraint, including the memberwise-clipping mean shift."
        )
        source = f"Granite-WxC {refinement_type} inference diagnostics"
    else:
        raise ValueError(f"Unsupported model product kind {product_kind!r}.")

    conventions = template_attrs.get("Conventions") or template_attrs.get(
        "conventions", "CF-1.8"
    )
    attrs: dict[str, object] = {
        "Conventions": str(conventions),
        "title": title,
        "summary": summary,
        "source": source,
        "history": (
            f"{datetime.now(timezone.utc).isoformat()}: generated by "
            f"{Path(__file__).name}"
        ),
        "product_kind": product_kind,
        "refinement_case": refinement_case,
        "refinement_sampling_mode": _sampling_mode(),
        "refinement_sampling_seed_rule": (
            "SHA256(stream_version,experiment_seed,predictor_filename|civil_timestamp,member_index,innovation_index)"
            if _sampling_mode() == "stable" else "legacy experiment_seed + batch_index"
        ),
        "refinement_randomness_stream_version": (
            "refinement.sample_member_innovation.v1" if _sampling_mode() == "stable" else "legacy"
        ),
        "refinement_canonical_neural_batch_size": 1 if _sampling_mode() == "stable" else -1,
        "refinement_member_nesting": "raw members nested; rerun coupled precipitation constraint for each ensemble size",
        "input_target_template": str(target_template_path.resolve()),
    }
    template_title = str(template_attrs.get("title", "")).strip()
    if template_title:
        attrs["input_target_template_title"] = template_title
    return attrs


def _build_constraint_diagnostics_dataset(
    *,
    target_vars: list[str],
    unbounded_ensemble_means: np.ndarray,
    memberwise_clipped_means: np.ndarray,
    memberwise_clipping_mean_shifts: np.ndarray,
    coords: dict[str, object],
    time_dim: str,
    lat_dim: str,
    lon_dim: str,
    target_attrs: dict[str, dict[str, object]],
    template_attrs: dict[str, object],
    target_template_path: Path,
    refinement_type: str,
    refinement_case: str,
    output_provenance: dict[str, object],
    nonnegative_ensemble_strategy: str,
    cf_auxiliaries: dict[str, xr.DataArray],
) -> xr.Dataset:
    """Build the separately persisted physical-constraint diagnostic product."""
    diagnostic_arrays = {
        "unbounded_ensemble_mean": np.asarray(unbounded_ensemble_means),
        "memberwise_clipped_mean": np.asarray(memberwise_clipped_means),
        "memberwise_clipping_mean_shift": np.asarray(
            memberwise_clipping_mean_shifts
        ),
    }
    reference_shape = diagnostic_arrays["unbounded_ensemble_mean"].shape
    if len(reference_shape) != 4 or reference_shape[1] != len(target_vars):
        raise ValueError(
            "Constraint diagnostics must use [time, channel, lat, lon] layout; "
            f"got {reference_shape} for target_vars={target_vars}."
        )
    for diagnostic_name, values in diagnostic_arrays.items():
        if values.shape != reference_shape:
            raise ValueError(
                "Constraint diagnostic arrays must have identical shapes; "
                f"unbounded_ensemble_mean={reference_shape}, "
                f"{diagnostic_name}={values.shape}."
            )

    constraint_ds = xr.Dataset(coords=coords)
    long_name_templates = {
        "unbounded_ensemble_mean": (
            "{name} ensemble mean before the final physical constraint"
        ),
        "memberwise_clipped_mean": (
            "{name} diagnostic mean under independent memberwise clipping"
        ),
        "memberwise_clipping_mean_shift": (
            "{name} memberwise-clipping mean shift relative to the unbounded "
            "ensemble mean"
        ),
    }
    for var_idx, name in enumerate(target_vars):
        if name not in target_attrs:
            raise ValueError(f"Missing physical output metadata for {name!r}.")
        units = str(target_attrs[name].get("units", "")).strip()
        if not units:
            raise ValueError(
                f"Physical output units are missing for target variable {name!r}."
            )
        for diagnostic_kind, values in diagnostic_arrays.items():
            diagnostic_name = f"{name}_{diagnostic_kind}"
            attrs = dict(target_attrs[name])
            # Derived diagnostics must not masquerade as the template's
            # original CF standard variable or inherit source validity bounds.
            for key in (
                "standard_name",
                "valid_min",
                "valid_max",
                "valid_range",
                "actual_range",
            ):
                attrs.pop(key, None)
            attrs.update(
                {
                    "units": units,
                    "source_predictand": name,
                    "diagnostic_quantity": diagnostic_kind,
                    "refinement_space": "physical",
                    "long_name": long_name_templates[diagnostic_kind].format(
                        name=name
                    ),
                }
            )
            constraint_ds[diagnostic_name] = xr.DataArray(
                values[:, var_idx],
                dims=(time_dim, lat_dim, lon_dim),
                coords=coords,
                attrs=attrs,
            ).astype(np.float32)

    constraint_ds.attrs.update(
        _model_product_global_attrs(
            template_attrs,
            product_kind="refinement_constraint_diagnostics",
            target_template_path=target_template_path,
            refinement_type=refinement_type,
            refinement_case=refinement_case,
        )
    )
    constraint_ds.attrs.update(output_provenance)
    constraint_ds.attrs["nonnegative_ensemble_strategy"] = str(
        nonnegative_ensemble_strategy
    )
    constraint_ds.attrs["diagnostic_semantics"] = (
        "All fields are computed after inverse residual normalization and "
        "Phase-1 addition. No intermediate flow/diffusion state is clipped."
    )
    copied_cf, removed_cf = _attach_cf_auxiliaries(
        constraint_ds,
        cf_auxiliaries,
    )
    constraint_ds.attrs["cf_auxiliary_variables"] = json.dumps(copied_cf)
    if removed_cf:
        constraint_ds.attrs["removed_dangling_cf_references"] = json.dumps(
            removed_cf
        )
    return constraint_ds


def _build_netcdf_encoding(var_names: list[str]) -> dict[str, dict[str, object]]:
    fill = np.float32(np.nan)
    return {
        name: {
            "dtype": NETCDF_OUTPUT_DTYPE,
            "_FillValue": fill,
            "zlib": True,
            "complevel": NETCDF_COMPRESSION_LEVEL,
            "shuffle": True,
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
    base_dataset = build_inference_dataset(
        config,
        predictor_paths,
        target_paths,
        allow_time_mismatch=True,
    )
    dataset = CordexWrappedDataset(base_dataset, preserve_metadata=_sampling_mode() == "stable")
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


def _resolve_project_path(value: str | os.PathLike) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def _resolve_yaml_path(value: str | os.PathLike, config_path: Path) -> Path:
    """Resolve a path using the directory containing its YAML as the base."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


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


def _resolve_refinement_checkpoint(config, config_path: Path = CONFIG_PATH) -> Path:
    """Resolve the refinement checkpoint declared by ``config_path``.

    ``model.refinement.checkpoint`` may name an explicit checkpoint. When it
    is unset, ``checkpoint_dir/<configured name>`` is used (``best.ckpt`` by
    default, overridable with ``GRANITE_REFINEMENT_CHECKPOINT``). Relative paths in both
    fields are relative to the YAML file, never to the process working
    directory.
    """
    config_path = Path(config_path).expanduser().resolve()
    refinement_config = getattr(config.model, "refinement", None) or {}
    explicit_checkpoint = refinement_config.get("checkpoint")

    if explicit_checkpoint:
        checkpoint_path = _resolve_yaml_path(explicit_checkpoint, config_path)
        if checkpoint_path.is_dir():
            checkpoint_path /= REFINEMENT_CHECKPOINT_NAME
        source = "model.refinement.checkpoint"
    else:
        configured_dir = getattr(config, "checkpoint_dir", None)
        if not configured_dir:
            raise ValueError(
                f"{config_path} must define checkpoint_dir or "
                "model.refinement.checkpoint"
            )
        checkpoint_path = (
            _resolve_yaml_path(configured_dir, config_path) / REFINEMENT_CHECKPOINT_NAME
        )
        source = "checkpoint_dir"

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Refinement checkpoint from {source} in {config_path} was not found: "
            f"{checkpoint_path}"
        )
    return checkpoint_path


def _load_refinement_checkpoint(path: Path, expected_type: str) -> dict:
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise RuntimeError(f"Invalid refinement checkpoint payload: {path}")

    checkpoint_type = payload.get("refinement_type")
    if checkpoint_type and checkpoint_type != expected_type:
        raise RuntimeError(
            f"Refinement checkpoint type mismatch: YAML requests {expected_type!r}, "
            f"but {path} contains {checkpoint_type!r}"
        )
    return payload


def _loaded_normalization_provenance(model) -> dict[str, object]:
    """Serialize fitted statistics after the lazy checkpoint has been restored."""
    if not getattr(model, "_last_checkpoint_loaded", False):
        raise RuntimeError("Cannot serialize residual statistics before checkpoint loading.")
    metadata = model.residual_normalization_metadata()
    if not metadata.get("fitted", False):
        raise RuntimeError("Inference output requires fitted checkpoint residual statistics.")
    from granitewxc.refinement.checkpoint import _state_fingerprint

    state = {
        name: tensor for name, tensor in model.state_dict().items()
        if name.startswith("residual_normalizer.")
    }
    if not state:
        raise RuntimeError("Loaded refinement has no residual-normalizer state.")
    fingerprint = _state_fingerprint(state)
    expected = model._last_checkpoint_payload["residual_normalizer_state_fingerprint"]
    if fingerprint != expected:
        raise RuntimeError("Loaded residual statistics differ from the checkpoint fingerprint.")
    return {
        "residual_normalization": json.dumps(metadata, sort_keys=True, allow_nan=False),
        "residual_normalizer_state_fingerprint": fingerprint,
    }


def _load_model_and_config() -> tuple[str, Path, Path, object, torch.nn.Module, Path]:
    global DATASET_ROOT, REFINEMENT_ENSEMBLE_SIZE
    config = get_config(str(CONFIG_PATH))
    assert_no_eccc_reference(CONFIG_PATH)
    DATASET_ROOT = _resolve_dataset_root(config)
    print(f"[config] source={CONFIG_PATH}")
    print(f"[data] YAML-derived dataset root={DATASET_ROOT}")
    refinement_config = getattr(config.model, "refinement", None) or {}
    yaml_ensemble_size = int(refinement_config.get("ensemble_size", 1))
    if REFINEMENT_ENSEMBLE_SIZE is None:
        REFINEMENT_ENSEMBLE_SIZE = yaml_ensemble_size
        ensemble_source = "model.refinement.ensemble_size"
    else:
        ensemble_source = "GRANITE_REFINEMENT_ENSEMBLE_SIZE"
    if REFINEMENT_ENSEMBLE_SIZE < 1:
        raise ValueError(
            f"Refinement ensemble size must be >= 1, got {REFINEMENT_ENSEMBLE_SIZE}."
        )
    print(
        f"[refinement] ensemble_size={REFINEMENT_ENSEMBLE_SIZE} "
        f"(source={ensemble_source}, YAML={yaml_ensemble_size})"
    )
    config.dl_num_workers = int(NUM_WORKERS) if NUM_WORKERS is not None else config.dl_num_workers
    config.batch_size = int(INFERENCE_BATCH_SIZE) if INFERENCE_BATCH_SIZE is not None else config.batch_size
    config.device_target = DEVICE_TARGET
    config.path_experiment = str(_resolve_project_path(config.path_experiment))
    if USE_STATIC is not None:
        config.data.use_static = bool(USE_STATIC)
    if STATIC_PATH:
        config.data.static_path = str(Path(STATIC_PATH).resolve())

    phase1_path = _resolve_repo_path(config.model.phase1["checkpoint"])
    refinement_path = _resolve_refinement_checkpoint(config, CONFIG_PATH)
    if not phase1_path.is_file():
        raise FileNotFoundError(f"Phase-1 checkpoint not found: {phase1_path}")
    assert_no_eccc_reference(phase1_path)
    assert_no_eccc_reference(refinement_path)

    model = build_two_phase_model(get_finetune_model_UNET(config), config)
    phase1_payload = torch.load(str(phase1_path), map_location="cpu", weights_only=True)
    print(f"[checkpoint] loaded Phase 1: {load_phase1_state_dict(model, phase1_payload).summary()}")
    refinement_payload = _load_refinement_checkpoint(
        refinement_path, str(config.model.refinement["type"])
    )
    validate_phase1_reference(refinement_payload, model.phase1.state_dict(), strict=True)
    model._last_checkpoint_payload = refinement_payload
    model._last_checkpoint_loaded = False
    model._loaded_phase1_checkpoint_path = phase1_path

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
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    print(f"[checkpoint] refinement file sha256={checkpoint_sha256}")
    refinement_output_header = _refinement_output_header(config, run_name)
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
        inference_output_root = _prediction_output_root(
            run_dir, refinement_output_header, INFERENCE_OUTPUT_SUBDIRS[idx]
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
            test_target_paths=[(DATASET_ROOT / TARGET_FILES[idx]).resolve()],
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
        prediction_output_stub = Path(
            prediction_output_name or f"{run_name}_predictions_{idx + 1:02d}.nc"
        )
        output_artifacts = _planned_output_artifacts(
            output_root,
            prediction_output_stub,
        )
        _refuse_existing_output_artifacts(output_artifacts)

        predictor_var_names = build_predictor_names(config)
        test_predictor_paths = _repair_predictor_files_if_needed(
            test_predictor_paths,
            output_root=output_root,
            predictor_var_names=predictor_var_names,
            run_index=idx + 1,
        )

        config.data.test_predictor_paths = test_predictor_paths
        config.data.test_target_paths = _resolve_inputs(params.test_target_paths)

        refinement_type = str(config.model.refinement["type"])
        checkpoint_payload = model._last_checkpoint_payload
        scientific_contract = checkpoint_payload.get("refinement_contract")
        if not isinstance(scientific_contract, dict):
            raise RuntimeError(
                "Loaded refinement checkpoint has no serialized scientific contract."
            )
        phase1_fingerprint = str(
            checkpoint_payload.get("phase1_fingerprint", "")
        ).strip()
        contract_fingerprint = str(
            checkpoint_payload.get("refinement_contract_fingerprint", "")
        ).strip()
        if not phase1_fingerprint or not contract_fingerprint:
            raise RuntimeError(
                "Loaded refinement checkpoint lacks Phase-1 or refinement-contract "
                "fingerprint provenance."
            )
        output_provenance = {
            "checkpoint_schema_version": int(
                checkpoint_payload["checkpoint_schema_version"]
            ),
            "refinement_contract_version": int(
                scientific_contract["contract_version"]
            ),
            "residual_contract": str(scientific_contract["residual_contract"]),
            "refinement_contract_fingerprint": contract_fingerprint,
            "refinement_checkpoint_sha256": checkpoint_sha256,
            "phase1_checkpoint": str(model._loaded_phase1_checkpoint_path),
            "phase1_fingerprint": phase1_fingerprint,
            "refinement_config_path": str(CONFIG_PATH),
            "refinement_seed": int(config.model.refinement.get("seed", -1)),
        }
        if (
            output_provenance["checkpoint_schema_version"]
            != CHECKPOINT_SCHEMA_VERSION
            or output_provenance["refinement_contract_version"]
            != REFINEMENT_CONTRACT_VERSION
        ):
            raise RuntimeError(
                "Refusing to write output from an obsolete refinement checkpoint "
                f"contract: {output_provenance}."
            )
        test_dl = _build_dataloader(config, config.data.test_predictor_paths, config.data.test_target_paths, device)
        base_dataset = test_dl.dataset.base
        target_vars = list(base_dataset.target_vars)
        output_budget_bytes = _check_output_disk_space(
            output_root,
            sample_count=len(base_dataset),
            target_shape=base_dataset.crop_size,
            variable_count=len(target_vars),
            ensemble_size=REFINEMENT_ENSEMBLE_SIZE,
        )
        params_json = export_params(
            params,
            output_artifacts["parameters_json"],
            extra={
                "run_name": run_name,
                "checkpoint": str(checkpoint_path),
                "refinement_output_header": refinement_output_header,
                "refinement_type": refinement_type,
                "netcdf_compression_level": NETCDF_COMPRESSION_LEVEL,
                "save_pickle_outputs": SAVE_PICKLE_OUTPUTS,
                "output_disk_budget_bytes": output_budget_bytes,
            },
        )

        print(f"[{active_position:02d}/{active_run_count}] Using predictors from {Path(test_predictor_paths[0]).parent}")
        print(f"[{active_position:02d}/{active_run_count}] Output directory: {output_root}")
        print(f"[{active_position:02d}/{active_run_count}] Saved parameter snapshot to {params_json}")

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
        output_provenance.update(_loaded_normalization_provenance(model))
        if SAVE_SAMPLING_STATES:
            np.savez_compressed(output_artifacts["sampling_states"], **inference_result["sampling_states"])
        full_outputs = inference_result["predictions"]
        full_deterministic_outputs = inference_result["deterministic_predictions"]
        full_ensemble_member_outputs = inference_result["ensemble_member_predictions"]
        full_unbounded_ensemble_means = inference_result["unbounded_ensemble_means"]
        full_memberwise_clipped_means = inference_result["memberwise_clipped_means"]
        full_memberwise_clipping_mean_shifts = inference_result[
            "memberwise_clipping_mean_shifts"
        ]
        full_outputs_pre_inverse = inference_result["predictions_pre_inverse"]
        full_targets = inference_result["targets"]

        outputs_np = full_outputs.numpy().astype(np.float32, copy=False)
        deterministic_outputs_np = full_deterministic_outputs.numpy().astype(np.float32, copy=False)
        ensemble_member_outputs_np = full_ensemble_member_outputs.numpy().astype(np.float32, copy=False)
        unbounded_ensemble_means_np = full_unbounded_ensemble_means.numpy().astype(
            np.float32, copy=False
        )
        memberwise_clipped_means_np = full_memberwise_clipped_means.numpy().astype(
            np.float32, copy=False
        )
        memberwise_clipping_mean_shifts_np = (
            full_memberwise_clipping_mean_shifts.numpy().astype(
                np.float32, copy=False
            )
        )
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
            diagnostics_path = output_artifacts["diagnostics_json"]
            save_json(diagnostics_payload, diagnostics_path)
            print(f"[diag] Saved diagnostics JSON to {diagnostics_path}")
        if SAVE_DISTRIBUTION_PLOT:
            if "tasmax" in predicted_var_values and "pr" in predicted_var_values:
                plot_path = output_artifacts["distribution_plot"]
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
        time_bounds_name = str(
            predictor_time_coord.attrs.get("bounds", "")
        ).strip()
        predictor_time_bounds = _concat_time_auxiliary(
            predictor_paths,
            time_dim,
            time_bounds_name or None,
        )

        if not target_template_paths:
            raise FileNotFoundError("Target template paths missing; ensure config.data.test_target_paths is set")

        target_template_path = Path(target_template_paths[0]).resolve()
        with xr.open_dataset(target_template_path, engine="h5netcdf") as template_ds:
            template_attrs = dict(template_ds.attrs)
            raw_target_attrs = {
                name: dict(template_ds[name].attrs)
                for name in target_vars
                if name in template_ds.data_vars
            }
            target_attrs = _physical_output_variable_attrs(
                raw_target_attrs,
                target_vars,
                list(getattr(base_dataset, "target_units", [])),
            )
            lat_coord = template_ds[lat_name].load()
            lon_coord = template_ds[lon_name].load()
            template_cf_auxiliaries = _load_template_cf_auxiliaries(
                template_ds,
                target_attrs,
                {
                    time_dim: predictor_time_coord,
                    lat_dim: lat_coord,
                    lon_dim: lon_coord,
                },
            )
        if time_bounds_name and predictor_time_bounds is not None:
            template_cf_auxiliaries[time_bounds_name] = predictor_time_bounds

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

        prediction_ds.attrs.update(
            _model_product_global_attrs(
                template_attrs,
                product_kind="refinement_ensemble_members",
                target_template_path=target_template_path,
                refinement_type=refinement_type,
                refinement_case=refinement_output_header,
            )
        )
        prediction_ds.attrs["prediction_kind"] = "refinement_ensemble_members"
        prediction_ds.attrs["refinement_case"] = refinement_output_header
        prediction_ds.attrs["refinement_type"] = refinement_type
        prediction_ds.attrs["refinement_checkpoint"] = str(checkpoint_path)
        prediction_ds.attrs["ensemble_size"] = REFINEMENT_ENSEMBLE_SIZE
        prediction_ds.attrs.update(output_provenance)
        for name in target_vars:
            prediction_ds[name].attrs["refinement_space"] = (
                "physical member after inverse residual normalization, "
                "Phase-1 addition and physical constraints"
            )
        copied_cf, removed_cf = _attach_cf_auxiliaries(
            prediction_ds,
            template_cf_auxiliaries,
        )
        prediction_ds.attrs["cf_auxiliary_variables"] = json.dumps(copied_cf)
        if removed_cf:
            prediction_ds.attrs["removed_dangling_cf_references"] = json.dumps(
                removed_cf
            )
        prediction_output_path = output_artifacts["prediction_netcdf"]
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
        deterministic_ds.attrs.update(
            _model_product_global_attrs(
                template_attrs,
                product_kind="phase1_unet_deterministic",
                target_template_path=target_template_path,
                refinement_type=refinement_type,
                refinement_case=refinement_output_header,
            )
        )
        deterministic_ds.attrs["prediction_kind"] = "phase1_unet_deterministic"
        deterministic_ds.attrs["inference_case"] = refinement_output_header
        deterministic_ds.attrs["phase1_checkpoint"] = output_provenance[
            "phase1_checkpoint"
        ]
        deterministic_ds.attrs["phase1_fingerprint"] = output_provenance[
            "phase1_fingerprint"
        ]
        deterministic_ds.attrs["paired_refinement_type"] = refinement_type
        deterministic_ds.attrs["paired_refinement_checkpoint"] = str(
            checkpoint_path
        )
        deterministic_ds.attrs["paired_refinement_checkpoint_sha256"] = (
            checkpoint_sha256
        )
        baseline_copied_cf, baseline_removed_cf = _attach_cf_auxiliaries(
            deterministic_ds,
            template_cf_auxiliaries,
        )
        deterministic_ds.attrs["cf_auxiliary_variables"] = json.dumps(
            baseline_copied_cf
        )
        if baseline_removed_cf:
            deterministic_ds.attrs["removed_dangling_cf_references"] = (
                json.dumps(baseline_removed_cf)
            )
        deterministic_output_path = output_artifacts["baseline_netcdf"]
        deterministic_ds.to_netcdf(
            deterministic_output_path,
            engine="h5netcdf",
            encoding=_build_netcdf_encoding(target_vars),
        )
        print(f"[{active_position:02d}/{active_run_count}] Saved deterministic U-Net output to {deterministic_output_path}")

        constraint_ds = _build_constraint_diagnostics_dataset(
            target_vars=target_vars,
            unbounded_ensemble_means=unbounded_ensemble_means_np,
            memberwise_clipped_means=memberwise_clipped_means_np,
            memberwise_clipping_mean_shifts=memberwise_clipping_mean_shifts_np,
            coords=coords,
            time_dim=time_dim,
            lat_dim=lat_dim,
            lon_dim=lon_dim,
            target_attrs=target_attrs,
            template_attrs=template_attrs,
            target_template_path=target_template_path,
            refinement_type=refinement_type,
            refinement_case=refinement_output_header,
            output_provenance=output_provenance,
            nonnegative_ensemble_strategy=(
                model.refinement_config.nonnegative_ensemble_strategy
            ),
            cf_auxiliaries=template_cf_auxiliaries,
        )
        constraint_names = [
            f"{name}_{diagnostic_kind}"
            for name in target_vars
            for diagnostic_kind in (
                "unbounded_ensemble_mean",
                "memberwise_clipped_mean",
                "memberwise_clipping_mean_shift",
            )
        ]
        constraint_output_path = output_artifacts[
            "constraint_diagnostics_netcdf"
        ]
        constraint_ds.to_netcdf(
            constraint_output_path,
            engine="h5netcdf",
            encoding=_build_netcdf_encoding(constraint_names),
        )
        print(
            f"[{active_position:02d}/{active_run_count}] Saved physical-constraint "
            f"diagnostics to {constraint_output_path}"
        )

        if SAVE_PICKLE_OUTPUTS:
            import pickle

            for artifact_name, values in (
                ("ensemble_pickle", ensemble_member_outputs_np),
                ("pre_inverse_pickle", outputs_pre_inverse_np),
            ):
                pickle_path = output_artifacts[artifact_name]
                with open(pickle_path, "wb") as handle:
                    pickle.dump(values, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"[{active_position:02d}/{active_run_count}] Saved {artifact_name} to {pickle_path}")

    if DIST_ENABLED and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--scientific-experiment":
        from granitewxc.refinement.scientific_entrypoints import dispatch_scientific
        raise SystemExit(dispatch_scientific(sys.argv[1:], expected_operation="predict"))
    main()
