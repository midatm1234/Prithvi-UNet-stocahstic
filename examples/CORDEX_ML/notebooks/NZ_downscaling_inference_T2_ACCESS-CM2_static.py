#!/usr/bin/env python
"""Python version of NZ_downscaling_inference_T2_ACCESS-CM2_static.ipynb.

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
from utils.postprocess_outputs import enforce_pr_nonnegative_xr  # noqa: E402
from granitewxc.models.model import get_finetune_model_UNET  # noqa: E402
from granitewxc.utils.config import get_config  # noqa: E402
from nz_params import (  # noqa: E402
    UserParams,
    export_params,
    resolve_checkpoint,
    resolve_existing_run_dir,
    validate_paths,
)
from run_utils import assert_no_eccc_reference, load_run_manifest  # noqa: E402


# ===================== USER PARAMETERS (EDIT ME) =====================
NUM_RUNS = 12

# Each list must have NUM_RUNS entries.
TEST_SPLITS = ["test/historical/predictors/perfect","test/historical/predictors/perfect",
        "test/historical/predictors/imperfect", "test/historical/predictors/imperfect",
        "test/mid_century/predictors/perfect","test/mid_century/predictors/perfect",
        "test/mid_century/predictors/imperfect", "test/mid_century/predictors/imperfect",
        "test/end_century/predictors/perfect","test/end_century/predictors/perfect",
        "test/end_century/predictors/imperfect", "test/end_century/predictors/imperfect"
]

PREDICTOR_FILES = ["ACCESS-CM2_1981-2000_regridded.nc","EC-Earth3_1981-2000_regridded.nc",
        "ACCESS-CM2_1981-2000_regridded.nc","EC-Earth3_1981-2000_regridded.nc",
        "ACCESS-CM2_2041-2060_regridded.nc","EC-Earth3_2041-2060_regridded.nc",
        "ACCESS-CM2_2041-2060_regridded.nc","EC-Earth3_2041-2060_regridded.nc",
        "ACCESS-CM2_2080-2099_regridded.nc","EC-Earth3_2080-2099_regridded.nc",
        "ACCESS-CM2_2080-2099_regridded.nc","EC-Earth3_2080-2099_regridded.nc"
]

PREDICTION_OUTPUT_NAMES = ["Predictions_pr_tasmax_ACCESS-CM2_1981-2000.nc", "Predictions_pr_tasmax_EC-Earth3_1981-2000.nc",
        "Predictions_pr_tasmax_ACCESS-CM2_1981-2000.nc", "Predictions_pr_tasmax_EC-Earth3_1981-2000.nc",
        "Predictions_pr_tasmax_ACCESS-CM2_2041-2060.nc", "Predictions_pr_tasmax_EC-Earth3_2041-2060.nc",
        "Predictions_pr_tasmax_ACCESS-CM2_2041-2060.nc", "Predictions_pr_tasmax_EC-Earth3_2041-2060.nc",
        "Predictions_pr_tasmax_ACCESS-CM2_2080-2099.nc", "Predictions_pr_tasmax_EC-Earth3_2080-2099.nc",
        "Predictions_pr_tasmax_ACCESS-CM2_2080-2099.nc", "Predictions_pr_tasmax_EC-Earth3_2080-2099.nc"
]

INFERENCE_OUTPUT_ROOTS = ["/mnt/data2/kyo/granite-wxc/examples/CORDEX_ML/runs/NZ_T2_ACCESS-CM2_static_train/predictions/historical/perfect/",
        "/mnt/data2/kyo/granite-wxc/examples/CORDEX_ML/runs/NZ_T2_ACCESS-CM2_static_train/predictions/historical/perfect/",
        "/mnt/data2/kyo/granite-wxc/examples/CORDEX_ML/runs/NZ_T2_ACCESS-CM2_static_train/predictions/historical/imperfect/",
        "/mnt/data2/kyo/granite-wxc/examples/CORDEX_ML/runs/NZ_T2_ACCESS-CM2_static_train/predictions/historical/imperfect/",
        "/mnt/data2/kyo/granite-wxc/examples/CORDEX_ML/runs/NZ_T2_ACCESS-CM2_static_train/predictions/mid-century/perfect/",
        "/mnt/data2/kyo/granite-wxc/examples/CORDEX_ML/runs/NZ_T2_ACCESS-CM2_static_train/predictions/mid-century/perfect/",
        "/mnt/data2/kyo/granite-wxc/examples/CORDEX_ML/runs/NZ_T2_ACCESS-CM2_static_train/predictions/mid-century/imperfect/",
        "/mnt/data2/kyo/granite-wxc/examples/CORDEX_ML/runs/NZ_T2_ACCESS-CM2_static_train/predictions/mid-century/imperfect/",
        "/mnt/data2/kyo/granite-wxc/examples/CORDEX_ML/runs/NZ_T2_ACCESS-CM2_static_train/predictions/end-century/perfect/",
        "/mnt/data2/kyo/granite-wxc/examples/CORDEX_ML/runs/NZ_T2_ACCESS-CM2_static_train/predictions/end-century/perfect/",
        "/mnt/data2/kyo/granite-wxc/examples/CORDEX_ML/runs/NZ_T2_ACCESS-CM2_static_train/predictions/end-century/imperfect/",
        "/mnt/data2/kyo/granite-wxc/examples/CORDEX_ML/runs/NZ_T2_ACCESS-CM2_static_train/predictions/end-century/imperfect/"
]

# Fixed config for the fine-tuned model
REPO_ROOT = REPO_ROOT.resolve()
PROJECT_DIR = PROJECT_DIR.resolve()
DATASET_ROOT = REPO_ROOT / "granite-geospatial-wxc-downscaling/CORDEX/NZ_domain"
RUNS_ROOT = PROJECT_DIR / "runs/NZ_T2_ACCESS-CM2_static_train"
CONFIG_PATH = PROJECT_DIR / "NZ_T2_ACCESS-CM2_static.yaml"

TRAIN_SPLIT = "train/Emulator_hist_future"
TARGET_TEMPLATE_FILE = "pr_tasmax_ACCESS-CM2_1961-1980_2080-2099.nc"
TRAIN_TARGETS = [DATASET_ROOT / TRAIN_SPLIT / "target" / TARGET_TEMPLATE_FILE]

FINETUNE_RUN_NAME = "NZ_T2_ACCESS-CM2_static"  # set None to auto-pick latest
USE_STATIC = True
STATIC_PATH = None  # e.g., DATASET_ROOT / TRAIN_SPLIT / "predictors" / "Static_fields.nc"

REPAIR_INVALID_INPUTS = True
CLAMP_PR_NONNEGATIVE = True

DEVICE_TARGET = "cuda"
NUM_WORKERS = 2
BATCH_SIZE = None
PREFERRED_CHECKPOINT = "best"
MIN_FREE_GB = 8  # minimum free GPU memory to consider "idle"
MAX_GPUS = None  # set to an int to cap how many GPUs to expose
RESPECT_CUDA_VISIBLE_DEVICES = False  # ignore preset CUDA_VISIBLE_DEVICES when auto-selecting idle GPUs

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
    if target == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device_target is 'cuda' but no CUDA device is available.")
        return torch.device("cuda")
    if target == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_full_inference(dataloader, model, device):
    predictions = []
    autocast_enabled = device.type == "cuda"
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
        for batch in tqdm(dataloader, desc="Running inference", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast_context():
                out = model(batch)
            predictions.append(out.detach().cpu())
    return torch.cat(predictions, dim=0)


def _concat_time_coordinate(paths, time_key):
    arrays = []
    attrs = None
    encoding = None
    for path in paths:
        with xr.open_dataset(path, engine="netcdf4") as ds:
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
        with xr.open_dataset(source_path, engine="netcdf4") as ds:
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


def _load_model_and_config() -> tuple[str, Path, Path, object, torch.nn.Module]:
    base_params = UserParams(
        repo_root=REPO_ROOT,
        project_dir=PROJECT_DIR,
        runs_root=RUNS_ROOT,
        config_path=CONFIG_PATH,
        inference_run_name=FINETUNE_RUN_NAME,
        preferred_checkpoint=PREFERRED_CHECKPOINT,
        device_target=DEVICE_TARGET,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        use_static=USE_STATIC,
    )

    validate_paths(base_params)
    run_name, run_dir = resolve_existing_run_dir(base_params)
    manifest = load_run_manifest(run_dir)
    resolved_config_path = Path(manifest["config_snapshot"]).resolve()

    os.chdir(PROJECT_DIR)
    config = get_config(str(resolved_config_path))
    assert_no_eccc_reference(resolved_config_path)

    manifest_scalars = manifest.get("scalars", {})
    if isinstance(manifest_scalars, dict):
        model_scaler_map = {
            "model.input_mu": "input_mu",
            "model.input_sigma": "input_sigma",
            "model.target_mu": "target_mu",
            "model.target_sigma": "target_sigma",
        }
        for key, attr in model_scaler_map.items():
            path = manifest_scalars.get(key)
            if path:
                setattr(config.model, attr, path)
        data_scalers = manifest_scalars.get("data.scalers")
        if data_scalers:
            config.data.scalers = data_scalers

    if NUM_WORKERS is not None:
        config.dl_num_workers = int(NUM_WORKERS)
    if BATCH_SIZE is not None:
        config.batch_size = int(BATCH_SIZE)
    config.device_target = DEVICE_TARGET
    if USE_STATIC is not None:
        config.data.use_static = bool(USE_STATIC)
    if STATIC_PATH:
        config.data.static_path = str(Path(STATIC_PATH).resolve())

    checkpoint_path = resolve_checkpoint(base_params, run_dir)
    assert_no_eccc_reference(checkpoint_path)

    model = get_finetune_model_UNET(config)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model_state = model.state_dict()
    weights_have_module_prefix = all(key.startswith("module.") for key in state_dict.keys())
    model_expects_module_prefix = all(key.startswith("module.") for key in model_state.keys())

    if model_expects_module_prefix and not weights_have_module_prefix:
        state_dict = state_dict.__class__((("module." + key), value) for key, value in state_dict.items())
    elif weights_have_module_prefix and not model_expects_module_prefix:
        prefix_len = len("module.")
        state_dict = state_dict.__class__((key[prefix_len:], value) for key, value in state_dict.items())

    model.load_state_dict(state_dict, strict=True)

    return run_name, run_dir, checkpoint_path, config, model


def main() -> None:
    logging.disable(logging.CRITICAL)
    warnings.simplefilter(action="ignore", category=FutureWarning)

    _check_list_lengths()

    _configure_idle_gpus()

    torch.jit.enable_onednn_fusion(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    device = _select_device()
    run_name, run_dir, checkpoint_path, config, model = _load_model_and_config()
    model.to(device)

    for idx in range(NUM_RUNS):
        test_split = TEST_SPLITS[idx]
        predictor_file = PREDICTOR_FILES[idx]
        prediction_output_name = PREDICTION_OUTPUT_NAMES[idx]
        inference_output_root = INFERENCE_OUTPUT_ROOTS[idx]

        if predictor_file:
            test_predictors = [DATASET_ROOT / test_split / predictor_file]
        else:
            test_predictors = sorted((DATASET_ROOT / test_split).glob("*.nc"))

        params = UserParams(
            repo_root=REPO_ROOT,
            project_dir=PROJECT_DIR,
            runs_root=RUNS_ROOT,
            config_path=CONFIG_PATH,
            inference_run_name=run_name,
            inference_output_root=Path(inference_output_root),
            inference_predictor_root=DATASET_ROOT / test_split,
            test_predictor_paths=test_predictors,
            test_target_paths=TRAIN_TARGETS,
            use_static=USE_STATIC,
            checkpoint_path=None,
            preferred_checkpoint=PREFERRED_CHECKPOINT,
            device_target=DEVICE_TARGET,
            num_workers=NUM_WORKERS,
            batch_size=BATCH_SIZE,
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

        print(f"[{idx + 1:02d}/{NUM_RUNS}] Using predictors from {Path(test_predictor_paths[0]).parent}")
        print(f"[{idx + 1:02d}/{NUM_RUNS}] Output directory: {output_root}")
        print(f"[{idx + 1:02d}/{NUM_RUNS}] Saved parameter snapshot to {params_json}")

        test_dl = _build_dataloader(config, config.data.test_predictor_paths, config.data.test_target_paths, device)

        base_dataset = test_dl.dataset.base
        target_vars = list(base_dataset.target_vars)
        predictor_paths = list(base_dataset.predictor_paths)
        target_template_paths = list(base_dataset.target_paths)

        full_outputs = _run_full_inference(test_dl, model, device)
        outputs_np = full_outputs.numpy()
        if outputs_np.shape[1] != len(target_vars):
            raise ValueError(
                f"Model produced {outputs_np.shape[1]} channels but target_vars expects {len(target_vars)}"
            )

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
                name: dict(template_ds[name].attrs)
                for name in target_vars
                if name in template_ds.data_vars
            }
            lat_coord = template_ds[lat_name].load()
            lon_coord = template_ds[lon_name].load()

        coords = {time_dim: predictor_time_coord, lat_dim: lat_coord, lon_dim: lon_coord}
        prediction_ds = xr.Dataset(coords=coords)
        for var_idx, name in enumerate(target_vars):
            prediction_ds[name] = xr.DataArray(
                outputs_np[:, var_idx],
                dims=(time_dim, lat_dim, lon_dim),
                coords=coords,
                attrs=target_attrs.get(name, {}),
            )

        if CLAMP_PR_NONNEGATIVE:
            prediction_ds = enforce_pr_nonnegative_xr(prediction_ds)
        else:
            print("[clamp] Precipitation clamp disabled by CLAMP_PR_NONNEGATIVE=False")

        prediction_ds.attrs.update(template_attrs)
        prediction_output_path = output_root / prediction_output_stub
        prediction_ds.to_netcdf(prediction_output_path, engine="h5netcdf")
        print(
            f"[{idx + 1:02d}/{NUM_RUNS}] Saved predictions to {prediction_output_path} "
            f"({predictor_time_coord.values[0]} -> {predictor_time_coord.values[-1]})"
        )

        pickle_path = output_root / prediction_output_stub.with_suffix(".pkl")
        with open(pickle_path, "wb") as handle:
            import pickle

            pickle.dump(outputs_np, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[{idx + 1:02d}/{NUM_RUNS}] Saved raw predictions array to {pickle_path}")


if __name__ == "__main__":
    main()
