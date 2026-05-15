"""Inference for the MERRA2-to-PRISM downscaling model.

Loads a trained checkpoint, runs prediction over the YAML-defined inference
date range, denormalizes outputs, and writes NetCDF files.

Usage:
    python merra_prism_inference.py --config MERRA_PRISM.yaml [--checkpoint path/to/best.ckpt]
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    import xarray as xr
except ImportError:
    xr = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from granitewxc.utils.config import get_config
from granitewxc.utils.predictands import build_predictand_specs

from merra_prism_dataset import MerraPrismDataset
from merra_prism_utils import load_yaml, parse_date_range_from_config, resolve_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_checkpoint(cfg: Dict[str, Any], explicit: Optional[str]) -> str:
    """Locate the model checkpoint to use for inference."""
    if explicit:
        return str(resolve_path(explicit))
    inf_cfg = cfg.get("inference", {})
    ckpt = inf_cfg.get("checkpoint_path")
    if ckpt:
        return str(resolve_path(ckpt))
    # Fall back to run_dir / checkpoints / best.ckpt
    run_dir = cfg.get("run_dir") or cfg.get("checkpoint_dir")
    if run_dir:
        best = resolve_path(run_dir) / "best.ckpt"
        if best.exists():
            return str(best)
    # Final fallback
    exp = cfg.get("path_experiment", ".")
    for candidate in ("best.ckpt", "last.ckpt"):
        p = resolve_path(exp) / "checkpoints" / candidate
        if p.exists():
            return str(p)
    raise FileNotFoundError("Cannot locate a trained checkpoint. Pass --checkpoint explicitly.")


def _load_model(config: Any, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Re-create the model architecture and load trained weights."""
    from granitewxc.models.model import get_finetune_model_UNET

    if not hasattr(config.data, "input_static_surface_vars"):
        config.data.input_static_surface_vars = []

    model = get_finetune_model_UNET(config)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        state = ckpt.state_dict() if hasattr(ckpt, "state_dict") else dict(ckpt)

    # Strip DDP/FSDP module prefix if needed
    cleaned = {}
    for k, v in state.items():
        key = k
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = v

    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()
    return model


def _load_scalars(cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Load target mean and std for denormalization."""
    data_cfg = cfg.get("data", {})
    scalers = data_cfg.get("scalers", {})
    mean_path = scalers.get("targets_mean") or os.path.join(
        str(data_cfg.get("scalar_dir", "")), "targets_mean.npy"
    )
    std_path = scalers.get("targets_std") or os.path.join(
        str(data_cfg.get("scalar_dir", "")), "targets_std.npy"
    )
    mean_path = str(resolve_path(mean_path))
    std_path = str(resolve_path(std_path))

    if not os.path.exists(mean_path) or not os.path.exists(std_path):
        raise FileNotFoundError(
            f"Target scalars not found: {mean_path}, {std_path}. Run compute_scalars first."
        )
    return np.load(mean_path), np.load(std_path)


def _denormalize(
    prediction: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> np.ndarray:
    """Reverse the normalisation: x = x_norm * std + mean."""
    mean = target_mean.copy()
    std = target_std.copy()
    # Broadcast to match prediction shape
    while mean.ndim < prediction.ndim:
        mean = mean[..., np.newaxis]
        std = std[..., np.newaxis]
    std = np.maximum(std, 1e-6)
    return prediction * std + mean


def _validate_output(
    prediction: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    target_variables: Sequence[str],
    inference_dates: Sequence[date],
) -> None:
    """Run basic sanity checks on the inference output."""
    expected_time = len(inference_dates)
    if prediction.shape[0] != expected_time:
        raise ValueError(
            f"Inference output time dimension ({prediction.shape[0]}) does not match "
            f"requested dates ({expected_time})"
        )
    n_vars = len(target_variables)
    if prediction.shape[1] != n_vars:
        raise ValueError(
            f"Inference output has {prediction.shape[1]} variables, expected {n_vars}"
        )
    if np.any(~np.isfinite(prediction)):
        n_bad = int(np.sum(~np.isfinite(prediction)))
        print(f"[inference] WARNING: {n_bad} non-finite values detected in output")


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_inference(
    cfg: Dict[str, Any],
    config: Any,
    checkpoint_path: str,
    output_dir: str,
    device: torch.device,
    batch_size: int = 1,
) -> Path:
    """Execute inference over the YAML-defined date range and write NetCDF output."""
    if xr is None:
        raise ImportError("xarray is required for inference")

    data_cfg = cfg.get("data", {})
    target_variables: List[str] = list(data_cfg.get("target_variables", []))

    # Load model
    print(f"[inference] loading checkpoint: {checkpoint_path}")
    model = _load_model(config, checkpoint_path, device)

    # Load scalars for denormalization
    target_mean, target_std = _load_scalars(cfg)

    # Build inference dataset
    dataset = MerraPrismDataset(
        config_path=None,  # We'll override by passing cfg directly
        mode="inference",
    )
    # Fallback: construct dataset from the config dict directly
    # Re-initialize properly
    from merra_prism_dataset import MerraPrismDataset as _DS

    class _InferenceDS(_DS):
        """Thin subclass that accepts a pre-loaded config dict."""
        def __init__(self, cfg_dict, mode):
            self.cfg = cfg_dict
            import xarray  # ensure available
            self.mode = mode
            self.dtype = torch.float32
            dc = cfg_dict.get("data", {})
            self.predictor_dir = resolve_path(dc["predictor_dir"])
            self.target_dir = resolve_path(dc["target_dir"])
            self.predictor_vars = list(dc.get("predictor_variables", []))
            self.target_vars = list(dc.get("target_variables", []))
            from merra_prism_utils import (
                align_dates, discover_all_prism_targets, discover_merra2_files,
                parse_date_range_from_config, validate_dates_exist, validate_target_variables,
            )
            validate_target_variables(self.target_dir, self.target_vars)
            self.start_date, self.end_date = parse_date_range_from_config(cfg_dict, mode)
            merra_files = discover_merra2_files(self.predictor_dir, self.start_date, self.end_date)
            prism_files = discover_all_prism_targets(
                self.target_dir, self.target_vars, self.start_date, self.end_date
            )
            self._predictor_map = {d: p for d, p in merra_files}
            self._target_maps = {}
            for var, fl in prism_files.items():
                self._target_maps[var] = {d: p for d, p in fl}
            target_date_lists = {v: list(dm.keys()) for v, dm in self._target_maps.items()}
            self._dates = align_dates(list(self._predictor_map.keys()), target_date_lists)
            validate_dates_exist(self._dates, list(self._predictor_map.keys()), "predictor")
            scalar_dir = dc.get("scalar_dir", "")
            self._scalars = self._load_scalars(scalar_dir)

    inf_dataset = _InferenceDS(cfg, "inference")
    print(f"[inference] {len(inf_dataset)} dates in inference period")

    loader = torch.utils.data.DataLoader(
        inf_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    all_predictions: List[np.ndarray] = []
    all_dates: List[date] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x = batch["x"].to(device)
            pred = model(x)
            if isinstance(pred, dict):
                pred = pred.get("y_hat", pred.get("output", next(iter(pred.values()))))
            pred_np = pred.cpu().numpy()
            all_predictions.append(pred_np)

            # Collect dates
            batch_dates = batch.get("date", [])
            if isinstance(batch_dates, (list, tuple)):
                all_dates.extend(batch_dates)
            elif hasattr(batch_dates, "tolist"):
                all_dates.extend(batch_dates.tolist())

            if (batch_idx + 1) % 50 == 0:
                print(f"[inference] processed batch {batch_idx + 1}")

    predictions = np.concatenate(all_predictions, axis=0)  # (T, C, H, W)

    # Denormalize
    predictions = _denormalize(predictions, target_mean, target_std)

    # Validate
    inference_dates = inf_dataset.dates
    # Get target grid from first PRISM file for metadata
    first_var = target_variables[0]
    first_date = inference_dates[0]
    first_path = inf_dataset._target_maps[first_var][first_date]
    with xr.open_dataset(str(first_path)) as ds_ref:
        lat_candidates = ("lat", "latitude", "y")
        lon_candidates = ("lon", "longitude", "x")
        lat_name = next((n for n in lat_candidates if n in ds_ref.coords), None)
        lon_name = next((n for n in lon_candidates if n in ds_ref.coords), None)
        if lat_name is None or lon_name is None:
            raise ValueError("Cannot find lat/lon in PRISM reference file")
        target_lat = ds_ref[lat_name].values
        target_lon = ds_ref[lon_name].values

    _validate_output(predictions, target_lat, target_lon, target_variables, inference_dates)

    # Write output NetCDF
    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / "merra_prism_inference.nc"

    time_values = [np.datetime64(str(d)) for d in inference_dates]
    data_vars = {}
    for ch_idx, var in enumerate(target_variables):
        data_vars[var] = (
            ("time", "lat", "lon"),
            predictions[:, ch_idx, :, :].astype(np.float32),
            {"long_name": var, "units": ""},
        )

    out_ds = xr.Dataset(
        data_vars,
        coords={
            "time": time_values,
            "lat": target_lat,
            "lon": target_lon,
        },
        attrs={
            "description": "MERRA2-to-PRISM downscaling inference output",
            "checkpoint": checkpoint_path,
            "inference_start": str(inference_dates[0]),
            "inference_end": str(inference_dates[-1]),
        },
    )
    out_ds.to_netcdf(str(out_path))
    print(f"[inference] output saved → {out_path}")

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MERRA-PRISM inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to MERRA_PRISM.yaml")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    config = get_config(str(Path(args.config).resolve()))

    checkpoint = _find_checkpoint(cfg, args.checkpoint)
    output_dir = args.output_dir or str(
        resolve_path(cfg.get("inference", {}).get("output_dir", "./examples/MERRA_PRISM/experiments/inference_output"))
    )

    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[inference] CUDA not available, falling back to CPU")
        device_str = "cpu"
    device = torch.device(device_str)

    run_inference(
        cfg=cfg,
        config=config,
        checkpoint_path=checkpoint,
        output_dir=output_dir,
        device=device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
