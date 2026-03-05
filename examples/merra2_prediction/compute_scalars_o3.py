from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from o3_pipeline_utils import (
    load_yaml_config,
    resolve_predictor_vars,
    resolve_target_var_name,
    resolve_runtime_paths,
)

DEFAULT_CONFIG_PATH = str((Path(__file__).resolve().parent / "o3_pipeline_config.yaml"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute channel-wise scalars for MERRA2 chemistry next-step fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML configuration file",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    paths = resolve_runtime_paths(cfg)

    data_cfg = cfg.get("data", {})
    scalers_cfg = cfg.get("scalers", {})

    train_pairs_file = paths["train_pairs_file"]
    if not train_pairs_file.exists():
        raise FileNotFoundError(
            f"Training pairs file not found: {train_pairs_file}. Run preprocess_o3_pairs.py first."
        )

    engine = "h5netcdf" if train_pairs_file.suffix.lower() in {".nc", ".nc4"} else None
    ds = xr.open_dataset(train_pairs_file, engine=engine)

    predictor_vars = resolve_predictor_vars(data_cfg, dataset_attrs=ds.attrs)
    target_var = str(ds.attrs.get("target_var", resolve_target_var_name(data_cfg)))

    missing = [v for v in predictor_vars + [target_var] if v not in ds.data_vars]
    if missing:
        raise KeyError(f"Missing variables in training pairs file: {missing}")

    x_da = ds[predictor_vars].to_array("channel")  # [channel, time, lat, lon]
    y_da = ds[target_var]  # [time, lat, lon]

    x_mean = x_da.mean(dim=("time", "lat", "lon"), skipna=True).values.astype(np.float32)
    x_std = x_da.std(dim=("time", "lat", "lon"), skipna=True).values.astype(np.float32)

    y_mean = np.array([float(y_da.mean(dim=("time", "lat", "lon"), skipna=True).values)], dtype=np.float32)
    y_std = np.array([float(y_da.std(dim=("time", "lat", "lon"), skipna=True).values)], dtype=np.float32)

    eps = float(scalers_cfg.get("eps", 1e-6))
    x_std = np.clip(x_std, eps, None)
    y_std = np.clip(y_std, eps, None)

    scalers_dir = paths["scalers_dir"]
    scalers_dir.mkdir(parents=True, exist_ok=True)

    input_mu_file = scalers_dir / "inputs_mean.npy"
    input_sigma_file = scalers_dir / "inputs_std.npy"
    target_mu_file = scalers_dir / "targets_mean.npy"
    target_sigma_file = scalers_dir / "targets_std.npy"
    metadata_file = scalers_dir / "metadata.json"

    np.save(input_mu_file, x_mean)
    np.save(input_sigma_file, x_std)
    np.save(target_mu_file, y_mean)
    np.save(target_sigma_file, y_std)

    metadata = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "train_pairs_file": str(train_pairs_file),
        "predictor_vars": predictor_vars,
        "target_var": target_var,
        "eps": eps,
        "inputs_mean_shape": list(x_mean.shape),
        "inputs_std_shape": list(x_std.shape),
        "targets_mean_shape": list(y_mean.shape),
        "targets_std_shape": list(y_std.shape),
        "input_mu_file": str(input_mu_file),
        "input_sigma_file": str(input_sigma_file),
        "target_mu_file": str(target_mu_file),
        "target_sigma_file": str(target_sigma_file),
    }
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Scaler computation complete")
    print(f"Train pairs file : {train_pairs_file}")
    print(f"Predictor vars   : {predictor_vars}")
    print(f"Target var       : {target_var}")
    print(f"Saved input mu   : {input_mu_file}")
    print(f"Saved input std  : {input_sigma_file}")
    print(f"Saved target mu  : {target_mu_file}")
    print(f"Saved target std : {target_sigma_file}")
    print(f"Metadata         : {metadata_file}")


if __name__ == "__main__":
    main()
