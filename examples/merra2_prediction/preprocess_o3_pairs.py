from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from o3_pipeline_utils import (
    build_inputs_targets_3h,
    load_chem_surface,
    load_met_surface,
    load_yaml_config,
    predictor_channel_order,
    resolve_runtime_paths,
    write_netcdf_robust,
)

DEFAULT_CONFIG_PATH = str((Path(__file__).resolve().parent / "o3_pipeline_config.yaml"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocess MERRA2 O3 next-step training pairs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML configuration file",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing preprocessed files",
    )
    return p.parse_args()


def split_indices(n: int, val_fraction: float, split_mode: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n < 2:
        raise ValueError(f"Need at least 2 samples to split train/val, got {n}")

    val_n = int(round(n * val_fraction))
    val_n = max(1, min(n - 1, val_n))

    if split_mode == "random":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        val_idx = np.sort(perm[-val_n:])
        train_idx = np.sort(perm[:-val_n])
    else:
        # chronological split
        cut = n - val_n
        train_idx = np.arange(0, cut)
        val_idx = np.arange(cut, n)

    return train_idx, val_idx


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    paths = resolve_runtime_paths(cfg)

    data_cfg = cfg.get("data", {})
    preprocess_cfg = cfg.get("preprocess", {})

    o3_name = str(data_cfg.get("o3_output_name", "O3_sfc"))
    chem_var = str(data_cfg.get("chem_var", "O3"))
    met_vars = list(data_cfg.get("met_vars", ["T", "U", "V", "PS"]))
    met_suffix = str(data_cfg.get("met_suffix", "_sfc"))

    start_time = str(data_cfg.get("start_time", "2020-01-01T00:00:00"))
    end_time = str(data_cfg.get("end_time", "2020-01-02T21:00:00"))
    delta_hours = int(data_cfg.get("delta_hours", 3))

    chem_pattern = str(data_cfg.get("chem_pattern", "MERRA2_*inst3_3d_chm_N[vV]*.nc4"))
    met_pattern = str(data_cfg.get("met_pattern", "MERRA2_*inst3_3d_asm_N[vV]*.nc4"))

    lev_dim_candidates = tuple(
        data_cfg.get("lev_dim_candidates", ["lev", "level", "model_level", "eta", "pfull"])
    )
    xr_chunks = data_cfg.get("xr_chunks", {"time": 8, "lat": 181, "lon": 288})

    train_out = paths["train_pairs_file"]
    val_out = paths["val_pairs_file"]

    if (train_out.exists() or val_out.exists()) and not args.overwrite:
        raise FileExistsError(
            f"Output exists ({train_out}, {val_out}). Pass --overwrite to regenerate."
        )

    ds_chem_3h = load_chem_surface(
        chem_root=paths["chem_root"],
        chem_pattern=chem_pattern,
        chem_var=chem_var,
        start_time=start_time,
        end_time=end_time,
        delta_hours=delta_hours,
        chunks=xr_chunks,
        lev_dim_candidates=lev_dim_candidates,
        output_name=o3_name,
    )
    ds_met_3h = load_met_surface(
        met_root=paths["met_root"],
        met_pattern=met_pattern,
        met_vars=met_vars,
        met_suffix=met_suffix,
        start_time=start_time,
        end_time=end_time,
        delta_hours=delta_hours,
        chunks=xr_chunks,
        lev_dim_candidates=lev_dim_candidates,
    )

    ds_chem_3h, ds_met_3h = xr.align(ds_chem_3h, ds_met_3h, join="inner")

    ds_inputs_3h, da_target_3h, time_out = build_inputs_targets_3h(
        ds_met_3h=ds_met_3h,
        ds_chem_3h=ds_chem_3h,
        o3_name=o3_name,
        delta_hours=delta_hours,
    )

    target_name = str(data_cfg.get("target_name", f"{o3_name}_target"))
    ds_pairs = ds_inputs_3h.assign({target_name: da_target_3h})
    ds_pairs = ds_pairs.assign_coords(time_out=("time", time_out.values.astype("datetime64[ns]")))

    predictor_vars = predictor_channel_order(o3_name=o3_name, met_vars=met_vars, met_suffix=met_suffix)
    missing = [v for v in predictor_vars if v not in ds_pairs.data_vars]
    if missing:
        raise KeyError(f"Missing predictors in preprocessed dataset: {missing}")

    val_fraction = float(preprocess_cfg.get("val_fraction", 0.2))
    split_mode = str(preprocess_cfg.get("split_mode", "chronological")).lower()
    seed = int(preprocess_cfg.get("seed", 42))
    if split_mode not in {"chronological", "random"}:
        raise ValueError("preprocess.split_mode must be one of {'chronological', 'random'}")

    n = int(ds_pairs.sizes["time"])
    train_idx, val_idx = split_indices(n=n, val_fraction=val_fraction, split_mode=split_mode, seed=seed)

    ds_train = ds_pairs.isel(time=train_idx)
    ds_val = ds_pairs.isel(time=val_idx)

    ds_train.attrs.update(
        {
            "split": "train",
            "predictor_vars": ",".join(predictor_vars),
            "target_var": target_name,
            "delta_hours": delta_hours,
            "split_mode": split_mode,
        }
    )
    ds_val.attrs.update(
        {
            "split": "val",
            "predictor_vars": ",".join(predictor_vars),
            "target_var": target_name,
            "delta_hours": delta_hours,
            "split_mode": split_mode,
        }
    )

    engine_train = write_netcdf_robust(ds_train.load(), train_out)
    engine_val = write_netcdf_robust(ds_val.load(), val_out)

    print("Preprocessing complete")
    print(f"Base dir          : {paths['base_dir']}")
    print(f"Chem root         : {paths['chem_root']}")
    print(f"Met root          : {paths['met_root']}")
    print(f"Predictor vars    : {predictor_vars}")
    print(f"Target var        : {target_name}")
    print(f"Total pairs       : {n}")
    print(f"Train pairs       : {len(train_idx)}")
    print(f"Val pairs         : {len(val_idx)}")
    print(f"Wrote train pairs : {train_out} (engine={engine_train})")
    print(f"Wrote val pairs   : {val_out} (engine={engine_val})")


if __name__ == "__main__":
    main()
