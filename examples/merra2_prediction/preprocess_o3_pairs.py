from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from o3_pipeline_utils import (
    build_inputs_targets_3h,
    load_chem_fields,
    load_met_surface,
    load_yaml_config,
    resolve_predictor_vars,
    resolve_target_input_name,
    resolve_target_var_name,
    resolve_runtime_paths,
    write_netcdf_robust,
)

DEFAULT_CONFIG_PATH = str((Path(__file__).resolve().parent / "o3_pipeline_config.yaml"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocess MERRA2 chemistry next-step training pairs",
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


def normalize_chem_predictor_specs(data_cfg: dict) -> list[dict[str, str]]:
    raw_specs = data_cfg.get("chem_predictors")
    if raw_specs is None:
        raw_specs = data_cfg.get("chem_predictor_vars", [])
    if raw_specs is None:
        raw_specs = []
    if not isinstance(raw_specs, (list, tuple)):
        raise TypeError("data.chem_predictors (or legacy data.chem_predictor_vars) must be a list")

    default_transform = str(data_cfg.get("chem_predictor_transform", "surface")).strip().lower()
    default_suffix = str(data_cfg.get("chem_predictor_suffix", "_sfc"))

    specs: list[dict[str, str]] = []
    for item in raw_specs:
        if isinstance(item, str):
            var = item
            transform = default_transform
            if transform in {"surface", "sfc"} and default_suffix:
                output_name = f"{var}{default_suffix}"
            else:
                output_name = var
        elif isinstance(item, dict):
            if "var" not in item:
                raise KeyError(f"chem predictor spec missing 'var': {item}")
            var = str(item["var"])
            transform = str(item.get("transform", default_transform)).strip().lower()
            output_name = item.get("output_name")
            if output_name is None:
                suffix = str(item.get("suffix", default_suffix if transform in {"surface", "sfc"} else ""))
                output_name = f"{var}{suffix}" if suffix else var
            output_name = str(output_name)
        else:
            raise TypeError(f"Unsupported chem predictor spec type: {type(item)}")

        specs.append(
            {
                "var": str(var),
                "output_name": str(output_name),
                "transform": str(transform),
            }
        )
    return specs


def assert_time_lat_lon_only(name: str, da: xr.DataArray) -> None:
    required = {"time", "lat", "lon"}
    missing = [d for d in required if d not in da.dims]
    if missing:
        raise ValueError(f"{name} is missing required dims {missing}. Found dims={da.dims}")
    extra = [d for d in da.dims if d not in required]
    if extra:
        raise ValueError(
            f"{name} has unsupported extra dims {extra}. "
            "Use transform='surface' for 3D fields or provide 2D column variables."
        )


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    paths = resolve_runtime_paths(cfg)

    data_cfg = cfg.get("data", {})
    preprocess_cfg = cfg.get("preprocess", {})

    target_var = str(data_cfg.get("target_var", data_cfg.get("chem_var", "O3")))
    target_input_name = resolve_target_input_name(data_cfg)
    target_name = resolve_target_var_name(data_cfg)
    target_transform = str(data_cfg.get("target_transform", "surface"))
    include_target_as_predictor = bool(data_cfg.get("include_target_as_predictor", True))

    chem_predictor_specs = normalize_chem_predictor_specs(data_cfg)
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

    chem_field_specs = [
        {
            "var": target_var,
            "output_name": target_input_name,
            "transform": target_transform,
        }
    ]
    chem_field_specs.extend(chem_predictor_specs)

    # De-duplicate by output name and reject conflicting remaps.
    dedup_specs: dict[str, dict[str, str]] = {}
    for spec in chem_field_specs:
        out_name = spec["output_name"]
        if out_name not in dedup_specs:
            dedup_specs[out_name] = spec
            continue
        prev = dedup_specs[out_name]
        if (prev["var"] != spec["var"]) or (prev["transform"] != spec["transform"]):
            raise ValueError(
                f"Conflicting chemistry field specs for output '{out_name}': {prev} vs {spec}"
            )

    ds_chem_3h = load_chem_fields(
        chem_root=paths["chem_root"],
        chem_pattern=chem_pattern,
        field_specs=list(dedup_specs.values()),
        start_time=start_time,
        end_time=end_time,
        delta_hours=delta_hours,
        chunks=xr_chunks,
        lev_dim_candidates=lev_dim_candidates,
    )

    if met_vars:
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
    else:
        ds_met_3h = xr.Dataset(coords={"time": ds_chem_3h.time})

    ds_chem_3h, ds_met_3h = xr.align(ds_chem_3h, ds_met_3h, join="inner")

    explicit_predictor_vars = data_cfg.get("predictor_vars") is not None
    predictor_vars = resolve_predictor_vars(data_cfg)
    if (not explicit_predictor_vars) and (not include_target_as_predictor) and target_input_name in predictor_vars:
        predictor_vars = [v for v in predictor_vars if v != target_input_name]

    if not predictor_vars:
        raise ValueError(
            "No predictors resolved. Set data.predictor_vars or enable target/met/chem predictors."
        )

    ds_predictor_pool = xr.merge([ds_chem_3h, ds_met_3h], compat="override")
    missing = [v for v in predictor_vars if v not in ds_predictor_pool.data_vars]
    if missing:
        raise KeyError(f"Missing predictors in loaded data: {missing}")

    if target_input_name not in ds_chem_3h.data_vars:
        raise KeyError(
            f"Target input '{target_input_name}' not found in chemistry fields. "
            "Check data.target_var/target_input_name/target_transform."
        )

    for v in predictor_vars:
        assert_time_lat_lon_only(f"predictor '{v}'", ds_predictor_pool[v])
    assert_time_lat_lon_only(f"target input '{target_input_name}'", ds_chem_3h[target_input_name])

    ds_inputs_3h, da_target_3h, time_out = build_inputs_targets_3h(
        ds_predictors_3h=ds_predictor_pool[predictor_vars],
        target_3h=ds_chem_3h[target_input_name],
        delta_hours=delta_hours,
    )

    ds_pairs = ds_inputs_3h.assign({target_name: da_target_3h})
    ds_pairs = ds_pairs.assign_coords(time_out=("time", time_out.values.astype("datetime64[ns]")))

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
            "target_input_name": target_input_name,
            "target_source_var": target_var,
            "delta_hours": delta_hours,
            "split_mode": split_mode,
        }
    )
    ds_val.attrs.update(
        {
            "split": "val",
            "predictor_vars": ",".join(predictor_vars),
            "target_var": target_name,
            "target_input_name": target_input_name,
            "target_source_var": target_var,
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
