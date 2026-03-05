from __future__ import annotations

import glob
import importlib.util
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
import yaml
from huggingface_hub import hf_hub_download


def load_yaml_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise TypeError(f"Config must parse to dict, got {type(cfg)}")
    cfg["_config_path"] = str(cfg_path)
    return cfg


def _resolve_path(base: Path, value: str | os.PathLike[str] | None, default: Path) -> Path:
    if value is None:
        return default
    p = Path(value).expanduser()
    if p.is_absolute():
        return p
    return (base / p).resolve()


def detect_base_dir(cfg: dict[str, Any]) -> Path:
    config_path = Path(cfg.get("_config_path", "")).resolve()
    config_parent = config_path.parent if config_path.exists() else Path.cwd().resolve()
    env_base = os.environ.get("MERRA2_PREDICTION_BASE")

    path_cfg = cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}
    base_cfg = path_cfg.get("base_dir")

    candidates: list[Path] = []
    if env_base:
        candidates.append(Path(env_base))
    if base_cfg:
        candidates.append(Path(base_cfg))

    cwd = Path.cwd().resolve()
    candidates.extend(
        [
            config_parent,
            cwd,
            cwd / "merra2_prediction",
            cwd / "examples" / "merra2_prediction",
            config_parent / "examples" / "merra2_prediction",
        ]
    )

    for p in [cwd] + list(cwd.parents):
        candidates.append(p / "examples" / "merra2_prediction")

    seen = set()
    checked: list[str] = []
    for cand in candidates:
        c = cand.expanduser().resolve()
        if c in seen:
            continue
        seen.add(c)
        checked.append(str(c))
        if (c / "merra2").exists():
            return c

    raise FileNotFoundError(
        "Could not detect merra2_prediction base directory (expects merra2/). "
        f"Checked: {checked}"
    )


def resolve_runtime_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    base_dir = detect_base_dir(cfg)
    path_cfg = cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}

    paths = {
        "base_dir": base_dir,
        "chem_root": _resolve_path(base_dir, path_cfg.get("chem_root"), base_dir / "merra2"),
        "met_root": _resolve_path(base_dir, path_cfg.get("met_root"), base_dir / "merra2"),
        "preprocessed_dir": _resolve_path(base_dir, path_cfg.get("preprocessed_dir"), base_dir / "preprocessed"),
        "scalers_dir": _resolve_path(base_dir, path_cfg.get("scalers_dir"), base_dir / "experiments" / "o3_scalars"),
        "checkpoints_dir": _resolve_path(base_dir, path_cfg.get("checkpoints_dir"), base_dir / "checkpoints"),
        "weights_dir": _resolve_path(base_dir, path_cfg.get("pretrained_weights_dir"), base_dir / "weights"),
    }

    train_pairs_default = paths["preprocessed_dir"] / "o3_pairs_train.nc"
    val_pairs_default = paths["preprocessed_dir"] / "o3_pairs_val.nc"
    paths["train_pairs_file"] = _resolve_path(base_dir, path_cfg.get("train_pairs_file"), train_pairs_default)
    paths["val_pairs_file"] = _resolve_path(base_dir, path_cfg.get("val_pairs_file"), val_pairs_default)

    for key in ("preprocessed_dir", "scalers_dir", "checkpoints_dir", "weights_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)

    return paths


def _pick_engine() -> str:
    return "h5netcdf" if importlib.util.find_spec("h5netcdf") is not None else "netcdf4"


def resolve_paths(root: str | os.PathLike[str], pattern: str) -> list[str]:
    root = Path(root)
    paths = sorted(glob.glob(str(root / pattern)))
    if not paths:
        raise FileNotFoundError(f"No files match {root / pattern}")
    return paths


def _to_utc_naive(ts: str) -> pd.Timestamp:
    t = pd.to_datetime(ts, utc=True)
    return t.tz_convert(None)


def open_mfdataset_utc(
    paths: Sequence[str],
    *,
    start_time: str,
    end_time: str,
    delta_hours: int,
    chunks: dict[str, int] | None = None,
    vars_to_keep: Sequence[str] | None = None,
) -> xr.Dataset:
    ds = xr.open_mfdataset(
        list(paths),
        combine="by_coords",
        decode_times=True,
        parallel=False,
        engine=_pick_engine(),
        chunks=chunks,
    )

    if vars_to_keep is not None:
        missing = [v for v in vars_to_keep if v not in ds.data_vars]
        if missing:
            raise KeyError(f"Missing variables {missing}. Available sample: {list(ds.data_vars)[:20]}")
        ds = ds[list(vars_to_keep)]

    ds = ds.sortby("time")
    t = pd.to_datetime(ds.time.values, utc=True).tz_convert(None)
    ds = ds.assign_coords(time=t)

    start_in = _to_utc_naive(start_time)
    end_in = _to_utc_naive(end_time)
    if end_in < start_in:
        raise ValueError(f"end_time ({end_time}) must be >= start_time ({start_time})")

    end_target = end_in + pd.Timedelta(hours=int(delta_hours))
    ds = ds.sel(time=slice(start_in, end_target))
    return ds


def _find_vertical_dim(da_3d: xr.DataArray, candidates: Sequence[str]) -> str:
    for dim in candidates:
        if dim in da_3d.dims:
            return dim
    raise ValueError(f"No vertical dim found in {da_3d.dims}")


def extract_surface_3d(
    da_3d: xr.DataArray,
    lev_dim_candidates: Sequence[str] = ("lev", "level", "model_level", "eta", "pfull"),
) -> xr.DataArray:
    lev_dim = _find_vertical_dim(da_3d, lev_dim_candidates)
    coord = da_3d[lev_dim]

    idx = coord.size - 1
    if np.issubdtype(coord.dtype, np.number) and coord.size > 0:
        vals = coord.values
        positive = str(coord.attrs.get("positive", "")).lower()
        if np.all(np.isnan(vals)):
            idx = coord.size - 1
        elif positive == "up":
            idx = int(np.nanargmin(vals))
        elif positive == "down":
            idx = int(np.nanargmax(vals))
        else:
            idx = int(np.nanargmax(vals))

    da_sfc = da_3d.isel({lev_dim: idx}).copy()
    da_sfc.attrs = dict(da_3d.attrs)
    if da_3d.name:
        da_sfc.attrs["long_name"] = f"{da_3d.name} surface (bottom-most model level)"
    return da_sfc


def apply_vertical_transform(
    da: xr.DataArray,
    *,
    transform: str,
    lev_dim_candidates: Sequence[str],
) -> xr.DataArray:
    mode = str(transform).strip().lower()
    if mode in {"surface", "sfc"}:
        if any(dim in da.dims for dim in lev_dim_candidates):
            return extract_surface_3d(da, lev_dim_candidates=lev_dim_candidates)
        return da
    if mode in {"none", "identity", "keep"}:
        return da
    raise ValueError(f"Unsupported transform='{transform}'. Use one of: surface, none")


def load_chem_fields(
    *,
    chem_root: Path,
    chem_pattern: str,
    field_specs: Sequence[Mapping[str, Any]],
    start_time: str,
    end_time: str,
    delta_hours: int,
    chunks: dict[str, int] | None,
    lev_dim_candidates: Sequence[str],
) -> xr.Dataset:
    if not field_specs:
        raise ValueError("field_specs must contain at least one chemistry field")

    vars_to_keep: list[str] = []
    for spec in field_specs:
        raw = spec.get("var")
        if raw is None:
            raise KeyError(f"Chem field spec missing 'var': {spec}")
        v = str(raw)
        if v not in vars_to_keep:
            vars_to_keep.append(v)

    paths = resolve_paths(chem_root, chem_pattern)
    ds = open_mfdataset_utc(
        paths,
        start_time=start_time,
        end_time=end_time,
        delta_hours=delta_hours,
        chunks=chunks,
        vars_to_keep=vars_to_keep,
    )

    out = xr.Dataset()
    for spec in field_specs:
        raw_var = str(spec.get("var"))
        output_name = str(spec.get("output_name") or raw_var)
        transform = str(spec.get("transform", "surface"))

        da = ds[raw_var]
        da = apply_vertical_transform(da, transform=transform, lev_dim_candidates=lev_dim_candidates).rename(output_name)
        out[output_name] = da

    return out


def load_chem_surface(
    *,
    chem_root: Path,
    chem_pattern: str,
    chem_var: str,
    start_time: str,
    end_time: str,
    delta_hours: int,
    chunks: dict[str, int] | None,
    lev_dim_candidates: Sequence[str],
    output_name: str,
) -> xr.Dataset:
    # Backward-compatible wrapper for legacy single-variable surface extraction.
    return load_chem_fields(
        chem_root=chem_root,
        chem_pattern=chem_pattern,
        field_specs=[
            {
                "var": chem_var,
                "output_name": output_name,
                "transform": "surface",
            }
        ],
        start_time=start_time,
        end_time=end_time,
        delta_hours=delta_hours,
        chunks=chunks,
        lev_dim_candidates=lev_dim_candidates,
    )


def load_met_surface(
    *,
    met_root: Path,
    met_pattern: str,
    met_vars: Sequence[str],
    met_suffix: str,
    start_time: str,
    end_time: str,
    delta_hours: int,
    chunks: dict[str, int] | None,
    lev_dim_candidates: Sequence[str],
) -> xr.Dataset:
    paths = resolve_paths(met_root, met_pattern)
    ds = open_mfdataset_utc(
        paths,
        start_time=start_time,
        end_time=end_time,
        delta_hours=delta_hours,
        chunks=chunks,
        vars_to_keep=list(met_vars),
    )

    ds_surface = xr.Dataset()
    for v in met_vars:
        da = ds[v]
        if any(dim in da.dims for dim in lev_dim_candidates):
            ds_surface[f"{v}{met_suffix}"] = extract_surface_3d(da, lev_dim_candidates=lev_dim_candidates)
        else:
            ds_surface[f"{v}{met_suffix}"] = da
    return ds_surface


def build_inputs_targets_3h(
    *,
    ds_predictors_3h: xr.Dataset,
    target_3h: xr.DataArray,
    delta_hours: int,
) -> tuple[xr.Dataset, xr.DataArray, xr.DataArray]:
    if delta_hours < 1:
        raise ValueError("delta_hours must be >= 1")

    t_in = pd.DatetimeIndex(ds_predictors_3h.time.values)
    t_out = t_in + pd.Timedelta(hours=delta_hours)
    t_target = pd.DatetimeIndex(target_3h.time.values)

    valid = t_out.isin(t_target)
    if not valid.any():
        raise ValueError(
            "No valid input/target pairs found. "
            "Check start/end coverage and source file range."
        )

    t_in_valid = t_in[valid]
    t_out_valid = t_out[valid]

    ds_predictors_3h = ds_predictors_3h.sel(time=t_in_valid)
    target_shifted = target_3h.sel(time=t_out_valid).assign_coords(time=t_in_valid)

    ds_predictors_3h, target_shifted = xr.align(ds_predictors_3h, target_shifted, join="inner")
    time_out = xr.DataArray(
        t_out_valid.values.astype("datetime64[ns]"),
        dims=["time"],
        coords={"time": t_in_valid.values},
        name="time_out",
    )
    return ds_predictors_3h, target_shifted, time_out


def build_inputs_targets_legacy_chem_met(
    *,
    ds_met_3h: xr.Dataset,
    ds_chem_3h: xr.Dataset,
    o3_name: str,
    delta_hours: int,
) -> tuple[xr.Dataset, xr.DataArray, xr.DataArray]:
    # Backward-compatible wrapper for legacy O3+met pipeline assumptions.
    ds_predictors_3h = xr.merge([ds_chem_3h[[o3_name]], ds_met_3h])
    return build_inputs_targets_3h(
        ds_predictors_3h=ds_predictors_3h,
        target_3h=ds_chem_3h[o3_name],
        delta_hours=delta_hours,
    )


def write_netcdf_robust(ds: xr.Dataset, out_path: Path) -> str:
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    engines: list[str] = []
    preferred = _pick_engine()
    if preferred:
        engines.append(preferred)
    for eng in ("netcdf4", "h5netcdf", "scipy"):
        if eng not in engines:
            engines.append(eng)

    last_err: Exception | None = None
    for eng in engines:
        try:
            if eng == "scipy":
                ds.to_netcdf(tmp_path, engine=eng, mode="w")
            else:
                ds.to_netcdf(tmp_path, engine=eng, mode="w", format="NETCDF4")
            tmp_path.replace(out_path)
            return eng
        except Exception as e:  # pragma: no cover - environment dependent
            last_err = e
            if tmp_path.exists():
                tmp_path.unlink()

    raise RuntimeError(f"Failed to write NetCDF to {out_path} with engines {engines}") from last_err


def predictor_channel_order(o3_name: str, met_vars: Sequence[str], met_suffix: str) -> list[str]:
    return [o3_name] + [f"{v}{met_suffix}" for v in met_vars]


def resolve_target_input_name(data_cfg: Mapping[str, Any]) -> str:
    if data_cfg.get("target_input_name") is not None:
        return str(data_cfg["target_input_name"])
    if data_cfg.get("target_output_name") is not None:
        return str(data_cfg["target_output_name"])
    if data_cfg.get("o3_output_name") is not None:
        return str(data_cfg["o3_output_name"])
    return "O3_sfc"


def resolve_target_var_name(data_cfg: Mapping[str, Any]) -> str:
    target_input_name = resolve_target_input_name(data_cfg)
    return str(data_cfg.get("target_name", f"{target_input_name}_target"))


def infer_chem_predictor_output_names(data_cfg: Mapping[str, Any]) -> list[str]:
    raw_specs = data_cfg.get("chem_predictors")
    if raw_specs is None:
        raw_specs = data_cfg.get("chem_predictor_vars", [])
    if raw_specs is None:
        raw_specs = []
    if not isinstance(raw_specs, (list, tuple)):
        raise TypeError("data.chem_predictors (or legacy data.chem_predictor_vars) must be a list")

    default_transform = str(data_cfg.get("chem_predictor_transform", "surface")).strip().lower()
    default_suffix = str(data_cfg.get("chem_predictor_suffix", "_sfc"))

    out: list[str] = []
    for item in raw_specs:
        if isinstance(item, str):
            var = item
            transform = default_transform
            if transform in {"surface", "sfc"} and default_suffix:
                name = f"{var}{default_suffix}"
            else:
                name = var
        elif isinstance(item, Mapping):
            if "var" not in item:
                raise KeyError(f"chem predictor spec missing 'var': {item}")
            var = str(item["var"])
            transform = str(item.get("transform", default_transform)).strip().lower()
            name = item.get("output_name")
            if name is None:
                suffix = str(item.get("suffix", default_suffix if transform in {"surface", "sfc"} else ""))
                name = f"{var}{suffix}" if suffix else var
            name = str(name)
        else:
            raise TypeError(f"Unsupported chem predictor spec type: {type(item)}")
        out.append(name)

    return out


def resolve_predictor_vars(
    data_cfg: Mapping[str, Any],
    *,
    dataset_attrs: Mapping[str, Any] | None = None,
) -> list[str]:
    explicit = data_cfg.get("predictor_vars")
    if explicit is not None:
        if not isinstance(explicit, (list, tuple)):
            raise TypeError("data.predictor_vars must be a list when provided")
        return [str(v) for v in explicit]

    if dataset_attrs is not None:
        attrs_val = dataset_attrs.get("predictor_vars")
        if attrs_val:
            return [v for v in str(attrs_val).split(",") if v]

    target_input_name = resolve_target_input_name(data_cfg)
    include_target = bool(data_cfg.get("include_target_as_predictor", True))

    chem_predictor_names_cfg = data_cfg.get("chem_predictor_output_names")
    if chem_predictor_names_cfg is not None:
        if not isinstance(chem_predictor_names_cfg, (list, tuple)):
            raise TypeError("data.chem_predictor_output_names must be a list when provided")
        chem_predictor_names = [str(v) for v in chem_predictor_names_cfg]
    else:
        chem_predictor_names = infer_chem_predictor_output_names(data_cfg)

    met_vars = list(data_cfg.get("met_vars", ["T", "U", "V", "PS"]))
    met_suffix = str(data_cfg.get("met_suffix", "_sfc"))
    met_names = [f"{v}{met_suffix}" for v in met_vars]

    out: list[str] = []
    if include_target:
        out.append(target_input_name)
    out.extend(chem_predictor_names)
    out.extend(met_names)

    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for v in out:
        if v not in seen:
            unique.append(v)
            seen.add(v)
    return unique


class O3Delta1Model(nn.Module):
    """Prithvi-WxC encoder-decoder wrapper for single-channel next-step forecasting."""

    def __init__(
        self,
        in_channels: int,
        *,
        embed_dim: int = 2560,
        n_blocks: int = 4,
        n_heads: int = 16,
        mlp_multiplier: int = 4,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        mask_unit_size: tuple[int, int] = (16, 16),
    ):
        super().__init__()

        try:
            from PrithviWxC.model import PrithviWxCEncoderDecoder
        except Exception as e:  # pragma: no cover - dependency guard
            raise ImportError("PrithviWxC package is required") from e

        if len(mask_unit_size) != 2:
            raise ValueError(f"mask_unit_size must be length 2, got {mask_unit_size}")
        self.mask_unit_size = (int(mask_unit_size[0]), int(mask_unit_size[1]))

        self.backbone = PrithviWxCEncoderDecoder(
            embed_dim=embed_dim,
            n_blocks=n_blocks,
            mlp_multiplier=mlp_multiplier,
            n_heads=n_heads,
            dropout=dropout,
            drop_path=drop_path,
        )
        self.in_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.proj = nn.Conv2d(embed_dim, 1, kernel_size=1)

    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, multiple_hw: tuple[int, int]) -> tuple[torch.Tensor, int, int]:
        _, _, h, w = x.shape
        mh, mw = multiple_hw
        pad_h = (mh - (h % mh)) % mh
        pad_w = (mw - (w % mw)) % mw
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x, pad_h, pad_w

    def _to_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        b, e, h, w = x.shape
        mu_h, mu_w = self.mask_unit_size
        if h % mu_h != 0 or w % mu_w != 0:
            raise ValueError(
                f"Input size {(h, w)} not divisible by mask_unit_size {self.mask_unit_size}"
            )

        g_h, g_w = h // mu_h, w // mu_w
        tokens = (
            x.reshape(b, e, g_h, mu_h, g_w, mu_w)
            .permute(0, 2, 4, 3, 5, 1)
            .flatten(3, 4)
            .flatten(1, 2)
        )
        return tokens, (g_h, g_w, mu_h, mu_w)

    @staticmethod
    def _from_tokens(tokens: torch.Tensor, token_shape: tuple[int, int, int, int]) -> torch.Tensor:
        g_h, g_w, l_h, l_w = token_shape
        b = tokens.shape[0]
        x = (
            tokens.reshape(b, g_h, g_w, l_h, l_w, -1)
            .permute(0, 5, 1, 3, 2, 4)
            .flatten(4, 5)
            .flatten(2, 3)
        )
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_orig, w_orig = x.shape[-2], x.shape[-1]

        x = self.in_proj(x)
        x, _, _ = self._pad_to_multiple(x, self.mask_unit_size)

        tokens, token_shape = self._to_tokens(x)
        feat_tokens = self.backbone(tokens)
        feat = self._from_tokens(feat_tokens, token_shape)
        feat = feat[..., :h_orig, :w_orig]

        return self.proj(feat)


def normalize_state_dict(state_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected dict-like checkpoint, got {type(state_dict)}")

    for wrap_key in ("model_state", "state_dict", "model", "module"):
        if wrap_key in state_dict and isinstance(state_dict[wrap_key], dict):
            state_dict = state_dict[wrap_key]

    return {k: v for k, v in state_dict.items() if torch.is_tensor(v)}


def extract_backbone_state_dict(state_dict: dict[str, torch.Tensor], backbone: nn.Module) -> dict[str, torch.Tensor]:
    backbone_keys = set(backbone.state_dict().keys())

    def candidates(k: str) -> list[str]:
        out = [k]
        for p in (
            "module.",
            "model.",
            "backbone.",
            "encoder.",
            "encoder_decoder.",
            "net.",
            "core.",
            "base_model.",
        ):
            if k.startswith(p):
                out.append(k[len(p):])

        parts = k.split(".")
        for i in range(1, min(8, len(parts))):
            out.append(".".join(parts[i:]))

        for i, part in enumerate(parts):
            if part in {"backbone", "encoder_decoder", "encoder"} and i + 1 < len(parts):
                out.append(".".join(parts[i + 1 :]))

        seen = set()
        uniq = []
        for cand in out:
            if cand not in seen:
                uniq.append(cand)
                seen.add(cand)
        return uniq

    mapped: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        for cand in candidates(k):
            if cand in backbone_keys:
                mapped[cand] = v
                break
    return mapped


def load_pretrained_backbone(
    model: O3Delta1Model,
    *,
    repo_id: str,
    filename: str,
    weights_dir: Path,
) -> tuple[Path, int, int]:
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / filename
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(weights_dir))

    state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    state_dict = normalize_state_dict(state_dict)
    backbone_state = extract_backbone_state_dict(state_dict, model.backbone)

    model_backbone_sd = model.backbone.state_dict()
    compatible = {
        k: v
        for k, v in backbone_state.items()
        if k in model_backbone_sd and tuple(v.shape) == tuple(model_backbone_sd[k].shape)
    }
    skipped = len(backbone_state) - len(compatible)

    model.backbone.load_state_dict(compatible, strict=False)
    return weights_path, len(compatible), skipped
