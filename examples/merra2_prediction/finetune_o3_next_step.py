from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from torch import nn
from torch.utils.data import DataLoader, Dataset

from o3_pipeline_utils import (
    O3Delta1Model,
    load_pretrained_backbone,
    load_yaml_config,
    normalize_state_dict,
    resolve_predictor_vars,
    resolve_target_var_name,
    resolve_runtime_paths,
)

DEFAULT_CONFIG_PATH = str((Path(__file__).resolve().parent / "o3_pipeline_config.yaml"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune next-step chemistry model on preprocessed MERRA2 pairs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML configuration",
    )
    p.add_argument(
        "--run-name",
        default=None,
        help="Optional run name (default: chem_ft_<UTC timestamp>)",
    )
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class O3PairsDataset(Dataset):
    def __init__(
        self,
        *,
        path: Path,
        predictor_vars: list[str],
        target_var: str,
        normalize_inputs: bool,
        normalize_target: bool,
        input_mu: np.ndarray | None,
        input_sigma: np.ndarray | None,
        target_mu: float | None,
        target_sigma: float | None,
        eps: float,
        crop_size: tuple[int, int] | None,
        random_crop: bool,
    ):
        self.path = path
        self.predictor_vars = predictor_vars
        self.target_var = target_var
        self.normalize_inputs = normalize_inputs
        self.normalize_target = normalize_target
        self.eps = eps

        self.input_mu = input_mu
        self.input_sigma = input_sigma
        self.target_mu = target_mu
        self.target_sigma = target_sigma
        self.crop_size = crop_size
        self.random_crop = random_crop

        ds = xr.open_dataset(path, engine="h5netcdf" if path.suffix in {".nc", ".nc4"} else None)
        missing = [v for v in predictor_vars + [target_var] if v not in ds.data_vars]
        if missing:
            raise KeyError(f"Missing variables in {path}: {missing}")

        x = ds[predictor_vars].to_array("channel").transpose("time", "channel", "lat", "lon")
        y = ds[target_var].transpose("time", "lat", "lon")

        self.x = x.values.astype(np.float32)
        self.y = y.values.astype(np.float32)
        self.time = ds["time"].values if "time" in ds.coords else None
        self.time_out = ds["time_out"].values if "time_out" in ds.coords else None

        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError(f"Input/target sample mismatch in {path}: {self.x.shape[0]} vs {self.y.shape[0]}")

        self.height = int(self.x.shape[2])
        self.width = int(self.x.shape[3])
        if self.crop_size is not None:
            ch, cw = int(self.crop_size[0]), int(self.crop_size[1])
            if ch <= 0 or cw <= 0:
                raise ValueError(f"crop_size must be positive, got {self.crop_size}")
            if ch > self.height or cw > self.width:
                raise ValueError(
                    f"crop_size {self.crop_size} exceeds data size {(self.height, self.width)}"
                )

        if self.normalize_inputs:
            if self.input_mu is None or self.input_sigma is None:
                raise ValueError("normalize_inputs=True requires input scalers")
            if self.input_mu.shape[0] != self.x.shape[1]:
                raise ValueError(
                    f"Input scaler channel mismatch: mu has {self.input_mu.shape[0]}, x has {self.x.shape[1]}"
                )

        if self.normalize_target and (self.target_mu is None or self.target_sigma is None):
            raise ValueError("normalize_target=True requires target scalers")

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x = self.x[idx]
        y = self.y[idx]

        if self.crop_size is not None:
            ch, cw = int(self.crop_size[0]), int(self.crop_size[1])
            if self.random_crop:
                top = np.random.randint(0, self.height - ch + 1)
                left = np.random.randint(0, self.width - cw + 1)
            else:
                top = (self.height - ch) // 2
                left = (self.width - cw) // 2
            x = x[:, top : top + ch, left : left + cw]
            y = y[top : top + ch, left : left + cw]

        if self.normalize_inputs:
            x = (x - self.input_mu[:, None, None]) / (self.input_sigma[:, None, None] + self.eps)

        if self.normalize_target:
            y = (y - self.target_mu) / (self.target_sigma + self.eps)

        return {
            "x": torch.from_numpy(x.astype(np.float32, copy=False)),
            "y": torch.from_numpy(y.astype(np.float32, copy=False)),
        }


def resolve_scaler_paths(cfg: dict[str, Any], paths: dict[str, Path]) -> dict[str, Path]:
    scalers_cfg = cfg.get("scalers", {})

    def _res(name: str, default_name: str) -> Path:
        v = scalers_cfg.get(name)
        if v is None:
            return paths["scalers_dir"] / default_name
        p = Path(v)
        if p.is_absolute():
            return p
        return (paths["base_dir"] / p).resolve()

    return {
        "input_mu": _res("input_mu_file", "inputs_mean.npy"),
        "input_sigma": _res("input_sigma_file", "inputs_std.npy"),
        "target_mu": _res("target_mu_file", "targets_mean.npy"),
        "target_sigma": _res("target_sigma_file", "targets_std.npy"),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool,
    loss_scale: float,
    normalize_target: bool,
    target_mu: float,
    target_sigma: float,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_rmse = 0.0
    n_batches = 0

    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            if x.is_cuda:
                x = x.contiguous(memory_format=torch.channels_last)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp and device.type == "cuda")):
                pred = model(x).squeeze(1)
                loss = F.mse_loss(pred * loss_scale, y * loss_scale)

            if normalize_target:
                pred_phys = pred * target_sigma + target_mu
                y_phys = y * target_sigma + target_mu
            else:
                pred_phys = pred
                y_phys = y

            rmse = torch.sqrt(torch.mean((pred_phys - y_phys) ** 2))

            total_loss += float(loss.item())
            total_rmse += float(rmse.item())
            n_batches += 1

    if n_batches == 0:
        return float("nan"), float("nan")
    return total_loss / n_batches, total_rmse / n_batches


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    paths = resolve_runtime_paths(cfg)

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    scalers_cfg = cfg.get("scalers", {})

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    device_cfg = str(train_cfg.get("device", "auto")).lower()
    if device_cfg == "cpu":
        device = torch.device("cpu")
    elif device_cfg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("training.device=cuda but CUDA is unavailable")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler_paths = resolve_scaler_paths(cfg, paths)
    eps = float(scalers_cfg.get("eps", 1e-6))

    normalize_inputs = bool(train_cfg.get("normalize_inputs", True))
    normalize_target = bool(train_cfg.get("normalize_target", False))

    input_mu = input_sigma = None
    target_mu = target_sigma = None
    if normalize_inputs or normalize_target:
        for name, p in scaler_paths.items():
            if not p.exists():
                raise FileNotFoundError(f"Scaler file missing: {p}")

    if normalize_inputs:
        input_mu = np.load(scaler_paths["input_mu"]).astype(np.float32)
        input_sigma = np.load(scaler_paths["input_sigma"]).astype(np.float32)

    if normalize_target:
        target_mu_arr = np.load(scaler_paths["target_mu"]).astype(np.float32)
        target_sigma_arr = np.load(scaler_paths["target_sigma"]).astype(np.float32)
        target_mu = float(target_mu_arr.reshape(-1)[0])
        target_sigma = float(target_sigma_arr.reshape(-1)[0])
    else:
        target_mu = 0.0
        target_sigma = 1.0

    train_file = paths["train_pairs_file"]
    val_file = paths["val_pairs_file"]
    if not train_file.exists() or not val_file.exists():
        raise FileNotFoundError(
            f"Missing train/val pairs files ({train_file}, {val_file}). Run preprocess_o3_pairs.py first."
        )

    train_meta = xr.open_dataset(
        train_file,
        engine="h5netcdf" if train_file.suffix.lower() in {".nc", ".nc4"} else None,
    )
    predictor_vars = resolve_predictor_vars(data_cfg, dataset_attrs=train_meta.attrs)
    target_var = str(train_meta.attrs.get("target_var", resolve_target_var_name(data_cfg)))
    train_meta.close()

    batch_size = int(train_cfg.get("batch_size", 1))
    num_workers = int(train_cfg.get("num_workers", 0))
    crop = train_cfg.get("crop_size")
    crop_size = tuple(crop) if isinstance(crop, (list, tuple)) and len(crop) == 2 else None
    random_crop = bool(train_cfg.get("random_crop", True))

    train_ds = O3PairsDataset(
        path=train_file,
        predictor_vars=predictor_vars,
        target_var=target_var,
        normalize_inputs=normalize_inputs,
        normalize_target=normalize_target,
        input_mu=input_mu,
        input_sigma=input_sigma,
        target_mu=target_mu,
        target_sigma=target_sigma,
        eps=eps,
        crop_size=crop_size,
        random_crop=random_crop,
    )
    val_ds = O3PairsDataset(
        path=val_file,
        predictor_vars=predictor_vars,
        target_var=target_var,
        normalize_inputs=normalize_inputs,
        normalize_target=normalize_target,
        input_mu=input_mu,
        input_sigma=input_sigma,
        target_mu=target_mu,
        target_sigma=target_sigma,
        eps=eps,
        crop_size=crop_size,
        random_crop=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = O3Delta1Model(
        in_channels=len(predictor_vars),
        embed_dim=int(model_cfg.get("embed_dim", 2560)),
        n_blocks=int(model_cfg.get("n_blocks", 4)),
        n_heads=int(model_cfg.get("n_heads", 16)),
        mlp_multiplier=int(model_cfg.get("mlp_multiplier", 4)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        drop_path=float(model_cfg.get("drop_path", 0.0)),
        mask_unit_size=tuple(model_cfg.get("mask_unit_size", [16, 16])),
    )

    ckpt_init = train_cfg.get("initialize_from_checkpoint")
    if ckpt_init:
        init_path = Path(ckpt_init)
        if not init_path.is_absolute():
            init_path = (paths["base_dir"] / init_path).resolve()
        if not init_path.exists():
            raise FileNotFoundError(f"initialize_from_checkpoint not found: {init_path}")
        loaded = torch.load(str(init_path), map_location="cpu", weights_only=False)
        loaded = normalize_state_dict(loaded)
        model.load_state_dict(loaded, strict=False)
        print(f"Initialized model from checkpoint: {init_path}")
    elif bool(train_cfg.get("use_pretrained_backbone", True)):
        repo_id = str(train_cfg.get("pretrained_repo_id", "ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M"))
        filename = str(train_cfg.get("pretrained_filename", "prithvi.wxc.2300m.v1.pt"))
        weights_path, n_loaded, n_skipped = load_pretrained_backbone(
            model,
            repo_id=repo_id,
            filename=filename,
            weights_dir=paths["weights_dir"],
        )
        print(f"Loaded pretrained backbone from {weights_path} (mapped={n_loaded}, skipped={n_skipped})")

    if bool(train_cfg.get("freeze_backbone", False)):
        for p in model.backbone.parameters():
            p.requires_grad = False
        print("Backbone frozen")

    model = model.to(device)

    lr = float(train_cfg.get("learning_rate", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)

    epochs = int(train_cfg.get("num_epochs", 10))
    loss_scale = float(train_cfg.get("loss_scale", 1.0))
    grad_clip = float(train_cfg.get("grad_clip", 0.0))
    amp = bool(train_cfg.get("amp", True))

    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"))

    run_name = args.run_name or str(train_cfg.get("run_name", "")).strip()
    if not run_name:
        run_name = f"chem_ft_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"

    run_dir = paths["checkpoints_dir"] / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = run_dir / "best.ckpt"
    last_ckpt = run_dir / "last.ckpt"

    manifest = {
        "run_name": run_name,
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "config_path": str(Path(cfg["_config_path"]).resolve()),
        "base_dir": str(paths["base_dir"]),
        "train_pairs_file": str(train_file),
        "val_pairs_file": str(val_file),
        "predictor_vars": predictor_vars,
        "target_var": target_var,
        "normalize_inputs": normalize_inputs,
        "normalize_target": normalize_target,
        "input_mu_file": str(scaler_paths["input_mu"]),
        "input_sigma_file": str(scaler_paths["input_sigma"]),
        "target_mu_file": str(scaler_paths["target_mu"]),
        "target_sigma_file": str(scaler_paths["target_sigma"]),
        "device": str(device),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    best_val_rmse = None
    save_every = int(train_cfg.get("save_every", 1))

    print(f"Training run: {run_name}")
    print(f"Device      : {device}")
    print(f"Train/Val   : {len(train_ds)} / {len(val_ds)} samples")
    print(f"Crop size   : {crop_size} (random={random_crop})")
    print(f"Predictors  : {predictor_vars}")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_rmse_sum = 0.0
        n_train_batches = 0

        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            if x.is_cuda:
                x = x.contiguous(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp and device.type == "cuda")):
                pred = model(x).squeeze(1)
                loss = F.mse_loss(pred * loss_scale, y * loss_scale)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            if normalize_target:
                pred_phys = pred * target_sigma + target_mu
                y_phys = y * target_sigma + target_mu
            else:
                pred_phys = pred
                y_phys = y
            rmse = torch.sqrt(torch.mean((pred_phys - y_phys) ** 2))

            train_loss_sum += float(loss.item())
            train_rmse_sum += float(rmse.item())
            n_train_batches += 1

        train_loss = train_loss_sum / max(n_train_batches, 1)
        train_rmse = train_rmse_sum / max(n_train_batches, 1)

        val_loss, val_rmse = evaluate(
            model,
            val_loader,
            device,
            amp=amp,
            loss_scale=loss_scale,
            normalize_target=normalize_target,
            target_mu=float(target_mu),
            target_sigma=float(target_sigma),
        )

        ckpt_state = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "train_rmse": train_rmse,
            "val_loss": val_loss,
            "val_rmse": val_rmse,
            "predictor_vars": predictor_vars,
            "target_var": target_var,
            "normalize_inputs": normalize_inputs,
            "normalize_target": normalize_target,
            "target_mu": float(target_mu),
            "target_sigma": float(target_sigma),
            "config": cfg,
        }

        torch.save(ckpt_state, last_ckpt)

        is_best = (best_val_rmse is None) or (val_rmse < best_val_rmse)
        if is_best:
            best_val_rmse = val_rmse
            torch.save(ckpt_state, best_ckpt)

        if (epoch % max(save_every, 1) == 0) or is_best or (epoch == epochs):
            epoch_ckpt = run_dir / f"epoch_{epoch:03d}.ckpt"
            torch.save(ckpt_state, epoch_ckpt)

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.6e} train_rmse={train_rmse:.6e} | "
            f"val_loss={val_loss:.6e} val_rmse={val_rmse:.6e} | "
            f"best_val_rmse={best_val_rmse:.6e}"
        )

    print("Fine-tuning complete")
    print(f"Run dir      : {run_dir}")
    print(f"Best ckpt    : {best_ckpt}")
    print(f"Last ckpt    : {last_ckpt}")


if __name__ == "__main__":
    main()
