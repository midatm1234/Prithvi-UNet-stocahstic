"""Helpers for building CORDEX inference datasets and dataloaders."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from cordex_dataset import CordexDownscaleDataset


REPO_ROOT = Path(__file__).resolve().parents[2]


def _coerce_path(path: Any) -> str:
    path_obj = Path(path).expanduser()
    if path_obj.is_absolute():
        return str(path_obj)
    return str((REPO_ROOT / path_obj).resolve())


def _coerce_paths(paths: Sequence[Any]) -> list[str]:
    return [_coerce_path(path) for path in paths]


def _level_suffix(level: Any) -> str:
    text = str(level)
    return text[:-2] if text.endswith(".0") else text


def build_predictor_names(config: Any) -> list[str]:
    input_vars = list(getattr(config.data, "input_vars", []))
    input_levels = list(getattr(config.data, "input_levels", []))
    names: list[str] = []
    for var in input_vars:
        for level in input_levels:
            names.append(f"{var}_{_level_suffix(level)}")
    return names


class CordexWrappedDataset(Dataset):
    """Expose static predictors separately for the model forward pass."""

    def __init__(self, base_dataset: CordexDownscaleDataset) -> None:
        self.base = base_dataset

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.base[idx]
        x = sample["x"]
        if not getattr(self.base, "use_static", True):
            return {"x": x, "y": sample["y"]}
        dynamic = x[:-1]
        static = x[-1:].clone()
        return {"x": dynamic, "y": sample["y"], "static_x": static, "static_y": static}


def build_inference_dataset(
    config: Any,
    predictor_paths: Sequence[Any],
    target_paths: Sequence[Any],
    *,
    crop_size: tuple[int, int] | None = None,
) -> CordexDownscaleDataset:
    predictor_files = _coerce_paths(predictor_paths)
    target_files = _coerce_paths(target_paths)
    if not predictor_files:
        raise ValueError("predictor_paths must contain at least one file")
    if len(predictor_files) != len(target_files):
        raise ValueError("predictor_paths and target_paths must be the same length")

    use_static = bool(getattr(config.data, "use_static", getattr(config, "finetune_w_static", True)))
    static_path = getattr(config.data, "static_path", None)
    if use_static and not static_path:
        raise ValueError("config.data.static_path is required for inference datasets")

    predictor_variables = build_predictor_names(config)
    target_variables = list(getattr(config.data, "output_vars", [])) or None

    return CordexDownscaleDataset(
        predictor_files=predictor_files,
        target_files=target_files,
        orography_file=_coerce_path(static_path) if use_static else None,
        predictor_variables=predictor_variables,
        target_variables=target_variables,
        crop_size=crop_size,
        random_crop=False,
        seed=None,
        use_static=use_static,
    )
