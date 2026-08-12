"""Target-free NARR preprocessing and inference-dataset contract tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml

from granitewxc.utils import normalization as norm
from granitewxc.utils.prism_grid import ensure_canonical_grid
from granitewxc.utils.prism_preprocessed import (
    PREDICTOR_PREPROCESSING_SIGNATURE_ATTR,
    split_source_artifact_signature,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
NARR_EXAMPLE_DIR = REPO_ROOT / "examples" / "NARR_PRISM"


def _load_example_module(script_name: str):
    name = f"_test_narr_target_free_{Path(script_name).stem}"
    sys.path.insert(0, str(NARR_EXAMPLE_DIR))
    try:
        spec = importlib.util.spec_from_file_location(
            name, NARR_EXAMPLE_DIR / script_name
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(NARR_EXAMPLE_DIR))


def _config(tmp_path: Path, *, case_name: str = "target_free_narr") -> dict:
    return {
        "case_name": case_name,
        "data": {
            "type": "narr_prism",
            "predictor_dir": str(tmp_path / "narr"),
            # No target_dir: independent inference must not discover PRISM.
            "preprocessed_dir": str(tmp_path / "preprocessed"),
            "use_preprocessed": True,
            "target_variables": ["tmax"],
            "output_vars": ["tmax"],
            "predictor_variables": {"shum": [850]},
            "input_vars": ["shum_850", "mask_shum_850"],
            "n_input_timestamps": 1,
            "train_crop_size_lat": 4,
            "train_crop_size_lon": 4,
            "regrid_method": "bilinear",
        },
        "dates": {
            "training": {
                "start": "1999-01-01",
                "end": "1999-01-01",
            },
            "validation": {
                "start": "2000-01-01",
                "end": "2000-01-01",
            },
            "inference": {
                "start": "2000-01-01",
                "end": "2000-01-01",
            },
        },
        "preprocess": {
            "save_train_targets": True,
            "save_val_targets": True,
            "save_inference_targets": False,
            "inference_include_observed_targets_for_eval": False,
        },
        "normalization": {
            "predictor_method": "standardize",
            "predictor_mode": "global",
            "target_method": "per_variable",
        },
        "predictands": {
            "tmax": {
                "normalization": {
                    "method": "standardize",
                    "mode": "gridpoint",
                }
            }
        },
    }


def _write_narr_month(root: Path) -> None:
    directory = root / "shum"
    directory.mkdir(parents=True)
    axis = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    lon, lat = np.meshgrid(10.0 + axis, axis)
    values = (lat + lon).astype(np.float32)
    dataset = xr.Dataset(
        {
            "shum": (
                ("time", "level", "y", "x"),
                values[None, None],
            )
        },
        coords={
            "time": [np.datetime64("2000-01-01")],
            "level": [850.0],
            "lat": (("y", "x"), lat),
            "lon": (("y", "x"), lon),
        },
    )
    dataset.to_netcdf(directory / "shum.200001.nc")


def _write_scalars(
    cfg: dict, predictor_signature: str, shape: tuple[int, int]
) -> None:
    scalar_dir = norm.case_preprocess_dir(cfg) / "scalars"
    scalar_dir.mkdir(parents=True)
    np.save(
        scalar_dir / "inputs_mean.npy",
        np.array([11.0, 0.0], dtype=np.float32),
    )
    np.save(
        scalar_dir / "inputs_std.npy",
        np.array([2.0, 1.0], dtype=np.float32),
    )
    np.save(
        scalar_dir / "targets_mean.npy",
        np.zeros((1, *shape), dtype=np.float32),
    )
    np.save(
        scalar_dir / "targets_std.npy",
        np.ones((1, *shape), dtype=np.float32),
    )
    mask_path = norm.target_valid_mask_path(scalar_dir)
    np.save(mask_path, np.ones(shape, dtype=bool))
    sources = {"1999-01-01": "a" * 64}
    split_signature = split_source_artifact_signature(sources)
    mask_entry = norm.build_target_valid_mask_manifest_entry(
        mask_path,
        cfg=cfg,
        training_source_artifact_split_signature=split_signature,
    )
    norm.write_manifest(
        scalar_dir,
        case_name=str(cfg["case_name"]),
        predictor_mode="global",
        cfg=cfg,
        train_date_range=["1999-01-01", "1999-01-01"],
        extra={
            "predictor_preprocessing_signature": predictor_signature,
            norm.TRAINING_SOURCE_ARTIFACT_SIGNATURES_KEY: sources,
            norm.TRAINING_SOURCE_ARTIFACT_SPLIT_SIGNATURE_KEY: (
                split_signature
            ),
            norm.TARGET_VALID_MASK_MANIFEST_KEY: mask_entry,
        },
    )


def test_target_free_narr_preprocessing_and_dataset(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _write_narr_month(Path(cfg["data"]["predictor_dir"]))
    canonical = ensure_canonical_grid(
        norm.case_preprocess_dir(cfg),
        np.array([0.25, 0.75, 1.25, 1.75]),
        np.array([10.25, 10.75, 11.25, 11.75]),
        source="training-prism.nc",
    )
    preprocessor = _load_example_module("preproc_narr_prism.py")
    output = preprocessor.preprocess(cfg, mode="inference")
    product = output / "narr_prism_20000101.nc"

    with xr.open_dataset(product) as dataset:
        assert "predictor_shum_850" in dataset
        assert "target_tmax" not in dataset
        assert dataset.attrs["contains_targets"] == "False"
        np.testing.assert_array_equal(dataset.lat.values, canonical.lat)
        np.testing.assert_array_equal(dataset.lon.values, canonical.lon)
        signature = str(
            dataset.attrs[PREDICTOR_PREPROCESSING_SIGNATURE_ATTR]
        )

    _write_scalars(cfg, signature, canonical.shape)
    np.testing.assert_array_equal(
        norm.load_target_valid_mask(
            cfg,
            role="deterministic inference test",
            expected_shape=canonical.shape,
        ),
        np.ones(canonical.shape, dtype=bool),
    )
    config_path = tmp_path / "narr.yaml"
    config_path.write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )
    dataset_module = _load_example_module("narr_prism_dataset.py")
    inference = dataset_module.NarrPrismDataset(
        config_path, mode="inference"
    )
    assert inference.has_observed_targets is False
    sample = inference[0]
    assert tuple(sample["x"].shape) == (2, *canonical.shape)
    assert tuple(sample["y"].shape) == (1, *canonical.shape)
    assert np.all(sample["y"].numpy() == 0.0)
    with pytest.raises(RuntimeError, match="without observed target"):
        inference._load_targets(inference.dates[0])


def test_target_free_narr_preprocessing_requires_canonical_grid(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path, case_name="missing_grid")
    _write_narr_month(Path(cfg["data"]["predictor_dir"]))
    preprocessor = _load_example_module("preproc_narr_prism.py")
    with pytest.raises(FileNotFoundError, match="No PRISM grid contract"):
        preprocessor.preprocess(cfg, mode="inference")


def test_validation_prediction_reader_never_discovers_or_loads_targets(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path, case_name="target_free_validation")
    cfg["data"]["use_preprocessed"] = False
    _write_narr_month(Path(cfg["data"]["predictor_dir"]))
    canonical = ensure_canonical_grid(
        norm.case_preprocess_dir(cfg),
        np.array([0.25, 0.75, 1.25, 1.75]),
        np.array([10.25, 10.75, 11.25, 11.75]),
        source="training-prism.nc",
    )
    config_path = tmp_path / "validation.yaml"
    config_path.write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )
    dataset_module = _load_example_module("narr_prism_dataset.py")
    validation = dataset_module.NarrPrismDataset(
        config_path,
        mode="validation",
        load_observed_targets=False,
    )
    assert validation.has_observed_targets is False
    assert validation.fine_shape == canonical.shape
    sample = validation[0]
    assert tuple(sample["y"].shape) == (1, *canonical.shape)
    assert np.all(sample["y"].numpy() == 0.0)
    with pytest.raises(RuntimeError, match="without observed target"):
        validation._load_targets(validation.dates[0])
