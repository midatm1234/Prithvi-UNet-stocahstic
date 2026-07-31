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
MERRA_EXAMPLE_DIR = REPO_ROOT / "examples" / "MERRA_PRISM"


def _load_example_module(script_name: str):
    module_name = f"_test_merra_target_free_{Path(script_name).stem}"
    sys.path.insert(0, str(MERRA_EXAMPLE_DIR))
    try:
        spec = importlib.util.spec_from_file_location(
            module_name, MERRA_EXAMPLE_DIR / script_name
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(MERRA_EXAMPLE_DIR))


def _target_free_config(tmp_path: Path, *, case_name: str = "target_free") -> dict:
    return {
        "case_name": case_name,
        "data": {
            "type": "merra_prism",
            "predictor_dir": str(tmp_path / "merra"),
            # data.target_dir is deliberately omitted: predictor-only inference
            # must not require or inspect raw PRISM truth.
            "preprocessed_dir": str(tmp_path / "preprocessed"),
            "use_preprocessed": True,
            "target_variables": ["tmax"],
            "output_vars": ["tmax"],
            "predictor_variables": {"T": [850]},
            "input_vars": ["T_850", "mask_T_850"],
            "n_input_timestamps": 1,
            "train_crop_size_lat": 4,
            "train_crop_size_lon": 4,
            "scalar_stride": 1,
            "regrid_method": "bilinear",
        },
        "dates": {
            "training": {"start": "1999-01-01", "end": "1999-01-01"},
            "inference": {"start": "2000-01-01", "end": "2000-01-01"},
        },
        "preprocess": {
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


def _write_merra_day(predictor_dir: Path) -> None:
    predictor_dir.mkdir(parents=True)
    lat = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    lon = np.array([10.0, 11.0, 12.0], dtype=np.float64)
    field = (lat[:, None] + lon[None, :]).astype(np.float32)
    ds = xr.Dataset(
        {
            "T": (
                ("time", "lev", "lat", "lon"),
                field[None, None, ...],
            )
        },
        coords={
            "time": [np.datetime64("2000-01-01")],
            "lev": [850.0],
            "lat": lat,
            "lon": lon,
        },
    )
    ds.to_netcdf(predictor_dir / "M2I3NPASM_subset_20000101.nc4")


def _write_case_scalars(cfg: dict, predictor_signature: str, shape: tuple[int, int]) -> None:
    scalar_dir = norm.case_preprocess_dir(cfg) / "scalars"
    scalar_dir.mkdir(parents=True)
    np.save(scalar_dir / "inputs_mean.npy", np.array([11.0, 0.0], dtype=np.float32))
    np.save(scalar_dir / "inputs_std.npy", np.array([2.0, 1.0], dtype=np.float32))
    np.save(
        scalar_dir / "targets_mean.npy",
        np.zeros((1, *shape), dtype=np.float32),
    )
    np.save(
        scalar_dir / "targets_std.npy",
        np.ones((1, *shape), dtype=np.float32),
    )
    training_sources = {"1999-01-01": "a" * 64}
    norm.write_manifest(
        scalar_dir,
        case_name=str(cfg["case_name"]),
        predictor_mode="global",
        cfg=cfg,
        train_date_range=["1999-01-01", "1999-01-01"],
        extra={
            "predictor_preprocessing_signature": predictor_signature,
            norm.TRAINING_SOURCE_ARTIFACT_SIGNATURES_KEY: training_sources,
            norm.TRAINING_SOURCE_ARTIFACT_SPLIT_SIGNATURE_KEY: (
                split_source_artifact_signature(training_sources)
            ),
        },
    )


def test_target_free_preprocessing_and_dataset_use_persisted_canonical_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _target_free_config(tmp_path)
    _write_merra_day(Path(cfg["data"]["predictor_dir"]))

    target_lat = np.array([0.25, 0.75, 1.25, 1.75], dtype=np.float64)
    target_lon = np.array([10.25, 10.75, 11.25, 11.75], dtype=np.float64)
    canonical = ensure_canonical_grid(
        norm.case_preprocess_dir(cfg), target_lat, target_lon, source="training-prism.nc"
    )

    preprocessor = _load_example_module("preproc_merra_prism.py")
    # Keep this unit test independent of an optional ESMF installation.
    monkeypatch.setattr(preprocessor, "xe", None)
    output_dir = preprocessor.preprocess(cfg, mode="inference")
    product = output_dir / "merra_prism_20000101.nc"

    with xr.open_dataset(product) as ds:
        assert "predictor_T_850" in ds
        assert "target_tmax" not in ds
        np.testing.assert_array_equal(ds["lat"].values, canonical.lat)
        np.testing.assert_array_equal(ds["lon"].values, canonical.lon)
        predictor_signature = str(
            ds.attrs[PREDICTOR_PREPROCESSING_SIGNATURE_ATTR]
        )

    _write_case_scalars(cfg, predictor_signature, canonical.shape)
    norm.log_scalar_summary(cfg, "target-free-test", logger=lambda _message: None)

    config_path = tmp_path / "target_free.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    dataset_module = _load_example_module("merra_prism_dataset.py")
    dataset = dataset_module.MerraPrismDataset(config_path, mode="inference")

    assert dataset.has_observed_targets is False
    sample = dataset[0]
    assert tuple(sample["x"].shape) == (2, *canonical.shape)
    assert tuple(sample["y"].shape) == (1, *canonical.shape)
    assert np.all(sample["y"].numpy() == 0.0)
    with pytest.raises(RuntimeError, match="without observed target"):
        dataset._load_target_valid_mask()


def test_target_free_preprocessing_requires_an_existing_canonical_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _target_free_config(tmp_path, case_name="missing_grid")
    _write_merra_day(Path(cfg["data"]["predictor_dir"]))
    preprocessor = _load_example_module("preproc_merra_prism.py")
    monkeypatch.setattr(preprocessor, "xe", None)

    with pytest.raises(FileNotFoundError, match="No PRISM grid contract"):
        preprocessor.preprocess(cfg, mode="inference")
