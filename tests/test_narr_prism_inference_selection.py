from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

from examples.NARR_PRISM.narr_prism_inference import (
    TARGET_VALID_MASK_CONTENT_SHA256_ATTR,
    TARGET_VALID_MASK_CRITERION_ATTR,
    TARGET_VALID_MASK_GRID_FINGERPRINT_ATTR,
    TARGET_VALID_MASK_SHA256_ATTR,
    TARGET_VALID_MASK_SOURCE_SPLIT_ATTR,
    TARGET_VALID_MASK_TRAINING_SOURCE_SIGNATURE_ATTR,
    _open_streaming_output,
    _select_inference_dates,
    _split_output_root,
    _target_valid_mask_content_sha256,
    _tensor_to_float32_numpy,
    _validate_dataset_dates,
    _validate_prediction_split,
)
from granitewxc.utils.normalization import TARGET_VALID_MASK_CRITERION
from granitewxc.utils.prism_grid import validate_prism_grid


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "examples" / "NARR_PRISM" / "notebooks"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_tensor_to_float32_numpy_supports_inference_dtypes(dtype) -> None:
    tensor = torch.tensor([1.25, -2.5], dtype=dtype, requires_grad=True)

    converted = _tensor_to_float32_numpy(tensor)

    assert converted.dtype == np.float32
    np.testing.assert_array_equal(converted, np.array([1.25, -2.5], dtype=np.float32))


def test_selected_inference_dates_preserve_requested_order_and_native_values() -> None:
    start = date(2020, 1, 1)
    available = [start + timedelta(days=offset) for offset in range(5)]

    selected = _select_inference_dates(
        available,
        [np.datetime64("2020-01-05"), "2020-01-02"],
    )

    assert selected == [available[4], available[1]]
    assert _select_inference_dates(available, None) == available


@pytest.mark.parametrize(
    "selected, message",
    [
        ([], "at least one"),
        (["2020-01-02", "2020-01-02"], "duplicates"),
        (["2020-02-01"], "outside"),
        (["not-a-date"], "invalid ISO date"),
    ],
)
def test_selected_inference_dates_reject_invalid_requests(selected, message) -> None:
    available = [date(2020, 1, 1), date(2020, 1, 2)]
    with pytest.raises(ValueError, match=message):
        _select_inference_dates(available, selected)


def test_prediction_split_dates_are_exact_inclusive_and_leap_safe() -> None:
    cfg = {
        "dates": {
            "validation": {
                "start": "2015-02-28",
                "end": "2015-03-01",
            },
            "inference": {
                "start": "2016-02-28",
                "end": "2016-03-01",
            },
        }
    }
    validation = [date(2015, 2, 28), date(2015, 3, 1)]
    inference = [
        date(2016, 2, 28),
        date(2016, 2, 29),
        date(2016, 3, 1),
    ]
    assert _validate_dataset_dates(cfg, validation, "validation") == validation
    assert _validate_dataset_dates(cfg, inference) == inference
    with pytest.raises(ValueError, match=r"dates\.validation"):
        _validate_dataset_dates(cfg, validation[:-1], "validation")
    with pytest.raises(ValueError, match="split must be one of"):
        _validate_prediction_split("training")


def test_validation_output_root_isolated_without_changing_inference(
    tmp_path: Path,
) -> None:
    assert _split_output_root(tmp_path, "inference") == tmp_path
    assert _split_output_root(tmp_path, "validation") == tmp_path / "validation"
    assert (
        _split_output_root(tmp_path / "validation", "validation")
        == tmp_path / "validation"
    )


def test_daily_deterministic_output_persists_support_mask_provenance(
    tmp_path: Path,
) -> None:
    lat = np.array([40.0, 40.5], dtype=np.float64)
    lon = np.array([-121.0, -120.5, -120.0], dtype=np.float64)
    mask = np.array([[True, False, True], [True, True, False]])
    provenance = {
        TARGET_VALID_MASK_SHA256_ATTR: "a" * 64,
        TARGET_VALID_MASK_CONTENT_SHA256_ATTR: (
            _target_valid_mask_content_sha256(mask)
        ),
        TARGET_VALID_MASK_CRITERION_ATTR: TARGET_VALID_MASK_CRITERION,
        TARGET_VALID_MASK_SOURCE_SPLIT_ATTR: "training",
        TARGET_VALID_MASK_TRAINING_SOURCE_SIGNATURE_ATTR: "b" * 64,
        TARGET_VALID_MASK_GRID_FINGERPRINT_ATTR: validate_prism_grid(
            lat, lon, context="deterministic-output-test"
        ).fingerprint,
    }
    output = tmp_path / "daily.nc"
    handle = _open_streaming_output(
        output,
        ["ppt"],
        lat,
        lon,
        "/checkpoints/phase1.ckpt",
        [date(2020, 1, 1)],
        "case",
        mask,
        provenance,
        "validation",
    )
    handle.variables["time"][:] = np.array([18262], dtype=np.int32)
    handle.variables["ppt"][0:1, :, :] = np.zeros(
        (1, *mask.shape), dtype=np.float32
    )
    handle.flush()
    handle.close()

    with xr.open_dataset(output) as dataset:
        np.testing.assert_array_equal(
            dataset["prism_valid_mask"].values.astype(bool), mask
        )
        for name, value in provenance.items():
            assert dataset.attrs[name] == value
        assert dataset.attrs["dataset_split"] == "validation"
        assert dataset.attrs["split_start"] == "2020-01-01"
        assert dataset.attrs["split_end"] == "2020-01-01"


def _notebook(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_main_inference_notebook_has_no_spatial_mean_section() -> None:
    notebook = _notebook(NOTEBOOK_DIR / "narr_prism_inference.ipynb")
    cell_ids = {cell.get("id") for cell in notebook["cells"]}
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "timeseries-section" not in cell_ids
    assert "plot-timeseries" not in cell_ids
    assert "spatial_mean" not in source


def test_random_ten_notebook_is_cpu_only_and_plots_all_targets() -> None:
    notebook = _notebook(NOTEBOOK_DIR / "narr_prism_inference_random10_cpu.ipynb")
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "SAMPLE_COUNT = 10" in source
    assert 'device = torch.device("cpu")' in source
    assert "selected_dates=selected_dates" in source
    assert 'PLOT_VARIABLES = ["ppt", "tmax", "tmin"]' in source
    assert 'f"target_{variable}"' in source
    assert "Inference − PRISM" in source
    assert "spatial_mean" not in source
