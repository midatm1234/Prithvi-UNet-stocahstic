from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
import torch

from examples.NARR_PRISM.narr_prism_inference import (
    _select_inference_dates,
    _tensor_to_float32_numpy,
)


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
