from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_training_module(workflow: str):
    example_dir = REPO_ROOT / "examples" / workflow
    script_name = f"{workflow.lower()}_training.py"
    module_name = f"_test_{workflow.lower()}_training"
    sys.path.insert(0, str(example_dir))
    try:
        spec = importlib.util.spec_from_file_location(module_name, example_dir / script_name)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(example_dir))


class _FakeBase:
    _elevation = None

    def __len__(self):
        return 1

    def __getitem__(self, _index):
        torch = pytest.importorskip("torch")
        return {
            "x": torch.zeros(4, 35, 37),
            "y": torch.zeros(3, 16, 18),
            "__scaler_offset": torch.tensor([11, 13]),
            "__input_scaler_offset": torch.tensor([-5, -3]),
            "__output_scaler_offset": torch.tensor([11, 13]),
            "__output_crop": torch.tensor([16, 16, 16, 18]),
        }


@pytest.mark.parametrize("workflow", ["NARR_PRISM", "MERRA_PRISM"])
def test_training_wrapper_preserves_spatial_scaler_and_halo_metadata(workflow):
    torch = pytest.importorskip("torch")
    module = _load_training_module(workflow)
    wrapped = module._WrappedDataset(_FakeBase(), num_static_channels=0, pad_multiple=32)
    sample = wrapped[0]

    assert sample["x"].shape[-2:] == (64, 64)
    assert sample["y"].shape[-2:] == (16, 18)
    torch.testing.assert_close(sample["__scaler_offset"], torch.tensor([11, 13]))
    torch.testing.assert_close(sample["__input_scaler_offset"], torch.tensor([-5, -3]))
    torch.testing.assert_close(sample["__output_scaler_offset"], torch.tensor([11, 13]))
    torch.testing.assert_close(sample["__output_crop"], torch.tensor([16, 16, 16, 18]))


@pytest.mark.parametrize("workflow", ["NARR_PRISM", "MERRA_PRISM"])
def test_validation_loader_uses_held_out_date_split(workflow, monkeypatch):
    module = _load_training_module(workflow)
    modes = []

    def fake_build(_path, _config, mode, **_kwargs):
        modes.append(mode)
        return mode

    monkeypatch.setattr(module, "_build_dataloader", fake_build)
    config = SimpleNamespace(
        dates={
            "training": {"start": "2000-01-01", "end": "2000-12-31"},
            "validation": {"start": "2001-01-01", "end": "2001-12-31"},
        }
    )
    train, validation = module.get_dataloaders("config.yaml", config)

    assert (train, validation) == ("training", "validation")
    assert modes == ["training", "validation"]
