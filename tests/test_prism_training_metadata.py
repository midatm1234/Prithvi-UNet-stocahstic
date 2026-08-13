from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from types import MethodType

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
            "date": "2001-02-03",
            "tile_index": 7,
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
    if workflow == "NARR_PRISM":
        assert sample["date"] == "2001-02-03"
        assert sample["tile_index"] == 7
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


def _date_groups(indices, tiles_per_date):
    return [
        [index // tiles_per_date for index in indices[start : start + tiles_per_date]]
        for start in range(0, len(indices), tiles_per_date)
    ]


def test_cached_training_sampler_groups_dates_and_is_resume_reproducible():
    module = _load_training_module("NARR_PRISM")
    sampler = module._DateGroupedTileSampler(
        num_dates=7, tiles_per_date=22, seed=1234
    )
    epoch_zero = list(sampler)
    assert len(epoch_zero) == 7 * 22
    assert all(len(set(group)) == 1 for group in _date_groups(epoch_zero, 22))
    assert len({group[0] for group in _date_groups(epoch_zero, 22)}) == 7

    duplicate = module._DateGroupedTileSampler(
        num_dates=7, tiles_per_date=22, seed=1234
    )
    assert list(duplicate) == epoch_zero
    sampler.set_epoch(3)
    epoch_three = list(sampler)
    assert epoch_three != epoch_zero
    resumed = module._DateGroupedTileSampler(
        num_dates=7, tiles_per_date=22, seed=1234
    )
    resumed.set_epoch(3)
    assert list(resumed) == epoch_three


def test_cached_training_sampler_has_equal_distributed_rank_lengths():
    module = _load_training_module("NARR_PRISM")
    samplers = [
        module._DateGroupedTileSampler(
            num_dates=7, tiles_per_date=22, seed=9, rank=rank, world_size=3
        )
        for rank in range(3)
    ]
    assert {len(sampler) for sampler in samplers} == {3 * 22}
    for sampler in samplers:
        indices = list(sampler)
        assert len(indices) == len(sampler)
        assert all(len(set(group)) == 1 for group in _date_groups(indices, 22))


def test_opt_in_daily_predictor_cache_is_exact_and_loads_once_per_date():
    torch = pytest.importorskip("torch")
    module = _load_training_module("NARR_PRISM")
    dataset_type = module.NarrPrismDataset
    dataset = object.__new__(dataset_type)
    dataset.fine_shape = (9, 11)
    dataset.cache_predictor_days = False
    dataset._cached_predictor_date = None
    dataset._cached_predictor_day = None
    full = torch.arange(4 * 9 * 11, dtype=torch.float32).reshape(4, 9, 11)
    loads = []

    def fake_load(self, sample_date, lat_slice=slice(None), lon_slice=slice(None)):
        loads.append(str(sample_date))
        return full[..., lat_slice, lon_slice].clone()

    dataset._load_predictor = MethodType(fake_load, dataset)
    slices = ((slice(0, 6), slice(1, 8)), (slice(3, 9), slice(4, 11)))
    direct = [
        dataset._load_predictor_crop("2001-02-03", *crop)
        for crop in slices
    ]
    assert loads == ["2001-02-03", "2001-02-03"]

    loads.clear()
    dataset.cache_predictor_days = True
    cached = [
        dataset._load_predictor_crop("2001-02-03", *crop)
        for crop in slices
    ]
    assert loads == ["2001-02-03"]
    for expected, observed in zip(direct, cached, strict=True):
        assert torch.equal(expected, observed)

    dataset._load_predictor_crop("2001-02-04", *slices[0])
    assert loads == ["2001-02-03", "2001-02-04"]
