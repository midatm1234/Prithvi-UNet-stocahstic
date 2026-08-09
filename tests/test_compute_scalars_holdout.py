from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1] / "examples" / "CORDEX_ML"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from compute_scalars_cordex import _select_scalar_training_partition


class _IndexDataset(torch.utils.data.Dataset):
    def __init__(self, count: int) -> None:
        self.count = int(count)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> int:
        return int(index)


def _config(tmp_path: Path, *, same_sources: bool = True, fraction=0.10):
    train_predictor = tmp_path / "train_predictor.nc"
    train_target = tmp_path / "train_target.nc"
    validation_predictor = (
        train_predictor if same_sources else tmp_path / "validation_predictor.nc"
    )
    validation_target = (
        train_target if same_sources else tmp_path / "validation_target.nc"
    )
    return SimpleNamespace(
        data=SimpleNamespace(
            training_predictor_paths=[str(train_predictor)],
            training_target_paths=[str(train_target)],
            validation_predictor_paths=[str(validation_predictor)],
            validation_target_paths=[str(validation_target)],
            validation_holdout_fraction=fraction,
            validation_holdout_strategy="contiguous_tail",
        )
    )


def test_scalar_selection_excludes_the_same_contiguous_validation_tail(tmp_path):
    config = _config(tmp_path)
    dataset = _IndexDataset(100)

    selected, audit = _select_scalar_training_partition(
        dataset,
        config=config,
        predictor_files=config.data.training_predictor_paths,
        target_files=config.data.training_target_paths,
    )

    assert len(selected) == 90
    assert [selected[index] for index in (0, 89)] == [0, 89]
    assert audit == {
        "policy": "leading_training_partition_excluding_contiguous_validation_tail",
        "source_sample_count": 100,
        "used_sample_count": 90,
        "used_index_start": 0,
        "used_index_stop_exclusive": 90,
        "excluded_index_start": 90,
        "excluded_index_stop_exclusive": 100,
        "config_training_validation_sources_identical": True,
        "validation_holdout_fraction": 0.10,
        "validation_holdout_strategy": "contiguous_tail",
    }


def test_scalar_selection_refuses_holdout_indices_on_different_cli_files(tmp_path):
    config = _config(tmp_path)

    with pytest.raises(ValueError, match="do not match config.data.training"):
        _select_scalar_training_partition(
            _IndexDataset(100),
            config=config,
            predictor_files=[str(tmp_path / "other_predictor.nc")],
            target_files=config.data.training_target_paths,
        )


def test_separate_validation_sources_leave_all_training_samples_available(tmp_path):
    config = _config(tmp_path, same_sources=False)
    dataset = _IndexDataset(37)

    selected, audit = _select_scalar_training_partition(
        dataset,
        config=config,
        predictor_files=config.data.training_predictor_paths,
        target_files=config.data.training_target_paths,
    )

    assert selected is dataset
    assert len(selected) == 37
    assert audit["policy"] == "all_supplied_samples"
    assert audit["source_sample_count"] == 37
    assert audit["used_sample_count"] == 37
    assert audit["config_training_validation_sources_identical"] is False


def test_scalar_selection_rejects_unknown_holdout_strategy(tmp_path):
    config = _config(tmp_path)
    config.data.validation_holdout_strategy = "random"

    with pytest.raises(ValueError, match="contiguous_tail"):
        _select_scalar_training_partition(
            _IndexDataset(10),
            config=config,
            predictor_files=config.data.training_predictor_paths,
            target_files=config.data.training_target_paths,
        )
