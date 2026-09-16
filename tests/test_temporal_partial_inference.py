"""Requested emission dates must not discard available within-split history."""
import cftime
import numpy as np
import pytest
import torch

from granitewxc.temporal.calendar import resolve_calendar
from granitewxc.temporal.inference import run_sequence_inference
from granitewxc.temporal.model import TemporalSequenceModel
from granitewxc.temporal.native_pair import pair_time_scalars
from granitewxc.temporal.config import NativePairBackendConfig, TemporalConfigError
from granitewxc.temporal.sequence_dataset import FrameRef
from tests.temporal_fixtures import make_sequence_batch, temporal_config
from tests.test_temporal_native_pair import _build_pair_model


class AvailableFrames:
    def __init__(self, days):
        self.days = list(days)
        self.batch = make_sequence_batch(batch=1, frames=len(days), use_static=False)
        self.loaded_days = []

    def frames(self):
        return [FrameRef(i, 0, 0, cftime.DatetimeGregorian(2000, 1, day)) for i, day in enumerate(self.days)]

    def calendar(self):
        return resolve_calendar("standard")

    def fine_shape(self):
        return (32, 32)

    def load_frame(self, frame, lat_slice, lon_slice):
        self.loaded_days.append(self.days[frame.file_index])
        i = frame.file_index
        return {"x": self.batch["x"][0, i, :, lat_slice, lon_slice],
                "y": self.batch["y"][0, i, :, lat_slice, lon_slice],
                "valid_mask": self.batch["__target_valid_mask"][0, i, :, lat_slice, lon_slice]}


def _setup(tmp_path, monkeypatch, *, days=range(1, 11), offsets=(1,), split_start=1, time_conditioning=True, cold_start_time_mode="legacy_nominal"):
    cfg = temporal_config(backend="native_pair", context_length=6, warmup_length=3,
        output_length=3, native_pair={"history_offsets": list(offsets), "time_conditioning": time_conditioning, "cold_start_time_mode": cold_start_time_mode})
    model, config = _build_pair_model(tmp_path, cfg, use_static=False)
    config.temporal_splits = {"test": {"start": f"2000-01-{split_start:02d}", "end": "2000-01-10"}}
    runner = TemporalSequenceModel(model, cfg, adapter=model.temporal_adapter).eval()
    source = AvailableFrames(days)
    monkeypatch.setattr("granitewxc.temporal.sources.build_frame_source", lambda *args: source)
    return runner, config, cfg, source


@pytest.mark.parametrize("start_day", [2, 5])
@pytest.mark.parametrize("offsets", [(1,), (1, 3)])
@pytest.mark.parametrize("cold_start_time_mode", ["legacy_nominal", "selected_timestamps"])
def test_partial_native_period_matches_slice_of_full_period(tmp_path, monkeypatch, offsets, cold_start_time_mode, start_day):
    runner, config, cfg, source = _setup(tmp_path, monkeypatch, offsets=offsets, cold_start_time_mode=cold_start_time_mode)
    full = run_sequence_inference(runner, config, cfg, date_start="2000-01-01",
        date_end="2000-01-10", chunk_length=4, verbose=False)[0]
    source.loaded_days.clear()
    partial = run_sequence_inference(runner, config, cfg, date_start=f"2000-01-{start_day:02d}",
        date_end="2000-01-10", chunk_length=4, verbose=False)[0]
    assert partial.dates == full.dates[start_day - 1:]
    np.testing.assert_allclose(partial.predictions, full.predictions[start_day - 1:], rtol=0, atol=0)
    assert start_day - 1 in source.loaded_days


def test_single_requested_date_has_context_without_training_window_requirement(tmp_path, monkeypatch):
    runner, config, cfg, source = _setup(tmp_path, monkeypatch, offsets=(1, 3))
    full = run_sequence_inference(runner, config, cfg, chunk_length=4, verbose=False)[0]
    source.loaded_days.clear()
    single = run_sequence_inference(runner, config, cfg, date_start="2000-01-10",
        date_end="2000-01-10", chunk_length=4, verbose=False)[0]
    assert [date[2] for date in single.dates] == [10]
    np.testing.assert_array_equal(single.predictions, full.predictions[-1:])
    assert set(source.loaded_days) == {7, 8, 9, 10}


def test_declared_split_start_remains_cold_and_excludes_earlier_archive_dates(tmp_path, monkeypatch):
    runner, config, cfg, source = _setup(tmp_path, monkeypatch, split_start=5)
    result = run_sequence_inference(runner, config, cfg, date_start="2000-01-05",
        date_end="2000-01-10", chunk_length=4, verbose=False)[0]
    assert len(result.dates) == 6
    assert min(source.loaded_days) == 5
    assert runner._allow_cold_start is False


def test_discontinuity_is_a_true_cold_start_not_history_across_a_gap(tmp_path, monkeypatch):
    runner, config, cfg, source = _setup(tmp_path, monkeypatch, days=[1, 2, 3, 6, 7, 8, 9, 10])
    full = run_sequence_inference(runner, config, cfg, chunk_length=4, verbose=False)
    assert len(full) == 2
    source.loaded_days.clear()
    partial = run_sequence_inference(runner, config, cfg, date_start="2000-01-06",
        date_end="2000-01-10", chunk_length=4, verbose=False)
    assert len(partial) == 1
    np.testing.assert_array_equal(partial[0].predictions, full[1].predictions)
    assert min(source.loaded_days) == 6
    assert partial[0].run_id == full[1].run_id


def test_partial_period_budget_counts_outputs_and_loads_only_required_context(tmp_path, monkeypatch):
    runner, config, cfg, source = _setup(tmp_path, monkeypatch)
    result = run_sequence_inference(runner, config, cfg, date_start="2000-01-05",
        date_end="2000-01-10", limit_frames_per_run=2, chunk_length=4, verbose=False)[0]
    assert [date[2] for date in result.dates] == [5, 6]
    assert set(source.loaded_days) == {4, 5, 6}


@pytest.mark.parametrize("mode,offsets,expected", [
    ("legacy_nominal", (1,), [24, 24, 24, 24]),
    ("legacy_nominal", (1, 3), [72, 72, 72, 72]),
    ("selected_timestamps", (1, 3), [0, 24, 24, 72]),
    ("selected_timestamps", (3,), [0, 0, 0, 72]),
])
def test_cold_time_modes_are_explicit_and_derive_selected_span(mode, offsets, expected):
    ratio = torch.ones(1, 4)
    actual = [pair_time_scalars(ratio, t, history_offsets=offsets, cadence_days=1,
        batch_size=1, device=torch.device("cpu"), allow_cold_start=True,
        cold_start_time_mode=mode)[0].item() for t in range(4)]
    assert actual == expected


@pytest.mark.parametrize("mode", ["legacy_nominal", "selected_timestamps"])
def test_cold_time_modes_still_measure_real_available_intervals(mode):
    # Calendar gaps terminate inference runs. This primitive must nevertheless
    # use measured elapsed time whenever all required states are available.
    ratio = torch.tensor([[1., 1., 2., 1.]])
    actual, _ = pair_time_scalars(ratio, 3, history_offsets=(1, 3), cadence_days=1,
        batch_size=1, device=torch.device("cpu"), allow_cold_start=True,
        cold_start_time_mode=mode)
    assert actual.item() == 96
    with pytest.raises(IndexError, match="cannot measure the input interval"):
        pair_time_scalars(ratio, 0, history_offsets=(1, 3), cadence_days=1,
            batch_size=1, device=torch.device("cpu"), position_offset=5,
            allow_cold_start=True, cold_start_time_mode=mode)


def test_old_configuration_keeps_legacy_time_mode_and_new_mode_changes_signature():
    legacy = NativePairBackendConfig.parse({"history_offsets": [1]}, "native_pair")
    measured = NativePairBackendConfig.parse({"cold_start_time_mode": "selected_timestamps"}, "native_pair")
    assert legacy.cold_start_time_mode == "legacy_nominal"
    assert legacy.to_dict()["cold_start_time_mode"] == "legacy_nominal"
    assert legacy.architecture_signature() != measured.architecture_signature()
    with pytest.raises(TemporalConfigError):
        NativePairBackendConfig.parse({"cold_start_time_mode": "guess"}, "native_pair")
