"""Progress reflects actual microbatches, update budgets, and exact resumes."""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from granitewxc.temporal import training
from tests.test_temporal_training_takeover import scalar_setup


def assert_json_events(events):
    # JSONL progress consumers must not receive NaN or tensor objects.
    json.dumps(events, allow_nan=False)
    for event in events:
        assert 0 <= event["completed"] <= event["total"]
        assert event["epoch"] >= 1


@pytest.mark.parametrize("n,accum,budget,completed", [(7, 4, 2, 7), (11, 2, 2, 4)])
def test_epoch_events_track_microbatches_and_final_partial_update(
    monkeypatch, tmp_path, n, accum, budget, completed
):
    config, cfg, _ = scalar_setup(monkeypatch, n=n)
    config.gradient_accumulation_steps = accum
    events = []
    result = training.train_temporal_model(
        config, cfg, device="cpu", output_dir=tmp_path, max_steps_per_epoch=budget,
        verbose=False, progress_callback=events.append,
    )
    assert [event["event"] for event in events] == (
        ["epoch_start"] + ["batch"] * completed + ["validation", "epoch_end"]
    )
    assert events[0] == {"event": "epoch_start", "epoch": 1, "epochs": 1,
                         "total": completed, "completed": 0, "updates": 0, "loss": None}
    assert [e["completed"] for e in events if e["event"] == "batch"] == list(range(1, completed + 1))
    assert events[-2]["updates"] == events[-1]["updates"] == result["global_step"] == budget
    assert events[-1]["completed"] == events[-1]["total"] == completed
    assert events[-1]["loss"] == result["history"][0]["train"]["total"]
    assert events[-1]["val_loss"] == result["history"][0]["validation"]["total"]
    assert events[-1]["status"] == "complete"
    saved = torch.load(tmp_path / "last.ckpt", weights_only=False)
    assert saved["training_state"]["epoch_complete"] is True
    assert_json_events(events)


def test_skipped_update_expands_planned_total(monkeypatch, tmp_path):
    config, cfg, _ = scalar_setup(monkeypatch, n=8)
    def before(model, optimizer, state):
        calls = 0
        def gradient(grad):
            nonlocal calls
            calls += 1
            return torch.full_like(grad, float("inf")) if calls <= 4 else grad
        model.temporal_adapter.weight.register_hook(gradient)
    events = []
    result = training.train_temporal_model(
        config, cfg, device="cpu", output_dir=tmp_path, max_steps_per_epoch=1,
        verbose=False, before_training_callback=before, progress_callback=events.append,
    )
    batches = [e for e in events if e["event"] == "batch"]
    assert events[0]["total"] == 4
    assert [e["total"] for e in batches] == [4, 4, 4, 8, 8, 8, 8, 8]
    assert [e["updates"] for e in batches] == [0] * 7 + [1]
    assert events[-1]["completed"] == events[-1]["total"] == 8
    assert result["counters"]["skipped_nonfinite_updates"] == 1
    assert_json_events(events)


def test_nonfinite_running_loss_is_json_null(monkeypatch, tmp_path):
    config, cfg, _ = scalar_setup(monkeypatch, n=8)
    calls = 0
    def loss(prediction, batch):
        nonlocal calls
        calls += 1
        return prediction.mean() * float("nan") if calls == 1 else ((prediction - batch["y"]) ** 2).mean()
    monkeypatch.setattr(training, "build_loss_fn", lambda *args: loss)
    events = []
    training.train_temporal_model(
        config, cfg, device="cpu", output_dir=tmp_path, max_steps_per_epoch=1,
        verbose=False, progress_callback=events.append,
    )
    assert all(e["loss"] is None for e in events)
    assert events[-1]["val_loss"] is not None
    assert_json_events(events)


@pytest.mark.parametrize("interrupt_after", [2, 4])
def test_resume_progress_starts_at_saved_microbatch_position(monkeypatch, tmp_path, interrupt_after):
    config, cfg, _ = scalar_setup(monkeypatch, n=11, noisy=True)
    config.gradient_accumulation_steps = 2
    interrupted_events = []
    def interrupt(event):
        interrupted_events.append(event)
        if event["event"] == "batch" and event["updates"] == interrupt_after:
            raise RuntimeError("stop after saved update")
    with pytest.raises(RuntimeError, match="stop after saved update"):
        training.train_temporal_model(
            config, cfg, device="cpu", output_dir=tmp_path / "interrupted",
            max_steps_per_epoch=4, verbose=False, checkpoint_interval_updates=1,
            progress_callback=interrupt,
        )
    assert not any(e["event"] == "epoch_end" for e in interrupted_events)
    resume_path = tmp_path / "interrupted" / "last.ckpt"
    saved = torch.load(resume_path, weights_only=False)
    assert saved["global_step"] == interrupt_after
    assert saved["training_state"]["epoch_complete"] is False
    resumed_cfg = replace(cfg, init_from_spatial_checkpoint=None,
                          resume_from_temporal_checkpoint=str(resume_path))
    events = []
    result = training.train_temporal_model(
        config, resumed_cfg, device="cpu", output_dir=tmp_path / "resumed",
        max_steps_per_epoch=4, verbose=False, progress_callback=events.append,
    )
    assert events[0]["event"] == "epoch_start"
    assert events[0]["completed"] == interrupt_after * 2
    assert events[0]["updates"] == interrupt_after
    assert events[0]["total"] == 8
    assert [e["completed"] for e in events if e["event"] == "batch"] == list(range(interrupt_after * 2 + 1, 9))
    assert [e["event"] for e in events].count("epoch_start") == 1
    assert [e["event"] for e in events][-2:] == ["validation", "epoch_end"]
    assert result["global_step"] == 4
    assert_json_events(events)


def _assert_nested_equal(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            _assert_nested_equal(first, second)
    else:
        assert left == right


def test_progress_does_not_change_weights_optimizer_protocol_or_rng(monkeypatch, tmp_path):
    config, cfg, _ = scalar_setup(monkeypatch, n=7, noisy=True)
    events = []
    for name, callback in (("quiet", None), ("progress", events.append)):
        training.train_temporal_model(
            config, cfg, device="cpu", output_dir=tmp_path / name,
            max_steps_per_epoch=2, num_epochs=2, verbose=False, progress_callback=callback,
        )
    quiet = torch.load(tmp_path / "quiet" / "last.ckpt", weights_only=False)
    reported = torch.load(tmp_path / "progress" / "last.ckpt", weights_only=False)
    for key in ("model", "optimizer", "rng_state"):
        _assert_nested_equal(quiet[key], reported[key])
    _assert_nested_equal(quiet["training_state"]["protocol"], reported["training_state"]["protocol"])
    assert [(e["event"], e["epoch"]) for e in events if e["event"] in ("epoch_start", "epoch_end")] == [
        ("epoch_start", 1), ("epoch_end", 1), ("epoch_start", 2), ("epoch_end", 2)
    ]


@pytest.mark.parametrize("verbose,external_callback", [(True, False), (False, False), (True, True)])
def test_default_renderer_lifecycle_and_suppression(monkeypatch, tmp_path, verbose, external_callback):
    from granitewxc.temporal import progress
    config, cfg, _ = scalar_setup(monkeypatch, n=7)
    renderers = []
    class Renderer:
        def __init__(self):
            self.events, self.closed = [], []
            renderers.append(self)
        def __call__(self, event):
            self.events.append(event)
        def close(self, status="interrupted"):
            self.closed.append(status)
    monkeypatch.setattr(progress, "EpochProgress", Renderer)
    external = []
    training.train_temporal_model(
        config, cfg, device="cpu", output_dir=tmp_path, max_steps_per_epoch=1,
        num_epochs=2, verbose=verbose, progress_callback=external.append if external_callback else None,
    )
    if verbose and not external_callback:
        assert len(renderers) == 1
        assert renderers[0].closed == ["complete"]
        assert [e["epoch"] for e in renderers[0].events if e["event"] == "epoch_start"] == [1, 2]
        assert [e["epoch"] for e in renderers[0].events if e["event"] == "epoch_end"] == [1, 2]
    else:
        assert renderers == []
        assert bool(external) == external_callback


def test_default_renderer_closes_when_callback_raises(monkeypatch, tmp_path):
    from granitewxc.temporal import progress
    config, cfg, _ = scalar_setup(monkeypatch)
    closed = []
    class Renderer:
        def __call__(self, event):
            if event["event"] == "batch":
                raise RuntimeError("progress observer failed")
        def close(self, status="interrupted"):
            closed.append(status)
    monkeypatch.setattr(progress, "EpochProgress", Renderer)
    with pytest.raises(RuntimeError, match="progress observer failed"):
        training.train_temporal_model(
            config, cfg, device="cpu", output_dir=tmp_path, max_steps_per_epoch=1,
            verbose=True,
        )
    assert closed == ["interrupted"]
    assert not (tmp_path / "last.ckpt").exists()
