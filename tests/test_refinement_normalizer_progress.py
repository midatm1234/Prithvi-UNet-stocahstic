"""Progress-reporting regressions for the Phase-2 residual-statistics scan."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from refinement_fixtures import TinyPhase1, make_batch

import granitewxc.refinement.two_phase as two_phase
from granitewxc.refinement import build_two_phase_model


class _RecordingProgress:
    instances: list["_RecordingProgress"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.n = 0
        self.closed = False
        self.instances.append(self)

    def update(self, amount: int) -> None:
        self.n += amount

    def close(self) -> None:
        self.closed = True


class _TrackingLoader:
    def __init__(self, batches):
        self.batches = list(batches)
        self.consumed = 0

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        for batch in self.batches:
            self.consumed += 1
            yield batch


def _normalizing_model():
    return build_two_phase_model(
        TinyPhase1(),
        {
            "refinement": {
                "type": "flow_matching_unet",
                "seed": 3,
                "conditioning": {
                    "deterministic_output": True,
                    "input_predictors": True,
                    "static_fields": False,
                    "masks": False,
                },
                "residual_normalization": {"enabled": True, "epsilon": 1e-6},
                "unet": {
                    "hidden_channels": 4,
                    "num_levels": 1,
                    "time_embedding_dim": 8,
                    "bottleneck_attention": False,
                },
                "flow_matching": {"integration_steps": 2},
            }
        },
    )


@pytest.mark.parametrize(
    ("max_batches", "expected"),
    ((None, 3), (2, 2), (20, 3)),
)
def test_residual_scan_progress_has_exact_total_and_does_not_overconsume(
    monkeypatch, max_batches, expected
):
    _RecordingProgress.instances.clear()
    monkeypatch.setattr(two_phase, "tqdm", _RecordingProgress)
    loader = _TrackingLoader(
        [
            make_batch(
                batch_size=1,
                height=8,
                width=8,
                seed=seed,
                with_static=False,
            )
            for seed in range(3)
        ]
    )

    metadata = _normalizing_model().fit_residual_normalizer(
        loader,
        max_batches=max_batches,
        show_progress=True,
    )

    assert metadata["fitted"] is True
    assert loader.consumed == expected
    assert len(_RecordingProgress.instances) == 1
    progress = _RecordingProgress.instances[0]
    assert progress.kwargs["total"] == expected
    assert progress.kwargs["desc"] == "Residual-normalization scan"
    assert progress.kwargs["unit"] == "batch"
    assert progress.kwargs["mininterval"] == 1.0
    assert progress.n == expected
    assert progress.closed is True


def test_residual_scan_is_silent_by_default(monkeypatch):
    def unexpected_progress(*args, **kwargs):
        raise AssertionError(
            "default normalizer scan unexpectedly created a progress bar"
        )

    monkeypatch.setattr(two_phase, "tqdm", unexpected_progress)
    loader = _TrackingLoader(
        [make_batch(batch_size=1, height=8, width=8, with_static=False)]
    )
    metadata = _normalizing_model().fit_residual_normalizer(loader)
    assert metadata["fitted"] is True
    assert loader.consumed == 1


def test_residual_scan_progress_closes_when_iteration_fails(monkeypatch):
    class FailingLoader:
        def __len__(self):
            return 2

        def __iter__(self):
            yield make_batch(
                batch_size=1, height=8, width=8, with_static=False
            )
            raise RuntimeError("loader failed")

    _RecordingProgress.instances.clear()
    monkeypatch.setattr(two_phase, "tqdm", _RecordingProgress)
    with pytest.raises(RuntimeError, match="loader failed"):
        _normalizing_model().fit_residual_normalizer(
            FailingLoader(), show_progress=True
        )

    progress = _RecordingProgress.instances[0]
    assert progress.kwargs["total"] == 2
    assert progress.n == 1
    assert progress.closed is True


def test_narr_prism_train_command_enables_residual_scan_progress():
    script = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "NARR_PRISM"
        / "narr_prism_refinement.py"
    )
    tree = ast.parse(script.read_text(encoding="utf-8"))
    normalizer_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "fit_residual_normalizer"
    ]
    assert len(normalizer_calls) == 1
    progress_keywords = [
        keyword.value
        for keyword in normalizer_calls[0].keywords
        if keyword.arg == "show_progress"
    ]
    assert len(progress_keywords) == 1
    assert isinstance(progress_keywords[0], ast.Constant)
    assert progress_keywords[0].value is True
