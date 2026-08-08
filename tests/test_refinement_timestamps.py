"""Predictor/target timestamp-alignment guarantees.

This is a spatial downscaling and bias-correction problem: for every sample the
predictors and the target refer to the *same* date. These tests assert that the
refinement layer never introduces a target offset, a forecast-horizon index or a
temporal window, and that the diffusion/flow process times are internal
mathematical variables that leave data alignment untouched.
"""

from __future__ import annotations

import inspect
import io
import tokenize

import pytest
import torch

from granitewxc.refinement import backbones, cache, config, diffusion, flow_matching, two_phase
from granitewxc.refinement.two_phase import TwoPhaseDownscalingModel
from refinement_fixtures import TinyPhase1, make_batch
from test_refinement_models import REFINERS, build

REFINEMENT_MODULES = [backbones, cache, config, diffusion, flow_matching, two_phase]


def _code_only(source: str) -> str:
    kept = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        kept.append(token.string)
    return "\n".join(kept)


class TimestampedPhase1(TinyPhase1):
    """Records the timestamps it was given, to prove nothing shifts them."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen_timestamps: list = []

    def forward(self, batch, **kwargs):
        if "timestamp" in batch:
            self.seen_timestamps.append(batch["timestamp"])
        return super().forward(batch, **kwargs)


def timestamped_batch(n: int = 3, **kwargs):
    batch = make_batch(batch_size=n, **kwargs)
    # Predictor and target timestamps are one and the same object.
    stamps = torch.tensor([20160101 + i for i in range(n)], dtype=torch.int64)
    batch["timestamp"] = stamps
    batch["predictor_timestamp"] = stamps.clone()
    batch["target_timestamp"] = stamps.clone()
    return batch


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("refiner_type", ["none", *REFINERS])
def test_predictor_and_target_timestamps_are_identical(refiner_type):
    batch = timestamped_batch(height=16, width=16)
    model = build(refiner_type, batch) if refiner_type != "none" else build("none", batch)
    model.eval()
    model.predict(batch, ensemble_size=1, seed=0)
    assert torch.equal(batch["predictor_timestamp"], batch["target_timestamp"])


@pytest.mark.parametrize("refiner_type", REFINERS)
def test_refinement_does_not_shift_or_reorder_samples(refiner_type):
    batch = timestamped_batch(n=4, height=16, width=16)
    model = build(refiner_type, batch)
    model.eval()
    out = model.predict(batch, ensemble_size=2, seed=0)
    # Sample b of the output corresponds to sample b of the input: the
    # deterministic output for a permuted batch is the permuted output.
    perm = torch.tensor([2, 0, 3, 1])
    permuted = {k: (v[perm] if torch.is_tensor(v) and v.shape[:1] == (4,) else v)
                for k, v in batch.items()}
    out_perm = model.predict(permuted, ensemble_size=2, seed=0)
    assert torch.allclose(out.deterministic[perm], out_perm.deterministic, atol=1e-6)
    assert torch.equal(batch["timestamp"][perm], permuted["timestamp"])


def test_phase1_receives_the_unmodified_timestamps():
    batch = timestamped_batch(height=16, width=16)
    phase1 = TimestampedPhase1()
    model = TwoPhaseDownscalingModel(phase1)
    model.run_phase1(batch)
    assert len(phase1.seen_timestamps) == 1
    assert torch.equal(phase1.seen_timestamps[0], batch["timestamp"])


@pytest.mark.parametrize("refiner_type", REFINERS)
def test_process_time_does_not_touch_dataset_timestamps(refiner_type):
    batch = timestamped_batch(height=16, width=16)
    before = batch["timestamp"].clone()
    model = build(refiner_type, batch)
    model.train()
    out = model.training_step(batch, generator=torch.Generator().manual_seed(0))
    assert torch.equal(batch["timestamp"], before)
    assert out.residual_target.shape == batch["y"].shape


@pytest.mark.parametrize("refiner_type", REFINERS)
def test_process_time_is_generated_internally(refiner_type):
    """The refiner must never read a time value out of the batch."""
    batch = timestamped_batch(height=16, width=16)
    model = build(refiner_type, batch)
    model.train()
    generator = torch.Generator().manual_seed(0)
    a = model.training_step(batch, generator=generator)

    shifted = dict(batch)
    shifted["timestamp"] = batch["timestamp"] + 10_000
    generator_b = torch.Generator().manual_seed(0)
    b = model.training_step(shifted, generator=generator_b)
    # Same seed, same data -> identical objective, regardless of the timestamps.
    assert torch.allclose(a.losses["loss"], b.losses["loss"], atol=1e-6)


def test_conditioning_never_includes_the_target():
    """Ground truth must not leak into the Phase-2 conditioning."""
    batch = make_batch(height=16, width=16)
    model = build("diffusion_unet", batch)
    _, normalized, _ = model.run_phase1(batch)
    reference = model.build_conditioning(batch, normalized)

    altered = dict(batch)
    altered["y"] = batch["y"] * 3.0 + 7.0
    perturbed = model.build_conditioning(altered, normalized)
    assert torch.equal(reference, perturbed)


# ---------------------------------------------------------------------------
# Static guarantees
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module", REFINEMENT_MODULES, ids=lambda m: m.__name__)
def test_no_target_offset_or_forecast_index_in_source(module):
    source = _code_only(inspect.getsource(module)).lower()
    for needle in (
        "lead_time",
        "forecast_hour",
        "forecast_horizon",
        "target_offset",
        "time_offset",
        "n_input_timestamps",
        "rollout",
    ):
        assert needle not in source, f"{module.__name__} references {needle!r}"


def test_refinement_config_rejects_temporal_window_settings():
    from granitewxc.refinement.config import ConfigValidationError, resolve_refinement_config

    for bad in ({"temporal_window": 3}, {"n_input_timestamps": 2}, {"lead_time": 6}):
        with pytest.raises(ConfigValidationError):
            resolve_refinement_config({"refinement": {"type": "diffusion_unet", **bad}})
