"""Refinement compatibility: conditioning width, checkpoint guards, AR(1) noise.

The properties pinned here are the ones that would otherwise silently corrupt an
ensemble:

* ``temporal_conditioning: none`` must add **zero** conditioning channels, so every
  existing Phase-2 checkpoint stays loadable and the legacy path is byte-identical.
* Widening the conditioning must be **rejected**, not loaded partially.
* AR(1) noise must keep each frame's marginal exactly N(0,1) -- otherwise the
  refiner is evaluated under a noise distribution it was never trained on, which is
  a documented failure mode (arXiv:2608.02575).
* ``rho = 0`` must reproduce i.i.d. noise bit-for-bit.
* Each ensemble member must own its stream, so a member's trajectory does not
  depend on how many members shared a batch.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from granitewxc.temporal.config import TemporalConfigError, parse_temporal_config
from granitewxc.temporal.refinement import (
    AR1NoiseSource,
    RefinerTemporalCompatibilityError,
    SequenceRefinementPlan,
    TemporalConditioningAdapter,
    append_temporal_conditioning,
    check_refiner_temporal_compatibility,
    temporal_conditioning_channels,
)


def _cfg(**refinement):
    raw = {
        "enabled": True,
        "backend": "recurrent",
        "init_from_spatial_checkpoint": "x.ckpt",
    }
    if refinement:
        raw["refinement"] = refinement
    return parse_temporal_config(raw)


# ---------------------------------------------------------------------------
# conditioning width and checkpoint compatibility
# ---------------------------------------------------------------------------
def test_none_adds_zero_channels_so_existing_checkpoints_stay_valid():
    cfg = _cfg()
    assert cfg.refinement.temporal_conditioning == "none"
    assert temporal_conditioning_channels(cfg.refinement, time_feature_dim=5) == 0
    # Same cond_channels => the existing checkpoint is accepted.
    check_refiner_temporal_compatibility(
        checkpoint_cond_channels=20,
        required_cond_channels=20,
        temporal_conditioning="none",
    )


@pytest.mark.parametrize(
    "mode,expected",
    [("none", 0), ("time_features", 5), ("latent_state", 21)],
)
def test_conditioning_channel_counts(mode, expected):
    r = dataclasses.replace(_cfg().refinement, temporal_conditioning=mode)
    assert temporal_conditioning_channels(r, time_feature_dim=5) == expected


def test_widened_conditioning_rejects_existing_checkpoint():
    with pytest.raises(RefinerTemporalCompatibilityError) as exc:
        check_refiner_temporal_compatibility(
            checkpoint_cond_channels=20,
            required_cond_channels=25,
            temporal_conditioning="time_features",
            checkpoint_path="phase2.ckpt",
        )
    message = str(exc.value)
    # The error must name both counts and both valid remedies.
    assert "20" in message and "25" in message
    assert "temporal_conditioning='none'" in message
    assert "retrain" in message.lower()


def test_unknown_checkpoint_width_is_not_guessed():
    """A checkpoint that does not record cond_channels must not be rejected blindly."""
    check_refiner_temporal_compatibility(
        checkpoint_cond_channels=None,
        required_cond_channels=25,
        temporal_conditioning="time_features",
    )


def test_conditioning_adapter_shapes_and_broadcast():
    adapter = TemporalConditioningAdapter(
        latent_channels=64, time_feature_dim=5, projection_channels=16
    )
    time_features = torch.randn(2, 5)
    latent = torch.randn(2, 64, 4, 4)
    extra = adapter(
        time_features=time_features, latent=latent, size=(16, 16), dtype=torch.float32
    )
    assert extra.shape == (2, 5 + 16, 16, 16)
    # Time features carry no spatial structure, so their planes must be constant.
    planes = extra[:, :5]
    assert torch.allclose(planes, planes[..., :1, :1].expand_as(planes))

    cond = torch.randn(2, 20, 16, 16)
    merged = append_temporal_conditioning(cond, extra)
    assert merged.shape == (2, 41, 16, 16)
    assert torch.equal(merged[:, :20], cond)

    # None => untouched, same object.
    assert append_temporal_conditioning(cond, None) is cond


def test_conditioning_adapter_rejects_batch_mismatch():
    cond = torch.randn(2, 8, 4, 4)
    with pytest.raises(ValueError, match="does not match"):
        append_temporal_conditioning(cond, torch.randn(3, 4, 4, 4))


# ---------------------------------------------------------------------------
# noise
# ---------------------------------------------------------------------------
def _generators(n: int, seed: int = 1234):
    out = []
    for m in range(n):
        g = torch.Generator(device="cpu")
        g.manual_seed(seed + 1000003 * m)
        out.append(g)
    return out


def test_rho_zero_reproduces_iid_noise_exactly():
    """The legacy path must be bit-for-bit preserved when correlation is off."""
    shape = (4, 2, 8, 8)  # batch 1 x 4 members
    a = AR1NoiseSource(_generators(4), batch_size=1, rho=0.0)
    b = AR1NoiseSource(_generators(4), batch_size=1, rho=0.0)
    for _ in range(5):
        x = a.frame_noise(shape, "cpu", torch.float32)
        y = b.randn(shape, "cpu", torch.float32)
        assert torch.equal(x, y)
        a.advance()


def test_ar1_preserves_the_marginal_distribution():
    """Every frame must stay standard normal; only the correlation changes."""
    shape = (8, 1, 16, 16)
    for rho in (0.0, 0.5, 0.9):
        src = AR1NoiseSource(_generators(8), batch_size=1, rho=rho)
        frames = []
        for _ in range(400):
            frames.append(src.frame_noise(shape, "cpu", torch.float32).numpy())
            src.advance()
        arr = np.stack(frames)
        assert abs(arr.mean()) < 0.02, f"rho={rho} mean drifted: {arr.mean()}"
        assert abs(arr.std() - 1.0) < 0.03, f"rho={rho} std drifted: {arr.std()}"


def test_ar1_produces_the_requested_lag1_correlation():
    shape = (8, 1, 16, 16)
    for rho in (0.0, 0.5, 0.9):
        src = AR1NoiseSource(_generators(8), batch_size=1, rho=rho)
        frames = []
        for _ in range(600):
            frames.append(src.frame_noise(shape, "cpu", torch.float32).numpy().reshape(8, -1))
            src.advance()
        arr = np.stack(frames)  # [T, members, cells]
        lag1 = float(
            np.mean(
                [
                    np.corrcoef(arr[1:, m].ravel(), arr[:-1, m].ravel())[0, 1]
                    for m in range(8)
                ]
            )
        )
        assert abs(lag1 - rho) < 0.06, f"requested rho={rho}, measured {lag1}"


def test_members_own_independent_streams():
    """Member m's noise must not depend on how many members shared the batch."""
    shape_one = (1, 1, 4, 4)
    solo = AR1NoiseSource(_generators(1), batch_size=1, rho=0.7)
    solo_traj = [solo.frame_noise(shape_one, "cpu", torch.float32).clone() for _ in range(4)]

    shape_many = (5, 1, 4, 4)
    many = AR1NoiseSource(_generators(5), batch_size=1, rho=0.7)
    many_traj = [many.frame_noise(shape_many, "cpu", torch.float32).clone() for _ in range(4)]

    for t in range(4):
        assert torch.equal(solo_traj[t][0], many_traj[t][0]), (
            f"member 0 frame {t} changed when other members were added to the batch"
        )


def test_members_are_mutually_independent():
    shape = (6, 1, 12, 12)
    src = AR1NoiseSource(_generators(6), batch_size=1, rho=0.0)
    draws = np.stack(
        [src.frame_noise(shape, "cpu", torch.float32).numpy().reshape(6, -1) for _ in range(200)]
    )
    for m in range(1, 6):
        r = np.corrcoef(draws[:, 0].ravel(), draws[:, m].ravel())[0, 1]
        assert abs(r) < 0.06, f"member 0 and {m} correlate at {r}"


def test_reset_starts_a_fresh_sequence():
    shape = (2, 1, 4, 4)
    src = AR1NoiseSource(_generators(2), batch_size=1, rho=0.8)
    for _ in range(3):
        src.frame_noise(shape, "cpu", torch.float32)
    src.reset()
    assert src._state is None
    assert src._frame == 0


def test_bad_leading_dimension_is_rejected():
    src = AR1NoiseSource(_generators(3), batch_size=2, rho=0.0)
    with pytest.raises(RuntimeError, match="leading dimension"):
        src.frame_noise((5, 1, 4, 4), "cpu", torch.float32)
    # 2 batch * 3 members = 6 is correct
    assert src.frame_noise((6, 1, 4, 4), "cpu", torch.float32).shape == (6, 1, 4, 4)


def test_rho_must_be_in_range():
    with pytest.raises(ValueError, match="0 <= rho < 1"):
        AR1NoiseSource(_generators(1), batch_size=1, rho=1.0)
    with pytest.raises(ValueError, match="0 <= rho < 1"):
        AR1NoiseSource(_generators(1), batch_size=1, rho=-0.1)


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------
def test_plan_defaults_to_causal_per_date_refinement():
    plan = SequenceRefinementPlan.from_config(_cfg())
    assert plan.refine_whole_sequence is False
    assert plan.temporal_conditioning == "none"
    assert plan.noise_kind == "iid_per_frame"
    assert plan.noise_rho == 0.0
    assert plan.per_member_state is True


def test_plan_builds_a_noise_source_with_one_generator_per_member():
    cfg = _cfg(noise="ar1_correlated", noise_rho=0.6, ensemble_size=4)
    plan = SequenceRefinementPlan.from_config(cfg)
    src = plan.build_noise_source(batch_size=2, device="cpu", seed=99)
    assert len(src.generators) == 4
    assert src.rho == pytest.approx(0.6)
    assert src.batch_size == 2
    assert src.describe()["kind"] == "ar1"


def test_plan_ignores_rho_when_noise_is_iid():
    plan = SequenceRefinementPlan.from_config(_cfg())
    src = plan.build_noise_source(batch_size=1, device="cpu", seed=7)
    assert src.rho == 0.0
    assert src.describe()["kind"] == "iid_per_frame"


def test_config_rejects_identical_noise_every_date():
    """rho = 1 would manufacture persistence rather than model it."""
    with pytest.raises(TemporalConfigError, match="artificial persistence"):
        _cfg(noise="ar1_correlated", noise_rho=1.0)
