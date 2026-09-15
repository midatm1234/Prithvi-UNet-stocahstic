"""Stable sample streams exercise actual nonzero heads and complete reconstruction."""
import copy

import pytest
import torch

from granitewxc.refinement.randomness import StableNoiseSource, stable_randn, stable_seed
from granitewxc.refinement.two_phase import _slice_prediction_batch
from refinement_fixtures import make_batch
from test_refinement_models import REFINERS, build


@pytest.fixture(autouse=True)
def fixed_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_fields_are_keyed_by_sample_member_innovation_not_order_or_batch():
    ids = ["2096-01-01", "2097-06-01", "2099-12-31"]
    source = StableNoiseSource(71, ids, [0, 1, 2])
    first = source.randn((9, 3, 9, 11), "cpu", torch.float32).reshape(3, 3, 3, 9, 11)
    later = source.randn((9, 3, 9, 11), "cpu", torch.float32).reshape_as(first)
    reverse = StableNoiseSource(71, ids[::-1], [2, 0])
    paired = reverse.randn((6, 3, 9, 11), "cpu", torch.float32).reshape(3, 2, 3, 9, 11)
    for i in range(3):
        assert torch.equal(first[i, 2], paired[2-i, 0])
        assert torch.equal(first[i, 0], paired[2-i, 1])
        assert torch.equal(later[i, 1], stable_randn(
            (3, 9, 11), experiment_seed=71, sample_id=ids[i],
            member_index=1, innovation_index=1,
        ))
    assert not torch.equal(first, later)
    seeds = {stable_seed(e, i, m, k) for e in (1, 2) for i in ids for m in (0, 1) for k in (0, 1)}
    assert len(seeds) == 24


def test_common_offsets_and_crops_are_not_mistaken_for_batch_vectors():
    batch = make_batch(batch_size=2, height=9, width=11)
    batch["__output_scaler_offset"] = torch.tensor([4, 7])
    batch["__input_scaler_offset"] = torch.tensor([[1, 2], [3, 4]])
    batch["__output_crop"] = torch.tensor([0, 0, 9, 11])
    one = _slice_prediction_batch(batch, 1, 2)
    assert torch.equal(one["__output_scaler_offset"], torch.tensor([4, 7]))
    assert torch.equal(one["__input_scaler_offset"], torch.tensor([[3, 4]]))
    assert one["__output_crop"] == [(0, 0, 9, 11)]
    assert one["x"].shape[0] == 1


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
@pytest.mark.parametrize("head", REFINERS)
def test_complete_nonzero_head_prediction_is_exact_under_batch_order_and_member_chunk_changes(head, device):
    torch.manual_seed(329)
    batch = {k: v.to(device) for k, v in make_batch(batch_size=3, height=9, width=11).items()}
    model = build(head, batch=None, diffusion={
        "training_timesteps": 50, "inference_steps": 4, "eta": 0.6,
    }).to(device)
    model.initialize_from_batch(batch)
    # Nonzero actual networks: do not let zero-init outputs hide batch arithmetic.
    with torch.no_grad():
        for parameter in model.refiner.parameters():
            if parameter.requires_grad:
                parameter.add_(torch.randn_like(parameter) * 0.015)
    model.eval()
    ids = ["date-A", "date-B", "date-C"]
    traces = {}
    def capture(sample_id, stage, step, time, state, member):
        traces[(sample_id, member, stage, step)] = state.cpu().clone()
    before_rng = torch.random.get_rng_state().clone()
    reference = model.predict(batch, seed=87, sample_ids=ids, ensemble_size=3,
                              chunk_size=3, sample_trajectory_callback=capture)
    assert torch.equal(before_rng, torch.random.get_rng_state())
    assert not model.phase1.training
    assert not any(p.requires_grad for p in model.phase1.parameters())
    assert reference.member_residuals.abs().max() > 0
    order = [2, 0, 1]
    shuffled = {k: v[order] for k, v in batch.items()}
    shuffled["__sample_timestamp"] = [ids[i] for i in order]
    replay = model.predict(shuffled, seed=87, ensemble_size=3, chunk_size=1, sampling_mode="stable")
    for name in reference.as_dict():
        a, b = getattr(reference, name), getattr(replay, name)
        if torch.is_tensor(a):
            torch.testing.assert_close(a[order], b, rtol=0, atol=0, equal_nan=True)
    for index, sample_id in enumerate(ids):
        one = _slice_prediction_batch(batch, index, 3)
        got = model.predict(one, seed=87, sample_ids=[sample_id], ensemble_size=3, chunk_size=2)
        for name in reference.as_dict():
            a, b = getattr(reference, name), getattr(got, name)
            if torch.is_tensor(a):
                torch.testing.assert_close(a[index:index+1], b, rtol=0, atol=0, equal_nan=True)
    # Raw members are nested across requested ensemble sizes. Ensemble-aware
    # postprocessing may legitimately change processed members with M.
    larger = model.predict(batch, seed=87, sample_ids=ids, ensemble_size=4)
    torch.testing.assert_close(reference.member_residuals, larger.member_residuals[:, :3], rtol=0, atol=0)
    initial = traces[(ids[0], 0, "initial", 0)]
    expected = stable_randn((3, 9, 11), experiment_seed=87, sample_id=ids[0], member_index=0)
    assert torch.equal(initial[0, 0], expected)
    changed = model.predict(_slice_prediction_batch(batch, 0, 3), seed=88,
                            sample_ids=[ids[0]], ensemble_size=3)
    assert not torch.equal(reference.member_residuals[:1], changed.member_residuals)


@pytest.mark.parametrize("head", REFINERS)
def test_explicit_legacy_path_ignores_identifiers_and_matches_original_seeded_path(head):
    batch = make_batch(height=9, width=11)
    model = build(head, batch).eval()
    old = model._predict_impl(batch, seed=9, ensemble_size=3, chunk_size=2)
    default = model.predict(batch, seed=9, ensemble_size=3, chunk_size=2)
    tagged = dict(batch, __sample_id=["one", "two"])
    explicit = model.predict(tagged, seed=9, ensemble_size=3, chunk_size=2, sampling_mode="legacy")
    metadata_default = model.predict(tagged, seed=9, ensemble_size=3, chunk_size=2)
    for name in old.as_dict():
        a, b, c = getattr(old, name), getattr(default, name), getattr(explicit, name)
        if torch.is_tensor(a):
            torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)
            torch.testing.assert_close(a, c, rtol=0, atol=0, equal_nan=True)
            torch.testing.assert_close(a, getattr(metadata_default, name), rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("kwargs", [
    {"sample_ids": ["one", "two"]},
    {"sample_ids": ["one", "two"], "seed": 1, "generator": torch.Generator()},
    {"sample_ids": ["one"], "seed": 1},
    {"sample_ids": ["one", "one"], "seed": 1},
    {"sample_ids": ["one", "two"], "seed": 1, "sampling_mode": "wrong"},
])
def test_invalid_stable_contracts_fail_explicitly(kwargs):
    batch = make_batch(height=9, width=11)
    model = build("diffusion_unet", batch).eval()
    with pytest.raises(ValueError):
        model.predict(batch, **kwargs)


@pytest.mark.parametrize("head", REFINERS)
def test_existing_timestamp_batches_keep_stateful_generator_legacy_contract(head):
    batch = make_batch(height=9, width=11)
    model = build(head, batch).eval()
    legacy_generator = torch.Generator().manual_seed(13)
    tagged_generator = torch.Generator().manual_seed(13)
    old = model.predict(batch, generator=legacy_generator, ensemble_size=2)
    tagged = dict(batch, __sample_timestamp=["date1", "date2"])
    current = model.predict(tagged, generator=tagged_generator, ensemble_size=2)
    assert torch.equal(old.member_residuals, current.member_residuals)
    assert torch.equal(legacy_generator.get_state(), tagged_generator.get_state())
