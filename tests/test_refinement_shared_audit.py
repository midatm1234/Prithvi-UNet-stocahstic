"""Numeric and semantic regressions from the four-head refinement audit."""
from __future__ import annotations

import copy

import pytest
import torch

from granitewxc.refinement.base import masked_loss, residual_structure_losses
from granitewxc.refinement.checkpoint import build_refinement_checkpoint, load_refinement_state_dict
from granitewxc.refinement.config import ResidualNormalizationConfig
from granitewxc.refinement.normalization import ResidualNormalizer
from granitewxc.refinement.target_space import NormalizedTargetSpace
from granitewxc.refinement.training import RefinementTrainer
from refinement_fixtures import TinyPhase1, make_batch
from test_refinement_models import REFINERS, build


@pytest.mark.parametrize('kind', ['mse', 'l1', 'huber'])
def test_masked_nonfinite_operands_have_zero_loss_and_gradient(kind):
    prediction = torch.tensor([[[[2., float('nan'), float('inf')]]]], requires_grad=True)
    target = torch.tensor([[[[1., float('inf'), float('nan')]]]])
    mask = torch.tensor([[[[True, False, False]]]])
    loss = masked_loss(prediction, target, mask, kind)
    expected = .5 if kind == 'huber' else 1.
    assert loss.item() == expected
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad[..., 1:]) == 0


def test_fractional_weights_reduce_by_weight_sum_and_equal_channel_means():
    prediction = torch.tensor([[[[2., 4.]], [[3., 99.]]]], requires_grad=True)
    target = torch.zeros_like(prediction)
    weights = torch.tensor([[[[.01, .03]], [[.02, 0.]]]])
    loss = masked_loss(prediction, target, weights)
    assert loss.item() == pytest.approx(((.01 * 4 + .03 * 16) / .04 + 9) / 2)
    loss.backward()
    assert prediction.grad[0, 1, 0, 1] == 0


@pytest.mark.parametrize('weight', [-1., float('nan'), float('inf')])
def test_invalid_loss_weights_are_rejected(weight):
    with pytest.raises(ValueError, match='finite and non-negative'):
        masked_loss(torch.ones(1, 1, 1, 1), torch.zeros(1, 1, 1, 1), torch.tensor(weight))


def test_fp16_difference_is_computed_in_float32():
    prediction = torch.full((1, 1, 1, 1), 60000., dtype=torch.float16)
    target = -prediction
    assert masked_loss(prediction, target, None, 'l1').item() == 120000.


def test_structure_losses_ignore_nonfinite_cells_and_broadcast_masks():
    target = torch.arange(48.).reshape(1, 3, 4, 4)
    prediction = target.clone().requires_grad_()
    mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    mask[..., 0, :] = False
    target[..., 0, :] = float('nan')
    with torch.no_grad():
        prediction[..., 0, :] = float('inf')
    losses = residual_structure_losses(prediction, target, mask)
    assert all(loss.item() == 0. for loss in losses.values())
    sum(losses.values()).backward()
    assert torch.isfinite(prediction.grad).all()


def test_small_residual_scales_remain_finite_under_fp16():
    normalizer = ResidualNormalizer(1, ResidualNormalizationConfig(minimum_scale=1e-12))
    normalizer.scale.fill_(1e-8)
    normalizer.fitted.fill_(True)
    normalized = normalizer.normalize(torch.ones(1, 1, 2, 2, dtype=torch.float16))
    assert normalized.dtype == torch.float32
    assert torch.isfinite(normalized).all()
    restored = normalizer.denormalize(normalized)
    torch.testing.assert_close(restored, torch.ones_like(restored))
    with pytest.raises(ValueError, match='expects'):
        normalizer.normalize(torch.ones(1, 2, 2, 2))


def test_log_precipitation_transform_uses_float32_and_keeps_signed_corrections():
    phase1 = TinyPhase1(out_channels=2, scaling_codes=(2, 0), nonneg=(True, False))
    phase1.output_scalers_mu.data.zero_()
    phase1.output_scalers_sigma.data.fill_(1.)
    space = NormalizedTargetSpace(phase1)
    normalized = torch.tensor([12., -3.], dtype=torch.float16).reshape(1, 2, 1, 1)
    physical = space.decode(normalized)
    assert physical.dtype == torch.float32
    assert torch.isfinite(physical).all()
    assert physical[0, 1, 0, 0] == -3.
    torch.testing.assert_close(space.encode(physical), normalized.float())
    baseline = torch.tensor([5., 300.]).reshape(1, 2, 1, 1)
    truth = torch.tensor([0., 298.]).reshape(1, 2, 1, 1)
    residual, valid = space.physical_residual_target(truth, baseline)
    assert valid.all()  # real dry-day zero remains observed
    assert (residual < 0).all()
    torch.testing.assert_close(space.reconstruct_physical(baseline, residual), truth)


@pytest.mark.parametrize('refiner_type', REFINERS)
def test_all_heads_update_with_three_variables_while_phase1_stays_frozen(refiner_type, tmp_path):
    batch = make_batch(batch_size=1, height=7, width=9, nan_fraction=.1)
    model = build(refiner_type, batch)
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=.001)
    trainer = RefinementTrainer(model, optimizer, checkpoint_dir=str(tmp_path))
    before_phase1 = {key: value.clone() for key, value in model.phase1.state_dict().items()}
    before_head = {key: value.clone() for key, value in model.refiner.named_parameters()}
    model.phase1.train()  # wrapper must repair accidental external mode changes
    loss = trainer.train_one_epoch([batch], show_progress=False)
    assert torch.isfinite(torch.tensor(loss))
    assert not model.phase1.training
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in model.phase1.parameters())
    assert all(torch.equal(before_phase1[key], value) for key, value in model.phase1.state_dict().items())
    assert any(not torch.equal(before_head[key], value) for key, value in model.refiner.named_parameters())
    model.eval()
    output = model.predict(batch, ensemble_size=2, seed=37)
    assert output.members.shape == (1, 2, 3, 7, 9)
    assert torch.isfinite(output.members).all()


def test_optimizer_missing_refinement_parameters_is_rejected(tmp_path):
    model = build('flow_matching_unet', make_batch(height=7, width=9))
    parameters = model.trainable_parameters()
    optimizer = torch.optim.Adam(parameters[:-1], lr=.001)
    with pytest.raises(ValueError, match='every trainable refinement parameter'):
        RefinementTrainer(model, optimizer, checkpoint_dir=str(tmp_path))


def test_checkpoint_rejects_same_shape_variable_order_and_transform_changes():
    batch = make_batch(height=7, width=9)
    source = build('flow_matching_unet', batch)
    source.phase1.output_var_names = ['pr', 'tasmax', 'wind']
    payload = copy.deepcopy(build_refinement_checkpoint(source))
    target = build('flow_matching_unet', batch)
    target.phase1.output_var_names = ['tasmax', 'pr', 'wind']
    with pytest.raises(RuntimeError, match='target-space contract mismatch.*channel_names'):
        load_refinement_state_dict(target, payload)
    target.phase1.output_var_names = source.phase1.output_var_names
    target.phase1.predictand_scaling_method_codes[0] = 2
    with pytest.raises(RuntimeError, match='target-space contract mismatch.*scaling_method'):
        load_refinement_state_dict(target, payload)


def test_legacy_checkpoint_without_target_contract_warns_explicitly():
    batch = make_batch(height=7, width=9)
    model = build('flow_matching_unet', batch)
    payload = copy.deepcopy(build_refinement_checkpoint(model))
    payload.pop('target_space_contract')
    payload.pop('target_space_contract_fingerprint')
    with pytest.warns(RuntimeWarning, match='channel ordering'):
        load_refinement_state_dict(model, payload)


@pytest.mark.parametrize('refiner_type', REFINERS)
def test_exact_cached_phase1_is_reused_for_training_and_inference(refiner_type, monkeypatch):
    batch = make_batch(batch_size=1, height=7, width=9)
    model = build(refiner_type, batch, residual_normalization={'method': 'standardize'})
    physical, normalized, _ = model.run_phase1(batch)
    physical = physical.clone()
    physical[:, 0] = 0.  # amount-normalized field cannot represent a dry hurdle decision
    model.phase1.precip_hurdle_enabled = True
    batch['__phase1_normalized'] = normalized
    batch['__phase1_physical'] = physical

    def no_phase1_forward(*args, **kwargs):
        raise AssertionError('Exact shared Phase-1 cache was not reused')

    monkeypatch.setattr(model.phase1, 'forward', no_phase1_forward)
    model.initialize_from_batch(batch)
    model.update_residual_statistics(batch)
    model.finalize_residual_statistics()
    training = model.training_step(batch, generator=torch.Generator().manual_seed(7))
    torch.testing.assert_close(training.residual_target_physical, batch['y'] - physical)
    model.eval()
    output = model.predict(batch, ensemble_size=2, seed=7)
    assert torch.equal(output.deterministic, physical)
    batch.pop('__phase1_physical')
    with pytest.raises(RuntimeError, match='wet-occurrence decision'):
        model.predict(batch, ensemble_size=2, seed=7)


@pytest.mark.parametrize('head', REFINERS)
def test_coordinate_convention_requires_compatible_refinement_checkpoint(head):
    batch = make_batch(height=7, width=9)
    model = build(head, batch)
    payload = copy.deepcopy(build_refinement_checkpoint(model))
    assert 'spatial_alignment' not in payload['refinement_contract']['network']['config']
    source = model.refinement_config.to_dict()
    architecture = 'transformer' if head.endswith('transformer') else 'unet'
    source[architecture]['spatial_alignment'] = 'coordinates'
    target = build(head, batch, **{architecture: source[architecture]})
    with pytest.raises(RuntimeError, match='scientific/config contract mismatch.*spatial_alignment'):
        load_refinement_state_dict(target, payload)


def test_cache_round_trip_preserves_physical_baseline_and_rejects_nonfinite_corruption(tmp_path):
    from granitewxc.refinement.cache import Phase1ConditioningCache
    from granitewxc.refinement.config import Phase1CachePerfConfig
    from test_refinement_performance import _cache_key

    batch = make_batch(batch_size=1, height=7, width=9)
    model = build('flow_matching_unet', batch)
    physical, normalized, _ = model.run_phase1(batch)
    cache = Phase1ConditioningCache(tmp_path, _cache_key(), Phase1CachePerfConfig(enabled=True, validate_samples=1))
    cache.open_for_write()
    cache.write('sample', timestamp=20000101, deterministic_normalized=normalized, deterministic_physical=physical)
    restored = cache.inject(dict(batch), cache.read('sample'))
    assert torch.equal(restored['__phase1_physical'], physical)
    assert cache.validate(model, [('sample', batch)])['max_abs'] == 0.
    corrupt = normalized.clone()
    corrupt[..., 0, 0] = float('nan')
    cache.write('sample', timestamp=20000101, deterministic_normalized=corrupt, deterministic_physical=physical)
    with pytest.raises(RuntimeError, match='incompatible finite mask'):
        cache.validate(model, [('sample', batch)])


@pytest.mark.parametrize('head', REFINERS)
def test_predict_trajectory_callback_preserves_members_and_chunk_order(head):
    batch = make_batch(batch_size=2, height=7, width=9)
    model = build(head, batch).eval()
    reference = model.predict(batch, ensemble_size=3, seed=91, chunk_size=2)
    events = []

    def capture(stage, step, process_time, state, member_start):
        events.append((stage, step, process_time, state.clone(), member_start))
        state.fill_(9876.)  # callbacks receive snapshots and cannot alter integration

    diagnosed = model.predict(batch, ensemble_size=3, seed=91, chunk_size=2, trajectory_callback=capture)
    assert torch.equal(reference.members, diagnosed.members)
    assert torch.equal(reference.member_residuals, diagnosed.member_residuals)
    initial = [event for event in events if event[0] == 'initial']
    assert [event[4] for event in initial] == [0, 2]
    assert [event[3].shape for event in initial] == [(2, 2, 3, 7, 9), (2, 1, 3, 7, 9)]
    final = [event for event in events if event[0] == 'final_normalized_residual']
    assert len(final) == 2
    joined = torch.cat([event[3] for event in final], dim=1)
    torch.testing.assert_close(joined, diagnosed.member_residuals)
