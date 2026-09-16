"""Physical clean-state boundary supervision, including all four wrappers."""
from dataclasses import replace

import pytest
import torch

from granitewxc.refinement import resolve_refinement_config
from granitewxc.refinement.checkpoint import _build_scientific_contract
from granitewxc.refinement.clean_boundary_loss import physical_clean_boundary_loss
from granitewxc.refinement.config import ConfigValidationError
from granitewxc.refinement.two_phase import TwoPhaseDownscalingModel
from refinement_fixtures import TinyPhase1, make_batch


def test_physical_huber_uses_smooth_weights_and_actual_valid_denominator():
    pred = torch.ones(1, 2, 9, 11, requires_grad=True)
    target = torch.zeros_like(pred)
    valid = torch.ones_like(pred, dtype=torch.bool)
    valid[:, 0, 3, 4] = False
    with torch.no_grad():
        pred[:, 0, 3, 4] = float('nan')
    loss = physical_clean_boundary_loss(pred, target, valid, (0,), alpha=3., length=2., delta=1.)
    assert loss.item() == pytest.approx(.5)
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad[:, 1].count_nonzero() == 0
    assert pred.grad[:, 0, 3, 4].item() == 0
    row = pred.grad[0, 0, :5, 5]
    assert torch.all(row[:-1] > row[1:])
    # Missing cells do not create a new boundary or change distance weights.
    assert pred.grad[0, 0, 3, 5] == pred.grad[0, 0, 3, 6]


def test_all_invalid_is_zero_and_valid_nonfinite_is_rejected():
    p = torch.full((1, 2, 3, 5), float('nan'), requires_grad=True)
    loss = physical_clean_boundary_loss(p, p.detach(), torch.zeros_like(p, dtype=torch.bool), (0,), alpha=3., length=8., delta=1.)
    assert loss.item() == 0
    loss.backward()
    assert torch.isfinite(p.grad).all()
    with pytest.raises(FloatingPointError):
        physical_clean_boundary_loss(p, p.detach(), None, (0,), alpha=3., length=8., delta=1.)


@pytest.mark.parametrize('head', ['flow_matching_unet', 'flow_matching_transformer', 'diffusion_unet', 'diffusion_transformer'])
def test_shared_wrapper_adds_physical_loss_without_changing_native_or_tmax_terms(head):
    cfg = resolve_refinement_config({'refinement': {'type': head,
        'unet': {'hidden_channels': 8, 'num_levels': 2, 'time_embedding_dim': 16, 'bottleneck_attention': False},
        'transformer': {'embedding_dim': 24, 'num_heads': 4, 'num_blocks': 1, 'patch_size': [2, 2]},
        'diffusion': {'training_timesteps': 10, 'inference_steps': 2},
        'flow_matching': {'integration_steps': 2}}})
    model = TwoPhaseDownscalingModel(TinyPhase1(out_channels=2, scaling_codes=(1, 0), nonneg=(True, False)), refinement=cfg)
    batch = make_batch(out_channels=2, height=9, width=11)
    model.update_residual_statistics(batch)
    model.finalize_residual_statistics()
    before = model.training_step(batch, generator=torch.Generator().manual_seed(73)).losses
    model.refinement_config = replace(cfg, clean_boundary_channels=(0,), clean_boundary_weight=.025)
    after = model.training_step(batch, generator=torch.Generator().manual_seed(73)).losses
    assert torch.equal(before['process_loss'], after['process_loss'])
    assert torch.equal(before['prediction'], after['prediction'])
    clean = after['clean_residual_prediction']
    direct = torch.autograd.grad(after['loss'], clean, retain_graph=True)[0]
    native = torch.autograd.grad(before['loss'], before['clean_residual_prediction'], retain_graph=True)[0]
    assert torch.equal(direct[:, 1], native[:, 1])
    torch.testing.assert_close(after['loss'], before['loss'] + .025 * after['clean_boundary_loss'])
    after['loss'].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.refiner.parameters() if p.grad is not None)
    assert 'clean_boundary_loss' not in _build_scientific_contract(cfg)
    assert _build_scientific_contract(model.refinement_config)['clean_boundary_loss']['delta'] == 1.


@pytest.mark.parametrize('patch', [
    {'clean_boundary_channels': [0, 0]}, {'clean_boundary_channels': [True]},
    {'clean_boundary_length': 0}, {'clean_boundary_delta': 0},
    {'clean_boundary_weight': float('nan')}, {'clean_boundary_alpha': -1},
    {'clean_boundary_weight': .1},
    {'clean_boundary_weight': .1, 'clean_boundary_channels': [0], 'process_boundary_balance_channels': [0]},
])
def test_invalid_config_rejected(patch):
    with pytest.raises(ConfigValidationError):
        resolve_refinement_config({'refinement': {'type': 'flow_matching_unet', **patch}})
