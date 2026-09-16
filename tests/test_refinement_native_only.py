"""Selected native objectives must not change the retained variable's losses."""
from dataclasses import replace

import pytest
import torch

from granitewxc.refinement import build_refiner, resolve_refinement_config
from granitewxc.refinement.base import residual_reconstruction_terms
from granitewxc.refinement.checkpoint import _build_scientific_contract
from granitewxc.refinement.config import ConfigValidationError


def test_auxiliary_removal_retains_tmax_gradients_and_original_diagnostics():
    pred = torch.randn(2, 2, 9, 11, requires_grad=True)
    target = torch.randn_like(pred)
    mask = torch.ones_like(pred, dtype=torch.bool)
    mask[0, 0, :2] = False
    anchor = torch.randn_like(pred)
    gate = torch.tensor([1., .4]).reshape(1, 2, 1, 1).requires_grad_()
    before = residual_reconstruction_terms(pred, target, mask, anchor, lambda x: x * gate)
    after = residual_reconstruction_terms(pred, target, mask, anchor, lambda x: x * gate, excluded_channels=(0,))
    assert torch.equal(before['effective_clean_residual_prediction'], after['effective_clean_residual_prediction'])
    for key in ('reconstruction_loss', 'multiscale_loss', 'gradient_loss', 'mean_bias_loss'):
        a = torch.autograd.grad(before[key], pred, retain_graph=True)[0]
        b = torch.autograd.grad(after[key], pred, retain_graph=True)[0]
        assert torch.equal(a[:, 1], b[:, 1]), key
        assert b[:, 0].count_nonzero() == 0, key
    a = torch.autograd.grad(before['gate_calibration_loss'], gate, retain_graph=True)[0]
    b = torch.autograd.grad(after['gate_calibration_loss'], gate, retain_graph=True)[0]
    assert torch.equal(a[:, 1], b[:, 1])
    assert b[:, 0].item() == 0


@pytest.mark.parametrize('head', ['flow_matching_unet', 'flow_matching_transformer', 'diffusion_unet', 'diffusion_transformer'])
def test_native_only_keeps_process_prediction_targets_and_sampling(head):
    cfg = resolve_refinement_config({'refinement': {'type': head, 'unit_correction_gate_channels': [0],
        'unet': {'hidden_channels': 8, 'num_levels': 2, 'time_embedding_dim': 16, 'bottleneck_attention': False},
        'transformer': {'embedding_dim': 24, 'num_heads': 4, 'num_blocks': 1, 'patch_size': [2, 2]},
        'diffusion': {'training_timesteps': 10, 'inference_steps': 2},
        'flow_matching': {'integration_steps': 2, 'mean_path_loss_weight': .05}}})
    native_cfg = replace(cfg, native_only_channels=(0,))
    before = build_refiner(cfg, residual_channels=2, cond_channels=3)
    after = build_refiner(native_cfg, residual_channels=2, cond_channels=3)
    after.load_state_dict(before.state_dict())
    captures = [[], []]
    for model, capture in zip((before, after), captures):
        original = model.predict_process
        def tracked(*args, original=original, capture=capture, **kwargs):
            prediction = original(*args, **kwargs)
            capture.append(prediction)
            return prediction
        model.predict_process = tracked
    target, cond = torch.randn(2, 2, 9, 11), torch.randn(2, 3, 9, 11)
    a = before.training_loss(target, cond, generator=torch.Generator().manual_seed(19))
    b = after.training_loss(target, cond, generator=torch.Generator().manual_seed(19))
    assert torch.equal(a['prediction'], b['prediction'])
    assert torch.equal(a['process_loss'], b['process_loss'])
    assert torch.equal(a['clean_residual_prediction'], b['clean_residual_prediction'])
    assert torch.equal(a['effective_clean_residual_prediction'], b['effective_clean_residual_prediction'])
    native_grad = torch.autograd.grad(b['process_loss'], b['prediction'], retain_graph=True)[0]
    total_grad = torch.autograd.grad(b['loss'], b['prediction'], retain_graph=True)[0]
    assert torch.equal(native_grad[:, 0], total_grad[:, 0])
    old_grad = torch.autograd.grad(a['loss'], a['prediction'], retain_graph=True)[0]
    assert torch.equal(old_grad[:, 1], total_grad[:, 1])
    if cfg.is_flow_matching:
        aux_grad = torch.autograd.grad(b['mean_path_loss'], captures[1][1], retain_graph=True)[0]
        old_aux_grad = torch.autograd.grad(a['mean_path_loss'], captures[0][1], retain_graph=True)[0]
        assert aux_grad[:, 0].count_nonzero() == 0
        assert torch.equal(aux_grad[:, 1], old_aux_grad[:, 1])
    b['loss'].backward()
    with torch.inference_mode():
        a_sample = before.sample(cond, generator=torch.Generator().manual_seed(5))
        b_sample = after.sample(cond, generator=torch.Generator().manual_seed(5))
    assert torch.equal(a_sample, b_sample)
    assert 'native_only_channels' not in _build_scientific_contract(cfg)
    assert _build_scientific_contract(native_cfg)['native_only_channels'] == [0]


@pytest.mark.parametrize('patch', [
    {'native_only_channels': [True]}, {'native_only_channels': [0, 0]},
    {'native_only_channels': [-1]},
    {'native_only_channels': [0], 'clean_boundary_weight': .025, 'clean_boundary_channels': [0]},
])
def test_invalid_native_only_config(patch):
    with pytest.raises(ConfigValidationError):
        resolve_refinement_config({'refinement': {'type': 'flow_matching_unet', **patch}})


def test_native_only_transformer_requires_unit_gate_for_excluded_channel():
    with pytest.raises(ConfigValidationError, match='unit correction gates'):
        resolve_refinement_config({'refinement': {'type': 'flow_matching_transformer', 'native_only_channels': [0]}})
