"""Opt-in conditioning scaling uses frozen training statistics before alignment."""
from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from granitewxc.refinement import build_two_phase_model
from granitewxc.refinement.checkpoint import build_refinement_checkpoint, load_refinement_state_dict
from granitewxc.refinement.config import ConfigValidationError, resolve_refinement_config
from refinement_fixtures import TinyPhase1, make_batch
from test_refinement_models import REFINERS, small_config


class ScaledPhase1(TinyPhase1):
    def __init__(self, timestamps=2):
        super().__init__()
        self.n_input_timestamps = timestamps
        channels = 4 // timestamps
        ramp = torch.arange(8 * 9, dtype=torch.float32).reshape(1, 1, 8, 9)
        self.input_scalers_mu = nn.Parameter(ramp.repeat(1, channels, 1, 1), requires_grad=False)
        self.input_scalers_sigma = nn.Parameter(torch.arange(1., channels+1).reshape(1, channels, 1, 1).expand(1, channels, 8, 9).clone(), requires_grad=False)
        self.input_scalers_epsilon = .125
        self.static_input_scalers_mu = nn.Parameter(torch.full((1, 1, 1, 1), 10.), requires_grad=False)
        self.static_input_scalers_sigma = nn.Parameter(torch.full((1, 1, 1, 1), 2.), requires_grad=False)
        self.static_output_scalers_mu = nn.Parameter(torch.full((1, 1, 1, 1), 30.), requires_grad=False)
        self.static_output_scalers_sigma = nn.Parameter(torch.full((1, 1, 1, 1), 4.), requires_grad=False)
        self.static_input_scalers_epsilon = .25
        self.last_input_offset = None

    def _resolve_input_scalers(self, reference, scaler_offset=None):
        from granitewxc.models.cordex_finetune_model import _resolve_spatial_scalers
        self.last_input_offset = scaler_offset
        return _resolve_spatial_scalers('input', self.input_scalers_mu.to(reference), self.input_scalers_sigma.to(reference), reference, scaler_offset=scaler_offset)


def build_scaled(head='flow_matching_unet', normalize=True, timestamps=2):
    config = small_config(head, conditioning={'normalize_predictors': normalize})
    return build_two_phase_model(ScaledPhase1(timestamps=timestamps), config)


@pytest.mark.parametrize('timestamps', [1, 2])
def test_scaled_conditioning_matches_frozen_statistics_offsets_and_crops(timestamps):
    wrapper = build_scaled(timestamps=timestamps)
    phase1 = wrapper.phase1
    batch = make_batch(batch_size=2, height=4, width=5)
    offsets = torch.tensor([[1, 2], [2, 3]])
    batch['__input_scaler_offset'] = offsets
    batch['__scaler_offset'] = torch.tensor([0, 0])  # specific input offset must win
    batch['__output_crop'] = torch.tensor([[1, 1, 2, 3], [1, 1, 2, 3]])
    desired = torch.arange(1., 5.).reshape(1, 4, 1, 1).expand(2, 4, 4, 5)
    means = torch.cat([phase1.input_scalers_mu[..., top:top+4, left:left+5] for top, left in offsets])
    scales = torch.cat([phase1.input_scalers_sigma[..., top:top+4, left:left+5] for top, left in offsets])
    means = means.repeat(1, timestamps, 1, 1)
    scales = scales.repeat(1, timestamps, 1, 1)
    batch['x'] = desired * (scales + phase1.input_scalers_epsilon) + means
    batch['static_x'].fill_(10. + 5. * 2.25)
    batch['static_y'].fill_(30. + 7. * 4.25)
    baseline = torch.zeros(2, 3, 2, 3)
    stats_before = {key: value.clone() for key, value in phase1.state_dict().items() if 'scalers' in key}
    cond = wrapper.build_conditioning(batch, baseline)
    assert cond.shape == (2, 10, 2, 3)
    torch.testing.assert_close(cond[:, 3:7], desired[..., 1:3, 1:4])
    torch.testing.assert_close(cond[:, 7:8], torch.full((2, 1, 2, 3), 5.))
    torch.testing.assert_close(cond[:, 8:9], torch.full((2, 1, 2, 3), 7.))
    assert torch.equal(phase1.last_input_offset, offsets)
    assert all(torch.equal(value, phase1.state_dict()[key]) for key, value in stats_before.items())


def test_opt_in_preserves_missing_predictor_mask_and_finite_conditioning():
    wrapper = build_scaled()
    batch = make_batch(batch_size=1, height=4, width=5)
    batch['__input_scaler_offset'] = torch.tensor([1, 2])
    batch['x'][0, 0, 1, 2] = float('nan')
    batch['static_x'][0, 0, 2, 2] = float('inf')
    cond = wrapper.build_conditioning(batch, torch.zeros_like(batch['y']))
    assert torch.isfinite(cond).all()
    assert cond[0, 3, 1, 2] == 0.
    assert cond[0, -1, 1, 2] == 0.
    assert cond[0, 7, 2, 2] == 0.


def test_disabled_scaling_is_exact_legacy_raw_conditioning():
    wrapper = build_scaled(normalize=False)
    batch = make_batch(batch_size=1, height=4, width=5)
    baseline = torch.zeros_like(batch['y'])
    cond = wrapper.build_conditioning(batch, baseline)
    expected = torch.cat((baseline, batch['x'], batch['static_x'], batch['static_y'], torch.ones(1, 1, 4, 5)), dim=1)
    assert torch.equal(cond, expected)
    assert wrapper.phase1.last_input_offset is None


@pytest.mark.parametrize('head', REFINERS)
def test_scaling_is_a_retraining_contract_for_all_heads(head):
    source = build_scaled(head, normalize=False)
    source.initialize_refiner(10)
    payload = copy.deepcopy(build_refinement_checkpoint(source))
    assert 'normalize_predictors' not in payload['refinement_contract']['conditioning']
    target = build_scaled(head, normalize=True)
    target.initialize_refiner(10)
    with pytest.raises(RuntimeError, match='scientific/config contract mismatch.*normalize_predictors'):
        load_refinement_state_dict(target, payload)
    normalized_payload = copy.deepcopy(build_refinement_checkpoint(target))
    assert normalized_payload['refinement_contract']['conditioning']['normalize_predictors'] is True
    target.phase1.input_scalers_mu.data.add_(1.)
    with pytest.raises(RuntimeError, match='target-space contract mismatch.*conditioning_normalization'):
        load_refinement_state_dict(target, normalized_payload)


def test_scaling_refuses_to_substitute_missing_training_statistics():
    config = small_config('flow_matching_unet', conditioning={'normalize_predictors': True})
    wrapper = build_two_phase_model(TinyPhase1(), config)
    batch = make_batch(height=4, width=5)
    with pytest.raises(RuntimeError, match='statistics cannot be fitted or substituted'):
        wrapper.build_conditioning(batch, torch.zeros_like(batch['y']))


def test_scaling_flag_does_not_count_as_a_conditioning_source():
    with pytest.raises(ConfigValidationError, match='disables every input'):
        resolve_refinement_config({'refinement': {'type': 'flow_matching_unet', 'conditioning': {
            'deterministic_output': False, 'input_predictors': False, 'static_fields': False,
            'masks': False, 'prithvi_features': False, 'unet_features': False, 'normalize_predictors': True,
        }}})
