"""Controls for paired audit metrics, input contracts, and loaded provenance."""
from pathlib import Path
import os
import runpy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from examples.CORDEX_ML.cordex_refinement_audit import input_contract, regions, summarize


def test_constant_and_ramp_climatology_metrics_include_signed_edges():
    yy, xx = np.indices((9, 11))
    truth = np.broadcast_to(yy + 2 * xx, (3, 9, 11)).astype(float)
    mask = regions(9, 11)
    prediction = truth.copy()
    prediction[:, :, -1] += 2
    prediction[:, :, 0] -= 2
    whole = summarize(prediction, truth, mask['domain'])
    assert whole['bias'] == pytest.approx(0)
    assert whole['climatology_mae'] == pytest.approx(4 / 11)
    assert summarize(prediction, truth, mask['interior_1'])['daily_rmse'] == 0
    assert summarize(prediction, truth, mask['column_end_1'])['bias'] == 2
    assert summarize(prediction, truth, mask['column_start_1'])['bias'] == -2


def test_missing_values_are_excluded_but_dry_zeros_are_retained():
    truth = np.zeros((2, 3, 4))
    truth[0, 0, 0] = np.nan
    prediction = np.ones_like(truth)
    metrics = summarize(prediction, truth, np.ones((3, 4), bool))
    assert metrics['valid_count'] == 23
    assert metrics['daily_mae'] == 1


def test_input_contract_ignores_head_but_rejects_order_and_conditioning_changes():
    import copy
    raw = dict(data=dict(output_vars=['pr', 'tasmax']), predictands={},
               model=dict(embed_dim=32, refinement=dict(type='diffusion_unet', conditioning=dict(masks=True))))
    other = copy.deepcopy(raw)
    other['model']['refinement']['type'] = 'flow_matching_transformer'
    assert input_contract(raw) == input_contract(other)
    other['data']['output_vars'].reverse()
    assert input_contract(raw) != input_contract(other)


@pytest.fixture(scope='module')
def provenance_helper():
    path = Path(__file__).resolve().parents[1] / 'examples/CORDEX_ML/notebooks/SA_downscaling_inference_T2_ACCESS-CM2_static.py'
    cwd = Path.cwd()
    try:
        return runpy.run_path(str(path))['_loaded_normalization_provenance']
    finally:
        os.chdir(cwd)


def test_provenance_requires_loaded_fitted_and_identical_statistics(provenance_helper):
    import json
    from granitewxc.refinement.checkpoint import _state_fingerprint
    state = {'residual_normalizer.mean': torch.tensor([2.0])}
    model = SimpleNamespace(_last_checkpoint_loaded=False)
    with pytest.raises(RuntimeError, match='before checkpoint'):
        provenance_helper(model)
    model._last_checkpoint_loaded = True
    model.residual_normalization_metadata = lambda: {'fitted': False}
    with pytest.raises(RuntimeError, match='fitted'):
        provenance_helper(model)
    model.residual_normalization_metadata = lambda: {'fitted': True, 'mean': [2.0]}
    model.state_dict = lambda: state
    model._last_checkpoint_payload = {'residual_normalizer_state_fingerprint': _state_fingerprint(state)}
    result = provenance_helper(model)
    assert json.loads(result['residual_normalization'])['mean'] == [2.0]
    state['residual_normalizer.mean'].add_(1)
    with pytest.raises(RuntimeError, match='differ'):
        provenance_helper(model)


def test_sa_notebook_uses_disjoint_training_cli_loader():
    import json
    notebook = Path(__file__).resolve().parents[1] / 'examples/CORDEX_ML/notebooks/SA_downscaling_refinement_T2_ACCESS-CM2_static.ipynb'
    payload = json.loads(notebook.read_text(encoding='utf8'))
    source = '\n'.join(''.join(cell.get('source', [])) for cell in payload['cells'] if cell['cell_type'] == 'code')
    assert 'from cordex_refinement_training import _build_loaders' in source
    assert 'train_dl, val_dl, validation_source = _build_loaders(' in source
    assert 'build_cordex_dataloaders(config, USE_GPU)' not in source


@pytest.mark.parametrize("head", ("diffusion_unet", "diffusion_transformer", "flow_matching_unet", "flow_matching_transformer"))
def test_normal_sa_driver_records_exact_sampler_and_physical_stages(head, provenance_helper, monkeypatch):
    from granitewxc.refinement.config import resolve_refinement_config
    from granitewxc.refinement.two_phase import TwoPhaseDownscalingModel
    from tests.refinement_fixtures import TinyPhase1
    phase1 = TinyPhase1(out_channels=2, scaling_codes=(1, 0), nonneg=(True, False))
    refinement = resolve_refinement_config({'refinement': {
        'enabled': True, 'type': head, 'ensemble_size': 3, 'seed': 41,
        'diffusion': {'training_timesteps': 4, 'inference_steps': 2},
        'flow_matching': {'integration_steps': 2},
        'unet': {'hidden_channels': 8, 'num_levels': 1, 'time_embedding_dim': 16},
        'transformer': {'embedding_dim': 16, 'num_heads': 4, 'num_blocks': 1, 'patch_size': 2},
    }})
    model = TwoPhaseDownscalingModel(phase1, refinement)
    batch = {'x': torch.ones(2, 4, 5, 7), 'y': torch.zeros(2, 2, 5, 7)}
    model.initialize_from_batch(batch)
    model.update_residual_statistics(batch)
    model.finalize_residual_statistics()
    model._last_checkpoint_loaded = True
    model.eval()
    namespace = provenance_helper.__globals__
    monkeypatch.setitem(namespace, '_normalize_predictors_for_diagnostics', lambda batch, model: batch['x'])
    monkeypatch.setitem(namespace, 'REFINEMENT_ENSEMBLE_SIZE', 3)
    function = namespace['_run_full_inference']
    monkeypatch.setitem(namespace, 'SAVE_SAMPLING_STATES', False)
    plain = function([batch], model, torch.device('cpu'), ['pr', 'tasmax'], SimpleNamespace(enabled=False))
    monkeypatch.setitem(namespace, 'SAVE_SAMPLING_STATES', True)
    traced = function([batch], model, torch.device('cpu'), ['pr', 'tasmax'], SimpleNamespace(enabled=False))
    torch.testing.assert_close(plain['ensemble_member_predictions'], traced['ensemble_member_predictions'], rtol=0, atol=0)
    states = traced['sampling_states']
    assert 'initial_0000' in states
    assert any(name.startswith('final_normalized_residual_') for name in states)
    assert states['postprocessed_physical'].shape == (3, 2, 5, 7)
    assert states['reconstructed_unbounded_physical'].shape == (3, 2, 5, 7)
    np.testing.assert_array_equal(states['postprocessed_physical'], traced['ensemble_member_predictions'][0].numpy())
