"""Scientific CLI dispatch and date identities survive the actual SA driver."""
import os
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import torch

from granitewxc.refinement.scientific_entrypoints import dispatch_scientific
from refinement_fixtures import make_batch
from test_refinement_models import REFINERS, build

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / 'examples/CORDEX_ML/notebooks/SA_downscaling_inference_T2_ACCESS-CM2_static.py'


@pytest.fixture(scope='module')
def driver():
    previous = Path.cwd()
    try:
        namespace = runpy.run_path(str(DRIVER))
    finally:
        os.chdir(previous)
    return namespace


def test_explicit_dispatch_preserves_legacy_and_operation(monkeypatch):
    import granitewxc.refinement.scientific_entrypoints as bridge
    calls = []
    monkeypatch.setattr(bridge, 'import_module', lambda name: SimpleNamespace(
        main=lambda args: calls.append((name, args)) or 7))
    assert dispatch_scientific(['--config', 'old.yaml']) is None
    assert dispatch_scientific(['--scientific-experiment', 'train', '--cache', 'a.h5'],
                               expected_operation='train') == 7
    assert calls == [('examples.CORDEX_ML.cordex_scientific_experiments',
                      ['train', '--cache', 'a.h5'])]
    with pytest.raises(ValueError, match='requires scientific operation'):
        dispatch_scientific(['--scientific-experiment', 'predict'], expected_operation='train')
    with pytest.raises(ValueError, match='requires train or predict'):
        dispatch_scientific(['--scientific-experiment'])


def test_inference_scientific_dispatch_precedes_notebook_directory_change(monkeypatch, tmp_path):
    import granitewxc.refinement.scientific_entrypoints as bridge
    recorded = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('sys.argv', [str(DRIVER), '--scientific-experiment', 'predict',
                                   '--cache', 'relative/phase1.h5'])
    monkeypatch.setattr(bridge, 'dispatch_scientific',
                        lambda args, **kwargs: recorded.append((Path.cwd(), args, kwargs)) or 0)
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(DRIVER), run_name='__main__')
    assert stopped.value.code == 0
    assert recorded == [(tmp_path, ['--scientific-experiment', 'predict', '--cache',
                                    'relative/phase1.h5'], {'expected_operation': 'predict'})]
    assert Path.cwd() == tmp_path


def test_wrapper_preserves_zero_validity_and_dates_only_on_explicit_request(driver):
    class Base:
        use_static = False
        def __len__(self): return 1
        def __getitem__(self, index):
            return {'x': torch.zeros(1, 2, 3), 'y': torch.zeros(2, 2, 3),
                    '__target_valid_mask': torch.ones(2, 2, 3, dtype=torch.bool),
                    '__sample_timestamp': '1977-01-07T12:00:00',
                    '__sample_predictor_path': 'case.nc'}
    wrapper = driver['CordexWrappedDataset']
    assert set(wrapper(Base())[0]) == {'x', 'y'}
    sample = wrapper(Base(), preserve_metadata=True)[0]
    assert sample['__target_valid_mask'].all() and sample['y'].sum() == 0
    assert sample['__sample_timestamp'] == '1977-01-07T12:00:00'


@pytest.mark.parametrize('head', REFINERS)
def test_normal_driver_samples_all_heads_with_stable_dates_across_batches(driver, monkeypatch, head):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        monkeypatch.setenv('GRANITE_REFINEMENT_SAMPLING_MODE', 'stable')
        function = driver['_run_full_inference']
        monkeypatch.setitem(function.__globals__, 'REFINEMENT_ENSEMBLE_SIZE', 3)
        monkeypatch.setitem(function.__globals__, 'SAVE_SAMPLING_STATES', True)
        torch.manual_seed(359)
        batch = make_batch(batch_size=2, height=9, width=11)
        model = build(head, batch, seed=41).eval()
        with torch.no_grad():
            for parameter in model.refiner.parameters():
                if parameter.requires_grad:
                    parameter.add_(0.01 * torch.randn_like(parameter))
        model.phase1.input_scalers_mu = torch.zeros(4)
        model.phase1.input_scalers_sigma = torch.ones(4)
        model._last_checkpoint_loaded = True
        batch['__sample_timestamp'] = ['1977-01-07T12:00:00', '2096-07-21T12:00:00']
        batch['__sample_predictor_path'] = ['development.nc'] * 2
        variables = ['pr', 'tasmax', 'other']
        full = function([batch], model, torch.device('cpu'), variables, SimpleNamespace(enabled=False))
        parts = []
        for i in (1, 0):
            parts.append({key: value[i:i+1] for key, value in batch.items()})
        replay = function(parts, model, torch.device('cpu'), variables, SimpleNamespace(enabled=False))
        for key in ('predictions', 'ensemble_member_predictions', 'deterministic_predictions'):
            torch.testing.assert_close(full[key][[1, 0]], replay[key], rtol=0, atol=0)
        assert any(name.startswith('initial_') for name in full['sampling_states'])
        assert not model.phase1.training
        assert not any(parameter.requires_grad for parameter in model.phase1.parameters())
    finally:
        torch.set_num_threads(previous_threads)
