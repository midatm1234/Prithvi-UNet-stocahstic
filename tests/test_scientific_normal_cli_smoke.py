"""Run real experimental fitting/sampling through both normal CLI entry points.

Only data/plan preparation uses miniature synthetic fixtures. Dispatch, parser,
optimizer, checkpoint serialization, model restoration and sampling execute.
This is an engineering smoke test, not climate-performance evidence.
"""
import importlib
import os
from pathlib import Path
import runpy

import h5py
import numpy as np
import pytest
import torch

from test_cordex_scientific_experiments import RUNNER, fixture_train_args, MemoryPrepared
from test_refinement_models import REFINERS

ROOT = Path(__file__).resolve().parents[1]
INFERENCE = ROOT/'examples/CORDEX_ML/notebooks/SA_downscaling_inference_T2_ACCESS-CM2_static.py'


@pytest.mark.parametrize('head', REFINERS)
def test_unmocked_normal_training_and_inference_scientific_entrypoints(tmp_path, monkeypatch, head):
    runner = importlib.import_module('examples.CORDEX_ML.cordex_scientific_experiments')
    args = fixture_train_args(tmp_path, monkeypatch, head, tmp_path/'training')
    # The real canonical module is imported by the unmodified dispatch bridge.
    monkeypatch.setattr(runner, '_load_registered_plan', RUNNER._load_registered_plan)
    monkeypatch.setattr(runner, 'prepare_conditioning', RUNNER.prepare_conditioning)
    training = importlib.import_module('examples.CORDEX_ML.cordex_refinement_training')
    code = training.main(['--scientific-experiment', 'train', '--config', args.config,
        '--cache', args.cache, '--plan', args.plan, '--contract', args.contract,
        '--output', args.output, '--head', head, '--stage', 'screen', '--device', 'cpu'])
    assert code == 0
    checkpoint = Path(args.output)/'nearest_provisional_candidate.ckpt'
    payload = torch.load(checkpoint, weights_only=False)
    assert payload['run_contract']['head'] == head
    assert payload['variables'] == ['pr', 'tasmax']
    assert payload['progress']['updates'] == 4
    assert all(torch.isfinite(value).all() for value in payload['candidate']['state_dict'].values())
    output = tmp_path/'normal_inference.h5'
    monkeypatch.setattr('sys.argv', [str(INFERENCE), '--scientific-experiment', 'predict',
        '--checkpoint', str(checkpoint), '--cache', args.cache, '--output', str(output),
        '--split', 'screen_validation', '--members', '3', '--device', 'cpu'])
    cwd = Path.cwd()
    try:
        with pytest.raises(SystemExit) as finished:
            runpy.run_path(str(INFERENCE), run_name='__main__')
        assert finished.value.code == 0
        assert Path.cwd() == cwd
    finally:
        os.chdir(cwd)
    truth = MemoryPrepared(validation=True)
    with h5py.File(output, 'r') as fields:
        assert fields.attrs['complete']
        assert fields['processed_members'].shape == (4, 3, 2, 9, 11)
        np.testing.assert_array_equal(fields['baseline'][:], truth.base.numpy())
        np.testing.assert_array_equal(fields['truth'][:], truth.truth.numpy())
        raw = fields['raw_members'][:]
        correction = raw - truth.base.numpy()[:, None]
        assert np.isfinite(raw).all()
        for variable in range(2):
            assert np.any(correction[:, :, variable] != 0)
            assert np.std(raw[:, :, variable], axis=1).mean() > 0
        assert np.min(fields['processed_members'][:, :, 0]) >= 0
    assert not (Path(args.output)/'best_scientific_candidate.ckpt').exists()
