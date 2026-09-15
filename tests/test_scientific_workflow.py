"""Execution receipts cannot silently skip heads, reuse weights or select early."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from examples.CORDEX_ML.cordex_scientific_workflow import (
    HEADS, condition_decision, create_queue, read, run_queue, train_arguments, write,
)
from examples.CORDEX_ML.cordex_scientific_full_plan import prepare_full


@pytest.fixture
def queue(tmp_path):
    plan = {'screening_seed': 101, 'training_seeds': [101, 202, 303],
            'acceptance_contract_sha256': 'a'*64}
    path = tmp_path/'registered_experiment_plan.json'
    text = json.dumps(plan, indent=2)+'\n'
    path.write_text(text, encoding='utf8')
    path.with_suffix('.sha256').write_text(hashlib.sha256(text.encode()).hexdigest())
    return create_queue(tmp_path)


def test_each_head_has_actual_config_overfit_native_existing_and_geometry(queue):
    path, data = queue
    for head in HEADS:
        jobs = [job for job in data['jobs'] if job['head'] == head]
        assert {f'{label}/{head}' for label in ('overfit', 'native', 'existing_full_recipe', 'coordinates')} <= {j['id'] for j in jobs}
        assert all(job['config'].endswith(head+'.yaml') for job in jobs)
        assert sum(job['stage'] == 'overfit' for job in jobs) == 1
        fresh = train_arguments(jobs[0], data)
        assert '--resume' not in fresh
        continued = train_arguments(jobs[0], data, resume=True)
        assert continued[-2:] == ['--resume', str(Path(jobs[0]['output'])/'last.ckpt')]
    assert create_queue(data['run_root'])[1] == data
    altered = read(path); altered['jobs'][0]['seed'] = 999; write(path, altered)
    with pytest.raises(ValueError, match='queue hash'):
        run_queue(path)


def test_cache_or_disk_pause_cannot_claim_completed_fitting(queue, monkeypatch):
    path, data = queue
    assert run_queue(path) == 75
    state_path = path.with_name(path.stem+'_execution.json')
    assert read(state_path)['status'] == 'WAITING_FOR_AUTHENTICATED_PHASE1_CACHE'
    write(Path(data['run_root'])/'cache'/'phase1.manifest.json', {'complete': True})
    import examples.CORDEX_ML.cordex_scientific_workflow as workflow
    monkeypatch.setattr(workflow.shutil, 'disk_usage', lambda _: SimpleNamespace(free=29*1024**3))
    assert run_queue(path) == 75
    assert read(state_path)['status'] == 'INCONCLUSIVE_RESOURCE_PAUSE'
    assert read(state_path)['next_job'].startswith('overfit/')
    with pytest.raises(ValueError, match='selection remains incomplete'):
        prepare_full(path)


def test_support_trigger_counts_valid_zero_cells_and_signed_raw_members(queue):
    _, data = queue
    source = next(job for job in data['jobs'] if job['id'] == 'native/flow_matching_unet')
    output = Path(source['output']); assessment = output/'assessment_epoch_0020'
    assessment.mkdir(parents=True)
    write(output/'execution_status.json', {'status': 'STOPPED_REGISTERED_PATIENCE_PROVISIONAL'})
    write(output/'scientific_selection.json', {'nearest_provisional': {'checkpoint': str(assessment/'candidate.ckpt')}})
    with h5py.File(assessment/'validation_endpoints.h5', 'w') as f:
        f.attrs['variables'] = json.dumps(['pr', 'tasmax'])
        f.create_dataset('dates', data=[0])
        raw = np.zeros((1, 2, 2, 1, 10), np.float32)
        raw[:, :, 0, 0, :2] = -1
        valid = np.ones((1, 2, 1, 10), bool)
        valid[:, 0, 0, 0] = False
        f.create_dataset('raw_members', data=raw)
        f.create_dataset('valid', data=valid)
    job = next(job for job in data['jobs'] if job['id'] == 'total_support/flow_matching_unet')
    result = condition_decision(job, data)
    assert result['execute'] is True
    assert result['valid_member_cells'] == 18
    assert result['negative_frequency'] == pytest.approx(1/9)


def test_full_selection_requires_finished_jobs_then_prepares_three_fresh_seeds(queue):
    path, data = queue
    states = {job['id']: {'status': 'CONDITION_NOT_TRIGGERED'} for job in data['jobs']}
    for head in HEADS:
        job = next(job for job in data['jobs'] if job['id'] == f'native/{head}')
        out = Path(job['output']); out.mkdir(parents=True)
        checkpoint = out/'nearest_provisional_candidate.ckpt'; checkpoint.write_bytes(head.encode())
        selection = {'score': [0, .1, .02], 'selected_checkpoint': str(checkpoint), 'checkpoint': str(checkpoint),
                     'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        write(out/'scientific_selection.json', {'nearest_provisional': selection})
        states[job['id']]['status'] = 'COMPLETED_REGISTERED_STOP'
    write(path.with_name(path.stem+'_execution.json'), {'queue_sha256': data['sha256'], 'jobs': states})
    full_path = prepare_full(path)
    full = read(full_path)
    assert len(full['jobs']) == 24
    for head in HEADS:
        for group in ('selected', 'control'):
            jobs = [j for j in full['jobs'] if j['head'] == head and j['comparison_group'] == group]
            assert {j['seed'] for j in jobs} == {101, 202, 303}
            assert all('--resume' not in train_arguments(j, full) for j in jobs)
    commands = read(Path(data['run_root'])/'workflow'/'full_inference_commands.json')
    assert len(commands) == 96
    assert all(not entry['executed'] for entry in commands)
    assert sum(not entry['diagnostic'] and entry['members'] == 20 for entry in commands) == 24
    assert full['production_promotion'] is False
