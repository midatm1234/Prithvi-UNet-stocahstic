"""All full seeds use one preregistered mean branch before stochastic fitting."""
import copy
from pathlib import Path

import pytest

from examples.CORDEX_ML.cordex_scientific_full_plan import prepare_full, register_full_mean_routing
from examples.CORDEX_ML.cordex_scientific_workflow import HEADS, condition_decision, read, train_arguments, write
from granitewxc.refinement.scientific_data import file_sha256
from test_scientific_workflow import queue


def full_queue(queue, force=False):
    path, development = queue
    states = {j['id']: {'status': 'CONDITION_NOT_TRIGGERED'} for j in development['jobs']}
    for head in HEADS:
        label = 'mean_remainder' if head in ('flow_matching_unet', 'diffusion_transformer') else 'native'
        job = next(j for j in development['jobs'] if j['id'] == f'{label}/{head}')
        directory = Path(job['output']); directory.mkdir(parents=True)
        checkpoint = directory/'nearest_provisional_candidate.ckpt'; checkpoint.write_bytes(head.encode())
        point = {'score': [0, .1, .02], 'selected_checkpoint': str(checkpoint),
                 'checkpoint': str(checkpoint), 'checkpoint_sha256': file_sha256(checkpoint)}
        write(directory/'scientific_selection.json', {'nearest_provisional': point})
        effective = copy.deepcopy(job)
        if force and head == 'flow_matching_unet':
            effective['formulation'] = 'crossfit_remainder'
        states[job['id']] = {'status': 'COMPLETED_REGISTERED_STOP', 'effective_job': effective}
    write(path.with_name(path.stem+'_execution.json'), {'queue_sha256': development['sha256'], 'jobs': states})
    return read(prepare_full(path))


def complete_mean(job, required):
    directory = Path(job['output']); directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory/'nearest_provisional_candidate.ckpt'
    checkpoint.write_bytes(f"independent-seed-{job['seed']}".encode())
    write(directory/'execution_status.json', {'status': 'REACHED_REGISTERED_MAXIMUM_PROVISIONAL'})
    write(directory/'mean_crossfit_assessment.json', {'crossfit_required': required,
          'checkpoint_sha256': file_sha256(checkpoint)})


def complete_fold(job):
    directory = Path(job['output']); directory.mkdir(parents=True, exist_ok=True)
    (directory/'nearest_provisional_candidate.ckpt').write_bytes(f"fold{job['fit_fold']}-seed{job['seed']}".encode())
    write(directory/'execution_status.json', {'status': 'STOPPED_REGISTERED_PATIENCE_PROVISIONAL'})


@pytest.mark.parametrize('triggers,force', [((False, False, False), False),
    ((True, False, False), False), ((False, True, False), False),
    ((False, False, True), False), ((False, False, False), True)])
def test_any_seed_trigger_or_development_force_routes_entire_family(queue, triggers, force):
    data = full_queue(queue, force)
    ordinary, folds, stochastic = data['jobs'][:3], data['jobs'][3:9], data['jobs'][9:]
    assert len(stochastic) == 24
    assert [j['seed'] for j in ordinary] == [101, 202, 303]
    assert all(j['id'].startswith('full_mean_control/') for j in ordinary)
    assert all(j['id'].startswith('full_mean_fold_') for j in folds)
    assert all(j['head'] in HEADS for j in stochastic)
    assert (Path(data['run_root'])/'full_mean_routing_registration.json').is_file()
    # The last ordinary mean is still unknown: no folding or stochastic fitting.
    for job, required in zip(ordinary[:2], triggers[:2]):
        complete_mean(job, required)
    assert condition_decision(copy.deepcopy(folds[0]), data)['execute'] is None
    assert all(condition_decision(copy.deepcopy(j), data)['execute'] is None for j in stochastic)
    complete_mean(ordinary[2], triggers[2])
    required = force or any(triggers)
    assert all(condition_decision(copy.deepcopy(j), data)['execute'] is required for j in folds)
    if required:
        # Even direct controls wait for every required fold, including the final seed.
        for job in folds[:-1]:
            complete_fold(job)
        assert all(condition_decision(copy.deepcopy(j), data)['execute'] is None for j in stochastic)
        complete_fold(folds[-1])
    for planned in stochastic:
        job = copy.deepcopy(planned)
        decision = condition_decision(job, data)
        assert decision['execute'] is True
        uses_mean = job['comparison_group'] == 'selected' and job['head'] in ('flow_matching_unet', 'diffusion_transformer')
        if uses_mean:
            assert decision['crossfit_used'] is required
            assert job['formulation'] == ('crossfit_remainder' if required else 'mean_remainder')
            args = train_arguments(job, data)
            if required:
                assert '--mean-checkpoint' not in args
                for index, pool in enumerate((1, 0)):
                    assert Path(job['crossfit_mean_checkpoints'][index]).as_posix().endswith(
                        f"full_mean_fold_{pool}/seed_{job['seed']}/nearest_provisional_candidate.ckpt")
            else:
                assert '--mean-checkpoint' in args and '--crossfit-mean-checkpoints' not in args
        assert '--resume' not in train_arguments(job, data)
    frozen = read(Path(data['run_root'])/'workflow/full_mean_routing_decision.json')
    assert frozen['crossfit_required'] is required
    assert frozen['route_applies_to_every_registered_training_seed']
    assert len(frozen['mean_assessments']) == 3


def test_registration_and_selected_source_cannot_change_a_frozen_route(queue):
    data = full_queue(queue)
    ordinary, fold = data['jobs'][:3], data['jobs'][3]
    for job in ordinary:
        complete_mean(job, False)
    assert condition_decision(copy.deepcopy(fold), data)['execute'] is False
    assessment = Path(ordinary[1]['output'])/'mean_crossfit_assessment.json'
    changed = read(assessment); changed['crossfit_required'] = True; write(assessment, changed)
    with pytest.raises(ValueError, match='already frozen'):
        condition_decision(copy.deepcopy(fold), data)
    changed['crossfit_required'] = False; write(assessment, changed)
    checkpoint = Path(ordinary[1]['output'])/'nearest_provisional_candidate.ckpt'
    checkpoint.write_bytes(b'changed weights')
    with pytest.raises(ValueError, match='does not authenticate'):
        condition_decision(copy.deepcopy(fold), data)
    registration = Path(data['run_root'])/'full_mean_routing_registration.json'
    value = read(registration); value['training_seeds'] = [101, 202]; write(registration, value)
    with pytest.raises(ValueError, match='registration changed'):
        condition_decision(copy.deepcopy(fold), data)


def test_cannot_register_after_full_training_started_and_parent_plan_is_unchanged(queue):
    _, development = queue
    root = Path(development['run_root'])
    plan = root/'registered_experiment_plan.json'
    original = plan.read_bytes()
    checkpoint = root/'workflow/runs/full_mean_control/seed_101/last.ckpt'
    checkpoint.parent.mkdir(parents=True); checkpoint.write_bytes(b'already started')
    with pytest.raises(ValueError, match='before any full training'):
        register_full_mean_routing(root)
    assert plan.read_bytes() == original


def test_external_full_artifact_path_also_blocks_late_registration(queue, tmp_path):
    _, development = queue
    outside = tmp_path/'separate_output_volume'
    path = outside/'workflow/runs/full_control/diffusion_unet/seed_101/run_contract.json'
    write(path, {'already': 'started'})
    with pytest.raises(ValueError, match='before any full training'):
        register_full_mean_routing(development['run_root'], outside)
