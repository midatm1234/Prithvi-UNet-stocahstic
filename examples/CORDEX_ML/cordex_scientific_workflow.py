#!/usr/bin/env python
"""Resumable execution of the registered four-head development experiments.

This controller records completed, failed, conditional and unexecuted jobs.
Selection is provisional. It never changes a production checkpoint or converts
an unfinished experiment into scientific acceptance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from granitewxc.refinement.scientific_data import canonical_sha256, file_sha256

HEADS = ('diffusion_unet', 'diffusion_transformer', 'flow_matching_unet', 'flow_matching_transformer')
TRAIN = 'examples/CORDEX_ML/cordex_refinement_training.py'
INFER = 'examples/CORDEX_ML/notebooks/SA_downscaling_inference_T2_ACCESS-CM2_static.py'
VERSION = 'registered_scientific_execution_queue_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')
    temporary.replace(path)


def shell_command(argv):
    """Reviewable PowerShell command; execution uses an argument list instead."""
    return '& ' + ' '.join("'" + str(value).replace("'", "''") + "'" for value in argv)


def configuration(head):
    name = 'flow_matching_unet' if head == 'deterministic_mean' else head
    return f'examples/CORDEX_ML/SA_T2_ACCESS-CM2_static_{name}.yaml'


def create_queue(run_root, artifact_root=None):
    run_root = Path(run_root).resolve()
    artifact_base = Path(artifact_root).resolve() if artifact_root else run_root
    plan_path = run_root / 'registered_experiment_plan.json'
    plan = read(plan_path)
    # The original registration hashes normalized UTF-8 text, not canonical JSON.
    plan_hash = hashlib.sha256(plan_path.read_text(encoding='utf8').encode()).hexdigest()
    if plan_hash != plan_path.with_suffix('.sha256').read_text().strip():
        raise ValueError('Registered experiment plan changed.')
    jobs = []

    def add(label, head, recipe='native', formulation='direct', stage='screen', **extra):
        job_id = f'{label}/{head}'
        jobs.append({'id': job_id, 'head': head, 'recipe': recipe, 'formulation': formulation,
                     'stage': stage, 'seed': plan['screening_seed'], 'config': configuration(head),
                     'output': str(artifact_base/'workflow'/'runs'/job_id), **extra})

    for head in HEADS:
        add('overfit', head, stage='overfit', role='capacity diagnostic; never recipe selection')
    for head in HEADS:
        add('existing_full_recipe', head, recipe='existing_full_recipe')
        add('native', head)
        if head.endswith('transformer'):
            add('native_production_gate', head, recipe='native_production_gate')
    for recipe in ('reconstruction_loss', 'multiscale_loss', 'gradient_loss', 'mean_bias_loss', 'zero_source_025'):
        add(recipe, 'flow_matching_unet', recipe=recipe)
    add('mean_control', 'deterministic_mean', formulation='deterministic_mean')
    for fold in (0, 1):
        add(f'mean_fold_{fold}', 'deterministic_mean', formulation='deterministic_mean',
            fit_fold=fold, fold_registration=str(run_root/'mean_fold_registration.json'),
            condition={'kind': 'mean_crossfit_required', 'source': 'mean_control/deterministic_mean'})
    for head in HEADS:
        add('total_support', head, formulation='total_support',
            condition={'kind': 'negative_frequency', 'source': f'native/{head}', 'threshold': .05})
    for head in ('flow_matching_unet', 'diffusion_transformer'):
        for recipe in ('pr_only', 'tasmax_only', 'fixed_pr_weight_2'):
            add(recipe, head, recipe=recipe,
                role='variable interference diagnostic' if recipe.endswith('_only') else 'joint fixed-weight candidate')
    for head in HEADS:
        add('coordinates', head, alignment='coordinates')
    mean_job = 'mean_control/deterministic_mean'
    mean_checkpoint = artifact_base/'workflow'/'runs'/mean_job/'nearest_provisional_candidate.ckpt'
    for head in HEADS:
        add('mean_remainder', head, formulation='mean_remainder', mean_checkpoint=str(mean_checkpoint),
            condition={'kind': 'frozen_mean_ready', 'source': mean_job,
                       'fold_jobs': ['mean_fold_0/deterministic_mean', 'mean_fold_1/deterministic_mean'],
                       'fold_registration': str(run_root/'mean_fold_registration.json')})
    queue = {'version': VERSION, 'run_root': str(run_root), 'registered_plan_sha256': plan_hash, 'artifact_root': str(artifact_base),
             'acceptance_contract_sha256': plan['acceptance_contract_sha256'], 'jobs': jobs,
             'head_order': list(HEADS), 'production_promotion': False,
             'deduplicated': ['U-Net native_production_gate equals native: no trainable correction gate'],
             'full_stage': 'Prepared after all applicable development jobs finish; selection stays provisional.'}
    queue['sha256'] = canonical_sha256(queue)
    filename = 'development_queue_artifacts.json' if artifact_root else 'development_queue.json'
    path = run_root/'workflow'/filename
    if path.exists() and read(path) != queue:
        raise ValueError('Refusing to change an existing execution queue.')
    write(path, queue)
    return path, queue


def train_arguments(job, queue, resume=False):
    root = Path(queue['run_root'])
    args = [TRAIN, '--scientific-experiment', 'train', '--config', job['config'],
            '--cache', str(root/'cache'/'phase1.h5'), '--plan', str(root/'registered_experiment_plan.json'),
            '--contract', str(root/'frozen_scientific_acceptance_v2.json'), '--output', job['output'],
            '--head', job['head'], '--recipe', job['recipe'], '--formulation', job['formulation'],
            '--stage', job['stage'], '--seed', str(job['seed']), '--device', 'cuda:0']
    for key in ('alignment', 'mean_checkpoint', 'fit_fold', 'fold_registration', 'crossfit_registration'):
        if job.get(key) is not None:
            args.extend(['--' + key.replace('_', '-'), str(job[key])])
    if job.get('crossfit_mean_checkpoints'):
        args.extend(['--crossfit-mean-checkpoints', *job['crossfit_mean_checkpoints']])
    if 'normalize_predictors' in job:
        args.append('--normalize-predictors' if job['normalize_predictors'] else '--no-normalize-predictors')
    if resume:
        args.extend(['--resume', str(Path(job['output'])/'last.ckpt')])
    return args


def selected_assessment(job):
    out = Path(job['output'])
    if job['formulation'] == 'deterministic_mean':
        decisions = [(read(path)['score'], path.parent) for path in out.glob('assessment_epoch_*/decision.json')]
        if not decisions:
            raise ValueError('Mean control has no completed sampled assessment.')
        return min(decisions, key=lambda pair: tuple(pair[0]))[1]
    state = read(out/'scientific_selection.json')
    selected = state.get('best_eligible') or state.get('nearest_provisional')
    if selected is None:
        raise ValueError('No selected provisional or eligible checkpoint.')
    return Path(selected['checkpoint']).parent



def common_full_mean_decision(condition, queue):
    """One immutable mean branch for all independently trained full seeds.

    Wait for every ordinary mean before any fold or remainder is started. The
    registration adds scheduling only: the existing physical-RMSE trigger and
    all training/selection/acceptance date rules remain unchanged.
    """
    path = Path(condition['mean_routing_registration'])
    registration = read(path)
    unsigned = {key: value for key, value in registration.items() if key != 'sha256'}
    if (canonical_sha256(unsigned) != registration.get('sha256')
            or registration.get('sha256') != condition.get('mean_routing_sha256')
            or registration.get('parent_experiment_plan_sha256') != queue['registered_plan_sha256']
            or registration.get('acceptance_contract_sha256') != queue['acceptance_contract_sha256']
            or registration.get('version') != 'common_full_seed_mean_routing_v1'
            or registration.get('registered_before_full_training') is not True):
        raise ValueError('Full mean family routing registration changed or belongs to another plan.')
    seed_values = registration['training_seeds']
    expected_ids = [f'full_mean_control/seed_{seed}' for seed in seed_values]
    if condition.get('mean_sources') != expected_ids:
        raise ValueError('Common mean routing must include every registered training seed in order.')
    assessments = []
    for source_id, seed in zip(expected_ids, seed_values):
        source = next(item for item in queue['jobs'] if item['id'] == source_id)
        if (source['head'] != 'deterministic_mean' or source['stage'] != 'full'
                or source['formulation'] != 'deterministic_mean' or source['seed'] != seed):
            raise ValueError('Common mean source is not the registered full-seed ordinary mean.')
        directory = Path(source['output'])
        status, assessment_path = directory/'execution_status.json', directory/'mean_crossfit_assessment.json'
        checkpoint = directory/'nearest_provisional_candidate.ckpt'
        if (not status.is_file() or not read(status).get('status', '').startswith(('STOPPED_REGISTERED', 'REACHED_REGISTERED'))
                or not assessment_path.is_file() or not checkpoint.is_file()):
            return {'execute': None, 'reason': 'All registered full-seed ordinary means must finish before choosing one common route.',
                    'pending_mean_job': source_id}
        assessment = read(assessment_path)
        if type(assessment.get('crossfit_required')) is not bool:
            raise ValueError('A full mean source lacks an explicit Boolean cross-fit assessment.')
        checkpoint_sha = file_sha256(checkpoint)
        if checkpoint_sha != assessment.get('checkpoint_sha256'):
            raise ValueError('Full mean assessment does not authenticate its selected checkpoint.')
        assessments.append({'job': source_id, 'training_seed': seed,
                            'checkpoint_sha256': checkpoint_sha,
                            'assessment_sha256': file_sha256(assessment_path), 'assessment': assessment})
    force = bool(condition.get('force_crossfit', False))
    required = force or any(value['assessment']['crossfit_required'] for value in assessments)
    decision = {'version': 'common_full_seed_mean_route_decision_v1',
                'registration_sha256': registration['sha256'], 'crossfit_required': required,
                'development_forces_crossfit': force, 'mean_assessments': assessments,
                'route_applies_to_every_registered_training_seed': True}
    decision['sha256'] = canonical_sha256(decision)
    destination = Path(queue['run_root'])/'workflow'/'full_mean_routing_decision.json'
    if destination.exists() and read(destination) != decision:
        raise ValueError('The common full-seed mean route is already frozen; changed sources cannot switch a running family.')
    write(destination, decision)
    return {'execute': True, 'crossfit_required': required, 'assessment': decision}


def condition_decision(job, queue):
    barrier = job.get('full_mean_barrier')
    if barrier is not None:
        common = common_full_mean_decision(barrier, queue)
        if common['execute'] is None:
            return common
        if common['crossfit_required']:
            for fold_id in barrier['all_fold_jobs']:
                source = next(item for item in queue['jobs'] if item['id'] == fold_id)
                directory = Path(source['output'])
                status = directory/'execution_status.json'
                if (not status.is_file() or not (directory/'nearest_provisional_candidate.ckpt').is_file()
                        or not read(status).get('status', '').startswith(('STOPPED_REGISTERED', 'REACHED_REGISTERED'))):
                    return {'execute': None, 'reason': 'All required full-seed mean folds must finish before any stochastic full job.',
                            'pending_fold_job': fold_id, 'common_mean_route': common['assessment']}
    condition = job.get('condition')
    if not condition:
        return {'execute': True}
    source = next(item for item in queue['jobs'] if item['id'] == condition['source'])
    status_path = Path(source['output'])/'execution_status.json'
    if not status_path.is_file() or not read(status_path)['status'].startswith(('STOPPED_REGISTERED', 'REACHED_REGISTERED')):
        return {'execute': None, 'reason': 'Required development source has not completed its stopping rule.'}
    if condition['kind'] == 'negative_frequency':
        import h5py
        import numpy as np
        path = selected_assessment(source)/'validation_endpoints.h5'
        negative = count = 0
        with h5py.File(path, 'r') as fields:
            variable = json.loads(fields.attrs['variables']).index('pr')
            for start in range(0, len(fields['dates']), 8):
                raw = fields['raw_members'][start:start+8, :, variable]
                valid = fields['valid'][start:start+8, variable].astype(bool)
                support = np.broadcast_to(valid[:, None], raw.shape)
                if np.any(support & ~np.isfinite(raw)):
                    raise ValueError('Cannot select support formulation from nonfinite precipitation.')
                count += int(support.sum())
                negative += int(((raw < 0) & support).sum())
        fraction = negative/count if count else None
        if fraction is None:
            return {'execute': None, 'reason': 'No valid precipitation cells.'}
        return {'execute': fraction > condition['threshold'], 'negative_frequency': fraction,
                'negative_count': negative, 'valid_member_cells': count, 'source': str(path),
                'threshold': condition['threshold']}
    if condition['kind'] in ('frozen_mean_ready', 'mean_crossfit_required'):
        check = Path(source['output'])/'mean_crossfit_assessment.json'
        if not check.exists():
            return {'execute': None, 'reason': 'Frozen mean crossfit trigger has not been evaluated.'}
        receipt = read(check)
        if condition.get('mean_sources') is not None:
            common = common_full_mean_decision(condition, queue)
            if common['execute'] is None:
                return common
            crossfit_required, receipt = common['crossfit_required'], common['assessment']
        else:
            crossfit_required = bool(receipt.get('crossfit_required')) or condition.get('force_crossfit', False)
        if condition['kind'] == 'mean_crossfit_required':
            return {'execute': crossfit_required, 'assessment': receipt}
        if crossfit_required:
            fold_ids = condition.get('fold_jobs', [])
            if len(fold_ids) != 2:
                return {'execute': None, 'reason': 'Two registered mean-fold jobs are required.'}
            checkpoints = []
            for fold_id in fold_ids:
                fold_job = next(item for item in queue['jobs'] if item['id'] == fold_id)
                status_path = Path(fold_job['output'])/'execution_status.json'
                checkpoint = Path(fold_job['output'])/'nearest_provisional_candidate.ckpt'
                if (not status_path.is_file() or not checkpoint.is_file()
                        or not read(status_path)['status'].startswith(('STOPPED_REGISTERED', 'REACHED_REGISTERED'))):
                    return {'execute': None, 'reason': 'Registered mean-overfit trigger requires finished crossfitted mean models.',
                            'assessment': receipt}
                checkpoints.append(str(checkpoint))
            job['formulation'] = 'crossfit_remainder'
            job['crossfit_registration'] = condition['fold_registration']
            # CLI order is held-out pool0, held-out pool1; trained pools are opposite.
            job['crossfit_mean_checkpoints'] = checkpoints[::-1]
            job.pop('mean_checkpoint', None)
        return {'execute': True, 'assessment': receipt, 'crossfit_used': crossfit_required}
    raise ValueError('Unknown registered conditional experiment.')


def run_queue(path, *, wait_for_cache=False, only_stage=None):
    path = Path(path).resolve()
    queue = read(path)
    if canonical_sha256({k: v for k, v in queue.items() if k != 'sha256'}) != queue['sha256']:
        raise ValueError('Execution queue hash changed.')
    root = Path(queue['run_root'])
    state_path = path.with_name(path.stem + '_execution.json')
    state = read(state_path) if state_path.exists() else {'queue_sha256': queue['sha256'], 'jobs': {}, 'status': 'READY'}
    if state['queue_sha256'] != queue['sha256']:
        raise ValueError('Execution receipts identify another queue.')
    for job in queue['jobs']:
        state['jobs'].setdefault(job['id'], {
            'status': 'NOT_EXECUTED',
            'planned_command': shell_command(['mamba', 'run', '-n', 'Prithvi', 'python',
                                              *train_arguments(job, queue)]),
        })
    cache = root/'cache'/'phase1.manifest.json'
    while not cache.exists():
        state.update(status='WAITING_FOR_AUTHENTICATED_PHASE1_CACHE', updated_utc=datetime.now(timezone.utc).isoformat())
        write(state_path, state)
        if not wait_for_cache or (root/'workflow'/'STOP').exists():
            return 75
        time.sleep(15)
    for original in queue['jobs']:
        job = dict(original)
        if only_stage and job['stage'] != only_stage:
            continue
        receipt = state['jobs'].get(job['id'], {})
        if receipt.get('status') in ('COMPLETED_REGISTERED_STOP', 'CONDITION_NOT_TRIGGERED', 'FAIL_NUMERICAL_EXECUTION'):
            continue
        output_volume = Path(job['output']).anchor or str(root)
        available = {str(root): shutil.disk_usage(root).free,
                     str(output_volume): shutil.disk_usage(output_volume).free}
        if (root/'workflow'/'STOP').exists() or min(available.values()) < 30*1024**3:
            state.update(status='INCONCLUSIVE_RESOURCE_PAUSE', next_job=job['id'], free_bytes=available)
            write(state_path, state)
            return 75
        condition = condition_decision(job, queue)
        if condition['execute'] is not True:
            receipt = {'status': 'CONDITION_NOT_TRIGGERED' if condition['execute'] is False else 'PENDING_DEPENDENCY',
                       'condition': condition}
            state['jobs'][job['id']] = receipt
            write(state_path, state)
            continue
        resume = (Path(job['output'])/'last.ckpt').exists()
        args = train_arguments(job, queue, resume)
        commands = ['mamba', 'run', '-n', 'Prithvi', 'python', *args]
        log = root/'workflow'/'logs'/(job['id'].replace('/', '__')+'.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        receipt = {'status': 'RUNNING', 'started_utc': datetime.now(timezone.utc).isoformat(),
                   'condition': condition, 'command': shell_command(commands), 'argv': commands,
                   'log': str(log), 'resume': resume, 'output': job['output'], 'effective_job': job}
        state['jobs'][job['id']] = receipt
        state.update(status='RUNNING', active_job=job['id'])
        write(state_path, state)
        print(f"Starting {job['id']}; log={log}", flush=True)
        with log.open('a', encoding='utf8') as output:
            result = subprocess.run([sys.executable, *args], cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
        actual = Path(job['output'])/'execution_status.json'
        execution = read(actual) if actual.exists() else {}
        status = execution.get('status', '')
        receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), returncode=result.returncode,
                       execution_status=status, epoch=execution.get('epoch'), updates=execution.get('updates'))
        if status.startswith(('STOPPED_REGISTERED', 'REACHED_REGISTERED')) and result.returncode == 0:
            receipt['status'] = 'COMPLETED_REGISTERED_STOP'
        elif status == 'FAIL_NUMERICAL_EXECUTION':
            receipt['status'] = status
        else:
            receipt['status'] = 'INCONCLUSIVE_UNFINISHED'
            receipt['exact_resume_command'] = shell_command(['mamba', 'run', '-n', 'Prithvi', 'python',
                *train_arguments(job, queue, (Path(job['output'])/'last.ckpt').exists())])
            state.update(status='INCONCLUSIVE_RESOURCE_PAUSE' if status.startswith('INCONCLUSIVE_RESOURCE') else 'STOPPED_EXECUTION_ERROR')
            write(state_path, state)
            return 75 if status.startswith('INCONCLUSIVE_RESOURCE') else 2
        write(state_path, state)
    state.update(status='DEVELOPMENT_FINISHED_WITH_PENDING_DEPENDENCIES' if any(
        entry.get('status') == 'PENDING_DEPENDENCY' for entry in state['jobs'].values()) else ('REQUESTED_STAGE_COMPLETE_OTHER_JOBS_UNEXECUTED' if any(
            entry.get('status') == 'NOT_EXECUTED' for entry in state['jobs'].values())
            else 'REGISTERED_QUEUE_COMPLETED'),
        production_promotion=False)
    write(state_path, state)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    p = sub.add_parser('plan'); p.add_argument('--run-root', required=True); p.add_argument('--artifact-root')
    r = sub.add_parser('run'); r.add_argument('--queue', required=True)
    r.add_argument('--wait-for-cache', action='store_true')
    r.add_argument('--only-stage', choices=('overfit', 'screen', 'full'))
    s = sub.add_parser('status'); s.add_argument('--queue', required=True)
    args = parser.parse_args(argv)
    if args.operation == 'plan':
        path, queue = create_queue(args.run_root, args.artifact_root)
        print(json.dumps({'queue': str(path), 'jobs': len(queue['jobs']), 'sha256': queue['sha256']}))
        return 0
    if args.operation == 'run':
        return run_queue(args.queue, wait_for_cache=args.wait_for_cache, only_stage=args.only_stage)
    path = Path(args.queue)
    state = path.with_name(path.stem + '_execution.json')
    print(json.dumps(read(state) if state.exists() else {'status': 'NOT_EXECUTED'}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
