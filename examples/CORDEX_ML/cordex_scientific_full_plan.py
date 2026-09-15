#!/usr/bin/env python
"""Freeze provisional development selections and prepare full multi-seed jobs.

This is executable preparation, not evidence that any command has been run.
It refuses to select recipes until the applicable development queue finishes.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.CORDEX_ML.cordex_scientific_workflow import (
    HEADS, VERSION, configuration, read, write, shell_command, INFER,
)
from granitewxc.refinement.scientific_data import canonical_sha256, file_sha256



def register_full_mean_routing(run_root, artifact_root=None):
    """Persist a common-seed routing policy before any full training begins."""
    root = Path(run_root).resolve()
    plan_path = root/'registered_experiment_plan.json'
    text = plan_path.read_text(encoding='utf8')
    plan, plan_hash = read(plan_path), hashlib.sha256(text.encode()).hexdigest()
    if plan_hash != plan_path.with_suffix('.sha256').read_text().strip():
        raise ValueError('Registered experiment plan changed before mean routing registration.')
    registration = {'version': 'common_full_seed_mean_routing_v1',
        'parent_experiment_plan_sha256': plan_hash,
        'acceptance_contract_sha256': plan['acceptance_contract_sha256'],
        'training_seeds': plan['training_seeds'], 'registered_before_full_training': True,
        'trigger': 'Any selected ordinary full-seed mean has fitting/validation physical remainder RMSE ratio < 0.5 in either variable.',
        'existing_trigger_threshold_unchanged': 0.5,
        'common_route': 'Use crossfit_remainder for every full training seed if any ordinary mean triggers or any selected development mean recipe already uses cross-fitting; otherwise all use mean_remainder.',
        'execution_order': ['all ordinary full-seed means', 'all required full-seed fold means', 'all selected/control stochastic jobs'],
        'independent_seed_family': 'Identical formulation and registered method for all seeds; individual selected means and remainder statistics remain separately fitted.',
        'training_selection_dates_and_acceptance_contract_unchanged': True}
    registration['sha256'] = canonical_sha256(registration)
    path = root/'full_mean_routing_registration.json'
    if path.exists():
        if read(path) != registration:
            raise ValueError('Refusing to overwrite a different full mean routing registration.')
        return path, registration
    bases = {root, Path(artifact_root).resolve() if artifact_root else root}
    for base in bases:
        for directory in (base/'workflow'/'runs').glob('full_*'):
            if any(p.name in ('last.ckpt', 'run_contract.json', 'execution_status.json') for p in directory.rglob('*')):
                raise ValueError('Full mean routing must be registered before any full training starts.')
    write(path, registration)
    return path, registration


def prepare_full(queue_path):
    path = Path(queue_path).resolve()
    development = read(path)
    if canonical_sha256({key: value for key, value in development.items() if key != 'sha256'}) != development['sha256']:
        raise ValueError('Development queue hash changed.')
    execution = read(path.with_name(path.stem+'_execution.json'))
    if execution['queue_sha256'] != development['sha256']:
        raise ValueError('Development execution receipt belongs to another queue.')
    unfinished = [job['id'] for job in development['jobs'] if execution['jobs'].get(job['id'], {}).get('status')
                  not in ('COMPLETED_REGISTERED_STOP', 'CONDITION_NOT_TRIGGERED', 'FAIL_NUMERICAL_EXECUTION')]
    if unfinished:
        raise ValueError('Development selection remains incomplete: ' + ', '.join(unfinished))
    root = Path(development['run_root'])
    artifact_root = Path(development.get('artifact_root', root))
    plan_path = root/'registered_experiment_plan.json'
    plan = read(plan_path)
    plan_hash = hashlib.sha256(plan_path.read_text(encoding='utf8').encode()).hexdigest()
    if (plan_hash != plan_path.with_suffix('.sha256').read_text().strip()
            or plan_hash != development['registered_plan_sha256']
            or plan['acceptance_contract_sha256'] != development['acceptance_contract_sha256']):
        raise ValueError('Registered full-fitting plan or acceptance link changed.')
    selected = {}
    for head in HEADS:
        candidates = []
        for job in development['jobs']:
            if (job['head'] != head or job['stage'] != 'screen' or job['recipe'].endswith('_only')
                    or execution['jobs'][job['id']]['status'] != 'COMPLETED_REGISTERED_STOP'):
                continue
            state = read(Path(job['output'])/'scientific_selection.json')
            selection = state.get('best_eligible') or state.get('nearest_provisional')
            if selection is None:
                continue
            if selection['score'][0] != 0:
                continue
            effective_job = execution['jobs'][job['id']].get('effective_job', job)
            candidates.append((tuple(selection['score']), effective_job, selection))
        if not candidates:
            raise ValueError(f'{head} has no complete joint-variable development point metrics.')
        _, job, point = min(candidates, key=lambda item: item[0])
        if file_sha256(point['selected_checkpoint']) != point.get('checkpoint_sha256'):
            raise ValueError('Selected development weights changed after their point metrics were recorded.')
        selected[head] = {'job': job, 'selection': point,
                          'selected_checkpoint_sha256': file_sha256(point['selected_checkpoint']),
                          'status': 'PROVISIONAL_DEVELOPMENT_RECIPE', 'production_accepted': False}
    selection = {'development_queue_sha256': development['sha256'],
                 'development_execution_receipt_sha256': file_sha256(path.with_name(path.stem+'_execution.json')),
                 'contract_sha256': development['acceptance_contract_sha256'],
                 'selected': selected, 'failed_experiments': [key for key, value in execution['jobs'].items()
                    if value['status'] == 'FAIL_NUMERICAL_EXECUTION'],
                 'production_promotion': False}
    selection['sha256'] = canonical_sha256(selection)
    selection_path = root/'workflow'/'selected_development_recipes.json'
    if selection_path.exists() and read(selection_path) != selection:
        raise ValueError('A full-run recipe selection is already frozen and cannot be overwritten.')
    write(selection_path, selection)
    jobs = []
    needs_mean = any(value['job']['formulation'] in ('mean_remainder', 'crossfit_remainder') for value in selected.values())
    force_crossfit = any(value['job']['formulation'] == 'crossfit_remainder' for value in selected.values())
    mean_routing = None
    mean_sources = [f'full_mean_control/seed_{seed}' for seed in plan['training_seeds']]
    if needs_mean:
        routing_path, mean_routing = register_full_mean_routing(root, artifact_root)
        # Every ordinary mean finishes before the first fold/remainder can run.
        for seed, job_id in zip(plan['training_seeds'], mean_sources):
            jobs.append({'id': job_id, 'head': 'deterministic_mean', 'recipe': 'native',
                         'formulation': 'deterministic_mean', 'stage': 'full', 'seed': seed,
                         'config': configuration('deterministic_mean'),
                         'output': str(artifact_root/'workflow'/'runs'/job_id)})
        for seed in plan['training_seeds']:
            for fold in (0, 1):
                fold_id = f'full_mean_fold_{fold}/seed_{seed}'
                jobs.append({'id': fold_id, 'head': 'deterministic_mean', 'recipe': 'native',
                             'formulation': 'deterministic_mean', 'stage': 'full', 'seed': seed,
                             'config': configuration('deterministic_mean'), 'fit_fold': fold,
                             'fold_registration': str(root/'mean_fold_registration.json'),
                             'condition': {'kind': 'mean_crossfit_required', 'source': f'full_mean_control/seed_{seed}',
                                           'force_crossfit': force_crossfit, 'mean_sources': mean_sources,
                                           'mean_routing_registration': str(routing_path), 'mean_routing_sha256': mean_routing['sha256']},
                             'output': str(artifact_root/'workflow'/'runs'/fold_id)})
    for seed in plan['training_seeds']:
        for head in HEADS:
            source = selected[head]['job']
            for group in ('selected', 'control'):
                job_id = f'full_{group}/{head}/seed_{seed}'
                job = {'id': job_id, 'head': head, 'recipe': source['recipe'] if group == 'selected' else 'existing_full_recipe',
                       'formulation': source['formulation'] if group == 'selected' else 'direct',
                       'stage': 'full', 'seed': seed, 'config': configuration(head),
                       'output': str(artifact_root/'workflow'/'runs'/job_id), 'comparison_group': group,
                       'role': 'fresh fitting on every registered fitting date; scientific acceptance pending'}
                if needs_mean:
                    job['full_mean_barrier'] = {
                        'mean_sources': mean_sources, 'force_crossfit': force_crossfit,
                        'mean_routing_registration': str(routing_path), 'mean_routing_sha256': mean_routing['sha256'],
                        'all_fold_jobs': [f'full_mean_fold_{fold}/seed_{item}' for item in plan['training_seeds'] for fold in (0, 1)]}
                if group == 'selected':
                    for key in ('alignment', 'normalize_predictors'):
                        if key in source:
                            job[key] = source[key]
                    if source['formulation'] in ('mean_remainder', 'crossfit_remainder'):
                        mean_job = f'full_mean_control/seed_{seed}'
                        job['condition'] = {'kind': 'frozen_mean_ready', 'source': mean_job,
                                            'force_crossfit': force_crossfit, 'mean_sources': mean_sources,
                                            'mean_routing_registration': str(routing_path), 'mean_routing_sha256': mean_routing['sha256'],
                                            'fold_jobs': [f'full_mean_fold_0/seed_{seed}', f'full_mean_fold_1/seed_{seed}'],
                                            'fold_registration': str(root/'mean_fold_registration.json')}
                        job['mean_checkpoint'] = str(artifact_root/'workflow'/'runs'/mean_job/'nearest_provisional_candidate.ckpt')
                jobs.append(job)
    full = {'version': VERSION, 'run_root': str(root), 'artifact_root': str(artifact_root), 'registered_plan_sha256': development['registered_plan_sha256'],
            'acceptance_contract_sha256': development['acceptance_contract_sha256'],
            'selection_sha256': selection['sha256'], 'jobs': jobs, 'production_promotion': False,
            'full_mean_routing_registration_sha256': mean_routing['sha256'] if mean_routing else None,
            'seed_policy': 'Independent fresh training seeds; shared paired sampler streams across all products.'}
    full['sha256'] = canonical_sha256(full)
    full_path = root/'workflow'/'full_queue.json'
    if full_path.exists() and read(full_path) != full:
        raise ValueError('Refusing to overwrite the frozen full-fitting queue.')
    write(full_path, full)
    prepare_inference_commands(full)
    return full_path


def prepare_inference_commands(queue):
    root = Path(queue['run_root'])
    artifact_root = Path(queue.get('artifact_root', root))
    commands = []
    for job in queue['jobs']:
        if job['head'] not in HEADS:
            continue
        directory = artifact_root/'workflow'/'assessments'/job['id']
        for size, diagnostic in ((20, False), (10, True), (20, True), (50, True)):
            output = directory/(f'diagnostic_members_{size}.h5' if diagnostic else 'validation_members_20.h5')
            args = ['mamba', 'run', '-n', 'Prithvi', 'python', 'examples/CORDEX_ML/cordex_scientific_prediction.py',
                    '--training-output', job['output'], '--cache', str(root/'cache'/'phase1.h5'),
                    '--output', str(output), '--split', 'validation', '--members', str(size), '--device', 'cuda:0']
            if diagnostic:
                args.append('--diagnostic-dates')
            commands.append({'training_job': job['id'], 'members': size, 'diagnostic': diagnostic,
                             'argv': args, 'command': shell_command(args), 'executed': False})
    write(root/'workflow'/'full_inference_commands.json', commands)
    powershell = root/'workflow'/'RUN_FULL_INFERENCE.ps1'
    powershell.write_text("$ErrorActionPreference = 'Stop'\nSet-Location -LiteralPath " + "'" + str(ROOT).replace("'", "''") + "'\n" +
                         '\n'.join(item['command'] + '\nif ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }' for item in commands) + '\n',
                         encoding='utf8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development-queue', required=True)
    args = parser.parse_args(argv)
    print(prepare_full(args.development_queue))


if __name__ == '__main__':
    main()
