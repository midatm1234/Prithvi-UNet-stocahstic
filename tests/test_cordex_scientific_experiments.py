"""Scientific runner controls: native sampling, replay, resume, and artifacts."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
import yaml

from granitewxc.refinement.config import resolve_refinement_config
from granitewxc.refinement.experiments import ExperimentalRefiner, experiment_recipes
from test_refinement_models import REFINERS, small_config

PATH = Path(__file__).resolve().parents[1]/'examples/CORDEX_ML/cordex_scientific_experiments.py'
SPEC = importlib.util.spec_from_file_location('scientific_runner_test_module', PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)
ROOT = PATH.parents[2]
ARTIFACT = ROOT/'artifacts/refinement_validation/scientific_acceptance_20260909T230807Z'


class MemoryPrepared(torch.utils.data.Dataset):
    def __init__(self, validation=False):
        generator = torch.Generator().manual_seed(19 if validation else 18)
        self.condition = torch.randn(4, 4, 9, 11, generator=generator)
        self.base = torch.ones(4, 2, 9, 11)
        self.base[:, 0] = 2.; self.base[:, 1] = 270.
        self.truth = self.base + torch.stack((.25*self.condition[:, 0]+.2, .5*self.condition[:, 1]-.3), 1)
        self.truth[:, 0, :2] = 0
        self.valid = torch.ones_like(self.truth, dtype=torch.bool)
        year = '1977' if validation else '1961'
        self.dates = [year+d for d in ('-01-03T12:00:00', '-01-11T12:00:00', '-04-03T12:00:00', '-04-11T12:00:00')]
        self.ids = ['case|'+d for d in self.dates]
        self.latitude = np.linspace(-34.5, -22., 9); self.longitude = np.linspace(21, 32.5, 11)
        self.manifest = {'variables': ['pr', 'tasmax'], 'units': {'pr': 'mm/day', 'tasmax': 'K'}, 'calendar': 'noleap',
            'cond_channels': 4, 'nonnegative': [True, False], 'sha256': 'validation_prepared' if validation else 'fitting_prepared',
            'identity': {'version': RUNNER.PREPARED_VERSION, 'source_cache_sha256': 'source', 'phase1_fingerprint': 'phase1',
                         'conditioning': {'input_predictors': True}, 'split': 'screen_validation' if validation else 'screen_fit',
                         'selected_dates_sha256': 'selection-validation' if validation else 'selection-fitting'}}
    def __len__(self): return 4
    def __getitem__(self, i):
        return {'conditioning': self.condition[i], 'baseline': self.base[i], 'truth': self.truth[i], 'valid': self.valid[i],
                'date': self.dates[i], 'sample_id': self.ids[i]}


def cfg(head):
    raw = small_config(head)
    raw['refinement']['residual_normalization'] = {'method': 'standardize'}
    return resolve_refinement_config(raw)


@pytest.mark.parametrize('head', REFINERS)
def test_nonzero_canonical_eight_sampler_nested_members_and_date_order(head):
    dataset = MemoryPrepared()
    config = cfg(head)
    model = ExperimentalRefiner(config, experiment_recipes(config, ('pr', 'tasmax'))['native'], cond_channels=4)
    # Avoid a trivial all-zero clean-output parity test.
    torch.manual_seed(22)
    with torch.no_grad():
        for p in model.refiner.net.parameters(): p.add_(.01*torch.randn_like(p))
    model.eval()
    condition = dataset.condition[:2]
    a, info = RUNNER.canonical_sample(model, condition, dataset.ids[:2], 10, 77)
    b, _ = RUNNER.canonical_sample(model, condition.flip(0), dataset.ids[:2][::-1], 20, 77)
    assert torch.equal(a, b.flip(0)[:, :10])
    assert a.std(dim=1).mean() > 0
    assert info['network_calls'] > 0 and info['canonical_member_batch'] == 8
    assert info['dummy_members_per_domain'] == 6
    small, _ = RUNNER.canonical_sample(model, condition[:1], dataset.ids[:1], 50, 77)
    assert torch.equal(a[:1], small[:, :10])


@pytest.mark.parametrize('head', REFINERS)
def test_prediction_artifacts_resumed_at_date_boundary_preserve_every_member(tmp_path, head):
    dataset = MemoryPrepared()
    config = cfg(head); recipe = experiment_recipes(config, ('pr', 'tasmax'))['native']
    model, target, _ = RUNNER.initialize_candidate(config, recipe, 'direct', dataset, device='cpu')
    kwargs = dict(model=model, target=target, formulation='direct', dataset=dataset, variables=recipe.variables,
                  members=3, seed=31, training_seed=101, strategy='mean_preserving', device='cpu')
    complete = tmp_path/'complete.h5'
    reference, artifact = RUNNER.predict_dataset(**kwargs, output_path=complete)
    partial = tmp_path/'resumed.h5'
    calls = [0]
    def interrupt():
        calls[0] += 1
        return calls[0] == 3
    with pytest.raises(InterruptedError):
        RUNNER.predict_dataset(**kwargs, output_path=partial, interrupt_check=interrupt)
    resumed, _ = RUNNER.predict_dataset(**kwargs, output_path=partial, resume=True)
    with h5py.File(complete) as a, h5py.File(partial) as b:
        for name in ('raw_members', 'processed_members', 'baseline', 'truth', 'valid'):
            assert np.array_equal(a[name][:], b[name][:]), name
        assert np.array_equal(a['baseline'][:], dataset.base.numpy())
        assert 'normalized_endpoints' not in a
        for date in RUNNER.diagnostic_dates(dataset.dates):
            for field in ('normalized_endpoints', 'ungated_residual_or_support_latent', 'raw_corrections', 'constraint_adjustment'):
                assert np.array_equal(a[f'diagnostics/{date}/{field}'][:], b[f'diagnostics/{date}/{field}'][:])
    assert all(len(d) == 10 for d in resumed['pr']['metadata']['date_ids'])
    assert set(artifact['auxiliary_metrics']['pr']) == {'p99_absolute_error', 'observed_p99_event_rmse'}
    assert np.array_equal(resumed['pr']['products']['refined']['squared_error_sum'], reference['pr']['products']['refined']['squared_error_sum'])
    replay, _ = RUNNER.predict_dataset(**kwargs, output_path=partial, resume=True)
    assert replay['pr']['metadata']['date_ids'] == resumed['pr']['metadata']['date_ids']


def fixture_train_args(tmp_path, monkeypatch, head, output):
    fit, valid = MemoryPrepared(), MemoryPrepared(validation=True)
    config_path = tmp_path/'case.yaml'
    config_path.write_text(yaml.safe_dump({'refinement': cfg(head).to_dict()}))
    cache_path = tmp_path/'cache.h5'
    cache_path.with_suffix('.manifest.json').write_text(json.dumps({'variables': ['pr', 'tasmax'], 'cache_sha256': 'source', 'phase1_checkpoint_sha256': 'phase1file'}))
    original_plan, _, contract = RUNNER._load_registered_plan(ARTIFACT/'registered_experiment_plan.json', ARTIFACT/'frozen_scientific_acceptance_v2.json')
    contract = copy.deepcopy(contract)
    contract['regions']['widths'] = [1, 2, 4]
    plan = copy.deepcopy(original_plan)
    plan['stopping']['screen'].update(maximum_epochs=2, minimum_epochs=2, assess_every_epochs=2, patience_assessments=4)
    plan['optimization']['screen_batch_size'] = 2
    plan['validation']['selection_members'] = 3
    # This miniature plan exists only inside the regression fixture, never a real experiment.
    monkeypatch.setattr(RUNNER, '_load_registered_plan', lambda *args: (plan, 'test_fixture_plan', contract))
    monkeypatch.setattr(RUNNER, 'prepare_conditioning', lambda cache, split, *args, **kwargs: valid if split == 'screen_validation' else fit)
    return RUNNER.build_parser().parse_args(['train', '--config', str(config_path), '--cache', str(cache_path),
        '--plan', str(ARTIFACT/'registered_experiment_plan.json'), '--contract', str(ARTIFACT/'frozen_scientific_acceptance_v2.json'),
        '--output', str(output), '--head', head, '--stage', 'screen', '--device', 'cpu'])


@pytest.mark.parametrize('head', REFINERS)
def test_complete_training_and_exact_optimizer_rng_resume(tmp_path, monkeypatch, head):
    args = fixture_train_args(tmp_path, monkeypatch, head, tmp_path/'uninterrupted')
    result = RUNNER.train(args)
    assert result['updates'] == 4 and result['epoch'] == 2
    original = RUNNER.training_result
    calls = [0]
    def interrupt(model, target, formulation, batch, generator):
        calls[0] += 1
        if calls[0] == 2: raise InterruptedError('simulated resource pause before second optimizer update')
        return original(model, target, formulation, batch, generator)
    monkeypatch.setattr(RUNNER, 'training_result', interrupt)
    args.output = str(tmp_path/'resumed')
    paused = RUNNER.train(args)
    assert paused['updates'] == 1 and paused['cursor'] == 2
    assert paused['status'] == 'INCONCLUSIVE_RESOURCE_INTERRUPTION'
    monkeypatch.setattr(RUNNER, 'training_result', original)
    args.resume = str(tmp_path/'resumed'/'last.ckpt')
    finished = RUNNER.train(args)
    assert finished['updates'] == 4 and finished['epoch'] == 2
    a = torch.load(tmp_path/'uninterrupted'/'last.ckpt', weights_only=False)
    b = torch.load(tmp_path/'resumed'/'last.ckpt', weights_only=False)
    for name, value in a['candidate']['state_dict'].items(): assert torch.equal(value, b['candidate']['state_dict'][name]), name
    assert torch.equal(a['process_rng'], b['process_rng'])
    assert torch.equal(a['order_rng'], b['order_rng'])
    assert a['scheduler'] == b['scheduler']
    assert (tmp_path/'resumed'/'nearest_provisional_candidate.ckpt').is_file()
    assert not (tmp_path/'resumed'/'best_scientific_candidate.ckpt').exists()

def test_selected_mean_has_authenticated_crossfit_trigger_and_can_initialize_remainder(tmp_path, monkeypatch):
    args = fixture_train_args(tmp_path, monkeypatch, 'flow_matching_unet', tmp_path/'mean')
    args.formulation = 'deterministic_mean'; args.head = 'deterministic_mean'
    RUNNER.train(args)
    selected = tmp_path/'mean'/'nearest_provisional_candidate.ckpt'
    sidecar = json.loads(selected.with_suffix('.mean_diagnostics.json').read_text())
    assert sidecar['checkpoint_sha256'] == RUNNER.file_sha256(selected)
    assert sidecar == json.loads((tmp_path/'mean'/'mean_crossfit_assessment.json').read_text())
    assert set(sidecar['fitting_to_validation_rmse_ratio']) == {'pr', 'tasmax'}
    assert sidecar['crossfit_required'] is False
    config = cfg('flow_matching_unet')
    recipe = experiment_recipes(config, ('pr', 'tasmax'))['native']
    _, target, _ = RUNNER.initialize_candidate(config, recipe, 'mean_remainder', MemoryPrepared(), device='cpu', mean_checkpoint=selected)
    assert target.normalizer.is_fitted
    sidecar['crossfit_required'] = True
    selected.with_suffix('.mean_diagnostics.json').write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match='cross-fitted control'):
        RUNNER.initialize_candidate(config, recipe, 'mean_remainder', MemoryPrepared(), device='cpu', mean_checkpoint=selected)


def test_prepared_conditioning_shares_all_heads_and_never_calls_phase1(tmp_path, monkeypatch):
    from refinement_fixtures import TinyPhase1
    from granitewxc.refinement.scientific_data import CACHE_VERSION, registered_splits
    from granitewxc.utils import config as config_module
    from granitewxc.models import model as model_module
    phase1 = TinyPhase1(out_channels=2, scaling_codes=(1, 0), nonneg=(True, False)).eval()
    checkpoint = tmp_path/'phase1.ckpt'
    torch.save(phase1.state_dict(), checkpoint)
    data = MemoryPrepared()
    x = data.condition[:2]
    with torch.no_grad(): physical, normalized = phase1({'x': x}, return_pre_inverse=True)
    raw = {'data': {'scalers': {}, 'output_vars': ['pr', 'tasmax'], 'input_vars': ['a', 'b', 'c', 'd']}}
    config_path = tmp_path/'case.yaml'; config_path.write_text(yaml.safe_dump(raw))
    path = tmp_path/'phase1.h5'
    dates = ['1961-01-03T12:00:00', '1977-01-07T12:00:00']
    with h5py.File(path, 'w') as f:
        group = f.create_group('fields')
        for name, value in {'x': x, 'y': data.truth[:2], '__phase1_physical': physical, '__phase1_normalized': normalized, '__target_valid_mask': data.valid[:2]}.items():
            group.create_dataset(name, data=value.numpy())
        f.create_dataset('latitude', data=data.latitude); f.create_dataset('longitude', data=data.longitude)
    manifest = {'version': CACHE_VERSION, 'complete': True, 'cache_sha256': RUNNER.file_sha256(path),
        'phase1_checkpoint_sha256': RUNNER.file_sha256(checkpoint), 'phase1_checkpoint': str(checkpoint),
        'phase1_state_fingerprint': RUNNER.phase1_state_fingerprint(phase1.state_dict()),
        'source_case': 'test', 'splits': registered_splits(dates), 'config': raw, 'scalers': {},
        'variables': ['pr', 'tasmax'], 'units': {'pr': 'mm/day', 'tasmax': 'K'}, 'calendar': 'noleap'}
    path.with_suffix('.manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(config_module, 'get_config', lambda path: None)
    constructions = [0]
    def make_model(config):
        constructions[0] += 1
        return copy.deepcopy(phase1)
    monkeypatch.setattr(model_module, 'get_finetune_model_UNET', make_model)
    first = RUNNER.prepare_conditioning(path, 'screen_fit', cfg('diffusion_unet'), config_path, tmp_path/'prepared', device='cpu')
    second = RUNNER.prepare_conditioning(path, 'screen_fit', cfg('flow_matching_transformer'), config_path, tmp_path/'prepared', device='cpu')
    assert first.path == second.path and constructions[0] == 1
    assert first.manifest['phase1_forward_calls'] == 0
    assert torch.equal(first[0]['baseline'], physical[0])
    assert torch.equal(first[0]['truth'], data.truth[0])
    assert first[0]['conditioning'].shape == (7, 9, 11)  # two baseline + four predictors + baseline-validity mask

def test_cache_setup_rng_consumption_does_not_change_fresh_weights(tmp_path, monkeypatch):
    args = fixture_train_args(tmp_path, monkeypatch, 'diffusion_unet', tmp_path/'cache_miss')
    prepare = RUNNER.prepare_conditioning
    def noisy_prepare(*a, **kw):
        torch.randn(137)
        return prepare(*a, **kw)
    monkeypatch.setattr(RUNNER, 'prepare_conditioning', noisy_prepare)
    monkeypatch.setattr(RUNNER, 'training_result', lambda *a, **kw: (_ for _ in ()).throw(InterruptedError('initial-state-only test')))
    RUNNER.train(args)
    monkeypatch.setattr(RUNNER, 'prepare_conditioning', prepare)
    args.output = str(tmp_path/'cache_hit')
    RUNNER.train(args)
    a = torch.load(tmp_path/'cache_miss'/'last.ckpt', weights_only=False)
    b = torch.load(tmp_path/'cache_hit'/'last.ckpt', weights_only=False)
    for key, value in a['candidate']['state_dict'].items(): assert torch.equal(value, b['candidate']['state_dict'][key]), key
    assert torch.equal(a['global_rng']['torch'], b['global_rng']['torch'])
    assert torch.equal(a['process_rng'], b['process_rng'])
    assert torch.equal(a['order_rng'], b['order_rng'])


def test_fold_option_rejects_unregistered_or_wrong_formulation(tmp_path, monkeypatch):
    args = fixture_train_args(tmp_path, monkeypatch, 'diffusion_unet', tmp_path/'fold')
    args.fit_fold = 0
    with pytest.raises(ValueError, match='preregistered deterministic mean folds'):
        RUNNER.train(args)


def test_changed_algorithm_source_rejects_predict_before_reading_data(tmp_path, monkeypatch):
    payload = {'kind': RUNNER.RUNNER_VERSION, 'run_contract': {'source_code_fingerprints': {}}, 'refinement_config': {}}
    payload['run_contract_sha256'] = RUNNER.config_fingerprint(payload['run_contract'])
    path = tmp_path/'old.ckpt'; torch.save(payload, path)
    with pytest.raises(ValueError, match='source code changed'):
        RUNNER.predict(SimpleNamespace(checkpoint=path, device='cpu'))

@pytest.mark.parametrize('head', REFINERS)
def test_crossfit_runner_fitting_evaluation_sampling_and_checkpoint(tmp_path, head):
    from test_refinement_crossfit import fixture, registration_kwargs, DATES
    from granitewxc.refinement.crossfit import make_crossfit_registration
    from granitewxc.refinement.mean_correction import SignedResidualMeanCorrector
    _, base, condition, truth, valid, _, _ = fixture()
    class CrossPrepared(MemoryPrepared):
        def __len__(self): return len(self.dates)
    data = CrossPrepared()
    data.base, data.condition, data.truth, data.valid = base, condition, truth, valid
    data.dates = DATES; data.ids = ['case|'+d for d in DATES]
    data.manifest['variables'] = ['pr', 'tasmax', 'tasmin']
    data.manifest['nonnegative'] = [True, False, False]
    provenance = RUNNER.provenance_for(data)
    kwargs = registration_kwargs()
    kwargs['mean_provenance'] = [{**provenance, 'training_selection_fingerprint': f'fold-{i}'} for i in range(2)]
    registration = make_crossfit_registration(**kwargs)
    paths = []
    for i, correction in enumerate(([2., -1., .5], [-4., 3., -1.5])):
        mean = SignedResidualMeanCorrector(4, data.manifest['variables'], hidden_channels=4, depth=1)
        mean.fit_scaling(torch.tensor([-1., 1.]).reshape(2, 1, 1, 1).expand(2, 3, 3, 5), split='train', provenance=kwargs['mean_provenance'][i])
        with torch.no_grad(): mean.net[-1].bias.copy_(torch.tensor(correction)/mean.scaling.scale.float().reshape(-1))
        path = tmp_path/f'mean{i}.ckpt'; torch.save(mean.checkpoint(), path); paths.append(path)
    planpath = tmp_path/'crossfit.json'; planpath.write_text(json.dumps(registration))
    config = cfg(head); recipe = experiment_recipes(config, data.manifest['variables'])['native']
    model, target, provenance = RUNNER.initialize_candidate(config, recipe, 'crossfit_remainder', data,
        device='cpu', crossfit_registration=planpath, crossfit_mean_checkpoints=paths)
    batch = next(iter(torch.utils.data.DataLoader(data, batch_size=8)))
    z, _ = RUNNER.prepare_target('crossfit_remainder', target, batch)
    fitting, fitmean, _ = RUNNER.reconstruct_samples(model, target, 'crossfit_remainder', batch, z[:, None], data_scope='fitting_diagnostic')
    assert torch.allclose(fitting[:, 0][valid], truth[valid], atol=2e-5)
    external = {**batch, 'date': ['1977-01-01T12:00:00']*8}
    zeval, _ = RUNNER.prepare_target('crossfit_remainder', target, external)
    evaluated, evalmean, _ = RUNNER.reconstruct_samples(model, target, 'crossfit_remainder', external, zeval[:, None])
    assert torch.allclose(evaluated[:, 0][valid], truth[valid], atol=2e-5)
    assert not torch.equal(evalmean, fitmean)
    result = RUNNER.training_result(model, target, 'crossfit_remainder', batch, torch.Generator().manual_seed(55))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    result['loss'].backward(); optimizer.step()
    assert torch.isfinite(result['loss'])
    candidate, target_state = RUNNER.candidate_state(model, target, 'crossfit_remainder', provenance)
    restored_model, restored_target = RUNNER.restore_candidate({'formulation': 'crossfit_remainder',
        'candidate': candidate, 'target_state': target_state, 'provenance': provenance}, 'cpu')
    assert target.mean_fingerprint == restored_target.mean_fingerprint
    a, _ = RUNNER.canonical_sample(model, condition[:1], data.ids[:1], 3, 7)
    b, _ = RUNNER.canonical_sample(restored_model, condition[:1], data.ids[:1], 3, 7)
    assert torch.equal(a, b)


def test_mean_ranking_uses_actual_non_screening_training_seed(monkeypatch):
    _, _, contract = RUNNER._load_registered_plan(ARTIFACT/'registered_experiment_plan.json', ARTIFACT/'frozen_scientific_acceptance_v2.json')
    observed = []
    def compare(records, local):
        observed.append(list(records['deterministic_mean']))
        return [], {}
    monkeypatch.setattr(RUNNER, 'comparison_rows', compare)
    auxiliary = {v: {metric: {'per_seed': {'202': {'reference': 1., 'candidate': .9}}}
        for metric in ('p99_absolute_error', 'observed_p99_event_rmse')} for v in ('pr', 'tasmax')}
    result = RUNNER._mean_rank({}, contract, auxiliary, 202)
    assert observed == [['202']]
    assert result['score'][0] == 0



def test_registered_mean_pool_artifact_binds_reversed_selected_folds():
    path = ARTIFACT/'mean_fold_registration.json'
    registered = json.loads(path.read_text())
    means = []
    for pool in (1, 0):
        provenance = {'phase1_fingerprint': 'phase1', 'conditioning_fingerprint': 'condition',
                      'training_selection_fingerprint': f'pool{pool}-fitting-dates'}
        means.append({'kind': RUNNER.RUNNER_VERSION,
                      'candidate': {'contract': {'scaling_provenance': provenance}},
                      'run_contract': {'fit_fold': pool, 'mean_fold_registration': registered,
                                       'plan_sha256': registered['parent_experiment_plan_sha256']}})
    bound, _ = RUNNER.resolve_crossfit_registration(path, means)
    assert bound['heldout_years'][0] == registered['fold_training_years']['0']
    assert bound['mean_training_years'][0] == registered['fold_fit_years']['1']
    assert bound['mean_selection_years'][1] == registered['fold_selection_years']['0']
    assert bound['registration_id'].endswith('5b95dc8ee7c1e066e9fc012a2c7cf999cf18038a08c75577731ba22199d71a45')
    with pytest.raises(ValueError, match='pool1 then pool0'):
        RUNNER.resolve_crossfit_registration(path, list(reversed(means)))



def test_mean_exact_best_checkpoint_is_separate_from_material_patience_progress(tmp_path):
    source = tmp_path/'candidate.ckpt'; source.write_bytes(b'better-A')
    progress = {'best_score': [0, 1., 1.]}
    first = {'score': [0, .999, .5]}
    assert RUNNER.retain_best_mean_checkpoint(progress, first, source, tmp_path, {'model': 'A'})
    source.write_bytes(b'worse-B')
    assert not RUNNER.retain_best_mean_checkpoint(progress, {'score': [0, .9995, .4]}, source, tmp_path, {'model': 'B'})
    assert (tmp_path/'nearest_provisional_candidate.ckpt').read_bytes() == b'better-A'
    assert json.loads((tmp_path/'mean_crossfit_assessment.json').read_text()) == {'model': 'A'}
    assert progress['best_score'] == [0, 1., 1.]
    assert progress['best_mean_score'] == first['score']



def test_remainder_physical_control_is_fair_without_changing_signed_mean_target(tmp_path, monkeypatch):
    from granitewxc.refinement.mean_correction import SignedResidualMeanCorrector, FrozenMeanComposition
    data = MemoryPrepared()
    provenance = RUNNER.provenance_for(data)
    mean = SignedResidualMeanCorrector(4, ('pr', 'tasmax'), hidden_channels=4, depth=1)
    mean.fit_scaling(data.truth-data.base, data.valid, split='train', provenance=provenance)
    with torch.no_grad(): mean.net[-1].bias.copy_(torch.tensor([-3., 1.])/mean.scaling.scale.float().reshape(-1))
    target = FrozenMeanComposition(mean)
    target.fit_remainder_statistics(torch.utils.data.DataLoader(data, batch_size=4), split='train', provenance=provenance)
    config = cfg('flow_matching_unet')
    recipe = experiment_recipes(config, ('pr', 'tasmax'))['native']
    model = ExperimentalRefiner(config, recipe, cond_channels=4)
    calls = []
    original = RUNNER.collect_year_statistics
    def collect(*args, **kwargs):
        calls.append(kwargs['deterministic_control'].copy())
        return original(*args, **kwargs)
    monkeypatch.setattr(RUNNER, 'collect_year_statistics', collect)
    path = tmp_path/'remainder.h5'
    RUNNER.predict_dataset(model, target, 'mean_remainder', data, recipe.variables, output_path=path,
        members=3, seed=31, training_seed=101, strategy='mean_preserving', device='cpu')
    with h5py.File(path) as f:
        assert np.allclose(f['mean_control_unbounded'][:, 0], -1.)
        assert np.allclose(f['mean_control'][:, 0], 0.)
        assert np.array_equal(f['mean_control_unbounded'][:, 1], f['mean_control'][:, 1])
        assert np.allclose(f['mean_control'][:, 1], 271.)
    for i, control in enumerate(calls): assert np.allclose(control, 0. if i % 4 < 2 else 271.)
    batch = next(iter(torch.utils.data.DataLoader(data, batch_size=4)))
    prepared = target.prepare(batch['baseline'], batch['conditioning'], batch['truth'], batch['valid'])
    assert torch.allclose(prepared['mean_physical'][:, 0], torch.full_like(batch['baseline'][:, 0], -1.))



@pytest.mark.parametrize('head', REFINERS)
def test_pending_assessment_resume_preserves_subsequent_dropout_training(tmp_path, monkeypatch, head):
    args = fixture_train_args(tmp_path, monkeypatch, head, tmp_path/'reference')
    plan, digest, contract = RUNNER._load_registered_plan(args.plan, args.contract)
    plan['stopping']['screen'].update(maximum_epochs=3, minimum_epochs=3, assess_every_epochs=1)
    raw = yaml.safe_load(Path(args.config).read_text())
    raw['refinement']['unet']['dropout'] = .2
    raw['refinement']['transformer']['dropout'] = .2
    Path(args.config).write_text(yaml.safe_dump(raw))
    RUNNER.train(args)
    original = RUNNER.predict_dataset
    calls = [0]
    def pause(*a, **kw):
        calls[0] += 1
        if calls[0] == 2:
            torch.rand(17)  # An interrupted evaluation utility consumed global RNG.
            raise InterruptedError('interrupted pending validation assessment')
        return original(*a, **kw)
    monkeypatch.setattr(RUNNER, 'predict_dataset', pause)
    args.output = str(tmp_path/'resumed')
    result = RUNNER.train(args)
    assert result['pending_assessment'] and result['epoch'] == 1
    monkeypatch.setattr(RUNNER, 'predict_dataset', original)
    args.resume = str(tmp_path/'resumed'/'last.ckpt')
    RUNNER.train(args)
    a = torch.load(tmp_path/'reference'/'last.ckpt', weights_only=False)
    b = torch.load(tmp_path/'resumed'/'last.ckpt', weights_only=False)
    for name, value in a['candidate']['state_dict'].items(): assert torch.equal(value, b['candidate']['state_dict'][name]), name
    assert torch.equal(a['global_rng']['torch'], b['global_rng']['torch'])
    assert torch.equal(a['process_rng'], b['process_rng'])
    assert a['progress']['best_score'] == b['progress']['best_score']


def test_partial_optimizer_oom_preserves_previous_atomic_checkpoint(tmp_path, monkeypatch):
    args = fixture_train_args(tmp_path, monkeypatch, 'flow_matching_unet', tmp_path/'reference')
    RUNNER.train(args)
    original = torch.optim.AdamW.step
    pristine = []
    def broken_step(optimizer, *a, **kw):
        path = Path(args.output)/'last.ckpt'
        pristine.append(path.read_bytes())
        with torch.no_grad(): optimizer.param_groups[0]['params'][0].add_(99.)
        raise torch.cuda.OutOfMemoryError('simulated partial AdamW state mutation')
    monkeypatch.setattr(torch.optim.AdamW, 'step', broken_step)
    args.output = str(tmp_path/'resumed')
    result = RUNNER.train(args)
    last = tmp_path/'resumed'/'last.ckpt'
    assert result['status'] == 'INCONCLUSIVE_RESOURCE_INTERRUPTION'
    assert result['resume_policy'] == 'replay_from_previous_atomic_checkpoint_after_partial_optimizer_failure'
    assert last.read_bytes() == pristine[0]
    monkeypatch.setattr(torch.optim.AdamW, 'step', original)
    args.resume = str(last); RUNNER.train(args)
    a = torch.load(tmp_path/'reference'/'last.ckpt', weights_only=False)
    b = torch.load(last, weights_only=False)
    for name, value in a['candidate']['state_dict'].items(): assert torch.equal(value, b['candidate']['state_dict'][name]), name
    assert torch.equal(a['process_rng'], b['process_rng'])


def test_source_fingerprints_include_every_refinement_module_and_shared_evaluator():
    fingerprints = RUNNER.source_code_fingerprints()
    for path in (ROOT/'granitewxc'/'refinement').glob('*.py'):
        assert path.relative_to(ROOT).as_posix() in fingerprints
    assert 'examples/CORDEX_ML/utils/evaluate_refinement_outputs.py' in fingerprints
