"""Authenticated scientific HDF5 collection and paired diagnostic regression."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from examples.CORDEX_ML import cordex_scientific_evidence as E
from granitewxc.refinement.acceptance import AcceptanceError, load_contract, load_year_statistics, paired_resampling_plan
from granitewxc.refinement.experiments import experiment_recipes
from granitewxc.refinement.scientific_data import registered_splits
from test_cordex_scientific_experiments import MemoryPrepared, cfg, ARTIFACT


@pytest.fixture
def authenticated_artifact(tmp_path, monkeypatch):
    """Actual native CPU sampler/checkpoint; miniature years never acceptance."""
    torch.set_num_threads(1)
    fit, validation = MemoryPrepared(), MemoryPrepared(validation=True)
    contract = copy.deepcopy(load_contract(ARTIFACT/'frozen_scientific_acceptance_v2.json'))
    contract['regions']['widths'] = [1, 2, 4]
    contract['bootstrap']['replicates'] = 19
    contract['sampling']['minimum_members'] = 3
    plan = {'stopping': {'full': {'minimum_epochs': 20}}}
    monkeypatch.setattr(E.runner, '_load_registered_plan', lambda *args: (plan, 'fixture_plan', contract))
    config = cfg('flow_matching_unet')
    recipe = experiment_recipes(config, ('pr', 'tasmax'))['native']
    model, target, provenance = E.runner.initialize_candidate(config, recipe, 'direct', fit, device='cpu')
    phase1 = tmp_path/'phase1.ckpt'; phase1.write_bytes(b'fixture frozen Phase-1 source')
    cache = tmp_path/'cache.h5'
    dates = fit.dates + validation.dates
    with h5py.File(cache, 'w') as f:
        f.create_dataset('dates', data=np.asarray(dates, dtype=object), dtype=h5py.string_dtype())
        f.create_dataset('latitude', data=fit.latitude); f.create_dataset('longitude', data=fit.longitude)
        fields = f.create_group('fields')
        for key, a, b in (('__phase1_physical', fit.base, validation.base),
                          ('y', fit.truth, validation.truth),
                          ('__target_valid_mask', fit.valid, validation.valid)):
            fields.create_dataset(key, data=torch.cat((a, b)).numpy())
        f.attrs['completed_samples'] = len(dates)
    cache_hash = E.file_sha256(cache); phase1_hash = E.file_sha256(phase1)
    source = {'version': E.CACHE_VERSION, 'complete': True, 'cache_sha256': cache_hash,
              'phase1_checkpoint': str(phase1), 'phase1_checkpoint_sha256': phase1_hash,
              'phase1_frozen': True, 'phase1_eval': True, 'source_case': 'case',
              'variables': ['pr', 'tasmax'], 'units': {'pr': 'mm/day', 'tasmax': 'K'},
              'calendar': 'noleap', 'splits': registered_splits(dates)}
    E.write_json(cache.with_suffix('.manifest.json'), source)
    run = {'acceptance_contract_sha256': E.canonical_hash(contract), 'source_cache_sha256': cache_hash,
           'source_phase1_sha256': phase1_hash, 'plan_sha256': 'fixture_plan',
           'stage': 'screen', 'seed': 101, 'head': 'flow_matching_unet',
           'recipe': recipe.to_dict(), 'source_config': {'fixture': True},
           'refinement_config': config.to_dict(),
           'source_code_fingerprints': E.runner.source_code_fingerprints(),
           'fit_selection_sha256': E.canonical_sha256(fit.dates),
           'validation_selection_sha256': E.canonical_sha256(validation.dates)}
    candidate, state = E.runner.candidate_state(model, target, 'direct', provenance)
    checkpoint = tmp_path/'candidate.ckpt'
    payload = {'kind': E.runner.RUNNER_VERSION, 'run_contract': run,
               'run_contract_sha256': E.config_fingerprint(run), 'candidate': candidate,
               'target_state': state, 'refinement_config': config.to_dict(),
               'variables': ['pr', 'tasmax'], 'formulation': 'direct', 'provenance': provenance,
               'progress': {'epoch': 2}, 'plan_path': 'fixture', 'contract_path': 'fixture'}
    torch.save(payload, checkpoint)
    authentication = {'checkpoint_path': str(checkpoint), 'checkpoint_sha256': E.file_sha256(checkpoint),
                      'run_contract_sha256': payload['run_contract_sha256'],
                      'plan_sha256': 'fixture_plan', 'acceptance_contract_sha256': E.canonical_hash(contract),
                      'source_cache_sha256': cache_hash, 'phase1_checkpoint_sha256': phase1_hash}
    prediction = tmp_path/'prediction.h5'
    E.runner.predict_dataset(model, target, 'direct', validation, recipe.variables,
                             members=3, seed=31, training_seed=101, strategy='mean_preserving',
                             device='cpu', output_path=prediction, authentication=authentication,
                             data_scope='component_validation')
    return SimpleNamespace(path=prediction, cache=cache, contract=contract, source=source,
                           payload=payload, checkpoint=checkpoint, model=model, target=target,
                           validation=validation, authentication=authentication)


def rehash_prediction(path):
    manifest = E._json(path.with_suffix('.manifest.json'))
    manifest['sha256'] = E.file_sha256(path)
    E.write_json(path.with_suffix('.manifest.json'), manifest)


def test_authenticates_actual_arrays_and_screening_cannot_be_upgraded(authenticated_artifact):
    a = authenticated_artifact
    result = E.authenticate_prediction(a.path, a.cache, a.contract, chunk_time=2)
    assert result['authenticated']
    assert not result['full_training_verified']
    assert any('screening' in value for value in result['full_scope_limitations'])
    assert any('complete registered' in value for value in result['full_scope_limitations'])
    assert result['fit_years'] == []
    assert result['date_ids'] == [date[:10] for date in a.validation.dates]
    assert result['recipe_sha256'] and result['sampling_settings']
    assert all(len(value) == 64 for fields in result['field_sha256'].values() for value in fields.values())


@pytest.mark.parametrize('field', ['baseline', 'truth', 'valid'])
def test_rehashed_file_cannot_hide_mismatched_paired_arrays(authenticated_artifact, field):
    a = authenticated_artifact
    with h5py.File(a.path, 'r+') as f:
        value = f[field][0]
        value[0, 0, 0] = not value[0, 0, 0] if field == 'valid' else value[0, 0, 0] + 1.
        f[field][0] = value
    rehash_prediction(a.path)
    with pytest.raises(AcceptanceError, match='differs from the immutable'):
        E.authenticate_prediction(a.path, a.cache, a.contract)


def test_rehashed_file_rejects_nan_physical_member(authenticated_artifact):
    a = authenticated_artifact
    with h5py.File(a.path, 'r+') as f:
        f['raw_members'][0, 0, 0, 0, 0] = np.nan
    rehash_prediction(a.path)
    with pytest.raises(AcceptanceError, match='Nonfinite physical'):
        E.authenticate_prediction(a.path, a.cache, a.contract)


def test_stored_mean_must_average_physical_members(authenticated_artifact):
    a = authenticated_artifact
    with h5py.File(a.path, 'r+') as f:
        f['processed_ensemble_mean'][0, 0, 0, 0] += 1.
    rehash_prediction(a.path)
    with pytest.raises(AcceptanceError, match='physical-member reconstruction'):
        E.authenticate_prediction(a.path, a.cache, a.contract)


def test_exact_member_quantiles_wet_brier_and_constraint_denominators():
    truth = np.array([[[0., 1.], [2., 0.]]])
    baseline = truth + .5
    raw = np.stack([truth - .2, truth + .2, truth + .5], axis=1)
    processed = np.maximum(raw, 0)
    valid = np.ones_like(truth, bool); valid[0, 1, 1] = False
    masks = {'full': np.ones((2, 2), bool), 'west': np.array([[True, False], [True, False]])}
    values = E.probabilistic_constraint_diagnostics(raw, processed, baseline, truth, valid, masks,
                                                   variable='pr', threshold=.01, chunk_time=1)
    full = values['regions']['full']
    assert full['valid_cell_days'] == 3 and full['physical_member_observations'] == 9
    assert full['raw']['negative_fraction'] == pytest.approx(1/9)
    assert full['processed']['fraction_changed_from_raw'] == pytest.approx(1/9)
    probability = (processed >= .01).mean(axis=1)
    expected = ((probability - (truth >= .01))**2)[valid].mean()
    wet = full['wet_day']['0.01']
    assert wet['role'] == 'configured_primary'
    assert wet['processed']['brier_score'] == pytest.approx(expected)
    assert sum(item['count'] for item in wet['processed']['reliability']) == 3
    distribution = E.exact_distribution_diagnostics(raw, processed, baseline, truth, valid, masks)
    assert distribution['members']['processed'][0]['quantiles'] == pytest.approx(
        np.quantile(processed[:, 0][valid], [.01, .05, .5, .95, .99, .999]).tolist())


def test_extreme_draws_keep_year_fields_and_exact_linear_quantiles():
    contract = copy.deepcopy(load_contract(ARTIFACT/'frozen_scientific_acceptance_v2.json'))
    contract['bootstrap']['replicates'] = 19
    plan = paired_resampling_plan(contract)
    years = np.repeat(plan['years'], 3)
    truth = np.arange(len(years)*6, dtype=float).reshape(-1, 2, 3)/3
    baseline = truth + np.sin(truth)
    prediction = truth + .3*np.cos(truth)
    valid = np.ones_like(truth, bool)
    values = E.extreme_year_draws(prediction, baseline, truth, years, valid, plan)
    assert values['resampling_plan_sha256'] == plan['sha256']
    for draw in (0, 4, 18):
        indices = np.concatenate([np.tile(np.flatnonzero(years == year), int(weight))
                                  for year, weight in zip(plan['years'], plan['year_weights'][draw])])
        expected = E._quantile_metrics(prediction[indices].ravel(), baseline[indices].ravel(), truth[indices].ravel())
        assert values['year_draw_values']['p99_absolute_error']['candidate'][draw] == pytest.approx(abs(expected['p99_error_ensemble_mean']))
        assert values['year_draw_values']['observed_p99_event_rmse']['candidate'][draw] == pytest.approx(expected['observed_p99_event_rmse_ensemble_mean'])


def test_collect_and_merge_incomplete_evidence_never_certifies(authenticated_artifact, tmp_path):
    a = authenticated_artifact
    evidence = E.collect_prediction_evidence(a.path, a.cache, a.contract, tmp_path/'collected',
                                             chunk_time=2, make_plots=False)
    stats = load_year_statistics(evidence['records']['flow_matching_unet']['101']['tasmax']['path'])
    assert stats['metadata']['data_scope'] == 'screening'
    assert stats['metadata']['diagnostics']['wet_day_brier_reliability']['status'] == 'not_applicable'
    assert stats['metadata']['diagnostics']['nested_10_20_50_member_sensitivity']['status'] == 'missing'
    result = E.merge_prediction_evidence([tmp_path/'collected/evidence.json'], a.contract,
                                         tmp_path/'merged', make_plots=False)
    assert result['production_promotion'] is False
    assert result['status'] == 'INCONCLUSIVE'
    assert (tmp_path/'merged/acceptance_report/scientific_acceptance_results.json').exists()
    map_path = Path(evidence["diagnostic_outputs"]["pr"]["pr_numerical_maps"])
    with map_path.open("ab") as stream:
        stream.write(b"changed diagnostic bytes")
    with pytest.raises(AcceptanceError, match="diagnostic map"):
        E.merge_prediction_evidence([tmp_path/"collected/evidence.json"], a.contract,
                                    tmp_path/"tampered_merge", make_plots=False)


@pytest.mark.parametrize("formulation", ["mean_remainder", "crossfit_remainder"])
def test_mean_control_replay_checks_raw_and_physical_fields(tmp_path, formulation):
    dates = ['1977-01-01T12:00:00']
    path, prepared_path = tmp_path/'prediction.h5', tmp_path/'prepared.h5'
    condition = np.zeros((1, 2, 3, 4), np.float32)
    baseline = np.ones((1, 2, 3, 4), np.float32); baseline[:, 1] = 270
    correction = np.empty_like(baseline); correction[:, 0] = -2; correction[:, 1] = -.4
    raw = baseline + correction
    physical = raw.copy(); physical[:, 0] = 0
    with h5py.File(prepared_path, 'w') as f:
        f.create_dataset('dates', data=np.asarray(dates, dtype=object), dtype=h5py.string_dtype())
        f.create_dataset('conditioning', data=condition)
    with h5py.File(path, 'w') as f:
        f.create_dataset('dates', data=np.asarray(dates, dtype=object), dtype=h5py.string_dtype())
        for name, value in (('baseline', baseline), ('mean_control_unbounded', raw),
                            ('mean_control', physical), ('valid', np.ones_like(baseline, bool))):
            f.create_dataset(name, data=value)
        f.attrs['nonnegative_strategy'] = 'mean_preserving'
    digest = E.file_sha256(prepared_path)
    E.write_json(prepared_path.with_suffix('.manifest.json'),
                 {'complete': True, 'sha256': digest, 'identity': {'source_cache_sha256': 'source'},
                  'variables': ['pr', 'tasmax'], 'nonnegative': [True, False]})
    class FrozenMean(torch.nn.Module):
        def mean_corrector(self, cond):
            return torch.from_numpy(correction).expand(len(cond), -1, -1, -1)
        def inference_mean(self, base, cond):
            return base + self.mean_corrector(cond)
    manifest = {'prepared_cache_path': str(prepared_path), 'prepared_cache_sha256': digest}
    report = E.authenticate_mean_control(path, manifest, FrozenMean(), formulation,
                                        ['pr', 'tasmax'], {'cache_sha256': 'source'})
    assert report['maximum_absolute_roundoff'] == {'unbounded': 0., 'physical': 0.}
    with h5py.File(path, 'r+') as f:
        f['mean_control'][0, 0, 0, 0] = -1.
    with pytest.raises(AcceptanceError, match='physical control differs'):
        E.authenticate_mean_control(path, manifest, FrozenMean(), formulation,
                                    ['pr', 'tasmax'], {'cache_sha256': 'source'})


def test_nested_raw_fields_share_prefix_and_projection_stays_ensemble_specific(authenticated_artifact, tmp_path):
    a = authenticated_artifact
    primary = E.authenticate_prediction(a.path, a.cache, a.contract)
    cache_auth = E.authenticate_cache(a.cache)
    paths = []
    for count in (10, 20, 50):
        path = tmp_path/f'nested_{count}.h5'
        E.runner.predict_dataset(a.model, a.target, 'direct', a.validation, ('pr', 'tasmax'),
                                 members=count, seed=31, training_seed=101, strategy='mean_preserving',
                                 device='cpu', output_path=path, authentication=a.authentication,
                                 data_scope='nested_member_diagnostic')
        paths.append(path)
    report = E.nested_member_diagnostics(paths, primary, a.cache, a.contract, cache_auth,
                                         tmp_path/'nested.json')
    assert report['status'] == 'complete'
    saved = E._json(report['artifact'])
    assert saved['raw_prefix_equality'] is True
    assert saved['primary_raw_prefix_equality'] is True
    with h5py.File(a.path, 'r+') as f:
        original = f['raw_members'][0]
        changed = original.copy(); changed[0, 0, 0, 0] += .125
        f['raw_members'][0] = changed
        f['raw_ensemble_mean'][0] = changed.mean(0)
    rehash_prediction(a.path)
    changed_primary = E.authenticate_prediction(a.path, a.cache, a.contract)
    with pytest.raises(AcceptanceError, match='Primary and nested'):
        E.nested_member_diagnostics(paths, changed_primary, a.cache, a.contract, cache_auth,
                                    tmp_path/'primary_mismatch.json')
    with h5py.File(a.path, 'r+') as f:
        f['raw_members'][0] = original
        f['raw_ensemble_mean'][0] = original.mean(0)
    rehash_prediction(a.path)
    with h5py.File(paths[1], 'r+') as f:
        changed = f['raw_members'][0]
        changed[0, 0, 0, 0] += .25
        f['raw_members'][0] = changed
        f['raw_ensemble_mean'][0] = changed.mean(0)
    rehash_prediction(paths[1])
    with pytest.raises(AcceptanceError, match='prefix'):
        E.nested_member_diagnostics(paths, primary, a.cache, a.contract, cache_auth,
                                    tmp_path/'nested_bad.json')


def test_temperature_wet_diagnostic_not_applicable_requires_artifact(tmp_path):
    # Applicability is validated with the full synthetic acceptance fixture in
    # test_scientific_acceptance; this reducer must never manufacture wet metrics.
    truth = np.full((1, 2, 2), 273.)
    members = np.stack((truth-.5, truth+.5), axis=1)
    value = E.probabilistic_constraint_diagnostics(members, members, truth, truth,
        np.ones_like(truth, bool), {'full': np.ones((2, 2), bool)}, variable='tasmax', threshold=.01)
    assert value['regions']['full']['wet_day'] == {}

def test_extreme_seed_resampling_averages_metrics_before_relative_change():
    contract = copy.deepcopy(load_contract(ARTIFACT/"frozen_scientific_acceptance_v2.json"))
    contract["bootstrap"]["replicates"] = 19
    plan = paired_resampling_plan(contract)
    seeds = [str(seed) for seed in contract["training_seeds"]]
    per_seed = {}
    metric = "p99_absolute_error"
    for index, seed in enumerate(seeds):
        per_seed[seed] = {"year_draw_values": {metric: {
            "reference": (1 + index + np.arange(19)*.03).tolist(),
            "candidate": (.5 + index*.2 + np.arange(19)*.01).tolist()}}}
    actual = E.merge_extreme_seed_draws(per_seed, seeds, metric, plan, .001)
    for draw, indices in enumerate(plan["seed_indices"]):
        reference = np.mean([per_seed[seeds[i]]["year_draw_values"][metric]["reference"][draw] for i in indices])
        candidate = np.mean([per_seed[seeds[i]]["year_draw_values"][metric]["candidate"][draw] for i in indices])
        assert actual[draw] == pytest.approx((candidate-reference)/reference)

def test_full_fit_with_registered_subset_ranking_is_distinct_from_full_acceptance():
    from test_scientific_acceptance import dates_for_years
    contract = load_contract(ARTIFACT/"frozen_scientific_acceptance_v2.json")
    fit = [date+"T12:00:00" for date in dates_for_years(E.registered_years(contract, "component_fit"))]
    validation = [date+"T12:00:00" for date in dates_for_years(E.registered_years(contract))]
    selection = [date for date in validation if int(date[8:10]) in (7, 21)]
    source = {"calendar": "noleap", "splits": registered_splits(fit+validation)}
    plan = {"stopping": {"full": {"minimum_epochs": 20}},
            "validation": {"selection_day_of_month": [7, 21]}}
    run = {"stage": "full", "fit_selection_sha256": E.canonical_sha256(fit),
           "validation_selection_sha256": E.canonical_sha256(selection)}
    payload = {"progress": {"epoch": 20},
               "provenance": {"training_selection_fingerprint": E.canonical_sha256(fit)}}
    assert E.full_training_limitations(run, payload, plan, source, contract) == []
    # Full acceptance prediction coverage is checked independently against all
    # 2920 validation dates in authenticate_prediction, never this 192-date rank.
    assert len(selection) == 192 and len(validation) == 2920
    run["stage"] = "screen"
    assert any("screening" in reason for reason in E.full_training_limitations(run, payload, plan, source, contract))
    run["stage"] = "full"
    run["validation_selection_sha256"] = E.canonical_sha256(validation)
    assert any("ranking" in reason for reason in E.full_training_limitations(run, payload, plan, source, contract))


@pytest.mark.parametrize("completed", [True, False])
def test_earlier_selected_checkpoint_uses_authenticated_completed_run(tmp_path, completed):
    contract = load_contract(ARTIFACT/"frozen_scientific_acceptance_v2.json")
    run = {"head": "flow_matching_unet", "stage": "full", "seed": 101}
    run_hash = E.config_fingerprint(run)
    selected = tmp_path/"nearest_provisional_candidate.ckpt"
    payload = {"kind": E.runner.RUNNER_VERSION, "run_contract": run,
               "run_contract_sha256": run_hash, "progress": {"epoch": 5, "updates": 50}}
    torch.save(payload, selected)
    progress = {"epoch": 20, "updates": 200, "cursor": 0, "pending_assessment": False, "patience": 3,
                "status": "STOPPED_REGISTERED_PATIENCE_PROVISIONAL" if completed else "INCONCLUSIVE_RESOURCE_INTERRUPTION"}
    torch.save({**payload, "progress": progress}, tmp_path/"last.ckpt")
    E.write_json(tmp_path/"execution_status.json", progress)
    E.write_json(tmp_path/"scientific_selection.json", {
        "head": run["head"], "contract_sha256": E.canonical_hash(contract),
        "best_eligible": None, "nearest_provisional": {
            "selected_checkpoint": str(selected), "checkpoint_sha256": E.file_sha256(selected)}})
    plan = {"stopping": {"full": {"minimum_epochs": 20, "maximum_epochs": 100, "patience_assessments": 3}}}
    result = E.authenticate_completed_run(selected, payload, plan, contract)
    assert result["status"] == ("complete" if completed else "incomplete")
    assert result["selected_checkpoint_epoch"] == 5
    assert result["completed_run_epoch"] == 20 and result["completed_run_updates"] == 200
    progress["epoch"] = 21
    E.write_json(tmp_path/"execution_status.json", progress)
    with pytest.raises(AcceptanceError, match="differs from final checkpoint"):
        E.authenticate_completed_run(selected, payload, plan, contract)


def test_matched_family_maps_use_one_scale_set_without_combining_gates(tmp_path, monkeypatch):
    contract = load_contract(ARTIFACT/"frozen_scientific_acceptance_v2.json")
    family_paths = {}
    for group, magnitude in (("selected", 1.), ("control", 3.)):
        sources = []
        for head in contract["heads"]:
            for seed in contract["training_seeds"]:
                folder = tmp_path/group/head/str(seed); folder.mkdir(parents=True)
                products, digests = {}, {}
                for variable in contract["variables"]:
                    map_path = folder/f"{variable}.npz"
                    field = np.full((4, 4), 3. if variable == "pr" else 275.)
                    maps = {"ground_truth": field, "phase1": field+.5,
                            "ensemble_mean": field+magnitude,
                            "member_climatologies": np.stack((field+magnitude-.1, field+magnitude+.1)),
                            "phase1_bias": field*0+.5, "ensemble_mean_bias": field*0+magnitude,
                            "correction": field*0+magnitude-.5, "ensemble_spread": field*0+.2}
                    np.savez(map_path, lat=np.arange(4), lon=np.arange(4), **maps)
                    summary = folder/f"{variable}.json"
                    E.write_json(summary, {"variables": {variable: {"units": contract["units"][variable]}}})
                    products[variable] = {f"{variable}_numerical_maps": str(map_path), "summary_json": str(summary)}
                    digests[variable] = {name: E.file_sha256(path) for name, path in products[variable].items()}
                path = folder/"evidence.json"
                E.write_json(path, {"records": {head: {str(seed): {var: {} for var in contract["variables"]}}},
                    "diagnostic_outputs": products, "diagnostic_output_sha256": digests})
                sources.append({"path": str(path), "sha256": E.file_sha256(path)})
        family = tmp_path/f"{group}_merged.json"
        E.write_json(family, {"contract_sha256": E.canonical_hash(contract), "input_evidence": sources})
        family_paths[group] = family
    rendered = []
    def render(summary, maps, **kwargs):
        rendered.append((summary, kwargs["plot_limits"]))
        E.write_json(Path(kwargs["output_dir"])/"rendered.json", summary)
        return {}
    monkeypatch.setattr(E, "write_evaluation_outputs", render)
    output = tmp_path/"matched_selected_control_maps"
    report = E.compare_family_maps(family_paths["selected"], family_paths["control"], contract, output)
    assert report["source_count"] == 24 and report["acceptance_families_combined"] is False
    assert len(rendered) == 24
    assert all(limits["pr"]["difference"] == [-3., 3.] for summary, limits in rendered)
    assert {summary["comparison_group"] for summary, limits in rendered} == {"selected", "control"}
    assert (output/"selected"/contract["heads"][0]/"101"/"rendered.json").exists()
    assert E._json(output/"completion.json")["operation"] == "compare"
    source = E._json(family_paths["control"])["input_evidence"][0]["path"]
    map_path = E._json(source)["diagnostic_outputs"]["pr"]["pr_numerical_maps"]
    Path(map_path).write_bytes(b"modified map bytes")
    with pytest.raises(AcceptanceError, match="diagnostic file changed"):
        E.compare_family_maps(family_paths["selected"], family_paths["control"], contract, tmp_path/"bad_maps")
