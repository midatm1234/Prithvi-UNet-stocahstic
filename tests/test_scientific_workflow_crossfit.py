"""Conditional mean cross-fitting is explicit and never silently skipped."""
from pathlib import Path

from examples.CORDEX_ML.cordex_scientific_workflow import condition_decision, train_arguments, write
from test_scientific_workflow import queue


def mean_receipt(data, required):
    source = next(j for j in data['jobs'] if j['id'] == 'mean_control/deterministic_mean')
    write(Path(source['output'])/'execution_status.json', {'status': 'REACHED_REGISTERED_MAXIMUM_PROVISIONAL'})
    write(Path(source['output'])/'mean_crossfit_assessment.json', {'crossfit_required': required})


def test_untriggered_mean_uses_original_frozen_route_without_fitting_folds(queue):
    _, data = queue
    mean_receipt(data, False)
    for i in (0, 1):
        fold = next(j for j in data['jobs'] if j['id'] == f'mean_fold_{i}/deterministic_mean')
        assert condition_decision(dict(fold), data)['execute'] is False
    remainder = dict(next(j for j in data['jobs'] if j['id'] == 'mean_remainder/flow_matching_unet'))
    decision = condition_decision(remainder, data)
    assert decision['execute'] is True and not decision['crossfit_used']
    assert remainder['formulation'] == 'mean_remainder'
    assert '--mean-checkpoint' in train_arguments(remainder, data)


def test_trigger_requires_both_completed_folds_and_correct_heldout_order(queue):
    _, data = queue
    mean_receipt(data, True)
    remainder = dict(next(j for j in data['jobs'] if j['id'] == 'mean_remainder/diffusion_transformer'))
    assert condition_decision(remainder, data)['execute'] is None
    trained = []
    for i in (0, 1):
        fold = next(j for j in data['jobs'] if j['id'] == f'mean_fold_{i}/deterministic_mean')
        assert condition_decision(dict(fold), data)['execute'] is True
        folder = Path(fold['output']); folder.mkdir(parents=True)
        checkpoint = folder/'nearest_provisional_candidate.ckpt'; checkpoint.write_bytes(b'unit-controller-fixture')
        write(folder/'execution_status.json', {'status': 'STOPPED_REGISTERED_PATIENCE_PROVISIONAL'})
        trained.append(str(checkpoint))
        if i == 0:
            assert condition_decision(remainder, data)['execute'] is None
    assert condition_decision(remainder, data)['execute'] is True
    assert remainder['formulation'] == 'crossfit_remainder'
    assert remainder['crossfit_mean_checkpoints'] == trained[::-1]
    args = train_arguments(remainder, data)
    index = args.index('--crossfit-mean-checkpoints')
    assert args[index+1:index+3] == trained[::-1]
    assert '--mean-checkpoint' not in args
