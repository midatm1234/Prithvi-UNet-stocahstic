"""Inference follows the authenticated selected candidate, not a filename guess."""
import hashlib
import json

import pytest

from examples.CORDEX_ML.cordex_scientific_prediction import resolve_selected_checkpoint


@pytest.mark.parametrize('eligible', [False, True])
def test_resolver_handles_eligible_only_or_provisional_only_and_detects_change(tmp_path, eligible):
    checkpoint = tmp_path/('best_scientific_candidate.ckpt' if eligible else 'nearest_provisional_candidate.ckpt')
    checkpoint.write_bytes(b'selected-weight-fixture')
    selected = {'selected_checkpoint': str(checkpoint), 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    key = 'best_eligible' if eligible else 'nearest_provisional'
    (tmp_path/'scientific_selection.json').write_text(json.dumps({key: selected}))
    (tmp_path/'execution_status.json').write_text(json.dumps({'status': 'REACHED_REGISTERED_MAXIMUM_PROVISIONAL'}))
    assert resolve_selected_checkpoint(tmp_path) == (checkpoint, key)
    checkpoint.write_bytes(b'changed')
    with pytest.raises(ValueError, match='bytes differ'):
        resolve_selected_checkpoint(tmp_path)


def test_unfinished_run_cannot_be_exported_as_completed_candidate(tmp_path):
    (tmp_path/'execution_status.json').write_text(json.dumps({'status': 'INCONCLUSIVE_RESOURCE_INTERRUPTION'}))
    with pytest.raises(ValueError, match='completed registered'):
        resolve_selected_checkpoint(tmp_path)
