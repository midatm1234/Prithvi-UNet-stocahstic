"""Each refiner must fit signed two-variable smooth and edge corrections."""
import importlib.util
from pathlib import Path

import pytest

_SOURCE = Path(__file__).resolve().parents[1] / 'examples' / 'refinement_signed_field_validation.py'
_SPEC = importlib.util.spec_from_file_location('signed_field_validation', _SOURCE)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


@pytest.mark.parametrize('head', _MODULE.HEADS)
def test_two_variable_signed_edge_field_can_be_learned(head):
    result = _MODULE.run_head(head)
    assert all(value > 0 for value in result['initial_process_gradient_l2_by_variable'])
    for row in result['results']:
        label = (head, row['variable'], row['region'])
        assert row['after_mae'] < .6 * row['phase1_mae'], label
        assert row['after_rmse'] < .6 * row['phase1_rmse'], label
        assert row['after_mae'] < row['before_mae'], label
