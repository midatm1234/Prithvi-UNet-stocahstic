"""Explicit bridge from existing entry points to versioned scientific runs."""
from __future__ import annotations
from importlib import import_module

def dispatch_scientific(argv, *, expected_operation=None):
    """Return None for the preserved legacy CLI; dispatch explicit new requests."""
    args=list(argv)
    if not args or args[0] != '--scientific-experiment': return None
    if len(args)<2 or args[1] not in ('train','predict'):
        raise ValueError('--scientific-experiment requires train or predict followed by scientific runner arguments.')
    if expected_operation and args[1] != expected_operation:
        raise ValueError(f'This entry point requires scientific operation {expected_operation}.')
    result = import_module('examples.CORDEX_ML.cordex_scientific_experiments').main(args[1:])
    if isinstance(result, int):
        return result
    if isinstance(result, dict):
        status = str(result.get('status', ''))
        if status.startswith('INCONCLUSIVE_RESOURCE'):
            return 75
        if status.startswith('FAIL'):
            return 2
    return 0

