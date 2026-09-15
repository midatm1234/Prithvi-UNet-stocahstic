"""Authenticated immutable Phase-1 fields and registered year-block datasets.

This cache stores the normal dataset's inference-time inputs, exact physical and
normalized Phase-1 outputs, and unchanged targets/masks. It does not store a
preselected model's conditioning, so controlled conditioning experiments can
reuse the identical physical baseline. Target values never choose the split.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

CACHE_VERSION = 'scientific-phase1-v1'
FIT_YEARS = tuple(range(1961, 1977)) + tuple(range(2080, 2096))
VALIDATION_YEARS = tuple(range(1977, 1981)) + tuple(range(2096, 2100))

def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def registered_splits(dates, *, fit_years=FIT_YEARS, validation_years=VALIDATION_YEARS):
    dates = [str(d) for d in dates]
    if len(set(dates)) != len(dates):
        raise ValueError('Duplicate sample timestamps cannot identify an immutable cache.')
    if set(fit_years) & set(validation_years):
        raise ValueError('Fitting and validation year blocks overlap.')
    years = np.array([int(d[:4]) for d in dates])
    fit = np.flatnonzero(np.isin(years, fit_years))
    validation = np.flatnonzero(np.isin(years, validation_years))
    if len(fit) == 0 or len(validation) == 0:
        raise ValueError('Both registered fitting and validation blocks are required.')
    uncovered = np.flatnonzero(~np.isin(years, tuple(fit_years) + tuple(validation_years)))
    if len(uncovered):
        raise ValueError('Cache contains dates outside registered development blocks.')
    # Deterministic seasonal screening; the fit/validation *split* is by year,
    # never randomly interleaved dates. Subsets cannot claim multiyear acceptance.
    def screen(indices, days):
        return [int(i) for i in indices if int(dates[i][8:10]) in days]
    result = {
        'fit_years': list(fit_years), 'validation_years': list(validation_years),
        'fit_indices': fit.tolist(), 'validation_indices': validation.tolist(),
        'screen_fit_indices': screen(fit, (3, 11, 19, 27)),
        'screen_validation_indices': screen(validation, (7, 21)),
        'dates': dates, 'scope': 'refinement_component_conditional_on_fixed_phase1',
        'phase1_has_seen_development_years': True,
        'historical_1981_2000': 'previously inspected development evidence',
        'end_to_end_independent_status': 'BLOCKED: independent target provenance not established',
    }
    result['sha256'] = canonical_sha256(result)
    return result

class ScientificCacheDataset(Dataset):
    """Read-only, per-worker HDF5 dataset with exact normal-model batch keys."""
    def __init__(self, cache_path, split='fit', *, expected_phase1_sha256=None,
                 authenticate=True, manifest_path=None):
        self.path = Path(cache_path).resolve()
        self.manifest_path = Path(manifest_path) if manifest_path else self.path.with_suffix('.manifest.json')
        self.manifest = json.loads(self.manifest_path.read_text(encoding='utf8'))
        if self.manifest.get('version') != CACHE_VERSION or not self.manifest.get('complete'):
            raise ValueError('Incomplete or unsupported scientific Phase-1 cache.')
        if authenticate and file_sha256(self.path) != self.manifest['cache_sha256']:
            raise ValueError('Scientific Phase-1 cache bytes changed after authentication.')
        if expected_phase1_sha256 and self.manifest['phase1_checkpoint_sha256'] != expected_phase1_sha256:
            raise ValueError('Cache belongs to a different Phase-1 checkpoint.')
        self.splits = self.manifest['splits']
        unsigned = {k: v for k, v in self.splits.items() if k != 'sha256'}
        if canonical_sha256(unsigned) != self.splits['sha256']:
            raise ValueError('Registered split manifest changed.')
        key = f'{split}_indices'
        if key not in self.splits:
            raise ValueError(f'Unregistered cache split: {split}')
        self.indices = list(self.splits[key])
        self.dates = [self.splits['dates'][i] for i in self.indices]
        self._file = None

    def __len__(self): return len(self.indices)
    def __getstate__(self):
        value = dict(self.__dict__); value['_file'] = None; return value
    def __getitem__(self, index):
        if self._file is None: self._file = h5py.File(self.path, 'r')
        i = self.indices[index]
        batch = {key: torch.from_numpy(np.asarray(value[i]).copy())
                 for key, value in self._file['fields'].items()}
        batch['__sample_timestamp'] = self.splits['dates'][i]
        batch['__sample_id'] = f"{self.manifest['source_case']}|{self.splits['dates'][i]}"
        batch['__sample_dataset_index'] = i
        return batch

    def close(self):
        if getattr(self, '_file', None) is not None: self._file.close(); self._file = None

    def __del__(self): self.close()
