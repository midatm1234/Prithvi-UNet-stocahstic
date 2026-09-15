"""Independent provenance controls for scientific data reuse."""
import json
import h5py
import numpy as np
import pytest
from granitewxc.refinement.scientific_data import (
    CACHE_VERSION, ScientificCacheDataset, canonical_sha256, file_sha256, registered_splits,
)

def test_registered_year_blocks_never_interleave_fitting_and_validation():
    dates=['1961-01-03T12:00:00','1976-12-27T12:00:00','1977-01-07T12:00:00',
           '1980-12-21T12:00:00','2080-01-11T12:00:00','2095-12-19T12:00:00',
           '2096-01-07T12:00:00','2099-12-21T12:00:00']
    result=registered_splits(dates)
    assert result['fit_indices']==[0,1,4,5]
    assert result['validation_indices']==[2,3,6,7]
    assert not set(result['fit_indices']) & set(result['validation_indices'])
    assert result['screen_fit_indices']==[0,1,4,5]
    assert result['screen_validation_indices']==[2,3,6,7]
    assert result['end_to_end_independent_status'].startswith('BLOCKED')

def test_duplicate_or_unregistered_dates_fail():
    with pytest.raises(ValueError,match='Duplicate'):
        registered_splits(['1961-01-01','1961-01-01','1977-01-01'])
    with pytest.raises(ValueError,match='outside'):
        registered_splits(['1961-01-01','1977-01-01','1981-01-01'])

@pytest.fixture
def cached(tmp_path):
    path=tmp_path/'phase1.h5'
    with h5py.File(path,'w') as f:
        fields=f.create_group('fields')
        fields.create_dataset('y',data=np.array([[[[0.,2.]]],[[[1.,3.]]]],np.float32))
        fields.create_dataset('__phase1_physical',data=np.array([[[[1.,4.]]],[[[2.,6.]]]],np.float32))
        fields.create_dataset('__phase1_normalized',data=np.zeros((2,1,1,2),np.float32))
        fields.create_dataset('__target_valid_mask',data=np.ones((2,1,1,2),bool))
    manifest={'version':CACHE_VERSION,'complete':True,'cache_sha256':file_sha256(path),
              'phase1_checkpoint_sha256':'fixed-phase1','source_case':'case',
              'splits':registered_splits(['1961-01-03T12:00:00','1977-01-07T12:00:00'])}
    path.with_suffix('.manifest.json').write_text(json.dumps(manifest))
    return path,manifest

def test_physical_zero_negative_correction_and_stable_identity_survive_cache(cached):
    path,_=cached
    d=ScientificCacheDataset(path,expected_phase1_sha256='fixed-phase1')
    b=d[0]
    assert b['__target_valid_mask'].all()
    assert b['y'][0,0,0].item()==0
    assert (b['y']-b['__phase1_physical']).tolist()==[[[-1.,-2.]]]
    assert b['__sample_id']=='case|1961-01-03T12:00:00'
    d.close()

def test_changed_cache_split_or_phase1_cannot_silently_load(cached):
    path,manifest=cached
    with pytest.raises(ValueError,match='different Phase-1'):
        ScientificCacheDataset(path,expected_phase1_sha256='wrong')
    manifest['splits']['fit_indices']=[1]
    path.with_suffix('.manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='split manifest changed'):
        ScientificCacheDataset(path)
    manifest['splits']=registered_splits(['1961-01-03','1977-01-07'])
    path.with_suffix('.manifest.json').write_text(json.dumps(manifest))
    with h5py.File(path,'r+') as f: f['fields/y'][0,0,0,0]=5
    with pytest.raises(ValueError,match='bytes changed'):
        ScientificCacheDataset(path)
