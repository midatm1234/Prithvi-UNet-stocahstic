#!/usr/bin/env python
"""Registered CORDEX experimental fitting and immutable-cache inference.

Experimental checkpoint semantics are separate from production Phase 2. All
physical baseline tensors come from the authenticated frozen Phase-1 cache.
No Phase-1 forward is executed while preparing conditioning, fitting or sampling.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time
from types import SimpleNamespace

import hashlib
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from granitewxc.refinement.acceptance import (canonical_hash, load_contract, collect_year_statistics,
    merge_year_statistics, save_year_statistics, load_year_statistics, ScientificSelector, comparison_rows)
from granitewxc.refinement.checkpoint import save_checkpoint_atomic, load_phase1_state_dict, phase1_state_fingerprint
from granitewxc.refinement.config import resolve_refinement_config, ResidualNormalizationConfig, config_fingerprint
from granitewxc.refinement.experiments import (ExperimentalRefiner, ExperimentRecipe, experiment_recipes,
    TotalPrecipitationSupport, RainfallRegimes, gradient_diagnostics, select_channels)
from granitewxc.refinement.mean_correction import (SignedResidualMeanCorrector, FrozenMeanComposition,
    tensor_fingerprint, checked_provenance)
from granitewxc.refinement.crossfit import CrossFitMeanComposition, make_crossfit_registration
from granitewxc.refinement.normalization import ResidualNormalizer
from granitewxc.refinement.randomness import StableNoiseSource, STREAM_VERSION, stable_seed
from granitewxc.refinement.scientific_data import ScientificCacheDataset, canonical_sha256, file_sha256
from granitewxc.refinement.target_space import NormalizedTargetSpace

RUNNER_VERSION = 'registered_scientific_experiment_v1'
PREPARED_VERSION = 'frozen_phase1_inference_conditioning_v1'
HEADS = ('diffusion_unet', 'diffusion_transformer', 'flow_matching_unet', 'flow_matching_transformer')
CANONICAL_MEMBERS = 8


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')


def move(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def seed_everything(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def source_code_fingerprints():
    paths = [Path(__file__), *sorted((ROOT/'granitewxc'/'refinement').glob('*.py')),
             ROOT/'examples'/'CORDEX_ML'/'utils'/'evaluate_refinement_outputs.py']
    return {str(path.resolve().relative_to(ROOT)).replace('\\', '/'): file_sha256(path) for path in paths}


def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state['cuda'])


class PreparedDataset(Dataset):
    """Shared conditioning cache independent of head, geometry, loss and gate."""
    def __init__(self, path, *, authenticate=True):
        self.path = Path(path)
        self.manifest = json.loads(self.path.with_suffix('.manifest.json').read_text())
        if self.manifest.get('version') != PREPARED_VERSION or not self.manifest.get('complete'):
            raise ValueError('Incomplete prepared conditioning cache.')
        if authenticate and file_sha256(self.path) != self.manifest['sha256']:
            raise ValueError('Prepared conditioning cache bytes changed.')
        with h5py.File(self.path, 'r') as f:
            self.dates = f['dates'].asstr()[:].tolist()
            self.ids = f['ids'].asstr()[:].tolist()
            self.latitude, self.longitude = f['latitude'][:], f['longitude'][:]
        self._file = None
    def __len__(self): return len(self.dates)
    def __getitem__(self, index):
        if self._file is None: self._file = h5py.File(self.path, 'r')
        return {**{key: torch.from_numpy(self._file[key][index].copy()) for key in ('conditioning', 'baseline', 'truth', 'valid')},
                'date': self.dates[index], 'sample_id': self.ids[index]}
    def close(self):
        if getattr(self, '_file', None) is not None: self._file.close(); self._file = None
    def __del__(self): self.close()


def prepare_conditioning(cache_path, split, cfg, config_path, output_root, *, device='cpu'):
    """Strict-load Phase 1 once solely for the original conditioning/scaler API."""
    source = ScientificCacheDataset(cache_path, split=split)
    manifest = source.manifest
    actual_raw = yaml.safe_load(Path(config_path).read_text(encoding='utf8'))
    if actual_raw.get('predictands') != manifest['config'].get('predictands'):
        raise ValueError('Phase-1 target configuration differs from the cached baseline contract.')
    for name, expected in manifest['scalers'].items():
        current = actual_raw['data']['scalers'].get(name)
        if current is None or file_sha256(current) != expected['sha256']:
            raise ValueError(f'Conditioning scaler {name} differs from the authenticated Phase-1 cache.')
    for name in ('input_vars', 'input_levels', 'input_surface_vars', 'input_static_surface_vars', 'output_vars'):
        if actual_raw['data'].get(name) != manifest['config']['data'].get(name):
            raise ValueError(f'Input/variable ordering {name} differs from the cache.')
    identity = {'version': PREPARED_VERSION, 'source_cache_sha256': manifest['cache_sha256'],
                'phase1_fingerprint': manifest['phase1_state_fingerprint'], 'scalers': manifest['scalers'],
                'input_config': {k: manifest['config']['data'].get(k) for k in ('input_vars', 'input_levels', 'input_surface_vars', 'input_static_surface_vars', 'output_vars')},
                'conditioning': cfg.conditioning.to_dict(), 'split': split,
                'selected_dates_sha256': canonical_sha256(source.dates)}
    key = canonical_sha256(identity)
    destination = Path(output_root) / (key + '.h5')
    if destination.exists():
        dataset = PreparedDataset(destination)
        if dataset.manifest['identity'] != identity: raise ValueError('Prepared conditioning identity mismatch.')
        source.close()
        return dataset
    destination.parent.mkdir(parents=True, exist_ok=True)
    if cfg.conditioning.prithvi_features or cfg.conditioning.unet_features:
        raise ValueError('This immutable cache has no captured features; feature-conditioned experiments need a separately authenticated feature cache.')
    from granitewxc.utils.config import get_config
    from granitewxc.models.model import get_finetune_model_UNET
    from granitewxc.refinement.two_phase import build_two_phase_model
    config = get_config(config_path)
    wrapper = build_two_phase_model(get_finetune_model_UNET(config), {'refinement': cfg.to_dict()}).eval()
    checkpoint_path = Path(manifest['phase1_checkpoint'])
    if file_sha256(checkpoint_path) != manifest['phase1_checkpoint_sha256']:
        raise ValueError('Phase-1 checkpoint differs from the immutable cache reference.')
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    report = load_phase1_state_dict(wrapper, payload)
    del payload
    if phase1_state_fingerprint(wrapper.phase1.state_dict()) != manifest['phase1_state_fingerprint']:
        raise ValueError('Strict restored Phase-1 weights do not match cache state identity.')
    if wrapper.phase1.training or any(p.requires_grad for p in wrapper.phase1.parameters()):
        raise RuntimeError('Phase 1 must stay frozen and in evaluation mode.')
    # A forward hook makes accidental Phase-1 execution a hard error.
    guard = wrapper.phase1.register_forward_pre_hook(lambda *args: (_ for _ in ()).throw(RuntimeError('Phase-1 forward forbidden in scientific cache conditioning')))
    partial = destination.with_suffix('.partial.h5')
    metadata = {'version': PREPARED_VERSION, 'identity': identity, 'variables': manifest['variables'],
                'units': manifest['units'], 'calendar': manifest['calendar'], 'complete': False,
                'phase1_load_report': report.summary(), 'phase1_forward_calls': 0,
                'nonnegative': wrapper.phase1.predictand_nonneg_enabled_mask.cpu().reshape(-1).tolist()}
    with h5py.File(cache_path, 'r') as original, h5py.File(partial, 'r+' if partial.exists() else 'w') as output:
        if 'conditioning_identity' in output.attrs:
            if output.attrs['conditioning_identity'] != key: raise ValueError('Partial conditioning cache belongs to another input contract.')
            completed = int(output.attrs['completed_samples'])
        elif len(output):
            raise ValueError('Unversioned partial conditioning cache cannot be resumed silently.')
        else:
            completed = 0
            output.attrs.update(conditioning_identity=key, completed_samples=0)
            output.create_dataset('latitude', data=original['latitude'][:]); output.create_dataset('longitude', data=original['longitude'][:])
            output.create_dataset('dates', data=np.asarray(source.dates, dtype=object), dtype=h5py.string_dtype())
            output.create_dataset('ids', shape=(len(source),), dtype=h5py.string_dtype())
        offset = 0
        for batch in DataLoader(source, batch_size=32, shuffle=False, num_workers=0):
            end = offset + len(batch['y'])
            if end <= completed:
                offset = end
                continue
            if offset < completed: raise ValueError('Prepared-cache resume batch boundary changed.')
            batch = move(batch, device)
            with torch.inference_mode():
                condition = wrapper.build_conditioning(batch, batch['__phase1_normalized'])
                baseline, truth = batch['__phase1_physical'], batch['y']
                valid = torch.isfinite(baseline) & torch.isfinite(truth)
                if '__target_valid_mask' in batch: valid &= batch['__target_valid_mask'].bool()
            values = dict(conditioning=condition, baseline=baseline, truth=truth, valid=valid)
            end = offset + len(truth)
            for name, value in values.items():
                array = value.detach().cpu().numpy()
                if name not in output:
                    output.create_dataset(name, shape=(len(source), *array.shape[1:]), dtype=array.dtype,
                                          chunks=(1, *array.shape[1:]), compression='lzf', shuffle=True)
                output[name][offset:end] = array
            output['ids'][offset:end] = batch['__sample_id']
            offset = end
            output.attrs['completed_samples'] = end; output.flush()
        if offset != len(source): raise RuntimeError('Prepared cache did not cover the registered split.')
    guard.remove(); source.close(); del wrapper
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    partial.rename(destination)
    with h5py.File(destination, 'r') as completed_file:
        cond_channels = int(completed_file['conditioning'].shape[1])
    metadata.update(complete=True, sha256=file_sha256(destination), cond_channels=cond_channels)
    write_json(destination.with_suffix('.manifest.json'), metadata)
    return PreparedDataset(destination, authenticate=False)


def dataset_selection(dataset, dates):
    lookup = {date: i for i, date in enumerate(dataset.dates)}
    if len(set(dates)) != len(dates) or any(date not in lookup for date in dates):
        raise ValueError('Requested registered dates are duplicated or absent from the split.')
    subset = Subset(dataset, [lookup[d] for d in dates])
    subset.dates, subset.ids = list(dates), [dataset.ids[lookup[d]] for d in dates]
    subset.manifest, subset.latitude, subset.longitude = dataset.manifest, dataset.latitude, dataset.longitude
    return subset


def selected_batch(batch, source_variables, variables):
    return {**batch, **{key: select_channels(batch[key], source_variables, variables) for key in ('baseline', 'truth', 'valid')}}


def provenance_for(dataset):
    identity = dataset.manifest['identity']
    return {'phase1_fingerprint': identity['phase1_fingerprint'],
            'conditioning_fingerprint': canonical_sha256({k: v for k, v in identity.items() if k not in ('split', 'selected_dates_sha256')}),
            'training_selection_fingerprint': canonical_sha256(dataset.dates)}


def physical_postprocess(raw_members, nonnegative, strategy):
    """Call the production constraint ONCE on the entire requested ensemble."""
    holder = SimpleNamespace(output_scalers_sigma=torch.ones(1, raw_members.shape[2], 1, 1),
                             predictand_nonneg_enabled_mask=torch.tensor(nonnegative, dtype=torch.bool))
    return NormalizedTargetSpace(holder).constrain_physical_ensemble(raw_members, strategy=strategy)


@torch.no_grad()
def canonical_sample(model, conditioning, sample_ids, members, seed, *, num_steps=None,
                     callback=None, canonical_members=CANONICAL_MEMBERS):
    """Fixed member microbatch arithmetic; keyed dummy members complete a block.

    Raw first N members are identical for N=10/20/50 and domain order changes.
    Whole-ensemble constraints are applied later and need not be nested when
    their mathematics explicitly depend on ensemble size.
    """
    if members < 1 or canonical_members < 1 or len(sample_ids) != len(conditioning):
        raise ValueError('Invalid canonical sample identifiers/member counts.')
    outputs, calls = [], [0]
    hook = model.refiner.net.register_forward_hook(lambda *args: calls.__setitem__(0, calls[0]+1))
    if conditioning.device.type == 'cuda': torch.cuda.synchronize(conditioning.device)
    start = time.perf_counter()
    try:
        for index, sample_id in enumerate(sample_ids):
            chunks = []
            for first in range(0, members, canonical_members):
                indices = list(range(first, first+canonical_members))
                count = min(canonical_members, members-first)
                source = StableNoiseSource(seed, [sample_id], indices)
                cond = conditioning[index:index+1].expand(canonical_members, -1, -1, -1).contiguous()
                cb = None if callback is None else lambda stage, step, t, state, sid=sample_id, first=first, count=count: callback(sid, first, stage, step, t, state[:count].detach().clone())
                with model.refiner.use_noise_source(source):
                    z = model.sample(cond, num_steps=num_steps, trajectory_callback=cb)
                chunks.append(z[:count])
            outputs.append(torch.cat(chunks, dim=0))
    finally:
        hook.remove()
    if conditioning.device.type == 'cuda': torch.cuda.synchronize(conditioning.device)
    return torch.stack(outputs, dim=0), {'network_calls': calls[0], 'network_field_evaluations_including_dummy_members': calls[0]*canonical_members,
            'elapsed_seconds': time.perf_counter()-start, 'canonical_member_batch': canonical_members,
            'dummy_members_per_domain': math.ceil(members/canonical_members)*canonical_members-members,
            'random_stream_version': STREAM_VERSION}


def target_contract_for(formulation, target):
    if formulation in ('mean_remainder', 'crossfit_remainder', 'total_support'): return target.semantic_contract()
    return {'version': 'direct_signed_physical_residual_v1', 'normalizer_fingerprint': tensor_fingerprint(target), 'normalizer': target.metadata()}


def resolve_crossfit_registration(path, mean_payloads):
    """Bind the untouched preregistered fold pools to their selected mean states."""
    text = Path(path).read_text(encoding='utf8')
    registration = json.loads(text)
    means = [p['candidate'] if p.get('kind') == RUNNER_VERSION else p for p in mean_payloads]
    if registration.get('version') != 'registered_mean_year_folds_v1':
        return registration, means
    if not registration.get('registered_before_mean_fold_training') or not registration.get('registered_before_training'):
        raise ValueError('Mean fold pools must be registered before either mean was trained.')
    for payload, pool in zip(mean_payloads, (1, 0)):
        contract = payload.get('run_contract', {})
        if payload.get('kind') != RUNNER_VERSION or contract.get('fit_fold') != pool:
            raise ValueError('Cross-fit mean checkpoints must be selected runner checkpoints ordered pool1 then pool0.')
        if contract.get('mean_fold_registration') != registration or contract.get('plan_sha256') != registration['parent_experiment_plan_sha256']:
            raise ValueError('Selected fold mean does not authenticate the registered year pools and parent plan.')
    pools = [registration['fold_training_years'][str(i)] for i in (0, 1)]
    years = sorted(set(pools[0]) | set(pools[1]))
    bound = make_crossfit_registration(
        registration_id='registered_fold_artifact_sha256:'+hashlib.sha256(text.encode('utf8')).hexdigest(),
        fitting_years=years, heldout_years=pools,
        mean_training_years=[registration['fold_fit_years'][str(i)] for i in (1, 0)],
        mean_selection_years=[registration['fold_selection_years'][str(i)] for i in (1, 0)],
        mean_provenance=[m['contract']['scaling_provenance'] for m in means],
        era_years={'historical': [y for y in years if y < 2000], 'future': [y for y in years if y >= 2000]},
        selection_rule=registration['stopping'])
    return bound, means


def initialize_candidate(cfg, recipe, formulation, dataset, *, device, mean_checkpoint=None, crossfit_registration=None, crossfit_mean_checkpoints=None):
    source_variables = dataset.manifest['variables']
    variables = recipe.variables
    provenance = checked_provenance(provenance_for(dataset))
    cond_channels = dataset.manifest['cond_channels']
    if formulation == 'deterministic_mean':
        model = SignedResidualMeanCorrector(cond_channels, variables, variable_weights=recipe.variable_weights).to(device)
        model.scaling_provenance = provenance
        target = None
    else:
        model = ExperimentalRefiner(cfg, recipe, cond_channels=cond_channels).to(device)
        if formulation == 'crossfit_remainder':
            if crossfit_registration is None or crossfit_mean_checkpoints is None or len(crossfit_mean_checkpoints) != 2:
                raise ValueError('Cross-fit remainder needs its preregistration and two selected mean checkpoints.')
            mean_payloads = [torch.load(path, map_location='cpu', weights_only=False) for path in crossfit_mean_checkpoints]
            registration, means = resolve_crossfit_registration(crossfit_registration, mean_payloads)
            target = CrossFitMeanComposition(means, preregistration=registration).to(device)
            if tuple(target.variables) != tuple(variables): raise ValueError('Cross-fit mean/remainder variable order differs.')
            def remainder_batches():
                for batch in DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0):
                    batch = selected_batch(move(batch, device), source_variables, variables)
                    yield {**batch, 'sample_ids': batch['date']}
            target.fit_remainder_statistics(remainder_batches(), split='train', provenance=provenance)
            return model, target, provenance
        elif formulation == 'mean_remainder':
            if mean_checkpoint is None: raise ValueError('mean_remainder requires a selected frozen --mean-checkpoint.')
            mean_payload = torch.load(mean_checkpoint, map_location='cpu', weights_only=False)
            if mean_payload.get('kind') == RUNNER_VERSION: mean_payload = mean_payload['candidate']
            mean = SignedResidualMeanCorrector.from_checkpoint(mean_payload)
            diagnostic_path = Path(mean_checkpoint).with_suffix('.mean_diagnostics.json')
            if not diagnostic_path.exists():
                raise ValueError('Selected mean requires its fitting/validation remainder diagnostic sidecar.')
            diagnostics = json.loads(diagnostic_path.read_text())
            if diagnostics.get('mean_fingerprint') != mean.fingerprint() or diagnostics.get('checkpoint_sha256') != file_sha256(mean_checkpoint):
                raise ValueError('Selected mean diagnostic does not authenticate this checkpoint.')
            if diagnostics.get('crossfit_required') is not False:
                raise ValueError('Selected mean overfits fitting remainders; run the preregistered cross-fitted control before stochastic remainder fitting.')
            for key in ('phase1_fingerprint', 'conditioning_fingerprint'):
                if mean.scaling_provenance[key] != provenance[key]:
                    raise ValueError('Selected frozen mean uses different Phase-1 or inference conditioning.')
            if mean.variables != tuple(variables): raise ValueError('Mean/remainder variable order differs.')
            target = FrozenMeanComposition(mean).to(device)
            target.remainder_provenance = provenance
        elif formulation == 'total_support':
            if recipe.gate_policy != 'fixed_unit_fresh': raise ValueError('Total support needs a fresh fixed-unit-gate recipe.')
            target = TotalPrecipitationSupport(variables, tuple(v for v in variables if v == 'pr')).to(device)
            target.provenance = provenance
        else:
            target = ResidualNormalizer(len(variables), cfg.residual_normalization).to(device)
            if target.is_identity: raise ValueError('Fresh registered controls require fitted residual standardization.')
    # Streaming statistics only on the designated fitting dates.
    for batch in DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0):
        batch = selected_batch(move(batch, device), source_variables, variables)
        with torch.no_grad():
            residual = batch['truth'] - batch['baseline']
            if formulation == 'deterministic_mean': model.scaling.update(residual, batch['valid'])
            elif formulation == 'mean_remainder':
                mean = batch['baseline'] + target.mean_corrector(batch['conditioning'])
                target.normalizer.update(batch['truth']-mean, batch['valid'])
            elif formulation == 'total_support': target.normalizer.update(target.encode_physical(batch['truth'], batch['baseline'], batch['valid']), batch['valid'])
            else: target.update(residual, batch['valid'])
    if formulation == 'deterministic_mean': model.scaling.finalize()
    elif formulation in ('mean_remainder', 'crossfit_remainder', 'total_support'): target.normalizer.finalize()
    else: target.finalize()
    return model, target, provenance


def prepare_target(formulation, target, batch):
    if formulation == 'crossfit_remainder':
        in_fit = [int(date[:4]) in target.preregistration['fitting_years'] for date in batch['date']]
        if all(in_fit):
            prepared = target.prepare(batch['baseline'], batch['conditioning'], batch['truth'], batch['valid'], dateids=batch['date'])
            return prepared['remainder_normalized'], prepared['zero_anchor']
        if any(in_fit): raise ValueError('Do not mix OOF fitting and external validation target mappings in one batch.')
        prepared = target.prepare_evaluation(batch['baseline'], batch['conditioning'], batch['truth'], batch['valid'], split='validation')
        return prepared['remainder_normalized'], prepared['zero_anchor']
    if formulation == 'mean_remainder':
        prepared = target.prepare(batch['baseline'], batch['conditioning'], batch['truth'], batch['valid'])
        return prepared['remainder_normalized'], prepared['zero_anchor']
    if formulation == 'total_support':
        z = target.target(batch['truth'], batch['baseline'], batch['valid'])
        return z, target.normalizer.normalize(torch.zeros_like(z))
    residual = batch['truth']-batch['baseline']
    return target.normalize(residual, batch['valid']), target.normalize(torch.zeros_like(residual))


def training_result(model, target, formulation, batch, generator):
    if formulation == 'deterministic_mean':
        return model.training_loss(batch['conditioning'], batch['baseline'], batch['truth'], batch['valid'])
    z, zero = prepare_target(formulation, target, batch)
    return model.training_loss(z, batch['conditioning'], batch['valid'], generator=generator, zero_residual=zero)


def candidate_state(model, target, formulation, provenance):
    if formulation == 'deterministic_mean': return model.checkpoint(), None
    contract = target_contract_for(formulation, target)
    candidate = model.checkpoint(target_contract=contract, provenance=provenance)
    target_state = target.checkpoint() if formulation in ('mean_remainder', 'crossfit_remainder', 'total_support') else {k: v.detach().cpu().clone() for k, v in target.state_dict().items()}
    return candidate, target_state


def restore_candidate(payload, device):
    formulation = payload['formulation']
    if formulation == 'deterministic_mean':
        return SignedResidualMeanCorrector.from_checkpoint(payload['candidate'], expected_provenance=payload['provenance']).to(device), None
    if formulation == 'crossfit_remainder':
        target = CrossFitMeanComposition.from_checkpoint(payload['target_state'], expected_provenance=payload['provenance'], expected_preregistration=payload['target_state']['contract']['preregistration'])
    elif formulation == 'mean_remainder':
        target = FrozenMeanComposition.from_checkpoint(payload['target_state'], expected_provenance=payload['provenance'])
    elif formulation == 'total_support':
        target = TotalPrecipitationSupport.from_checkpoint(payload['target_state'], expected_provenance=payload['provenance'])
    else:
        cfg = resolve_refinement_config({'refinement': payload['refinement_config']})
        target = ResidualNormalizer(len(payload['variables']), cfg.residual_normalization)
        target.load_state_dict(payload['target_state'], strict=True)
    contract = target_contract_for(formulation, target)
    model = ExperimentalRefiner.from_checkpoint(payload['candidate'], expected_target_contract=contract, expected_provenance=payload['provenance'])
    return model.to(device), target.to(device)

def _select_overfit_dates(dataset, count_per_stratum):
    selected = []
    for future in (False, True):
        for months in ((12, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)):
            eligible = sorted(d for d in dataset.dates if (int(d[:4]) >= 2000) == future and int(d[5:7]) in months)
            if len(eligible) < count_per_stratum: raise ValueError('Insufficient registered overfit stratum dates.')
            selected.extend(eligible[:count_per_stratum])
    return sorted(selected)


def diagnostic_dates(dates):
    chosen = []
    for future in (False, True):
        for months in ((12, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)):
            stratum = sorted(d for d in dates if (int(d[:4]) >= 2000) == future and int(d[5:7]) in months)
            if stratum: chosen.extend((stratum[0], stratum[len(stratum)//2]))
    return sorted(set(chosen))


def fit_rainfall_regimes(dataset, *, seed):
    """At most one million uniform fitting cells; no changes to training draws."""
    variables = dataset.manifest['variables']
    if 'pr' not in variables: return None
    i = variables.index('pr')
    cells = dataset[0]['truth'][i].numel(); total = len(dataset)*cells
    positions = np.sort(np.random.default_rng(seed).choice(total, size=min(total, 1_000_000), replace=False))
    values = []
    for row in range(len(dataset)):
        left, right = np.searchsorted(positions, [row*cells, (row+1)*cells])
        if left == right: continue
        batch = dataset[row]
        local = torch.as_tensor(positions[left:right]-row*cells)
        truth, valid = batch['truth'][i].flatten()[local], batch['valid'][i].flatten()[local]
        values.append(truth[valid & torch.isfinite(truth)])
    selected = torch.cat(values)
    return RainfallRegimes.fit(selected, torch.ones_like(selected, dtype=torch.bool), split='train',
                               fitting_fingerprint=canonical_sha256(dataset.dates), wet_threshold=.01)


def heldout_loss(model, target, formulation, dataset, variables, *, device, seed):
    model.eval()
    sums = np.zeros(len(variables)); counts = np.zeros(len(variables))
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0):
            batch = selected_batch(move(batch, device), dataset.manifest['variables'], variables)
            generator = torch.Generator(device='cpu').manual_seed(stable_seed(seed, '|'.join(batch['sample_id']), 0, 999))
            result = training_result(model, target, formulation, batch, generator)
            loss = result['per_variable_loss'] if formulation == 'deterministic_mean' else result['per_variable_total']
            weight = batch['valid'].sum(dim=(0, 2, 3)).cpu().numpy()
            sums += loss.detach().cpu().numpy()*weight; counts += weight
    return {v: float(sums[i]/counts[i]) if counts[i] else None for i, v in enumerate(variables)}


def reconstruct_samples(model, target, formulation, batch, z, *, data_scope='screening'):
    if formulation == 'crossfit_remainder':
        parts = target.reconstruct(batch['baseline'], batch['conditioning'], z, model,
            dateids=batch['date'] if data_scope == 'fitting_diagnostic' else None)
        return parts['raw_members_physical'], parts['mean_physical'], parts['ungated_remainder_physical']
    if formulation == 'mean_remainder':
        parts = target.reconstruct(batch['baseline'], batch['conditioning'], z, model)
        return parts['raw_members_physical'], parts['mean_physical'], parts['ungated_remainder_physical']
    if formulation == 'total_support':
        parts = target.reconstruct(z, batch['baseline'], gate_policy=model.recipe.gate_policy)
        return parts['members_physical'], batch['baseline'], parts['unconstrained_latent']
    b, m, c, h, w = z.shape
    raw_correction = target.denormalize(z.reshape(b*m, c, h, w)).reshape(b, m, c, h, w)
    return batch['baseline'][:, None] + model.apply_correction_gate(raw_correction), batch['baseline'], raw_correction


@torch.no_grad()
def predict_dataset(model, target, formulation, dataset, variables, *, output_path, members,
                    seed, training_seed, strategy, device, data_scope='screening', num_steps=None, resume=False, interrupt_check=None, authentication=None):
    """Write all requested member fields; collect physical metrics additively."""
    if formulation == 'deterministic_mean':
        effective_steps, sampler_settings = 0, {'kind': 'deterministic_mean_no_stochastic_sampler'}
    else:
        effective_steps = num_steps if num_steps is not None else model.config.diffusion.inference_steps if model.config.is_diffusion else model.config.flow_matching.integration_steps
        if effective_steps < 1: raise ValueError('Sampling steps must be positive.')
        sampler_settings = (model.config.diffusion if model.config.is_diffusion else model.config.flow_matching).to_dict()
    prepared_source = dataset
    while hasattr(prepared_source, 'dataset'): prepared_source = prepared_source.dataset
    prepared_path = getattr(prepared_source, 'path', None)
    path = Path(output_path)
    identity = config_fingerprint({'model_state': tensor_fingerprint(model),
        'target': None if target is None else tensor_fingerprint(target), 'formulation': formulation,
        'prepared_sha256': dataset.manifest['sha256'], 'dates': dataset.dates, 'members': members,
        'seed': seed, 'variables': list(variables), 'strategy': strategy, 'num_steps': num_steps,
        'canonical_member_batch': CANONICAL_MEMBERS, 'authentication': authentication})
    if path.exists():
        if not resume: raise FileExistsError('Completed predictions are immutable; select a new path or explicitly resume.')
        manifest = json.loads(path.with_suffix('.manifest.json').read_text())
        if manifest.get('prediction_identity') != identity or file_sha256(path) != manifest['sha256']:
            raise ValueError('Completed prediction identity/bytes differ from requested replay.')
        records = {v: load_year_statistics(manifest['statistics'][f"{v}_{'raw' if formulation == 'deterministic_mean' else 'processed'}"]['path'], manifest['statistics'][f"{v}_{'raw' if formulation == 'deterministic_mean' else 'processed'}"]['sha256']) for v in variables}
        return records, {'path': str(path), 'statistics': manifest['statistics'], 'performance': manifest['performance'], 'auxiliary_metrics': manifest['auxiliary_metrics']}
    if path.with_suffix('.partial.h5').exists() and not resume:
        raise FileExistsError('Partial predictions require explicit resume.')
    path.parent.mkdir(parents=True, exist_ok=True)
    estimated_bytes = (2*len(dataset)+12*len(diagnostic_dates(dataset.dates)))*members*len(variables)*len(dataset.latitude)*len(dataset.longitude)*4
    if shutil.disk_usage(path.parent).free < estimated_bytes + 30*1024**3:
        raise InterruptedError('Insufficient disk for retained member diagnostics and 30 GiB reserve.')
    partial = path.with_suffix('.partial.h5')
    source_variables = dataset.manifest['variables']
    nonnegative = [bool(dataset.manifest['nonnegative'][source_variables.index(v)]) for v in variables]
    records, raw_records = {v: None for v in variables}, {v: None for v in variables}
    diagnostics = set(diagnostic_dates(dataset.dates))
    model.eval()
    totals = {'network_calls': 0, 'network_field_evaluations_including_dummy_members': 0, 'elapsed_seconds': 0.}
    correction_nonzero = {v: False for v in variables}
    with h5py.File(partial, 'r+' if partial.exists() else 'w') as output:
        if 'prediction_identity' in output.attrs:
            if output.attrs['prediction_identity'] != identity: raise ValueError('Partial prediction identity mismatch.')
            completed = int(output.attrs.get('completed_dates', 0))
            totals.update(json.loads(output.attrs.get('performance', '{}')))
        else:
            completed = 0
            output.attrs['prediction_identity'] = identity
            output.create_dataset('dates', data=np.asarray(dataset.dates, dtype=object), dtype=h5py.string_dtype())
            output.create_dataset('sample_ids', data=np.asarray(dataset.ids, dtype=object), dtype=h5py.string_dtype())
            output.create_dataset('latitude', data=dataset.latitude); output.create_dataset('longitude', data=dataset.longitude)
            output.attrs.update(formulation=formulation, variables=json.dumps(list(variables)), ensemble_size=members,
                                data_scope=data_scope, sampling_seed=seed, training_seed=training_seed,
                                canonical_member_batch=CANONICAL_MEMBERS, random_stream_version=STREAM_VERSION,
                                nonnegative_strategy=strategy, complete=False, effective_sampling_steps=effective_steps, sampler_settings=json.dumps(sampler_settings))
        for row, batch in enumerate(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)):
            batch = selected_batch(move(batch, device), source_variables, variables)
            date, sid = batch['date'][0], batch['sample_id'][0]
            def save_stage(sample_id, first, stage, step, process_time, state):
                if date not in diagnostics: return
                if 'initial' not in stage and 'final' not in stage and step not in (1, 10, 25, 50, 100, 200, 400): return
                group = output.require_group(f'trajectories/{date}/members_{first:04d}')
                name = f'{stage}_{step:04d}'
                if name not in group: group.create_dataset(name, data=state.cpu().numpy(), compression='lzf', shuffle=True)
                group[name].attrs['process_time'] = json.dumps(process_time.detach().cpu().reshape(-1).tolist())
            if interrupt_check is not None and interrupt_check():
                raise InterruptedError('Prediction interrupted at a completed date boundary.')
            if row < completed:
                raw = torch.from_numpy(output['raw_members'][row]).unsqueeze(0).to(device)
                mean = torch.from_numpy(output['mean_control_unbounded'][row]).unsqueeze(0).to(device)
                z = ungated = None
                if date in diagnostics:
                    z = torch.from_numpy(output[f'diagnostics/{date}/normalized_endpoints'][:]).unsqueeze(0).to(device)
                    ungated = torch.from_numpy(output[f'diagnostics/{date}/ungated_residual_or_support_latent'][:]).unsqueeze(0).to(device)
            elif formulation == 'deterministic_mean':
                mean = batch['baseline'] + model(batch['conditioning'])
                raw = mean[:, None].expand(-1, members, -1, -1, -1).clone()
                ungated = raw-batch['baseline'][:, None]
                z = ungated/model.scaling.scale.to(ungated).unsqueeze(1)
            else:
                z, performance = canonical_sample(model, batch['conditioning'], [sid], members, seed, num_steps=num_steps, callback=save_stage)
                for key in totals: totals[key] += performance[key]
                raw, mean, ungated = reconstruct_samples(model, target, formulation, batch, z, data_scope=data_scope)
            # No observation or observation-validity mask enters sampling/constraints.
            constrained = physical_postprocess(raw, nonnegative, strategy)
            processed = constrained['members']
            physical_mean_control = physical_postprocess(mean[:, None], nonnegative, strategy)['members'][:, 0]
            fields = {'raw_members': raw, 'processed_members': processed,
                      'baseline': batch['baseline'], 'truth': batch['truth'], 'valid': batch['valid'],
                      'mean_control': physical_mean_control, 'mean_control_unbounded': mean, 'raw_ensemble_mean': raw.mean(dim=1),
                      'processed_ensemble_mean': processed.mean(dim=1)}
            if date in diagnostics and row >= completed:
                group = output.require_group(f'diagnostics/{date}')
                group.attrs['sample_id'] = sid
                for name, value in {'normalized_endpoints': z, 'ungated_residual_or_support_latent': ungated,
                                    'raw_corrections': raw-batch['baseline'][:, None], 'constraint_adjustment': processed-raw}.items():
                    if name in group: del group[name]
                    group.create_dataset(name, data=value[0].cpu().numpy(), compression='lzf', shuffle=True)
            for key, tensor in fields.items():
                array = tensor[0].cpu().numpy()
                if key not in output:
                    output.create_dataset(key, shape=(len(dataset), *array.shape), dtype=array.dtype,
                                          chunks=(1, *array.shape), compression='lzf', shuffle=True)
                output[key][row] = array
            for i, variable in enumerate(variables):
                valid = batch['valid'][:, i].cpu().numpy()
                baseline = batch['baseline'][:, i].cpu().numpy(); truth = batch['truth'][:, i].cpu().numpy()
                correction_nonzero[variable] |= bool(np.any(valid & (raw[:, :, i].mean(1).cpu().numpy() != baseline)))
                metadata = {'data_scope': data_scope, 'date_ids': [date[:10]], 'calendar': dataset.manifest['calendar'],
                            'effective_sampled_correction': formulation != 'deterministic_mean' and correction_nonzero[variable],
                            'edge_fallback_or_suppression': False, 'training_seed': training_seed,
                            'units': dataset.manifest['units'].get(variable), 'formulation': formulation,
                            'mean_component_used': formulation in ('mean_remainder', 'crossfit_remainder'),
                            'deterministic_control_sha256': target.mean_fingerprint if formulation in ('mean_remainder', 'crossfit_remainder') else None,
                            'canonical_member_batch': CANONICAL_MEMBERS, 'random_stream_version': STREAM_VERSION}
                kwargs = dict(valid_mask=valid, metadata=metadata)
                if formulation in ('mean_remainder', 'crossfit_remainder'): kwargs['deterministic_control'] = physical_mean_control[:, i].cpu().numpy()
                statistics = collect_year_statistics(processed[:, :, i].cpu().numpy(), baseline, truth,
                    [int(date[:4])], dataset.latitude, dataset.longitude, **kwargs)
                raw_statistics = collect_year_statistics(raw[:, :, i].cpu().numpy(), baseline, truth,
                    [int(date[:4])], dataset.latitude, dataset.longitude, **kwargs)
                records[variable] = merge_year_statistics(records[variable], statistics)
                raw_records[variable] = merge_year_statistics(raw_records[variable], raw_statistics)
            output.attrs['completed_dates'] = max(completed, row+1)
            output.attrs['performance'] = json.dumps(totals); output.flush()
            if (row+1) % 16 == 0: print(f'sampling {row+1}/{len(dataset)} dates, {members} members', flush=True)
        output.attrs.update(complete=True, performance=json.dumps(totals))
    partial.rename(path)
    saved = {}
    for variable in variables:
        for label, record in (('processed', records[variable]), ('raw', raw_records[variable])):
            record['metadata'].update(date_ids=[d[:10] for d in dataset.dates], effective_sampled_correction=formulation != 'deterministic_mean' and correction_nonzero[variable])
            destination = path.with_name(f'{path.stem}_{variable}_{label}_year_statistics.npz')
            digest = save_year_statistics(destination, record)
            saved[f'{variable}_{label}'] = {'path': str(destination.resolve()), 'sha256': digest}
    auxiliary = prediction_extremes(path, variables, training_seed, product='raw_ensemble_mean' if formulation == 'deterministic_mean' else 'processed_ensemble_mean')
    write_json(path.with_suffix('.manifest.json'), {'version': RUNNER_VERSION, 'complete': True,
               'path': str(path.resolve()), 'sha256': file_sha256(path), 'statistics': saved,
               'performance': totals, 'members': members, 'seed': seed, 'variables': list(variables),
               'prediction_identity': identity, 'prepared_cache_sha256': dataset.manifest['sha256'],
               'prepared_cache_path': str(Path(prepared_path).resolve()) if prepared_path is not None else None, 'dates_sha256': canonical_sha256(dataset.dates),
               'scope': data_scope, 'formulation': formulation, 'production_accepted': False,
               'authentication': authentication, 'auxiliary_metrics': auxiliary,
               'stage_diagnostic_scope': 'registered_first_and_middle_dates_per_season_and_climate_regime',
               'diagnostic_dates': sorted(diagnostics), 'full_member_fields': ['raw_members', 'processed_members'],
               'full_signed_correction_definition': 'raw_members - baseline[:, None]',
               'deterministic_control_representations': {'mean_control_unbounded': 'signed physical baseline plus frozen mean correction used in target and reconstruction',
                   'mean_control': 'point prediction after the same variable physical postprocessing; probabilistic control reference'},
               'mean_component_used': formulation in ('mean_remainder', 'crossfit_remainder'),
               'deterministic_control_sha256': target.mean_fingerprint if formulation in ('mean_remainder', 'crossfit_remainder') else None,
               'effective_sampling_steps': effective_steps, 'sampler_settings': sampler_settings,
               'sampling_settings': {'effective_sampling_steps': effective_steps, **sampler_settings},
               'source_code_fingerprints': source_code_fingerprints(), 'candidate_state_fingerprint': tensor_fingerprint(model),
               'target_normalizer_fingerprint': None if target is None else target_contract_for(formulation, target)['normalizer_fingerprint']})
    return (raw_records if formulation == 'deterministic_mean' else records), {'path': str(path), 'statistics': saved, 'performance': totals, 'auxiliary_metrics': auxiliary}


def prediction_extremes(path, variables, seed, *, product='processed_ensemble_mean'):
    from utils.evaluate_refinement_outputs import _quantile_metrics
    result = {}
    with h5py.File(path, 'r') as f:
        for i, variable in enumerate(variables):
            truth = np.asarray(f['truth'][:, i], dtype=np.float64)
            baseline = np.asarray(f['baseline'][:, i], dtype=np.float64)
            prediction = np.asarray(f[product][:, i], dtype=np.float64)
            valid = f['valid'][:, i].astype(bool) & np.isfinite(truth) & np.isfinite(baseline)
            if np.any(valid & ~np.isfinite(prediction)): raise ValueError('Nonfinite predicted extremes on valid support.')
            metrics = _quantile_metrics(prediction[valid], baseline[valid], truth[valid])
            result[variable] = {
                'p99_absolute_error': {'per_seed': {str(seed): {
                    'reference': abs(metrics['p99_error_phase1']), 'candidate': abs(metrics['p99_error_ensemble_mean'])}}},
                'observed_p99_event_rmse': {'per_seed': {str(seed): {
                    'reference': metrics['observed_p99_event_rmse_phase1'], 'candidate': metrics['observed_p99_event_rmse_ensemble_mean']}}}}
    return result


def _mean_rank(records, contract, auxiliary, training_seed):
    local = copy.deepcopy(contract); local['heads'] = ['deterministic_mean']
    seed = str(training_seed)
    rows, _ = comparison_rows({'deterministic_mean': {seed: records}}, local)
    rows = [r for r in rows if r['metric'] not in ('crps', 'spread_to_phase1_mae', 'identical_fraction')]
    for variable, metrics in auxiliary.items():
        for metric, gate in (('p99_absolute_error', 'maximum_p99_absolute_error_relative_change'), ('observed_p99_event_rmse', 'maximum_observed_p99_event_rmse_relative_change')):
            item = metrics[metric]['per_seed'][seed]
            change = (item['candidate']-item['reference'])/max(abs(item['reference']), contract['near_zero_reference'][variable])
            rows.append({'variable': variable, 'region': 'full', 'metric': metric, 'change': change, 'limit': contract['gates'][gate]})
    finite = [r['change']-r['limit'] for r in rows if r['change'] is not None and math.isfinite(r['change'])]
    return {'score': [len(rows)-len(finite), max(finite) if finite else 1e30, float(np.mean(finite)) if finite else 1e30],
            'rows': rows, 'scope': 'deterministic_control_provisional_ranking', 'production_accepted': False,
            'scientifically_eligible_candidate': False}


def retain_best_mean_checkpoint(progress, decision, checkpoint_path, output, diagnostics):
    """Keep the exact lexicographic best separately from the patience threshold."""
    score = decision['score']
    previous = progress.get('best_mean_score')
    if previous is not None and tuple(score) >= tuple(previous): return False
    shutil.copy2(checkpoint_path, output/'nearest_provisional_candidate.ckpt')
    write_json(output/'nearest_provisional_candidate.mean_diagnostics.json', diagnostics)
    write_json(output/'mean_crossfit_assessment.json', diagnostics)
    progress['best_mean_score'] = copy.deepcopy(score)
    return True


def _load_registered_plan(path, contract_path):
    plan_text = Path(path).read_text(encoding='utf8')
    plan = json.loads(plan_text); digest = hashlib.sha256(plan_text.encode('utf8')).hexdigest()
    sidecar = Path(path).with_suffix('.sha256')
    if not sidecar.is_file() or sidecar.read_text().strip() != digest:
        raise ValueError('Registered experiment plan hash missing or changed.')
    contract = load_contract(contract_path)
    if plan.get('registered_before_training') is not True or canonical_hash(contract) != plan['acceptance_contract_sha256']:
        raise ValueError('Experiment plan does not reference this frozen scientific contract.')
    return plan, digest, contract


def _configuration(args):
    raw = yaml.safe_load(Path(args.config).read_text(encoding='utf8'))
    cfg = resolve_refinement_config(raw)
    if args.head in HEADS: cfg = replace(cfg, type=args.head, enabled=True, checkpoint=None)
    if args.alignment is not None:
        cfg = replace(cfg, unet=replace(cfg.unet, spatial_alignment=args.alignment), transformer=replace(cfg.transformer, spatial_alignment=args.alignment))
    normalized = args.normalize_predictors
    if normalized is None and args.formulation in ('deterministic_mean', 'mean_remainder', 'crossfit_remainder'): normalized = True
    if normalized is not None: cfg = replace(cfg, conditioning=replace(cfg.conditioning, normalize_predictors=normalized))
    return raw, cfg

def train(args):
    plan, plan_sha, contract = _load_registered_plan(args.plan, args.contract)
    if args.seed not in plan['training_seeds']: raise ValueError('Unregistered training seed.')
    if args.stage != 'full' and args.seed != plan['screening_seed']: raise ValueError('Screening/overfit uses the registered screening seed.')
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()) and args.resume is None:
        raise FileExistsError('An experiment directory already exists; explicitly resume it or select a new directory.')
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    raw, cfg = _configuration(args)
    source_manifest = json.loads(Path(args.cache).with_suffix('.manifest.json').read_text())
    variables = tuple(source_manifest['variables'])
    recipes = experiment_recipes(cfg, variables)
    if args.recipe not in recipes: raise ValueError(f'Unknown recipe; choose one of {list(recipes)}')
    recipe = recipes[args.recipe]; variables = recipe.variables
    if args.formulation == 'deterministic_mean' and args.head != 'deterministic_mean':
        raise ValueError('Label the deterministic control explicitly with --head deterministic_mean.')
    if args.formulation != 'deterministic_mean' and args.head not in HEADS:
        raise ValueError('A stochastic experiment requires one of the four registered heads.')
    train_split = 'fit' if args.stage == 'full' else 'screen_fit'
    prepared_root = Path(args.prepared_cache) if args.prepared_cache else Path(args.cache).parent/'prepared_conditioning'
    train_data = prepare_conditioning(args.cache, train_split, cfg, args.config, prepared_root, device=args.device)
    validation_data = prepare_conditioning(args.cache, 'screen_validation', cfg, args.config, prepared_root, device=args.device)
    overfit_registration = None
    if args.stage == 'overfit':
        registration_path = Path(args.overfit_registration or Path(args.plan).with_name('overfit_registration.json'))
        overfit_registration = json.loads(registration_path.read_text())
        if overfit_registration.get('parent_experiment_plan_sha256') != plan_sha or not overfit_registration.get('registered_before_overfit_training'):
            raise ValueError('Overfit selection was not registered against this plan.')
        train_data = dataset_selection(train_data, _select_overfit_dates(train_data, 2))
        validation_data = dataset_selection(validation_data, _select_overfit_dates(validation_data, 4))
        if len(train_data) != 16 or len(validation_data) != 32: raise ValueError('Registered overfit size mismatch.')
    fold_registration = None
    if args.fit_fold is not None:
        if args.formulation != 'deterministic_mean' or args.fold_registration is None:
            raise ValueError('--fit-fold is only for separately preregistered deterministic mean folds.')
        fold_registration = json.loads(Path(args.fold_registration).read_text())
        if fold_registration.get('parent_experiment_plan_sha256') != plan_sha or fold_registration.get('registered_before_mean_fold_training') is not True:
            raise ValueError('Mean fold selection is not preregistered against this experiment plan.')
        folds = fold_registration['fold_training_years']
        first, second = set(folds['0']), set(folds['1'])
        if first & second or first | second != set(plan['fitting']['years']):
            raise ValueError('Mean folds must partition all fitting years without overlap.')
        if any(not any(y < 2000 for y in fold) or not any(y >= 2000 for y in fold) for fold in (first, second)):
            raise ValueError('Every mean fitting fold must span historical and future climates.')
        pool = set(folds[str(args.fit_fold)])
        years = set(fold_registration['fold_fit_years'][str(args.fit_fold)])
        selection_years = set(fold_registration['fold_selection_years'][str(args.fit_fold)])
        if years & selection_years or years | selection_years != pool:
            raise ValueError('Fold fitting and internal selection must partition its fitting-year pool.')
        if any(not any(y < 2000 for y in group) or not any(y >= 2000 for y in group) for group in (years, selection_years)):
            raise ValueError('Mean fitting and internal selection each require historical and future years.')
        fold_source = train_data
        validation_data = dataset_selection(fold_source, [d for d in fold_source.dates if int(d[:4]) in selection_years])
        train_data = dataset_selection(fold_source, [d for d in fold_source.dates if int(d[:4]) in years])
    selection = {'fit_dates': train_data.dates, 'validation_dates': validation_data.dates,
                 'fit_sha256': canonical_sha256(train_data.dates), 'validation_sha256': canonical_sha256(validation_data.dates),
                 'overfit_registration': overfit_registration, 'mean_fold_registration': fold_registration, 'fit_fold': args.fit_fold}
    selection_path = out/'selected_dates_before_training.json'
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError('Previously registered selected dates changed.')
    write_json(selection_path, selection)
    run_contract = {'runner_version': RUNNER_VERSION, 'plan_sha256': plan_sha, 'acceptance_contract_sha256': canonical_hash(contract),
                    'source_cache_sha256': source_manifest['cache_sha256'], 'source_phase1_sha256': source_manifest['phase1_checkpoint_sha256'],
                    'prepared_fit_sha256': train_data.manifest['sha256'], 'prepared_validation_sha256': validation_data.manifest['sha256'],
                    'fit_selection_sha256': selection['fit_sha256'], 'validation_selection_sha256': selection['validation_sha256'],
                    'formulation': args.formulation, 'head': args.head, 'refinement_config': cfg.to_dict(),
                    'recipe': recipe.to_dict(), 'seed': args.seed, 'stage': args.stage, 'mean_fold_registration': fold_registration, 'fit_fold': args.fit_fold,
                    'canonical_member_batch': CANONICAL_MEMBERS, 'source_config': raw, 'source_code_fingerprints': source_code_fingerprints(),
                    'crossfit_registration': json.loads(Path(args.crossfit_registration).read_text()) if args.crossfit_registration else None,
                    'crossfit_registration_sha256': file_sha256(args.crossfit_registration) if args.crossfit_registration else None,
                    'crossfit_mean_checkpoint_sha256': [file_sha256(path) for path in args.crossfit_mean_checkpoints] if args.crossfit_mean_checkpoints else None}
    run_sha = config_fingerprint(run_contract)
    # Validate the stored run before writing metadata during resume.
    if (out/'run_contract.json').exists() and json.loads((out/'run_contract.json').read_text()).get('sha256') != run_sha:
        raise ValueError('Existing experiment contract differs; leave its artifacts unchanged.')
    write_json(out/'run_contract.json', {'contract': run_contract, 'sha256': run_sha})
    progress = {'epoch': 0, 'cursor': 0, 'updates': 0, 'order': None,
                'best_score': None, 'patience': 0, 'history': [], 'status': 'RUNNING'}
    order_generator = torch.Generator(device='cpu').manual_seed(args.seed+11000)
    process_generator = torch.Generator(device='cpu').manual_seed(args.seed+22000)
    resume_payload = None
    if args.resume:
        resume_payload = torch.load(args.resume, map_location='cpu', weights_only=False)
        if resume_payload.get('kind') != RUNNER_VERSION or resume_payload.get('run_contract_sha256') != run_sha:
            raise ValueError('Resume checkpoint experiment/data/config/plan contract differs.')
        model, target = restore_candidate(resume_payload, args.device)
        provenance = resume_payload['provenance']
    else:
        # Cache preparation may instantiate a large Phase-1 module and consume
        # global RNG. Fresh candidate weights must depend only on their seed.
        seed_everything(args.seed)
        model, target, provenance = initialize_candidate(cfg, recipe, args.formulation, train_data, device=args.device, mean_checkpoint=args.mean_checkpoint, crossfit_registration=args.crossfit_registration, crossfit_mean_checkpoints=args.crossfit_mean_checkpoints)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=plan['optimization']['learning_rate'])
    stopping = plan['stopping'][args.stage]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stopping['maximum_epochs'], eta_min=.000001)
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload['optimizer']); scheduler.load_state_dict(resume_payload['scheduler'])
        progress = resume_payload['progress']; progress['status'] = 'RUNNING'
        order_generator.set_state(resume_payload['order_rng']); process_generator.set_state(resume_payload['process_rng'])
        restore_rng(resume_payload['global_rng'])
    regimes = None
    if 'pr' in variables:
        regime_path = out/'fitting_rainfall_regimes.json'
        if regime_path.exists(): regimes = RainfallRegimes(**json.loads(regime_path.read_text()))
        else:
            regimes = fit_rainfall_regimes(train_data, seed=args.seed)
            write_json(regime_path, asdict(regimes))
    # Start stochastic-layer RNG independently of cache misses and statistic
    # loader construction; restore resume RNG after every setup operation.
    if resume_payload is None:
        seed_everything(args.seed+33000)
    else:
        restore_rng(resume_payload['global_rng'])
    def checkpoint(path):
        candidate, target_state = candidate_state(model, target, args.formulation, provenance)
        payload = {'kind': RUNNER_VERSION, 'run_contract': run_contract, 'run_contract_sha256': run_sha,
                   'formulation': args.formulation, 'candidate': candidate, 'target_state': target_state,
                   'variables': list(variables), 'provenance': provenance, 'refinement_config': cfg.to_dict(),
                   'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                   'progress': copy.deepcopy(progress), 'order_rng': order_generator.get_state(),
                   'process_rng': process_generator.get_state(), 'global_rng': capture_rng(),
                   'source_config_path': str(Path(args.config).resolve()), 'cache_path': str(Path(args.cache).resolve()),
                   'plan_path': str(Path(args.plan).resolve()), 'contract_path': str(Path(args.contract).resolve())}
        save_checkpoint_atomic(payload, path)
    last = out/'last.ckpt'; checkpoint(last)
    selector = None if args.formulation == 'deterministic_mean' else ScientificSelector(out, contract, args.head)
    interrupted = [False]
    old_handler = signal.signal(signal.SIGINT, lambda *unused: interrupted.__setitem__(0, True))
    batch_size = plan['optimization']['full_batch_size' if args.stage == 'full' else 'screen_batch_size']
    start = time.perf_counter()
    def assess_epoch_impl():
        row = progress['history'][-1]
        assessment = out/f'assessment_epoch_{progress["epoch"]:04d}'
        assessment.mkdir(exist_ok=True)
        checkpoint_path = assessment/'candidate.ckpt'
        if not checkpoint_path.exists(): checkpoint(checkpoint_path)
        else:
            previous = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            candidate, _ = candidate_state(model, target, args.formulation, provenance)
            if previous['candidate']['fingerprint'] != candidate['fingerprint']:
                raise ValueError('Existing assessment checkpoint has different model weights.')
        authentication = {'checkpoint_path': str(checkpoint_path.resolve()), 'checkpoint_sha256': file_sha256(checkpoint_path),
                          'run_contract_sha256': run_sha, 'plan_sha256': plan_sha, 'acceptance_contract_sha256': canonical_hash(contract),
                          'source_cache_sha256': source_manifest['cache_sha256'], 'phase1_checkpoint_sha256': source_manifest['phase1_checkpoint_sha256']}
        row['fitting_loss'] = heldout_loss(model, target, args.formulation, train_data, variables, device=args.device, seed=args.seed)
        row['heldout_loss'] = heldout_loss(model, target, args.formulation, validation_data, variables, device=args.device, seed=args.seed)
        mean_diagnostics = None
        if args.formulation == 'deterministic_mean':
            scales = model.scaling.scale.detach().cpu().reshape(-1).tolist()
            fitting_rmse = {v: math.sqrt(row['fitting_loss'][v])*scales[i] for i, v in enumerate(variables)}
            validation_rmse = {v: math.sqrt(row['heldout_loss'][v])*scales[i] for i, v in enumerate(variables)}
            ratios = {v: fitting_rmse[v]/max(validation_rmse[v], 1e-12) for v in variables}
            mean_diagnostics = {'fitting_remainder_rmse_physical': fitting_rmse, 'validation_remainder_rmse_physical': validation_rmse,
                'fitting_to_validation_rmse_ratio': ratios, 'crossfit_required': any(value < .5 for value in ratios.values()),
                'registered_trigger': 'any fitting/validation physical remainder RMSE ratio < 0.5',
                'mean_fingerprint': model.fingerprint(), 'checkpoint_sha256': authentication['checkpoint_sha256'],
                'plan_sha256': plan_sha, 'fitting_dates_sha256': selection['fit_sha256'], 'validation_dates_sha256': selection['validation_sha256']}
            row['mean_overfit_diagnostic'] = mean_diagnostics
            write_json(checkpoint_path.with_suffix('.mean_diagnostics.json'), mean_diagnostics)
        fitting_diagnostic = train_data if args.stage == 'overfit' else dataset_selection(train_data, diagnostic_dates(train_data.dates))
        _, fit_artifact = predict_dataset(model, target, args.formulation, fitting_diagnostic, variables,
            output_path=assessment/'fitting_endpoints.h5', members=plan['validation']['selection_members'],
            seed=plan['validation']['sampling_seed'], training_seed=args.seed, strategy=cfg.nonnegative_ensemble_strategy,
            device=args.device, data_scope='fitting_diagnostic', resume=True, interrupt_check=lambda: interrupted[0], authentication=authentication)
        records, artifact = predict_dataset(model, target, args.formulation, validation_data, variables,
            output_path=assessment/'validation_endpoints.h5', members=plan['validation']['selection_members'],
            seed=plan['validation']['sampling_seed'], training_seed=args.seed, strategy=cfg.nonnegative_ensemble_strategy,
            device=args.device, data_scope='screening', resume=True, interrupt_check=lambda: interrupted[0], authentication=authentication)
        if args.formulation == 'deterministic_mean':
            decision = _mean_rank(records, contract, artifact['auxiliary_metrics'], args.seed)
            retain_best_mean_checkpoint(progress, decision, checkpoint_path, out, mean_diagnostics)
        else:
            evidence = {'contract_sha256': canonical_hash(contract), 'scope': 'component_validation',
                        'records': {args.head: {str(args.seed): records}}, 'auxiliary_metrics': {args.head: artifact['auxiliary_metrics']}}
            decision = selector.consider(checkpoint_path, evidence, training_progress={'epoch': progress['epoch'], 'stage': args.stage})
        row.update(fitting_artifact=fit_artifact, validation_artifact=artifact, selection_score=decision['score'], selection_provisional=True)
        write_json(assessment/'decision.json', decision)
        old = progress['best_score']; score = decision['score']
        improved = old is None or score[0] < old[0] or (score[0] == old[0] and score[1] < old[1]-stopping['minimum_worst_margin_improvement'])
        if improved: progress['best_score'], progress['patience'] = score, 0
        else: progress['patience'] += 1
        progress['pending_assessment'] = False
        write_json(out/'learning_curve.json', progress['history'])
        if progress['epoch'] >= stopping['minimum_epochs'] and progress['patience'] >= stopping['patience_assessments']:
            progress['status'] = 'STOPPED_REGISTERED_PATIENCE_PROVISIONAL'
    def assess_epoch():
        # DataLoader iterators and future evaluation utilities may consume global
        # RNG even in eval mode. Replay must never advance training dropout RNG.
        training_rng = capture_rng()
        try:
            assess_epoch_impl()
        finally:
            restore_rng(training_rng)
            checkpoint(last)
    unsafe_optimizer_failure = False
    try:
        while progress['epoch'] < stopping['maximum_epochs'] or progress.get('pending_assessment'):
            if progress.get('pending_assessment'):
                assess_epoch()
                if progress['status'] != 'RUNNING' or progress['epoch'] >= stopping['maximum_epochs']: break
            epoch = progress['epoch']
            if progress['order'] is None:
                progress['order'] = torch.randperm(len(train_data), generator=order_generator).tolist()
                progress['cursor'] = 0
            while progress['cursor'] < len(train_data):
                if interrupted[0]: raise InterruptedError('Interrupt requested at a completed optimizer boundary.')
                indices = progress['order'][progress['cursor']:progress['cursor']+batch_size]
                batch = torch.utils.data.default_collate([train_data[i] for i in indices])
                batch = selected_batch(move(batch, args.device), train_data.manifest['variables'], variables)
                process_before, global_before = process_generator.get_state(), capture_rng()
                optimizer.zero_grad(set_to_none=True); model.train()
                try:
                    result = training_result(model, target, args.formulation, batch, process_generator)
                    if not torch.isfinite(result['loss']): raise FloatingPointError('Nonfinite fitting loss.')
                    if progress['updates'] == 0 or (progress['cursor'] == 0 and (epoch+1) % stopping['assess_every_epochs'] == 0):
                        if args.formulation != 'deterministic_mean':
                            diag = gradient_diagnostics(result, model, truth_physical=batch['truth'], valid=batch['valid'], regimes=regimes,
                                normalizer=target if args.formulation == 'direct' else target.normalizer if args.formulation in ('mean_remainder', 'crossfit_remainder') else None)
                            write_json(out/f'gradients_epoch_{epoch+1:04d}.json', diag)
                    result['loss'].backward()
                    grad = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], plan['optimization']['gradient_clip_l2'])
                    if not torch.isfinite(grad): raise FloatingPointError('Nonfinite fitting gradient.')
                except BaseException:
                    process_generator.set_state(process_before); restore_rng(global_before)
                    optimizer.zero_grad(set_to_none=True)
                    raise
                try:
                    optimizer.step()
                except torch.cuda.OutOfMemoryError:
                    # AdamW may already have mutated a subset of parameters/state.
                    # Keep the previous atomic checkpoint; never save partial state.
                    unsafe_optimizer_failure = True
                    raise
                progress['cursor'] += len(indices); progress['updates'] += 1
                if not all(torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError('Nonfinite updated model; resume the last finite checkpoint.')
            progress['epoch'] += 1; progress['cursor'] = 0; progress['order'] = None
            scheduler.step()
            row = {'epoch': progress['epoch'], 'updates': progress['updates'], 'learning_rate': scheduler.get_last_lr()[0],
                   'last_training_batch_loss': float(result['loss'].detach()), 'elapsed_seconds': time.perf_counter()-start}
            progress['history'].append(row)
            progress['pending_assessment'] = (progress['epoch'] % stopping['assess_every_epochs'] == 0 or progress['epoch'] == stopping['maximum_epochs'])
            checkpoint(last); write_json(out/'learning_curve.json', progress['history'])
            print(json.dumps(row), flush=True)
            if shutil.disk_usage(out).free < plan['stopping']['minimum_free_disk_gib']*1024**3:
                raise InterruptedError('Disk reserve reached; preserve exact resumable state.')
        if progress['status'] == 'RUNNING': progress['status'] = 'REACHED_REGISTERED_MAXIMUM_PROVISIONAL'
    except (InterruptedError, KeyboardInterrupt, torch.cuda.OutOfMemoryError) as exc:
        progress['status'] = 'INCONCLUSIVE_RESOURCE_INTERRUPTION'; progress['interruption'] = str(exc)
        if unsafe_optimizer_failure:
            progress['resume_checkpoint'] = str(last)
            progress['resume_policy'] = 'replay_from_previous_atomic_checkpoint_after_partial_optimizer_failure'
    except FloatingPointError as exc:
        progress['status'] = 'FAIL_NUMERICAL_EXECUTION'; progress['failure'] = str(exc)
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            write_json(out/'execution_status.json', progress)
            raise
    finally:
        signal.signal(signal.SIGINT, old_handler)
    if not unsafe_optimizer_failure: checkpoint(last)
    write_json(out/'execution_status.json', progress)
    print(json.dumps({'status': progress['status'], 'last_checkpoint': str(last), 'epoch': progress['epoch'], 'updates': progress['updates']}), flush=True)
    return progress


def predict(args):
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if payload.get('kind') != RUNNER_VERSION: raise ValueError('Predict requires a registered experimental runner checkpoint.')
    if config_fingerprint(payload['run_contract']) != payload['run_contract_sha256']: raise ValueError('Run contract fingerprint changed.')
    if payload['run_contract']['source_code_fingerprints'] != source_code_fingerprints(): raise ValueError('Experimental source code changed; silent checkpoint reinterpretation is forbidden.')
    cfg = resolve_refinement_config({'refinement': payload['refinement_config']})
    model, target = restore_candidate(payload, args.device)
    source = json.loads(Path(args.cache).with_suffix('.manifest.json').read_text())
    if source['cache_sha256'] != payload['run_contract']['source_cache_sha256']: raise ValueError('Prediction source cache changed.')
    dataset = prepare_conditioning(args.cache, args.split, cfg, payload['source_config_path'], args.prepared_cache or Path(args.cache).parent/'prepared_conditioning', device=args.device)
    plan, _, contract = _load_registered_plan(payload['plan_path'], payload['contract_path'])
    if args.diagnostic_dates: dataset = dataset_selection(dataset, diagnostic_dates(dataset.dates))
    if args.split == 'validation' and not args.diagnostic_dates and args.members < plan['validation']['acceptance_members']:
        raise ValueError('Full validation assessment needs at least the registered 20 members.')
    return predict_dataset(model, target, payload['formulation'], dataset, payload['variables'], output_path=args.output,
        members=args.members, seed=plan['validation']['sampling_seed'], training_seed=payload['run_contract']['seed'],
        strategy=cfg.nonnegative_ensemble_strategy, device=args.device, resume=args.resume,
        data_scope='nested_member_diagnostic' if args.diagnostic_dates else 'component_validation' if args.split == 'validation' else 'screening', num_steps=args.num_steps,
        authentication={'checkpoint_path': str(Path(args.checkpoint).resolve()), 'checkpoint_sha256': file_sha256(args.checkpoint),
                        'run_contract_sha256': payload['run_contract_sha256'], 'plan_sha256': payload['run_contract']['plan_sha256'],
                        'acceptance_contract_sha256': canonical_hash(contract), 'source_cache_sha256': source['cache_sha256'],
                        'phase1_checkpoint_sha256': source['phase1_checkpoint_sha256']})


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='operation', required=True)
    t = sub.add_parser('train')
    for option in ('config', 'cache', 'plan', 'contract', 'output'): t.add_argument('--'+option, required=True)
    t.add_argument('--prepared-cache'); t.add_argument('--head', required=True, choices=(*HEADS, 'deterministic_mean'))
    t.add_argument('--recipe', default='native')
    t.add_argument('--formulation', choices=('direct', 'deterministic_mean', 'mean_remainder', 'crossfit_remainder', 'total_support'), default='direct')
    t.add_argument('--stage', choices=('screen', 'overfit', 'full'), required=True)
    t.add_argument('--seed', type=int, default=101); t.add_argument('--device', default='cuda:0')
    t.add_argument('--alignment', choices=('legacy', 'coordinates'))
    t.add_argument('--normalize-predictors', action=argparse.BooleanOptionalAction, default=None)
    t.add_argument('--mean-checkpoint'); t.add_argument('--overfit-registration'); t.add_argument('--resume')
    t.add_argument('--fit-fold', type=int, choices=(0, 1)); t.add_argument('--fold-registration')
    t.add_argument('--crossfit-registration'); t.add_argument('--crossfit-mean-checkpoints', nargs=2)
    v = sub.add_parser('predict')
    for option in ('checkpoint', 'cache', 'output'): v.add_argument('--'+option, required=True)
    v.add_argument('--prepared-cache'); v.add_argument('--resume', action='store_true')
    v.add_argument('--split', choices=('fit', 'validation', 'screen_fit', 'screen_validation'), default='validation')
    v.add_argument('--diagnostic-dates', action='store_true')
    v.add_argument('--members', type=int, default=20); v.add_argument('--device', default='cuda:0'); v.add_argument('--num-steps', type=int)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return train(args) if args.operation == 'train' else predict(args)


if __name__ == '__main__': main()