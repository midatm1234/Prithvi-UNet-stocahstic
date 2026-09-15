"""Paired, full-domain, all-head refinement learning and sampling audit.

This is a controlled subset diagnostic, never a substitute for held-out climate
validation. It reuses one frozen Phase-1 pass and stores every physical member.
Run from the repository root in the Prithvi environment. Existing outputs are
never overwritten. No test data are used for fitting or selecting checkpoints.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

HEADS = ('diffusion_unet', 'diffusion_transformer', 'flow_matching_unet', 'flow_matching_transformer')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def input_contract(raw):
    from granitewxc.refinement.config import config_fingerprint
    return config_fingerprint(dict(data=raw['data'], predictands=raw.get('predictands'),
        conditioning=raw['model']['refinement']['conditioning'],
        phase1_model={k: v for k, v in raw['model'].items() if k not in ('refinement', 'phase1')}))


def summarize(prediction, truth, region):
    valid = np.isfinite(prediction) & np.isfinite(truth) & region[None]
    error = np.where(valid, prediction - truth, np.nan)
    counts = valid.sum(axis=0)
    def time_mean(values):
        return np.divide(np.where(valid, values, 0).sum(axis=0, dtype=np.float64),
                         counts, out=np.full(counts.shape, np.nan), where=counts > 0)
    clim = time_mean(prediction-truth)
    pred_clim = time_mean(prediction)
    target_clim = time_mean(truth)
    good = np.isfinite(pred_clim) & np.isfinite(target_clim)
    corr = float(np.corrcoef(pred_clim[good], target_clim[good])[0, 1]) if good.sum() > 1 and np.std(pred_clim[good]) > 0 and np.std(target_clim[good]) > 0 else None
    return dict(bias=float(np.nanmean(error)), daily_mae=float(np.nanmean(abs(error))),
                daily_rmse=float(np.sqrt(np.nanmean(error ** 2))),
                climatology_mae=float(np.nanmean(abs(clim))),
                climatology_rmse=float(np.sqrt(np.nanmean(clim ** 2))),
                climatology_correlation=corr, valid_count=int(valid.sum()))


def regions(height, width):
    yy, xx = np.indices((height, width))
    distance = np.minimum.reduce((yy, xx, height - 1 - yy, width - 1 - xx))
    result = {'domain': np.ones((height, width), bool)}
    for n in (1, 2, 4, 8, 16):
        if 2 * n < min(height, width):
            result[f'boundary_{n}'] = distance < n
            result[f'interior_{n}'] = distance >= n
            for name, mask in (('row_start', yy < n), ('row_end', yy >= height-n),
                               ('column_start', xx < n), ('column_end', xx >= width-n)):
                result[f'{name}_{n}'] = mask
    return result


def build_cache(config, phase1, output, args):
    from cordex_refinement_training import _build_loaders
    from granitewxc.refinement.two_phase import build_two_phase_model

    train_loader, valid_loader, source = _build_loaders(
        config, use_gpu=True, tiny_overfit=False, validation_fraction=args.validation_fraction)
    if valid_loader is None:
        raise ValueError('This audit requires a disjoint validation loader.')
    wrapper = build_two_phase_model(phase1, config).to(args.device).eval()
    cache = {'split_source': source, 'variables': list(config.data.output_vars)}
    cache['input_contract'] = input_contract(yaml.safe_load(Path(args.config).read_text()))
    for split, loader, count in (('train', train_loader, args.train_count),
                                  ('validation', valid_loader, args.validation_count)):
        indices = np.unique(np.linspace(0, len(loader.dataset)-1, min(count, len(loader.dataset)), dtype=int))
        selected = Subset(loader.dataset, indices.tolist())
        batches = DataLoader(selected, batch_size=args.batch_size, shuffle=False, num_workers=0)
        records = []
        dates = []
        for batch in batches:
            batch = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            with torch.no_grad():
                base, normalized, cond = wrapper._prepare(batch)
            dates.extend(list(batch['__sample_timestamp']))
            valid = torch.isfinite(batch['y']) & torch.isfinite(base)
            if '__target_valid_mask' in batch:
                valid &= batch['__target_valid_mask'].bool()
            records.append({'base': base.cpu(), 'normalized': normalized.cpu(), 'cond': cond.cpu(),
                            'truth': batch['y'].cpu(), 'valid': valid.cpu()})
        cache[split] = {key: torch.cat([record[key] for record in records]) for key in records[0]}
        dataset = loader.dataset
        absolute_indices = list(indices)
        if isinstance(dataset, Subset):
            absolute_indices = [int(dataset.indices[i]) for i in indices]
            dataset = dataset.dataset
        cache[split]['dataset_indices'] = absolute_indices
        cache[split]['dates'] = dates
        print('cached', split, len(indices), 'full domains', flush=True)
    target_path = config.data.training_target_paths[0]
    with xr.open_dataset(target_path) as ds:
        cache['latitude'] = ds.lat.values
        cache['longitude'] = ds.lon.values
        cache['units'] = {name: ds[name].attrs.get('units', '') for name in cache['variables']}
    cache['phase1_checkpoint'] = str((ROOT / config.model.phase1['checkpoint']).resolve())
    cache['phase1_checkpoint_sha256'] = digest(cache['phase1_checkpoint'])
    torch.save(cache, output / 'paired_phase1_cache.pt')
    np.savez_compressed(output / 'training_residual_statistics_maps.npz',
                       mean=torch.nanmean(torch.where(cache['train']['valid'], cache['train']['truth']-cache['train']['base'], torch.nan), 0).numpy(),
                       std=np.nanstd(np.where(cache['train']['valid'].numpy(), (cache['train']['truth']-cache['train']['base']).numpy(), np.nan), axis=0),
                       count=cache['train']['valid'].sum(0).numpy(),
                       latitude=cache['latitude'], longitude=cache['longitude'])
    return cache


@torch.no_grad()
def recondition_cache(config, phase1, cache, output, args):
    """Reuse exact cached Phase-1 fields while applying a new conditioning convention."""
    from cordex_refinement_training import _build_loaders
    from granitewxc.refinement.two_phase import build_two_phase_model
    loaders = _build_loaders(config, use_gpu=True, tiny_overfit=False,
                             validation_fraction=args.validation_fraction)[:2]
    wrapper = build_two_phase_model(phase1, config).to(args.device).eval()
    for split, loader in zip(('train', 'validation'), loaders):
        dataset = loader.dataset
        if isinstance(dataset, Subset):
            dataset = dataset.dataset
        selected = Subset(dataset, cache[split]['dataset_indices'])
        batches = DataLoader(selected, batch_size=args.batch_size, shuffle=False, num_workers=0)
        conditions, timestamps = [], []
        start = 0
        for batch in batches:
            batch = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            count = len(batch['x'])
            normalized = cache[split]['normalized'][start:start+count].to(args.device)
            conditions.append(wrapper.build_conditioning(batch, normalized).cpu())
            timestamps.extend(list(batch['__sample_timestamp']))
            start += count
        cache[split]['cond'] = torch.cat(conditions)
        # Exact sample timestamp strings from the loader, not guessed positional dates.
        if not np.array_equal(np.asarray(timestamps, dtype='datetime64[ns]'), np.asarray(cache[split]['dates'], dtype='datetime64[ns]')):
            raise ValueError('Reconditioned samples do not match the cached dates.')
    cache['input_contract'] = input_contract(args.runtime_raw_config)
    torch.save(cache, output / 'paired_phase1_cache.pt')
    return cache


@torch.no_grad()
def evaluate(wrapper, cache, output, label, args):
    wrapper.eval()
    refiner = wrapper.refiner
    split = cache['validation']
    raw, physical, unbounded = [], [], []
    traces = {}
    for start in range(0, len(split['base']), args.batch_size):
        stop = start + args.batch_size
        cond = split['cond'][start:stop].to(args.device)
        base = split['base'][start:stop].to(args.device)
        draws = []
        for member in range(args.members):
            generator = torch.Generator(device=args.device).manual_seed(args.seed + member * 100003 + start)
            def callback(stage, step, process_time, state):
                if start == 0 and member == 0:
                    # Every step for the first full-domain sample, in model space.
                    traces[f'{stage}_{step:04d}'] = state[:1].detach().float().cpu().numpy()
            with torch.no_grad():
                sample = refiner.sample(cond, generator=generator, trajectory_callback=callback)
            draws.append(sample)
        raw_batch = torch.stack(draws, 1)
        fields = []
        for member in range(args.members):
            correction = wrapper.residual_normalizer.denormalize(raw_batch[:, member])
            fields.append(base + wrapper._effective_physical_residual(correction))
        fields = torch.stack(fields, 1)
        selected = wrapper.target_space.constrain_physical_ensemble(
            fields, strategy=wrapper.refinement_config.nonnegative_ensemble_strategy)
        raw.append(raw_batch.cpu())
        physical.append(selected['members'].cpu())
        unbounded.append(fields.cpu())
    members = torch.cat(physical).numpy()
    unbounded = torch.cat(unbounded).numpy()
    raw = torch.cat(raw).numpy()
    truth = split['truth'].numpy()
    baseline = split['base'].numpy()
    common_mask = split['valid'].numpy()
    if not np.isfinite(members)[np.broadcast_to(common_mask[:, None], members.shape)].all():
        raise FloatingPointError('Nonfinite predicted member on a valid cell; refusing a changing evaluation mask.')
    mean = members.mean(axis=1, dtype=np.float64)
    np.savez_compressed(output / f'{label}_members.npz', members=members,
                       unbounded=unbounded, normalized_residual=raw,
                       physical_residual=members-baseline[:, None], mean=mean,
                       truth=truth, baseline=baseline, valid_mask=common_mask,
                       dates=np.array(split['dates']), latitude=cache['latitude'], longitude=cache['longitude'])
    np.savez_compressed(output / f'{label}_trajectory.npz', **traces,
                       reconstructed_unbounded=unbounded[:1, :1], postprocessed=members[:1, :1])
    rows = []
    for channel, name in enumerate(cache['variables']):
        for region_name, region in regions(*truth.shape[-2:]).items():
            for kind, values in (('phase1', baseline), (label, mean)):
                masked_truth = np.where(common_mask[:, channel], truth[:, channel], np.nan)
                rows.append(dict(variable=name, product=kind, region=region_name,
                                 **summarize(values[:, channel], masked_truth, region)))
    with (output / f'{label}_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        if row['region'] == 'domain':
            print(row, flush=True)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--cache', help='Previously generated paired_phase1_cache.pt')
    parser.add_argument('--heads', nargs='+', choices=HEADS, default=list(HEADS))
    parser.add_argument('--train-count', type=int, default=128)
    parser.add_argument('--validation-count', type=int, default=32)
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--updates', type=int, default=200)
    parser.add_argument('--learning-rate', type=float, default=0.00005)
    parser.add_argument('--members', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--alignment', choices=('legacy', 'coordinates'), default='legacy')
    parser.add_argument('--checkpoint', action='append', default=[], metavar='HEAD=PATH')
    parser.add_argument('--normalize-conditioning', action='store_true', help='Separate retraining ablation using frozen Phase-1 input scalers.')
    parser.add_argument('--cache-only', action='store_true')
    args = parser.parse_args(argv)
    if min(args.train_count, args.validation_count, args.batch_size, args.members) < 1 or args.updates < 0:
        parser.error('Counts must be positive and updates nonnegative.')
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    from granitewxc.models.model import get_finetune_model_UNET
    from granitewxc.refinement.checkpoint import (load_phase1_state_dict, load_refinement_state_dict,
        validate_phase1_reference, phase1_state_fingerprint, build_refinement_checkpoint)
    from granitewxc.refinement.two_phase import build_two_phase_model
    from granitewxc.utils.config import get_config
    config = get_config(args.config)
    raw_config = yaml.safe_load(Path(args.config).read_text())
    config.batch_size = args.batch_size
    config.dl_num_workers = 0
    config.dl_pin_memory = False
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    phase1 = get_finetune_model_UNET(config)
    phase1_path = (ROOT / config.model.phase1['checkpoint']).resolve()
    first = build_two_phase_model(phase1, config)
    payload = torch.load(phase1_path, map_location='cpu', weights_only=True)
    print(load_phase1_state_dict(first, payload).summary(), flush=True)
    del payload
    phase1.to(args.device).eval().requires_grad_(False)
    fingerprint = phase1_state_fingerprint(phase1.state_dict())
    cache = torch.load(args.cache, map_location='cpu', weights_only=False) if args.cache else build_cache(config, phase1, output, args)
    if cache['phase1_checkpoint_sha256'] != digest(phase1_path):
        raise ValueError('Cached Phase-1 checkpoint differs from the configured checkpoint.')
    if cache.get('input_contract') != input_contract(raw_config):
        raise ValueError('Cached conditioning/data/normalization contract differs from the requested case.')
    if cache['variables'] != list(config.data.output_vars):
        raise ValueError('Cached variable order differs from the configured order.')
    if args.normalize_conditioning:
        config.model.refinement['conditioning']['normalize_predictors'] = True
        raw_config['model']['refinement']['conditioning']['normalize_predictors'] = True
        args.runtime_raw_config = raw_config
        cache = recondition_cache(config, phase1, cache, output, args)
        del args.runtime_raw_config
    manifest = dict(vars(args), phase1_fingerprint=fingerprint,
                    phase1_checkpoint_sha256=cache['phase1_checkpoint_sha256'],
                    split_source=cache['split_source'], variables=cache['variables'], units=cache['units'],
                    training_dates=cache['train']['dates'], validation_dates=cache['validation']['dates'],
                    scope='Controlled subset engineering/scientific diagnostic; not full climatology or untouched test acceptance.')
    manifest['member_seeds'] = [args.seed + member * 100003 for member in range(args.members)]
    manifest['batch_seed_rule'] = 'member_seed + batch_start_index; unchanged before/after'
    manifest['existing_checkpoint_validation_caveat'] = 'Existing notebook checkpoints may have trained on audit validation dates; these comparisons are diagnostic only.'
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    if args.cache_only:
        return
    checkpoints = dict(item.split('=', 1) for item in args.checkpoint)
    for head in args.heads:
        head_output = output / head
        head_output.mkdir()
        source = Path(args.config).with_name('SA_T2_ACCESS-CM2_static_' + head + '.yaml')
        head_config = get_config(str(source)) if source.exists() else copy.deepcopy(config)
        head_raw = yaml.safe_load(source.read_text()) if source.exists() else copy.deepcopy(raw_config)
        if args.normalize_conditioning:
            head_config.model.refinement['conditioning']['normalize_predictors'] = True
            head_raw['model']['refinement']['conditioning']['normalize_predictors'] = True
        if input_contract(head_raw) != cache['input_contract']:
            raise ValueError(f'{head} uses a different Phase-1, input, or conditioning contract.')
        head_config.model.refinement['type'] = head
        architecture = 'transformer' if 'transformer' in head else 'unet'
        head_config.model.refinement.setdefault(architecture, {})['spatial_alignment'] = args.alignment
        head_raw['model']['refinement'] = head_config.model.refinement
        head_raw['checkpoint_dir'] = str(head_output)
        head_raw['case_name'] = 'four_option_subset_' + head
        (head_output / 'resolved.yaml').write_text(yaml.safe_dump(head_raw, sort_keys=False))
        torch.manual_seed(args.seed)
        wrapper = build_two_phase_model(phase1, head_config).to(args.device)
        wrapper.initialize_refiner(cache['train']['cond'].shape[1])
        if head in checkpoints:
            payload = torch.load(checkpoints[head], map_location='cpu', weights_only=True)
            validate_phase1_reference(payload, phase1.state_dict(), strict=True)
            print(head, load_refinement_state_dict(wrapper, payload).summary(), flush=True)
            del payload
        else:
            for start in range(0, len(cache['train']['base']), args.batch_size):
                split = cache['train']
                residual = (split['truth'][start:start+args.batch_size] - split['base'][start:start+args.batch_size]).to(args.device)
                wrapper.residual_normalizer.update(residual, split['valid'][start:start+args.batch_size].to(args.device))
            wrapper.finalize_residual_statistics()
        before_label = 'existing_checkpoint' if head in checkpoints else 'initialized_control'
        evaluate(wrapper, cache, head_output, before_label, args)
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
        optimizer = torch.optim.AdamW(wrapper.trainable_parameters(), lr=args.learning_rate)
        history = []
        split = cache['train']
        for update in range(args.updates):
            indices = torch.randint(len(split['base']), (args.batch_size,), generator=torch.Generator().manual_seed(args.seed+update))
            cond = split['cond'][indices].to(args.device)
            valid = split['valid'][indices].to(args.device)
            residual = (split['truth'][indices] - split['base'][indices]).to(args.device)
            target = wrapper.residual_normalizer.normalize(residual, valid)
            zero = wrapper.residual_normalizer.normalize(torch.zeros_like(residual), valid)
            wrapper.train()
            assert not phase1.training and not any(p.requires_grad for p in phase1.parameters())
            optimizer.zero_grad(set_to_none=True)
            losses = wrapper.refiner.training_loss(target, cond, valid, generator=generator, zero_residual=zero)
            if not torch.isfinite(losses['loss']):
                raise FloatingPointError(f'{head} nonfinite training loss at {update}')
            losses['loss'].backward()
            norm = torch.nn.utils.clip_grad_norm_(wrapper.trainable_parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            history.append(dict(update=update+1, loss=float(losses['loss'].detach()), gradient_norm=float(norm)))
            if (update+1) % 25 == 0:
                print(head, history[-1], flush=True)
        (head_output / 'training_history.json').write_text(json.dumps(history, indent=2))
        evaluate(wrapper, cache, head_output, 'after', args)
        assert phase1_state_fingerprint(phase1.state_dict()) == fingerprint
        payload = build_refinement_checkpoint(wrapper, epoch=1, global_step=args.updates,
                    phase1_checkpoint=str(phase1_path), phase1_fingerprint=fingerprint,
                    resolved_config=head_raw, extra={'case_name': head_raw['case_name'], 'audit_scope': manifest['scope']})
        torch.save(payload, head_output / 'audit.ckpt')
        (head_output / 'normalization.json').write_text(json.dumps(wrapper.residual_normalization_metadata(), indent=2))
        del wrapper
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
