"""CPU synthetic capability check for all four refinement heads and two variables.

This fits one known spatial residual (not a held-out climate experiment). New
source draws probe the full sampler; all pre/post comparisons use paired seeds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from granitewxc.refinement.base import build_refiner, masked_loss
from granitewxc.refinement.config import resolve_refinement_config
from granitewxc.refinement.normalization import ResidualNormalizer

HEADS = ('diffusion_unet', 'diffusion_transformer', 'flow_matching_unet', 'flow_matching_transformer')
SEEDS = (1701, 1702, 1703, 1704, 1705, 1706, 1707, 1708)


def run_head(head, output_dir=None):
    torch.set_num_threads(1)
    torch.manual_seed(409)
    rows = torch.linspace(-1., 1., 9).view(1, 1, 9, 1).expand(1, 1, 9, 11)
    cols = torch.linspace(-1., 1., 11).view(1, 1, 1, 11).expand_as(rows)
    rainfall = .8 * torch.sin(torch.pi * cols) + .4 * rows - .7 * (cols > .65) + .5 * (rows < -.65)
    temperature = -.5 + .6 * torch.cos(torch.pi * rows) - .5 * cols + .8 * (cols < -.65)
    physical_target = torch.cat((rainfall, temperature), dim=1)
    conditioning = torch.cat((rows, cols, torch.sin(torch.pi * cols), torch.cos(torch.pi * rows)), dim=1)
    config = resolve_refinement_config({'refinement': {
        'type': head,
        'unet': {'hidden_channels': 8, 'num_levels': 2, 'attention_heads': 4, 'spatial_alignment': 'coordinates'},
        'transformer': {'embedding_dim': 16, 'num_heads': 4, 'num_blocks': 1, 'patch_size': 2, 'spatial_alignment': 'coordinates'},
        **({'diffusion': {'training_timesteps': 20, 'inference_steps': 20, 'prediction_type': 'sample'}} if head.startswith('diffusion') else
           {'flow_matching': {'integration_steps': 20, 'solver': 'heun'}}),
    }})
    normalizer = ResidualNormalizer(2, config.residual_normalization)
    normalizer.update(physical_target)
    normalizer.finalize()
    target = normalizer.normalize(physical_target)
    zero = normalizer.normalize(torch.zeros_like(physical_target))
    model = build_refiner(config, residual_channels=2, cond_channels=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)

    @torch.no_grad()
    def sample():
        model.eval()
        raw = torch.stack([model.sample(conditioning, generator=torch.Generator().manual_seed(seed)) for seed in SEEDS], dim=1)
        corrections = torch.stack([model.apply_correction_gate(normalizer.denormalize(raw[:, member])) for member in range(len(SEEDS))], dim=1)
        return raw, corrections

    raw_before, before = sample()
    model.train()
    updates = 400 if head.startswith('diffusion') else 600
    generator = torch.Generator().manual_seed(314)
    started = time.perf_counter()
    process_by_variable = []
    initial_gradient_by_variable = []
    initial_loss = None
    for step in range(updates):
        optimizer.zero_grad(set_to_none=True)
        losses = model.training_loss(target, conditioning, generator=generator, zero_residual=zero)
        if step == 0:
            initial_loss = float(losses['loss'].detach())
            process_target = losses.get('process_target', losses.get('target_velocity'))
            for channel in range(2):
                variable_loss = masked_loss(losses['prediction'][:, channel:channel+1], process_target[:, channel:channel+1], None)
                gradients = torch.autograd.grad(variable_loss, tuple(model.parameters()), retain_graph=True, allow_unused=True)
                process_by_variable.append(float(variable_loss.detach()))
                initial_gradient_by_variable.append(float(sum(g.detach().square().sum() for g in gradients if g is not None).sqrt()))
        losses['loss'].backward()
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise RuntimeError(f'Nonfinite gradients in {head} at update {step}')
        optimizer.step()
    raw_after, after = sample()
    border = torch.zeros_like(rows[0, 0], dtype=torch.bool)
    border[:2] = border[-2:] = True
    border[:, :2] = border[:, -2:] = True
    regions = {'full': torch.ones_like(border), 'boundary_2': border, 'interior_2': ~border}
    results = []
    for channel, variable in enumerate(('pr', 'tasmax')):
        for region, mask in regions.items():
            truth = physical_target[0, channel][mask]
            base_error = -truth
            item = {'head': head, 'variable': variable, 'region': region,
                    'phase1_mae': float(base_error.abs().mean()), 'phase1_rmse': float(base_error.square().mean().sqrt()),
                    'phase1_bias': float(base_error.mean())}
            for name, members in (('before', before), ('after', after)):
                selected = members[0, :, channel][:, mask]
                error = selected.mean(0) - truth
                item.update({f'{name}_mae': float(error.abs().mean()), f'{name}_rmse': float(error.square().mean().sqrt()),
                             f'{name}_bias': float(error.mean()), f'{name}_spread': float(selected.std(0).mean()),
                             f'{name}_member_rmse': float((selected - truth).square().mean().sqrt())})
            results.append(item)
    summary = {'head': head, 'scope': 'one synthetic spatial field, training capability only; no climate/generalization claim',
               'variables': ['pr', 'tasmax'], 'units': ['mm/day residual', 'K residual'], 'domain': [9, 11],
               'seed_set': list(SEEDS), 'ensemble_size': len(SEEDS), 'updates': updates, 'cpu_seconds': time.perf_counter()-started,
               'configuration': config.to_dict(), 'residual_statistics': normalizer.metadata(),
               'initial_loss': initial_loss, 'final_training_loss': float(losses['loss'].detach()),
               'initial_process_loss_by_variable': process_by_variable,
               'initial_process_gradient_l2_by_variable': initial_gradient_by_variable,
               'correction_gate': model.correction_gate.detach().tolist() if hasattr(model, 'correction_gate') else None,
               'results': results}
    if output_dir is not None:
        destination = Path(output_dir) / head
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        torch.save({'physical_target_residual': physical_target, 'conditioning': conditioning,
                    'initial_normalized_samples': raw_before, 'initial_physical_corrections': before,
                    'trained_normalized_samples': raw_after, 'trained_physical_corrections': after,
                    'model': model.state_dict(), 'normalizer': normalizer.state_dict()}, destination / 'fields.pt')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='artifacts/refinement_validation/four_option_20260909/synthetic_signed_fields')
    parser.add_argument('--head', choices=HEADS, action='append')
    args = parser.parse_args()
    for head in args.head or HEADS:
        result = run_head(head, args.output_dir)
        print(json.dumps({'head': head, 'seconds': result['cpu_seconds'], 'results': [r for r in result['results'] if r['region'] == 'full']}), flush=True)


if __name__ == '__main__':
    main()
