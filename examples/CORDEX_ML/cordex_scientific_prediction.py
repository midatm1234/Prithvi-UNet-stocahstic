#!/usr/bin/env python
"""Resolve a completed run's actual selected checkpoint, then use the SA CLI."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import uuid
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from granitewxc.refinement.scientific_data import file_sha256


def resolve_selected_checkpoint(directory):
    directory = Path(directory)
    execution = json.loads((directory/'execution_status.json').read_text(encoding='utf8'))
    if not str(execution.get('status', '')).startswith(('STOPPED_REGISTERED', 'REACHED_REGISTERED')):
        raise ValueError('Full prediction requires a completed registered stopping rule.')
    state = json.loads((directory/'scientific_selection.json').read_text(encoding='utf8'))
    selected = state.get('best_eligible') or state.get('nearest_provisional')
    if selected is None:
        raise ValueError('No eligible or explicitly provisional candidate was selected.')
    path = Path(selected['selected_checkpoint'])
    if not path.is_file() or file_sha256(path) != selected['checkpoint_sha256']:
        raise ValueError('Selected checkpoint is missing or its bytes differ from the selection receipt.')
    return path.resolve(), ('best_eligible' if state.get('best_eligible') else 'nearest_provisional')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-output', required=True)
    for name in ('cache', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--members', type=int, default=20)
    parser.add_argument('--split', choices=('validation', 'screen_validation'), default='validation')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--diagnostic-dates', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    checkpoint, selection = resolve_selected_checkpoint(args.training_output)
    command = [sys.executable, str(ROOT/'examples/CORDEX_ML/notebooks/SA_downscaling_inference_T2_ACCESS-CM2_static.py'),
               '--scientific-experiment', 'predict', '--checkpoint', str(checkpoint),
               '--cache', str(Path(args.cache).resolve()), '--output', str(Path(args.output).resolve()),
               '--members', str(args.members), '--split', args.split, '--device', args.device]
    if args.diagnostic_dates:
        command.append('--diagnostic-dates')
    if args.resume:
        command.append('--resume')
    print(json.dumps({'checkpoint': str(checkpoint), 'selection': selection,
                      'production_accepted': False, 'normal_inference_command': command}), flush=True)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = {'started_utc': datetime.now(timezone.utc).isoformat(),
               'selected_checkpoint': str(checkpoint), 'selection': selection,
               'checkpoint_sha256': file_sha256(checkpoint), 'normal_inference_argv': command,
               'output': str(output), 'members': args.members, 'diagnostic_dates': args.diagnostic_dates,
               'production_accepted': False}
    result = subprocess.run(command, cwd=ROOT)
    receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), returncode=result.returncode,
                   prediction_present=output.is_file())
    attempt = output.with_name(output.stem + '.entrypoint_attempt_' + uuid.uuid4().hex + '.json')
    attempt.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf8')
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
