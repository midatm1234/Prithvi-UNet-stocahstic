#!/usr/bin/env python
"""CLI entry-point for MERRA2-to-PRISM fine-tuning.

Usage:
    python merra_prism_finetune.py --config MERRA_PRISM_subdomain.yaml [--num-gpus 1] [--save-every 5]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from granitewxc.utils.config import get_config

from merra_prism_training import run_training


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MERRA-PRISM fine-tuning (single-GPU or multi-GPU).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to MERRA_PRISM YAML config.")
    parser.add_argument(
        "--num-gpus", type=int, default=None,
        help="Number of GPUs (default: from YAML or 1).",
    )
    parser.add_argument("--save-every", type=int, default=5, help="Checkpoint interval (epochs).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = str(Path(args.config).resolve())
    config = get_config(config_path)
    print(f"[finetune] config={config_path}")

    # Scaler wiring + case-context logging + missing-scaler guard live in
    # run_training so the CLI and notebook paths behave identically.
    train_losses, val_losses = run_training(
        config=config,
        config_path=config_path,
        num_gpus=args.num_gpus,
        save_every=args.save_every,
    )
    if train_losses is not None and val_losses is not None:
        print(
            f"[finetune] done: epochs={len(train_losses)}, "
            f"final_train_loss={train_losses[-1]:.6f}, "
            f"final_val_loss={val_losses[-1]:.6f}"
        )


if __name__ == "__main__":
    main()
