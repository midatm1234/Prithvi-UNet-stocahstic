#!/usr/bin/env python
"""CLI entrypoint for CORDEX fine-tuning."""

from __future__ import annotations

import argparse
from pathlib import Path

from granitewxc.utils.config import get_config

from cordex_training import run_training


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CORDEX fine-tuning (single GPU, multi-GPU spawn, or torchrun).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to CORDEX YAML config.")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to use when launching internally (ignored under torchrun).",
    )
    parser.add_argument("--save-every", type=int, default=5, help="Checkpoint save interval (epochs).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config).resolve()
    config = get_config(str(config_path))
    print(f"[finetune] config={config_path}")

    train_losses, val_losses = run_training(
        config=config,
        num_gpus=args.num_gpus,
        save_every=args.save_every,
    )
    if train_losses is not None and val_losses is not None:
        print(
            f"[finetune] completed: epochs={len(train_losses)}, "
            f"final_train_loss={train_losses[-1]}, final_val_loss={val_losses[-1]}"
        )


if __name__ == "__main__":
    main()
