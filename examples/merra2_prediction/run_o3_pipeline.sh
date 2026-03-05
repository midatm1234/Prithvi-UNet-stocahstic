#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${1:-$SCRIPT_DIR/o3_pipeline_config.yaml}"

"$PYTHON_BIN" "$SCRIPT_DIR/preprocess_o3_pairs.py" --config "$CONFIG_PATH" --overwrite
"$PYTHON_BIN" "$SCRIPT_DIR/compute_scalars_o3.py" --config "$CONFIG_PATH"
"$PYTHON_BIN" "$SCRIPT_DIR/finetune_o3_next_step.py" --config "$CONFIG_PATH"
