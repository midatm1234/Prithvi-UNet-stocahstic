#!/usr/bin/env bash
# Run scalar computation + preprocessing for NARR-to-PRISM downscaling.
# Usage:  bash run_preprocess.sh [/path/to/NARR_PRISM_subdomain.yaml]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="${1:-$SCRIPT_DIR/NARR_PRISM_subdomain.yaml}"

echo "=== NARR-PRISM preprocessing pipeline ==="
echo "Config : $CONFIG"
echo "Repo   : $REPO_ROOT"
echo ""

cd "$REPO_ROOT"

printf '%s\n' '-- Step 1/3: Computing scalars (training period only) ...'
python "$SCRIPT_DIR/compute_scalars_narr_prism.py" --config "$CONFIG"
echo "    Scalars done."
echo ""

printf '%s\n' '-- Step 2/3: Preprocessing training data ...'
python "$SCRIPT_DIR/preproc_narr_prism.py" --config "$CONFIG" --mode training
echo "    Training preprocessing done."
echo ""

printf '%s\n' '-- Step 3/3: Preprocessing inference data ...'
python "$SCRIPT_DIR/preproc_narr_prism.py" --config "$CONFIG" --mode inference
echo "    Inference preprocessing done."
echo ""

echo "=== All preprocessing complete ==="
