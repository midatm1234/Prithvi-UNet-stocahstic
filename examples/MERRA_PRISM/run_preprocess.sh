#!/usr/bin/env bash
# Run scalar computation + preprocessing for MERRA2-to-PRISM downscaling.
# Usage:  bash run_preprocess.sh [/path/to/MERRA_PRISM_subdomain.yaml]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="${1:-$SCRIPT_DIR/MERRA_PRISM_subdomain.yaml}"

echo "=== MERRA-PRISM preprocessing pipeline ==="
echo "Config : $CONFIG"
echo "Repo   : $REPO_ROOT"
echo ""

cd "$REPO_ROOT"

printf '%s\n' '-- Step 1/3: Computing scalars (training period only) ...'
python "$SCRIPT_DIR/compute_scalars_merra_prism.py" --config "$CONFIG"
echo "    Scalars done."
echo ""

printf '%s\n' '-- Step 2/3: Preprocessing training data ...'
python "$SCRIPT_DIR/preproc_merra_prism.py" --config "$CONFIG" --mode training
echo "    Training preprocessing done."
echo ""

HAS_VALIDATION="$(
python - "$CONFIG" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh) or {}
validation = (cfg.get("dates") or {}).get("validation") or {}
print("1" if validation.get("start") and validation.get("end") else "0")
PY
)"

if [[ "$HAS_VALIDATION" == "1" ]]; then
  printf '%s\n' '-- Step 3/4: Preprocessing validation data ...'
  python "$SCRIPT_DIR/preproc_merra_prism.py" --config "$CONFIG" --mode validation
  echo "    Validation preprocessing done."
  echo ""
  printf '%s\n' '-- Step 4/4: Preprocessing inference data ...'
else
  printf '%s\n' '-- Step 3/3: Preprocessing inference data ...'
fi
python "$SCRIPT_DIR/preproc_merra_prism.py" --config "$CONFIG" --mode inference
echo "    Inference preprocessing done."
echo ""

echo "=== All preprocessing complete ==="
