#!/usr/bin/env bash
# Run the complete NARR-to-PRISM preprocessing contract.
#
# A successful run always executes, in order:
#   1. training daily products
#   2. training-only, case-scoped scalar recomputation
#   3. optional held-out validation daily products
#   4. inference daily products
#
# Usage:
#   bash run_preprocess.sh [CONFIG]
#   bash run_preprocess.sh --config CONFIG [--shards N] [--overwrite]
#
# Backward-compatible environment variables:
#   PREPROCESS_SHARDS=N  OVERWRITE=0|1  PYTHON_BIN=python
# Scalars are always replaced; OVERWRITE controls daily NetCDF products only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="${CONFIG:-$SCRIPT_DIR/NARR_PRISM_subdomain.yaml}"
PREPROCESS_SHARDS="${PREPROCESS_SHARDS:-1}"
OVERWRITE="${OVERWRITE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
  sed -n '2,16s/^# \{0,1\}//p' "${BASH_SOURCE[0]}"
}

# Retain the original positional CONFIG interface while supporting explicit
# flags that are easier to audit in logs and job schedulers.
if [[ $# -gt 0 && "${1:-}" != --* ]]; then
  CONFIG="$1"
  shift
fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      [[ $# -ge 2 ]] || { echo "ERROR: --config requires a path" >&2; exit 2; }
      CONFIG="$2"
      shift 2
      ;;
    --shards)
      [[ $# -ge 2 ]] || { echo "ERROR: --shards requires an integer" >&2; exit 2; }
      PREPROCESS_SHARDS="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --no-overwrite)
      OVERWRITE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! "$PREPROCESS_SHARDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: PREPROCESS_SHARDS/--shards must be a positive integer; got '$PREPROCESS_SHARDS'" >&2
  exit 2
fi
if [[ "$OVERWRITE" != "0" && "$OVERWRITE" != "1" ]]; then
  echo "ERROR: OVERWRITE must be 0 or 1; got '$OVERWRITE'" >&2
  exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config file not found: $CONFIG" >&2
  exit 2
fi
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi

OVERWRITE_FLAG=()
if [[ "$OVERWRITE" == "1" ]]; then
  OVERWRITE_FLAG+=(--overwrite)
fi

ACTIVE_PIDS=()
cleanup_shards() {
  local pid
  for pid in "${ACTIVE_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${ACTIVE_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  ACTIVE_PIDS=()
}
trap cleanup_shards INT TERM EXIT

run_preprocess_mode() {
  local mode="$1"
  local shard pid failed=0
  ACTIVE_PIDS=()
  for ((shard=0; shard<PREPROCESS_SHARDS; shard++)); do
    "$PYTHON_BIN" "$SCRIPT_DIR/preproc_narr_prism.py" \
      --config "$CONFIG" --mode "$mode" "${OVERWRITE_FLAG[@]}" \
      --date-shard-index "$shard" --date-shard-count "$PREPROCESS_SHARDS" &
    ACTIVE_PIDS+=("$!")
  done
  for pid in "${ACTIVE_PIDS[@]}"; do
    if ! wait "$pid"; then
      failed=1
      cleanup_shards
      break
    fi
  done
  ACTIVE_PIDS=()
  if ((failed)); then
    if [[ "$mode" == "training" ]]; then
      echo "ERROR: NARR training preprocessing failed; scalars were not recomputed." >&2
    else
      echo "ERROR: NARR preprocessing failed for mode '$mode'." >&2
    fi
    return 1
  fi
}

cd "$REPO_ROOT"

HAS_VALIDATION="$(
"$PYTHON_BIN" - "$CONFIG" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh) or {}
validation = (cfg.get("dates") or {}).get("validation") or {}
start, end = validation.get("start"), validation.get("end")
if bool(start) != bool(end):
    raise SystemExit(
        "dates.validation must define both start and end, or neither"
    )
print("1" if start and end else "0")
PY
)"
if [[ "$HAS_VALIDATION" != "0" && "$HAS_VALIDATION" != "1" ]]; then
  echo "ERROR: could not determine dates.validation from $CONFIG" >&2
  exit 2
fi

TOTAL_STEPS=3
if [[ "$HAS_VALIDATION" == "1" ]]; then
  TOTAL_STEPS=4
fi

echo "=== NARR-PRISM preprocessing pipeline ==="
echo "Config             : $CONFIG"
echo "Repo               : $REPO_ROOT"
echo "Python             : $(command -v "$PYTHON_BIN")"
echo "Date shards        : $PREPROCESS_SHARDS"
echo "Overwrite daily products: $OVERWRITE"
echo "Scalars            : always recomputed from training products"
echo ""

printf '%s\n' "-- Step 1/$TOTAL_STEPS: Preprocessing training data ..."
run_preprocess_mode training
echo "    Training preprocessing done."
echo ""

printf '%s\n' "-- Step 2/$TOTAL_STEPS: Recomputing case-scoped scalars from training products only ..."
"$PYTHON_BIN" "$SCRIPT_DIR/compute_scalars_narr_prism.py" --config "$CONFIG"
echo "    Scalar recomputation done."
echo ""

if [[ "$HAS_VALIDATION" == "1" ]]; then
  printf '%s\n' '-- Step 3/4: Preprocessing held-out validation data ...'
  run_preprocess_mode validation
  echo "    Validation preprocessing done."
  echo ""
  INFERENCE_STEP=4
else
  INFERENCE_STEP=3
fi

printf '%s\n' "-- Step $INFERENCE_STEP/$TOTAL_STEPS: Preprocessing inference data ..."
run_preprocess_mode inference
echo "    Inference preprocessing done."
echo ""

echo "=== NARR-PRISM preprocessing and scalar recomputation complete ==="
