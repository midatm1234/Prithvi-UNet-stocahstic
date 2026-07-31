#!/usr/bin/env bash
# Run the MERRA-PRISM fine-tuning notebook in a detached tmux session.
# Usage:
#   bash run_finetune_notebook_tmux.sh [session_name] [output_notebook]
#
# Examples:
#   bash run_finetune_notebook_tmux.sh
#   bash run_finetune_notebook_tmux.sh merra_prism_ft notebooks/merra_prism_finetune_executed.ipynb
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
NOTEBOOK="$SCRIPT_DIR/notebooks/merra_prism_finetune.ipynb"
DEFAULT_OUTPUT="$SCRIPT_DIR/notebooks/merra_prism_finetune_executed.ipynb"
LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/merra_prism_finetune_notebook.log"

run_worker() {
    local output_notebook="${1:-$DEFAULT_OUTPUT}"
    if [[ "$output_notebook" != /* ]]; then
        output_notebook="$SCRIPT_DIR/$output_notebook"
    fi

    mkdir -p "$LOG_DIR" "$(dirname "$output_notebook")"
    exec > >(tee -a "$LOG_FILE") 2>&1

    echo "=== MERRA-PRISM fine-tuning notebook ==="
    echo "Started : $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "Repo    : $REPO_ROOT"
    echo "Workdir : $SCRIPT_DIR"
    echo "Input   : $NOTEBOOK"
    echo "Output  : $output_notebook"
    echo "Log     : $LOG_FILE"
    echo ""

    cd "$SCRIPT_DIR"
    export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
    export PYTHONUNBUFFERED=1

    if command -v conda >/dev/null 2>&1; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate Prithvi
    elif command -v mamba >/dev/null 2>&1; then
        eval "$(mamba shell hook --shell bash)"
        mamba activate Prithvi
    else
        echo "Neither mamba nor conda was found on PATH; cannot activate Prithvi." >&2
        exit 1
    fi
    echo "Python  : $(command -v python)"
    echo ""

    if python -m papermill --version >/dev/null 2>&1; then
        # Forward notebook cell output to this tmux pane so tqdm epoch bars and
        # training metrics remain visible while the session is attached.
        python -m papermill --log-output "$NOTEBOOK" "$output_notebook"
    else
        output_dir="$(cd "$(dirname "$output_notebook")" && pwd)"
        output_name="$(basename "$output_notebook")"
        jupyter nbconvert \
            --to notebook \
            --execute "$NOTEBOOK" \
            --output "$output_name" \
            --output-dir "$output_dir" \
            --ExecutePreprocessor.timeout=-1
    fi

    echo ""
    echo "Finished: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
}

if [[ "${1:-}" == "--worker" ]]; then
    shift
    run_worker "${1:-$DEFAULT_OUTPUT}"
    exit 0
fi

SESSION_NAME="${1:-merra_prism_finetune}"
OUTPUT_NOTEBOOK="${2:-$DEFAULT_OUTPUT}"
if [[ "$OUTPUT_NOTEBOOK" != /* ]]; then
    OUTPUT_NOTEBOOK="$SCRIPT_DIR/$OUTPUT_NOTEBOOK"
fi

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not installed or not on PATH." >&2
    exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session '$SESSION_NAME' already exists." >&2
    echo "Attach with: tmux attach -t $SESSION_NAME" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

tmux new-session -d -s "$SESSION_NAME" \
    bash "$SCRIPT_DIR/run_finetune_notebook_tmux.sh" --worker "$OUTPUT_NOTEBOOK"

echo "Started tmux session: $SESSION_NAME"
echo "Attach with         : tmux attach -t $SESSION_NAME"
echo "Watch log           : tail -f $LOG_FILE"
echo "Output notebook     : $OUTPUT_NOTEBOOK"
