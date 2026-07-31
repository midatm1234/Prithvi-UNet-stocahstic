#!/usr/bin/env bash
# Run the NARR-PRISM fine-tuning notebook in a detached tmux session.
# Usage:
#   bash run_finetune_notebook_tmux.sh [start|attach|logs|status|stop]
#   bash run_finetune_notebook_tmux.sh [session_name] [output_notebook]
#
# Examples:
#   bash run_finetune_notebook_tmux.sh
#   bash run_finetune_notebook_tmux.sh narr_prism_ft notebooks/narr_prism_finetune_executed.ipynb
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
NOTEBOOK="$SCRIPT_DIR/notebooks/narr_prism_finetune.ipynb"
DEFAULT_OUTPUT="$SCRIPT_DIR/notebooks/narr_prism_finetune_executed.ipynb"
LOG_DIR="$SCRIPT_DIR/logs"
CONDA_ENV="Prithvi"
MAMBA_BIN="${MAMBA_BIN:-$(command -v mamba || true)}"
DEFAULT_SESSION="narr_prism_finetune"

run_worker() {
    local output_notebook="${1:-$DEFAULT_OUTPUT}"
    local log_file="${2:?worker log file is required}"
    if [[ "$output_notebook" != /* ]]; then
        output_notebook="$SCRIPT_DIR/$output_notebook"
    fi

    mkdir -p "$LOG_DIR" "$(dirname "$output_notebook")"
    exec > >(tee -a "$log_file") 2>&1

    echo "=== NARR-PRISM fine-tuning notebook ==="
    echo "Started : $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "Repo    : $REPO_ROOT"
    echo "Workdir : $SCRIPT_DIR"
    echo "Input   : $NOTEBOOK"
    echo "Output  : $output_notebook"
    echo "Log     : $log_file"
    echo ""

    cd "$SCRIPT_DIR"
    export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
    export PYTHONUNBUFFERED=1
    export GRANITEWXC_TQDM_MODE=terminal

    if [[ -z "$MAMBA_BIN" || ! -x "$MAMBA_BIN" ]]; then
        echo "mamba was not found on PATH; cannot run environment $CONDA_ENV." >&2
        exit 1
    fi
    echo "Env     : $CONDA_ENV (via $MAMBA_BIN run)"
    echo "Python  : $($MAMBA_BIN run -n "$CONDA_ENV" python -c 'import sys; print(sys.executable)')"
    echo ""

    if "$MAMBA_BIN" run -n "$CONDA_ENV" python -m papermill --version >/dev/null 2>&1; then
        # Stream raw cell output rather than only saving it in the executed
        # notebook. stderr carries tqdm's in-place progress updates; both
        # streams are inherited by tee above and remain visible in tmux.
        "$MAMBA_BIN" run -a "" -n "$CONDA_ENV" python -m papermill \
            --stdout-file /dev/stdout \
            --stderr-file /dev/stderr \
            "$NOTEBOOK" "$output_notebook"
    else
        output_dir="$(cd "$(dirname "$output_notebook")" && pwd)"
        output_name="$(basename "$output_notebook")"
        "$MAMBA_BIN" run -a "" -n "$CONDA_ENV" python -m jupyter nbconvert \
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
    run_worker "${1:-$DEFAULT_OUTPUT}" "${2:?worker log file is required}"
    exit 0
fi

COMMAND="${1:-start}"
SESSION_NAME="${SESSION_NAME:-$DEFAULT_SESSION}"

case "$COMMAND" in
    attach)
        exec tmux attach -t "$SESSION_NAME"
        ;;
    logs)
        latest="$(ls -1t "$LOG_DIR"/narr_prism_finetune_*.log 2>/dev/null | head -1 || true)"
        if [[ -z "$latest" ]]; then
            echo "No fine-tuning logs found in $LOG_DIR" >&2
            exit 1
        fi
        echo "Tailing: $latest"
        exec tail -f "$latest"
        ;;
    status)
        if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
            echo "RUNNING: tmux session '$SESSION_NAME' is active."
            tmux list-panes -t "$SESSION_NAME" \
                -F '  pane pid=#{pane_pid} cmd=#{pane_current_command}'
        else
            echo "NOT RUNNING: no tmux session '$SESSION_NAME'."
        fi
        exit 0
        ;;
    stop)
        tmux kill-session -t "$SESSION_NAME" 2>/dev/null \
            && echo "Stopped '$SESSION_NAME'." \
            || echo "No session '$SESSION_NAME'."
        exit 0
        ;;
    start)
        OUTPUT_NOTEBOOK="${2:-$DEFAULT_OUTPUT}"
        ;;
    *)
        # Preserve the original positional interface where the first argument
        # is a custom tmux session name.
        SESSION_NAME="$COMMAND"
        OUTPUT_NOTEBOOK="${2:-$DEFAULT_OUTPUT}"
        ;;
esac

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
TIMESTAMP="$(date -u '+%Y%m%d_%H%M%S')"
LOG_FILE="$LOG_DIR/narr_prism_finetune_${TIMESTAMP}.log"

tmux new-session -d -s "$SESSION_NAME" \
    bash "$SCRIPT_DIR/run_finetune_notebook_tmux.sh" \
        --worker "$OUTPUT_NOTEBOOK" "$LOG_FILE"

echo "Started tmux session: $SESSION_NAME"
echo "Attach with         : tmux attach -t $SESSION_NAME"
echo "Watch log           : $SCRIPT_DIR/run_finetune_notebook_tmux.sh logs"
echo "Log file            : $LOG_FILE"
echo "Output notebook     : $OUTPUT_NOTEBOOK"
