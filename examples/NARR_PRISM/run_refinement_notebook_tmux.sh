#!/usr/bin/env bash
# Run the NARR-PRISM refinement notebook in a detached tmux session.
# Usage:
#   bash run_refinement_notebook_tmux.sh [start|attach|logs|status|stop]
#   bash run_refinement_notebook_tmux.sh [session_name] [output_notebook]
#
# Examples:
#   bash run_refinement_notebook_tmux.sh
#   bash run_refinement_notebook_tmux.sh narr_prism_refine notebooks/narr_prism_refinement_executed.ipynb
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
NOTEBOOK="$SCRIPT_DIR/notebooks/narr_prism_refinement.ipynb"
DEFAULT_OUTPUT="$SCRIPT_DIR/notebooks/narr_prism_refinement_executed.ipynb"
LOG_DIR="$SCRIPT_DIR/logs"
CONDA_ENV="Prithvi"
MAMBA_BIN="${MAMBA_BIN:-$(command -v mamba || true)}"
DEFAULT_SESSION="narr_prism_refinement"

run_worker() {
    local output_notebook="${1:-$DEFAULT_OUTPUT}"
    local log_file="${2:?worker log file is required}"
    if [[ "$output_notebook" != /* ]]; then
        output_notebook="$SCRIPT_DIR/$output_notebook"
    fi

    mkdir -p "$LOG_DIR" "$(dirname "$output_notebook")"
    exec > >(tee -a "$log_file") 2>&1

    echo "=== NARR-PRISM refinement notebook ==="
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

    local min_free_gpu_gb="${NARR_PRISM_REFINEMENT_MIN_FREE_GPU_GB:-40}"
    local min_free_gpu_mb=$((min_free_gpu_gb * 1024))
    local selected_gpus="${NARR_PRISM_REFINEMENT_GPUS:-}"
    if [[ -z "$selected_gpus" ]]; then
        if ! command -v nvidia-smi >/dev/null 2>&1; then
            echo "nvidia-smi is required to select refinement GPUs." >&2
            exit 1
        fi
        local inherited_visible="${CUDA_VISIBLE_DEVICES:-}"
        local gpu_index gpu_free_mb
        local -a eligible_gpus=()
        while IFS=',' read -r gpu_index gpu_free_mb; do
            gpu_index="${gpu_index//[[:space:]]/}"
            gpu_free_mb="${gpu_free_mb//[[:space:]]/}"
            [[ "$gpu_index" =~ ^[0-9]+$ && "$gpu_free_mb" =~ ^[0-9]+$ ]] || continue
            if [[ -n "$inherited_visible" ]]; then
                case ",$inherited_visible," in
                    *",$gpu_index,"*) ;;
                    *) continue ;;
                esac
            fi
            if (( gpu_free_mb > min_free_gpu_mb )); then
                eligible_gpus+=("$gpu_index")
            fi
        done < <(
            nvidia-smi \
                --query-gpu=index,memory.free \
                --format=csv,noheader,nounits
        )
        if (( ${#eligible_gpus[@]} == 0 )); then
            echo "No visible GPU has more than ${min_free_gpu_gb} GiB free." >&2
            exit 1
        fi
        local IFS=,
        selected_gpus="${eligible_gpus[*]}"
    fi
    selected_gpus="${selected_gpus//[[:space:]]/}"
    if [[ ! "$selected_gpus" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        echo "Invalid GPU list: '$selected_gpus'" >&2
        exit 1
    fi
    export CUDA_VISIBLE_DEVICES="$selected_gpus"
    local comma_characters="${selected_gpus//[^,]/}"
    export NARR_PRISM_REFINEMENT_NUM_GPUS=$(( ${#comma_characters} + 1 ))
    echo "GPUs    : $CUDA_VISIBLE_DEVICES ($NARR_PRISM_REFINEMENT_NUM_GPUS ranks)"
    echo "Policy  : free memory > ${min_free_gpu_gb} GiB"

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
        latest="$(ls -1t "$LOG_DIR"/narr_prism_refinement_*.log 2>/dev/null | head -1 || true)"
        if [[ -z "$latest" ]]; then
            echo "No refinement logs found in $LOG_DIR" >&2
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
LOG_FILE="$LOG_DIR/narr_prism_refinement_${TIMESTAMP}.log"

tmux new-session -d -s "$SESSION_NAME" \
    bash "$SCRIPT_DIR/run_refinement_notebook_tmux.sh" \
        --worker "$OUTPUT_NOTEBOOK" "$LOG_FILE"

echo "Started tmux session: $SESSION_NAME"
echo "Attach with         : tmux attach -t $SESSION_NAME"
echo "Watch log           : $SCRIPT_DIR/run_refinement_notebook_tmux.sh logs"
echo "Log file            : $LOG_FILE"
echo "Output notebook     : $OUTPUT_NOTEBOOK"
