#!/usr/bin/env bash
#
# Run the NARR-PRISM inference notebook headless in a detached tmux session,
# so it keeps running after you close the notebook / disconnect.
#
# Usage:
#   ./run_inference_tmux.sh                 # start the run
#   ./run_inference_tmux.sh attach          # attach to the live session
#   ./run_inference_tmux.sh logs            # tail the log file
#   ./run_inference_tmux.sh status          # is it still running?
#   ./run_inference_tmux.sh stop            # kill the session
#
set -euo pipefail

# ----------------------------- configuration -------------------------------
SESSION="narr_infer"
CONDA_ENV="Prithvi"
REPO_ROOT="/data/granite-wxc"
NOTEBOOK="${REPO_ROOT}/examples/NARR_PRISM/notebooks/narr_prism_inference.ipynb"
LOG_DIR="${REPO_ROOT}/examples/NARR_PRISM/logs"
# Cell timeout in seconds (-1 = no per-cell timeout). Inference can be long.
CELL_TIMEOUT="${CELL_TIMEOUT:--1}"
# ---------------------------------------------------------------------------

MAMBA_BIN="/home/azureuser/miniforge3/bin/mamba"

cmd="${1:-start}"

case "${cmd}" in
  attach)
    exec tmux attach -t "${SESSION}"
    ;;
  logs)
    latest="$(ls -1t "${LOG_DIR}"/inference_*.log 2>/dev/null | head -1 || true)"
    [ -z "${latest}" ] && { echo "No log files in ${LOG_DIR}"; exit 1; }
    echo "Tailing: ${latest}"
    exec tail -f "${latest}"
    ;;
  status)
    if tmux has-session -t "${SESSION}" 2>/dev/null; then
      echo "RUNNING: tmux session '${SESSION}' is active."
      tmux list-panes -t "${SESSION}" -F '  pane pid=#{pane_pid} cmd=#{pane_current_command}' 2>/dev/null || true
    else
      echo "NOT RUNNING: no tmux session '${SESSION}'."
    fi
    exit 0
    ;;
  stop)
    tmux kill-session -t "${SESSION}" 2>/dev/null && echo "Stopped '${SESSION}'." || echo "No session '${SESSION}'."
    exit 0
    ;;
  start)
    : # fall through
    ;;
  *)
    echo "Usage: $0 [start|attach|logs|status|stop]" >&2
    exit 2
    ;;
esac

# -------------------------------- start ------------------------------------
if ! command -v tmux >/dev/null 2>&1; then
  echo "ERROR: tmux not found." >&2; exit 1
fi
if [ ! -x "${MAMBA_BIN}" ]; then
  echo "ERROR: mamba not found at ${MAMBA_BIN}" >&2; exit 1
fi
if [ ! -f "${NOTEBOOK}" ]; then
  echo "ERROR: notebook not found at ${NOTEBOOK}" >&2; exit 1
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "A session '${SESSION}' is already running. Use '$0 attach' or '$0 stop'." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/inference_${TS}.log"

# Generate an inner runner script so the tmux pane executes a single file
# (avoids fragile nested shell quoting). Executed with the Prithvi env via
# `mamba run`, which activates the environment for the whole process.
RUNNER="${LOG_DIR}/.runner_${TS}.sh"
cat > "${RUNNER}" <<EOF
#!/usr/bin/env bash
set -o pipefail
cd '${REPO_ROOT}'
echo "[runner] start: \$(date)"
echo "[runner] env: ${CONDA_ENV}"
'${MAMBA_BIN}' run -n '${CONDA_ENV}' python -m jupyter nbconvert \\
  --to notebook --execute --inplace \\
  --ExecutePreprocessor.timeout=${CELL_TIMEOUT} \\
  '${NOTEBOOK}'
rc=\$?
echo "[runner] nbconvert exit: \${rc} (\$(date))"
exit \${rc}
EOF
chmod +x "${RUNNER}"

# Run the runner, tee output to the log, then drop into an interactive shell
# so the pane stays open for inspection after completion.
tmux new-session -d -s "${SESSION}" \
  "bash '${RUNNER}' 2>&1 | tee '${LOG}'; echo; echo '[done] Ctrl-b d to detach'; exec bash"

echo "Started inference in tmux session '${SESSION}'."
echo "  Env      : ${CONDA_ENV} (via mamba run)"
echo "  Log file : ${LOG}"
echo "  Tail log : $0 logs"
echo "  Attach   : $0 attach   (detach with Ctrl-b then d)"
echo "  Status   : $0 status"
echo "  Stop     : $0 stop"
