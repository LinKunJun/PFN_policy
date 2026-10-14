#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Standard 8x MTT S5000 / 2x EPYC 9T34 deployment defaults.
# Each rank starts 16 persistent single-thread CPU simulators and up to 4
# CPU compute threads: ~160 slots across 8 ranks, leaving headroom on 256
# logical threads for the OS, MUSA runtime and communication.
# Environment overrides are supported for host-specific scheduling.
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
CPU_THREADS=${CPU_THREADS:-4}
ENV_WORKERS=${ENV_WORKERS:-16}
CONFIG=${CONFIG:-configs/default.json}
OBJECTIVE=${OBJECTIVE:-both}
RESUME=${RESUME:-}
PYTHON=${PYTHON:-python}
export NPROC_PER_NODE
export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export PYTHONUNBUFFERED=1

ACTION=${1:-start}
mkdir -p logs
LATEST=logs/pretrain_latest.txt

case "$ACTION" in
  status|logs|raw)
    [[ -f "$LATEST" ]] || { echo 'No run recorded by this launcher yet.'; exit 1; }
    mapfile -t INFO < "$LATEST"
    printf 'PID: %s\nOutput: %s\nLog: %s\n' "${INFO[0]}" "${INFO[1]}" "${INFO[2]}"
    if [[ "$ACTION" == status ]]; then
      if kill -0 "${INFO[0]}" 2>/dev/null; then
        ps -p "${INFO[0]}" -o pid,etime,args
      else
        echo 'Launcher has exited. Inspect the end of the log for completion or errors.'
      fi
      tail -n 15 -- "${INFO[2]}"
    elif [[ "$ACTION" == raw ]]; then
      tail -n 80 -F -- "${INFO[2]}"
    else
      # Hide ONLY these two known advisory lines in the VIEW. Full log is preserved.
      tail -n 80 -F -- "${INFO[2]}" | grep --line-buffered -v -E \
        'Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters|Warning: Time stats are currently only collected for CPU and CUDA devices'
    fi
    exit 0 ;;
  start|check|dry-run) ;;
  *) echo 'Usage: bash PFN_network_interference_musa_start.sh [start|check|status|logs|raw|dry-run]'; exit 2 ;;
esac

[[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ && "$CPU_THREADS" =~ ^[1-9][0-9]*$ && "$ENV_WORKERS" =~ ^[0-9]+$ ]] || {
  echo 'Invalid process/thread settings.'; exit 2;
}
[[ "$OBJECTIVE" == both || "$OBJECTIVE" == static || "$OBJECTIVE" == sequential ]] || exit 2
[[ -f "$CONFIG" && -f pfn_network/distributed.py ]] || { echo 'Run from an installed project.'; exit 2; }
[[ -z "$RESUME" || ( "$OBJECTIVE" != both && -f "$RESUME" ) ]] || {
  echo 'RESUME requires an existing checkpoint and OBJECTIVE=static or sequential.'; exit 2;
}
STAMP=$(date +%Y%m%d_%H%M%S)_$$
RUN_DIR="runs/pretrain_${STAMP}"
LOG="logs/pretrain_${STAMP}.log"
CMD=("$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="$NPROC_PER_NODE")
if [[ "$ACTION" == check ]]; then
  CMD+=(scripts/check_distributed.py --device musa --output "runs/ddp_check_${STAMP}")
else
  CMD+=(-m pfn_network.train_eval.run --device musa --stage train
        --config "$CONFIG" --objective "$OBJECTIVE" --env-workers "$ENV_WORKERS" --output "$RUN_DIR")
  [[ -z "$RESUME" ]] || CMD+=(--resume "$RESUME")
fi
printf 'GPUs/processes=%s; CPU threads per process=%s; environment workers per rank=%s\n' \
  "$NPROC_PER_NODE" "$CPU_THREADS" "$ENV_WORKERS"
printf 'Command: '; printf '%q ' "${CMD[@]}"; printf '\n'
[[ "$ACTION" != dry-run ]] || exit 0

# The child inherits this lock. Repeated starts through this script are rejected.
command -v flock >/dev/null || { echo 'flock is required (Ubuntu package: util-linux).'; exit 2; }
exec 9>logs/pretrain.lock
flock -n 9 || { echo 'A training/check process launched by this script is still running.'; exit 1; }
"$PYTHON" -c 'import os, torch, torch_musa; n=int(os.environ["NPROC_PER_NODE"]); print("CPU affinity:",len(os.sched_getaffinity(0)),"logical CPUs; visible GPUs:",torch.musa.device_count()); assert torch.musa.is_available() and torch.musa.device_count() >= n'
if [[ "$ACTION" == check ]]; then
  "${CMD[@]}" 2>&1 | tee "logs/ddp_check_${STAMP}.log"
  exit 0
fi
{
  printf 'CPU_THREADS=%s ENV_WORKERS=%s NPROC_PER_NODE=%s\n' "$CPU_THREADS" "$ENV_WORKERS" "$NPROC_PER_NODE"
  printf 'Command: '; printf '%q ' "${CMD[@]}"; printf '\n'
} > "$LOG"
nohup "${CMD[@]}" >> "$LOG" 2>&1 < /dev/null &
TRAIN_PID=$!
printf '%s\n' "$TRAIN_PID" > "logs/pretrain_${STAMP}.pid"
printf '%s\n%s\n%s\n' "$TRAIN_PID" "$RUN_DIR" "$LOG" > "${LATEST}.tmp"
mv -- "${LATEST}.tmp" "$LATEST"
printf 'Submitted PID: %s\nOutput: %s\nLog: %s\n' "$TRAIN_PID" "$RUN_DIR" "$LOG"
echo 'View: bash PFN_network_interference_musa_start.sh logs'
