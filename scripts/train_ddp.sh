#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
# Additional arguments include --config, --output, --objective and --resume.
python -m torch.distributed.run --standalone --nnodes=1 \
  --nproc-per-node="${NPROC_PER_NODE:-8}" \
  -m pfn_network.train_eval.run --device musa --stage train --env-workers 2 "$@"
