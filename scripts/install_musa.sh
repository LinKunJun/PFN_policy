#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
# Check the vendor stack before installing anything. Do not replace torch wheels.
"$PYTHON" scripts/check_environment.py --device musa --stack-only
"$PYTHON" -m venv --system-site-packages .venv-musa
.venv-musa/bin/python -m pip install -r requirements-minimal.txt
# Run from the project root; an editable install is not required.
.venv-musa/bin/python scripts/check_environment.py --device musa
echo 'Ready for smoke tests. Activate with: source .venv-musa/bin/activate'
