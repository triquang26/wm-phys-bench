#!/usr/bin/env bash
# run_all.sh — thin wrapper around `make all` for backward compat.
#
# Prefer using the Makefile directly:
#   make help        # see all targets
#   make all         # full pipeline
#   make smoke       # smoke test (1 prompt × 5 videos)
#   make full        # full gen
#   make harvest     # videos → image_no_bg/
#
# Variables (pass-through to Makefile):
#   N_GPUS=1 N_VIDEOS=50 N_PROMPTS=23 MODEL_SIZE=2B bash run_all.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT}"

# Activate venv if it exists
if [ -f cosmos-predict2/.venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source cosmos-predict2/.venv/bin/activate
fi

exec make all
