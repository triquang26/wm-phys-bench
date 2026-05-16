#!/usr/bin/env bash
# setup.sh — install cosmos-predict2 and download checkpoints for GR00T-Dreams-GR1.
# Tested against cosmos-predict2 main branch (last verified release: Aug-Dec 2025).
#
# Requirements before running:
#   - CUDA 12.6+, NVIDIA driver supporting Hopper or newer
#   - Python 3.10
#   - ~250 GB free disk for 14B GR1 + base ckpts (or ~50 GB if --only_2b)
#   - HF token with access to Llama-Guard-3-8B (free, requires Meta T&C accept)
#       export HF_TOKEN=hf_xxx
#
# Optional env vars:
#   COSMOS_DIR              override clone location
#   ONLY_2B=1               skip 14B checkpoint download (saves ~200GB)
#   ONLY_GR1=1              skip base checkpoint download
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
COSMOS_DIR="${ROOT}/cosmos-predict2"
CKPT_DIR="${COSMOS_DIR}/checkpoints"

# ---------------------------------------------------------------------------
# 1. Clone cosmos-predict2 (engine of DreamGen / GR00T-Dreams)
# ---------------------------------------------------------------------------
if [ ! -d "${COSMOS_DIR}" ]; then
    echo "[setup] cloning cosmos-predict2…"
    git clone https://github.com/nvidia-cosmos/cosmos-predict2.git "${COSMOS_DIR}"
else
    echo "[setup] cosmos-predict2 already cloned, skipping"
fi

# ---------------------------------------------------------------------------
# 2. Install via uv (recommended by Nvidia)
# ---------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] installing uv…"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi

cd "${COSMOS_DIR}"
uv venv --python 3.10 --allow-existing
# CUDA 12.6 + torch 2.6 wheels (adjust if your CUDA differs)
uv pip install -U "cosmos-predict2[cu126]" \
    --extra-index-url https://nvidia-cosmos.github.io/cosmos-dependencies/cu126_torch260/simple

# Extra deps used by our wrapper scripts
uv pip install -U "datasets>=3.0" "huggingface_hub[cli]" "opencv-python-headless" "pillow" "tqdm"

# ---------------------------------------------------------------------------
# 3. HuggingFace auth (needed for Llama-Guard-3-8B guardrail)
# ---------------------------------------------------------------------------
if [ -z "${HF_TOKEN:-}" ]; then
    echo "[setup] WARNING: HF_TOKEN env var not set."
    echo "[setup] Accept Llama-Guard-3-8B terms at"
    echo "        https://huggingface.co/meta-llama/Llama-Guard-3-8B"
    echo "        then: export HF_TOKEN=hf_xxx ; rerun setup.sh"
fi
uv run huggingface-cli login --token "${HF_TOKEN:-}" --add-to-git-credential || true

# ---------------------------------------------------------------------------
# 4. Download checkpoints
#    - Cosmos-Predict2-14B-Video2World          (base — used for `hallucinate` profile)
#    - Cosmos-Predict2-14B-Sample-GR00T-Dreams-GR1 (post-trained — used for `high` profile)
#    Resolution 480p / 16fps because that's what the GR1 sample checkpoint supports.
# ---------------------------------------------------------------------------
mkdir -p "${CKPT_DIR}"

# Default: 2B (single-GPU friendly, ~50GB). Set USE_14B=1 to download full 14B (~250GB).
MODEL_TYPES=()
if [ -n "${USE_14B:-}" ]; then
    MODEL_TYPES+=(video2world sample_gr00t_dreams_gr1)
    MODEL_SIZE="14B"
    echo "[setup] downloading 14B GR1 + base ckpts (~250GB) — set unset USE_14B for 2B only"
else
    MODEL_TYPES+=(video2world)
    MODEL_SIZE="2B"
    echo "[setup] downloading 2B base ckpt only (~50GB). Set USE_14B=1 for full quality."
fi

uv run python -m scripts.download_checkpoints \
    --model_types "${MODEL_TYPES[@]}" \
    --model_sizes "${MODEL_SIZE}" \
    --resolution 480 \
    --fps 16

echo "[setup] done."
echo "[setup] cosmos-predict2 lives at: ${COSMOS_DIR}"
echo "[setup] activate env with: cd ${COSMOS_DIR} && source .venv/bin/activate"
echo "[setup] next: cd .. && python prepare_data.py --out_dir data --max_items 100"
