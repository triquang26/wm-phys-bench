#!/usr/bin/env bash
# scripts/setup_env.sh — create the warp-score conda env on a new machine.
#
# Usage:
#   bash scripts/setup_env.sh              # creates 'warp-score' env + installs everything
#   bash scripts/setup_env.sh --check      # dry-run: just prints what will be done
#   CUDA_VER=12.4 bash scripts/setup_env.sh  # override CUDA wheel index
#
# After this script, activate with:
#   conda activate warp-score
#   export PYTHONPATH=$(pwd)/third_party/RoMaV2/src:$PYTHONPATH

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
ENV_NAME="warp-score"
PYTHON_VER="3.10"
# Detect CUDA version from driver unless overridden
if [ -z "${CUDA_VER:-}" ]; then
  CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[\d.]+" | cut -d. -f1,2 || echo "12.1")
fi
# Map CUDA version to PyTorch wheel index
case "$CUDA_VER" in
  11.8) TORCH_INDEX="https://download.pytorch.org/whl/cu118" ;;
  12.1) TORCH_INDEX="https://download.pytorch.org/whl/cu121" ;;
  12.4) TORCH_INDEX="https://download.pytorch.org/whl/cu124" ;;
  12.6) TORCH_INDEX="https://download.pytorch.org/whl/cu126" ;;
  *)    TORCH_INDEX="https://download.pytorch.org/whl/cu121" ; echo "[warn] Unknown CUDA $CUDA_VER, defaulting to cu121 wheel" ;;
esac

CHECK_ONLY=${1:-}

echo "============================================================"
echo " warp-score env setup"
echo "   conda env : $ENV_NAME"
echo "   python    : $PYTHON_VER"
echo "   CUDA      : $CUDA_VER  (torch index: $TORCH_INDEX)"
echo "============================================================"
[ "$CHECK_ONLY" = "--check" ] && { echo "[check] dry-run — exiting"; exit 0; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------------------------------------------------------------------
# 1. Create conda env
# ---------------------------------------------------------------------------
if conda env list | grep -qE "^${ENV_NAME}\b"; then
  echo "[1] conda env '$ENV_NAME' already exists — skipping create"
else
  echo "[1] creating conda env '$ENV_NAME' (Python $PYTHON_VER)..."
  conda create -y -n "$ENV_NAME" python="$PYTHON_VER"
fi

PY="$(conda run -n "$ENV_NAME" which python)"
PIP="$(conda run -n "$ENV_NAME" which pip)"

# ---------------------------------------------------------------------------
# 2. Install PyTorch + torchvision matching CUDA
# ---------------------------------------------------------------------------
echo "[2] installing torch + torchvision (CUDA $CUDA_VER)..."
conda run -n "$ENV_NAME" pip install -q \
  torch torchvision \
  --index-url "$TORCH_INDEX"

# ---------------------------------------------------------------------------
# 3. Install all project Python deps
# ---------------------------------------------------------------------------
echo "[3] installing project dependencies..."
conda run -n "$ENV_NAME" pip install -q \
  "numpy>=2.0" \
  "scipy>=1.11" \
  "opencv-python-headless>=4.8" \
  "tqdm>=4.65" \
  "pyyaml>=6.0" \
  "matplotlib>=3.7" \
  "scikit-learn>=1.3" \
  "Pillow>=12.0" \
  "transformers>=5.0" \
  "einops>=0.8.1" \
  "rich>=14.0" \
  "huggingface_hub>=1.0"

# ---------------------------------------------------------------------------
# 4. Install RoMaV2 from third_party/
# ---------------------------------------------------------------------------
ROMAV2_DIR="$REPO_ROOT/third_party/RoMaV2"
if [ -d "$ROMAV2_DIR" ]; then
  echo "[4] installing RoMaV2 from $ROMAV2_DIR..."
  conda run -n "$ENV_NAME" pip install -q -e "$ROMAV2_DIR"
else
  echo "[4] WARN: $ROMAV2_DIR not found — RoMaV2 will not be installed"
  echo "    Make sure to add it to PYTHONPATH manually or clone it:"
  echo "    git submodule update --init third_party/RoMaV2"
fi

# ---------------------------------------------------------------------------
# 5. Install warp_score itself (editable)
# ---------------------------------------------------------------------------
echo "[5] installing warp_score (editable)..."
conda run -n "$ENV_NAME" pip install -q -e "$REPO_ROOT"

# ---------------------------------------------------------------------------
# 6. Verify
# ---------------------------------------------------------------------------
echo "[6] verifying install..."
conda run -n "$ENV_NAME" python - <<'EOF'
import sys, torch, numpy, cv2, scipy, sklearn, transformers, einops
print(f"  python       : {sys.version.split()[0]}")
print(f"  torch        : {torch.__version__}")
print(f"  cuda avail   : {torch.cuda.is_available()} ({torch.version.cuda})")
print(f"  numpy        : {numpy.__version__}")
print(f"  opencv       : {cv2.__version__}")
print(f"  scipy        : {scipy.__version__}")
print(f"  transformers : {transformers.__version__}")
print(f"  einops       : {einops.__version__}")
from warp_score.config import WarpScoreConfig
print(f"  warp_score   : OK")
try:
    from romav2.model import RoMa
    print(f"  RoMaV2       : OK")
except ImportError:
    print(f"  RoMaV2       : NOT FOUND (add third_party/RoMaV2/src to PYTHONPATH)")
EOF

echo ""
echo "============================================================"
echo " Done! Activate with:"
echo "   conda activate $ENV_NAME"
echo "   export PYTHONPATH=\$(pwd)/third_party/RoMaV2/src:\$PYTHONPATH"
echo "============================================================"
