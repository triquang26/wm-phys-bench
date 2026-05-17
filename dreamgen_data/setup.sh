#!/usr/bin/env bash
# setup.sh -- install cosmos-predict2 + download checkpoints used by the dreamgen pipeline.
# Models downloaded:
#   - nvidia/Cosmos-Predict2-14B-Video2World   (HIGH profile DiT)
#   - nvidia/Cosmos-Predict2-2B-Video2World    (optional; smaller/faster)
#   - nvidia/Cosmos-Predict2-14B-Sample-GR00T-Dreams-GR1  (GR00T fine-tune DiT)
#   - nvidia/Cosmos-Reason1-7B                (prompt refiner for HIGH profile)
#   - google-t5/t5-11b                        (text encoder)
#
# This is idempotent: each step short-circuits if its product already exists.
# Logs everything to setup.log (in addition to stdout).
#
# REQUIREMENTS
#   - NVIDIA GPU with driver compatible with CUDA 12.6 runtime
#     (driver 535.x = CUDA 12.4 max -- wheels ship their own runtime so this works)
#   - Linux x86-64, glibc >= 2.31
#   - uv on PATH (auto-installed if missing)
#   - ~120 GB free disk for full checkpoint family
#   - HF_TOKEN env var with access to gated repos (Llama-Guard-3-8B):
#       export HF_TOKEN=hf_xxx
#
# NETWORK
#   The cosmos-predict2 `[cu126]` install pulls torch/torchvision/flash-attn/
#   transformer-engine/apex/natten from a custom Nvidia index hosted on
#   `nvidia-cosmos.github.io` (GitHub Pages, 185.199.108-111.153).
#   If that host is firewalled, fall back to:
#     - torch / torchvision from https://download.pytorch.org/whl/cu126
#     - flash-attn 2.6.3 prebuilt wheels (cu123, torch 2.4) from
#       https://github.com/Dao-AILab/flash-attention/releases  -- which
#       requires *downgrading* torch to 2.4. cosmos-predict2 pins torch==2.6.0,
#       so that path needs a `--no-deps` install + manual override.
#   See ENV.md for the current blocker.
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Tee everything to setup.log (overwrite per run, easier to scan)
exec > >(tee setup.log) 2>&1
echo "===== setup.sh start: $(date -Iseconds) ====="

# ---------------------------------------------------------------------------
# 1. HF_TOKEN check
# ---------------------------------------------------------------------------
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set. Source ../.env.dreamgen first:"
    echo "       set -a && source ../.env.dreamgen && set +a"
    exit 1
fi
echo "[setup] HF_TOKEN: <set, first 6=${HF_TOKEN:0:6}>"

# ---------------------------------------------------------------------------
# 2. Ensure uv is installed
# ---------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi
echo "[setup] uv: $(command -v uv) $(uv --version)"

# ---------------------------------------------------------------------------
# 3. Clone cosmos-predict2 (skip if exists)
# ---------------------------------------------------------------------------
if [ ! -d cosmos-predict2 ]; then
    echo "[setup] cloning cosmos-predict2..."
    git clone --depth 1 https://github.com/nvidia-cosmos/cosmos-predict2.git
else
    echo "[setup] cosmos-predict2 already cloned, skipping clone"
fi
COSMOS_SHA=$(git -C cosmos-predict2 rev-parse HEAD)
echo "$COSMOS_SHA" > _cosmos_sha.txt
echo "[setup] cosmos-predict2 SHA: $COSMOS_SHA"

# ---------------------------------------------------------------------------
# 4. Create venv with uv (Python 3.11; cosmos-predict2 supports 3.10/3.11)
# ---------------------------------------------------------------------------
VENV_DIR="cosmos-predict2/.venv"
VENV_PY="$VENV_DIR/bin/python"
if [ ! -f "$VENV_PY" ] || ! "$VENV_PY" -m pip --version >/dev/null 2>&1; then
    echo "[setup] creating uv venv (python 3.11, seeded with pip)..."
    # --seed installs pip/setuptools/wheel into the venv so the venv-local pip
    # is available; without it, modern uv venvs are pip-less.
    rm -rf "$VENV_DIR"
    uv venv "$VENV_DIR" --python 3.11 --seed
else
    echo "[setup] venv already exists at $VENV_DIR (pip OK)"
fi
PIP=("$VENV_PY" -m pip)

# ---------------------------------------------------------------------------
# 5. Install wrapper requirements (HF hub + opencv + datasets)
# ---------------------------------------------------------------------------
echo "[setup] installing wrapper requirements..."
"${PIP[@]}" install --upgrade pip
"${PIP[@]}" install -r requirements.txt

# ---------------------------------------------------------------------------
# 6. Install cosmos-predict2 + its CUDA stack.
#    Primary path (uses Nvidia's cu126 wheel index):
# ---------------------------------------------------------------------------
COSMOS_INDEX="https://nvidia-cosmos.github.io/cosmos-dependencies/cu126_torch260/simple"
echo "[setup] testing reachability of $COSMOS_INDEX ..."
if "$VENV_PY" - <<'PY' >/dev/null 2>&1
import urllib.request
urllib.request.urlopen(
    "https://nvidia-cosmos.github.io/cosmos-dependencies/cu126_torch260/simple/cosmos-predict2/",
    timeout=15,
)
PY
then
    echo "[setup] cosmos-cu126 index reachable -- using primary install path"
    "${PIP[@]}" install -U "cosmos-predict2[cu126]==1.0.9" \
        --extra-index-url "$COSMOS_INDEX"
else
    echo "[setup] WARN: cosmos-cu126 index NOT reachable from this host."
    echo "       (Likely firewalled GitHub-Pages range 185.199.108-111.153)"
    echo "       Falling back to pytorch.org + PyPI install (no flash-attn/apex/natten)."
    echo "       Inference WILL still attempt to run but performance / numerics"
    echo "       may differ from the reference setup."

    # 6a. Install torch+cu126 from pytorch.org
    "${PIP[@]}" install \
        --index-url https://download.pytorch.org/whl/cu126 \
        --extra-index-url https://pypi.org/simple \
        "torch==2.6.0" "torchvision==0.21.0"

    # 6b. Install cosmos-predict2 itself + its python-only deps from PyPI.
    #     Skip the [cu126] extra (those deps live only on the unreachable index).
    "${PIP[@]}" install "cosmos-predict2==1.0.9" \
        --extra-index-url https://pypi.org/simple

    # 6c. Optional flash-attn from PyPI (will compile from sdist -- slow, needs
    #     matching nvcc; will likely fail without CUDA 12.6 toolkit). Skipped
    #     unless caller sets COSMOS_BUILD_FLASHATTN=1.
    if [ "${COSMOS_BUILD_FLASHATTN:-0}" = "1" ]; then
        echo "[setup] building flash-attn 2.6.3 from source (slow)..."
        "${PIP[@]}" install --no-build-isolation flash-attn==2.6.3 || \
            echo "[setup] WARN: flash-attn build failed; continuing without it"
    fi
fi

# Editable install of the clone so generate.py picks up `imaginaire.utils.io`
# and any in-repo modules.
"${PIP[@]}" install --no-deps -e cosmos-predict2/ || \
    echo "[setup] WARN: editable install of cosmos-predict2/ failed (continuing)"

# ---------------------------------------------------------------------------
# 7. HuggingFace login (writes ~/.cache/huggingface/token)
# ---------------------------------------------------------------------------
HFCLI="$VENV_DIR/bin/huggingface-cli"
if [ -x "$HFCLI" ]; then
    echo "[setup] logging in to HuggingFace..."
    "$HFCLI" login --token "$HF_TOKEN" --add-to-git-credential || true
else
    echo "[setup] WARN: huggingface-cli not found at $HFCLI -- using HF_TOKEN env var only"
fi

# ---------------------------------------------------------------------------
# 8. Download checkpoints.
#    cosmos-predict2 expects: $PWD/checkpoints/{org}/{repo}/...
#    For Video2World-2B inference we need:
#      - nvidia/Cosmos-Predict2-2B-Video2World    (the DiT, ~50 GB)
#      - google-t5/t5-11b                         (text encoder, ~45 GB)
#      - nvidia/Cosmos-Guardrail1                 (guardrail, ~5 GB)        [optional if disable_guardrail]
#      - meta-llama/Llama-Guard-3-8B              (~16 GB, gated)           [only with guardrail]
#      - nvidia/Cosmos-Reason1-7B                 (~15 GB, prompt refiner)  [optional if disable_prompt_refiner]
#
#    We download via huggingface_hub.snapshot_download (always reachable on
#    this host) and pin the revision recorded by cosmos-predict2 upstream.
# ---------------------------------------------------------------------------
mkdir -p checkpoints
echo "[setup] downloading checkpoints..."
"$VENV_PY" - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

# Pinned revisions copied from cosmos-predict2 scripts/download_checkpoints.py
PINS = {
    "nvidia/Cosmos-Predict2-14B-Video2World": "03b03a377fede782647afac998f674d9f358e319",
    "nvidia/Cosmos-Predict2-2B-Video2World":  "f50c09f5d8ab133a90cac3f4886a6471e9ba3f18",
    "google-t5/t5-11b":                       "90f37703b3334dfe9d2b009bfcbfbf1ac9d28ea3",
    "nvidia/Cosmos-Reason1-7B":               "8fe96c1fa10db9e666b6fa6a87fea57dd9635649",
    "nvidia/Cosmos-Guardrail1":               "d6d4bfa899a71454a700907664f3e88f503950cf",
}
# Llama-Guard-3-8B is only needed with guardrail enabled; disabled in profiles.py.

root = Path("checkpoints").resolve()
for repo_id, rev in PINS.items():
    out = root / repo_id  # nested {org}/{repo}/ matches CHECKPOINTS_DIR expectation
    if (out / "config.json").exists() or any(out.glob("*.safetensors")) or any(out.glob("*.pt")):
        print(f"[ckpt] skip (exists): {repo_id}")
        continue
    out.mkdir(parents=True, exist_ok=True)
    print(f"[ckpt] downloading {repo_id}@{rev[:8]} -> {out}")
    snapshot_download(
        repo_id=repo_id,
        revision=rev,
        local_dir=str(out),
        max_workers=4,
    )
    print(f"[ckpt] done: {repo_id}")

# Record the headline revision for reproducibility
Path("_ckpt_revision.txt").write_text(PINS["nvidia/Cosmos-Predict2-14B-Video2World"])
print("[ckpt] all done")
PY

# ---------------------------------------------------------------------------
# 9. Sanity import
#    Verified path: cosmos_predict2.pipelines.video2world.Video2WorldPipeline
#    (Confirmed by reading cosmos-predict2/cosmos_predict2/pipelines/video2world.py:255)
# ---------------------------------------------------------------------------
echo "[setup] sanity-checking cosmos-predict2 import..."
"$VENV_PY" -c "
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from cosmos_predict2.configs.base.config_video2world import get_cosmos_predict2_video2world_pipeline
from imaginaire.utils.io import save_image_or_video
print('cosmos-predict2 OK -- Video2WorldPipeline / save_image_or_video importable')
" || {
    echo "[setup] WARN: sanity import failed."
    echo "       Look in cosmos-predict2/cosmos_predict2/pipelines/ for the real layout"
    echo "       and update generate.py:_import_cosmos()."
}

echo "===== setup.sh done: $(date -Iseconds) ====="
echo "Setup complete. HF_TOKEN: <set>. Cosmos SHA: $COSMOS_SHA"
