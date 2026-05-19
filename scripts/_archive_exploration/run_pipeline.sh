#!/usr/bin/env bash
# Full end-to-end pipeline — run from feature_matching_eval_hallucination/ root.
#
# Data layout (all under data/, .gitignored):
#   data/cosmos_synthetic_data/{high,low}/<task>.mp4    ← source videos
#   data/cosmos_frames_raw/{high,low}/<task>/*.png      ← extracted frames
#   data/reference/<task>/*.png                          ← calibration references (LOO)
#   data/query/high/<task>/*.png                         ← clean held-out queries (AUROC label=0)
#   data/query/low/<task>/*.png                          ← hallucinated queries (AUROC label=1)
#
# Steps:
#   1. Extract 50 frames per video  (fast, cpu)
#   2. SAM3 bg removal              (gpu, ~1-2h)
#   3. Build labels.csv
#   4. Calibrate warp_score         (gpu, ~3-4h)
#   5. Detect on query/{high,low}   (gpu, ~1-2h)
#   6. Eval AUROC/AP
#
# Usage:  conda activate groot && bash scripts/run_pipeline.sh
# Resume: re-run — all steps skip already-done work
#
# Prefer the python entry point: `python -m warp_score ...` (see Makefile).

set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$REPO/data"
REF_DIR="$DATA/reference"
QUERY_HIGH_DIR="$DATA/query/high"
QUERY_LOW_DIR="$DATA/query/low"
# RoMaV2 source path — override with: ROMAV2=/your/path bash scripts/run_pipeline.sh
ROMAV2=${ROMAV2:-/mnt/data/sftp/data/quangpt3/gcvwm/calibration/RoMaV2/src}
# Fallback: use bundled copy in third_party/ if external not found
if [ ! -d "$ROMAV2" ]; then
    ROMAV2="$REPO/third_party/RoMaV2/src"
fi
ARTIFACTS="$REPO/artifacts/v9_dreamgen"
mkdir -p "$ARTIFACTS"

cd "$REPO"

# ─────────────────────────────────────────────────────────────────────────────
echo ""; echo "=== STEP 1: Extract frames ==="
EXPECTED=2300   # 23 tasks × 50 frames × 2 splits
ACTUAL=$(find "$DATA/cosmos_frames_raw" -name "*.png" 2>/dev/null | wc -l || echo 0)
if [ "$ACTUAL" -ge "$EXPECTED" ]; then
    echo "  Already done ($ACTUAL frames). Skipping."
else
    python scripts/extract_frames.py \
        --src_root "$DATA/cosmos_synthetic_data" \
        --dst_root "$DATA/cosmos_frames_raw" \
        --n_frames 50
fi

# ─────────────────────────────────────────────────────────────────────────────
echo ""; echo "=== STEP 2: SAM3 background removal ==="
# (Output is now expected to live under data/reference and data/query/{high,low}
#  — the Agent A data migration step handles layout. This step still produces
#  bg-removed PNGs; see APPROACH.md for the mapping.)
DONE_BG=$(find "$REF_DIR" "$QUERY_HIGH_DIR" "$QUERY_LOW_DIR" -name "*.png" 2>/dev/null | wc -l || echo 0)
if [ "$DONE_BG" -ge "$EXPECTED" ]; then
    echo "  Already done ($DONE_BG frames). Skipping."
else
    python scripts/run_bg_removal_batch.py \
        --src_root "$DATA/cosmos_frames_raw" \
        --out_root "$DATA"
fi

# ─────────────────────────────────────────────────────────────────────────────
echo ""; echo "=== STEP 3: Build weak labels CSV ==="
if [ ! -f labels.csv ]; then
    python scripts/build_weak_labels.py \
        --query_high_dir "$QUERY_HIGH_DIR" \
        --query_low_dir  "$QUERY_LOW_DIR" \
        --out labels.csv
    echo "  Written: labels.csv"
else
    echo "  labels.csv exists. Skipping."
fi

# ─────────────────────────────────────────────────────────────────────────────
echo ""; echo "=== STEP 4: Calibrate ==="
if [ -f "$ARTIFACTS/calibration.npz" ]; then
    echo "  Calibration exists. Skipping (delete $ARTIFACTS/calibration.npz to re-run)."
else
    PYTHONPATH="$ROMAV2:$PYTHONPATH" python -m warp_score \
        --ref_dir       "$REF_DIR" \
        --artifacts_dir "$ARTIFACTS" \
        calibrate 2>&1 | tee "$ARTIFACTS/calibrate.log"
fi

# ─────────────────────────────────────────────────────────────────────────────
echo ""; echo "=== STEP 5: Detect ==="
if [ -f "$ARTIFACTS/summary.csv" ]; then
    echo "  summary.csv exists. Skipping (delete to re-run)."
else
    PYTHONPATH="$ROMAV2:$PYTHONPATH" python -m warp_score \
        --ref_dir        "$REF_DIR" \
        --query_high_dir "$QUERY_HIGH_DIR" \
        --query_low_dir  "$QUERY_LOW_DIR" \
        --artifacts_dir  "$ARTIFACTS" \
        detect 2>&1 | tee "$ARTIFACTS/detect.log"
fi

# ─────────────────────────────────────────────────────────────────────────────
echo ""; echo "=== STEP 6: Eval ==="
python -m warp_score \
    --artifacts_dir "$ARTIFACTS" \
    eval --labels labels.csv 2>&1 | tee "$ARTIFACTS/eval.log"

echo ""
echo "All done. Results in: $ARTIFACTS/"
echo "  calibration.npz  — per-task empirical null distributions"
echo "  summary.csv      — H_score + is_hallucination per frame"
echo "  eval_report.json — AUROC / AP / FPR@95TPR"
