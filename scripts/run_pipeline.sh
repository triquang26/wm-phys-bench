#!/usr/bin/env bash
# Full end-to-end pipeline — run from feature_matching_eval_hallucination/ root.
#
# Data layout (all under data/, .gitignored):
#   data/cosmos_synthetic_data/{high,low}/<task>.mp4   ← source videos
#   data/cosmos_frames_raw/{high,low}/<task>/*.png     ← extracted frames
#   data/image_no_bg/{high,low}/<task>/*.png           ← SAM3 bg-removed (label 0/1)
#
# Steps:
#   1. Extract 50 frames per video  (fast, cpu)
#   2. SAM3 bg removal              (gpu, ~1-2h)
#   3. Build labels.csv
#   4. Calibrate warp_score         (gpu, ~3-4h)
#   5. Detect on low/ frames        (gpu, ~1-2h)
#   6. Eval AUROC/AP
#
# Usage:  conda activate groot && bash scripts/run_pipeline.sh
# Resume: re-run — all steps skip already-done work

set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$REPO/data"
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
DONE_BG=$(find "$DATA/image_no_bg" -name "*.png" 2>/dev/null | wc -l || echo 0)
if [ "$DONE_BG" -ge "$EXPECTED" ]; then
    echo "  Already done ($DONE_BG frames). Skipping."
else
    python scripts/run_bg_removal_batch.py \
        --src_root "$DATA/cosmos_frames_raw" \
        --out_root "$DATA/image_no_bg"
fi

# ─────────────────────────────────────────────────────────────────────────────
echo ""; echo "=== STEP 3: Build weak labels CSV ==="
if [ ! -f labels.csv ]; then
    python scripts/build_weak_labels.py \
        --high_dir "$DATA/image_no_bg/high" \
        --low_dir  "$DATA/image_no_bg/low" \
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
        --high_dir      "$DATA/image_no_bg/high" \
        --artifacts_dir "$ARTIFACTS" \
        calibrate 2>&1 | tee "$ARTIFACTS/calibrate.log"
fi

# ─────────────────────────────────────────────────────────────────────────────
echo ""; echo "=== STEP 5: Detect ==="
if [ -f "$ARTIFACTS/summary.csv" ]; then
    echo "  summary.csv exists. Skipping (delete to re-run)."
else
    PYTHONPATH="$ROMAV2:$PYTHONPATH" python -m warp_score \
        --low_dir       "$DATA/image_no_bg/low" \
        --artifacts_dir "$ARTIFACTS" \
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
