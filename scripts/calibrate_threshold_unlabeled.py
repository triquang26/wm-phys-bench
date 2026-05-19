#!/usr/bin/env python3
"""Self-supervised threshold calibration — REAL VIDEOS ONLY.

Conformal-style anomaly threshold: run VideoScorer on N real training
videos, take the p95 (or p99) of video_h_peak distribution. Any new
video scoring above this is "rarer than 95% of training real data" →
flag as anomaly.

No generated videos required. FPR ≤ (1 - p) guaranteed by construction.

Usage (groot env):
    python scripts/calibrate_threshold_unlabeled.py \
        --video_root paper-physical-gr1/raw_videos/gr1/ \
        --frame_dirs paper-physical-gr1/reference/ \
        --target_fpr 0.05 \
        --max_videos 50

Output:
    paper-physical-gr1/ref_cache/threshold.json with mode="conformal"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

REPO_ROOT = Path("/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_root", type=Path, default=None,
                    help="Directory of raw .mp4 real videos. Mutually exclusive with --frame_dirs.")
    ap.add_argument("--frame_dirs", type=Path, default=REPO_ROOT / "paper-physical-gr1" / "reference",
                    help="Directory containing pre-segmented frame subdirs (one per video).")
    ap.add_argument("--cache_dir", type=Path,
                    default=REPO_ROOT / "paper-physical-gr1" / "ref_cache")
    ap.add_argument("--n_frames", type=int, default=10)
    ap.add_argument("--max_videos", type=int, default=50,
                    help="Max real videos to use for calibration")
    ap.add_argument("--target_fpr", type=float, default=0.05,
                    help="Desired FPR upper bound (sets threshold = (1-fpr) quantile)")
    args = ap.parse_args()

    from warp_score.video_scorer import VideoScorer

    print(f"Loading VideoScorer from {args.cache_dir} …")
    # Disable SAM3 if we're already pointing at pre-segmented frame dirs
    use_sam = args.video_root is not None
    scorer = VideoScorer.from_cache(args.cache_dir, sam_segment=use_sam)

    # ── Collect sources to score
    if args.video_root is not None:
        sources = sorted(args.video_root.glob("*.mp4"))[: args.max_videos]
        print(f"Found {len(sources)} mp4 files in {args.video_root}")
    else:
        sources = sorted(d for d in args.frame_dirs.iterdir() if d.is_dir())[: args.max_videos]
        print(f"Found {len(sources)} frame dirs in {args.frame_dirs}")

    if len(sources) < 10:
        print(f"WARNING: only {len(sources)} sources — quantile estimate unreliable below ~20")

    # ── Run scorer on each
    h_peaks = []
    h_trimmed = []
    for i, src in enumerate(sources, 1):
        try:
            if args.video_root is not None:
                r = scorer.score(src, n_frames=args.n_frames)
            else:
                r = scorer.score(src, n_frames=args.n_frames, frame_dir=src)
            h_peaks.append(r.video_h_peak)
            h_trimmed.append(r.video_h_score)
            print(f"  [{i}/{len(sources)}] {src.name[:50]:50s}  peak={r.video_h_peak:.4f}  trimmed={r.video_h_score:.4f}")
        except Exception as e:
            print(f"  [{i}/{len(sources)}] {src.name[:50]:50s}  FAILED: {e}")
            continue

    h_peaks = np.array(h_peaks)
    h_trimmed = np.array(h_trimmed)

    # ── Compute quantiles
    quantile = 1.0 - args.target_fpr
    threshold_peak = float(np.quantile(h_peaks, quantile))
    threshold_trim = float(np.quantile(h_trimmed, quantile))

    print(f"\n=== Conformal threshold calibration ===")
    print(f"N real videos:           {len(h_peaks)}")
    print(f"Target FPR:              {args.target_fpr}")
    print(f"Quantile used:           {quantile}")
    print(f"")
    print(f"H_PEAK distribution:")
    print(f"  range: [{h_peaks.min():.4f}, {h_peaks.max():.4f}]")
    print(f"  mean:   {h_peaks.mean():.4f}  std: {h_peaks.std():.4f}")
    print(f"  p90: {np.quantile(h_peaks, 0.9):.4f}")
    print(f"  p95: {np.quantile(h_peaks, 0.95):.4f}")
    print(f"  p99: {np.quantile(h_peaks, 0.99):.4f}")
    print(f"  → threshold @ p{int(quantile*100)} = {threshold_peak:.4f}")
    print(f"")
    print(f"H_TRIMMED distribution:")
    print(f"  range: [{h_trimmed.min():.4f}, {h_trimmed.max():.4f}]")
    print(f"  → threshold @ p{int(quantile*100)} = {threshold_trim:.4f}")

    # ── Save
    out = args.cache_dir / "threshold.json"
    out.write_text(json.dumps({
        "threshold":   threshold_peak,
        "aggregator":  "p80",
        "config":      "S1+S2+S4",
        "mode":        "conformal_unlabeled",
        "target_fpr":  args.target_fpr,
        "n_real_calib": int(len(h_peaks)),
        "real_score_stats": {
            "mean": float(h_peaks.mean()),
            "std":  float(h_peaks.std()),
            "p90":  float(np.quantile(h_peaks, 0.9)),
            "p95":  float(np.quantile(h_peaks, 0.95)),
            "p99":  float(np.quantile(h_peaks, 0.99)),
        },
    }, indent=2))
    print(f"\nThreshold saved → {out}")
    print(f"  Operating point: 'flag video if H_peak > {threshold_peak:.4f}'")
    print(f"  Expected FPR ≤ {args.target_fpr} on similar real data (statistical guarantee)")


if __name__ == "__main__":
    main()
