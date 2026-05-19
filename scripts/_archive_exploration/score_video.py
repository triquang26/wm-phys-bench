#!/usr/bin/env python3
"""Production-style CLI: throw a video in, get a robust hallucination score out.

Usage (groot env):
    python scripts/score_video.py path/to/video.mp4 \
        --cache_dir paper-physical-gr1/ref_cache \
        --n_frames 10

Output:
    JSON with video_h_score (robust trimmed mean), video_h_peak (80th
    percentile), is_hallucination flag, and per-frame breakdown.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

REPO_ROOT = Path("/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path, help="Path to video (.mp4) OR directory of .png frames")
    ap.add_argument("--cache_dir", type=Path,
                    default=REPO_ROOT / "paper-physical-gr1" / "ref_cache")
    ap.add_argument("--n_frames", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Decision threshold on video_h_score (trimmed mean).")
    ap.add_argument("--out", type=Path, default=None, help="Optional JSON output path")
    args = ap.parse_args()

    from warp_score.video_scorer import VideoScorer

    print(f"Loading reference cache from {args.cache_dir} …")
    scorer = VideoScorer.from_cache(args.cache_dir, threshold=args.threshold)
    print(f"  Pool size: {len(scorer.cache.pool_paths)}")
    print(f"  cycle null: n_mean={len(scorer.cache.cycle_null_mean)}, "
          f"n_peak={len(scorer.cache.cycle_null_peak)}")
    print(f"  jaccard null: n={len(scorer.cache.jaccard_null) if scorer.cache.jaccard_null is not None else 0}")

    is_dir = args.video.is_dir()
    if is_dir:
        result = scorer.score(args.video, n_frames=args.n_frames, frame_dir=args.video)
    else:
        result = scorer.score(args.video, n_frames=args.n_frames)

    print(f"\n=== Video: {args.video.name} ===")
    print(f"  Frames processed:   {result.n_frames}")
    print(f"  H_score (robust):   {result.video_h_score:.4f}")
    print(f"  H_score (p80 peak): {result.video_h_peak:.4f}")
    print(f"  Decision (>{args.threshold}):  "
          f"{'HALLUCINATION' if result.is_hallucination else 'CLEAN'}")
    print(f"\n  Aggregator breakdown:")
    for k, v in result.aggregate_breakdown.items():
        print(f"    {k:14s} {v:.4f}")
    print(f"\n  Per-frame H_scores:")
    for fs in result.per_frame:
        print(f"    frame {fs.frame_idx:4d}: H={fs.h_score:.4f}  "
              f"p_cycle={fs.p_cycle if fs.p_cycle is not None else 'NA':.4} "
              f"p_jaccard={fs.p_jaccard if fs.p_jaccard is not None else 'NA':.4}")

    if args.out:
        out = {
            "video": str(args.video),
            "video_h_score": result.video_h_score,
            "video_h_peak":  result.video_h_peak,
            "is_hallucination": result.is_hallucination,
            "threshold":     result.threshold,
            "aggregate":     result.aggregate_breakdown,
            "per_frame":     [asdict(f) for f in result.per_frame],
        }
        args.out.write_text(json.dumps(out, indent=2))
        print(f"\nJSON → {args.out}")


if __name__ == "__main__":
    main()
