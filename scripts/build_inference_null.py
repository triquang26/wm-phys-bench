#!/usr/bin/env python3
"""Build null distributions that match the INFERENCE pipeline exactly.

The original cycle/jaccard nulls were built from pre-segmented reference
frames at lag-1. But VideoScorer at inference time:
  1. Reads raw .mp4
  2. Samples n_frames=10 via np.linspace (large temporal gap)
  3. SAM3-segments each frame at runtime
  4. Computes cycle on consecutive sampled frames

These mismatches cause cycle scores to saturate (~0.99) on both real and
generated videos because the original null underestimates expected drift.

This script samples N real training mp4s the EXACT same way and builds
matched nulls. Resulting null is what VideoScorer actually sees at
inference, so empirical p-values become meaningful.

Output (overwrites cache):
    paper-physical-gr1/ref_cache/cycle_null.npz
    paper-physical-gr1/ref_cache/nn_jaccard_null.npy

Usage (groot env):
    python scripts/build_inference_null.py --max_videos 50 --n_frames 10
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

REPO_ROOT = Path("/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_root", type=Path,
                    default=REPO_ROOT / "paper-physical-gr1" / "raw_videos" / "gr1")
    ap.add_argument("--cache_dir", type=Path,
                    default=REPO_ROOT / "paper-physical-gr1" / "ref_cache")
    ap.add_argument("--max_videos", type=int, default=50,
                    help="How many real training videos to sample (default 50)")
    ap.add_argument("--n_frames", type=int, default=10,
                    help="Frames per video — must match inference n_frames")
    args = ap.parse_args()

    # Discover real videos
    mp4s = sorted(args.video_root.glob("*.mp4"))
    if len(mp4s) > args.max_videos:
        idx = np.linspace(0, len(mp4s) - 1, args.max_videos).astype(int)
        mp4s = [mp4s[i] for i in idx]
    print(f"Found {len(mp4s)} real training videos")

    # Reuse VideoScorer machinery: SAM3 + DINOv2 + RoMa cycle + Jaccard
    from warp_score.video_scorer import VideoScorer, ReferenceCache
    from warp_score.temporal_signals import cycle_signal
    from warp_score.nn_consistency import nn_set_jaccard_distance

    # Load existing cache for pool_feats (used by jaccard signal)
    cache = ReferenceCache.load(args.cache_dir)
    print(f"Loaded pool: {cache.pool_feats.shape[0]} refs × {cache.pool_feats.shape[1]} dim")

    # Spin up scorer for SAM3 + DINOv2 + RoMa lazy-loading
    scorer = VideoScorer(cache=cache, sam_segment=True)
    _ = scorer._get_sam()
    _ = scorer._get_dino()
    _ = scorer._get_matcher()

    cycle_means = []
    cycle_peaks = []
    jaccard_dists = []

    t_start = time.time()
    for i, mp4 in enumerate(mp4s, 1):
        try:
            # Same frame extraction as VideoScorer
            raw = scorer._extract_spread_frames(mp4, args.n_frames)
            if len(raw) < 2:
                continue

            # SAM3 segment
            sam = scorer._get_sam()
            seg_frames = [(idx, sam.segment_frame(bgr)) for idx, bgr in raw]

            # Save to tmp + DINOv2 embed
            import tempfile, cv2
            tmp = Path(tempfile.mkdtemp(prefix="inull_"))
            paths = []
            for j, (idx, bgr) in enumerate(seg_frames):
                p = tmp / f"f_{j:04d}.png"
                cv2.imwrite(str(p), bgr)
                paths.append(p)

            dino = scorer._get_dino()
            q_feats = dino.extract(paths)

            matcher = scorer._get_matcher()
            for t in range(len(paths) - 1):
                # Cycle
                fwd = matcher.match(paths[t], paths[t + 1])
                bwd = matcher.match(paths[t + 1], paths[t])
                sig = cycle_signal(fwd.warp, bwd.warp, cert_fwd=fwd.cert, cert_floor=0.1)
                cycle_means.append(sig["mean"])
                cycle_peaks.append(sig["peak"])

                # Jaccard
                ja = nn_set_jaccard_distance(
                    q_feats[t], q_feats[t + 1], cache.pool_feats, k=50,
                )
                jaccard_dists.append(ja)

            # Cleanup tmp
            for p in paths:
                p.unlink(missing_ok=True)
            tmp.rmdir()

            elapsed = time.time() - t_start
            rate = i / elapsed
            eta = (len(mp4s) - i) / max(rate, 1e-6)
            print(f"  [{i}/{len(mp4s)}] {mp4.name:15s}  "
                  f"pairs added: {len(paths)-1}  "
                  f"rate={rate:.2f} vid/s   eta={eta/60:.1f} min")
        except Exception as e:
            print(f"  [{i}/{len(mp4s)}] {mp4.name:15s}  FAILED: {e}")
            continue

    cycle_means = np.sort(np.asarray(cycle_means, dtype=np.float32))
    cycle_peaks = np.sort(np.asarray(cycle_peaks, dtype=np.float32))
    jaccard_dists = np.sort(np.asarray(jaccard_dists, dtype=np.float32))

    print(f"\n=== Inference-matched null distributions ===")
    print(f"n_pairs:        {len(cycle_means)}")
    print(f"cycle_mean:     med={np.median(cycle_means):.3f}  "
          f"p95={np.percentile(cycle_means, 95):.3f}  "
          f"p99={np.percentile(cycle_means, 99):.3f}  "
          f"max={cycle_means.max():.3f}")
    print(f"cycle_peak:     med={np.median(cycle_peaks):.3f}  "
          f"p95={np.percentile(cycle_peaks, 95):.3f}  "
          f"p99={np.percentile(cycle_peaks, 99):.3f}  "
          f"max={cycle_peaks.max():.3f}")
    print(f"jaccard:        med={np.median(jaccard_dists):.3f}  "
          f"p95={np.percentile(jaccard_dists, 95):.3f}  "
          f"p99={np.percentile(jaccard_dists, 99):.3f}  "
          f"max={jaccard_dists.max():.3f}")

    # Save: OVERWRITE cache nulls
    np.savez(args.cache_dir / "cycle_null.npz",
             cycle_mean=cycle_means, cycle_peak=cycle_peaks)
    np.save(args.cache_dir / "nn_jaccard_null.npy", jaccard_dists)
    print(f"\nNulls overwritten in {args.cache_dir}")
    print(f"  cycle_null.npz       (n={len(cycle_means)} pairs)")
    print(f"  nn_jaccard_null.npy  (n={len(jaccard_dists)} pairs)")


if __name__ == "__main__":
    main()
