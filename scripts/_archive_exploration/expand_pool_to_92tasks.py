#!/usr/bin/env python3
"""Expand reference pool from 5 tasks (200 frames) to all 92 tasks (~4600).

Rebuilds:
    paper-physical-gr1/ref_cache/pool_feats.npy   (large pool DINOv2)
    paper-physical-gr1/ref_cache/pool_paths.txt   (matching paths)

cycle_null.npz and nn_jaccard_null.npy are LEFT AS-IS — they were
already built from 92-task consecutive pairs, so they're already
representative of the full distribution. Only k-NN candidate pool widens.

Cost: ~30 sec DINOv2 embed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

REPO_ROOT = Path("/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_root", type=Path,
                    default=REPO_ROOT / "paper-physical-gr1" / "reference")
    ap.add_argument("--cache_dir", type=Path,
                    default=REPO_ROOT / "paper-physical-gr1" / "ref_cache")
    ap.add_argument("--frames_per_task", type=int, default=50,
                    help="How many frames per task to include in pool")
    args = ap.parse_args()

    # Discover all task dirs
    task_dirs = sorted(d for d in args.ref_root.iterdir() if d.is_dir())
    print(f"Found {len(task_dirs)} task dirs under {args.ref_root}")

    # Build pool path list
    pool_paths = []
    for td in task_dirs:
        pngs = sorted(td.glob("*.png"))[: args.frames_per_task]
        pool_paths.extend(pngs)
    print(f"Total pool frames: {len(pool_paths)}")

    # Embed with DINOv2
    from warp_score.adaptive_refs import DinoFeatureExtractor
    print("Loading DINOv2 …")
    dino = DinoFeatureExtractor("dinov2_vits14")
    dino._load()
    print(f"Embedding {len(pool_paths)} frames …")
    pool_feats = dino.extract(pool_paths, batch_size=32)
    print(f"  Feature shape: {pool_feats.shape}")

    # Save into existing cache (preserving nulls + threshold)
    cache_dir = args.cache_dir
    np.save(cache_dir / "pool_feats.npy", pool_feats.astype(np.float32))
    (cache_dir / "pool_paths.txt").write_text("\n".join(str(p) for p in pool_paths))

    print(f"\nExpanded pool saved → {cache_dir}")
    print(f"  pool_feats.npy : {pool_feats.shape}")
    print(f"  pool_paths.txt : {len(pool_paths)} lines")
    print(f"  cycle_null.npz : UNCHANGED")
    print(f"  nn_jaccard_null.npy : UNCHANGED")
    print(f"  threshold.json : UNCHANGED (may want to re-calibrate)")


if __name__ == "__main__":
    main()
