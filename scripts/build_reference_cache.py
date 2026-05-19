#!/usr/bin/env python3
"""Build a ReferenceCache directory from existing benchmark artifacts.

Compiles into a single inference-ready directory:
    pool_feats.npy        DINOv2 embeddings for pool refs
    pool_paths.txt        matching ref paths
    cycle_null.npz        sorted cycle_mean / cycle_peak null arrays
    nn_jaccard_null.npy   sorted S4 jaccard null array

Once built, the cache is fully portable — point VideoScorer.from_cache()
at this dir on any machine with the warp_score package + RoMa weights
and you can score new videos without re-running calibration.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

REPO_ROOT = Path("/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination")
POOL = REPO_ROOT / "paper-physical-gr1" / "pool"
V2_OUT = POOL / "results_warpdyn_v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path,
                    default=REPO_ROOT / "paper-physical-gr1" / "ref_cache")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from warp_score.adaptive_refs import DinoFeatureExtractor
    from warp_score.video_scorer import ReferenceCache

    # Pool refs + DINOv2 embeddings
    print("Embedding pool refs …")
    pool_paths = sorted((POOL / "reference" / "POOL").glob("*.png"))
    dino = DinoFeatureExtractor("dinov2_vits14")
    dino._load()
    pool_feats = dino.extract(pool_paths)
    print(f"  Pool size: {len(pool_paths)}   Feat dim: {pool_feats.shape[1]}")

    # Cycle null (cert-weighted, from v2 run)
    cycle_npz = V2_OUT / "cycle_null_v2.npz"
    if not cycle_npz.exists():
        raise FileNotFoundError(f"Cycle null not found: {cycle_npz}. Run run_warpdyn_v2.py first.")
    cycle = np.load(cycle_npz)
    cycle_mean = np.sort(cycle["cycle_mean"].astype(np.float32))
    cycle_peak = np.sort(cycle["cycle_peak"].astype(np.float32))

    # Jaccard null
    jaccard_path = V2_OUT / "nn_jaccard_null.npy"
    if not jaccard_path.exists():
        raise FileNotFoundError(f"Jaccard null not found: {jaccard_path}")
    jaccard_null = np.sort(np.load(jaccard_path).astype(np.float32))

    cache = ReferenceCache(
        pool_paths=pool_paths,
        pool_feats=pool_feats,
        cycle_null_mean=cycle_mean,
        cycle_null_peak=cycle_peak,
        jaccard_null=jaccard_null,
    )
    cache.save(args.out_dir)

    print(f"\nReferenceCache saved → {args.out_dir}")
    print(f"  pool_feats.npy        ({pool_feats.shape})")
    print(f"  pool_paths.txt        ({len(pool_paths)} lines)")
    print(f"  cycle_null.npz        (cycle_mean n={cycle_mean.size}, cycle_peak n={cycle_peak.size})")
    print(f"  nn_jaccard_null.npy   (n={jaccard_null.size})")


if __name__ == "__main__":
    main()
