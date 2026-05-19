#!/usr/bin/env python3
"""Build diversity coreset for pool via Farthest-Point Sampling in DINOv2 space.

Current pool: 50 uniform frames/task — adjacent frames are visually
near-identical → redundancy. FPS picks N diverse frames per task that
span the DINOv2 feature manifold of that task.

For each task:
  feats = DINOv2(all_frames)
  centroid = feats[argmax(||feats - mean||)]   # farthest from center
  picked = [centroid]
  while len(picked) < target_per_task:
      next = argmax(min_dist_to_already_picked)
      picked.append(next)

Output: paper-physical-gr1/ref_cache/pool_feats.npy + pool_paths.txt
        with diversity-pruned subset (default 30 per task → 2760 total).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

REPO_ROOT = Path("/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination")


def fps_select(feats: np.ndarray, target_n: int) -> list[int]:
    """Farthest-point sampling. feats: (N, D) L2-normalized. Returns indices."""
    n = len(feats)
    if target_n >= n:
        return list(range(n))
    # Start from the frame farthest from the mean (semantic centroid)
    mean = feats.mean(axis=0)
    mean /= max(np.linalg.norm(mean), 1e-8)
    dists_to_mean = 1.0 - feats @ mean
    selected = [int(np.argmax(dists_to_mean))]
    # Min distance from each point to the selected set
    min_dist = 1.0 - feats @ feats[selected[0]]
    for _ in range(target_n - 1):
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        d_new = 1.0 - feats @ feats[nxt]
        min_dist = np.minimum(min_dist, d_new)
    return sorted(selected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_root", type=Path,
                    default=REPO_ROOT / "paper-physical-gr1" / "reference")
    ap.add_argument("--cache_dir", type=Path,
                    default=REPO_ROOT / "paper-physical-gr1" / "ref_cache")
    ap.add_argument("--per_task", type=int, default=30,
                    help="Diverse frames to keep per task")
    args = ap.parse_args()

    from warp_score.adaptive_refs import DinoFeatureExtractor
    print("Loading DINOv2 …")
    dino = DinoFeatureExtractor("dinov2_vits14")
    dino._load()

    task_dirs = sorted(d for d in args.ref_root.iterdir() if d.is_dir())
    print(f"Tasks: {len(task_dirs)}")

    selected_paths: list[Path] = []
    selected_feats: list[np.ndarray] = []
    for td in task_dirs:
        pngs = sorted(td.glob("*.png"))
        if not pngs:
            continue
        feats = dino.extract(pngs)
        idxs = fps_select(feats, args.per_task)
        selected_paths.extend(pngs[i] for i in idxs)
        selected_feats.append(feats[idxs])
        print(f"  {td.name[:50]:50s}  {len(pngs):3d} → {len(idxs):3d}")

    selected_feats = np.concatenate(selected_feats, axis=0)
    print(f"\nCoreset size: {len(selected_paths)}  (feats: {selected_feats.shape})")

    np.save(args.cache_dir / "pool_feats.npy", selected_feats.astype(np.float32))
    (args.cache_dir / "pool_paths.txt").write_text("\n".join(str(p) for p in selected_paths))
    print(f"Saved → {args.cache_dir}")


if __name__ == "__main__":
    main()
