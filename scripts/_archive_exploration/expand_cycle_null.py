#!/usr/bin/env python3
"""Compute dense cycle-null distribution across ALL 92 GR1 tasks.

Current WarpDyn null distribution is built from only the 5 evaluation
tasks (n=45 pairs) — too sparse for robust empirical p-values.

This script computes cycle_mean / cycle_peak on sampled consecutive frame
pairs across the 92 task reference dirs already on disk
(~4600 frames total → ~16 pairs/task × 92 = ~1500 null samples).

Output: paper-physical-gr1/cycle_null_large.npz
        Containing sorted arrays of cycle_mean / cycle_peak.

Usage (groot env):
    python scripts/expand_cycle_null.py [--pairs_per_task 16]
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
REFERENCE_ROOT = REPO_ROOT / "paper-physical-gr1" / "reference"
OUT_NPZ = REPO_ROOT / "paper-physical-gr1" / "cycle_null_large.npz"


def sample_pairs_for_task(task_dir: Path, n_pairs: int) -> list[tuple[Path, Path]]:
    """Pick n_pairs evenly-spaced consecutive frame pairs from one task dir."""
    pngs = sorted(task_dir.glob("*.png"))
    if len(pngs) < 2:
        return []
    n_consecutive = len(pngs) - 1
    if n_consecutive <= n_pairs:
        idxs = list(range(n_consecutive))
    else:
        idxs = np.linspace(0, n_consecutive - 1, n_pairs).astype(int).tolist()
    return [(pngs[i], pngs[i + 1]) for i in idxs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_per_task", type=int, default=16,
                    help="Consecutive pairs to sample per task (default 16 → ~1500 total)")
    ap.add_argument("--limit_tasks", type=int, default=None,
                    help="For testing: limit number of tasks to process")
    args = ap.parse_args()

    task_dirs = sorted(d for d in REFERENCE_ROOT.iterdir() if d.is_dir())
    if args.limit_tasks:
        task_dirs = task_dirs[: args.limit_tasks]
    print(f"Tasks available: {len(task_dirs)}")

    # Build pair list
    all_pairs: list[tuple[str, Path, Path]] = []
    for td in task_dirs:
        for a, b in sample_pairs_for_task(td, args.pairs_per_task):
            all_pairs.append((td.name, a, b))
    print(f"Total pairs to compute: {len(all_pairs)}")

    # Load RoMa
    from warp_score.matcher import RoMaMatcher
    from warp_score.temporal_signals import cycle_signal

    matcher = RoMaMatcher(setting="turbo", device="cuda", use_precision=True, vis_size=224)
    print("Loading RoMaV2 …")
    matcher._load_model()
    print("  OK")

    means: list[float] = []
    peaks: list[float] = []
    t_start = time.time()
    for i, (task, p_a, p_b) in enumerate(all_pairs, 1):
        fwd = matcher.match(p_a, p_b)
        bwd = matcher.match(p_b, p_a)
        sig = cycle_signal(fwd.warp, bwd.warp, cert_fwd=fwd.cert)
        means.append(sig["mean"])
        peaks.append(sig["peak"])
        if i % 50 == 0 or i == len(all_pairs):
            rate = i / (time.time() - t_start)
            eta = (len(all_pairs) - i) / max(rate, 1e-6)
            print(f"  [{i}/{len(all_pairs)}] rate={rate:.2f} pair/s   eta={eta/60:.1f} min   "
                  f"latest: {task[:40]} mean={sig['mean']:.3f} peak={sig['peak']:.3f}")

    means_arr = np.sort(np.asarray(means, dtype=np.float32))
    peaks_arr = np.sort(np.asarray(peaks, dtype=np.float32))

    np.savez(OUT_NPZ,
             cycle_mean=means_arr, cycle_peak=peaks_arr,
             n_tasks=len(task_dirs), n_pairs=len(all_pairs))
    print(f"\nSaved → {OUT_NPZ}")
    print(f"  n_pairs={len(all_pairs)}")
    print(f"  cycle_mean: med={np.median(means_arr):.3f}  p95={np.percentile(means_arr, 95):.3f}  "
          f"p99={np.percentile(means_arr, 99):.3f}  max={means_arr.max():.3f}")
    print(f"  cycle_peak: med={np.median(peaks_arr):.3f}  p95={np.percentile(peaks_arr, 95):.3f}  "
          f"p99={np.percentile(peaks_arr, 99):.3f}  max={peaks_arr.max():.3f}")


if __name__ == "__main__":
    main()
