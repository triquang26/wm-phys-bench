#!/usr/bin/env python3
"""Quick test: cycle_signal + trajectory_accel on 1 real pair vs 1 generated pair.

Expected: real pair shows LOWER cycle drift / accel than generated.
If this holds at small N, full benchmark is worth running.
"""
from __future__ import annotations

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from warp_score.matcher import RoMaMatcher
from warp_score.temporal_signals import cycle_signal, trajectory_accel

POOL = Path("/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination/paper-physical-gr1/pool")
REAL = POOL / "query_high" / "POOL"
GEN  = POOL / "query_low"  / "POOL"


def consecutive_pairs_per_video(dirpath: Path, video_prefix: bool):
    """Group frames by (task, video) and return sorted consecutive pairs."""
    from collections import defaultdict
    groups: dict[tuple[str, str], list[tuple[int, Path]]] = defaultdict(list)
    for p in sorted(dirpath.glob("*.png")):
        stem = p.stem  # e.g. t1__v0000_frame_0010 or t1__frame_0015
        task_part, _, rest = stem.partition("__")
        if video_prefix:
            vid_part, _, frame_part = rest.partition("_frame_")
        else:
            vid_part = "real"
            frame_part = rest.replace("frame_", "")
        frame_idx = int(frame_part)
        groups[(task_part, vid_part)].append((frame_idx, p))

    pairs = []
    for key, items in groups.items():
        items.sort()
        for (idx_a, p_a), (idx_b, p_b) in zip(items[:-1], items[1:]):
            pairs.append((key, idx_a, idx_b, p_a, p_b))
    return pairs


def main():
    matcher = RoMaMatcher(setting="turbo", device="cuda", use_precision=True, vis_size=224)
    print("Loading RoMaV2…")
    matcher._load_model()
    print("OK\n")

    real_pairs = consecutive_pairs_per_video(REAL, video_prefix=False)[:5]
    gen_pairs = consecutive_pairs_per_video(GEN, video_prefix=True)[:5]

    print(f"Testing on {len(real_pairs)} real pairs + {len(gen_pairs)} generated pairs\n")

    def eval_pairs(pairs, label):
        cycles_mean, cycles_peak = [], []
        for key, idx_a, idx_b, p_a, p_b in pairs:
            fwd = matcher.match(p_a, p_b)
            bwd = matcher.match(p_b, p_a)
            sig = cycle_signal(fwd.warp, bwd.warp, cert_fwd=fwd.cert)
            cycles_mean.append(sig["mean"])
            cycles_peak.append(sig["peak"])
            print(f"  [{label}] {key[0]}/{key[1]} {idx_a}->{idx_b}  "
                  f"cycle_mean={sig['mean']:.4f}  peak={sig['peak']:.4f}")
        import numpy as np
        return np.array(cycles_mean), np.array(cycles_peak)

    print("=== REAL pairs ===")
    real_mean, real_peak = eval_pairs(real_pairs, "REAL")
    print("\n=== GENERATED pairs ===")
    gen_mean, gen_peak = eval_pairs(gen_pairs, "GEN ")

    print("\n=== SUMMARY ===")
    print(f"REAL cycle_mean: mean={real_mean.mean():.4f}  std={real_mean.std():.4f}")
    print(f"GEN  cycle_mean: mean={gen_mean.mean():.4f}  std={gen_mean.std():.4f}")
    print(f"REAL cycle_peak: mean={real_peak.mean():.4f}")
    print(f"GEN  cycle_peak: mean={gen_peak.mean():.4f}")
    diff = gen_mean.mean() - real_mean.mean()
    print(f"\nGEN - REAL (mean): {diff:+.4f}  ({'GEN larger ✓' if diff > 0 else 'REAL larger ✗'})")


if __name__ == "__main__":
    main()
