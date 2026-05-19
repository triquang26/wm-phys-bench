#!/usr/bin/env python3
"""Per-task null eval — does the detector spare training data when null is task-scoped?

For each of the 5 eval tasks (1, 2, 3, 4, 6):
  Null pairs:   consecutive frame pairs from reference/<task>/*.png  (~49 pairs)
                (these are the SAM3-segmented frames extracted from the
                 task's real training video).
  Real test:    sample the raw training mp4 at lag-10, SAM3-segment,
                compute cycle / PA / lost-pixel signals (9 pair scores → p80).
  Gen test:     same pipeline on all 5 Cosmos generations for that task.

Reports per-task:
  - Real H_peak  (should be LOW if null absorbs the training distribution)
  - Gen H_peak   (informative — should be higher if cycle picks up artifacts)
  - Per-signal: cycle / PA / lost-pixel / bidir
"""
from __future__ import annotations

import csv
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

REPO_ROOT = Path("/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination")
BENCH = REPO_ROOT / "paper-physical-gr1"
REF_ROOT = BENCH / "reference"
RAW_VIDEO_ROOT = BENCH / "raw_videos" / "gr1"
GEN_ROOT = BENCH / "generated"
OUT = BENCH / "per_task_eval"

EVAL_TASKS = [
    "1_Use the right hand to pick up green bok choy from tan table right side to bottom level of wire basket.",
    "2_Use the right hand to pick up rubik's cube from top level of the shelf to bottom level of the shelf.",
    "3_Use the right hand to pick up banana from teal plate to wooden table.",
    "4_Use the left hand to pick up dragonfruit from pink plate to teal plate.",
    "6_Use the right hand to pick up orange from middle of table to bottom white shelf.",
]


def sample_frames(mp4: Path, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(mp4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []
    indices = np.linspace(0, total - 1, min(n, total), dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        if ok:
            frames.append(bgr)
    cap.release()
    return frames


def cauchy_combine(ps):
    ps = [p for p in ps if p is not None and 0 < p < 1]
    if not ps:
        return 0.5
    t = float(np.mean([np.tan(np.pi * (0.5 - p)) for p in ps]))
    return float(0.5 - np.arctan(t) / np.pi)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    from warp_score.temporal_signals import (
        CycleSignal, PrecisionAnomalySignal, LostPixelSignal,
        BidirectionalCertSignal, empirical_p_value,
    )
    from warp_score.matcher import RoMaMatcher
    from warp_score.sam_segmenter import VideoFrameSegmenter

    print("Loading models …")
    matcher = RoMaMatcher(setting="turbo", device="cuda", use_precision=True, vis_size=224)
    matcher._load_model()
    seg = VideoFrameSegmenter()
    signals = [
        CycleSignal(cert_floor=0.1),
        PrecisionAnomalySignal(cert_floor=0.1),
        LostPixelSignal(cert_thresh=0.3, prec_thresh=0.1),
        BidirectionalCertSignal(cert_floor=0.0),
    ]
    sig_names = [s.name for s in signals]
    print(f"Signals: {sig_names}\n")

    def score_pair(p_a: Path, p_b: Path) -> dict:
        fwd = matcher.match(p_a, p_b)
        bwd = matcher.match(p_b, p_a)
        return {s.name: s.compute(fwd, bwd) for s in signals}

    def score_video_pairs(mp4_or_dir: Path, n_frames: int, is_dir: bool = False) -> list[dict]:
        """Returns list of pair-stat dicts {sig: {mean, peak}}."""
        if is_dir:
            # Pre-segmented PNG dir → just use them sorted at lag-1
            pngs = sorted(mp4_or_dir.glob("*.png"))[:n_frames]
            paths = pngs
        else:
            bgrs = sample_frames(mp4_or_dir, n_frames)
            if len(bgrs) < 2:
                return []
            seg_bgrs = [seg.segment_frame(b) for b in bgrs]
            tmp = Path(tempfile.mkdtemp(prefix="ptask_"))
            paths = []
            for i, b in enumerate(seg_bgrs):
                p = tmp / f"f_{i:04d}.png"
                cv2.imwrite(str(p), b)
                paths.append(p)
        out = []
        for t in range(len(paths) - 1):
            sigs = score_pair(paths[t], paths[t + 1])
            out.append({n: {"mean": sigs[n].mean, "peak": sigs[n].peak} for n in sig_names})
        if not is_dir:
            for p in paths:
                p.unlink(missing_ok=True)
            paths[0].parent.rmdir()
        return out

    # ── Iterate over the 5 eval tasks
    all_results = []
    for ti, task in enumerate(EVAL_TASKS, 1):
        task_short = task.split("_")[0]
        print(f"\n{'=' * 70}")
        print(f"Task {task_short}: {task[:60]}")
        print('=' * 70)

        # Build per-task null from pre-segmented reference PNGs (lag-1 pairs)
        ref_dir = REF_ROOT / task
        if not ref_dir.exists():
            print(f"  ref_dir missing: {ref_dir}")
            continue
        pngs = sorted(ref_dir.glob("*.png"))
        print(f"  null pairs (lag-1 ref): {len(pngs)-1}")

        null = {n: {"mean": [], "peak": []} for n in sig_names}
        for t in range(len(pngs) - 1):
            sigs = score_pair(pngs[t], pngs[t + 1])
            for n in sig_names:
                null[n]["mean"].append(sigs[n].mean)
                null[n]["peak"].append(sigs[n].peak)
        sorted_null = {n: {"mean": np.sort(np.asarray(null[n]["mean"], dtype=np.float32)),
                           "peak": np.sort(np.asarray(null[n]["peak"], dtype=np.float32))}
                       for n in sig_names}
        print(f"  null cycle_peak  p99: {np.percentile(sorted_null['cycle']['peak'], 99):.3f}")
        print(f"  null lost_pixel  p99: {np.percentile(sorted_null['lost_pixel']['peak'], 99):.3f}")

        def agg_video(mp4: Path, vtype: str, label: int):
            t0 = time.time()
            pairs = score_video_pairs(mp4, n_frames=10, is_dir=False)
            row = {"task": task_short, "video": mp4.name, "type": vtype, "label": label,
                   "time_sec": time.time() - t0}
            for n in sig_names:
                h_pairs = []
                for pp in pairs:
                    p_m = empirical_p_value(pp[n]["mean"], sorted_null[n]["mean"])
                    p_p = empirical_p_value(pp[n]["peak"], sorted_null[n]["peak"])
                    h_pairs.append(1.0 - cauchy_combine([p_m, p_p]))
                if h_pairs:
                    row[f"{n}_peak"] = float(np.percentile(h_pairs, 80))
                    row[f"{n}_mean"] = float(np.mean(h_pairs))
                else:
                    row[f"{n}_peak"] = 0.0
                    row[f"{n}_mean"] = 0.0
            return row

        # Real training video for this task
        real_mp4 = RAW_VIDEO_ROOT / f"{task_short}.mp4"
        if real_mp4.exists():
            r_real = agg_video(real_mp4, "REAL", 0)
            all_results.append(r_real)
            print(f"  [REAL] {real_mp4.name:<8} "
                  + "  ".join(f"{n}={r_real[n+'_peak']:.3f}" for n in sig_names))
        else:
            print(f"  REAL mp4 missing: {real_mp4}")

        # Gen videos for this task
        gen_dir = GEN_ROOT / task
        for mp4 in sorted(gen_dir.glob("v*.mp4")):
            r_gen = agg_video(mp4, "GEN", 1)
            all_results.append(r_gen)
            print(f"  [GEN ] {mp4.name:<8} "
                  + "  ".join(f"{n}={r_gen[n+'_peak']:.3f}" for n in sig_names))

    # ── Save + summarize
    csv_path = OUT / "per_task_table.csv"
    with open(csv_path, "w", newline="") as f:
        if all_results:
            w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            w.writeheader()
            w.writerows(all_results)
    print(f"\nTable → {csv_path}")

    from sklearn.metrics import roc_auc_score
    y = np.array([r["label"] for r in all_results])
    print("\n" + "=" * 70)
    print("PER-TASK NULL SUMMARY")
    print("=" * 70)
    print(f"{'Signal':<22} {'AUROC':>7} {'mean_R':>8} {'mean_G':>8} {'gen@FPR=0':>11}")
    for n in sig_names:
        scores = np.array([r[f"{n}_peak"] for r in all_results])
        real_s = scores[y == 0]
        gen_s = scores[y == 1]
        T = float(real_s.max()) + 1e-6
        gen_above = int((gen_s > T).sum())
        try:
            auroc = float(roc_auc_score(y, scores))
        except Exception:
            auroc = float("nan")
        print(f"{n:<22} {auroc:>7.4f} {real_s.mean():>8.4f} {gen_s.mean():>8.4f} "
              f"{gen_above:>3d}/{len(gen_s):d}")

    print("\nReal-by-real H_peak (cycle) — should be low if per-task null works:")
    for r in all_results:
        if r["type"] != "REAL":
            continue
        print(f"  task {r['task']:<4} → cycle_peak = {r['cycle_peak']:.4f}  "
              f"pa_peak = {r['precision_anomaly_peak']:.4f}")


if __name__ == "__main__":
    main()
