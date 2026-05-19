#!/usr/bin/env python3
"""Standalone PA-signal eval — clean comparison of CycleSignal vs PrecisionAnomalySignal.

Pipeline:
  1. Build null distributions on 50 real training videos, computing BOTH
     CycleSignal and PrecisionAnomalySignal on every consecutive pair
     through the inference pipeline (SAM3 + DINOv2 + RoMa with precision).
  2. Score all 54 batch-eval videos with BOTH signals.
  3. For each signal, set conformal threshold (alpha=0.05 then 0.00 of real).
  4. Report side-by-side: real outliers, gen catch rate, AUROC.

No modification to VideoScorer — uses signal classes directly. Outputs
written under paper-physical-gr1/pa_eval/.
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

REPO_ROOT = Path("/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination")
BENCH = REPO_ROOT / "paper-physical-gr1"
RAW_VIDEO_ROOT = BENCH / "raw_videos" / "gr1"
GEN_ROOT = BENCH / "generated"
OUT = BENCH / "pa_eval"
OUT.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class VideoStats:
    """Per-video aggregated stats for one signal."""
    name:       str
    h_peak:     float
    h_robust:   float
    h_max:      float


@dataclass
class PerVideoResult:
    video:      str
    type:       str          # "REAL" | "GEN"
    task:       str
    label:      int          # 0=real, 1=gen
    cycle:      VideoStats
    pa:         VideoStats
    time_sec:   float


# ─────────────────────────────────────────────────────────────────────────────
# Video → per-frame signals computation
# ─────────────────────────────────────────────────────────────────────────────


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


def score_video(video_path: Path,
                n_frames: int,
                cycle_sig,
                pa_sig,
                segmenter,
                matcher,
                ) -> tuple[list[dict], list[dict]]:
    """Score one video → list of cycle pair-stats + PA pair-stats."""
    frames_bgr = sample_frames(video_path, n_frames)
    if len(frames_bgr) < 2:
        return [], []
    # SAM3 segment
    seg = [segmenter.segment_frame(b) for b in frames_bgr]

    # Save to tmp
    tmp = Path(tempfile.mkdtemp(prefix="paeval_"))
    paths = []
    for i, b in enumerate(seg):
        p = tmp / f"f_{i:04d}.png"
        cv2.imwrite(str(p), b)
        paths.append(p)

    cycle_stats = []
    pa_stats = []
    for t in range(len(paths) - 1):
        fwd = matcher.match(paths[t], paths[t + 1])
        bwd = matcher.match(paths[t + 1], paths[t])
        c_res = cycle_sig.compute(fwd, bwd)
        pa_res = pa_sig.compute(fwd, bwd)
        cycle_stats.append({"mean": c_res.mean, "peak": c_res.peak})
        pa_stats.append({"mean": pa_res.mean, "peak": pa_res.peak})

    # Cleanup
    for p in paths:
        p.unlink(missing_ok=True)
    tmp.rmdir()

    return cycle_stats, pa_stats


# ─────────────────────────────────────────────────────────────────────────────


def aggregate_pair_stats(stats: list[dict]) -> dict:
    """Compute per-video aggregators from a list of pair stats."""
    if not stats:
        return {"h_peak": 0.0, "h_robust": 0.0, "h_max": 0.0,
                "means": np.array([]), "peaks": np.array([])}
    means = np.array([s["mean"] for s in stats])
    peaks = np.array([s["peak"] for s in stats])
    # Combine mean+peak via simple max (per-frame Cauchy fusion happens elsewhere
    # — here we keep raw stats so the eval pipeline can fuse properly).
    return {"means": means, "peaks": peaks}


# ─────────────────────────────────────────────────────────────────────────────


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_real_calib", type=int, default=50,
                    help="Real videos to build null from")
    ap.add_argument("--max_real_eval", type=int, default=30,
                    help="Real videos to evaluate (separate from calib)")
    ap.add_argument("--n_frames", type=int, default=10)
    args = ap.parse_args()

    from warp_score.temporal_signals import (
        CycleSignal, PrecisionAnomalySignal, empirical_p_value,
    )
    from warp_score.matcher import RoMaMatcher
    from warp_score.sam_segmenter import VideoFrameSegmenter

    print("Loading models …")
    matcher = RoMaMatcher(setting="turbo", device="cuda", use_precision=True, vis_size=224)
    matcher._load_model()
    seg = VideoFrameSegmenter()
    cycle_sig = CycleSignal(cert_floor=0.1)
    pa_sig = PrecisionAnomalySignal(cert_floor=0.1)
    print("OK\n")

    # ── Discover videos
    all_real = sorted(RAW_VIDEO_ROOT.glob("*.mp4"))
    all_gen = sorted(GEN_ROOT.glob("*/v*.mp4"))

    # Stratified split: even spread across 92 task ids
    idx = np.linspace(0, len(all_real) - 1, args.max_real_eval + args.max_real_calib).astype(int)
    sampled = [all_real[i] for i in idx]
    real_calib = sampled[:args.max_real_calib]
    real_eval = sampled[args.max_real_calib:]
    print(f"Calibration real videos: {len(real_calib)}")
    print(f"Evaluation real videos:  {len(real_eval)}")
    print(f"Evaluation gen videos:   {len(all_gen)}")

    # ── Step 1: Build null distributions
    null_cycle_means = []
    null_cycle_peaks = []
    null_pa_means    = []
    null_pa_peaks    = []
    t0 = time.time()
    print("\n=== Building null on calibration real videos ===")
    for i, mp4 in enumerate(real_calib, 1):
        cycle_stats, pa_stats = score_video(mp4, args.n_frames, cycle_sig, pa_sig, seg, matcher)
        for s in cycle_stats:
            null_cycle_means.append(s["mean"])
            null_cycle_peaks.append(s["peak"])
        for s in pa_stats:
            null_pa_means.append(s["mean"])
            null_pa_peaks.append(s["peak"])
        rate = i / (time.time() - t0)
        eta = (len(real_calib) - i) / max(rate, 1e-6)
        print(f"  [{i}/{len(real_calib)}] {mp4.name:15s}  pairs+={len(cycle_stats)}  "
              f"rate={rate:.2f} vid/s   eta={eta/60:.1f} min")

    null_cycle_means = np.sort(np.asarray(null_cycle_means, dtype=np.float32))
    null_cycle_peaks = np.sort(np.asarray(null_cycle_peaks, dtype=np.float32))
    null_pa_means    = np.sort(np.asarray(null_pa_means,    dtype=np.float32))
    null_pa_peaks    = np.sort(np.asarray(null_pa_peaks,    dtype=np.float32))

    print(f"\nNull sizes: {len(null_cycle_means)} pairs")
    print(f"  cycle_mean  med={np.median(null_cycle_means):.4f}  p99={np.percentile(null_cycle_means, 99):.4f}")
    print(f"  cycle_peak  med={np.median(null_cycle_peaks):.4f}  p99={np.percentile(null_cycle_peaks, 99):.4f}")
    print(f"  pa_mean     med={np.median(null_pa_means):.4f}  p99={np.percentile(null_pa_means, 99):.4f}")
    print(f"  pa_peak     med={np.median(null_pa_peaks):.4f}  p99={np.percentile(null_pa_peaks, 99):.4f}")

    np.savez(OUT / "null.npz",
             cycle_mean=null_cycle_means, cycle_peak=null_cycle_peaks,
             pa_mean=null_pa_means, pa_peak=null_pa_peaks)

    # ── Step 2: Score eval videos
    print("\n=== Scoring eval videos ===")
    eval_rows = []

    def cauchy_combine(ps):
        ps = [p for p in ps if p is not None and 0 < p < 1]
        if not ps:
            return 0.5
        t = float(np.mean([np.tan(np.pi * (0.5 - p)) for p in ps]))
        return float(0.5 - np.arctan(t) / np.pi)

    def aggregate_h_scores(pair_stats: list[dict], null_mean, null_peak) -> dict:
        """For each pair: empirical p of mean and peak → Cauchy fuse → H_pair.
        Then aggregate per-video: h_peak, h_robust, h_max."""
        h_per_pair = []
        for s in pair_stats:
            p_m = empirical_p_value(s["mean"], null_mean)
            p_p = empirical_p_value(s["peak"], null_peak)
            p_c = cauchy_combine([p_m, p_p])
            h_per_pair.append(1.0 - p_c)
        h = np.array(h_per_pair, dtype=np.float32)
        if h.size == 0:
            return {"h_peak": 0.0, "h_robust": 0.0, "h_max": 0.0}
        k = max(int(len(h) * 0.1), 0)
        sorted_h = np.sort(h)
        h_robust = float(sorted_h[k:len(h)-k].mean()) if (len(h) - 2*k) > 0 else float(h.mean())
        return {
            "h_peak":   float(np.percentile(h, 80)),
            "h_robust": h_robust,
            "h_max":    float(h.max()),
        }

    def score_for_eval(mp4: Path, vtype: str, label: int, task: str):
        t1 = time.time()
        cycle_stats, pa_stats = score_video(mp4, args.n_frames, cycle_sig, pa_sig, seg, matcher)
        cyc = aggregate_h_scores(cycle_stats, null_cycle_means, null_cycle_peaks)
        pa  = aggregate_h_scores(pa_stats,    null_pa_means,    null_pa_peaks)
        return PerVideoResult(
            video=mp4.name, type=vtype, task=task, label=label,
            cycle=VideoStats(name="cycle", **cyc),
            pa   =VideoStats(name="pa",    **pa),
            time_sec=time.time() - t1,
        )

    for i, mp4 in enumerate(real_eval, 1):
        r = score_for_eval(mp4, "REAL", 0, mp4.stem)
        eval_rows.append(r)
        print(f"  [REAL {i:2d}/{len(real_eval)}] {mp4.name:15s}  "
              f"cycle_peak={r.cycle.h_peak:.4f}  pa_peak={r.pa.h_peak:.4f}  "
              f"({r.time_sec:.1f}s)")
    for i, mp4 in enumerate(all_gen, 1):
        r = score_for_eval(mp4, "GEN", 1, mp4.parent.name)
        eval_rows.append(r)
        print(f"  [GEN  {i:2d}/{len(all_gen)}] {mp4.name:15s}  "
              f"cycle_peak={r.cycle.h_peak:.4f}  pa_peak={r.pa.h_peak:.4f}  "
              f"({r.time_sec:.1f}s)")

    # ── Step 3: Save CSV
    csv_path = OUT / "eval_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "type", "task", "label",
                    "cycle_peak", "cycle_robust", "cycle_max",
                    "pa_peak", "pa_robust", "pa_max", "time_sec"])
        for r in eval_rows:
            w.writerow([r.video, r.type, r.task, r.label,
                        r.cycle.h_peak, r.cycle.h_robust, r.cycle.h_max,
                        r.pa.h_peak, r.pa.h_robust, r.pa.h_max,
                        r.time_sec])

    # ── Step 4: Compare signals
    from sklearn.metrics import roc_auc_score, average_precision_score
    print("\n" + "=" * 70)
    print("SIDE-BY-SIDE: CycleSignal vs PrecisionAnomalySignal")
    print("=" * 70)

    y = np.array([r.label for r in eval_rows])
    for sig_name, getter in [("cycle", lambda r: r.cycle), ("pa", lambda r: r.pa)]:
        scores = np.array([getter(r).h_peak for r in eval_rows])
        real_s = scores[y == 0]
        gen_s = scores[y == 1]
        T_safe = float(real_s.max()) + 1e-6
        T_p95 = float(np.quantile(real_s, 0.95))
        gen_above_safe = int((gen_s > T_safe).sum())
        gen_above_p95 = int((gen_s > T_p95).sum())
        real_above_p95 = int((real_s > T_p95).sum())
        auroc = float(roc_auc_score(y, scores))
        ap = float(average_precision_score(y, scores))
        print(f"\n{sig_name.upper()} signal:")
        print(f"  AUROC = {auroc:.4f}   AP = {ap:.4f}")
        print(f"  Real: range [{real_s.min():.4f}, {real_s.max():.4f}]  mean={real_s.mean():.4f}")
        print(f"  Gen:  range [{gen_s.min():.4f}, {gen_s.max():.4f}]  mean={gen_s.mean():.4f}")
        print(f"  Threshold @ FPR=0%:  {T_safe:.4f}  →  gen caught: {gen_above_safe}/{len(gen_s)}")
        print(f"  Threshold @ FPR≤5%:  {T_p95:.4f}  →  real flagged: {real_above_p95}/{len(real_s)}, "
              f"gen caught: {gen_above_p95}/{len(gen_s)}")

    # ── Step 5: Top-10 real outliers per signal
    print("\n" + "=" * 70)
    print("Top 10 REAL outliers — does PA reduce false positives?")
    print("=" * 70)
    reals = [r for r in eval_rows if r.type == "REAL"]
    print(f"\n{'Rank':>4} {'Task':<10} {'Video':<15} {'cycle_peak':>12} {'pa_peak':>12} {'ratio':>8}")
    cycle_sorted = sorted(reals, key=lambda r: -r.cycle.h_peak)[:10]
    for i, r in enumerate(cycle_sorted, 1):
        ratio = r.pa.h_peak / max(r.cycle.h_peak, 1e-6)
        print(f"{i:>4} {r.task:<10} {r.video:<15} {r.cycle.h_peak:>12.4f} {r.pa.h_peak:>12.4f} {ratio:>8.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
