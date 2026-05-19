#!/usr/bin/env python3
"""Per-task DENSE null eval — multi-lag pairs to lock training safety.

Goal: maximise zero-false-positive guarantee on real training videos.
Per-task null built from MULTIPLE lags (1, 2, 5, 10) of the same task's
reference frames so the null distribution covers the full range of
motion magnitudes that real videos at any inference-time lag could land in.

For 50 reference frames per task and lags {1, 2, 5, 10}:
    49 + 48 + 45 + 40 = 182 pairs per task
Compared to lag-1 only (~49 pairs) this is ~3.7× denser and covers wider
motion magnitudes — should absorb most "real outlier" videos.

Eval metric priority:
  1. Real H_peak should be LOW (mean and max across 5 tasks).
  2. Gen H_peak is informative — we still report it, but no hallu label
     is asserted on gen videos.

Per-signal report keeps Cauchy mean+peak fusion intact.
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
OUT = BENCH / "per_task_dense_eval"

EVAL_TASKS = [
    "1_Use the right hand to pick up green bok choy from tan table right side to bottom level of wire basket.",
    "2_Use the right hand to pick up rubik's cube from top level of the shelf to bottom level of the shelf.",
    "3_Use the right hand to pick up banana from teal plate to wooden table.",
    "4_Use the left hand to pick up dragonfruit from pink plate to teal plate.",
    "6_Use the right hand to pick up orange from middle of table to bottom white shelf.",
]

NULL_LAGS = [1, 2, 3, 4, 5, 6, 8, 10, 15]   # 9 lags → ~410 null pairs per task
N_TEST_FRAMES = 10


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
        CycleSignal, PrecisionAnomalySignal, empirical_p_value,
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
    ]
    sig_names = [s.name for s in signals]
    print(f"Signals: {sig_names}\n")

    def score_pair(p_a: Path, p_b: Path) -> dict:
        fwd = matcher.match(p_a, p_b)
        bwd = matcher.match(p_b, p_a)
        return {s.name: s.compute(fwd, bwd) for s in signals}

    def video_pair_scores(mp4: Path) -> list[dict]:
        bgrs = sample_frames(mp4, N_TEST_FRAMES)
        if len(bgrs) < 2:
            return []
        seg_bgrs = [seg.segment_frame(b) for b in bgrs]
        tmp = Path(tempfile.mkdtemp(prefix="ptdense_"))
        paths = []
        for i, b in enumerate(seg_bgrs):
            p = tmp / f"f_{i:04d}.png"
            cv2.imwrite(str(p), b)
            paths.append(p)
        out = []
        for t in range(len(paths) - 1):
            sigs = score_pair(paths[t], paths[t + 1])
            out.append({n: {"mean": sigs[n].mean, "peak": sigs[n].peak} for n in sig_names})
        for p in paths:
            p.unlink(missing_ok=True)
        paths[0].parent.rmdir()
        return out

    all_rows = []
    null_summary = {}

    for task in EVAL_TASKS:
        task_short = task.split("_")[0]
        print(f"\n{'=' * 70}")
        print(f"Task {task_short}: {task[:60]}")
        print('=' * 70)

        ref_dir = REF_ROOT / task
        if not ref_dir.exists():
            print(f"  ref_dir missing")
            continue
        pngs = sorted(ref_dir.glob("*.png"))

        # ── Build dense per-task null at multiple lags
        null = {n: {"mean": [], "peak": []} for n in sig_names}
        for lag in NULL_LAGS:
            n_pairs = max(0, len(pngs) - lag)
            for i in range(n_pairs):
                sigs = score_pair(pngs[i], pngs[i + lag])
                for n in sig_names:
                    null[n]["mean"].append(sigs[n].mean)
                    null[n]["peak"].append(sigs[n].peak)
            print(f"  lag {lag:>2}: +{n_pairs} pairs   (total: {len(null[sig_names[0]]['mean'])})")

        sorted_null = {n: {"mean": np.sort(np.asarray(null[n]["mean"], dtype=np.float32)),
                           "peak": np.sort(np.asarray(null[n]["peak"], dtype=np.float32))}
                       for n in sig_names}
        null_summary[task_short] = {n: {"n": len(null[n]["mean"]),
                                         "mean_p99": float(np.percentile(sorted_null[n]["mean"], 99)),
                                         "peak_p99": float(np.percentile(sorted_null[n]["peak"], 99))}
                                     for n in sig_names}

        # ── Score real and gen videos
        def agg_video(mp4: Path, vtype: str, label: int):
            t0 = time.time()
            pairs = video_pair_scores(mp4)
            row = {"task": task_short, "video": mp4.name, "type": vtype, "label": label,
                   "time_sec": time.time() - t0,
                   "null_n": len(sorted_null["cycle"]["mean"])}
            for n in sig_names:
                h_pairs = []
                for pp in pairs:
                    p_m = empirical_p_value(pp[n]["mean"], sorted_null[n]["mean"])
                    p_p = empirical_p_value(pp[n]["peak"], sorted_null[n]["peak"])
                    h_pairs.append(1.0 - cauchy_combine([p_m, p_p]))
                if h_pairs:
                    row[f"{n}_peak"] = float(np.percentile(h_pairs, 80))
                    row[f"{n}_robust"] = float(np.sort(h_pairs)[len(h_pairs)//10:-max(1, len(h_pairs)//10)].mean()
                                                if len(h_pairs) > 2 else np.mean(h_pairs))
                else:
                    row[f"{n}_peak"] = 0.0
                    row[f"{n}_robust"] = 0.0
            return row

        real_mp4 = RAW_VIDEO_ROOT / f"{task_short}.mp4"
        if real_mp4.exists():
            r = agg_video(real_mp4, "REAL", 0)
            all_rows.append(r)
            print(f"  [REAL] {real_mp4.name:<8}  cycle={r['cycle_peak']:.4f}  pa={r['precision_anomaly_peak']:.4f}")

        for mp4 in sorted((GEN_ROOT / task).glob("v*.mp4")):
            r = agg_video(mp4, "GEN", 1)
            all_rows.append(r)
            print(f"  [GEN ] {mp4.name:<8}  cycle={r['cycle_peak']:.4f}  pa={r['precision_anomaly_peak']:.4f}")

    # ── Save
    csv_path = OUT / "per_task_dense_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nTable → {csv_path}")

    # ── Training-safety report (PRIMARY metric)
    from sklearn.metrics import roc_auc_score
    print("\n" + "=" * 70)
    print("PRIMARY METRIC: REAL TRAINING SAFETY")
    print("=" * 70)
    reals = [r for r in all_rows if r["type"] == "REAL"]
    print(f"{'Task':<8} {'cycle_peak':>11} {'cycle_robust':>14} {'pa_peak':>10} {'pa_robust':>11}")
    for r in reals:
        print(f"{r['task']:<8} {r['cycle_peak']:>11.4f} {r['cycle_robust']:>14.4f} "
              f"{r['precision_anomaly_peak']:>10.4f} {r['precision_anomaly_robust']:>11.4f}")

    print(f"\nReal max(cycle_peak):   {max(r['cycle_peak'] for r in reals):.4f}  ← FPR=0% threshold")
    print(f"Real max(cycle_robust): {max(r['cycle_robust'] for r in reals):.4f}")
    print(f"Real max(pa_peak):      {max(r['precision_anomaly_peak'] for r in reals):.4f}")

    # ── Informative gen scores
    print("\n" + "=" * 70)
    print("INFORMATIVE: GEN scores at real-max thresholds")
    print("=" * 70)
    for n in sig_names:
        for agg in ("peak", "robust"):
            T = max(r[f"{n}_{agg}"] for r in reals)
            gens = [r for r in all_rows if r["type"] == "GEN"]
            gen_above = sum(1 for r in gens if r[f"{n}_{agg}"] > T)
            try:
                y = np.array([r["label"] for r in all_rows])
                s = np.array([r[f"{n}_{agg}"] for r in all_rows])
                auroc = float(roc_auc_score(y, s))
            except Exception:
                auroc = float("nan")
            print(f"  {n:<20} {agg:<7}  threshold={T:.4f}  "
                  f"gen above: {gen_above}/{len(gens)}  AUROC={auroc:.4f}")


if __name__ == "__main__":
    main()
