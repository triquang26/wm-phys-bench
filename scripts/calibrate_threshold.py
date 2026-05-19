#!/usr/bin/env python3
"""Pick optimal video-level threshold via Youden's J on v2 benchmark data.

Loads per-frame H_scores from results_warpdyn_v2/raw_signals_v2.csv,
aggregates to video level (80th-percentile peak — the best metric),
then finds the threshold T that maximizes (TPR - FPR).

Writes paper-physical-gr1/ref_cache/threshold.json so VideoScorer can
load a calibrated default.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination")
POOL = REPO_ROOT / "paper-physical-gr1" / "pool"


def parse_pool_frame(stem):
    task_part, _, rest = stem.partition("__")
    if rest.startswith("v") and "_frame_" in rest:
        vid_part, _, frame_part = rest.partition("_frame_")
    else:
        vid_part, frame_part = "real", rest.replace("frame_", "")
    return task_part, vid_part, int(frame_part)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_csv", type=Path,
                    default=POOL / "results_warpdyn_v2" / "raw_signals_v2.csv")
    ap.add_argument("--cache_dir", type=Path,
                    default=REPO_ROOT / "paper-physical-gr1" / "ref_cache")
    ap.add_argument("--config", default="S1+S2+S4",
                    help="Which fused column to threshold on")
    ap.add_argument("--aggregator", default="p80",
                    choices=["p80", "trimmed_mean", "mean", "max"])
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.raw_csv)))
    by_video = defaultdict(list)
    for r in rows:
        task, vid, _ = parse_pool_frame(r["frame"])
        by_video[(task, vid)].append(float(r[f"H_{args.config}"]))

    video_scores = []
    video_labels = []
    for (task, vid), h_vals in by_video.items():
        h = np.array(h_vals)
        if args.aggregator == "p80":
            score = float(np.percentile(h, 80))
        elif args.aggregator == "trimmed_mean":
            s = np.sort(h)
            n = len(s)
            k = int(n * 0.1)
            score = float(s[k:n-k].mean()) if n - 2 * k > 0 else float(s.mean())
        elif args.aggregator == "mean":
            score = float(h.mean())
        else:
            score = float(h.max())
        video_scores.append(score)
        video_labels.append(0 if vid == "real" else 1)

    y = np.array(video_labels)
    s = np.array(video_scores)

    # Sweep thresholds, find Youden's J = TPR - FPR
    thresholds = np.linspace(s.min(), s.max(), 200)
    best_j = -1.0
    best_t = float(s.mean())
    for t in thresholds:
        pred = s > t
        tp = ((pred == 1) & (y == 1)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        tn = ((pred == 0) & (y == 0)).sum()
        if (tp + fn) == 0 or (fp + tn) == 0:
            continue
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_t = float(t)
            best_tpr, best_fpr = float(tpr), float(fpr)

    # Eval at best threshold
    pred = s > best_t
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    print(f"Config:       {args.config}")
    print(f"Aggregator:   {args.aggregator}")
    print(f"Best threshold:  {best_t:.4f}  (Youden's J = {best_j:.4f})")
    print(f"  TPR={best_tpr:.4f}   FPR={best_fpr:.4f}")
    print(f"  Confusion: TP={tp}  FN={fn}  FP={fp}  TN={tn}")
    print(f"  Accuracy:  {(tp + tn) / len(y):.4f}")
    print(f"  Real(neg) score range: [{s[y==0].min():.4f}, {s[y==0].max():.4f}] mean={s[y==0].mean():.4f}")
    print(f"  Gen (pos) score range: [{s[y==1].min():.4f}, {s[y==1].max():.4f}] mean={s[y==1].mean():.4f}")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    threshold_path = args.cache_dir / "threshold.json"
    threshold_path.write_text(json.dumps({
        "threshold":      best_t,
        "config":         args.config,
        "aggregator":     args.aggregator,
        "youden_j":       best_j,
        "tpr_at_best":    best_tpr,
        "fpr_at_best":    best_fpr,
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "n_videos":       int(len(y)),
        "n_real":         int((y == 0).sum()),
        "n_gen":          int((y == 1).sum()),
    }, indent=2))
    print(f"\nThreshold saved → {threshold_path}")


if __name__ == "__main__":
    main()
