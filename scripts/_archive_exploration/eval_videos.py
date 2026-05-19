#!/usr/bin/env python3
"""End-to-end batch eval: input training videos + all generated videos.

Discovers:
  - Real videos:      paper-physical-gr1/raw_videos/gr1/*.mp4
  - Generated videos: paper-physical-gr1/generated/<task>/v*.mp4

Each video is scored through the full WarpDyn pipeline (SAM3 → DINOv2
→ RoMa cycle + S4 → Cauchy fuse → 80%-peak aggregation). Output:

  paper-physical-gr1/eval_results/
    video_table.csv     one row per video with H_peak, decision, time
    summary.json        AUROC, confusion matrix, mean scores, total time
    summary_table.md    pretty markdown table (sorted by H_peak desc)

Usage (groot env):
    python scripts/eval_videos.py [--n_frames 10] [--max_real 30]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

REPO_ROOT = Path("/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination")
BENCH = REPO_ROOT / "paper-physical-gr1"


def discover_real_videos(max_real: Optional[int] = None) -> list[Path]:
    """Real training mp4s from raw_videos/gr1/."""
    pool = sorted((BENCH / "raw_videos" / "gr1").glob("*.mp4"))
    if max_real is not None:
        # Sample evenly across the 92 tasks for diversity
        if len(pool) > max_real:
            idx = np.linspace(0, len(pool) - 1, max_real).astype(int)
            pool = [pool[i] for i in idx]
    return pool


def discover_generated_videos() -> list[Path]:
    """All Cosmos-generated mp4s under generated/<task>/v*.mp4."""
    gen_root = BENCH / "generated"
    return sorted(gen_root.glob("*/v*.mp4"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=Path, default=BENCH / "ref_cache")
    ap.add_argument("--out_dir",   type=Path, default=BENCH / "eval_results")
    ap.add_argument("--n_frames",  type=int, default=10)
    ap.add_argument("--max_real",  type=int, default=30,
                    help="How many real training videos to sample (default 30)")
    ap.add_argument("--no_sam",    action="store_true",
                    help="Disable SAM3 (videos already segmented)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from warp_score.video_scorer import VideoScorer

    # ── Load scorer once
    print(f"Loading reference cache from {args.cache_dir} …")
    scorer = VideoScorer.from_cache(args.cache_dir, sam_segment=not args.no_sam)
    print(f"  pool size:    {len(scorer.cache.pool_paths)}")
    print(f"  threshold:    {scorer.threshold:.4f}")
    print(f"  cycle null:   n={len(scorer.cache.cycle_null_mean)}")
    print(f"  jaccard null: n={len(scorer.cache.jaccard_null) if scorer.cache.jaccard_null is not None else 0}")
    print()

    # ── Discover videos
    reals = discover_real_videos(args.max_real)
    gens  = discover_generated_videos()
    print(f"Real training videos:    {len(reals)}")
    print(f"Generated videos (gen):  {len(gens)}")
    print(f"Total to score:          {len(reals) + len(gens)}\n")

    rows = []
    total_start = time.time()

    def score_one(path: Path, label: int, video_type: str, task: str):
        t0 = time.time()
        try:
            r = scorer.score(path, n_frames=args.n_frames)
            elapsed = time.time() - t0
            return {
                "video":          path.name,
                "type":           video_type,
                "task":           task,
                "label":          label,
                "h_peak":         r.video_h_peak,
                "h_robust":       r.video_h_score,
                "h_max":          r.aggregate_breakdown["max"],
                "is_hallu":       bool(r.video_h_peak > scorer.threshold),
                "decision":       "HALLU" if r.video_h_peak > scorer.threshold else "CLEAN",
                "correct":        ((r.video_h_peak > scorer.threshold) == bool(label)),
                "n_frames_used":  r.n_frames,
                "time_sec":       elapsed,
            }
        except Exception as e:
            return {
                "video": path.name, "type": video_type, "task": task,
                "label": label, "h_peak": None, "h_robust": None, "h_max": None,
                "is_hallu": None, "decision": "ERROR", "correct": False,
                "n_frames_used": 0, "time_sec": time.time() - t0,
                "error": str(e),
            }

    def fmt_h(v):
        return f"{v:.4f}" if v is not None else "  ERR "

    # ── Score real videos
    print("=" * 80)
    print("REAL TRAINING VIDEOS")
    print("=" * 80)
    for i, p in enumerate(reals, 1):
        task = p.stem  # task number from filename
        row = score_one(p, label=0, video_type="REAL", task=task)
        rows.append(row)
        print(f"  [{i:2d}/{len(reals)}] {p.name:20s}  "
              f"H_peak={fmt_h(row['h_peak'])}  "
              f"→ {row['decision']:5s}  ({row['time_sec']:.1f}s)")

    # ── Score generated videos
    print("\n" + "=" * 80)
    print("GENERATED VIDEOS (Cosmos)")
    print("=" * 80)
    for i, p in enumerate(gens, 1):
        task = p.parent.name
        row = score_one(p, label=1, video_type="GEN", task=task)
        rows.append(row)
        print(f"  [{i:2d}/{len(gens)}] {task[:40]:40s}/{p.name:10s}  "
              f"H_peak={fmt_h(row['h_peak'])}  "
              f"→ {row['decision']:5s}  ({row['time_sec']:.1f}s)")

    total_time = time.time() - total_start

    # ── CSV
    csv_path = args.out_dir / "video_table.csv"
    fieldnames = list({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nVideo table → {csv_path}")

    # ── Aggregate metrics
    valid = [r for r in rows if r["h_peak"] is not None]
    y = np.array([r["label"] for r in valid])
    s = np.array([r["h_peak"] for r in valid])

    from sklearn.metrics import roc_auc_score, average_precision_score

    if len(set(y.tolist())) >= 2:
        auroc = float(roc_auc_score(y, s))
        ap    = float(average_precision_score(y, s))
    else:
        auroc = float("nan")
        ap    = float("nan")

    pred = (s > scorer.threshold)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)

    summary = {
        "n_real":            int((y == 0).sum()),
        "n_gen":             int((y == 1).sum()),
        "threshold":         float(scorer.threshold),
        "auroc":             auroc,
        "ap":                ap,
        "tpr":               float(tpr),
        "fpr":               float(fpr),
        "accuracy":          float((tp + tn) / max(len(y), 1)),
        "confusion":         {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "mean_score_real":   float(s[y == 0].mean()) if (y == 0).any() else None,
        "mean_score_gen":    float(s[y == 1].mean()) if (y == 1).any() else None,
        "total_time_sec":    float(total_time),
        "avg_time_per_video": float(total_time / max(len(rows), 1)),
        "n_frames_per_video": int(args.n_frames),
    }

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # ── Pretty markdown table
    rows_sorted = sorted(valid, key=lambda r: -r["h_peak"])
    md = ["# WarpDyn batch eval results\n"]
    md.append(f"- Threshold:  `{scorer.threshold:.4f}`")
    md.append(f"- AUROC:      `{auroc:.4f}` | AP: `{ap:.4f}`")
    md.append(f"- TPR / FPR:  `{tpr:.4f}` / `{fpr:.4f}` (accuracy `{summary['accuracy']:.4f}`)")
    md.append(f"- mean(real) `{summary['mean_score_real']:.4f}`  |  mean(gen) `{summary['mean_score_gen']:.4f}`")
    md.append(f"- total time: `{total_time:.1f}s` over `{len(rows)}` videos "
              f"(`{summary['avg_time_per_video']:.1f}s` / video)\n")
    md.append("## Videos sorted by H_peak (descending)\n")
    md.append("| # | type | task | video | H_peak | H_robust | decision | correct? | time |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows_sorted, 1):
        md.append(
            f"| {i} | {r['type']} | {r['task'][:35]} | {r['video']} | "
            f"{r['h_peak']:.4f} | {r['h_robust']:.4f} | "
            f"{'**HALLU**' if r['is_hallu'] else 'CLEAN'} | "
            f"{'✓' if r['correct'] else '✗'} | {r['time_sec']:.1f}s |"
        )
    (args.out_dir / "summary_table.md").write_text("\n".join(md))

    # ── Console summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"N videos scored:          {len(rows)} ({summary['n_real']} real + {summary['n_gen']} gen)")
    print(f"AUROC:                    {auroc:.4f}")
    print(f"AP:                       {ap:.4f}")
    print(f"Threshold:                {scorer.threshold:.4f}")
    print(f"Confusion:                TP={tp}  FN={fn}  FP={fp}  TN={tn}")
    print(f"TPR / FPR / accuracy:     {tpr:.4f} / {fpr:.4f} / {summary['accuracy']:.4f}")
    print(f"Mean score (real):        {summary['mean_score_real']:.4f}")
    print(f"Mean score (gen):         {summary['mean_score_gen']:.4f}")
    print(f"")
    print(f"Total time:               {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Avg time per video:       {summary['avg_time_per_video']:.2f}s")
    print(f"")
    print(f"Outputs:")
    print(f"  {csv_path}")
    print(f"  {args.out_dir / 'summary.json'}")
    print(f"  {args.out_dir / 'summary_table.md'}")


if __name__ == "__main__":
    main()
