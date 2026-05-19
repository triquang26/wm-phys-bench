#!/usr/bin/env python3
"""Per-task ratio score: H_test / H_train_task — task-fair anomaly metric.

For each task T:
  H_train_task = WarpDyn H_peak when scoring the task's REAL training video
                 against its own per-task multi-lag null.

For any test video at task T:
  ratio = H_test_peak / H_train_task
  > 1.0  → more anomalous than training → likely hallu
  ~ 1.0  → on the boundary of training distribution
  < 1.0  → less anomalous than training → clean

FPR = 0% by construction (each task's threshold == its training H).

Reads paper-physical-gr1/per_task_dense_eval/per_task_dense_table.csv,
writes a ranked CSV + markdown summary.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path("/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination")
EVAL_DIR = REPO_ROOT / "paper-physical-gr1" / "per_task_dense_eval"
INPUT_CSV = EVAL_DIR / "per_task_dense_table.csv"
OUT_CSV = EVAL_DIR / "per_task_ratio_table.csv"
OUT_MD = EVAL_DIR / "per_task_ratio_ranking.md"


def main():
    rows = list(csv.DictReader(open(INPUT_CSV)))

    # Per-task training baselines
    task_real_h = {r["task"]: float(r["cycle_peak"]) for r in rows if r["type"] == "REAL"}

    # Compute ratio for every video (real and gen)
    out_rows = []
    for r in rows:
        h = float(r["cycle_peak"])
        h_train = task_real_h[r["task"]]
        ratio = h / h_train if h_train > 1e-8 else 0.0
        out_rows.append({
            "type":       r["type"],
            "task":       r["task"],
            "video":      r["video"],
            "h_test":     h,
            "h_train":    h_train,
            "ratio":      ratio,
            "delta":      h - h_train,
            "label":      r.get("label", ""),
            "verdict":    ("HALLU" if ratio > 1.0
                           else "borderline" if ratio > 0.95
                           else "clean"),
        })

    # Sort by ratio desc
    out_rows.sort(key=lambda x: -x["ratio"])

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"CSV → {OUT_CSV}")

    # ── Markdown
    md = ["# WarpDyn — Per-task ratio ranking\n"]
    md.append("Score = `H_test / H_train_task`. Per-task baseline calibration "
              "(FPR = 0% by construction).\n")
    md.append("## Per-task training baselines\n")
    md.append("| Task | H_train (training video) |")
    md.append("|---|---|")
    for t, h in sorted(task_real_h.items()):
        md.append(f"| {t} | {h:.4f} |")

    gens = [r for r in out_rows if r["type"] == "GEN"]
    n_hallu = sum(1 for r in gens if r["verdict"] == "HALLU")
    n_border = sum(1 for r in gens if r["verdict"] == "borderline")
    n_clean = sum(1 for r in gens if r["verdict"] == "clean")
    md.append(f"\n## Summary\n")
    md.append(f"- Total gen: {len(gens)}")
    md.append(f"- HALLU (ratio > 1.0): **{n_hallu}/{len(gens)}**")
    md.append(f"- Borderline (0.95-1.0): {n_border}/{len(gens)}")
    md.append(f"- Clean (< 0.95): {n_clean}/{len(gens)}")

    md.append(f"\n## Full ranking (gen videos)\n")
    md.append("| Rank | Task | Vid | H_test | H_train | Ratio | Δ | Verdict |")
    md.append("|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(gens, 1):
        verdict_emoji = {"HALLU": "🔴", "borderline": "⚠", "clean": "✓"}.get(r["verdict"], "?")
        md.append(f"| {i:2d} | {r['task']} | {r['video']} | {r['h_test']:.4f} | "
                  f"{r['h_train']:.4f} | **{r['ratio']:.3f}** | {r['delta']:+.3f} | "
                  f"{verdict_emoji} {r['verdict']} |")

    OUT_MD.write_text("\n".join(md))
    print(f"Markdown → {OUT_MD}")

    print(f"\nHallu detected (ratio > 1.0): {n_hallu}/{len(gens)}")
    print(f"Borderline (0.95-1.0):        {n_border}/{len(gens)}")
    print(f"Clean (< 0.95):               {n_clean}/{len(gens)}")


if __name__ == "__main__":
    main()
