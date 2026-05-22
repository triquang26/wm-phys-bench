"""Bench 23 doanh tasks: high as training ref, low as query.

For each task, calls scripts/benchmark_one_task.py with:
  --query   raw_videos/low/<i>_<task>.mp4
  --ref_dir reference/<i>_<task>/
  --real_mp4 raw_videos/high/<i>_<task>.mp4

Aggregates ratios into a CSV + summary.
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO = Path("/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination")
BENCH_SCRIPT = REPO / ".claude/worktrees/feat-knn-pool-gr1/scripts/benchmark_one_task.py"
BENCH = REPO / "paper-doanh-eval"
PYTHON = sys.executable

eval_tasks = json.loads((BENCH / "eval_tasks.json").read_text())
OUT_BASE = REPO / ".claude/worktrees/feat-knn-pool-gr1/outputs/doanh_eval_demo"
OUT_BASE.mkdir(parents=True, exist_ok=True)

rows = []
for ts, task in sorted(eval_tasks.items(), key=lambda x: int(x[0])):
    task_full = task["task_full"]
    folder = f"{ts}_{task_full}"[:200]
    high_mp4 = BENCH / "raw_videos/high" / f"{folder}.mp4"
    low_mp4 = BENCH / "raw_videos/low" / f"{folder}.mp4"
    ref_dir = BENCH / "reference" / folder
    out_dir = OUT_BASE / folder

    print(f"\n{'='*70}\n[{ts}] {task_full[:60]}\n{'='*70}")
    if not low_mp4.exists() or not high_mp4.exists() or not ref_dir.exists():
        print(f"  [skip] missing files")
        continue
    rc = subprocess.run([
        PYTHON, str(BENCH_SCRIPT),
        "--task", folder,
        "--query", str(low_mp4),
        "--out_dir", str(out_dir),
        "--ref_dir", str(ref_dir),
        "--real_mp4", str(high_mp4),
    ], check=False).returncode
    print(f"  exit: {rc}")
    if rc == 0:
        t = json.loads((out_dir / "timing.json").read_text())
        o = t["online"]
        rows.append({
            "task_short": ts,
            "task_full": task_full,
            "eval_subfolder": task.get("eval_subfolder", ""),
            "H_train_cycle": t["h_train"]["cycle_peak"],
            "H_train_knn": t["h_train"]["knn_peak"],
            "H_train_fused": t["h_train"]["fused_peak"],
            "H_test_cycle": o["cycle_peak"],
            "H_test_knn": o["knn_peak"],
            "H_test_fused": o["fused_peak"],
            "ratio_cycle": o["ratio_cycle"],
            "ratio_knn": o["ratio_knn"],
            "ratio_fused": o["ratio_fused"],
            "verdict": "HALLU" if o["ratio_fused"] > 1.0 else ("borderline" if o["ratio_fused"] > 0.95 else "clean"),
        })

if rows:
    csv_path = OUT_BASE / "ratio_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n→ {csv_path}")
    print(f"\nSummary across {len(rows)} tasks:")
    import statistics
    ratios = [r["ratio_fused"] for r in rows]
    print(f"  ratio_fused mean = {statistics.mean(ratios):.3f}")
    print(f"  ratio_fused median = {statistics.median(ratios):.3f}")
    hallu = sum(1 for r in rows if r["ratio_fused"] > 1.0)
    bord = sum(1 for r in rows if 0.95 < r["ratio_fused"] <= 1.0)
    clean = sum(1 for r in rows if r["ratio_fused"] <= 0.95)
    print(f"  HALLU      (>1.0)  : {hallu}/{len(rows)}")
    print(f"  borderline (0.95-1): {bord}/{len(rows)}")
    print(f"  clean      (≤0.95) : {clean}/{len(rows)}")
