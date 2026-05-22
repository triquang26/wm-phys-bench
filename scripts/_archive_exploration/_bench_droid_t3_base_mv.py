"""Bench DROID task 3 base Cosmos — 2 base videos × 3 views (single-view per view)."""
import json
import math
import subprocess
import sys
from pathlib import Path

REPO = Path("/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination")
BENCH = REPO / ".claude/worktrees/feat-knn-pool-gr1/scripts/benchmark_one_task.py"
PYTHON = sys.executable
TASK_FULL = "3_Move the cup to the right"

VIDEOS = ["v0000", "v0001"]
VIEWS = ["exterior_1", "exterior_2", "wrist"]
OUT_BASE = REPO / ".claude/worktrees/feat-knn-pool-gr1/outputs/benchmark_demo"
BASE_GEN_ROOT = REPO / "paper-physical-droid/generated_base_cosmos" / TASK_FULL


def cauchy_combine(ps):
    ps = [p for p in ps if 0.0 < p < 1.0]
    if not ps:
        return 0.5
    t = sum(math.tan(math.pi * (0.5 - p)) for p in ps) / len(ps)
    return 0.5 - math.atan(t) / math.pi


results = {}
for vid in VIDEOS:
    results[vid] = {}
    for view in VIEWS:
        out = OUT_BASE / f"droid_t3_base_{vid}_{view}"
        query = BASE_GEN_ROOT / f"{vid}_views" / f"{view}.mp4"
        ref_dir = REPO / f"paper-physical-droid/reference/{TASK_FULL}/{view}"
        real_mp4 = REPO / f"paper-physical-droid/raw_videos/droid/3/{view}.mp4"
        print(f"\n{'='*70}\nBASE {vid} / {view}\n{'='*70}")
        if not query.exists():
            print(f"  [skip] missing query")
            continue
        rc = subprocess.run([
            PYTHON, str(BENCH),
            "--task", TASK_FULL,
            "--query", str(query),
            "--out_dir", str(out),
            "--ref_dir", str(ref_dir),
            "--real_mp4", str(real_mp4),
        ], check=False).returncode
        print(f"  exit: {rc}")
        if rc == 0:
            t = json.loads((out / "timing.json").read_text())
            results[vid][view] = t["online"]

# Cross-view fuse
print("\n" + "=" * 70)
print("BASE COSMOS MULTI-VIEW SUMMARY")
print("=" * 70)
summary = {}
for vid in VIDEOS:
    print(f"\n[{vid}]")
    multi_view = {}
    for view in VIEWS:
        if view in results[vid]:
            r = results[vid][view]["ratio_fused"]
            multi_view[f"ratio_{view}_fused"] = r
            print(f"  {view}: cycle={results[vid][view]['ratio_cycle']:.3f}  knn={results[vid][view]['ratio_knn']:.3f}  fused={r:.3f}")
    h_views = [results[vid][v]["fused_peak"] for v in VIEWS if v in results[vid]]
    p_views = [max(1e-6, min(1 - 1e-6, 1 - h)) for h in h_views]
    h_cross = 1 - cauchy_combine(p_views)
    h_train_views = [results[vid][v]["fused_peak"] / results[vid][v]["ratio_fused"]
                     if results[vid][v]["ratio_fused"] > 1e-8 else 0
                     for v in VIEWS if v in results[vid]]
    p_train_views = [max(1e-6, min(1 - 1e-6, 1 - h)) for h in h_train_views]
    h_train_cross = 1 - cauchy_combine(p_train_views)
    ratio_mv = h_cross / max(h_train_cross, 1e-8)
    multi_view["H_cross"] = h_cross
    multi_view["H_train_cross"] = h_train_cross
    multi_view["ratio_multiview"] = ratio_mv
    multi_view["verdict"] = "HALLU" if ratio_mv > 1.0 else ("borderline" if ratio_mv > 0.95 else "clean")
    print(f"  → ratio_multiview = {ratio_mv:.3f}  →  {multi_view['verdict']}")
    summary[vid] = multi_view

summary_path = OUT_BASE / "droid_t3_base_multiview_summary.json"
summary_path.write_text(json.dumps(summary, indent=2))
print(f"\n→ {summary_path}")
