#!/usr/bin/env python3
"""Re-evaluate WarpDyn with dense null + bootstrap CI.

Reuses pre-computed signals:
  - Appearance p-value from pool benchmark (summary.csv)
  - Cycle stats per query frame from earlier WarpDyn run (raw_signals.csv)

Plugs in the expanded cycle null (paper-physical-gr1/cycle_null_large.npz)
and reports:
  - AUROC + AP with 95% bootstrap CI (n_boot=1000)
  - Per-original-task breakdown
  - Comparison vs sparse null (n=45) baseline
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

REPO_ROOT = Path("/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination")
POOL = REPO_ROOT / "paper-physical-gr1" / "pool"


def cauchy_combine(ps: list[float]) -> float:
    ps = [p for p in ps if p is not None and 0 < p < 1]
    if not ps:
        return 0.5
    t = np.mean([np.tan(np.pi * (0.5 - p)) for p in ps])
    return float(0.5 - np.arctan(t) / np.pi)


def empirical_p(value: float, sorted_null: np.ndarray) -> float:
    n = sorted_null.size
    if n == 0:
        return 0.5
    rank = int(np.searchsorted(sorted_null, value, side="right"))
    p = (n - rank + 0.5) / (n + 1.0)
    return float(np.clip(p, 1.0 / (n + 1), 1.0 - 1.0 / (n + 1)))


def bootstrap_auroc(y: np.ndarray, s: np.ndarray, n_boot: int = 1000) -> tuple[float, float, float]:
    """Return (point estimate, lo_95, hi_95)."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(42)
    n = len(y)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yi, si = y[idx], s[idx]
        if len(set(yi.tolist())) < 2:
            continue
        boots.append(roc_auc_score(yi, si))
    boots = np.array(boots)
    return float(roc_auc_score(y, s)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--null_npz", type=Path,
                    default=REPO_ROOT / "paper-physical-gr1" / "cycle_null_large.npz")
    ap.add_argument("--summary",  type=Path,
                    default=POOL / "results" / "summary.csv")
    ap.add_argument("--raw_signals", type=Path,
                    default=POOL / "results_warpdyn" / "raw_signals.csv")
    ap.add_argument("--out_dir", type=Path,
                    default=POOL / "results_warpdyn_robust")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load dense null
    null = np.load(args.null_npz)
    null_cycle_mean = np.sort(null["cycle_mean"].astype(np.float32))
    null_cycle_peak = np.sort(null["cycle_peak"].astype(np.float32))
    n_null = null_cycle_mean.size
    print(f"Loaded dense cycle null: n_pairs={n_null}")
    print(f"  cycle_mean  med={np.median(null_cycle_mean):.3f}  p99={np.percentile(null_cycle_mean,99):.3f}")
    print(f"  cycle_peak  med={np.median(null_cycle_peak):.3f}  p99={np.percentile(null_cycle_peak,99):.3f}")

    # Load query signals
    s1_by_frame = {r["frame"]: r for r in csv.DictReader(open(args.summary))}
    raw_by_frame = {r["frame"]: r for r in csv.DictReader(open(args.raw_signals))}

    mapping = json.loads((POOL / "frame_to_task.json").read_text())

    # Fuse per frame
    fused = []
    for stem, s1 in s1_by_frame.items():
        raw = raw_by_frame.get(stem)
        if raw is None:
            continue
        p_appear = float(s1["p_combined"])
        cm = raw.get("cycle_mean")
        cp = raw.get("cycle_peak")
        p_cm = empirical_p(float(cm), null_cycle_mean) if cm not in (None, "", "None") else None
        p_cp = empirical_p(float(cp), null_cycle_peak) if cp not in (None, "", "None") else None
        p_cycle = cauchy_combine([p_cm, p_cp])
        p_fused = cauchy_combine([p_appear, p_cycle])
        fused.append({
            "frame": stem,
            "split": s1["split"],
            "p_appear": p_appear,
            "p_cycle": p_cycle,
            "H_S1": 1.0 - p_appear,
            "H_S1+S2": 1.0 - p_fused,
        })

    y = np.array([0 if r["split"] == "high" else 1 for r in fused])

    print(f"\n=== Overall AUROC with bootstrap 95% CI (n_boot=1000) ===")
    results = {}
    for cfg in ("H_S1", "H_S1+S2"):
        s = np.array([r[cfg] for r in fused])
        au, lo, hi = bootstrap_auroc(y, s)
        results[cfg] = {"auroc": au, "ci95_lo": lo, "ci95_hi": hi,
                        "mean_real": float(s[y == 0].mean()),
                        "mean_gen":  float(s[y == 1].mean())}
        label = cfg.replace("H_", "")
        print(f"  {label:8s}  AUROC = {au:.4f}  [{lo:.4f}, {hi:.4f}]  "
              f"Δmean={s[y==1].mean()-s[y==0].mean():+.4f}")

    print("\n=== Per-original-task (S1+S2) ===")
    by_task: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r, label in zip(fused, y):
        t = mapping.get(r["frame"])
        if t is None:
            continue
        by_task[t].append((int(label), r["H_S1+S2"]))

    per_task = {}
    for task, pairs in sorted(by_task.items()):
        yy = np.array([p[0] for p in pairs])
        ss = np.array([p[1] for p in pairs])
        if len(set(yy.tolist())) < 2:
            continue
        au, lo, hi = bootstrap_auroc(yy, ss, n_boot=500)
        per_task[task] = {"n": len(yy), "n_pos": int(yy.sum()),
                          "auroc": au, "ci95_lo": lo, "ci95_hi": hi}
        print(f"  {task[:55]:55s}  AUROC={au:.4f}  [{lo:.4f}, {hi:.4f}]")

    out = args.out_dir / "robust_report.json"
    out.write_text(json.dumps({
        "n_null_pairs": int(n_null),
        "overall": results,
        "per_task": per_task,
    }, indent=2))
    print(f"\nReport → {out}")


if __name__ == "__main__":
    main()
