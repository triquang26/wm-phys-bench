#!/usr/bin/env python3
"""WarpDyn benchmark — fuse appearance (S1) + cycle (S2) + trajectory (S3).

Pipeline:
  1. Group pool frames by (task, video) → temporal neighbors map.
  2. Compute cycle signal for each consecutive pair using RoMa.
  3. Compute traj signal for each consecutive triplet.
  4. Build null distributions from REAL-only pairs/triplets.
  5. Empirical p-values → Cauchy fusion with appearance signal (from existing
     pool benchmark summary.csv).
  6. Eval AUROC for each ablation: S1, S1+S2, S1+S3, S1+S2+S3.

No optical-flow model — RoMa dense warps are the only source of motion info.

Usage (groot env):
    python scripts/run_gr1_warpdyn.py [--out_dir paper-physical-gr1/warpdyn]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

REPO_ROOT = Path("/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination")
POOL = REPO_ROOT / "paper-physical-gr1" / "pool"


# ─────────────────────────────────────────────────────────────────────────────
# Frame grouping
# ─────────────────────────────────────────────────────────────────────────────


def parse_pool_frame(stem: str) -> tuple[str, str, int]:
    """Parse `t<N>__v<NNNN>_frame_<MMMM>` or `t<N>__frame_<MMMM>` → (task, vid, idx).

    Real held-out has no video prefix; we tag it 'real' so each task's real
    sequence is one virtual video.
    """
    task_part, _, rest = stem.partition("__")
    if rest.startswith("v") and "_frame_" in rest:
        vid_part, _, frame_part = rest.partition("_frame_")
    else:
        vid_part = "real"
        frame_part = rest.replace("frame_", "")
    return task_part, vid_part, int(frame_part)


def group_frames_by_video(dirpath: Path) -> dict[tuple[str, str], list[tuple[int, Path]]]:
    groups: dict[tuple[str, str], list[tuple[int, Path]]] = defaultdict(list)
    for p in sorted(dirpath.glob("*.png")):
        task, vid, idx = parse_pool_frame(p.stem)
        groups[(task, vid)].append((idx, p))
    for k in groups:
        groups[k].sort()
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# Cycle + trajectory signals per pair/triplet
# ─────────────────────────────────────────────────────────────────────────────


def compute_pair_signals(matcher, p_a: Path, p_b: Path) -> dict:
    """Return cycle signal for pair (a, b). Forward + backward RoMa match."""
    from warp_score.temporal_signals import cycle_signal

    fwd = matcher.match(p_a, p_b)
    bwd = matcher.match(p_b, p_a)
    sig = cycle_signal(fwd.warp, bwd.warp, cert_fwd=fwd.cert)
    return {
        "cycle_mean": sig["mean"],
        "cycle_peak": sig["peak"],
        # Keep warps around for trajectory recompute
        "warp_ab": fwd.warp,
        "warp_ba": bwd.warp,
    }


def compute_triplet_signals(warp_01: np.ndarray, warp_12: np.ndarray) -> dict:
    from warp_score.temporal_signals import trajectory_accel

    sig = trajectory_accel(warp_01, warp_12, grid_size=16)
    return {"traj_mean": sig["mean"], "traj_peak": sig["peak"]}


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path,
                    default=POOL / "results_warpdyn")
    ap.add_argument("--summary",
                    type=Path,
                    default=POOL / "results" / "summary.csv",
                    help="Appearance summary.csv from pool benchmark (S1).")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load existing appearance scores (S1)
    print(f"Loading appearance scores from {args.summary}")
    s1_rows = list(csv.DictReader(open(args.summary)))
    s1_by_frame: dict[str, dict] = {}
    for r in s1_rows:
        # Stem comparison key
        s1_by_frame[r["frame"]] = {
            "split": r["split"],
            "H_appear": float(r["H_score"]),
            "p_appear": float(r["p_combined"]),
        }
    print(f"  Loaded {len(s1_by_frame)} appearance scores")

    # Group frames by (task, video) — both query_high (real) and query_low (gen)
    print("\nGrouping frames by (task, video) ...")
    real_groups = group_frames_by_video(POOL / "query_high" / "POOL")
    gen_groups = group_frames_by_video(POOL / "query_low" / "POOL")
    all_groups = {**real_groups, **gen_groups}
    n_real_videos = len(real_groups)
    n_gen_videos = len(gen_groups)
    print(f"  Real videos: {n_real_videos}   Gen videos: {n_gen_videos}")

    # Load matcher
    from warp_score.matcher import RoMaMatcher
    matcher = RoMaMatcher(setting="turbo", device="cuda", use_precision=True, vis_size=224)
    print("Loading RoMaV2 ...")
    matcher._load_model()
    print("  OK\n")

    # Cache pair signals → keyed by (frame_a_stem, frame_b_stem)
    pair_cache: dict[tuple[str, str], dict] = {}

    print("=== Computing cycle + traj signals per video ===")
    per_frame_temporal: dict[str, dict] = {}

    for (task, vid), frames in sorted(all_groups.items()):
        is_real_video = vid == "real"
        kind = "REAL" if is_real_video else "GEN "
        n = len(frames)
        if n < 2:
            continue

        # Pair signals for consecutive (i, i+1)
        warps: list[Optional[np.ndarray]] = [None] * n
        cycles_for_frame: list[Optional[dict]] = [None] * n

        for i in range(n - 1):
            (idx_a, p_a), (idx_b, p_b) = frames[i], frames[i + 1]
            key = (p_a.stem, p_b.stem)
            res = compute_pair_signals(matcher, p_a, p_b)
            pair_cache[key] = res
            warps[i] = res["warp_ab"]
            cycles_for_frame[i] = {
                "cycle_mean": res["cycle_mean"],
                "cycle_peak": res["cycle_peak"],
            }

        # Trajectory needs warp_01 and warp_12 — defined for frame i if both pairs exist
        for i in range(n - 2):
            if warps[i] is None or warps[i + 1] is None:
                continue
            traj = compute_triplet_signals(warps[i], warps[i + 1])
            f_stem = frames[i][1].stem
            cyc = cycles_for_frame[i] or {}
            per_frame_temporal[f_stem] = {
                "cycle_mean": cyc.get("cycle_mean"),
                "cycle_peak": cyc.get("cycle_peak"),
                "traj_mean":  traj["traj_mean"],
                "traj_peak":  traj["traj_peak"],
                "split":      "high" if is_real_video else "low",
            }

        # Edge frames (i = n-2, n-1) get cycle but no traj
        for i in (n - 2, n - 1):
            if i < 0 or i >= n:
                continue
            f_stem = frames[i][1].stem
            if f_stem in per_frame_temporal:
                continue
            cyc = cycles_for_frame[i] if i < n - 1 else None
            per_frame_temporal[f_stem] = {
                "cycle_mean": cyc["cycle_mean"] if cyc else None,
                "cycle_peak": cyc["cycle_peak"] if cyc else None,
                "traj_mean":  None,
                "traj_peak":  None,
                "split":      "high" if is_real_video else "low",
            }

        print(f"  [{kind}] {task}/{vid}  n={n}  pairs={n-1}  triplets={n-2}")

    print(f"\nFrames with temporal signal: {len(per_frame_temporal)}")

    # ── Build null distributions from REAL pairs/triplets ────────────────────
    print("\n=== Building null distributions from REAL only ===")
    null_cycle_mean = []
    null_cycle_peak = []
    null_traj_mean = []
    null_traj_peak = []
    for stem, sig in per_frame_temporal.items():
        if sig["split"] != "high":
            continue
        if sig["cycle_mean"] is not None:
            null_cycle_mean.append(sig["cycle_mean"])
            null_cycle_peak.append(sig["cycle_peak"])
        if sig["traj_mean"] is not None:
            null_traj_mean.append(sig["traj_mean"])
            null_traj_peak.append(sig["traj_peak"])

    null_cycle_mean = np.sort(np.asarray(null_cycle_mean, dtype=np.float32))
    null_cycle_peak = np.sort(np.asarray(null_cycle_peak, dtype=np.float32))
    null_traj_mean  = np.sort(np.asarray(null_traj_mean,  dtype=np.float32))
    null_traj_peak  = np.sort(np.asarray(null_traj_peak,  dtype=np.float32))

    print(f"  null_cycle_mean: n={null_cycle_mean.size}  "
          f"med={np.median(null_cycle_mean):.4f}  p99={np.percentile(null_cycle_mean, 99):.4f}")
    print(f"  null_cycle_peak: n={null_cycle_peak.size}  "
          f"med={np.median(null_cycle_peak):.4f}  p99={np.percentile(null_cycle_peak, 99):.4f}")
    print(f"  null_traj_mean:  n={null_traj_mean.size}  "
          f"med={np.median(null_traj_mean):.4f}  p99={np.percentile(null_traj_mean, 99):.4f}")
    print(f"  null_traj_peak:  n={null_traj_peak.size}  "
          f"med={np.median(null_traj_peak):.4f}  p99={np.percentile(null_traj_peak, 99):.4f}")

    # Save raw signals
    raw_csv = args.out_dir / "raw_signals.csv"
    with open(raw_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "split", "cycle_mean", "cycle_peak", "traj_mean", "traj_peak"])
        for stem, sig in per_frame_temporal.items():
            w.writerow([stem, sig["split"],
                        sig["cycle_mean"], sig["cycle_peak"],
                        sig["traj_mean"],  sig["traj_peak"]])
    print(f"\nRaw signals → {raw_csv}")

    # ── Empirical p-values + ablation Cauchy fusion ──────────────────────────
    from warp_score.temporal_signals import empirical_p_value

    def cauchy_combine(ps: list[float]) -> float:
        """ACAT (Cauchy) combination of valid p-values."""
        ps = [p for p in ps if p is not None and 0 < p < 1]
        if not ps:
            return 0.5
        # T = mean(tan(pi*(0.5 - p))) ; p_comb = 0.5 - arctan(T)/pi
        t = np.mean([np.tan(np.pi * (0.5 - p)) for p in ps])
        return float(0.5 - np.arctan(t) / np.pi)

    fused_rows = []
    for stem, sig in per_frame_temporal.items():
        s1 = s1_by_frame.get(stem)
        if s1 is None:
            continue
        p_appear = s1["p_appear"]
        p_cycle_mean = empirical_p_value(sig["cycle_mean"], null_cycle_mean) if sig["cycle_mean"] is not None else None
        p_cycle_peak = empirical_p_value(sig["cycle_peak"], null_cycle_peak) if sig["cycle_peak"] is not None else None
        p_traj_mean  = empirical_p_value(sig["traj_mean"],  null_traj_mean)  if sig["traj_mean"]  is not None else None
        p_traj_peak  = empirical_p_value(sig["traj_peak"],  null_traj_peak)  if sig["traj_peak"]  is not None else None

        # Combined cycle p (mean + peak)
        p_cycle = cauchy_combine([p_cycle_mean, p_cycle_peak])
        p_traj  = cauchy_combine([p_traj_mean, p_traj_peak])

        ablations = {
            "S1":         p_appear,
            "S1+S2":      cauchy_combine([p_appear, p_cycle]),
            "S1+S3":      cauchy_combine([p_appear, p_traj]),
            "S1+S2+S3":   cauchy_combine([p_appear, p_cycle, p_traj]),
        }

        row = {
            "frame": stem, "split": sig["split"],
            "p_appear": p_appear, "p_cycle": p_cycle, "p_traj": p_traj,
        }
        for name, pv in ablations.items():
            row[f"H_{name}"] = 1.0 - pv
        fused_rows.append(row)

    print(f"\nFused {len(fused_rows)} frames")

    # ── Eval AUROC per ablation ──────────────────────────────────────────────
    from sklearn.metrics import roc_auc_score, average_precision_score

    mapping = json.loads((POOL / "frame_to_task.json").read_text())
    y = np.array([0 if r["split"] == "high" else 1 for r in fused_rows])

    print("\n=== ABLATION RESULTS ===")
    ablation_names = ["S1", "S1+S2", "S1+S3", "S1+S2+S3"]
    overall = {}
    for name in ablation_names:
        s = np.array([r[f"H_{name}"] for r in fused_rows])
        au = roc_auc_score(y, s)
        ap = average_precision_score(y, s)
        overall[name] = {"auroc": float(au), "ap": float(ap),
                         "mean_real": float(s[y == 0].mean()),
                         "mean_gen":  float(s[y == 1].mean())}
        print(f"  {name:10s}  AUROC={au:.4f}  AP={ap:.4f}  "
              f"mean(real)={s[y==0].mean():.4f}  mean(gen)={s[y==1].mean():.4f}")

    # Per-original-task
    print("\n=== PER-TASK (best ablation S1+S2+S3) ===")
    by_task: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r, label in zip(fused_rows, y):
        orig_task = mapping.get(r["frame"])
        if orig_task is None:
            continue
        by_task[orig_task].append((int(label), r["H_S1+S2+S3"]))

    per_task = {}
    for task, pairs in sorted(by_task.items()):
        yy = np.array([p[0] for p in pairs])
        ss = np.array([p[1] for p in pairs])
        if len(set(yy.tolist())) < 2:
            continue
        per_task[task] = {
            "n": len(yy), "n_pos": int(yy.sum()),
            "auroc": float(roc_auc_score(yy, ss)),
            "ap":    float(average_precision_score(yy, ss)),
        }
        print(f"  {task[:60]:60s}  AUROC={per_task[task]['auroc']:.4f}")

    out_report = args.out_dir / "ablation_report.json"
    out_report.write_text(json.dumps({"overall": overall, "per_task": per_task}, indent=2))
    print(f"\nReport → {out_report}")


if __name__ == "__main__":
    main()
