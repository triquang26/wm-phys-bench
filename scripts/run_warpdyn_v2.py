#!/usr/bin/env python3
"""WarpDyn v2 robust benchmark — all signals + per-video aggregation.

What's new vs run_gr1_warpdyn.py:
  - Cert-weighted cycle peak (robust on textured / symmetric scenes)
  - Adds S4 NN-set Jaccard signal (cheap, novel)
  - Computes DINOv2 jaccard null from the same 1472 real reference pairs
  - Per-video aggregation (trimmed mean over frames in same video)
  - Bootstrap 95% CI on both per-frame AND per-video AUROC

Outputs:
  paper-physical-gr1/pool/results_warpdyn_v2/
    raw_signals_v2.csv      per-frame raw cycle (recomputed) + jaccard
    nn_jaccard_null.npy     sorted null array (same 1472 pairs)
    per_frame_report.json   AUROC + CI on 290-frame eval
    per_video_report.json   AUROC + CI on (5+25) video eval
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

REPO_ROOT = Path("/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination")
BENCH = REPO_ROOT / "paper-physical-gr1"
POOL = BENCH / "pool"
REF_ROOT = BENCH / "reference"

# Sampled-pair list used to build the cycle null — must match expand_cycle_null.py
PAIRS_PER_TASK = 16


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def cauchy_combine(ps: list[float]) -> float:
    ps = [p for p in ps if p is not None and 0 < p < 1]
    if not ps:
        return 0.5
    t = float(np.mean([np.tan(np.pi * (0.5 - p)) for p in ps]))
    return float(0.5 - np.arctan(t) / np.pi)


def empirical_p(value: float, sorted_null: np.ndarray) -> float:
    n = sorted_null.size
    if n == 0:
        return 0.5
    rank = int(np.searchsorted(sorted_null, value, side="right"))
    p = (n - rank + 0.5) / (n + 1.0)
    return float(np.clip(p, 1.0 / (n + 1), 1.0 - 1.0 / (n + 1)))


def trimmed_mean(values: np.ndarray, trim: float = 0.1) -> float:
    if values.size == 0:
        return 0.0
    sorted_v = np.sort(values)
    n = len(sorted_v)
    k = int(n * trim)
    return float(sorted_v[k : n - k].mean()) if n - 2 * k > 0 else float(sorted_v.mean())


def bootstrap_auroc(y: np.ndarray, s: np.ndarray, n_boot: int = 1000) -> tuple[float, float, float]:
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


def parse_pool_frame(stem: str) -> tuple[str, str, int]:
    task_part, _, rest = stem.partition("__")
    if rest.startswith("v") and "_frame_" in rest:
        vid_part, _, frame_part = rest.partition("_frame_")
    else:
        vid_part = "real"
        frame_part = rest.replace("frame_", "")
    return task_part, vid_part, int(frame_part)


def group_frames_by_video(dirpath: Path):
    groups = defaultdict(list)
    for p in sorted(dirpath.glob("*.png")):
        task, vid, idx = parse_pool_frame(p.stem)
        groups[(task, vid)].append((idx, p))
    for k in groups:
        groups[k].sort()
    return groups


def sample_pairs_for_task(task_dir: Path, n_pairs: int):
    pngs = sorted(task_dir.glob("*.png"))
    if len(pngs) < 2:
        return []
    n_consecutive = len(pngs) - 1
    if n_consecutive <= n_pairs:
        idxs = list(range(n_consecutive))
    else:
        idxs = np.linspace(0, n_consecutive - 1, n_pairs).astype(int).tolist()
    return [(pngs[i], pngs[i + 1]) for i in idxs]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, default=POOL / "results_warpdyn_v2")
    ap.add_argument("--appearance", type=Path,
                    default=POOL / "results" / "summary.csv")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. DINOv2 cache for pool refs + all queries + all reference pairs
    from warp_score.adaptive_refs import DinoFeatureExtractor
    print("Loading DINOv2 …")
    dino = DinoFeatureExtractor("dinov2_vits14")
    dino._load()

    # Pool refs
    pool_paths = sorted((POOL / "reference" / "POOL").glob("*.png"))
    print(f"Embedding pool refs ({len(pool_paths)}) …")
    pool_feats = dino.extract(pool_paths)

    # Query frames
    query_paths = sorted((POOL / "query_high" / "POOL").glob("*.png")) + \
                  sorted((POOL / "query_low" / "POOL").glob("*.png"))
    print(f"Embedding query frames ({len(query_paths)}) …")
    q_feats = dino.extract(query_paths)
    q_feat_by_stem = {p.stem: q_feats[i] for i, p in enumerate(query_paths)}

    # ── 2. Compute NN-Jaccard for query consecutive pairs (S4 raw signal)
    print("\n=== Computing per-query NN-Jaccard ===")
    from warp_score.nn_consistency import nn_set_jaccard_distance

    real_groups = group_frames_by_video(POOL / "query_high" / "POOL")
    gen_groups = group_frames_by_video(POOL / "query_low" / "POOL")
    all_groups = {**real_groups, **gen_groups}

    jaccard_per_stem = {}
    for (task, vid), frames in sorted(all_groups.items()):
        is_real = vid == "real"
        for i in range(len(frames) - 1):
            p_a = frames[i][1]
            p_b = frames[i + 1][1]
            ja = nn_set_jaccard_distance(
                q_feat_by_stem[p_a.stem], q_feat_by_stem[p_b.stem],
                pool_feats, k=50,
            )
            jaccard_per_stem[p_a.stem] = {"jaccard": ja, "split": "high" if is_real else "low"}

    # ── 3. NN-Jaccard NULL from 92-task real reference pairs
    print("\n=== Building NN-Jaccard null from 1472 real reference pairs ===")
    task_dirs = sorted(d for d in REF_ROOT.iterdir() if d.is_dir())
    null_pairs = []
    for td in task_dirs:
        null_pairs.extend(sample_pairs_for_task(td, PAIRS_PER_TASK))
    print(f"  Sampled {len(null_pairs)} pairs across {len(task_dirs)} tasks")

    # Embed all unique frames in null_pairs
    unique_paths = sorted({p for pair in null_pairs for p in pair})
    print(f"  Embedding {len(unique_paths)} unique frames for jaccard null …")
    t0 = time.time()
    null_feats = dino.extract(unique_paths)
    feat_idx = {p: i for i, p in enumerate(unique_paths)}
    print(f"  DINOv2 embed: {time.time()-t0:.1f}s")

    null_jaccard = []
    for (a, b) in null_pairs:
        ja = nn_set_jaccard_distance(
            null_feats[feat_idx[a]], null_feats[feat_idx[b]], pool_feats, k=50,
        )
        null_jaccard.append(ja)
    null_jaccard_arr = np.sort(np.asarray(null_jaccard, dtype=np.float32))
    np.save(args.out_dir / "nn_jaccard_null.npy", null_jaccard_arr)
    print(f"  null_jaccard: n={null_jaccard_arr.size}  "
          f"med={np.median(null_jaccard_arr):.3f}  "
          f"p95={np.percentile(null_jaccard_arr, 95):.3f}  "
          f"p99={np.percentile(null_jaccard_arr, 99):.3f}")

    # ── 4. Recompute cycle signals for queries with cert-weighted formula
    print("\n=== Recomputing query cycle signals (cert-weighted) ===")
    from warp_score.matcher import RoMaMatcher
    from warp_score.temporal_signals import cycle_signal

    matcher = RoMaMatcher(setting="turbo", device="cuda", use_precision=True, vis_size=224)
    print("Loading RoMaV2 …")
    matcher._load_model()

    cycle_per_stem = {}
    t0 = time.time()
    total_pairs = sum(max(0, len(f) - 1) for f in all_groups.values())
    n_done = 0
    for (task, vid), frames in sorted(all_groups.items()):
        is_real = vid == "real"
        for i in range(len(frames) - 1):
            p_a = frames[i][1]
            p_b = frames[i + 1][1]
            fwd = matcher.match(p_a, p_b)
            bwd = matcher.match(p_b, p_a)
            sig = cycle_signal(fwd.warp, bwd.warp, cert_fwd=fwd.cert, cert_floor=0.1)
            cycle_per_stem[p_a.stem] = {
                "cycle_mean": sig["mean"],
                "cycle_peak": sig["peak"],
                "split":      "high" if is_real else "low",
            }
            n_done += 1
            if n_done % 30 == 0:
                rate = n_done / (time.time() - t0)
                print(f"  [{n_done}/{total_pairs}] rate={rate:.2f} pair/s")

    # ── 5. Cycle null (recompute with cert-weighted formula)
    print("\n=== Recomputing cycle NULL with cert-weighted formula on 1472 real pairs ===")
    cycle_null_mean = []
    cycle_null_peak = []
    t0 = time.time()
    for j, (a, b) in enumerate(null_pairs, 1):
        fwd = matcher.match(a, b)
        bwd = matcher.match(b, a)
        sig = cycle_signal(fwd.warp, bwd.warp, cert_fwd=fwd.cert, cert_floor=0.1)
        cycle_null_mean.append(sig["mean"])
        cycle_null_peak.append(sig["peak"])
        if j % 100 == 0 or j == len(null_pairs):
            rate = j / (time.time() - t0)
            eta = (len(null_pairs) - j) / max(rate, 1e-6)
            print(f"  [{j}/{len(null_pairs)}] rate={rate:.2f} pair/s   eta={eta/60:.1f} min")
    cycle_null_mean = np.sort(np.asarray(cycle_null_mean, dtype=np.float32))
    cycle_null_peak = np.sort(np.asarray(cycle_null_peak, dtype=np.float32))

    np.savez(args.out_dir / "cycle_null_v2.npz",
             cycle_mean=cycle_null_mean, cycle_peak=cycle_null_peak)
    print(f"  cycle_mean: med={np.median(cycle_null_mean):.3f}  p99={np.percentile(cycle_null_mean,99):.3f}")
    print(f"  cycle_peak: med={np.median(cycle_null_peak):.3f}  p99={np.percentile(cycle_null_peak,99):.3f}")

    # ── 6. Load appearance H_score from existing pool benchmark
    print("\n=== Loading appearance signal from pool benchmark ===")
    appear_by_stem = {r["frame"]: r for r in csv.DictReader(open(args.appearance))}

    # ── 7. Per-frame fusion
    print("\n=== Per-frame fusion ===")
    mapping = json.loads((POOL / "frame_to_task.json").read_text())
    per_frame = []
    for stem, cyc in cycle_per_stem.items():
        ja = jaccard_per_stem.get(stem, {}).get("jaccard")
        appear = appear_by_stem.get(stem)
        if appear is None:
            continue
        p_appear = float(appear["p_combined"])
        p_cm = empirical_p(cyc["cycle_mean"], cycle_null_mean)
        p_cp = empirical_p(cyc["cycle_peak"], cycle_null_peak)
        p_cycle = cauchy_combine([p_cm, p_cp])
        p_jacc = empirical_p(ja, null_jaccard_arr) if ja is not None else None

        configs = {
            "S1":            p_appear,
            "S1+S2":         cauchy_combine([p_appear, p_cycle]),
            "S1+S2+S4":      cauchy_combine([p_appear, p_cycle, p_jacc]),
            "S2+S4":         cauchy_combine([p_cycle, p_jacc]),
        }
        row = {"frame": stem, "split": cyc["split"],
               "task": mapping.get(stem),
               "p_appear": p_appear, "p_cycle": p_cycle, "p_jaccard": p_jacc,
               "cycle_mean": cyc["cycle_mean"], "cycle_peak": cyc["cycle_peak"],
               "jaccard": ja}
        for name, pv in configs.items():
            row[f"H_{name}"] = 1.0 - pv
        per_frame.append(row)

    # Save raw
    with open(args.out_dir / "raw_signals_v2.csv", "w", newline="") as f:
        if per_frame:
            w = csv.DictWriter(f, fieldnames=list(per_frame[0].keys()))
            w.writeheader()
            w.writerows(per_frame)

    # ── 8. Per-frame AUROC + CI
    print("\n=== PER-FRAME AUROC (95% CI bootstrap n_boot=1000) ===")
    y = np.array([0 if r["split"] == "high" else 1 for r in per_frame])
    per_frame_report = {}
    for cfg in ["S1", "S1+S2", "S1+S2+S4", "S2+S4"]:
        s = np.array([r[f"H_{cfg}"] for r in per_frame])
        au, lo, hi = bootstrap_auroc(y, s)
        per_frame_report[cfg] = {
            "auroc": au, "ci95_lo": lo, "ci95_hi": hi,
            "mean_real": float(s[y == 0].mean()),
            "mean_gen":  float(s[y == 1].mean()),
        }
        print(f"  {cfg:10s}  AUROC={au:.4f}  [{lo:.4f}, {hi:.4f}]  "
              f"Δ={s[y==1].mean()-s[y==0].mean():+.4f}")

    # ── 9. Per-VIDEO aggregation (trimmed mean across frames in same video)
    print("\n=== PER-VIDEO AGGREGATION ===")
    by_video = defaultdict(list)
    for r in per_frame:
        task, vid, _ = parse_pool_frame(r["frame"])
        by_video[(task, vid)].append(r)

    video_rows = []
    for (task, vid), rows in by_video.items():
        is_real = vid == "real"
        agg_row = {"task": task, "vid": vid, "split": "high" if is_real else "low",
                   "label": 0 if is_real else 1, "n_frames": len(rows)}
        for cfg in ["S1", "S1+S2", "S1+S2+S4", "S2+S4"]:
            vals = np.array([r[f"H_{cfg}"] for r in rows])
            agg_row[f"video_H_{cfg}"]      = trimmed_mean(vals, trim=0.1)
            agg_row[f"video_Hpeak_{cfg}"]  = float(np.percentile(vals, 80))
        video_rows.append(agg_row)

    y_vid = np.array([r["label"] for r in video_rows])
    per_video_report = {}
    print("Aggregator: TRIMMED MEAN (trim=0.1)")
    for cfg in ["S1", "S1+S2", "S1+S2+S4", "S2+S4"]:
        s = np.array([r[f"video_H_{cfg}"] for r in video_rows])
        if len(set(y_vid.tolist())) < 2:
            continue
        au, lo, hi = bootstrap_auroc(y_vid, s)
        per_video_report[cfg] = {
            "auroc": au, "ci95_lo": lo, "ci95_hi": hi,
            "n_videos": int(len(y_vid)),
            "n_real": int((y_vid == 0).sum()),
            "n_gen":  int((y_vid == 1).sum()),
            "mean_real": float(s[y_vid == 0].mean()),
            "mean_gen":  float(s[y_vid == 1].mean()),
        }
        print(f"  {cfg:10s}  AUROC={au:.4f}  [{lo:.4f}, {hi:.4f}]  "
              f"Δ={s[y_vid==1].mean()-s[y_vid==0].mean():+.4f}")

    print("\nAggregator: 80th PERCENTILE PEAK")
    per_video_peak_report = {}
    for cfg in ["S1", "S1+S2", "S1+S2+S4", "S2+S4"]:
        s = np.array([r[f"video_Hpeak_{cfg}"] for r in video_rows])
        if len(set(y_vid.tolist())) < 2:
            continue
        au, lo, hi = bootstrap_auroc(y_vid, s)
        per_video_peak_report[cfg] = {"auroc": au, "ci95_lo": lo, "ci95_hi": hi}
        print(f"  {cfg:10s}  AUROC={au:.4f}  [{lo:.4f}, {hi:.4f}]")

    # ── 10. Per-task breakdown for best config
    best_cfg = max(per_frame_report.keys(), key=lambda k: per_frame_report[k]["auroc"])
    print(f"\n=== PER-TASK (best config: {best_cfg}, per-frame) ===")
    by_task = defaultdict(list)
    for r, label in zip(per_frame, y):
        if r["task"] is None:
            continue
        by_task[r["task"]].append((int(label), r[f"H_{best_cfg}"]))
    per_task_report = {}
    for task, pairs in sorted(by_task.items()):
        yy = np.array([p[0] for p in pairs])
        ss = np.array([p[1] for p in pairs])
        if len(set(yy.tolist())) < 2:
            continue
        au, lo, hi = bootstrap_auroc(yy, ss, n_boot=500)
        per_task_report[task] = {"n": len(yy), "auroc": au, "ci95_lo": lo, "ci95_hi": hi}
        print(f"  {task[:55]:55s}  AUROC={au:.4f}  [{lo:.4f}, {hi:.4f}]")

    # ── Save full report
    report = {
        "n_null_pairs": int(cycle_null_mean.size),
        "best_config":  best_cfg,
        "per_frame":    per_frame_report,
        "per_video_trimmed_mean": per_video_report,
        "per_video_p80_peak":     per_video_peak_report,
        "per_task_best_config":   per_task_report,
    }
    out = args.out_dir / "final_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nFinal report → {out}")


if __name__ == "__main__":
    main()
