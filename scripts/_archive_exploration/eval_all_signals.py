#!/usr/bin/env python3
"""End-to-end eval of ALL 4 temporal signals + AND/OR voting.

Signals tested:
  1. CycleSignal              — fwd/bwd warp composition drift
  2. PrecisionAnomalySignal   — cycle × sqrt(det(precision))
  3. LostPixelSignal          — fraction of "lost" pixels (cert/precision low)
  4. BidirectionalCertSignal  — fwd cert high but bwd cert at destination low

Plus voting strategies:
  AND2     — flag if BOTH cycle and lost agree (intersect of top-tail)
  OR-all   — flag if any signal triggers (union)

Pipeline (identical to inference time):
  - 50 real videos for null calibration
  - 30 real + 24 gen for evaluation
  - 10 frames per video via np.linspace
  - SAM3 segmentation + RoMa with precision matrix output
"""
from __future__ import annotations

import argparse
import csv
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

REPO_ROOT = Path("/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination")
BENCH = REPO_ROOT / "paper-physical-gr1"
RAW_VIDEO_ROOT = BENCH / "raw_videos" / "gr1"
GEN_ROOT = BENCH / "generated"
OUT = BENCH / "eval_all_signals"


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_real_calib", type=int, default=50)
    ap.add_argument("--max_real_eval",  type=int, default=30)
    ap.add_argument("--n_frames",       type=int, default=10)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    from warp_score.temporal_signals import (
        CycleSignal, PrecisionAnomalySignal, LostPixelSignal,
        BidirectionalCertSignal, empirical_p_value,
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
        LostPixelSignal(cert_thresh=0.3, prec_thresh=0.1),
        BidirectionalCertSignal(cert_floor=0.0),
    ]
    signal_names = [s.name for s in signals]
    print(f"Signals: {signal_names}\n")

    # ── Discover videos
    all_real = sorted(RAW_VIDEO_ROOT.glob("*.mp4"))
    all_gen = sorted(GEN_ROOT.glob("*/v*.mp4"))
    idx = np.linspace(0, len(all_real) - 1,
                      args.max_real_calib + args.max_real_eval).astype(int)
    sampled = [all_real[i] for i in idx]
    real_calib = sampled[:args.max_real_calib]
    real_eval = sampled[args.max_real_calib:]
    print(f"Calib reals: {len(real_calib)}   Eval reals: {len(real_eval)}   Eval gens: {len(all_gen)}\n")

    def score_pair_signals(p_a: Path, p_b: Path) -> dict[str, dict]:
        """Compute all signals on one pair. Returns {sig_name: {"mean":, "peak":}}."""
        fwd = matcher.match(p_a, p_b)
        bwd = matcher.match(p_b, p_a)
        out = {}
        for s in signals:
            r = s.compute(fwd, bwd)
            out[s.name] = {"mean": r.mean, "peak": r.peak}
        return out

    def score_video(mp4: Path) -> list[dict]:
        """Sample, segment, score all pairs in the video. Returns list of pair-dicts."""
        bgrs = sample_frames(mp4, args.n_frames)
        if len(bgrs) < 2:
            return []
        seg_bgrs = [seg.segment_frame(b) for b in bgrs]
        tmp = Path(tempfile.mkdtemp(prefix="evalall_"))
        paths = []
        for i, b in enumerate(seg_bgrs):
            p = tmp / f"f_{i:04d}.png"
            cv2.imwrite(str(p), b)
            paths.append(p)
        pair_stats = []
        for t in range(len(paths) - 1):
            pair_stats.append(score_pair_signals(paths[t], paths[t + 1]))
        for p in paths:
            p.unlink(missing_ok=True)
        tmp.rmdir()
        return pair_stats

    # ── Step 1: null distributions
    print("=== Building null on calibration reals ===")
    null = {n: {"mean": [], "peak": []} for n in signal_names}
    t0 = time.time()
    for i, mp4 in enumerate(real_calib, 1):
        pairs = score_video(mp4)
        for pair in pairs:
            for n, s in pair.items():
                null[n]["mean"].append(s["mean"])
                null[n]["peak"].append(s["peak"])
        rate = i / (time.time() - t0)
        eta = (len(real_calib) - i) / max(rate, 1e-6)
        print(f"  [{i:2d}/{len(real_calib)}] {mp4.name:15s}  +{len(pairs)} pairs   "
              f"rate={rate:.2f} vid/s  eta={eta/60:.1f} min")

    sorted_null = {n: {"mean": np.sort(np.asarray(v["mean"], dtype=np.float32)),
                       "peak": np.sort(np.asarray(v["peak"], dtype=np.float32))}
                   for n, v in null.items()}

    np.savez(OUT / "null.npz", **{
        f"{n}_{stat}": sorted_null[n][stat]
        for n in signal_names for stat in ("mean", "peak")
    })

    print(f"\nNull sizes: {len(sorted_null[signal_names[0]]['mean'])} pairs each")
    for n in signal_names:
        m = sorted_null[n]["mean"]
        p = sorted_null[n]["peak"]
        print(f"  {n:18s}  mean p99={np.percentile(m, 99):.4f}   peak p99={np.percentile(p, 99):.4f}")

    # ── Step 2: score eval videos
    print("\n=== Scoring eval videos ===")
    eval_results = []

    def aggregate_per_video(pair_stats: list[dict], sig_name: str) -> dict:
        """Cauchy-fuse mean+peak p-values per pair, then aggregate."""
        h_pairs = []
        for pair in pair_stats:
            s = pair[sig_name]
            p_m = empirical_p_value(s["mean"], sorted_null[sig_name]["mean"])
            p_p = empirical_p_value(s["peak"], sorted_null[sig_name]["peak"])
            h_pairs.append(1.0 - cauchy_combine([p_m, p_p]))
        h = np.array(h_pairs, dtype=np.float32)
        if h.size == 0:
            return {"h_peak": 0.0, "h_mean": 0.0}
        return {
            "h_peak": float(np.percentile(h, 80)),
            "h_mean": float(h.mean()),
        }

    def score_one(mp4: Path, vtype: str, label: int, task: str):
        t1 = time.time()
        pairs = score_video(mp4)
        per_sig = {n: aggregate_per_video(pairs, n) for n in signal_names}
        elapsed = time.time() - t1
        row = {"video": mp4.name, "type": vtype, "task": task, "label": label, "time_sec": elapsed}
        for n in signal_names:
            row[f"{n}_peak"] = per_sig[n]["h_peak"]
            row[f"{n}_mean"] = per_sig[n]["h_mean"]
        return row

    for i, mp4 in enumerate(real_eval, 1):
        r = score_one(mp4, "REAL", 0, mp4.stem)
        eval_results.append(r)
        print(f"  [REAL {i:2d}/{len(real_eval)}] {mp4.name:12s}  "
              + "  ".join(f"{n}={r[n+'_peak']:.3f}" for n in signal_names)
              + f"  ({r['time_sec']:.1f}s)")
    for i, mp4 in enumerate(all_gen, 1):
        r = score_one(mp4, "GEN", 1, mp4.parent.name)
        eval_results.append(r)
        print(f"  [GEN  {i:2d}/{len(all_gen)}] {mp4.name:12s}  "
              + "  ".join(f"{n}={r[n+'_peak']:.3f}" for n in signal_names)
              + f"  ({r['time_sec']:.1f}s)")

    # ── Step 3: save CSV
    csv_path = OUT / "eval_table.csv"
    with open(csv_path, "w", newline="") as f:
        fns = list(eval_results[0].keys())
        w = csv.DictWriter(f, fieldnames=fns)
        w.writeheader()
        w.writerows(eval_results)
    print(f"\nTable → {csv_path}")

    # ── Step 4: comparison
    from sklearn.metrics import roc_auc_score, average_precision_score
    y = np.array([r["label"] for r in eval_results])

    print("\n" + "=" * 70)
    print("SIDE-BY-SIDE per-signal results")
    print("=" * 70)
    print(f"{'Signal':<22} {'AUROC':>7} {'mean_R':>8} {'mean_G':>8} "
          f"{'T@FPR=0':>9} {'gen@FPR=0':>10}  {'T@FPR≤5%':>10} {'gen@5%':>8}")
    print("-" * 95)
    per_signal_results = {}
    for n in signal_names:
        scores = np.array([r[f"{n}_peak"] for r in eval_results])
        real_s = scores[y == 0]
        gen_s = scores[y == 1]
        auroc = float(roc_auc_score(y, scores))
        T_safe = float(real_s.max()) + 1e-6
        T_p95 = float(np.quantile(real_s, 0.95))
        gen_safe = int((gen_s > T_safe).sum())
        gen_p95 = int((gen_s > T_p95).sum())
        per_signal_results[n] = {
            "auroc": auroc,
            "real_mean": float(real_s.mean()),
            "gen_mean": float(gen_s.mean()),
            "T_safe": T_safe, "gen_safe": gen_safe,
            "T_p95": T_p95, "gen_p95": gen_p95,
            "scores": scores,
        }
        print(f"{n:<22} {auroc:>7.4f} {real_s.mean():>8.4f} {gen_s.mean():>8.4f} "
              f"{T_safe:>9.4f} {gen_safe:>3d}/{len(gen_s):d}{'':>5} "
              f"{T_p95:>10.4f} {gen_p95:>3d}/{len(gen_s):d}")

    # ── Step 5: voting strategies
    print("\n" + "=" * 70)
    print("VOTING strategies (per-signal threshold = max(real) + ε)")
    print("=" * 70)
    thresholds_safe = {n: per_signal_results[n]["T_safe"] for n in signal_names}

    # OR-all: flag if any signal above its threshold
    or_flags = np.zeros(len(eval_results), dtype=bool)
    and_flags = np.ones(len(eval_results), dtype=bool)
    for n in signal_names:
        above = per_signal_results[n]["scores"] > thresholds_safe[n]
        or_flags |= above
        and_flags &= above

    for label_name, flags in [("OR-all (any signal)", or_flags),
                              ("AND-all (every signal)", and_flags)]:
        real_caught = int(flags[y == 0].sum())
        gen_caught = int(flags[y == 1].sum())
        print(f"  {label_name:<26}  real flagged: {real_caught}/{(y==0).sum()}  "
              f"gen caught: {gen_caught}/{(y==1).sum()}")

    # OR pairs (cycle + lost)
    cycle_above = per_signal_results["cycle"]["scores"] > thresholds_safe["cycle"]
    lost_above = per_signal_results["lost_pixel"]["scores"] > thresholds_safe["lost_pixel"]
    or_cl = cycle_above | lost_above
    print(f"  {'OR(cycle, lost_pixel)':<26}  real flagged: {int(or_cl[y==0].sum())}/{(y==0).sum()}  "
          f"gen caught: {int(or_cl[y==1].sum())}/{(y==1).sum()}")

    # ── Step 6: top real outliers per signal
    print("\n" + "=" * 70)
    print("Top 10 REAL outliers per signal")
    print("=" * 70)
    real_rows = [r for r in eval_results if r["type"] == "REAL"]
    for n in signal_names:
        srt = sorted(real_rows, key=lambda r: -r[f"{n}_peak"])[:10]
        print(f"\n{n}:")
        for i, r in enumerate(srt, 1):
            print(f"  {i:>2}. {r['task']:<8} {r['video']:<14} → {r[f'{n}_peak']:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
