#!/usr/bin/env python3
"""Per-task DENSE null eval — multi-lag cycle + DINOv2 k-NN fusion.

OFFLINE per task T:
  1. Load N refs from REF_ROOT/T (50 or 182 — densified by scripts/extract_refs_dense.py)
  2. Cycle null: NULL_LAGS = [1,2,5,10] over the N refs → ~182 (N=50) or ~720 (N=182) pairs
  3. k-NN pool: DINOv2 features for all N refs
  4. k-NN LOO null: for each ref_i, score it as pseudo-query against top-k=15 of {refs - i}
  5. Routing decision: CV(null_ivar) < 0.50 → use peak signal, else ivar
  6. Compute H_train for real training mp4 (cycle + knn + fused)

ONLINE per query video Q:
  1. Sample 10 frames, SAM3 segment
  2. Cycle: 9 H_pair → p80 → H_cycle
  3. k-NN: 10 H_frame → p80 → H_knn
  4. Fuse: H_fused = cauchy_combine_video(H_cycle, H_knn)
  5. Output cycle_peak, knn_peak, fused_peak columns (+ ivar/peak diagnostics)

Flags:
  --use_knn   (default True)  enable k-NN signal; requires DINOv2 download
  --no_knn                    cycle-only regression mode (matches pre-fusion main)
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
REF_ROOT = BENCH / "reference"
RAW_VIDEO_ROOT = BENCH / "raw_videos" / "gr1"
GEN_ROOT = BENCH / "generated"
OUT = BENCH / "per_task_dense_eval"
DINO_CACHE_DIR = BENCH / "ref_cache"

EVAL_TASKS = [
    "1_Use the right hand to pick up green bok choy from tan table right side to bottom level of wire basket.",
    "2_Use the right hand to pick up rubik's cube from top level of the shelf to bottom level of the shelf.",
    "3_Use the right hand to pick up banana from teal plate to wooden table.",
    "4_Use the left hand to pick up dragonfruit from pink plate to teal plate.",
    "6_Use the right hand to pick up orange from middle of table to bottom white shelf.",
]

NULL_LAGS = [1, 2, 5, 10]
N_TEST_FRAMES = 10
KNN_K = 15


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


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


def cauchy_combine(ps: list[float]) -> float:
    ps = [p for p in ps if p is not None and 0 < p < 1]
    if not ps:
        return 0.5
    t = float(np.mean([np.tan(np.pi * (0.5 - p)) for p in ps]))
    return float(0.5 - np.arctan(t) / np.pi)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use_knn", action=argparse.BooleanOptionalAction, default=True,
                    help="Enable k-NN signal + Cauchy fusion (default True)")
    ap.add_argument("--out_suffix", default="",
                    help="Append suffix to CSV filename (e.g. '_cycle_only')")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    from warp_score.temporal_signals import (
        CycleSignal, PrecisionAnomalySignal, empirical_p_value,
    )
    from warp_score.matcher import RoMaMatcher
    from warp_score.sam_segmenter import VideoFrameSegmenter
    from warp_score.fusion import cauchy_combine_video

    print(f"Loading models  (use_knn={args.use_knn}) …")
    matcher = RoMaMatcher(setting="turbo", device="cuda", use_precision=True, vis_size=224)
    matcher._load_model()
    seg = VideoFrameSegmenter()

    signals = [
        CycleSignal(cert_floor=0.1),
        PrecisionAnomalySignal(cert_floor=0.1),
    ]
    sig_names = [s.name for s in signals]
    print(f"Cycle signals: {sig_names}")

    knn = None
    if args.use_knn:
        from warp_score.knn_signal import KNNFrameSignal, fg_mask_at_size
        knn = KNNFrameSignal(matcher, k=KNN_K)
        print(f"k-NN signal: k={KNN_K}, DINOv2 ViT-S/14\n")

    # ── Pair scorer (cycle/precision_anomaly) ────────────────────────────
    def score_pair_cycle(p_a: Path, p_b: Path) -> dict:
        fwd = matcher.match(p_a, p_b)
        bwd = matcher.match(p_b, p_a)
        return {s.name: s.compute(fwd, bwd) for s in signals}

    # ── Per-video cycle pair scores ──────────────────────────────────────
    def video_pair_scores(seg_pngs: list[Path]) -> list[dict]:
        out = []
        for t in range(len(seg_pngs) - 1):
            sigs = score_pair_cycle(seg_pngs[t], seg_pngs[t + 1])
            out.append({n: {"mean": sigs[n].mean, "peak": sigs[n].peak} for n in sig_names})
        return out

    # ── Per-video k-NN frame scores ──────────────────────────────────────
    def video_knn_scores(seg_pngs: list[Path], pool: dict, null_knn: dict) -> list[dict]:
        out = []
        for p in seg_pngs:
            from warp_score.knn_signal import fg_mask_at_size
            fg = fg_mask_at_size(p, matcher.vis_size)
            res = knn.score_frame(p, pool, null_knn, query_fg_mask=fg)
            out.append(res)
        return out

    # ── Score 1 video end-to-end ─────────────────────────────────────────
    def score_video(mp4: Path, task: str, vtype: str, label: int,
                    sorted_null_cycle: dict, pool: dict, null_knn: dict) -> dict:
        t0 = time.time()
        bgrs = sample_frames(mp4, N_TEST_FRAMES)
        if len(bgrs) < 2:
            return None
        seg_bgrs = [seg.segment_frame(b) for b in bgrs]
        tmp_dir = Path(tempfile.mkdtemp(prefix="ptdense_"))
        seg_pngs = []
        for i, b in enumerate(seg_bgrs):
            p = tmp_dir / f"f_{i:04d}.png"
            cv2.imwrite(str(p), b)
            seg_pngs.append(p)

        row = {
            "task": task.split("_")[0], "video": mp4.name, "type": vtype, "label": label,
            "time_sec": 0.0,
            "null_n": int(len(sorted_null_cycle["cycle"]["mean"])),
        }

        # ── Cycle pair signals ─────────────────────────────────────────
        pairs = video_pair_scores(seg_pngs)
        for n in sig_names:
            h_pairs = []
            for pp in pairs:
                p_m = empirical_p_value(pp[n]["mean"], sorted_null_cycle[n]["mean"])
                p_p = empirical_p_value(pp[n]["peak"], sorted_null_cycle[n]["peak"])
                h_pairs.append(1.0 - cauchy_combine([p_m, p_p]))
            if h_pairs:
                row[f"{n}_peak"] = float(np.percentile(h_pairs, 80))
                row[f"{n}_robust"] = float(np.sort(h_pairs)[len(h_pairs)//10:-max(1, len(h_pairs)//10)].mean()
                                            if len(h_pairs) > 2 else np.mean(h_pairs))
            else:
                row[f"{n}_peak"] = 0.0
                row[f"{n}_robust"] = 0.0

        # ── k-NN frame signals ──────────────────────────────────────────
        if args.use_knn and pool is not None and null_knn is not None:
            knn_frames = video_knn_scores(seg_pngs, pool, null_knn)
            h_knns = [r["H"] for r in knn_frames]
            row["knn_peak"] = float(np.percentile(h_knns, 80)) if h_knns else 0.0
            row["knn_robust"] = float(np.sort(h_knns)[len(h_knns)//10:-max(1, len(h_knns)//10)].mean()
                                       if len(h_knns) > 2 else (np.mean(h_knns) if h_knns else 0.0))
            row["knn_mean_ivar"] = float(np.mean([r["ivar"] for r in knn_frames]))
            row["knn_mean_peak_raw"] = float(np.mean([r["peak"] for r in knn_frames]))
            row["knn_route"] = null_knn["route"]
            # Fusion
            row["fused_peak"] = cauchy_combine_video(row["cycle_peak"], row["knn_peak"])
            row["fused_robust"] = cauchy_combine_video(row["cycle_robust"], row["knn_robust"])

        # Cleanup
        for p in seg_pngs:
            p.unlink(missing_ok=True)
        tmp_dir.rmdir()
        row["time_sec"] = time.time() - t0
        return row

    # ─────────────────────────────────────────────────────────────────────
    # Per-task loop
    # ─────────────────────────────────────────────────────────────────────
    all_rows: list[dict] = []
    null_per_task = OUT.parent / "null_per_task"
    null_per_task.mkdir(exist_ok=True)

    for task in EVAL_TASKS:
        task_short = task.split("_")[0]
        print(f"\n{'=' * 70}")
        print(f"Task {task_short}: {task[:60]}")
        print('=' * 70)

        ref_dir = REF_ROOT / task
        if not ref_dir.exists():
            print(f"  ref_dir missing")
            continue
        pngs = sorted(ref_dir.glob("frame_*.png"))
        if len(pngs) < 30:
            print(f"  not enough refs ({len(pngs)})")
            continue
        print(f"  refs: {len(pngs)}")

        # ── Cycle null (multi-lag) ────────────────────────────────────
        print("  Building cycle null …")
        null = {n: {"mean": [], "peak": []} for n in sig_names}
        t0 = time.time()
        for lag in NULL_LAGS:
            n_pairs = max(0, len(pngs) - lag)
            for i in range(n_pairs):
                sigs = score_pair_cycle(pngs[i], pngs[i + lag])
                for n in sig_names:
                    null[n]["mean"].append(sigs[n].mean)
                    null[n]["peak"].append(sigs[n].peak)
            print(f"    lag {lag:>2}: +{n_pairs} pairs  total {len(null[sig_names[0]]['mean'])}  ({time.time()-t0:.0f}s)")

        sorted_null = {n: {"mean": np.sort(np.asarray(null[n]["mean"], dtype=np.float32)),
                           "peak": np.sort(np.asarray(null[n]["peak"], dtype=np.float32))}
                       for n in sig_names}

        # ── k-NN pool + LOO null ──────────────────────────────────────
        pool, null_knn = None, None
        if args.use_knn:
            print("  Building k-NN pool (DINOv2) …")
            pool = knn.build_pool(task, pngs, DINO_CACHE_DIR)
            print(f"    pool: {len(pool['paths'])} frames × {pool['feats'].shape[1]} dim")
            print(f"  Calibrating k-NN LOO null ({len(pngs)} samples) …")
            null_knn = knn.calibrate_loo(pool)
            print(f"    null_ivar: mean={null_knn['null_ivar'].mean():.3f}, "
                  f"std={null_knn['null_ivar'].std():.3f}, CV={null_knn['cv']:.3f}")
            print(f"    null_peak: mean={null_knn['null_peak'].mean():.3f}, "
                  f"std={null_knn['null_peak'].std():.3f}")
            print(f"    route: {null_knn['route']}  (CV>={knn.cv_threshold:.2f} → ivar, else peak)")

        # ── Save combined null (cycle + knn) for this task ────────────
        save_data = {
            "null_lags": NULL_LAGS,
            "n_ref_frames": len(pngs),
        }
        for n in sig_names:
            save_data[f"cycle_{n}_mean"] = sorted_null[n]["mean"]
            save_data[f"cycle_{n}_peak"] = sorted_null[n]["peak"]
        if null_knn is not None:
            save_data["knn_null_ivar"] = null_knn["null_ivar"]
            save_data["knn_null_peak"] = null_knn["null_peak"]
            save_data["knn_route"] = np.str_(null_knn["route"])
            save_data["knn_cv"] = null_knn["cv"]
        np.savez(null_per_task / f"task_{task_short}.npz", **save_data)

        # ── Score real + gens ─────────────────────────────────────────
        real_mp4 = RAW_VIDEO_ROOT / f"{task_short}.mp4"
        if real_mp4.exists():
            r = score_video(real_mp4, task, "REAL", 0, sorted_null, pool, null_knn)
            if r is not None:
                all_rows.append(r)
                _print_row(r, args.use_knn)

        for mp4 in sorted((GEN_ROOT / task).glob("v*.mp4")):
            r = score_video(mp4, task, "GEN", 1, sorted_null, pool, null_knn)
            if r is not None:
                all_rows.append(r)
                _print_row(r, args.use_knn)

    # ── Save table ───────────────────────────────────────────────────────
    csv_path = OUT / f"per_task_dense_table{args.out_suffix}.csv"
    if all_rows:
        # union of all keys (some rows may lack knn cols if use_knn=False)
        fieldnames = []
        for r in all_rows:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nTable → {csv_path}")

    # ── Training-safety report ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PRIMARY METRIC: REAL TRAINING SAFETY")
    print("=" * 70)
    reals = [r for r in all_rows if r["type"] == "REAL"]
    cols = ["cycle_peak", "precision_anomaly_peak"]
    if args.use_knn:
        cols += ["knn_peak", "fused_peak"]
    header = f"{'Task':<8} " + " ".join(f"{c:>14}" for c in cols)
    print(header)
    for r in reals:
        line = f"{r['task']:<8} " + " ".join(f"{r.get(c, 0):>14.4f}" for c in cols)
        print(line)

    if reals:
        print()
        for c in cols:
            mx = max(r.get(c, 0) for r in reals)
            print(f"  Real max({c}) = {mx:.4f}")

    # ── AUROC + gen catch (informational only) ─────────────────────────
    try:
        from sklearn.metrics import roc_auc_score
        print("\n" + "=" * 70)
        print("INFORMATIONAL: gen catch + AUROC (gen labels are noisy proxies)")
        print("=" * 70)
        for c in cols:
            T = max(r.get(c, 0) for r in reals)
            gens = [r for r in all_rows if r["type"] == "GEN"]
            gen_above = sum(1 for r in gens if r.get(c, 0) > T)
            try:
                y = np.array([r["label"] for r in all_rows])
                s = np.array([r.get(c, 0) for r in all_rows])
                auroc = float(roc_auc_score(y, s))
            except Exception:
                auroc = float("nan")
            print(f"  {c:<25} threshold={T:.4f}  catch={gen_above}/{len(gens)}  AUROC={auroc:.4f}")
    except ImportError:
        pass


def _print_row(r: dict, use_knn: bool) -> None:
    base = f"  [{r['type']:<4}] {r['video']:<10} cycle={r.get('cycle_peak', 0):.4f}"
    if use_knn:
        base += f"  knn={r.get('knn_peak', 0):.4f}  fused={r.get('fused_peak', 0):.4f}"
    print(base + f"  ({r['time_sec']:.0f}s)")


if __name__ == "__main__":
    main()
