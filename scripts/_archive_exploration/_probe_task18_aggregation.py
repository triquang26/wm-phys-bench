"""Probe task 18 (red glass) — sweep percentile cutoff + frame count.

Outputs a table: how H_cycle, H_knn, H_fused, ratio_fused change when
we use p80/90/95/99/100 aggregation AND 10/20/30 sampled frames.

Reuses existing refs (50 PNGs already extracted in paper-doanh-eval/reference/).
Does NOT re-run offline null — re-uses it by rebuilding from same refs.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

REPO = Path("/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination")
BENCH = REPO / "paper-doanh-eval"
TASK_FOLDER = "8_15_Use the left hand to pick up red glass from light red plate to turquoise plate."
HIGH_MP4 = BENCH / "raw_videos" / "high" / f"{TASK_FOLDER}.mp4"
LOW_MP4  = BENCH / "raw_videos" / "low"  / f"{TASK_FOLDER}.mp4"
REF_DIR  = BENCH / "reference" / TASK_FOLDER

NULL_LAGS = [1, 2, 5, 10]
KNN_K = 15

PERCENTILES = [80, 90, 95, 99, 100]
FRAME_COUNTS = [10, 20, 30]


def sample_frames(mp4: Path, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(mp4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, min(n, total), dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        if ok:
            frames.append(bgr)
    cap.release()
    return frames


def main():
    print("Loading models …")
    from warp_score.matcher import RoMaMatcher
    from warp_score.sam_segmenter import VideoFrameSegmenter
    from warp_score.knn_signal import KNNFrameSignal, fg_mask_at_size
    from warp_score.temporal_signals import CycleSignal, empirical_p_value, cycle_error_map
    from warp_score.statistics import MahalanobisStatistics
    from warp_score.fusion import (cauchy_combine, cauchy_combine_video,
                                   bootstrap_baseline_sigma, sigmoid_normalize_ratio)

    matcher = RoMaMatcher(setting="turbo", device="cuda", use_precision=True, vis_size=224)
    matcher._load_model()
    seg = VideoFrameSegmenter()
    knn = KNNFrameSignal(matcher, k=KNN_K)
    cycle_signal = CycleSignal(cert_floor=0.1)

    pngs = sorted(REF_DIR.glob("frame_*.png"))
    print(f"Refs: {len(pngs)}")

    # ── Build offline null (once, shared across all experiments) ───────────────
    print("\n── Building cycle null …")
    t0 = time.time()
    null_mean_list, null_peak_list = [], []
    for lag in NULL_LAGS:
        for i in range(len(pngs) - lag):
            fwd = matcher.match(pngs[i], pngs[i + lag])
            bwd = matcher.match(pngs[i + lag], pngs[i])
            s = cycle_signal.compute(fwd, bwd)
            null_mean_list.append(s.mean)
            null_peak_list.append(s.peak)
    null_cycle_mean = np.sort(np.asarray(null_mean_list, dtype=np.float32))
    null_cycle_peak = np.sort(np.asarray(null_peak_list, dtype=np.float32))
    print(f"  cycle null: {len(null_mean_list)} pairs in {time.time()-t0:.0f}s")

    print("\n── Building kNN LOO null …")
    t0 = time.time()
    pool = knn.build_pool(TASK_FOLDER, pngs, BENCH / "ref_cache")
    null_knn = knn.calibrate_loo(pool, verbose=False)
    print(f"  kNN null: {len(null_knn['null_ivar'])} LOO samples in {time.time()-t0:.0f}s, route={null_knn['route']}")

    # ── H_train (high video) ────────────────────────────────────────────────────
    print("\n── Scoring HIGH video (training ref) …")
    h_train = _score_video_raw(HIGH_MP4, pngs, pool, null_cycle_mean, null_cycle_peak,
                               null_knn, matcher, seg, knn, cycle_signal,
                               empirical_p_value, cauchy_combine,
                               MahalanobisStatistics, fg_mask_at_size,
                               n_frames=10)
    h_train_knn_raw = h_train["h_frames"]
    h_train_cycle_raw = h_train["h_pairs"]
    print(f"  h_pairs (high): {[f'{v:.3f}' for v in h_train_cycle_raw]}")
    print(f"  h_frames (high): {[f'{v:.3f}' for v in h_train_knn_raw]}")

    # ── Score LOW video at different frame counts ────────────────────────────────
    low_raw: dict[int, dict] = {}
    for n_frames in FRAME_COUNTS:
        print(f"\n── Scoring LOW video — {n_frames} frames …")
        t0 = time.time()
        low_raw[n_frames] = _score_video_raw(
            LOW_MP4, pngs, pool, null_cycle_mean, null_cycle_peak,
            null_knn, matcher, seg, knn, cycle_signal,
            empirical_p_value, cauchy_combine,
            MahalanobisStatistics, fg_mask_at_size,
            n_frames=n_frames)
        print(f"  done in {time.time()-t0:.0f}s")
        print(f"  h_pairs  ({n_frames}f): {[f'{v:.3f}' for v in low_raw[n_frames]['h_pairs']]}")
        print(f"  h_frames ({n_frames}f): {[f'{v:.3f}' for v in low_raw[n_frames]['h_frames']]}")

    # ── Sweep percentiles ────────────────────────────────────────────────────────
    print("\n\n" + "="*72)
    print("RESULTS — how ratio_fused changes with percentile and frame count")
    print("="*72)
    header = f"{'pct':>5} | {'n_frames':>8} | {'H_train_c':>9} {'H_test_c':>9} {'r_cycle':>7} | {'H_train_k':>9} {'H_test_k':>9} {'r_knn':>7} | {'r_fused':>9} | verdict"
    print(header)
    print("-"*len(header))

    rows = []
    for n_frames in FRAME_COUNTS:
        raw = low_raw[n_frames]
        # For H_train we always use 10 frames (same as benchmark)
        for pct in PERCENTILES:
            ht_c = float(np.percentile(h_train_cycle_raw, pct))
            ht_k = float(np.percentile(h_train_knn_raw, pct))
            ht_f = cauchy_combine_video(ht_c, ht_k)

            ho_c = float(np.percentile(raw["h_pairs"], pct))
            ho_k = float(np.percentile(raw["h_frames"], pct))
            ho_f = cauchy_combine_video(ho_c, ho_k)

            r_c = ho_c / max(ht_c, 1e-8)
            r_k = ho_k / max(ht_k, 1e-8)
            r_f = ho_f / max(ht_f, 1e-8)
            v = "🔴 HALLU" if r_f > 1.0 else ("⚠ border" if r_f > 0.95 else "✓ clean")
            rows.append({
                "pct": pct, "n_frames": n_frames,
                "ht_c": ht_c, "ho_c": ho_c, "r_cycle": r_c,
                "ht_k": ht_k, "ho_k": ho_k, "r_knn": r_k,
                "r_fused": r_f, "verdict": v,
            })
            print(f"  p{pct:2d} | {n_frames:8d} | {ht_c:9.3f} {ho_c:9.3f} {r_c:7.3f} | "
                  f"{ht_k:9.3f} {ho_k:9.3f} {r_k:7.3f} | {r_f:9.3f} | {v}")

    # ── Sigmoid normalization via bootstrap baseline σ ─────────────────────────
    print("\n\n" + "="*72)
    print("SIGMOID NORMALIZATION — bootstrap H_train sigma → α = 1/σ")
    print("="*72)
    sigma, dist = bootstrap_baseline_sigma(h_train_cycle_raw, h_train_knn_raw,
                                            n_boot=200, pct=80, seed=42)
    alpha = 1.0 / max(sigma, 1e-6)
    print(f"  Bootstrap H_train_fused (n=200, p80): σ = {sigma:.4f}")
    print(f"  α = 1/σ = {alpha:.2f}")
    print(f"  σ-spread examples:")
    print(f"    ratio = 1.00         → score = 0.500 (at baseline)")
    print(f"    ratio = 1 + 1σ_norm  → score = {sigmoid_normalize_ratio(1.0 + sigma, sigma):.3f}")
    print(f"    ratio = 1 + 2σ_norm  → score = {sigmoid_normalize_ratio(1.0 + 2*sigma, sigma):.3f}")
    print(f"    ratio = 1 - 1σ_norm  → score = {sigmoid_normalize_ratio(1.0 - sigma, sigma):.3f}")

    print("\n  Applied to task-18 ratios:")
    print(f"  {'pct':>5} | {'n_frames':>8} | {'r_fused':>9} | {'score_norm':>10} | interpretation")
    print("  " + "-"*72)
    for row in rows:
        s = sigmoid_normalize_ratio(row["r_fused"], sigma)
        interp = ("strongly anomalous" if s > 0.7 else
                  "weakly anomalous"   if s > 0.55 else
                  "ambiguous"           if s > 0.45 else
                  "weakly clean"        if s > 0.3  else
                  "strongly clean")
        print(f"  p{row['pct']:2d} | {row['n_frames']:8d} | {row['r_fused']:9.3f} | "
              f"{s:10.3f} | {interp}")
        row["score_norm"] = s
    print(f"\n  → Note: σ = {sigma:.4f}. The sigmoid is fairly STEEP because")
    print(f"    aggregation variance is small (only 9 pairs + 10 frames). ratio=0.81 → score≈0")

    # Save results
    out = BASE / "outputs" / "probe_task18_aggregation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\n→ {out}")


def _score_video_raw(mp4, pngs, pool, null_cycle_mean, null_cycle_peak,
                     null_knn, matcher, seg, knn, cycle_signal,
                     empirical_p_value, cauchy_combine,
                     Mahal, fg_mask_at_size, n_frames: int) -> dict:
    """Returns raw per-pair / per-frame H lists (no percentile applied yet)."""
    from warp_score.fusion import cauchy_combine

    bgrs = sample_frames(mp4, n_frames)
    tmp = Path(tempfile.mkdtemp())
    seg_pngs = []
    for i, b in enumerate(bgrs):
        p = tmp / f"q_{i:04d}.png"
        cv2.imwrite(str(p), seg.segment_frame(b))
        seg_pngs.append(p)

    # Cycle pairs (n_frames - 1 consecutive)
    h_pairs = []
    for t in range(len(seg_pngs) - 1):
        fwd = matcher.match(seg_pngs[t], seg_pngs[t + 1])
        bwd = matcher.match(seg_pngs[t + 1], seg_pngs[t])
        s = cycle_signal.compute(fwd, bwd)
        p_m = empirical_p_value(s.mean, null_cycle_mean)
        p_p = empirical_p_value(s.peak, null_cycle_peak)
        h_pairs.append(1.0 - cauchy_combine([p_m, p_p]))

    # kNN frames
    h_frames = []
    for p in seg_pngs:
        fg = fg_mask_at_size(p, matcher.vis_size)
        res = knn.score_frame(p, pool, null_knn, query_fg_mask=fg)
        h_frames.append(res["H"])

    for p in seg_pngs:
        try: p.unlink()
        except: pass
    try: tmp.rmdir()
    except: pass

    return {"h_pairs": h_pairs, "h_frames": h_frames}


if __name__ == "__main__":
    main()
