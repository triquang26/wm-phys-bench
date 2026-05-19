"""Timing benchmark: process frames sampled from a video through the full detector pipeline.

Prints per-stage latency for each frame, then summary stats.

Usage:
    python scripts/timing_single_video.py \
        --video "data/cosmos_synthetic_data/query/low/0_Open the box.mp4" \
        --fps 2          # sample 2 frames per second from the video
        --config warp_score/configs/test_v11_bidi.yaml
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


# ── helpers ───────────────────────────────────────────────────────────────────

def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0

def _bar(ms: float, scale: float = 0.05) -> str:
    n = max(1, int(ms * scale))
    return "█" * min(n, 60)

def _fmt_row(label: str, ms: float, pct: float) -> str:
    return f"    {label:<22s} {ms:7.1f} ms  {pct:5.1f}%  {_bar(ms)}"


# ── extract frames from video ─────────────────────────────────────────────────

def extract_frames(video_path: Path, target_fps: float) -> list[tuple[int, float, np.ndarray]]:
    """Return list of (frame_idx, timestamp_s, bgr_frame) sampled at target_fps."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")

    src_fps  = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step     = max(1, round(src_fps / target_fps))

    print(f"\n{'='*70}")
    print(f"  Video : {video_path.name}")
    print(f"  Source: {src_fps:.1f} fps  |  {n_frames} frames  |  {n_frames/src_fps:.2f}s")
    print(f"  Sample: every {step} frames  →  target {target_fps} fps")
    print(f"{'='*70}")

    frames = []
    idx = 0
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        if idx % step == 0:
            ts = idx / src_fps
            frames.append((idx, ts, bgr.copy()))
        idx += 1
    cap.release()
    print(f"  Extracted {len(frames)} frames for processing\n")
    return frames


# ── main pipeline ─────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    t_startup = time.perf_counter()

    # ── Load config + calibration ──────────────────────────────────────────
    print("[1/4] Loading config & calibration...", end=" ", flush=True)
    from warp_score.config import WarpScoreConfig
    from warp_score.calibrator import CalibrationArtifact
    from warp_score.matcher import RoMaMatcher
    from warp_score.mask import ForegroundMask, InteriorMask
    from warp_score.signals import build_signals
    from warp_score.fuser import build_fuser
    from warp_score.statistics import CertWeightedStatistics, MahalanobisStatistics
    from warp_score.signals import per_pixel_p_value
    from warp_score.adaptive_refs import AdaptiveRefSelector, DinoFeatureExtractor

    cfg    = WarpScoreConfig.from_yaml(args.config) if args.config else WarpScoreConfig()
    calib  = CalibrationArtifact.load(cfg.calib_path)
    t_cfg  = _ms(t_startup)
    print(f"done  ({t_cfg:.0f} ms)")

    # ── Init matcher ───────────────────────────────────────────────────────
    print("[2/4] Initialising RoMa matcher...", end=" ", flush=True)
    t0 = time.perf_counter()
    matcher = RoMaMatcher(
        setting=cfg.setting, device=cfg.device,
        use_precision=cfg.use_precision,
        bidirectional=getattr(cfg, "bidirectional", False),
        vis_size=cfg.vis_size,
    )
    t_matcher = _ms(t0)
    print(f"done  ({t_matcher:.0f} ms)  bidi={getattr(cfg, 'bidirectional', False)}")

    # ── Init DINO + load ref features from cache ───────────────────────────
    print("[3/4] Loading DINO + ref cache...", end=" ", flush=True)
    t0 = time.perf_counter()
    dino_model = getattr(cfg, "dino_model", "dinov2_vits14")
    selector   = AdaptiveRefSelector(DinoFeatureExtractor(dino_model))
    dino_cache_dir = getattr(cfg, "dino_cache_dir", None) or (cfg.artifacts_dir / "dino_cache")

    # infer task from video name
    video_path = Path(args.video).expanduser().resolve()
    task_name  = video_path.stem          # e.g. "0_Open the box"
    refs       = sorted((cfg.reference_dir / task_name).glob("*.png"))
    if not refs:
        sys.exit(f"No refs found under {cfg.reference_dir / task_name}")

    ref_feats = selector.load_cache(task_name, refs, dino_cache_dir)
    if ref_feats is None:
        print("\n  (cache miss — building...)", end=" ", flush=True)
        ref_feats = selector.build_cache(task_name, refs, dino_cache_dir)
    t_dino_init = _ms(t0)
    print(f"done  ({t_dino_init:.0f} ms)  {len(refs)} refs  cache={dino_cache_dir.name}")

    # Calibration for this task
    task_calib = calib.tasks.get(task_name, calib.global_)
    interior   = InteriorMask(cfg.erosion_k)
    k          = getattr(cfg, "k_per_frame", 15)

    print(f"[4/4] task='{task_name}'  k={k}  vis_size={cfg.vis_size}")

    # ── Extract frames ─────────────────────────────────────────────────────
    raw_frames = extract_frames(video_path, target_fps=args.fps)

    # ── Per-frame pipeline ─────────────────────────────────────────────────
    header = (
        f"{'#':>3}  {'ts':>6}  "
        f"{'decode':>8}  {'mask':>8}  {'dino':>8}  "
        f"{'knn':>8}  {'roma':>10}  {'signal':>8}  "
        f"{'total':>8}  H_score"
    )
    print(header)
    print("-" * len(header))

    all_timings: list[dict] = []

    for seq_i, (frame_idx, ts, bgr_raw) in enumerate(raw_frames):
        T = {}
        t_frame_start = time.perf_counter()

        # Stage 1: decode + resize
        t0 = time.perf_counter()
        bgr = cv2.resize(bgr_raw, (cfg.vis_size, cfg.vis_size), interpolation=cv2.INTER_NEAREST)
        T["decode_ms"] = _ms(t0)

        # Stage 2: save temp PNG for the matchers that need a path
        tmp_png = Path(f"/tmp/_timing_frame_{seq_i:04d}.png")
        cv2.imwrite(str(tmp_png), bgr)

        # Stage 3: FG mask + interior mask
        t0 = time.perf_counter()
        fg_mask       = ForegroundMask.from_image(bgr)
        interior_mask = interior.apply(fg_mask)
        T["mask_ms"] = _ms(t0)

        if not interior_mask.any():
            print(f"  frame {frame_idx}: empty interior, skipping")
            continue

        # Stage 4: DINO embed query
        t0 = time.perf_counter()
        query_feat = selector.extractor.extract([tmp_png])[0]
        T["dino_ms"] = _ms(t0)

        # Stage 5: k-NN ref selection
        t0 = time.perf_counter()
        top_k_idx   = selector.select_for_query(query_feat, ref_feats, k)
        selected_refs = [refs[i] for i in top_k_idx]
        T["knn_ms"] = _ms(t0)

        # Stage 6: RoMa batch matching (1 forward pass for all k refs)
        t0 = time.perf_counter()
        batch_results = matcher.match_batch(tmp_png, selected_refs, fg_mask=fg_mask)
        warps      = [r.warp for r in batch_results]
        certs      = [r.cert for r in batch_results]
        precisions = [r.precision for r in batch_results if r.precision is not None]
        ok_refs    = selected_refs
        T["roma_ms"]      = _ms(t0)
        T["roma_per_ref"] = [T["roma_ms"] / len(selected_refs)] * len(selected_refs)  # amortized

        if not warps:
            print(f"  frame {frame_idx}: all refs failed")
            continue

        # Stage 7: signal computation
        t0 = time.perf_counter()
        warps_a = np.stack(warps)
        certs_a = np.stack(certs)
        var_map = CertWeightedStatistics.variance_per_pixel(warps_a, certs_a)
        raw = {
            "ivar": CertWeightedStatistics.interior_mean(var_map, interior_mask),
            "peak": CertWeightedStatistics.peak_max_z(var_map, interior_mask),
        }
        D_map = None
        if precisions and len(precisions) == len(warps):
            precisions_a = np.stack(precisions)
            D_map, logdetΛ_map, _ = MahalanobisStatistics.ivar_per_pixel(warps_a, precisions_a)
            raw["ivar_maha"] = MahalanobisStatistics.interior_mean(D_map, interior_mask)
            raw["peak_maha"] = MahalanobisStatistics.peak_max_z(D_map, interior_mask)
            raw["evidence"]  = MahalanobisStatistics.interior_mean(-logdetΛ_map, interior_mask)
            if task_calib.T_null is not None:
                p_px = per_pixel_p_value(D_map, task_calib.T_null, interior_mask)
                vals = (1.0 - p_px)[interior_mask]
                raw["ivar_px"] = float(vals.mean()) if vals.size > 0 else 0.5
        T["signal_ms"] = _ms(t0)

        # Stage 8: p-values + Cauchy fuse (fast, fold into signal timing)
        from warp_score.signals import build_signals
        from warp_score.fuser import build_fuser
        from warp_score.calibrator import TaskCalibration
        t0 = time.perf_counter()
        signals_list = build_signals(cfg.signal_names)
        fuser = build_fuser(cfg.fuser)
        p_per = {}
        for sig in signals_list:
            if sig.name in raw:
                p_per[sig.name] = sig.p_value(raw[sig.name], task_calib)
        p_combined = fuser.fuse(p_per) if p_per else 0.5
        H_score = 1.0 - p_combined
        T["fuse_ms"] = _ms(t0)

        T["total_ms"] = _ms(t_frame_start)
        all_timings.append(T)

        # Print one-liner
        print(
            f"{seq_i+1:>3}  {ts:>5.2f}s  "
            f"{T['decode_ms']:>7.1f}ms  {T['mask_ms']:>7.1f}ms  {T['dino_ms']:>7.1f}ms  "
            f"{T['knn_ms']:>7.1f}ms  {T['roma_ms']:>9.1f}ms  {T['signal_ms']:>7.1f}ms  "
            f"{T['total_ms']:>7.1f}ms  {H_score:.3f}"
        )

        # Per-ref RoMa times
        if args.verbose:
            valid = [t for t in T["roma_per_ref"] if t > 0]
            if valid:
                print(f"       RoMa per-ref: min={min(valid):.0f}ms  mean={np.mean(valid):.0f}ms  max={max(valid):.0f}ms  ({len(valid)} refs)")

    # ── Summary ────────────────────────────────────────────────────────────
    if not all_timings:
        print("No frames processed.")
        return

    print(f"\n{'='*70}")
    print(f"  SUMMARY  ({len(all_timings)} frames)")
    print(f"{'='*70}")

    for stage, label in [
        ("decode_ms",  "decode+resize"),
        ("mask_ms",    "FG mask"),
        ("dino_ms",    "DINO embed"),
        ("knn_ms",     "k-NN select"),
        ("roma_ms",    f"RoMa (k={k} refs)"),
        ("signal_ms",  "signal compute"),
        ("fuse_ms",    "p-val + fuse"),
        ("total_ms",   "TOTAL"),
    ]:
        vals = [t[stage] for t in all_timings]
        pct  = np.mean(vals) / np.mean([t["total_ms"] for t in all_timings]) * 100
        print(f"  {label:<20s}  mean={np.mean(vals):7.1f}ms  "
              f"min={min(vals):6.1f}  max={max(vals):6.1f}  p50={np.median(vals):6.1f}  {pct:5.1f}%")

    totals = [t["total_ms"] for t in all_timings]
    fps_achieved = 1000.0 / np.mean(totals)
    print(f"\n  Throughput: {fps_achieved:.2f} frames/s  ({np.mean(totals):.0f} ms/frame avg)")
    print(f"  Startup (config+model+cache): {t_cfg + t_matcher + t_dino_init:.0f} ms (one-time)")

    # Per-ref breakdown summary
    all_ref_times = []
    for t in all_timings:
        all_ref_times.extend([x for x in t.get("roma_per_ref", []) if x > 0])
    if all_ref_times:
        print(f"\n  RoMa per single ref match:")
        print(f"    mean={np.mean(all_ref_times):.0f}ms  min={min(all_ref_times):.0f}ms  "
              f"max={max(all_ref_times):.0f}ms  (n={len(all_ref_times)} ref-frame pairs)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video",   required=True,  help="Path to input MP4")
    p.add_argument("--config",  default="warp_score/configs/test_v11_bidi.yaml")
    p.add_argument("--fps",     type=float, default=2.0, help="Sample rate (frames/sec from video)")
    p.add_argument("--verbose", action="store_true", help="Print per-ref RoMa times")
    run(p.parse_args())
