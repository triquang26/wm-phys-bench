#!/usr/bin/env python3
"""End-to-end benchmark + visualization for 1 task × 1 query video.

Measures EVERY step's wall-clock time precisely and produces informative
visualizations of both branches (cycle + kNN) so reviewers can SEE what
the method is doing.

Default: GR-1 task 1 (green bok choy), pick the first generated v0000.mp4
as query. Override via --task / --query.

Outputs (under outputs/benchmark_demo/<task_short>/):
  timing.json                            — raw measurements per step
  timing_summary.md                      — human-readable summary
  offline/
    01_sampled_refs.png                  — grid of segmented refs
    02_cycle_pair_example.png            — A | B | RoMa fwd flow | RoMa bwd flow | cycle error
    03_cycle_null_histograms.png         — null distributions for mean+peak
    04_dinov2_pool_pca.png               — pool features projected to 2D
    05_knn_loo_example.png               — 1 LOO query + its top-k retrieved refs
    06_cochran_dmap_example.png          — D-map for 1 LOO query
    07_knn_null_histograms.png           — null_ivar + null_peak distributions
  online/
    01_query_frames.png                  — 10 sampled query frames
    02_query_pair_example.png            — 1 consecutive query pair + cycle viz
    03_cycle_pairs_bar.png               — H_pair × 9 + H_cycle (p80 line)
    04_knn_topk_per_frame.png            — for 3 query frames: top-5 nearest pool refs
    05_knn_dmap_per_frame.png            — D-map per query frame (10 panels)
    06_knn_frames_bar.png                — H_frame × 10 + H_knn (p80 line)
    07_final_scores.png                  — H_cycle / H_knn / H_fused / H_train / ratio
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

REPO_ROOT = Path("/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination")
BENCH_GR1 = REPO_ROOT / "paper-physical-gr1"

DEFAULT_TASK = "1_Use the right hand to pick up green bok choy from tan table right side to bottom level of wire basket."
DEFAULT_TASK_SHORT = "1"

NULL_LAGS = [1, 2, 5, 10]
N_QUERY_FRAMES = 10
KNN_K = 15


# ─────────────────────────────────────────────────────────────────────────────
# Timing utility
# ─────────────────────────────────────────────────────────────────────────────


class StepTimer:
    """Records named step timings. Use as `with timer.step('name'):`."""

    def __init__(self):
        self.timings: dict[str, float] = {}

    @contextmanager
    def step(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.timings[name] = self.timings.get(name, 0.0) + dt

    def summary(self) -> str:
        lines = [f"  {n:<48s} {t*1000:>10.1f} ms"
                 if t < 1.0 else
                 f"  {n:<48s} {t:>10.2f} s"
                 for n, t in self.timings.items()]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


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


def fg_mask_from_seg(bgr: np.ndarray) -> np.ndarray:
    return ~np.all(bgr == np.array([127, 127, 127])[None, None, :], axis=-1)


def warp_to_flow_rgb(warp_norm: np.ndarray) -> np.ndarray:
    """Convert (H,W,2) normalized warp to a colorful RGB flow visualization."""
    H, W = warp_norm.shape[:2]
    # Convert normalized [-1,1] to displacement in pixels relative to identity grid
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    target_x = (warp_norm[..., 0] + 1) * (W - 1) / 2
    target_y = (warp_norm[..., 1] + 1) * (H - 1) / 2
    dx = target_x - xx
    dy = target_y - yy
    mag = np.sqrt(dx * dx + dy * dy)
    ang = (np.arctan2(dy, dx) + np.pi) / (2 * np.pi)  # 0..1
    hsv = np.zeros((H, W, 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 180).astype(np.uint8)
    hsv[..., 1] = 255
    mag_norm = np.clip(mag / max(1.0, np.percentile(mag, 95)), 0, 1)
    hsv[..., 2] = (mag_norm * 255).astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def heatmap_overlay(bgr: np.ndarray, heat: np.ndarray, alpha: float = 0.55,
                    vmax: float | None = None) -> np.ndarray:
    """Overlay TURBO heatmap on bgr image."""
    if vmax is None:
        vmax = float(np.percentile(heat, 95)) + 1e-6
    h_norm = np.clip(heat / vmax, 0, 1)
    colored = cv2.applyColorMap((h_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    blend = (bgr * (1 - alpha) + colored * alpha).astype(np.uint8)
    return blend


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark
# ─────────────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=DEFAULT_TASK,
                    help="Full task folder name under reference/ and generated/")
    ap.add_argument("--query", default=None,
                    help="Path to query mp4 (default: first v*.mp4 in generated/<task>/)")
    ap.add_argument("--out_dir", default=None,
                    help="Output dir (default: outputs/benchmark_demo/<task_short>/)")
    args = ap.parse_args()

    task = args.task
    task_short = task.split("_")[0]
    out_dir = Path(args.out_dir) if args.out_dir else (
        BASE / "outputs" / "benchmark_demo" / task_short)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "offline").mkdir(exist_ok=True)
    (out_dir / "online").mkdir(exist_ok=True)

    ref_dir = BENCH_GR1 / "reference" / task
    real_mp4 = BENCH_GR1 / "raw_videos" / "gr1" / f"{task_short}.mp4"
    if args.query is None:
        args.query = sorted((BENCH_GR1 / "generated" / task).glob("v*.mp4"))[0]
    query_mp4 = Path(args.query)

    print(f"\n{'='*70}\nBenchmark — task {task_short}\n{'='*70}")
    print(f"Task:    {task}")
    print(f"Refs:    {ref_dir}")
    print(f"Real:    {real_mp4}")
    print(f"Query:   {query_mp4}")
    print(f"Out:     {out_dir}")

    timer = StepTimer()

    # ────────────────────── MODEL LOAD ──────────────────────
    with timer.step("00_model_load"):
        from warp_score.matcher import RoMaMatcher
        from warp_score.sam_segmenter import VideoFrameSegmenter
        from warp_score.knn_signal import KNNFrameSignal, fg_mask_at_size
        from warp_score.temporal_signals import (
            CycleSignal, empirical_p_value, cycle_error_map,
        )
        from warp_score.statistics import MahalanobisStatistics
        from warp_score.fusion import cauchy_combine, cauchy_combine_video

        matcher = RoMaMatcher(setting="turbo", device="cuda", use_precision=True, vis_size=224)
        matcher._load_model()
        seg = VideoFrameSegmenter()
        knn = KNNFrameSignal(matcher, k=KNN_K)
        cycle_signal = CycleSignal(cert_floor=0.1)

    print(f"\n[load] models in {timer.timings['00_model_load']:.1f}s\n")

    # ════════════════════════════════════════════════════════════════════
    # OFFLINE — build null
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}\nOFFLINE\n{'─'*70}")

    # ── O1: load refs ─────────────────────────────────────────────
    with timer.step("offline_01_load_refs"):
        pngs = sorted(ref_dir.glob("frame_*.png"))
    print(f"[O1] loaded {len(pngs)} ref PNGs")

    # ── O2: viz sampled refs ──────────────────────────────────────
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    sample_idx = np.linspace(0, len(pngs) - 1, 10, dtype=int)
    for ax, i in zip(axes.flat, sample_idx):
        ax.imshow(cv2.cvtColor(cv2.imread(str(pngs[i])), cv2.COLOR_BGR2RGB))
        ax.set_title(f"ref {i}", fontsize=10)
        ax.axis("off")
    fig.suptitle(f"Task {task_short} — 10 sampled refs (of {len(pngs)})", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "offline" / "01_sampled_refs.png", dpi=110)
    plt.close(fig)

    # ── O3: cycle null ────────────────────────────────────────────
    with timer.step("offline_02_cycle_null_build"):
        null_mean_list, null_peak_list = [], []
        n_pairs = 0
        for lag in NULL_LAGS:
            for i in range(len(pngs) - lag):
                fwd = matcher.match(pngs[i], pngs[i + lag])
                bwd = matcher.match(pngs[i + lag], pngs[i])
                s = cycle_signal.compute(fwd, bwd)
                null_mean_list.append(s.mean)
                null_peak_list.append(s.peak)
                n_pairs += 1
        null_cycle_mean = np.sort(np.asarray(null_mean_list, dtype=np.float32))
        null_cycle_peak = np.sort(np.asarray(null_peak_list, dtype=np.float32))
    print(f"[O2] cycle null built: {n_pairs} pairs in {timer.timings['offline_02_cycle_null_build']:.1f}s")

    # ── O4: visualize 1 cycle pair ────────────────────────────────
    pair_i, pair_j = 0, 1
    fwd_demo = matcher.match(pngs[pair_i], pngs[pair_j])
    bwd_demo = matcher.match(pngs[pair_j], pngs[pair_i])
    err_demo = cycle_error_map(fwd_demo.warp, bwd_demo.warp)
    img_a = cv2.imread(str(pngs[pair_i]))
    img_b = cv2.imread(str(pngs[pair_j]))
    img_a_r = cv2.resize(img_a, (matcher.vis_size, matcher.vis_size))
    img_b_r = cv2.resize(img_b, (matcher.vis_size, matcher.vis_size))

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    axes[0].imshow(cv2.cvtColor(img_a_r, cv2.COLOR_BGR2RGB)); axes[0].set_title("ref A"); axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(img_b_r, cv2.COLOR_BGR2RGB)); axes[1].set_title(f"ref B (lag={pair_j-pair_i})"); axes[1].axis("off")
    axes[2].imshow(warp_to_flow_rgb(fwd_demo.warp)); axes[2].set_title("RoMa warp A→B\n(hue=direction, value=magnitude)"); axes[2].axis("off")
    axes[3].imshow(warp_to_flow_rgb(bwd_demo.warp)); axes[3].set_title("RoMa warp B→A"); axes[3].axis("off")
    im = axes[4].imshow(err_demo, cmap="turbo", vmin=0, vmax=float(np.percentile(err_demo, 99)) + 1e-6)
    axes[4].set_title(f"cycle error map (px)\nmean={null_mean_list[0]:.2f}, p99={null_peak_list[0]:.2f}")
    axes[4].axis("off")
    plt.colorbar(im, ax=axes[4], fraction=0.046)
    fig.suptitle(f"Cycle branch — 1 example pair (lag {pair_j-pair_i})", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "offline" / "02_cycle_pair_example.png", dpi=110)
    plt.close(fig)

    # ── O5: cycle null histograms ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(null_cycle_mean, bins=40, color="steelblue", edgecolor="white")
    axes[0].axvline(np.median(null_cycle_mean), color="red", ls="--", label=f"median={np.median(null_cycle_mean):.2f}")
    axes[0].set_xlabel("mean cycle drift (px)"); axes[0].set_ylabel("count")
    axes[0].set_title(f"null_cycle_mean (n={len(null_cycle_mean)})")
    axes[0].legend()
    axes[1].hist(null_cycle_peak, bins=40, color="darkorange", edgecolor="white")
    axes[1].axvline(np.median(null_cycle_peak), color="red", ls="--", label=f"median={np.median(null_cycle_peak):.2f}")
    axes[1].set_xlabel("peak cycle drift (p99 of err map, px)")
    axes[1].set_title(f"null_cycle_peak (n={len(null_cycle_peak)})")
    axes[1].legend()
    fig.suptitle("Cycle null distributions (per-task)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "offline" / "03_cycle_null_histograms.png", dpi=110)
    plt.close(fig)

    # ── O6: DINOv2 pool ───────────────────────────────────────────
    with timer.step("offline_03_dinov2_pool_build"):
        pool = knn.build_pool(task, pngs, BENCH_GR1 / "ref_cache")
    feats = pool["feats"]
    print(f"[O3] DINOv2 pool: {feats.shape} in {timer.timings['offline_03_dinov2_pool_build']:.1f}s")

    # PCA viz of pool
    feats_centered = feats - feats.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(feats_centered, full_matrices=False)
    pool_2d = feats_centered @ Vt[:2].T  # (N, 2)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(pool_2d[:, 0], pool_2d[:, 1], c=np.arange(len(pool_2d)), cmap="viridis", s=40, edgecolors="white", linewidths=0.5)
    plt.colorbar(sc, ax=ax, label="ref frame index (time)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"DINOv2 pool (n={len(pool_2d)}, dim={feats.shape[1]}) — PCA 2D")
    fig.tight_layout()
    fig.savefig(out_dir / "offline" / "04_dinov2_pool_pca.png", dpi=110)
    plt.close(fig)

    # ── O7: kNN LOO null ──────────────────────────────────────────
    with timer.step("offline_04_knn_loo_null"):
        null_knn = knn.calibrate_loo(pool, verbose=False)
    print(f"[O4] kNN LOO null built ({len(null_knn['null_ivar'])} samples) "
          f"in {timer.timings['offline_04_knn_loo_null']:.1f}s, route={null_knn['route']}, CV={null_knn['cv']:.3f}")

    # ── O8: 1 LOO query example + Cochran D-map ───────────────────
    loo_q_idx = 50  # mid pool
    q_feat = feats[loo_q_idx]
    cand = [j for j in range(len(pngs)) if j != loo_q_idx]
    top_k_idx = knn.selector.select_for_query(q_feat, feats[cand], KNN_K)
    top_k_paths = [pngs[cand[j]] for j in top_k_idx]
    q_path = pngs[loo_q_idx]

    # Compute D-map for viz
    fg = fg_mask_at_size(q_path, matcher.vis_size)
    matches = matcher.match_batch(q_path, top_k_paths, fg_mask=fg)
    warps = np.stack([m.warp for m in matches], axis=0)
    precs = np.stack([m.precision for m in matches], axis=0)
    D_map, _, _ = MahalanobisStatistics.ivar_per_pixel(warps, precs)
    ivar_demo = MahalanobisStatistics.interior_mean(D_map, fg)
    peak_demo = MahalanobisStatistics.peak_max_z(D_map, fg)

    # Viz: query + top-5 nearest
    fig, axes = plt.subplots(1, 6, figsize=(24, 4.5))
    axes[0].imshow(cv2.cvtColor(cv2.resize(cv2.imread(str(q_path)), (matcher.vis_size, matcher.vis_size)), cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"LOO query\n(ref {loo_q_idx})", fontsize=10)
    axes[0].axis("off")
    for i, idx in enumerate(top_k_idx[:5]):
        axes[i+1].imshow(cv2.cvtColor(cv2.resize(cv2.imread(str(top_k_paths[i])), (matcher.vis_size, matcher.vis_size)), cv2.COLOR_BGR2RGB))
        axes[i+1].set_title(f"top-{i+1} nearest\n(ref {cand[idx]})", fontsize=10)
        axes[i+1].axis("off")
    fig.suptitle(f"kNN LOO — 1 query + top-{KNN_K} retrieval (showing top-5)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "offline" / "05_knn_loo_example.png", dpi=110)
    plt.close(fig)

    # Cochran D-map viz
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    q_bgr = cv2.cvtColor(cv2.resize(cv2.imread(str(q_path)), (matcher.vis_size, matcher.vis_size)), cv2.COLOR_BGR2RGB)
    axes[0].imshow(q_bgr); axes[0].set_title("Query (LOO)"); axes[0].axis("off")
    im1 = axes[1].imshow(D_map, cmap="turbo", vmin=0, vmax=float(np.percentile(D_map[fg], 99)) + 1e-6)
    axes[1].set_title(f"Cochran D-map\n(higher = refs disagree)"); axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    overlay = heatmap_overlay(cv2.cvtColor(q_bgr, cv2.COLOR_RGB2BGR), D_map)
    axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"D-map overlay on query\nivar={ivar_demo:.2f}, peak_z={peak_demo:.2f}")
    axes[2].axis("off")
    fig.suptitle(f"Cochran deviance — 1 example query against k={KNN_K} refs", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "offline" / "06_cochran_dmap_example.png", dpi=110)
    plt.close(fig)

    # ── O9: kNN null histograms ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(null_knn["null_ivar"], bins=40, color="steelblue", edgecolor="white")
    axes[0].axvline(np.median(null_knn["null_ivar"]), color="red", ls="--",
                    label=f"median={np.median(null_knn['null_ivar']):.3f}")
    axes[0].set_xlabel("ivar_maha (interior mean of D-map)")
    axes[0].set_title(f"null_knn_ivar (n={len(null_knn['null_ivar'])})  [ROUTED={null_knn['route']=='ivar'}]")
    axes[0].legend()
    axes[1].hist(null_knn["null_peak"], bins=40, color="darkorange", edgecolor="white")
    axes[1].axvline(np.median(null_knn["null_peak"]), color="red", ls="--",
                    label=f"median={np.median(null_knn['null_peak']):.3f}")
    axes[1].set_xlabel("peak_maha (z-score max of D-map)")
    axes[1].set_title(f"null_knn_peak  [ROUTED={null_knn['route']=='peak'}]")
    axes[1].legend()
    fig.suptitle(f"kNN null distributions  (route={null_knn['route']}, CV(null_ivar)={null_knn['cv']:.3f})", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "offline" / "07_knn_null_histograms.png", dpi=110)
    plt.close(fig)

    # ── O10: H_train baseline ─────────────────────────────────────
    with timer.step("offline_05_h_train_baseline"):
        h_train = score_video(real_mp4, pngs, pool, null_cycle_mean, null_cycle_peak,
                              null_knn, matcher, seg, knn, cycle_signal,
                              MahalanobisStatistics, empirical_p_value,
                              cauchy_combine, cauchy_combine_video,
                              fg_mask_at_size,
                              return_intermediates=False)
    print(f"[O5] H_train baseline computed in {timer.timings['offline_05_h_train_baseline']:.1f}s — "
          f"H_train: cycle={h_train['cycle_peak']:.3f}, knn={h_train['knn_peak']:.3f}, "
          f"fused={h_train['fused_peak']:.3f}")

    # ════════════════════════════════════════════════════════════════════
    # ONLINE — score 1 query video
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}\nONLINE — query: {query_mp4.name}\n{'─'*70}")

    with timer.step("online_total"):
        online_result = score_video(query_mp4, pngs, pool, null_cycle_mean, null_cycle_peak,
                                    null_knn, matcher, seg, knn, cycle_signal,
                                    MahalanobisStatistics, empirical_p_value,
                                    cauchy_combine, cauchy_combine_video,
                                    fg_mask_at_size,
                                    return_intermediates=True,
                                    step_timer=timer, step_prefix="online")
    online_result["h_train"] = h_train
    ratio_cycle = online_result["cycle_peak"] / max(h_train["cycle_peak"], 1e-8)
    ratio_knn = online_result["knn_peak"] / max(h_train["knn_peak"], 1e-8)
    ratio_fused = online_result["fused_peak"] / max(h_train["fused_peak"], 1e-8)

    print(f"[Q]  H_cycle={online_result['cycle_peak']:.3f}  H_knn={online_result['knn_peak']:.3f}  H_fused={online_result['fused_peak']:.3f}")
    print(f"[Q]  ratio_cycle={ratio_cycle:.3f}  ratio_knn={ratio_knn:.3f}  ratio_fused={ratio_fused:.3f}")

    # ── Online viz panels ────────────────────────────────────────
    _render_online_viz(out_dir / "online", online_result, h_train,
                       ratio_cycle, ratio_knn, ratio_fused,
                       matcher, cycle_error_map)

    # ────────────────────── Save timing ──────────────────────
    out_timing = {
        "task_short": task_short,
        "task_full": task,
        "query": str(query_mp4),
        "n_refs": len(pngs),
        "n_cycle_null_pairs": n_pairs,
        "knn_pool_size": int(feats.shape[0]),
        "knn_k": KNN_K,
        "knn_route": null_knn["route"],
        "knn_cv": float(null_knn["cv"]),
        "h_train": {k: float(v) for k, v in h_train.items()
                    if k != "_intermediates" and not isinstance(v, (list, tuple))},
        "online": {
            "cycle_peak": float(online_result["cycle_peak"]),
            "knn_peak": float(online_result["knn_peak"]),
            "fused_peak": float(online_result["fused_peak"]),
            "ratio_cycle": float(ratio_cycle),
            "ratio_knn": float(ratio_knn),
            "ratio_fused": float(ratio_fused),
        },
        "timings_seconds": {k: float(v) for k, v in timer.timings.items()},
    }
    (out_dir / "timing.json").write_text(json.dumps(out_timing, indent=2))

    # Markdown summary
    md = ["# Benchmark — task " + task_short + "\n"]
    md.append(f"Task: `{task}`\n")
    md.append(f"Query: `{query_mp4.name}`\n")
    md.append(f"Refs: **{len(pngs)}**  |  Cycle null pairs: **{n_pairs}**  |  kNN k={KNN_K}, route=**{null_knn['route']}**\n")
    md.append("\n## Timing\n")
    md.append("| Phase | Step | Time |")
    md.append("|---|---|---|")
    for n in sorted(timer.timings.keys()):
        t = timer.timings[n]
        time_str = f"{t*1000:.0f} ms" if t < 1.0 else f"{t:.2f} s"
        phase = "OFFLINE" if "offline" in n else ("ONLINE" if "online" in n else "SETUP")
        md.append(f"| {phase} | {n} | **{time_str}** |")
    md.append("")
    md.append("## Scores\n")
    md.append("| Branch | H_train | H_test | ratio | verdict |")
    md.append("|---|---|---|---|---|")
    for sig in ["cycle", "knn", "fused"]:
        ht = h_train[f"{sig}_peak"]
        ho = online_result[f"{sig}_peak"]
        r = ho / max(ht, 1e-8)
        v = "🔴 HALLU" if r > 1.0 else ("⚠ borderline" if r > 0.95 else "✓ clean")
        md.append(f"| {sig} | {ht:.4f} | {ho:.4f} | **{r:.3f}** | {v} |")
    (out_dir / "timing_summary.md").write_text("\n".join(md))

    print(f"\n{'='*70}\nDone\n{'='*70}")
    print(f"Outputs: {out_dir}")
    print(f"  - timing.json")
    print(f"  - timing_summary.md")
    print(f"  - offline/ (7 viz)")
    print(f"  - online/ (7 viz)")
    print(f"\nFinal: ratio_fused = {ratio_fused:.3f}  → {'🔴 HALLU' if ratio_fused > 1.0 else ('⚠ borderline' if ratio_fused > 0.95 else '✓ clean')}")


# ─────────────────────────────────────────────────────────────────────────────
# score_video helper — full online pipeline w/ optional intermediate capture
# ─────────────────────────────────────────────────────────────────────────────


def score_video(mp4, pngs, pool, null_cycle_mean, null_cycle_peak, null_knn,
                matcher, seg, knn, cycle_signal,
                Mahal, empirical_p_value,
                cauchy_combine, cauchy_combine_video,
                fg_mask_at_size,
                return_intermediates=False,
                step_timer=None, step_prefix=""):
    """One-shot online scoring of mp4. Returns row dict."""
    def _step(name):
        return step_timer.step(f"{step_prefix}_{name}") if step_timer else _noop_ctx()

    @contextmanager
    def _noop_ctx():
        yield

    # Sample + SAM3
    with _step("01_sample_frames"):
        bgrs = sample_frames(mp4, N_QUERY_FRAMES)

    with _step("02_sam3_segment"):
        seg_bgrs = [seg.segment_frame(b) for b in bgrs]
        tmp_dir = Path(tempfile.mkdtemp(prefix="bench_"))
        seg_pngs = []
        for i, b in enumerate(seg_bgrs):
            p = tmp_dir / f"q_{i:04d}.png"
            cv2.imwrite(str(p), b)
            seg_pngs.append(p)

    # Cycle 9 pairs
    intermediates = {"cycle_pairs": [], "knn_frames": [], "seg_pngs": seg_pngs} if return_intermediates else None

    with _step("03_cycle_9_pairs"):
        h_pairs = []
        for t in range(len(seg_pngs) - 1):
            fwd = matcher.match(seg_pngs[t], seg_pngs[t + 1])
            bwd = matcher.match(seg_pngs[t + 1], seg_pngs[t])
            s = cycle_signal.compute(fwd, bwd)
            p_m = empirical_p_value(s.mean, null_cycle_mean)
            p_p = empirical_p_value(s.peak, null_cycle_peak)
            p_pair = cauchy_combine([p_m, p_p])
            h_pair = 1.0 - p_pair
            h_pairs.append(h_pair)
            if return_intermediates:
                intermediates["cycle_pairs"].append({
                    "t": t, "fwd": fwd, "bwd": bwd, "s_mean": s.mean, "s_peak": s.peak,
                    "p_m": p_m, "p_p": p_p, "h_pair": h_pair,
                    "img_a": seg_bgrs[t], "img_b": seg_bgrs[t+1],
                })
        cycle_peak = float(np.percentile(h_pairs, 80))
        cycle_robust = float(np.sort(h_pairs)[len(h_pairs)//10:-max(1, len(h_pairs)//10)].mean() if len(h_pairs) > 2 else np.mean(h_pairs))

    # kNN 10 frames
    with _step("04_knn_10_frames"):
        h_frames = []
        for i, p in enumerate(seg_pngs):
            fg = fg_mask_at_size(p, matcher.vis_size)
            res = knn.score_frame(p, pool, null_knn, query_fg_mask=fg)
            h_frames.append(res["H"])
            if return_intermediates:
                # Also capture D-map for viz
                q_feat = knn.dino.extract([p])[0]
                top_k_idx = knn.selector.select_for_query(q_feat, pool["feats"], knn.k)
                top_k_paths = [pool["paths"][j] for j in top_k_idx]
                matches = matcher.match_batch(p, top_k_paths, fg_mask=fg)
                warps = np.stack([m.warp for m in matches], axis=0)
                precs = np.stack([m.precision for m in matches], axis=0)
                D_map, _, _ = Mahal.ivar_per_pixel(warps, precs)
                intermediates["knn_frames"].append({
                    "i": i, "h_frame": res["H"], "ivar": res["ivar"], "peak": res["peak"],
                    "p_ivar": res["p_ivar"], "p_peak": res["p_peak"], "route": res["route"],
                    "img": seg_bgrs[i], "D_map": D_map, "top_k_paths": top_k_paths,
                })
        knn_peak = float(np.percentile(h_frames, 80))
        knn_robust = float(np.sort(h_frames)[len(h_frames)//10:-max(1, len(h_frames)//10)].mean() if len(h_frames) > 2 else np.mean(h_frames))

    # Fusion
    with _step("05_fusion_ratio"):
        fused_peak = cauchy_combine_video(cycle_peak, knn_peak)
        fused_robust = cauchy_combine_video(cycle_robust, knn_robust)

    # Cleanup
    for p in seg_pngs:
        try:
            p.unlink()
        except FileNotFoundError:
            pass
    try:
        tmp_dir.rmdir()
    except Exception:
        pass

    row = {
        "cycle_peak": cycle_peak, "cycle_robust": cycle_robust,
        "knn_peak": knn_peak, "knn_robust": knn_robust,
        "fused_peak": fused_peak, "fused_robust": fused_robust,
        "h_pairs": h_pairs, "h_frames": h_frames,
    }
    if return_intermediates:
        row["_intermediates"] = intermediates
    return row


# ─────────────────────────────────────────────────────────────────────────────
# Online viz panels
# ─────────────────────────────────────────────────────────────────────────────


def _render_online_viz(out_dir, online_result, h_train,
                       ratio_cycle, ratio_knn, ratio_fused,
                       matcher, cycle_error_map):
    inter = online_result["_intermediates"]

    # 01 — 10 query frames
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for ax, info in zip(axes.flat, inter["knn_frames"]):
        ax.imshow(cv2.cvtColor(info["img"], cv2.COLOR_BGR2RGB))
        ax.set_title(f"q[{info['i']}]", fontsize=10)
        ax.axis("off")
    fig.suptitle("Query — 10 sampled frames (after SAM3)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "01_query_frames.png", dpi=110)
    plt.close(fig)

    # 02 — 1 query pair cycle viz
    pair = inter["cycle_pairs"][len(inter["cycle_pairs"]) // 2]
    fwd, bwd = pair["fwd"], pair["bwd"]
    err = cycle_error_map(fwd.warp, bwd.warp)
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    axes[0].imshow(cv2.cvtColor(cv2.resize(pair["img_a"], (matcher.vis_size, matcher.vis_size)), cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"query A (frame {pair['t']})"); axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(cv2.resize(pair["img_b"], (matcher.vis_size, matcher.vis_size)), cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"query B (frame {pair['t']+1})"); axes[1].axis("off")
    axes[2].imshow(warp_to_flow_rgb(fwd.warp)); axes[2].set_title("RoMa A→B"); axes[2].axis("off")
    axes[3].imshow(warp_to_flow_rgb(bwd.warp)); axes[3].set_title("RoMa B→A"); axes[3].axis("off")
    im = axes[4].imshow(err, cmap="turbo", vmin=0, vmax=float(np.percentile(err, 99)) + 1e-6)
    axes[4].set_title(f"cycle error (px)\nmean={pair['s_mean']:.2f}, peak={pair['s_peak']:.2f}\nH_pair={pair['h_pair']:.3f}")
    axes[4].axis("off"); plt.colorbar(im, ax=axes[4], fraction=0.046)
    fig.suptitle(f"Cycle branch — 1 query pair (consecutive frames {pair['t']}, {pair['t']+1})", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "02_query_pair_example.png", dpi=110)
    plt.close(fig)

    # 03 — Cycle H_pairs bar
    h_pairs = online_result["h_pairs"]
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(h_pairs)), h_pairs, color="steelblue", edgecolor="white")
    ax.axhline(online_result["cycle_peak"], color="red", ls="--",
               label=f"H_cycle = p80 = {online_result['cycle_peak']:.3f}")
    ax.axhline(h_train["cycle_peak"], color="green", ls=":",
               label=f"H_train_cycle = {h_train['cycle_peak']:.3f}")
    ax.set_xlabel("consecutive pair index (0-8)")
    ax.set_ylabel("H_pair (1 - p_pair)")
    ax.set_title(f"Cycle — 9 pair scores  →  ratio = {ratio_cycle:.3f}")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_dir / "03_cycle_pairs_bar.png", dpi=110)
    plt.close(fig)

    # 04 — kNN top-k per frame (3 sample frames)
    sample_frames_idx = [0, 5, 9]
    fig, axes = plt.subplots(len(sample_frames_idx), 6, figsize=(24, 4.5 * len(sample_frames_idx)))
    for r, idx in enumerate(sample_frames_idx):
        info = inter["knn_frames"][idx]
        axes[r, 0].imshow(cv2.cvtColor(info["img"], cv2.COLOR_BGR2RGB))
        axes[r, 0].set_title(f"query q[{idx}]\nH={info['h_frame']:.3f}, route={info['route']}", fontsize=10)
        axes[r, 0].axis("off")
        for k, ref_path in enumerate(info["top_k_paths"][:5]):
            axes[r, k+1].imshow(cv2.cvtColor(cv2.resize(cv2.imread(str(ref_path)), (matcher.vis_size, matcher.vis_size)), cv2.COLOR_BGR2RGB))
            axes[r, k+1].set_title(f"top-{k+1}", fontsize=9)
            axes[r, k+1].axis("off")
    fig.suptitle(f"kNN branch — top-5 of k={KNN_K} retrieved refs for 3 query frames", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "04_knn_topk_per_frame.png", dpi=110)
    plt.close(fig)

    # 05 — D-map per query frame
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    for ax, info in zip(axes.flat, inter["knn_frames"]):
        q = cv2.resize(info["img"], (matcher.vis_size, matcher.vis_size))
        overlay = heatmap_overlay(q, info["D_map"], alpha=0.55)
        ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax.set_title(f"q[{info['i']}]  ivar={info['ivar']:.2f}  peak_z={info['peak']:.2f}\nH={info['h_frame']:.3f}", fontsize=9)
        ax.axis("off")
    fig.suptitle("kNN — Cochran D-map overlay per query frame  (high = refs disagree)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "05_knn_dmap_per_frame.png", dpi=110)
    plt.close(fig)

    # 06 — kNN H_frames bar
    h_frames = online_result["h_frames"]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(h_frames)), h_frames, color="darkorange", edgecolor="white")
    ax.axhline(online_result["knn_peak"], color="red", ls="--",
               label=f"H_knn = p80 = {online_result['knn_peak']:.3f}")
    ax.axhline(h_train["knn_peak"], color="green", ls=":",
               label=f"H_train_knn = {h_train['knn_peak']:.3f}")
    ax.set_xlabel("query frame index (0-9)")
    ax.set_ylabel("H_frame (1 - p_routed)")
    ax.set_title(f"kNN — 10 frame scores  →  ratio = {ratio_knn:.3f}")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_dir / "06_knn_frames_bar.png", dpi=110)
    plt.close(fig)

    # 07 — Final scores bar
    fig, ax = plt.subplots(figsize=(11, 6))
    sigs = ["cycle", "knn", "fused"]
    h_train_vals = [h_train[f"{s}_peak"] for s in sigs]
    h_test_vals = [online_result[f"{s}_peak"] for s in sigs]
    ratios = [h_test_vals[i] / max(h_train_vals[i], 1e-8) for i in range(3)]
    x = np.arange(3)
    w = 0.35
    ax.bar(x - w/2, h_train_vals, w, color="green", alpha=0.7, label="H_train")
    ax.bar(x + w/2, h_test_vals, w, color="orange", alpha=0.7, label="H_test (query)")
    for i, r in enumerate(ratios):
        verd = "🔴" if r > 1.0 else ("⚠" if r > 0.95 else "✓")
        ax.text(x[i], max(h_train_vals[i], h_test_vals[i]) + 0.03, f"ratio={r:.3f} {verd}", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(sigs)
    ax.set_ylabel("H score (1 - p)")
    ax.set_title("Final — H_train vs H_test per branch  →  ratio = decision metric")
    ax.legend()
    ax.set_ylim(0, max(max(h_train_vals), max(h_test_vals)) + 0.18)
    fig.tight_layout()
    fig.savefig(out_dir / "07_final_scores.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
