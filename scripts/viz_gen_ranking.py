#!/usr/bin/env python3
"""Generate reviewer-friendly visualizations cho 24 gen videos.

Per video output:
  - 10 sampled frames in a grid (with H_pair scores)
  - Cycle heatmaps for each of the 9 consecutive pairs
  - Summary card with score, ratio, verdict + thumbnails

Per task output:
  - Real (training) vs all 5 gens for that task — heatmaps side-by-side

Overview:
  - Grid of all 24 gens ranked by H_peak with thumbnails

Final output:
  paper-physical-gr1/viz/
    per_video/<rank>_<task>_<vid>/...
    per_task_comparison/<task>/...
    overview_ranking.png
"""
from __future__ import annotations

import csv
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

REPO_ROOT = Path("/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination")
BENCH = REPO_ROOT / "paper-physical-gr1"
REF_ROOT = BENCH / "reference"
GEN_ROOT = BENCH / "generated"
RAW_VIDEO_ROOT = BENCH / "raw_videos" / "gr1"
EVAL_DIR = BENCH / "per_task_dense_eval"
VIZ_OUT = BENCH / "viz"

NULL_LAGS = [1, 2, 5, 10]
N_FRAMES = 10

TASK_FULL = {
    "1": "1_Use the right hand to pick up green bok choy from tan table right side to bottom level of wire basket.",
    "2": "2_Use the right hand to pick up rubik's cube from top level of the shelf to bottom level of the shelf.",
    "3": "3_Use the right hand to pick up banana from teal plate to wooden table.",
    "4": "4_Use the left hand to pick up dragonfruit from pink plate to teal plate.",
    "6": "6_Use the right hand to pick up orange from middle of table to bottom white shelf.",
}


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
    ps = [p for p in ps if 0 < p < 1]
    if not ps:
        return 0.5
    t = float(np.mean([np.tan(np.pi * (0.5 - p)) for p in ps]))
    return float(0.5 - np.arctan(t) / np.pi)


def score_pairs_with_heatmap(seg_paths: list[Path], matcher, cycle_sig) -> list[dict]:
    """Per consecutive pair: return {mean, peak, drift_map, frame_a, frame_b}."""
    out = []
    for t in range(len(seg_paths) - 1):
        fwd = matcher.match(seg_paths[t], seg_paths[t + 1])
        bwd = matcher.match(seg_paths[t + 1], seg_paths[t])
        sig = cycle_sig.compute(fwd, bwd)
        out.append({
            "mean":      sig.mean,
            "peak":      sig.peak,
            "drift_map": sig.pixel_map,        # (H, W)
            "cert_fwd":  fwd.cert,
        })
    return out


def overlay_heatmap_on_frame(frame_bgr: np.ndarray,
                              heatmap: np.ndarray,
                              cert: Optional[np.ndarray] = None,
                              cert_floor: float = 0.1,
                              alpha: float = 0.6,
                              vmin: float = 0.0,
                              vmax: float = 30.0,
                              ) -> np.ndarray:
    """Overlay heatmap (cycle drift) on frame, MASKED by RoMa cert.

    Background / uniform-texture pixels (cert < cert_floor) → heatmap
    suppressed to 0 so the visualization matches the actual signal
    aggregation. Without this masking the background looks "bright"
    because RoMa's warp on textureless regions is arbitrary and the
    cycle composition there is noisy — but those pixels are NOT used
    in the score computation.
    """
    H, W = heatmap.shape
    h_clipped = np.clip(heatmap, vmin, vmax)
    h_norm = (h_clipped - vmin) / max(vmax - vmin, 1e-6)

    if cert is not None:
        # Sample cert to heatmap resolution, then zero out low-cert pixels
        cert_r = cv2.resize(cert.astype(np.float32), (W, H),
                            interpolation=cv2.INTER_LINEAR)
        mask = (cert_r > cert_floor).astype(np.float32)
        h_norm = h_norm * mask

    frame_r = cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_LINEAR)
    colored = cv2.applyColorMap((h_norm * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    # Only blend where mask is non-zero (so frame shows through on bg)
    if cert is not None:
        alpha_map = (h_norm > 0.001)[..., None].astype(np.float32) * alpha
        overlay = (frame_r * (1 - alpha_map) + colored * alpha_map).astype(np.uint8)
    else:
        overlay = cv2.addWeighted(frame_r, 1.0 - alpha, colored, alpha, 0)
    return overlay


def make_per_video_card(video_label: str,
                        score_info: dict,
                        seg_bgrs: list[np.ndarray],
                        pair_data: list[dict],
                        H_pairs: list[float],
                        H_train: float,
                        out_path: Path):
    """Big summary card: video info + 10 frames + 9 heatmaps + H_pair bars."""
    n_frames = len(seg_bgrs)
    n_pairs = len(pair_data)

    fig = plt.figure(figsize=(20, 12), constrained_layout=True)
    gs = fig.add_gridspec(4, n_frames, height_ratios=[1, 1, 1, 0.6])

    # Title
    H_video = float(np.percentile(H_pairs, 80))
    ratio = H_video / max(H_train, 1e-6)
    verdict = "HALLU" if ratio > 1.0 else ("borderline" if ratio > 0.95 else "clean")
    color = {"HALLU": "red", "borderline": "orange", "clean": "green"}[verdict]
    fig.suptitle(
        f"{video_label}  |  H_video={H_video:.4f}  H_train={H_train:.4f}  "
        f"ratio={ratio:.3f}  →  {verdict}",
        fontsize=14, fontweight="bold", color=color,
    )

    # Row 1: 10 sampled frames
    for i, bgr in enumerate(seg_bgrs):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ax.set_title(f"f{i}", fontsize=8)
        ax.axis("off")

    # Row 2: cycle heatmaps (between f_t and f_{t+1})
    for t, pd in enumerate(pair_data):
        ax = fig.add_subplot(gs[1, t])
        overlay = overlay_heatmap_on_frame(seg_bgrs[t], pd["drift_map"],
                                            cert=pd["cert_fwd"], cert_floor=0.1,
                                            alpha=0.6)
        ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax.set_title(f"f{t}→{t+1}", fontsize=8)
        ax.axis("off")
    # Spare cell
    if n_pairs < n_frames:
        ax = fig.add_subplot(gs[1, -1])
        ax.text(0.5, 0.5, "—", ha="center", va="center", fontsize=20, color="gray")
        ax.axis("off")

    # Row 3: per-pair cycle peak values (raw cycle drift peak — informative)
    for t, pd in enumerate(pair_data):
        ax = fig.add_subplot(gs[2, t])
        peaks = pd["drift_map"][pd["cert_fwd"] > 0.1]
        ax.hist(peaks.flatten()[:5000], bins=30, color="steelblue")
        ax.axvline(np.percentile(peaks, 99) if peaks.size > 0 else 0,
                   color="red", linewidth=1.0, linestyle="--")
        ax.set_title(f"pair {t}\npeak={pd['peak']:.2f}", fontsize=7)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=6)

    # Row 4: H_pair bar chart
    ax = fig.add_subplot(gs[3, :])
    bars = ax.bar(range(len(H_pairs)), H_pairs,
                  color=["red" if h > 0.8 else "orange" if h > 0.5 else "green"
                         for h in H_pairs])
    ax.axhline(0.5, color="gray", linewidth=0.5, linestyle="--")
    ax.axhline(np.percentile(H_pairs, 80), color="blue", linewidth=1.0,
               linestyle="--", label=f"p80 = {np.percentile(H_pairs, 80):.3f}")
    ax.set_xticks(range(len(H_pairs)))
    ax.set_xticklabels([f"pair {i}" for i in range(len(H_pairs))], fontsize=8)
    ax.set_ylabel("H_pair", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("Per-pair anomaly score (red = HALLU pair, green = clean)", fontsize=9)
    ax.legend(loc="upper right", fontsize=7)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def make_task_comparison(task: str,
                         real_card_data: dict,
                         gen_cards_data: list[dict],
                         out_path: Path):
    """Side-by-side: real video frames+heatmaps on top, all gens below."""
    n_frames = N_FRAMES
    n_rows = 1 + len(gen_cards_data)
    fig = plt.figure(figsize=(20, 2.5 * n_rows), constrained_layout=True)
    gs = fig.add_gridspec(n_rows, n_frames + 1)

    def render_video_row(row, label, seg_bgrs, pair_data, score_info):
        # Label cell
        ax_lbl = fig.add_subplot(gs[row, 0])
        H = score_info.get("H_peak", 0)
        ratio = score_info.get("ratio", 0)
        color = score_info.get("color", "black")
        ax_lbl.text(0.5, 0.5,
                    f"{label}\nH={H:.3f}\nratio={ratio:.2f}",
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color=color, transform=ax_lbl.transAxes)
        ax_lbl.axis("off")
        # Frame cells with heatmap overlay where pair exists
        for i in range(n_frames):
            ax = fig.add_subplot(gs[row, i + 1])
            if i < len(seg_bgrs):
                if i < len(pair_data):
                    overlay = overlay_heatmap_on_frame(
                        seg_bgrs[i], pair_data[i]["drift_map"],
                        cert=pair_data[i]["cert_fwd"], cert_floor=0.1,
                        alpha=0.5,
                    )
                    ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                else:
                    ax.imshow(cv2.cvtColor(seg_bgrs[i], cv2.COLOR_BGR2RGB))
            ax.axis("off")

    fig.suptitle(f"Task {task}: REAL training (top) vs {len(gen_cards_data)} GEN videos",
                 fontsize=12, fontweight="bold")
    render_video_row(0, "REAL", real_card_data["seg_bgrs"], real_card_data["pair_data"],
                     {"H_peak": real_card_data["H_video"], "ratio": 1.0, "color": "green"})
    for i, gc in enumerate(gen_cards_data, 1):
        verdict_color = ("red" if gc["ratio"] > 1.0 else
                          "orange" if gc["ratio"] > 0.95 else "green")
        render_video_row(i, gc["label"], gc["seg_bgrs"], gc["pair_data"],
                         {"H_peak": gc["H_video"], "ratio": gc["ratio"],
                          "color": verdict_color})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def make_overview_ranking(rows: list[dict], thumbs: dict[str, np.ndarray], out_path: Path):
    """Big poster: all 24 gens ranked, with first-frame thumbnail."""
    n = len(rows)
    cols = 6
    rows_n = (n + cols - 1) // cols
    fig = plt.figure(figsize=(cols * 3.5, rows_n * 3.0), constrained_layout=True)
    gs = fig.add_gridspec(rows_n, cols)
    fig.suptitle("All 24 gen videos ranked by ratio (most → least hallu)",
                 fontsize=14, fontweight="bold")
    for i, r in enumerate(rows):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        thumb = thumbs.get(r["video_id"])
        if thumb is not None:
            ax.imshow(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB))
        color = "red" if r["ratio"] > 1.0 else ("orange" if r["ratio"] > 0.95 else "green")
        ax.set_title(
            f"#{i+1}  task {r['task']}/{r['video']}\n"
            f"H={r['h_test']:.3f}  ratio={r['ratio']:.3f}  →  {r['verdict']}",
            fontsize=9, color=color, fontweight="bold",
        )
        ax.axis("off")
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────


def main():
    VIZ_OUT.mkdir(parents=True, exist_ok=True)
    print("Loading models …")
    from warp_score.temporal_signals import (
        CycleSignal, empirical_p_value,
    )
    from warp_score.matcher import RoMaMatcher
    from warp_score.sam_segmenter import VideoFrameSegmenter
    matcher = RoMaMatcher(setting="turbo", device="cuda", use_precision=True, vis_size=224)
    matcher._load_model()
    sam = VideoFrameSegmenter()
    cycle = CycleSignal(cert_floor=0.1)

    def score_pair(p_a, p_b):
        fwd = matcher.match(p_a, p_b)
        bwd = matcher.match(p_b, p_a)
        return cycle.compute(fwd, bwd), fwd

    def process_video(mp4: Path) -> dict:
        """Sample + SAM3 + pair signals."""
        bgrs = sample_frames(mp4, N_FRAMES)
        if len(bgrs) < 2:
            return None
        seg_bgrs = [sam.segment_frame(b) for b in bgrs]
        tmp = Path(tempfile.mkdtemp(prefix="viz_"))
        paths = []
        for i, b in enumerate(seg_bgrs):
            p = tmp / f"f{i:04d}.png"
            cv2.imwrite(str(p), b)
            paths.append(p)
        pair_data = []
        for t in range(len(paths) - 1):
            sig, fwd = score_pair(paths[t], paths[t + 1])
            pair_data.append({
                "mean":      sig.mean,
                "peak":      sig.peak,
                "drift_map": sig.pixel_map,
                "cert_fwd":  fwd.cert,
            })
        for p in paths:
            p.unlink(missing_ok=True)
        tmp.rmdir()
        return {"seg_bgrs": seg_bgrs, "pair_data": pair_data}

    # ── Load per-task null + H_train from existing csv
    raw_rows = list(csv.DictReader(open(EVAL_DIR / "per_task_dense_table.csv")))
    task_H_train = {r["task"]: float(r["cycle_peak"])
                    for r in raw_rows if r["type"] == "REAL"}

    # Build per-task null from reference PNGs (re-compute or load)
    # For viz we need the null distributions for p-value lookup
    null_per_task = {}
    print("Building per-task nulls …")
    for task_short, task_full in TASK_FULL.items():
        pngs = sorted((REF_ROOT / task_full).glob("*.png"))
        means, peaks = [], []
        for lag in NULL_LAGS:
            for i in range(len(pngs) - lag):
                sig, _ = score_pair(pngs[i], pngs[i + lag])
                means.append(sig.mean)
                peaks.append(sig.peak)
        null_per_task[task_short] = (
            np.sort(np.asarray(means, dtype=np.float32)),
            np.sort(np.asarray(peaks, dtype=np.float32)),
        )
        print(f"  task {task_short}: {len(means)} null pairs")

    def aggregate_H(pair_data, null_mean, null_peak):
        H_pairs = []
        for pd in pair_data:
            p_m = empirical_p_value(pd["mean"], null_mean)
            p_p = empirical_p_value(pd["peak"], null_peak)
            H_pairs.append(1 - cauchy_combine([p_m, p_p]))
        return H_pairs

    # ── Per-video cards for all gens (sorted by ratio)
    print("\nGenerating per-video cards for gens …")
    gen_rows = []
    gen_data = {}   # task → list of dict {label, seg_bgrs, pair_data, H_video, ratio}
    thumbs = {}
    for task_short, task_full in TASK_FULL.items():
        gen_dir = GEN_ROOT / task_full
        for mp4 in sorted(gen_dir.glob("v*.mp4")):
            data = process_video(mp4)
            if data is None:
                continue
            null_m, null_p = null_per_task[task_short]
            H_pairs = aggregate_H(data["pair_data"], null_m, null_p)
            H_video = float(np.percentile(H_pairs, 80))
            H_train = task_H_train[task_short]
            ratio = H_video / H_train
            verdict = ("HALLU" if ratio > 1.0
                       else "borderline" if ratio > 0.95
                       else "clean")
            video_id = f"task{task_short}_{mp4.stem}"
            gen_rows.append({"task": task_short, "video": mp4.name,
                              "video_id": video_id, "h_test": H_video,
                              "h_train": H_train, "ratio": ratio, "verdict": verdict,
                              "H_pairs": H_pairs})
            gen_data.setdefault(task_short, []).append({
                "label": f"gen {mp4.stem}", "video_id": video_id,
                "seg_bgrs": data["seg_bgrs"], "pair_data": data["pair_data"],
                "H_video": H_video, "ratio": ratio,
            })
            thumbs[video_id] = data["seg_bgrs"][0]
            print(f"  {video_id}  ratio={ratio:.3f}  {verdict}")

    # Sort by ratio
    gen_rows.sort(key=lambda r: -r["ratio"])

    # Render per-video cards
    print("\nRendering per-video summary cards …")
    for rank, r in enumerate(gen_rows, 1):
        video_id = r["video_id"]
        gc = next(g for gs in gen_data.values() for g in gs if g["video_id"] == video_id)
        out_dir = VIZ_OUT / "per_video" / f"rank{rank:02d}_{video_id}"
        make_per_video_card(
            f"rank #{rank} | task {r['task']} | {r['video']}",
            r, gc["seg_bgrs"], gc["pair_data"], r["H_pairs"],
            r["h_train"], out_dir / "summary_card.png",
        )
        if rank <= 3 or rank > len(gen_rows) - 3:
            print(f"  rank {rank}: {video_id}")

    # ── Per-task real vs gens comparison
    print("\nRendering per-task real-vs-gens comparison …")
    for task_short, task_full in TASK_FULL.items():
        real_mp4 = RAW_VIDEO_ROOT / f"{task_short}.mp4"
        if not real_mp4.exists():
            continue
        real_data = process_video(real_mp4)
        if real_data is None:
            continue
        null_m, null_p = null_per_task[task_short]
        real_H_pairs = aggregate_H(real_data["pair_data"], null_m, null_p)
        real_card = {**real_data, "H_video": float(np.percentile(real_H_pairs, 80))}
        out_path = VIZ_OUT / "per_task_comparison" / f"task{task_short}.png"
        make_task_comparison(task_short, real_card, gen_data[task_short], out_path)
        print(f"  task {task_short} → {out_path}")

    # ── Overview ranking
    print("\nRendering overview ranking …")
    make_overview_ranking(gen_rows, thumbs, VIZ_OUT / "overview_ranking.png")

    # ── INDEX
    md = ["# WarpDyn — Visualization gallery\n"]
    md.append("All 24 generated videos scored against per-task multi-lag null. "
              "Ratio = H_test / H_train_task. Sorted by ratio (most → least hallu).\n")
    md.append("## Overview\n\n![ranking](overview_ranking.png)\n")
    md.append("## Per-task real-vs-gens comparison\n")
    for task_short in TASK_FULL:
        md.append(f"### Task {task_short}\n")
        md.append(f"![task{task_short}](per_task_comparison/task{task_short}.png)\n")
    md.append("## Per-video summary cards (ranked)\n")
    md.append("| Rank | Task | Video | H_test | H_train | Ratio | Verdict | Card |")
    md.append("|---|---|---|---|---|---|---|---|")
    for rank, r in enumerate(gen_rows, 1):
        md.append(f"| {rank} | {r['task']} | {r['video']} | {r['h_test']:.4f} | "
                  f"{r['h_train']:.4f} | **{r['ratio']:.3f}** | {r['verdict']} | "
                  f"[card](per_video/rank{rank:02d}_{r['video_id']}/summary_card.png) |")

    (VIZ_OUT / "INDEX.md").write_text("\n".join(md))
    print(f"\nAll viz → {VIZ_OUT}")
    print(f"  INDEX:    {VIZ_OUT}/INDEX.md")
    print(f"  Overview: {VIZ_OUT}/overview_ranking.png")
    print(f"  Per-video cards: {VIZ_OUT}/per_video/ (24 dirs)")
    print(f"  Per-task comparisons: {VIZ_OUT}/per_task_comparison/ (5 files)")


if __name__ == "__main__":
    main()
