#!/usr/bin/env python3
"""Visualize the 8 fusion-complementarity catches on GR-1.

Each catch is a gen where one signal (cycle XOR k-NN) flags it as HALLU and
the other signal misses it — concrete evidence that the two modalities are
orthogonal. We render one wide PNG per catch showing:

  1. Title bar: "CYCLE-ONLY CATCH" or "KNN-ONLY CATCH" + scores
  2. Row of 10 sampled query frames (with H_cycle pair and H_knn frame scores)
  3. Cycle row: 9 cycle drift heatmaps (TURBO + cert mask + HALLU bbox)
  4. k-NN row: per-frame D_map heatmap + the routed signal value
  5. Top-3 nearest DINOv2 refs per query frame (small strip per query)

Output:
  paper-physical-gr1/viz/complementarity/{cycle_only,knn_only}/<task>_<vid>_viz.png

Then uploaded to HF dataset `twanghcmut/wmbench` at
`gr-1/fusion_analysis/{cycle_only,knn_only}/`.
"""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

# Always use the main repo's data (this worktree only has pool/)
REPO_ROOT = Path("/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination")
BENCH = REPO_ROOT / "paper-physical-gr1"
REF_ROOT = BENCH / "reference"
GEN_ROOT = BENCH / "generated"
EVAL_DIR = BENCH / "per_task_dense_eval"
NULL_DIR = BENCH / "null_per_task"
DINO_CACHE_DIR = BENCH / "ref_cache"

# Worktree-local viz output (won't pollute main repo)
WORKTREE = Path(__file__).resolve().parent.parent
VIZ_OUT = WORKTREE / "paper-physical-gr1" / "viz" / "complementarity"

N_FRAMES = 10
KNN_K = 15

HF_REPO = "twanghcmut/wmbench"
HF_PREFIX = "gr-1/fusion_analysis"

TASK_FULL = {
    "1": "1_Use the right hand to pick up green bok choy from tan table right side to bottom level of wire basket.",
    "2": "2_Use the right hand to pick up rubik's cube from top level of the shelf to bottom level of the shelf.",
    "3": "3_Use the right hand to pick up banana from teal plate to wooden table.",
    "4": "4_Use the left hand to pick up dragonfruit from pink plate to teal plate.",
    "6": "6_Use the right hand to pick up orange from middle of table to bottom white shelf.",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (sample frames + cycle heatmap overlay) — copied from viz_gen_ranking
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


def cauchy_combine(ps):
    ps = [p for p in ps if 0 < p < 1]
    if not ps:
        return 0.5
    t = float(np.mean([np.tan(np.pi * (0.5 - p)) for p in ps]))
    return float(0.5 - np.arctan(t) / np.pi)


def overlay_heatmap_on_frame(
    frame_bgr: np.ndarray,
    heatmap: np.ndarray,
    cert: np.ndarray | None = None,
    cert_floor: float = 0.1,
    alpha: float = 0.85,
    percentile_clip: float = 95.0,
    annotate_top_cluster: bool = True,
) -> np.ndarray:
    """TURBO overlay with cert-mask + HALLU bbox (lifted from viz_gen_ranking)."""
    H, W = heatmap.shape
    frame_r = cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_LINEAR)
    if cert is not None:
        cert_r = cv2.resize(cert.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        mask = cert_r > cert_floor
    else:
        mask = np.ones_like(heatmap, dtype=bool)
    masked = np.where(mask, heatmap, 0.0).astype(np.float32)
    if mask.any():
        vmax = max(float(np.percentile(heatmap[mask], percentile_clip)), 5.0)
    else:
        vmax = 30.0
    h_norm = np.clip(masked / max(vmax, 1e-6), 0, 1)
    colored = cv2.applyColorMap((h_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    alpha_map = (h_norm > 0.05)[..., None].astype(np.float32) * alpha
    overlay = (frame_r * (1 - alpha_map) + colored * alpha_map).astype(np.uint8)
    if annotate_top_cluster and mask.any():
        high_thresh = float(np.percentile(heatmap[mask], 95))
        if high_thresh > 1.0:
            high_mask = ((heatmap > high_thresh) & mask).astype(np.uint8) * 255
            num, _, stats, _ = cv2.connectedComponentsWithStats(high_mask, connectivity=8)
            if num > 1:
                biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                x, y, w, h, area = stats[biggest]
                if area > 10:
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.putText(overlay, "HALLU", (x, max(y - 5, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1,
                                cv2.LINE_AA)
    return overlay


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline state — build once, reuse across all 8 catches
# ─────────────────────────────────────────────────────────────────────────────


def load_catches() -> tuple[list[dict], list[dict]]:
    rows = list(csv.DictReader(open(EVAL_DIR / "per_task_ratio_table.csv")))
    gens = [r for r in rows if r["type"] == "GEN"]
    cycle_only = [r for r in gens
                  if float(r["ratio_cycle"]) > 1.0 and float(r["ratio_knn"]) <= 1.0]
    knn_only = [r for r in gens
                if float(r["ratio_cycle"]) <= 1.0 and float(r["ratio_knn"]) > 1.0]
    return cycle_only, knn_only


def build_per_task_state(task_short: str, matcher, sam, cycle_sig, knn) -> dict:
    """Cycle null + DINOv2 pool + LOO null for one task. Cached on disk."""
    task_full = TASK_FULL[task_short]
    npz_path = NULL_DIR / f"task_{task_short}.npz"
    npz = np.load(npz_path, allow_pickle=True)

    null_cycle_mean = np.asarray(npz["cycle_cycle_mean"], dtype=np.float32)
    null_cycle_peak = np.asarray(npz["cycle_cycle_peak"], dtype=np.float32)
    null_ivar = np.asarray(npz["knn_null_ivar"], dtype=np.float32)
    null_peak = np.asarray(npz["knn_null_peak"], dtype=np.float32)
    knn_route = str(npz["knn_route"])

    # k-NN pool: DINOv2 features for the 120 refs (cached on disk by build_cache)
    pngs = sorted((REF_ROOT / task_full).glob("frame_*.png"))
    pool = knn.build_pool(task_full, pngs, DINO_CACHE_DIR)
    null_knn = {
        "null_ivar": np.sort(null_ivar),
        "null_peak": np.sort(null_peak),
        "route": knn_route,
    }
    return {
        "task_short": task_short,
        "task_full": task_full,
        "null_cycle_mean": np.sort(null_cycle_mean),
        "null_cycle_peak": np.sort(null_cycle_peak),
        "pool": pool,
        "null_knn": null_knn,
        "ref_pngs": pngs,
    }


def process_video(mp4: Path, state: dict, matcher, sam, cycle_sig, knn) -> dict:
    """Sample + SAM3 + cycle pair maps + per-frame k-NN D_map + top-k ref ids."""
    from warp_score.temporal_signals import empirical_p_value
    from warp_score.knn_signal import fg_mask_at_size
    from warp_score.statistics import MahalanobisStatistics

    bgrs = sample_frames(mp4, N_FRAMES)
    if len(bgrs) < 2:
        return None
    seg_bgrs = [sam.segment_frame(b) for b in bgrs]
    tmp = Path(tempfile.mkdtemp(prefix="vizcomp_"))
    seg_pngs: list[Path] = []
    for i, b in enumerate(seg_bgrs):
        p = tmp / f"f_{i:04d}.png"
        cv2.imwrite(str(p), b)
        seg_pngs.append(p)

    null_m = state["null_cycle_mean"]
    null_p = state["null_cycle_peak"]

    # ── Cycle: 9 consecutive pairs ──────────────────────────────────────
    cycle_pairs: list[dict] = []
    cycle_H_pairs: list[float] = []
    for t in range(len(seg_pngs) - 1):
        fwd = matcher.match(seg_pngs[t], seg_pngs[t + 1])
        bwd = matcher.match(seg_pngs[t + 1], seg_pngs[t])
        sig = cycle_sig.compute(fwd, bwd)
        p_m = empirical_p_value(sig.mean, null_m)
        p_p = empirical_p_value(sig.peak, null_p)
        H_pair = 1.0 - cauchy_combine([p_m, p_p])
        cycle_pairs.append({
            "mean": sig.mean,
            "peak": sig.peak,
            "drift_map": sig.pixel_map,
            "cert_fwd": fwd.cert,
            "H_pair": H_pair,
        })
        cycle_H_pairs.append(H_pair)

    # ── k-NN: per-frame D_map + top-3 ref previews ─────────────────────
    knn_frames: list[dict] = []
    knn_H_frames: list[float] = []
    for i, q_png in enumerate(seg_pngs):
        q_feat = knn.dino.extract([q_png])[0]
        top_k_idx = knn.selector.select_for_query(q_feat, state["pool"]["feats"], knn.k)
        k_refs = [state["pool"]["paths"][j] for j in top_k_idx]
        fg = fg_mask_at_size(q_png, matcher.vis_size)
        match_results = matcher.match_batch(q_png, k_refs, fg_mask=fg)
        warps = np.stack([m.warp for m in match_results], axis=0)
        precisions = np.stack([m.precision for m in match_results], axis=0)
        D_map, _, _ = MahalanobisStatistics.ivar_per_pixel(warps, precisions)
        ivar = MahalanobisStatistics.interior_mean(D_map, fg)
        peak = MahalanobisStatistics.peak_max_z(D_map, fg)
        p_ivar = empirical_p_value(ivar, state["null_knn"]["null_ivar"])
        p_peak = empirical_p_value(peak, state["null_knn"]["null_peak"])
        p_routed = p_peak if state["null_knn"]["route"] == "peak" else p_ivar
        H_frame = 1.0 - p_routed
        # Top-3 ref previews (use full k_refs[0:3])
        top3_paths = k_refs[:3]
        top3_bgrs = [cv2.imread(str(p)) for p in top3_paths]
        knn_frames.append({
            "D_map": D_map,
            "fg_mask": fg,
            "ivar": ivar,
            "peak": peak,
            "H_frame": H_frame,
            "top3_bgrs": top3_bgrs,
            "route": state["null_knn"]["route"],
        })
        knn_H_frames.append(H_frame)

    # ── Cleanup ─────────────────────────────────────────────────────────
    for p in seg_pngs:
        p.unlink(missing_ok=True)
    tmp.rmdir()

    return {
        "seg_bgrs": seg_bgrs,
        "cycle_pairs": cycle_pairs,
        "cycle_H_pairs": cycle_H_pairs,
        "knn_frames": knn_frames,
        "knn_H_frames": knn_H_frames,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Rendering — one PNG per catch
# ─────────────────────────────────────────────────────────────────────────────


def render_knn_overlay(
    frame_bgr: np.ndarray,
    D_map: np.ndarray,
    fg_mask: np.ndarray,
    alpha: float = 0.75,
    percentile_clip: float = 95.0,
) -> np.ndarray:
    """TURBO overlay of D_map on frame, masked by fg_mask."""
    H, W = D_map.shape
    frame_r = cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_LINEAR)
    masked = np.where(fg_mask, D_map, 0.0).astype(np.float32)
    if fg_mask.any():
        vmax = max(float(np.percentile(D_map[fg_mask], percentile_clip)), 5.0)
    else:
        vmax = 30.0
    h_norm = np.clip(masked / max(vmax, 1e-6), 0, 1)
    colored = cv2.applyColorMap((h_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    alpha_map = (h_norm > 0.05)[..., None].astype(np.float32) * alpha
    overlay = (frame_r * (1 - alpha_map) + colored * alpha_map).astype(np.uint8)
    return overlay


def render_catch(
    catch_row: dict,
    group: str,       # "cycle_only" or "knn_only"
    state: dict,
    data: dict,
    out_path: Path,
):
    """One wide PNG per catch."""
    task_short = catch_row["task"]
    task_desc = TASK_FULL[task_short][2:]   # strip leading "N_"
    task_desc_short = task_desc[:60] + ("…" if len(task_desc) > 60 else "")

    ratio_cycle = float(catch_row["ratio_cycle"])
    ratio_knn = float(catch_row["ratio_knn"])
    ratio_fused = float(catch_row["ratio_fused"])
    verdict_fused = catch_row["verdict_fused"]
    if group == "cycle_only":
        header = "CYCLE-ONLY CATCH  (k-NN missed)"
        header_color = "#cc0000"
    else:
        header = "k-NN-ONLY CATCH  (cycle missed)"
        header_color = "#cc7700"

    n = len(data["seg_bgrs"])         # 10
    n_pairs = len(data["cycle_pairs"]) # 9

    # Grid: 5 rows × n cols
    #   r0: query frames           (n cells)
    #   r1: cycle heatmap overlays (n_pairs cells, last empty)
    #   r2: k-NN D_map overlays    (n cells)
    #   r3: top-3 refs (concat into 3 mini-thumbs per cell)
    #   r4: summary scores bar     (spans all n cells)
    fig = plt.figure(figsize=(22, 13), constrained_layout=True)
    gs = fig.add_gridspec(5, n, height_ratios=[1.0, 1.0, 1.0, 0.7, 0.45])

    # ── Title bar ───────────────────────────────────────────────────
    fig.suptitle(
        f"{header}    task {task_short} {catch_row['video']}    {task_desc_short}\n"
        f"ratio_cycle = {ratio_cycle:.3f}    "
        f"ratio_knn = {ratio_knn:.3f}    "
        f"ratio_fused = {ratio_fused:.3f}    →    {verdict_fused}",
        fontsize=14, fontweight="bold", color=header_color,
    )

    # ── Row 0: 10 query frames with H_cycle (pair) + H_knn (frame) ────
    H_knns = data["knn_H_frames"]
    H_cycs = data["cycle_H_pairs"]
    for i, bgr in enumerate(data["seg_bgrs"]):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        # H_pair label is between frame i-i+1 (so apply to first 9 columns)
        cyc_lbl = f"H_pair={H_cycs[i]:.2f}" if i < n_pairs else ""
        knn_lbl = f"H_knn={H_knns[i]:.2f}"
        ax.set_title(f"f{i}\n{cyc_lbl}\n{knn_lbl}", fontsize=7)
        ax.axis("off")

    # ── Row 1: cycle drift heatmaps (TURBO + cert mask + HALLU box) ──
    for t, pd in enumerate(data["cycle_pairs"]):
        ax = fig.add_subplot(gs[1, t])
        overlay = overlay_heatmap_on_frame(
            data["seg_bgrs"][t], pd["drift_map"],
            cert=pd["cert_fwd"], cert_floor=0.1,
            alpha=0.85, percentile_clip=95.0, annotate_top_cluster=True,
        )
        ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax.set_title(f"cycle {t}→{t+1}\npeak={pd['peak']:.1f}px", fontsize=7)
        ax.axis("off")
    # Last cell: empty placeholder
    ax = fig.add_subplot(gs[1, n - 1])
    ax.text(0.5, 0.5, "—", ha="center", va="center", fontsize=20, color="gray",
            transform=ax.transAxes)
    ax.axis("off")
    # Y-axis row label (annotation on first cycle cell)
    fig.text(0.005, 0.71, "CYCLE", rotation=90, fontsize=10, fontweight="bold",
             color="#cc0000", va="center")

    # ── Row 2: k-NN D_map heatmap overlays ─────────────────────────────
    for i, kf in enumerate(data["knn_frames"]):
        ax = fig.add_subplot(gs[2, i])
        overlay = render_knn_overlay(
            data["seg_bgrs"][i], kf["D_map"], kf["fg_mask"],
            alpha=0.75, percentile_clip=95.0,
        )
        ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        # Show routed signal value
        if kf["route"] == "peak":
            label = f"D_map\npeak={kf['peak']:.1f}"
        else:
            label = f"D_map\nivar={kf['ivar']:.1f}"
        ax.set_title(label, fontsize=7)
        ax.axis("off")
    fig.text(0.005, 0.49, "k-NN", rotation=90, fontsize=10, fontweight="bold",
             color="#cc7700", va="center")

    # ── Row 3: top-3 refs per query — concatenate horizontally ─────────
    for i, kf in enumerate(data["knn_frames"]):
        ax = fig.add_subplot(gs[3, i])
        # stack top-3 ref thumbs side by side
        thumbs = []
        for b in kf["top3_bgrs"]:
            if b is None:
                continue
            r = cv2.resize(b, (128, 128), interpolation=cv2.INTER_AREA)
            thumbs.append(r)
        if thumbs:
            strip = np.concatenate(thumbs, axis=1)
            ax.imshow(cv2.cvtColor(strip, cv2.COLOR_BGR2RGB))
        ax.set_title(f"top-3 refs (f{i})", fontsize=6)
        ax.axis("off")
    fig.text(0.005, 0.255, "TOP-3 REFS", rotation=90, fontsize=9, fontweight="bold",
             color="#444444", va="center")

    # ── Row 4: side-by-side H_cycle vs H_knn comparison bar ────────────
    ax = fig.add_subplot(gs[4, :])
    x = np.arange(n)
    width = 0.4
    cyc_vals = list(H_cycs) + [0.0]   # pad cycle to n
    knn_vals = list(H_knns)
    ax.bar(x - width / 2, cyc_vals[:n], width, label="H_cycle (per pair)",
           color="#cc0000", alpha=0.8)
    ax.bar(x + width / 2, knn_vals, width, label="H_knn (per frame)",
           color="#cc7700", alpha=0.8)
    ax.axhline(0.5, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"f{i}" for i in range(n)], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("H ∈ [0,1]", fontsize=8)
    ax.set_title("Per-frame anomaly score: CYCLE vs k-NN side-by-side", fontsize=9)
    ax.legend(loc="upper right", fontsize=7)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    VIZ_OUT.mkdir(parents=True, exist_ok=True)

    cycle_only, knn_only = load_catches()
    print(f"cycle-only catches: {len(cycle_only)}")
    print(f"knn-only catches:   {len(knn_only)}")

    print("\nLoading models …")
    from warp_score.temporal_signals import CycleSignal
    from warp_score.matcher import RoMaMatcher
    from warp_score.sam_segmenter import VideoFrameSegmenter
    from warp_score.knn_signal import KNNFrameSignal

    matcher = RoMaMatcher(setting="turbo", device="cuda", use_precision=True, vis_size=224)
    matcher._load_model()
    sam = VideoFrameSegmenter()
    cycle_sig = CycleSignal(cert_floor=0.1)
    knn = KNNFrameSignal(matcher, k=KNN_K)

    # ── Per-task state cache ────────────────────────────────────────────
    needed_tasks = sorted({r["task"] for r in (cycle_only + knn_only)})
    states: dict[str, dict] = {}
    for task_short in needed_tasks:
        print(f"\n[task {task_short}] building cycle null + k-NN pool …")
        states[task_short] = build_per_task_state(task_short, matcher, sam, cycle_sig, knn)
        print(f"  cycle null: {len(states[task_short]['null_cycle_mean'])} pairs")
        print(f"  k-NN pool:  {len(states[task_short]['pool']['paths'])} refs, "
              f"route={states[task_short]['null_knn']['route']}")

    # ── Render each catch ───────────────────────────────────────────────
    catches_meta = ([("cycle_only", r) for r in cycle_only]
                    + [("knn_only", r) for r in knn_only])
    out_files: list[tuple[str, str, Path]] = []   # (group, label, png path)
    for group, row in catches_meta:
        task_short = row["task"]
        task_full = TASK_FULL[task_short]
        vid = row["video"]
        gen_mp4 = GEN_ROOT / task_full / vid
        if not gen_mp4.exists():
            print(f"  [skip] missing {gen_mp4}")
            continue
        out_png = VIZ_OUT / group / f"task_{task_short}__{vid.replace('.mp4','')}_viz.png"
        print(f"\n>>> {group}  task {task_short} {vid} …")
        data = process_video(gen_mp4, states[task_short], matcher, sam, cycle_sig, knn)
        if data is None:
            print(f"  [skip] could not sample frames")
            continue
        render_catch(row, group, states[task_short], data, out_png)
        print(f"    → {out_png}")
        out_files.append((group, f"task_{task_short}__{vid}", out_png))

    # ── Upload to HF ────────────────────────────────────────────────────
    print("\nUploading to HF …")
    from huggingface_hub import HfApi
    token = (Path.home() / ".cache" / "huggingface" / "token").read_text().strip()
    api = HfApi(token=token)
    hf_urls: list[str] = []
    for group, label, png in out_files:
        dst = f"{HF_PREFIX}/{group}/{png.name}"
        api.upload_file(
            path_or_fileobj=str(png),
            path_in_repo=dst,
            repo_id=HF_REPO,
            repo_type="dataset",
        )
        url = f"https://huggingface.co/datasets/{HF_REPO}/blob/main/{dst}"
        hf_urls.append(url)
        print(f"    → {url}")

    print(f"\n=== Done: {len(out_files)} viz uploaded ===")
    print(f"Tree: https://huggingface.co/datasets/{HF_REPO}/tree/main/{HF_PREFIX}")


if __name__ == "__main__":
    main()
