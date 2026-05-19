#!/usr/bin/env python3
"""DROID multi-view per-catch visualization.

Reads `per_task_ratio_table.csv` (output of compute_droid_multiview_ratio.py)
and renders one PNG per catch, grouped into 3 buckets:

  * multiview_HALLU       ratio_multiview > 1.0
  * single_view_only      exactly one fused_<view> > 1.0 (edge case)
  * multiview_borderline  ratio_multiview in [0.95, 1.05)

Each PNG is a 3-row × 5-col grid (3 views × 5 sampled frames):
  * each cell: query frame + cycle/k-NN heatmap overlay (TURBO, alpha 0.6)
  * below each view row: bar chart of H_cycle_<view> + H_knn_<view> per frame
  * right-side text margin: ratio_<view>_fused, ratio_multiview, verdict

Output:
    paper-physical-droid/viz/multiview_hallu/<task>_<vid>.png
    paper-physical-droid/viz/single_view_only/<task>_<vid>.png
    paper-physical-droid/viz/multiview_borderline/<task>_<vid>.png

Re-uses the pipeline machinery + TURBO overlay style from
scripts/viz_complementarity.py (the GR-1 viz). Models are loaded once and
per-(task, view) state is cached across all catches.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
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

REPO_ROOT = Path("/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination")
BENCH = REPO_ROOT / "paper-physical-droid"
REF_ROOT = BENCH / "reference"
GEN_ROOT = BENCH / "generated"
RAW_VIDEO_ROOT = BENCH / "raw_videos" / "droid"
EVAL_DIR = BENCH / "per_task_dense_eval"
NULL_DIR = BENCH / "null_per_task"
DINO_CACHE_DIR = BENCH / "ref_cache"
VIZ_OUT = BENCH / "viz"

DEFAULT_VIEWS = ["exterior_1", "exterior_2", "wrist"]
N_VIZ_FRAMES = 5     # sampled query frames per view for the grid
KNN_K = 15


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (frame I/O + overlays — same style as viz_complementarity.py)
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


def cauchy_combine_local(ps: list[float]) -> float:
    ps = [p for p in ps if p is not None and 0.0 < p < 1.0]
    if not ps:
        return 0.5
    t = float(np.mean([np.tan(np.pi * (0.5 - p)) for p in ps]))
    return float(0.5 - np.arctan(t) / np.pi)


def overlay_heatmap_on_frame(
    frame_bgr: np.ndarray,
    heatmap: np.ndarray,
    cert: np.ndarray | None = None,
    cert_floor: float = 0.1,
    alpha: float = 0.6,
    percentile_clip: float = 95.0,
) -> np.ndarray:
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
    return (frame_r * (1 - alpha_map) + colored * alpha_map).astype(np.uint8)


def render_knn_overlay(
    frame_bgr: np.ndarray,
    D_map: np.ndarray,
    fg_mask: np.ndarray,
    alpha: float = 0.6,
    percentile_clip: float = 95.0,
) -> np.ndarray:
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
    return (frame_r * (1 - alpha_map) + colored * alpha_map).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Catch selection
# ─────────────────────────────────────────────────────────────────────────────


def _f(r: dict, k: str, default=float("nan")) -> float:
    v = r.get(k, "")
    if v in ("", None, "nan"):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load_catches(views: list[str]) -> dict[str, list[dict]]:
    rows = list(csv.DictReader(open(EVAL_DIR / "per_task_ratio_table.csv")))
    gens = [r for r in rows if r["type"] == "GEN"]

    multiview_hallu: list[dict] = []
    single_view_only: list[dict] = []
    multiview_borderline: list[dict] = []

    for r in gens:
        rmv = _f(r, "ratio_multiview")
        rfused = {v: _f(r, f"ratio_fused_{v}") for v in views}
        flags = [v for v in views
                 if not math.isnan(rfused[v]) and rfused[v] > 1.0]
        if not math.isnan(rmv) and rmv > 1.0:
            multiview_hallu.append(r)
        if len(flags) == 1:
            single_view_only.append(r)
        if not math.isnan(rmv) and 0.95 <= rmv < 1.05:
            multiview_borderline.append(r)

    return {
        "multiview_hallu": multiview_hallu,
        "single_view_only": single_view_only,
        "multiview_borderline": multiview_borderline,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-(task, view) state (cycle null + k-NN pool/null)
# ─────────────────────────────────────────────────────────────────────────────


def build_view_state(
    task_short: str,
    task_full: str,
    view: str,
    knn,
) -> dict | None:
    npz_path = NULL_DIR / f"{task_short}__{view}.npz"
    if not npz_path.exists():
        return None
    npz = np.load(npz_path, allow_pickle=True)

    null_cycle_mean = np.asarray(npz["cycle_cycle_mean"], dtype=np.float32)
    null_cycle_peak = np.asarray(npz["cycle_cycle_peak"], dtype=np.float32)
    state: dict = {
        "task_short": task_short,
        "task_full": task_full,
        "view": view,
        "null_cycle_mean": np.sort(null_cycle_mean),
        "null_cycle_peak": np.sort(null_cycle_peak),
    }

    if "knn_null_ivar" in npz.files:
        ref_dir = REF_ROOT / task_full / view
        pngs = sorted(ref_dir.glob("frame_*.png"))
        if pngs:
            pool_key = f"{task_full}__{view}"
            pool = knn.build_pool(pool_key, pngs, DINO_CACHE_DIR)
            state["pool"] = pool
            state["null_knn"] = {
                "null_ivar": np.sort(np.asarray(npz["knn_null_ivar"], dtype=np.float32)),
                "null_peak": np.sort(np.asarray(npz["knn_null_peak"], dtype=np.float32)),
                "route": str(npz["knn_route"]) if "knn_route" in npz.files else "ivar",
            }
            state["ref_pngs"] = pngs
        else:
            state["pool"] = None
            state["null_knn"] = None
    else:
        state["pool"] = None
        state["null_knn"] = None

    return state


# ─────────────────────────────────────────────────────────────────────────────
# Run pipeline on one (mp4, state) — returns frames + cycle pair maps + k-NN
# ─────────────────────────────────────────────────────────────────────────────


def process_one_view_video(
    mp4: Path,
    state: dict,
    matcher,
    seg,
    cycle_sig,
    knn,
) -> dict | None:
    from warp_score.temporal_signals import empirical_p_value
    from warp_score.knn_signal import fg_mask_at_size
    from warp_score.statistics import MahalanobisStatistics

    bgrs = sample_frames(mp4, N_VIZ_FRAMES)
    if len(bgrs) < 2:
        return None
    seg_bgrs = [seg.segment_frame(b) for b in bgrs]
    tmp = Path(tempfile.mkdtemp(prefix="droid_mvviz_"))
    seg_pngs: list[Path] = []
    for i, b in enumerate(seg_bgrs):
        p = tmp / f"f_{i:04d}.png"
        cv2.imwrite(str(p), b)
        seg_pngs.append(p)

    null_m = state["null_cycle_mean"]
    null_p = state["null_cycle_peak"]

    # Cycle pairs (between consecutive sampled frames)
    cycle_pairs: list[dict] = []
    cycle_H: list[float] = []
    for t in range(len(seg_pngs) - 1):
        fwd = matcher.match(seg_pngs[t], seg_pngs[t + 1])
        bwd = matcher.match(seg_pngs[t + 1], seg_pngs[t])
        sig = cycle_sig.compute(fwd, bwd)
        p_m = empirical_p_value(sig.mean, null_m)
        p_p = empirical_p_value(sig.peak, null_p)
        H_pair = 1.0 - cauchy_combine_local([p_m, p_p])
        cycle_pairs.append({
            "drift_map": sig.pixel_map,
            "cert_fwd": fwd.cert,
            "peak": float(sig.peak),
            "H_pair": float(H_pair),
        })
        cycle_H.append(H_pair)

    # k-NN per frame
    knn_frames: list[dict] = []
    knn_H: list[float] = []
    if knn is not None and state.get("pool") is not None and state.get("null_knn") is not None:
        pool = state["pool"]
        null_knn = state["null_knn"]
        for q_png in seg_pngs:
            q_feat = knn.dino.extract([q_png])[0]
            top_k_idx = knn.selector.select_for_query(q_feat, pool["feats"], knn.k)
            k_refs = [pool["paths"][j] for j in top_k_idx]
            fg = fg_mask_at_size(q_png, matcher.vis_size)
            match_results = matcher.match_batch(q_png, k_refs, fg_mask=fg)
            warps = np.stack([m.warp for m in match_results], axis=0)
            precisions = np.stack([m.precision for m in match_results], axis=0)
            D_map, _, _ = MahalanobisStatistics.ivar_per_pixel(warps, precisions)
            ivar = MahalanobisStatistics.interior_mean(D_map, fg)
            peak = MahalanobisStatistics.peak_max_z(D_map, fg)
            p_ivar = empirical_p_value(ivar, null_knn["null_ivar"])
            p_peak = empirical_p_value(peak, null_knn["null_peak"])
            p_routed = p_peak if null_knn["route"] == "peak" else p_ivar
            H_frame = 1.0 - p_routed
            knn_frames.append({
                "D_map": D_map,
                "fg_mask": fg,
                "ivar": float(ivar),
                "peak": float(peak),
                "H_frame": float(H_frame),
                "route": null_knn["route"],
            })
            knn_H.append(H_frame)

    for p in seg_pngs:
        p.unlink(missing_ok=True)
    tmp.rmdir()

    return {
        "seg_bgrs": seg_bgrs,
        "cycle_pairs": cycle_pairs,
        "cycle_H": cycle_H,
        "knn_frames": knn_frames,
        "knn_H": knn_H,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Render one catch (3 views × 5 frames)
# ─────────────────────────────────────────────────────────────────────────────


def render_catch(
    catch_row: dict,
    group: str,
    views: list[str],
    per_view_data: dict[str, dict],
    out_path: Path,
) -> None:
    n = N_VIZ_FRAMES
    n_views = len(views)

    # Layout — for each view: 2 rows (cycle, k-NN overlays) + 1 bar row → 3 rows
    # Grid total: (3 * n_views) rows × n cols + 1 right margin col for text
    rows_per_view = 3
    total_rows = rows_per_view * n_views
    fig = plt.figure(
        figsize=(3.4 * n + 5.0, 2.5 * total_rows),
        constrained_layout=True,
    )
    gs = fig.add_gridspec(total_rows, n + 1, width_ratios=[1.0] * n + [1.3])

    # Header
    task = catch_row["task"]
    vid = catch_row["video"]
    rmv = _f(catch_row, "ratio_multiview")
    verd = catch_row.get("verdict_multiview", "?")
    if group == "multiview_hallu":
        title_color = "#cc0000"
        title_pfx = "MULTI-VIEW HALLU"
    elif group == "single_view_only":
        title_color = "#cc7700"
        title_pfx = "SINGLE-VIEW-ONLY CATCH"
    else:
        title_color = "#888800"
        title_pfx = "MULTI-VIEW BORDERLINE"

    fig.suptitle(
        f"{title_pfx}    task {task}  {vid}    "
        f"ratio_multiview = {rmv:.3f}  →  {verd}",
        fontsize=15, fontweight="bold", color=title_color,
    )

    for vi, view in enumerate(views):
        data = per_view_data.get(view)
        row_cyc = vi * rows_per_view + 0
        row_knn = vi * rows_per_view + 1
        row_bar = vi * rows_per_view + 2

        ratio_fused = _f(catch_row, f"ratio_fused_{view}")
        ratio_cycle = _f(catch_row, f"ratio_cycle_{view}")
        ratio_knn = _f(catch_row, f"ratio_knn_{view}")

        if data is None:
            # All-empty rows + an info cell
            for ri in range(rows_per_view):
                for ci in range(n):
                    ax = fig.add_subplot(gs[ri + vi * rows_per_view, ci])
                    ax.text(0.5, 0.5, f"{view}\n(no data)",
                            ha="center", va="center", fontsize=10, color="gray",
                            transform=ax.transAxes)
                    ax.axis("off")
            ax_info = fig.add_subplot(gs[vi * rows_per_view : vi * rows_per_view + rows_per_view, n])
            ax_info.text(
                0.02, 0.5,
                f"View: {view}\n  ratio_fused = nan\n  ratio_cycle = nan\n  ratio_knn = nan",
                fontsize=11, va="center", ha="left", transform=ax_info.transAxes,
            )
            ax_info.axis("off")
            continue

        # Cycle row
        cycle_pairs = data["cycle_pairs"]
        for ci in range(n):
            ax = fig.add_subplot(gs[row_cyc, ci])
            if ci < len(cycle_pairs):
                cp = cycle_pairs[ci]
                overlay = overlay_heatmap_on_frame(
                    data["seg_bgrs"][ci], cp["drift_map"],
                    cert=cp["cert_fwd"], cert_floor=0.1, alpha=0.6,
                )
                ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                ax.set_title(f"cycle {ci}→{ci+1}\npeak={cp['peak']:.1f}", fontsize=7)
            else:
                # Last column for cycle has no pair — show plain frame
                ax.imshow(cv2.cvtColor(data["seg_bgrs"][ci], cv2.COLOR_BGR2RGB))
                ax.set_title(f"f{ci}", fontsize=7)
            ax.axis("off")

        # k-NN row
        knn_frames = data["knn_frames"]
        for ci in range(n):
            ax = fig.add_subplot(gs[row_knn, ci])
            if ci < len(knn_frames):
                kf = knn_frames[ci]
                overlay = render_knn_overlay(
                    data["seg_bgrs"][ci], kf["D_map"], kf["fg_mask"], alpha=0.6,
                )
                ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                if kf["route"] == "peak":
                    ax.set_title(f"k-NN f{ci}\npeak={kf['peak']:.1f}", fontsize=7)
                else:
                    ax.set_title(f"k-NN f{ci}\nivar={kf['ivar']:.1f}", fontsize=7)
            else:
                ax.imshow(cv2.cvtColor(data["seg_bgrs"][ci], cv2.COLOR_BGR2RGB))
                ax.set_title(f"f{ci}", fontsize=7)
            ax.axis("off")

        # Bar row (spans n cells)
        ax_bar = fig.add_subplot(gs[row_bar, :n])
        x = np.arange(n)
        width = 0.4
        cyc_vals = list(data["cycle_H"]) + [0.0] * (n - len(data["cycle_H"]))
        knn_vals = list(data["knn_H"])  + [0.0] * (n - len(data["knn_H"]))
        ax_bar.bar(x - width / 2, cyc_vals[:n], width, label="H_cycle (pair)",
                   color="#cc0000", alpha=0.8)
        ax_bar.bar(x + width / 2, knn_vals[:n], width, label="H_knn (frame)",
                   color="#cc7700", alpha=0.8)
        ax_bar.axhline(0.5, color="gray", linewidth=0.5, linestyle="--")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels([f"f{i}" for i in range(n)], fontsize=8)
        ax_bar.set_ylim(0, 1)
        ax_bar.set_ylabel("H ∈ [0,1]", fontsize=8)
        ax_bar.set_title(f"view = {view}", fontsize=10, fontweight="bold")
        ax_bar.legend(loc="upper right", fontsize=7)

        # Right info column (spans the 3 rows for this view)
        ax_info = fig.add_subplot(gs[vi * rows_per_view : vi * rows_per_view + rows_per_view, n])
        info_lines = [
            f"View: {view}",
            f"  ratio_fused = {ratio_fused:.3f}",
            f"  ratio_cycle = {ratio_cycle:.3f}",
            f"  ratio_knn   = {ratio_knn:.3f}",
        ]
        verd_v = catch_row.get(f"verdict_fused_{view}", "?")
        info_lines.append(f"  verdict_fused: {verd_v}")
        ax_info.text(
            0.02, 0.5, "\n".join(info_lines),
            fontsize=11, va="center", ha="left", transform=ax_info.transAxes,
        )
        ax_info.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def _gen_view_mp4(gen_mp4: Path, view: str) -> Path:
    cand = gen_mp4.parent / f"{gen_mp4.stem}_views" / f"{view}.mp4"
    return cand if cand.exists() else gen_mp4


def main() -> None:
    ap = argparse.ArgumentParser(
        description="DROID multi-view per-catch visualization (cycle + k-NN, 3 views)."
    )
    ap.add_argument("--views", nargs="+", default=DEFAULT_VIEWS,
                    help="views to render (default: 3 DROID views)")
    ap.add_argument("--max_per_group", type=int, default=None,
                    help="limit number of catches per group (default: no limit)")
    ap.add_argument("--groups", nargs="+",
                    default=["multiview_hallu", "single_view_only", "multiview_borderline"])
    ap.add_argument("--out_root", type=Path, default=VIZ_OUT)
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)

    catches = load_catches(args.views)
    print("Catches:")
    for g in args.groups:
        print(f"  {g}: {len(catches.get(g, []))}")

    # ── Load models once ───────────────────────────────────────────────
    print("\nLoading models …")
    from warp_score.temporal_signals import CycleSignal
    from warp_score.matcher import RoMaMatcher
    from warp_score.sam_segmenter import VideoFrameSegmenter
    from warp_score.knn_signal import KNNFrameSignal

    matcher = RoMaMatcher(setting="turbo", device="cuda", use_precision=True, vis_size=224)
    matcher._load_model()
    seg = VideoFrameSegmenter()
    cycle_sig = CycleSignal(cert_floor=0.1)
    knn = KNNFrameSignal(matcher, k=KNN_K)

    # ── eval_tasks.json for task_full lookup ───────────────────────────
    eval_tasks = json.loads((BENCH / "eval_tasks.json").read_text())

    # ── Build per-(task, view) state lazily, cache across catches ───────
    state_cache: dict[tuple[str, str], dict | None] = {}

    def get_state(task_short: str, view: str) -> dict | None:
        key = (task_short, view)
        if key not in state_cache:
            task_meta = eval_tasks.get(task_short)
            if task_meta is None:
                state_cache[key] = None
            else:
                state_cache[key] = build_view_state(task_short, task_meta["task_full"], view, knn)
        return state_cache[key]

    # ── Iterate groups + catches ───────────────────────────────────────
    out_files: list[Path] = []
    for group in args.groups:
        group_catches = catches.get(group, [])
        if args.max_per_group is not None:
            group_catches = group_catches[: args.max_per_group]
        if not group_catches:
            print(f"\n[{group}] no catches → skip")
            continue
        print(f"\n[{group}] rendering {len(group_catches)} catches …")
        for row in group_catches:
            task_short = row["task"]
            vid = row["video"]
            task_meta = eval_tasks.get(task_short)
            if task_meta is None:
                print(f"  [skip] task {task_short} missing from eval_tasks.json")
                continue
            task_full = task_meta["task_full"]
            gen_mp4 = GEN_ROOT / task_full / vid
            if not gen_mp4.exists():
                # Maybe row is REAL (shouldn't happen for gens, but defensive)
                print(f"  [skip] gen mp4 missing: {gen_mp4}")
                continue

            print(f"  >>> {group}  task {task_short}  {vid}")
            per_view_data: dict[str, dict] = {}
            for view in args.views:
                state = get_state(task_short, view)
                if state is None:
                    print(f"    [view {view}] no state (null/refs missing)")
                    per_view_data[view] = None
                    continue
                view_mp4 = _gen_view_mp4(gen_mp4, view)
                if not view_mp4.exists():
                    print(f"    [view {view}] mp4 missing")
                    per_view_data[view] = None
                    continue
                data = process_one_view_video(view_mp4, state, matcher, seg, cycle_sig, knn)
                if data is None:
                    print(f"    [view {view}] could not sample frames")
                per_view_data[view] = data

            out_png = args.out_root / group / f"task_{task_short}__{Path(vid).stem}.png"
            render_catch(row, group, args.views, per_view_data, out_png)
            print(f"    → {out_png}")
            out_files.append(out_png)

    print(f"\n=== Done: {len(out_files)} viz written under {args.out_root} ===")


if __name__ == "__main__":
    main()
