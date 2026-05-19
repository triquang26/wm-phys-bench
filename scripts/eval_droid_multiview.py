#!/usr/bin/env python3
"""DROID multi-view WarpDyn evaluation.

For each (task, view) pair we run the SAME single-view pipeline as
`eval_per_task_dense_null.py` (multi-lag cycle null + DINOv2 k-NN LOO null +
Cauchy(cycle, knn) at the video level) independently. Then per-task we
cross-view fuse the three per-view fused H values via a second Cauchy combine
across views to obtain a single `multiview_fused_peak` per video.

DROID layout (see scripts/setup_droid_multiview.py):

    paper-physical-droid/
    ├── eval_tasks.json
    ├── raw_videos/droid/<task_short>/{exterior_1,exterior_2,wrist}.mp4
    ├── reference/<task_full>/<view>/frame_NNNN.png   (SAM3 refs, 1 dir/view)
    └── generated/<task_full>/
        ├── v0000.mp4                  Cosmos 2x2 composite (raw)
        └── v0000_views/<view>.mp4     per-view demuxed (run cosmos_demux_views.py)

For each gen we look for `<gen>_views/<view>.mp4`; if missing we fall back to
the original composite and log a warning (the per-view scores will be
contaminated by the other panels in that case).

Output:
    paper-physical-droid/per_task_dense_eval/per_task_dense_table.csv

Columns:
    task, video, type, label, time_sec
    # per-view raw H scores (one block per view)
    cycle_<view>_peak, cycle_<view>_robust,
    precision_anomaly_<view>_peak, precision_anomaly_<view>_robust,
    knn_<view>_peak, knn_<view>_robust,
    knn_<view>_route, knn_<view>_mean_ivar, knn_<view>_mean_peak_raw,
    fused_<view>_peak, fused_<view>_robust,
    # cross-view multiview fusion
    multiview_fused_peak, multiview_fused_robust,
    # nulls used (one per view)
    null_n_<view>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

REPO_ROOT = Path("/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination")

# ── Pipeline constants — match single-view GR-1 production ──────────────────
NULL_LAGS = [1, 2, 5, 10]
N_TEST_FRAMES = 10
KNN_K = 15
DEFAULT_VIEWS = ["exterior_1", "exterior_2", "wrist"]


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
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


def resolve_view_video(gen_mp4: Path, view: str) -> tuple[Path, bool]:
    """Return (path-to-use, is_demuxed). Falls back to composite if missing."""
    demux_dir = gen_mp4.parent / f"{gen_mp4.stem}_views"
    candidate = demux_dir / f"{view}.mp4"
    if candidate.exists():
        return candidate, True
    return gen_mp4, False


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="DROID multi-view WarpDyn eval (per-view + cross-view Cauchy)."
    )
    ap.add_argument("--bench", default="paper-physical-droid",
                    help="bench dir name under REPO_ROOT (default: paper-physical-droid)")
    ap.add_argument("--tasks_json", default="eval_tasks.json",
                    help="path to eval_tasks.json, resolved relative to bench dir "
                         "(default: eval_tasks.json)")
    ap.add_argument("--raw_video_subdir", default="raw_videos/droid",
                    help="subdir under bench/ holding <task_short>/<view>.mp4 "
                         "(default: raw_videos/droid)")
    ap.add_argument("--views", nargs="+", default=DEFAULT_VIEWS,
                    help=f"views to evaluate (default: {' '.join(DEFAULT_VIEWS)})")
    ap.add_argument("--use_knn", action=argparse.BooleanOptionalAction, default=True,
                    help="enable k-NN signal + Cauchy fusion (default True)")
    ap.add_argument("--limit_tasks", type=int, default=None,
                    help="for debug: only process first N tasks")
    ap.add_argument("--out_suffix", default="",
                    help="append suffix to CSV filename")
    args = ap.parse_args()

    BENCH = REPO_ROOT / args.bench
    REF_ROOT = BENCH / "reference"
    RAW_VIDEO_ROOT = BENCH / args.raw_video_subdir
    GEN_ROOT = BENCH / "generated"
    OUT = BENCH / "per_task_dense_eval"
    NULL_DIR = BENCH / "null_per_task"
    DINO_CACHE_DIR = BENCH / "ref_cache"

    tasks_json_path = BENCH / args.tasks_json
    if not tasks_json_path.exists():
        sys.exit(f"tasks_json not found: {tasks_json_path}")
    eval_tasks_raw = json.loads(tasks_json_path.read_text())

    def _sort_key(kv):
        try:
            return (0, int(kv[0]))
        except ValueError:
            return (1, kv[0])

    eval_tasks = dict(sorted(eval_tasks_raw.items(), key=_sort_key))
    if args.limit_tasks is not None:
        eval_tasks = dict(list(eval_tasks.items())[: args.limit_tasks])

    print(f"bench         : {BENCH}")
    print(f"ref_root      : {REF_ROOT}")
    print(f"raw_video_root: {RAW_VIDEO_ROOT}")
    print(f"gen_root      : {GEN_ROOT}")
    print(f"tasks_json    : {tasks_json_path}")
    print(f"tasks         : {len(eval_tasks)}")
    print(f"views         : {args.views}")

    OUT.mkdir(parents=True, exist_ok=True)
    NULL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Heavy imports + single-load of models ──────────────────────────────
    from warp_score.temporal_signals import (
        CycleSignal, PrecisionAnomalySignal, empirical_p_value,
    )
    from warp_score.matcher import RoMaMatcher
    from warp_score.sam_segmenter import VideoFrameSegmenter
    from warp_score.fusion import cauchy_combine_video, cauchy_combine

    print(f"\nLoading models  (use_knn={args.use_knn}) …")
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
        print(f"k-NN signal: k={KNN_K}, DINOv2 ViT-S/14")

    # ─────────────────────────────────────────────────────────────────────
    # Per-pair / per-frame scoring closures (matcher captured from above)
    # ─────────────────────────────────────────────────────────────────────

    def score_pair_cycle(p_a: Path, p_b: Path) -> dict:
        fwd = matcher.match(p_a, p_b)
        bwd = matcher.match(p_b, p_a)
        return {s.name: s.compute(fwd, bwd) for s in signals}

    def video_pair_scores(seg_pngs: list[Path]) -> list[dict]:
        out = []
        for t in range(len(seg_pngs) - 1):
            sigs = score_pair_cycle(seg_pngs[t], seg_pngs[t + 1])
            out.append({n: {"mean": sigs[n].mean, "peak": sigs[n].peak} for n in sig_names})
        return out

    def video_knn_scores(seg_pngs: list[Path], pool: dict, null_knn: dict) -> list[dict]:
        from warp_score.knn_signal import fg_mask_at_size
        out = []
        for p in seg_pngs:
            fg = fg_mask_at_size(p, matcher.vis_size)
            res = knn.score_frame(p, pool, null_knn, query_fg_mask=fg)
            out.append(res)
        return out

    # ─────────────────────────────────────────────────────────────────────
    # Score one (view, video) pair → flat dict of cycle_*, knn_*, fused_*
    # ─────────────────────────────────────────────────────────────────────

    def score_view_video(
        mp4: Path,
        sorted_null_cycle: dict,
        pool: dict | None,
        null_knn: dict | None,
    ) -> dict | None:
        bgrs = sample_frames(mp4, N_TEST_FRAMES)
        if len(bgrs) < 2:
            return None
        seg_bgrs = [seg.segment_frame(b) for b in bgrs]
        tmp_dir = Path(tempfile.mkdtemp(prefix="droid_mv_"))
        seg_pngs: list[Path] = []
        for i, b in enumerate(seg_bgrs):
            p = tmp_dir / f"f_{i:04d}.png"
            cv2.imwrite(str(p), b)
            seg_pngs.append(p)

        view_row: dict = {}
        try:
            # ── Cycle pair signals ────────────────────────────────────
            pairs = video_pair_scores(seg_pngs)
            for n in sig_names:
                h_pairs: list[float] = []
                for pp in pairs:
                    p_m = empirical_p_value(pp[n]["mean"], sorted_null_cycle[n]["mean"])
                    p_p = empirical_p_value(pp[n]["peak"], sorted_null_cycle[n]["peak"])
                    h_pairs.append(1.0 - cauchy_combine_local([p_m, p_p]))
                if h_pairs:
                    view_row[f"{n}_peak"] = float(np.percentile(h_pairs, 80))
                    if len(h_pairs) > 2:
                        trimmed = np.sort(h_pairs)[len(h_pairs)//10:-max(1, len(h_pairs)//10)]
                        view_row[f"{n}_robust"] = float(trimmed.mean())
                    else:
                        view_row[f"{n}_robust"] = float(np.mean(h_pairs))
                else:
                    view_row[f"{n}_peak"] = 0.0
                    view_row[f"{n}_robust"] = 0.0

            # ── k-NN frame signals ────────────────────────────────────
            if knn is not None and pool is not None and null_knn is not None:
                knn_frames = video_knn_scores(seg_pngs, pool, null_knn)
                h_knns = [r["H"] for r in knn_frames]
                view_row["knn_peak"] = float(np.percentile(h_knns, 80)) if h_knns else 0.0
                if len(h_knns) > 2:
                    trimmed = np.sort(h_knns)[len(h_knns)//10:-max(1, len(h_knns)//10)]
                    view_row["knn_robust"] = float(trimmed.mean())
                elif h_knns:
                    view_row["knn_robust"] = float(np.mean(h_knns))
                else:
                    view_row["knn_robust"] = 0.0
                view_row["knn_mean_ivar"] = float(np.mean([r["ivar"] for r in knn_frames]))
                view_row["knn_mean_peak_raw"] = float(np.mean([r["peak"] for r in knn_frames]))
                view_row["knn_route"] = null_knn["route"]
                # Within-view fusion: cycle + knn
                view_row["fused_peak"] = cauchy_combine_video(view_row["cycle_peak"],
                                                              view_row["knn_peak"])
                view_row["fused_robust"] = cauchy_combine_video(view_row["cycle_robust"],
                                                                view_row["knn_robust"])
        finally:
            for p in seg_pngs:
                p.unlink(missing_ok=True)
            tmp_dir.rmdir()
        return view_row

    # ─────────────────────────────────────────────────────────────────────
    # Build per-(task, view) null calibration — cache to disk if absent
    # ─────────────────────────────────────────────────────────────────────

    def build_view_null(task_short: str, task_full: str, view: str) -> dict | None:
        ref_dir = REF_ROOT / task_full / view
        if not ref_dir.exists():
            print(f"    [view {view}] ref dir missing: {ref_dir}")
            return None
        pngs = sorted(ref_dir.glob("frame_*.png"))
        if len(pngs) < 20:
            print(f"    [view {view}] too few refs ({len(pngs)})")
            return None

        print(f"    [view {view}] refs: {len(pngs)}")

        # ── Cycle null ────────────────────────────────────────────────
        print(f"    [view {view}] building cycle null …")
        null = {n: {"mean": [], "peak": []} for n in sig_names}
        t0 = time.time()
        for lag in NULL_LAGS:
            n_pairs = max(0, len(pngs) - lag)
            for i in range(n_pairs):
                sigs = score_pair_cycle(pngs[i], pngs[i + lag])
                for n in sig_names:
                    null[n]["mean"].append(sigs[n].mean)
                    null[n]["peak"].append(sigs[n].peak)
            print(f"      lag {lag:>2}: +{n_pairs} pairs  total {len(null[sig_names[0]]['mean'])}  "
                  f"({time.time()-t0:.0f}s)")

        sorted_null = {
            n: {
                "mean": np.sort(np.asarray(null[n]["mean"], dtype=np.float32)),
                "peak": np.sort(np.asarray(null[n]["peak"], dtype=np.float32)),
            } for n in sig_names
        }

        # ── k-NN pool + LOO null ──────────────────────────────────────
        pool = None
        null_knn = None
        if knn is not None:
            # Cache feature pool with a task-and-view qualified key
            pool_key = f"{task_full}__{view}"
            print(f"    [view {view}] building k-NN pool (DINOv2) …")
            pool = knn.build_pool(pool_key, pngs, DINO_CACHE_DIR)
            print(f"      pool: {len(pool['paths'])} frames × {pool['feats'].shape[1]} dim")
            print(f"    [view {view}] calibrating k-NN LOO null ({len(pngs)} samples) …")
            null_knn = knn.calibrate_loo(pool, verbose=False)
            print(f"      null_ivar: mean={null_knn['null_ivar'].mean():.3f}, "
                  f"std={null_knn['null_ivar'].std():.3f}, CV={null_knn['cv']:.3f}")
            print(f"      route: {null_knn['route']}  "
                  f"(CV>={knn.cv_threshold:.2f} → ivar, else peak)")

        # ── Save combined null for this (task, view) ──────────────────
        save_data = {
            "null_lags": NULL_LAGS,
            "n_ref_frames": len(pngs),
            "view": np.str_(view),
        }
        for n in sig_names:
            save_data[f"cycle_{n}_mean"] = sorted_null[n]["mean"]
            save_data[f"cycle_{n}_peak"] = sorted_null[n]["peak"]
        if null_knn is not None:
            save_data["knn_null_ivar"] = null_knn["null_ivar"]
            save_data["knn_null_peak"] = null_knn["null_peak"]
            save_data["knn_route"] = np.str_(null_knn["route"])
            save_data["knn_cv"] = null_knn["cv"]
        np.savez(NULL_DIR / f"{task_short}__{view}.npz", **save_data)

        return {
            "ref_pngs": pngs,
            "sorted_null": sorted_null,
            "pool": pool,
            "null_knn": null_knn,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Per-task loop
    # ─────────────────────────────────────────────────────────────────────

    all_rows: list[dict] = []

    for task_short, task_meta in eval_tasks.items():
        task_full = task_meta["task_full"]
        task_views = [v for v in args.views if v in task_meta.get("views", DEFAULT_VIEWS)]
        if not task_views:
            print(f"\n[task {task_short}] no overlap between --views and eval_tasks views — skip")
            continue

        print(f"\n{'=' * 72}")
        print(f"Task {task_short}: {task_full[:60]}")
        print(f"  views to eval: {task_views}")
        print('=' * 72)

        # Build a per-view state once for this task
        view_states: dict[str, dict] = {}
        for view in task_views:
            state = build_view_null(task_short, task_full, view)
            if state is not None:
                view_states[view] = state

        if not view_states:
            print(f"[task {task_short}] no view states ready — skip task")
            continue

        # Score one video (mp4 path differs per view!) and aggregate
        def score_video_multiview(
            video_id: str,
            video_type: str,
            label: int,
            view_mp4_lookup: dict,           # view -> Path (real input video)
            missing_demux_views: set[str] | None = None,
        ) -> dict | None:
            t0 = time.time()
            row = {
                "task": task_short,
                "video": video_id,
                "type": video_type,
                "label": label,
            }
            fused_per_view: list[float] = []
            fused_robust_per_view: list[float] = []
            any_scored = False

            for view, state in view_states.items():
                view_mp4 = view_mp4_lookup.get(view)
                if view_mp4 is None or not view_mp4.exists():
                    print(f"    [view {view}] mp4 missing: {view_mp4}")
                    row[f"null_n_{view}"] = int(len(state["sorted_null"][sig_names[0]]["mean"]))
                    continue
                vr = score_view_video(
                    view_mp4,
                    state["sorted_null"],
                    state["pool"],
                    state["null_knn"],
                )
                row[f"null_n_{view}"] = int(len(state["sorted_null"][sig_names[0]]["mean"]))
                if vr is None:
                    print(f"    [view {view}] could not sample frames")
                    continue
                any_scored = True
                # Inject the view tag into every per-view raw key.
                for k, v in vr.items():
                    row[_inject_view(k, view)] = v

                if "fused_peak" in vr:
                    fused_per_view.append(float(vr["fused_peak"]))
                if "fused_robust" in vr:
                    fused_robust_per_view.append(float(vr["fused_robust"]))

            if not any_scored:
                return None

            # Cross-view Cauchy fusion: convert each H to a p-value then combine
            def _multiview_fuse(hs: list[float]) -> float:
                eps = 1e-6
                ps = [1.0 - float(np.clip(h, eps, 1.0 - eps)) for h in hs]
                return 1.0 - cauchy_combine(ps)

            if fused_per_view:
                row["multiview_fused_peak"] = _multiview_fuse(fused_per_view)
            else:
                row["multiview_fused_peak"] = 0.0
            if fused_robust_per_view:
                row["multiview_fused_robust"] = _multiview_fuse(fused_robust_per_view)
            else:
                row["multiview_fused_robust"] = 0.0

            row["time_sec"] = time.time() - t0
            row["missing_demux_views"] = ",".join(sorted(missing_demux_views or []))
            return row

        # ── Real training video (per-view mp4s from raw_videos) ───────
        real_lookup = {
            v: RAW_VIDEO_ROOT / task_short / f"{v}.mp4" for v in view_states.keys()
        }
        # Print mp4 existence summary
        print(f"  REAL videos:")
        for v, p in real_lookup.items():
            print(f"    {v}: {'OK' if p.exists() else 'MISSING'}  {p}")

        if all(p.exists() for p in real_lookup.values()):
            r = score_video_multiview(
                video_id=f"{task_short}_real",
                video_type="REAL",
                label=0,
                view_mp4_lookup=real_lookup,
                missing_demux_views=None,
            )
            if r is not None:
                all_rows.append(r)
                _print_row(r, view_states.keys())

        # ── Generated videos ──────────────────────────────────────────
        gen_dir = GEN_ROOT / task_full
        if gen_dir.exists():
            for gen_mp4 in sorted(gen_dir.glob("v*.mp4")):
                # Skip demuxed sub-dirs that look like v0000_views/exterior_1.mp4
                # (they live in subdirs, not in this glob, but defensive)
                missing_demux: set[str] = set()
                view_lookup: dict[str, Path] = {}
                for v in view_states.keys():
                    p, is_demuxed = resolve_view_video(gen_mp4, v)
                    if not is_demuxed:
                        missing_demux.add(v)
                    view_lookup[v] = p
                if missing_demux:
                    print(f"  [warn] {gen_mp4.name}: no demuxed views for {sorted(missing_demux)} "
                          f"— falling back to composite mp4 (scores contaminated)")
                r = score_video_multiview(
                    video_id=gen_mp4.name,
                    video_type="GEN",
                    label=1,
                    view_mp4_lookup=view_lookup,
                    missing_demux_views=missing_demux,
                )
                if r is not None:
                    all_rows.append(r)
                    _print_row(r, view_states.keys())
        else:
            print(f"  [skip] no generated dir: {gen_dir}")

    # ─────────────────────────────────────────────────────────────────────
    # Save CSV
    # ─────────────────────────────────────────────────────────────────────
    csv_path = OUT / f"per_task_dense_table{args.out_suffix}.csv"
    if all_rows:
        fieldnames: list[str] = []
        for r in all_rows:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nTable → {csv_path}")
    else:
        print("\n[warn] no rows produced")

    # ── Training-safety report ───────────────────────────────────────────
    reals = [r for r in all_rows if r["type"] == "REAL"]
    if reals:
        print("\n" + "=" * 72)
        print("PRIMARY METRIC: REAL TRAINING SAFETY (multiview_fused_peak)")
        print("=" * 72)
        for r in reals:
            print(f"  task {r['task']:<4} multiview_fused_peak = {r.get('multiview_fused_peak', 0):.4f}")


def _inject_view(key: str, view: str) -> str:
    """Insert view tag into raw signal key.

    Mapping:
        cycle_peak                  → cycle_<view>_peak
        cycle_robust                → cycle_<view>_robust
        precision_anomaly_peak      → precision_anomaly_<view>_peak
        precision_anomaly_robust    → precision_anomaly_<view>_robust
        knn_peak                    → knn_<view>_peak
        knn_robust                  → knn_<view>_robust
        knn_route                   → knn_<view>_route
        knn_mean_ivar               → knn_<view>_mean_ivar
        knn_mean_peak_raw           → knn_<view>_mean_peak_raw
        fused_peak                  → fused_<view>_peak
        fused_robust                → fused_<view>_robust
    """
    if key.endswith("_peak") and not key.startswith("precision_anomaly"):
        prefix = key[: -len("_peak")]
        return f"{prefix}_{view}_peak"
    if key.endswith("_robust") and not key.startswith("precision_anomaly"):
        prefix = key[: -len("_robust")]
        return f"{prefix}_{view}_robust"
    if key.startswith("precision_anomaly_"):
        suffix = key[len("precision_anomaly_"):]
        return f"precision_anomaly_{view}_{suffix}"
    if key.startswith("knn_") and key not in ("knn_peak", "knn_robust"):
        suffix = key[len("knn_"):]
        return f"knn_{view}_{suffix}"
    return f"{key.split('_')[0]}_{view}_{'_'.join(key.split('_')[1:])}"


def _print_row(r: dict, views) -> None:
    parts = [f"  [{r['type']:<4}] {r['video']:<14}"]
    for v in views:
        f = r.get(f"fused_{v}_peak", float("nan"))
        parts.append(f"{v}={f:.3f}")
    parts.append(f"MV={r.get('multiview_fused_peak', 0):.3f}")
    if r.get("time_sec"):
        parts.append(f"({r['time_sec']:.0f}s)")
    print("  ".join(parts))


if __name__ == "__main__":
    main()
