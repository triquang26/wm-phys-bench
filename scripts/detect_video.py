"""detect_video.py — CLI for video-level hallucination detection.

Single video (with SAM3 segmentation, default):
    python scripts/detect_video.py \\
        --video "data/cosmos_synthetic_data/query/gr00t/0_Open the box/v0006.mp4" \\
        --config warp_score/configs/test_knn15.yaml --fps 2

Without SAM3 (frames already pre-segmented, or for speed testing):
    python scripts/detect_video.py --video v0006.mp4 --no_sam

Folder (recursive):
    python scripts/detect_video.py \\
        --video_dir "data/cosmos_synthetic_data/query/gr00t/" \\
        --recursive --fps 2 \\
        --summary_csv /tmp/gr00t_summary.csv

Skip cache:
    python scripts/detect_video.py --video v0006.mp4 --no-cache
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _build_detector(args: argparse.Namespace):
    """Load config, model, calibration, optional SAM3; return (VideoDetector, config)."""
    from warp_score.config import WarpScoreConfig
    from warp_score.calibrator import CalibrationArtifact
    from warp_score.matcher import RoMaMatcher
    from warp_score.mask import InteriorMask
    from warp_score.signals import build_signals
    from warp_score.fuser import build_fuser
    from warp_score.video_detector import VideoDetector

    cfg = WarpScoreConfig.from_yaml(args.config) if args.config else WarpScoreConfig()

    print(f"[init] config   : {args.config or '(default)'}")
    print(f"[init] artifacts: {cfg.artifacts_dir}")
    print(f"[init] calib    : {cfg.calib_path}")
    print(f"[init] SAM3     : {'enabled' if not args.no_sam else 'disabled (--no_sam)'}")

    t0 = time.perf_counter()
    calib = CalibrationArtifact.load(cfg.calib_path)
    print(f"[init] calibration loaded in {(time.perf_counter()-t0)*1000:.0f}ms")

    t0 = time.perf_counter()
    matcher = RoMaMatcher(
        setting=cfg.setting,
        device=cfg.device,
        use_precision=cfg.use_precision,
        bidirectional=getattr(cfg, "bidirectional", False),
        vis_size=cfg.vis_size,
    )
    matcher._load_model()
    print(f"[init] RoMa model loaded in {(time.perf_counter()-t0)*1000:.0f}ms")

    # SAM3 segmenter — loaded lazily (first segment_frame() call triggers it)
    segmenter = None
    if not args.no_sam:
        t0 = time.perf_counter()
        segmenter = VideoDetector.load_segmenter(threshold=args.sam_threshold)
        # Force model load now so it shows up in init timing
        segmenter._load()
        print(f"[init] SAM3 model loaded in {(time.perf_counter()-t0)*1000:.0f}ms")

    signals  = build_signals(cfg.signal_names)
    fuser    = build_fuser(cfg.fuser)
    interior = InteriorMask(cfg.erosion_k)

    detector = VideoDetector.from_config(
        cfg, matcher, calib, fuser, signals, interior,
        segmenter=segmenter,
    )
    return detector, cfg


def _collect_videos(args: argparse.Namespace) -> list[Path]:
    if args.video:
        p = Path(args.video).expanduser().resolve()
        if not p.exists():
            sys.exit(f"Video not found: {p}")
        return [p]

    root = Path(args.video_dir).expanduser().resolve()
    if not root.exists():
        sys.exit(f"Directory not found: {root}")

    pattern = "**/*.mp4" if args.recursive else "*.mp4"
    videos = sorted(root.glob(pattern))
    if not videos:
        sys.exit(f"No .mp4 files found in {root} (recursive={args.recursive})")
    return videos


def _print_header(n_videos: int, fps: float, use_sam: bool) -> None:
    sam_str = "SAM3 seg ON" if use_sam else "SAM3 seg OFF"
    print(f"\n{'='*72}")
    print(f"  Video hallucination detector  |  {n_videos} video(s)  |  fps={fps}  |  {sam_str}")
    print(f"{'='*72}")


def _write_summary_csv(rows: list[dict], csv_path: Path) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n[summary] CSV written → {csv_path}  ({len(rows)} rows)")


def run(args: argparse.Namespace) -> None:
    from warp_score.video_detector import VideoDetector

    detector, cfg = _build_detector(args)
    videos = _collect_videos(args)
    fps = args.fps

    _print_header(len(videos), fps, use_sam=not args.no_sam)

    summary_rows: list[dict] = []
    t_wall_start = time.perf_counter()

    for vi, video_path in enumerate(videos, 1):
        task = args.task or None   # None → auto-infer inside VideoDetector
        print(f"\n[{vi}/{len(videos)}] {video_path.name}")
        print(f"        {video_path}")

        try:
            result = detector.detect_video(
                video_path=video_path,
                fps=fps,
                n_frames=args.n_frames if args.n_frames else None,
                task=task,
                use_cache=not args.no_cache,
            )
        except Exception as e:
            print(f"  [ERROR] {e}")
            summary_rows.append({
                "video_path": str(video_path),
                "task": str(video_path.parent.name),
                "n_frames": 0,
                "H_video": float("nan"),
                "decision": "ERROR",
                "t_total_s": 0.0,
                "error": str(e),
            })
            continue

        VideoDetector.print_video_summary(result)

        summary_rows.append({
            "video_path":    str(result.video_path),
            "task":          result.task,
            "n_frames":      result.n_frames,
            "H_video":       round(result.H_video, 4),
            "p_video":       round(result.p_video, 4),
            "decision":      result.decision,
            "mean_H":        round(result.mean_H, 4),
            "t_total_s":     round(result.t_total_s, 2),
            "mean_frame_ms": round(result.mean_total_ms, 1),
            "mean_roma_ms":  round(result.mean_roma_ms, 1),
            "from_cache":    int(result.from_cache),
            "cache_key":     result.cache_key,
        })

    # ── Final summary table ────────────────────────────────────────────────────
    t_wall = time.perf_counter() - t_wall_start
    valid_rows = [r for r in summary_rows if r.get("decision") != "ERROR"]

    print(f"\n{'='*72}")
    print(f"  RESULTS  ({len(valid_rows)}/{len(videos)} processed)  wall={t_wall:.1f}s")
    print(f"{'='*72}")
    print(f"  {'Video':<40}  {'Task':<30}  {'H_video':>7}  {'Decision':<12}  {'T(s)':>6}")
    print(f"  {'-'*40}  {'-'*30}  {'-'*7}  {'-'*12}  {'-'*6}")
    for r in summary_rows:
        h = r.get("H_video", float("nan"))
        h_str = f"{h:.4f}" if h == h else "   NaN"
        print(
            f"  {str(r['video_path']).split('/')[-1]:<40}  "
            f"{r['task'][:30]:<30}  "
            f"{h_str:>7}  "
            f"{r.get('decision','?'):<12}  "
            f"{r.get('t_total_s', 0):>6.1f}"
        )

    if valid_rows:
        import numpy as np
        hs = [r["H_video"] for r in valid_rows]
        print(f"\n  H_video:  mean={np.mean(hs):.4f}  min={min(hs):.4f}  "
              f"max={max(hs):.4f}  median={np.median(hs):.4f}")
        hallu     = sum(1 for r in valid_rows if r["decision"] == "hallucinated")
        clean     = sum(1 for r in valid_rows if r["decision"] == "clean")
        uncertain = sum(1 for r in valid_rows if r["decision"] == "uncertain")
        print(f"  Decisions: hallucinated={hallu}  clean={clean}  uncertain={uncertain}")

    if args.summary_csv:
        _write_summary_csv(summary_rows, Path(args.summary_csv))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Video hallucination detector")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video",     help="Single .mp4 file to process")
    src.add_argument("--video_dir", help="Directory of .mp4 files to process")

    p.add_argument("--recursive",      action="store_true",
                   help="Recurse into subdirectories when using --video_dir")
    p.add_argument("--config",         default="warp_score/configs/test_knn15.yaml",
                   help="WarpScoreConfig YAML")
    p.add_argument("--fps",            type=float, default=2.0,
                   help="Sample rate in frames/sec (default: 2.0)")
    p.add_argument("--n_frames",       type=int, default=None,
                   help="Fixed frame count to sample (overrides --fps)")
    p.add_argument("--task",           default=None,
                   help="Override task name (auto-inferred if not set)")
    p.add_argument("--no_sam",         action="store_true",
                   help="Disable SAM3 segmentation (use when frames already pre-segmented)")
    p.add_argument("--sam_threshold",  type=float, default=0.3,
                   help="SAM3 detection threshold (default: 0.3)")
    p.add_argument("--no-cache",       action="store_true",
                   help="Force recompute, ignore existing cache")
    p.add_argument("--summary_csv",    default=None,
                   help="Write per-video summary to this CSV path")

    run(p.parse_args())
