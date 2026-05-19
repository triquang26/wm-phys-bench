#!/usr/bin/env python3
"""Demux Cosmos DROID 2×2 composite outputs into per-view mp4s.

Cosmos-Predict2-14B-Sample-GR00T-Dreams-DROID emits 640×384 composite frames
laid out exactly as our `setup_droid_multiview.make_2x2_composite` produced
the conditioning image:

    +-----------------+-----------------+      640
    |   exterior_1    |   exterior_2    |      ↕ 192
    +-----------------+-----------------+
    |     wrist       |   (filler, =    |      ↕ 192
    |                 |    exterior_1)  |
    +-----------------+-----------------+
            320              320

For each `paper-physical-droid/generated/<task_full>/v*.mp4` we emit:
    paper-physical-droid/generated/<task_full>/<vid_stem>_views/
        exterior_1.mp4   exterior_2.mp4   wrist.mp4

The bottom-right tile is duplicate filler and is **not** written out. The
script is idempotent: existing per-view mp4s are skipped (`--force` overrides).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path("/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination")

# Tile coordinates inside the 640×384 composite (y0:y1, x0:x1)
TILE_COORDS = {
    "exterior_1": ((0,   192), (0,   320)),
    "exterior_2": ((0,   192), (320, 640)),
    "wrist":      ((192, 384), (0,   320)),
}
EXPECTED_W = 640
EXPECTED_H = 384


def demux_one(src_mp4: Path, dst_dir: Path, force: bool = False) -> dict[str, Path]:
    """Demux a single composite mp4 into the three view mp4s. Returns mapping."""
    dst_dir.mkdir(parents=True, exist_ok=True)

    out_paths = {v: dst_dir / f"{v}.mp4" for v in TILE_COORDS}
    if not force and all(p.exists() for p in out_paths.values()):
        return out_paths

    cap = cv2.VideoCapture(str(src_mp4))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {src_mp4}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (width, height) != (EXPECTED_W, EXPECTED_H):
        print(f"  [warn] {src_mp4.name}: expected {EXPECTED_W}×{EXPECTED_H}, "
              f"got {width}×{height} — resizing before demux")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers = {
        v: cv2.VideoWriter(str(out_paths[v]), fourcc, fps, (320, 192))
        for v in TILE_COORDS
    }

    n_frames = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if (bgr.shape[1], bgr.shape[0]) != (EXPECTED_W, EXPECTED_H):
            bgr = cv2.resize(bgr, (EXPECTED_W, EXPECTED_H), interpolation=cv2.INTER_AREA)
        for view, ((y0, y1), (x0, x1)) in TILE_COORDS.items():
            tile = bgr[y0:y1, x0:x1]
            writers[view].write(tile)
        n_frames += 1
    cap.release()
    for w in writers.values():
        w.release()
    return out_paths


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Demux 2×2 Cosmos DROID composites into per-view mp4s."
    )
    ap.add_argument("--bench", default="paper-physical-droid",
                    help="bench dir name under REPO_ROOT (default: paper-physical-droid)")
    ap.add_argument("--force", action="store_true",
                    help="re-demux even if per-view mp4s already exist")
    ap.add_argument("--task", default=None,
                    help="restrict to this task_full (default: all)")
    ap.add_argument("--limit", type=int, default=None,
                    help="for debug: only process the first N composites per task")
    args = ap.parse_args()

    BENCH = REPO_ROOT / args.bench
    GEN_ROOT = BENCH / "generated"
    if not GEN_ROOT.exists():
        sys.exit(f"generated/ not found: {GEN_ROOT}")

    task_dirs = sorted(d for d in GEN_ROOT.iterdir() if d.is_dir())
    if args.task is not None:
        task_dirs = [d for d in task_dirs if d.name == args.task]
        if not task_dirs:
            sys.exit(f"no generated/ subdir matches --task={args.task!r}")

    total_demuxed = 0
    total_skipped = 0
    for task_dir in task_dirs:
        mp4s = sorted(p for p in task_dir.glob("v*.mp4") if p.is_file())
        if args.limit is not None:
            mp4s = mp4s[: args.limit]
        if not mp4s:
            print(f"[task {task_dir.name}] (no composite mp4s)")
            continue
        print(f"[task {task_dir.name}] {len(mp4s)} composite mp4(s)")
        for src in mp4s:
            dst_dir = task_dir / f"{src.stem}_views"
            already = all((dst_dir / f"{v}.mp4").exists() for v in TILE_COORDS)
            if already and not args.force:
                total_skipped += 1
                print(f"  [skip] {src.name} (already demuxed)")
                continue
            try:
                demux_one(src, dst_dir, force=args.force)
                total_demuxed += 1
                print(f"  + {src.name} → {dst_dir.name}/{{exterior_1,exterior_2,wrist}}.mp4")
            except Exception as e:
                print(f"  [err] {src.name}: {e}")

    print(f"\n=== done. demuxed {total_demuxed}, skipped {total_skipped} ===")


if __name__ == "__main__":
    main()
