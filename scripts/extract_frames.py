"""Extract N uniformly-spaced frames from every MP4 in cosmos_synthetic_data/{high,low}
and save raw (no bg removal) PNGs to a staging directory.

Usage:
    python extract_frames.py \
        --src_root cosmos_synthetic_data \
        --dst_root cosmos_frames_raw \
        --n_frames 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def extract_uniform(video_path: Path, dst_dir: Path, n_frames: int) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return 0

    indices = np.linspace(0, total - 1, min(n_frames, total), dtype=int)
    saved = 0
    for frame_no in indices:
        out_path = dst_dir / f"frame_{frame_no:04d}.png"
        if out_path.exists():
            saved += 1
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))
        ok, frame = cap.read()
        if not ok:
            continue
        cv2.imwrite(str(out_path), frame)
        saved += 1
    cap.release()
    return saved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", required=True)
    ap.add_argument("--dst_root", required=True)
    ap.add_argument("--n_frames", type=int, default=50)
    args = ap.parse_args()

    src = Path(args.src_root)
    dst = Path(args.dst_root)
    total_saved = 0

    for split in ("high", "low"):
        split_dir = src / split
        if not split_dir.exists():
            print(f"[skip] {split_dir} not found")
            continue
        mp4s = sorted(split_dir.glob("*.mp4"))
        print(f"\n[{split}] {len(mp4s)} videos")
        for mp4 in mp4s:
            task = mp4.stem  # filename without .mp4
            out_dir = dst / split / task
            n = extract_uniform(mp4, out_dir, args.n_frames)
            print(f"  {n:3d} frames → {split}/{task[:60]}")
            total_saved += n

    print(f"\nTotal: {total_saved} frames saved to {dst}")


if __name__ == "__main__":
    main()
