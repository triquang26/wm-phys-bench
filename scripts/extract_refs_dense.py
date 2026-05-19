#!/usr/bin/env python3
"""Extract 182 SAM3-segmented reference frames per task from training video.

Densifies the existing 50-frame ref pool to 182 frames — matches the cycle null
sample count and provides a larger candidate set for k-NN reference selection.

Idempotent: if N_DENSE PNGs already present in the task's reference dir, skip.
Otherwise wipes and re-extracts (avoids mixing 50-frame and 182-frame indices).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

REPO_ROOT = Path("/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination")
BENCH = REPO_ROOT / "paper-physical-gr1"
REF_ROOT = BENCH / "reference"
RAW_VIDEO_ROOT = BENCH / "raw_videos" / "gr1"

EVAL_TASKS = [
    "1_Use the right hand to pick up green bok choy from tan table right side to bottom level of wire basket.",
    "2_Use the right hand to pick up rubik's cube from top level of the shelf to bottom level of the shelf.",
    "3_Use the right hand to pick up banana from teal plate to wooden table.",
    "4_Use the left hand to pick up dragonfruit from pink plate to teal plate.",
    "6_Use the right hand to pick up orange from middle of table to bottom white shelf.",
]

N_DENSE = 182


def sample_indices(mp4: Path, n: int) -> list[int]:
    cap = cv2.VideoCapture(str(mp4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total < 2:
        raise RuntimeError(f"Bad video (frame count {total}): {mp4}")
    return [int(i) for i in np.linspace(0, total - 1, min(n, total), dtype=int)]


def extract_one_task(task: str, seg) -> bool:
    task_short = task.split("_")[0]
    ref_dir = REF_ROOT / task
    real_mp4 = RAW_VIDEO_ROOT / f"{task_short}.mp4"

    if not real_mp4.exists():
        print(f"  [skip] missing training mp4: {real_mp4}")
        return False

    existing = sorted(ref_dir.glob("frame_*.png"))
    if len(existing) >= N_DENSE:
        print(f"  [skip] {ref_dir.name[:50]} already has {len(existing)} frames")
        return True

    # Wipe old + re-extract
    ref_dir.mkdir(parents=True, exist_ok=True)
    for p in existing:
        p.unlink()

    indices = sample_indices(real_mp4, N_DENSE)
    cap = cv2.VideoCapture(str(real_mp4))
    t0 = time.time()
    for j, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        if not ok:
            print(f"  [warn] failed to read frame {idx}")
            continue
        seg_bgr = seg.segment_frame(bgr)
        out_path = ref_dir / f"frame_{j:04d}.png"
        cv2.imwrite(str(out_path), seg_bgr)
        if (j + 1) % 20 == 0:
            print(f"  {j+1:3d}/{N_DENSE}  ({time.time()-t0:.0f}s)")
    cap.release()
    print(f"  done {len(indices)} frames in {time.time()-t0:.0f}s")
    return True


def main():
    from warp_score.sam_segmenter import VideoFrameSegmenter

    print("Loading SAM3 …")
    seg = VideoFrameSegmenter()

    for task in EVAL_TASKS:
        print(f"\n=== {task[:70]} ===")
        extract_one_task(task, seg)


if __name__ == "__main__":
    main()
