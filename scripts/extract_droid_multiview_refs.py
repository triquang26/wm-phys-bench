#!/usr/bin/env python3
"""Extract 120 SAM3-segmented refs per (task, view) for DROID multi-view eval.

Reads paper-physical-droid/eval_tasks.json, iterates over (task, view) pairs,
extracts dense refs into reference/<task_full>/<view>/.

Idempotent: skips any (task, view) whose ref dir is already populated.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

REPO_ROOT = Path("/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination")
BENCH = REPO_ROOT / "paper-physical-droid"
REF_ROOT = BENCH / "reference"
EVAL_TASKS_JSON = BENCH / "eval_tasks.json"

N_DENSE = 120
MIN_FOR_SKIP = 30  # if ≥ this many PNGs present, treat as done


def sample_indices(mp4: Path, n: int) -> list[int]:
    cap = cv2.VideoCapture(str(mp4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total < 2:
        raise RuntimeError(f"Bad video: {mp4} (frame count {total})")
    return [int(i) for i in np.linspace(0, total - 1, min(n, total), dtype=int)]


def extract_one_view(task_full: str, view: str, mp4: Path, seg) -> bool:
    ref_dir = REF_ROOT / task_full / view
    existing = sorted(ref_dir.glob("frame_*.png"))
    if len(existing) >= MIN_FOR_SKIP:
        print(f"    [skip] {task_full[:40]}/{view}: {len(existing)} refs already")
        return True

    ref_dir.mkdir(parents=True, exist_ok=True)
    for p in existing:
        p.unlink()

    indices = sample_indices(mp4, N_DENSE)
    cap = cv2.VideoCapture(str(mp4))
    t0 = time.time()
    n_ok = 0
    for j, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        if not ok:
            continue
        seg_bgr = seg.segment_frame(bgr)
        out_path = ref_dir / f"frame_{j:04d}.png"
        cv2.imwrite(str(out_path), seg_bgr)
        n_ok += 1
        if (j + 1) % 30 == 0:
            print(f"      {j+1:3d}/{len(indices)}  ({time.time()-t0:.0f}s)")
    cap.release()
    print(f"    done {n_ok} frames in {time.time()-t0:.0f}s")
    return True


def main():
    eval_tasks = json.loads(EVAL_TASKS_JSON.read_text())
    print(f"Loaded {len(eval_tasks)} tasks from {EVAL_TASKS_JSON}")

    from warp_score.sam_segmenter import VideoFrameSegmenter
    print("Loading SAM3 (fallback=gray for DROID — no-detect → all-gray, not full frame) …")
    seg = VideoFrameSegmenter(fallback="gray")

    for ts in sorted(eval_tasks.keys(), key=lambda k: int(k)):
        task = eval_tasks[ts]
        task_full = task["task_full"]
        views = task["views"]
        print(f"\n=== Task {ts}: {task_full[:60]} ===")
        for view in views:
            rel = task["view_videos"][view]
            mp4 = BENCH / rel
            if not mp4.exists():
                print(f"    [skip] missing {mp4}")
                continue
            print(f"  view {view}: {mp4.name}")
            extract_one_view(task_full, view, mp4, seg)


if __name__ == "__main__":
    main()
