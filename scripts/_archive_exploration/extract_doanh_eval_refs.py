#!/usr/bin/env python3
"""Extract SAM3-segmented reference frames from doanh's high/<task>.mp4.

For each task, sample 50 frames from the high-quality generated video
(treat it as the "training reference") and SAM3-segment them. Background
gray-filled (fallback=gray for unsegmented frames).
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
BENCH = REPO_ROOT / "paper-doanh-eval"

N_REFS = 50


def sample_indices(mp4: Path, n: int) -> list[int]:
    cap = cv2.VideoCapture(str(mp4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total < 2:
        return []
    return [int(i) for i in np.linspace(0, total - 1, min(n, total), dtype=int)]


def main():
    eval_tasks = json.loads((BENCH / "eval_tasks.json").read_text())
    from warp_score.sam_segmenter import VideoFrameSegmenter
    seg = VideoFrameSegmenter(fallback="gray")
    print(f"Loaded SAM3 (fallback=gray). Processing {len(eval_tasks)} tasks …\n")

    for ts, task in sorted(eval_tasks.items(), key=lambda x: int(x[0])):
        task_full = task["task_full"]
        # high mp4 path (truncated to 200 chars per setup script)
        high_name = f"{ts}_{task_full}.mp4"[:200]
        high_mp4 = BENCH / "raw_videos" / "high" / high_name
        ref_dir = BENCH / "reference" / f"{ts}_{task_full}"[:200]
        ref_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(ref_dir.glob("frame_*.png"))
        if len(existing) >= 30:
            print(f"[{ts}] skip — {len(existing)} refs already in {ref_dir.name[:60]}")
            continue
        if not high_mp4.exists():
            print(f"[{ts}] missing {high_mp4}")
            continue
        # Wipe partial
        for p in existing:
            p.unlink()
        indices = sample_indices(high_mp4, N_REFS)
        cap = cv2.VideoCapture(str(high_mp4))
        t0 = time.time()
        n_ok = 0
        for j, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, bgr = cap.read()
            if not ok:
                continue
            seg_bgr = seg.segment_frame(bgr)
            cv2.imwrite(str(ref_dir / f"frame_{j:04d}.png"), seg_bgr)
            n_ok += 1
        cap.release()
        print(f"[{ts}] {task_full[:50]} → {n_ok} refs in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
