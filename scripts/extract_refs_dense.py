#!/usr/bin/env python3
"""Extract 182 SAM3-segmented reference frames per task from training video.

Densifies the existing 50-frame ref pool to 182 frames — matches the cycle null
sample count and provides a larger candidate set for k-NN reference selection.

Idempotent: if N_DENSE PNGs already present in the task's reference dir, skip.
Otherwise wipes and re-extracts (avoids mixing 50-frame and 182-frame indices).

Configurable per benchmark via --bench / --tasks_json / --raw_video_subdir.
Defaults preserve GR-1 behavior for backwards compat.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

REPO_ROOT = Path("/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination")

# ── Hardcoded GR-1 fallback list (used iff --tasks_json not provided) ───────
EVAL_TASKS_GR1 = [
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


def load_tasks(tasks_json: Path | None) -> list[str]:
    """Return list of task_full strings.

    If tasks_json provided: parse a JSON dict {task_short: {task_full, ...}} and
    return [task_full, ...] sorted by task_short. Otherwise fall back to the
    hardcoded GR-1 list.
    """
    if tasks_json is None:
        return list(EVAL_TASKS_GR1)
    raw = json.loads(Path(tasks_json).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{tasks_json}: expected top-level dict of {{task_short: {{...}}}}")

    def _sort_key(kv):
        k = kv[0]
        try:
            return (0, int(k))
        except ValueError:
            return (1, k)

    items = sorted(raw.items(), key=_sort_key)
    tasks: list[str] = []
    for short, meta in items:
        if not isinstance(meta, dict) or "task_full" not in meta:
            raise ValueError(f"{tasks_json}: entry {short!r} missing 'task_full'")
        tasks.append(str(meta["task_full"]))
    return tasks


def resolve_tasks_json(arg: str | None, bench: Path) -> Path | None:
    """Resolve --tasks_json path. Tries <bench>/, <REPO_ROOT>/, and cwd in order."""
    if arg is None:
        return None
    candidates = [bench / arg, REPO_ROOT / arg, Path(arg)]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"tasks_json not found in any of: {[str(c) for c in candidates]}")


def extract_one_task(task: str, seg, ref_root: Path, raw_video_root: Path) -> bool:
    task_short = task.split("_")[0]
    ref_dir = ref_root / task
    real_mp4 = raw_video_root / f"{task_short}.mp4"

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
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench", default="paper-physical-gr1",
                    help="bench dir name under REPO_ROOT (default: paper-physical-gr1)")
    ap.add_argument("--tasks_json", default=None,
                    help="path to eval_tasks.json (override hardcoded EVAL_TASKS_GR1). "
                         "Relative paths resolve under <REPO_ROOT>/<bench>/ first, "
                         "then under REPO_ROOT, then as cwd-relative.")
    ap.add_argument("--raw_video_subdir", default="raw_videos/gr1",
                    help="subdir under <bench>/ holding the training <task_short>.mp4 files "
                         "(default: raw_videos/gr1)")
    args = ap.parse_args()

    bench = REPO_ROOT / args.bench
    ref_root = bench / "reference"
    raw_video_root = bench / args.raw_video_subdir

    tasks_json_path = resolve_tasks_json(args.tasks_json, bench)
    tasks = load_tasks(tasks_json_path)

    print(f"bench         : {bench}")
    print(f"ref_root      : {ref_root}")
    print(f"raw_video_root: {raw_video_root}")
    print(f"tasks_json    : {tasks_json_path}")
    print(f"tasks         : {len(tasks)}")

    from warp_score.sam_segmenter import VideoFrameSegmenter

    print("Loading SAM3 …")
    seg = VideoFrameSegmenter()

    for task in tasks:
        print(f"\n=== {task[:70]} ===")
        extract_one_task(task, seg, ref_root, raw_video_root)


if __name__ == "__main__":
    main()
