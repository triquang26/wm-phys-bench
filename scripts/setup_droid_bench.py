#!/usr/bin/env python3
"""
Set up paper-physical-droid/ benchmark dir mirroring paper-physical-gr1/.

Reads episode annotations from droid_subset_100/annotation/train/*.json,
groups by canonical language instruction, picks top-5 most-frequent tasks,
and copies the exterior_1_left mp4 (videos/train/<ep>/0.mp4) into
paper-physical-droid/raw_videos/droid/<N>.mp4, extracts frame 0 as PNG
conditioning, and writes eval_tasks.json.

Idempotent: skips outputs that already exist.
"""
from __future__ import annotations

import json
import re
import shutil
import string
from collections import defaultdict
from pathlib import Path

import cv2

# ---- paths ---------------------------------------------------------------
DROID_ROOT = Path(
    "/mnt/data/sftp/data/quangpt3/Ctrl-World/dataset_example/droid_subset_100"
)
ANNOT_DIR = DROID_ROOT / "annotation" / "train"
OUT_ROOT = Path(
    "/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/"
    "feature_matching_eval_hallucination/paper-physical-droid"
)
RAW_VID_DIR = OUT_ROOT / "raw_videos" / "droid"
COND_DIR = OUT_ROOT / "conditioning"
REF_DIR = OUT_ROOT / "reference"
EVAL_TASKS_PATH = OUT_ROOT / "eval_tasks.json"

N_TASKS = 5
EXTERIOR_1_LEFT_VIEW_IDX = 0  # confirmed via Ctrl-World/dataset_example/extract_latent.py

# ---- helpers -------------------------------------------------------------
def canonical(text: str) -> str:
    """Lowercase, collapse whitespace, strip surrounding punctuation."""
    t = text.lower().strip()
    # remove trailing/leading punctuation; collapse internal whitespace
    t = t.translate(str.maketrans("", "", string.punctuation))
    t = re.sub(r"\s+", " ", t).strip()
    return t


def clean_for_filename(text: str, max_len: int = 80) -> str:
    """Keep alnum + space; collapse repeats; trim to max_len."""
    cleaned = re.sub(r"[^a-zA-Z0-9 ]+", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:max_len].strip()


def extract_frame0(mp4_path: Path, png_path: Path) -> bool:
    cap = cv2.VideoCapture(str(mp4_path))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return False
    png_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(png_path), frame)
    return True


# ---- main ----------------------------------------------------------------
def main() -> None:
    assert ANNOT_DIR.is_dir(), f"missing annotation dir: {ANNOT_DIR}"

    # 1. Scan annotations, group episodes by canonical instruction
    by_task: dict[str, list[dict]] = defaultdict(list)
    n_files = 0
    for jp in sorted(ANNOT_DIR.glob("*.json")):
        with open(jp, "r") as f:
            meta = json.load(f)
        texts = meta.get("texts") or []
        if not texts:
            continue
        # use first instruction as canonical text
        raw = texts[0]
        key = canonical(raw)
        if not key:
            continue
        ep_id = meta.get("episode_id", int(jp.stem))
        by_task[key].append(
            {
                "annotation_file": jp.name,
                "raw_text": raw,
                "episode_id": ep_id,
                "video_length": meta.get("video_length"),
                "success": bool(meta.get("success", 0)),
            }
        )
        n_files += 1

    # 2. Rank tasks by episode count desc, then by canonical text for determinism
    ranked = sorted(by_task.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    chosen = ranked[:N_TASKS]

    print(f"scanned {n_files} annotations, {len(by_task)} unique tasks")
    print(f"top {N_TASKS} task episode counts: {[len(v) for _, v in chosen]}")

    # 3. Build output dirs
    RAW_VID_DIR.mkdir(parents=True, exist_ok=True)
    COND_DIR.mkdir(parents=True, exist_ok=True)
    REF_DIR.mkdir(parents=True, exist_ok=True)

    eval_tasks: dict[str, dict] = {}

    for task_short, (canon_key, episodes) in enumerate(chosen):
        # pick first episode (lowest episode_id) as the "training video"
        episodes_sorted = sorted(episodes, key=lambda e: e["episode_id"])
        first = episodes_sorted[0]
        ep_id = first["episode_id"]
        raw_text = first["raw_text"]

        cleaned = clean_for_filename(raw_text)
        task_full = f"{task_short}_{cleaned}"

        # exterior_1_left mp4 = videos/train/<ep_id>/0.mp4
        src_mp4 = (
            DROID_ROOT / "videos" / "train" / str(ep_id) / f"{EXTERIOR_1_LEFT_VIEW_IDX}.mp4"
        )
        dst_mp4 = RAW_VID_DIR / f"{task_short}.mp4"
        dst_png = COND_DIR / f"{task_full}.png"
        ref_subdir = REF_DIR / task_full

        if not src_mp4.is_file():
            print(f"  [skip task {task_short}] missing source mp4: {src_mp4}")
            continue

        # copy mp4 (idempotent)
        if dst_mp4.exists():
            print(f"  [task {task_short}] mp4 exists, skip copy: {dst_mp4.name}")
        else:
            shutil.copy(src_mp4, dst_mp4)
            print(f"  [task {task_short}] copied {src_mp4} -> {dst_mp4}")

        # extract frame 0 PNG (idempotent)
        if dst_png.exists():
            print(f"  [task {task_short}] png exists, skip: {dst_png.name}")
        else:
            ok = extract_frame0(dst_mp4, dst_png)
            if not ok:
                print(f"  [task {task_short}] WARNING: failed to read frame 0 from {dst_mp4}")
            else:
                print(f"  [task {task_short}] frame0 -> {dst_png.name}")

        # ensure empty reference subdir
        ref_subdir.mkdir(parents=True, exist_ok=True)

        eval_tasks[str(task_short)] = {
            "task_full": task_full,
            "training_episode": f"ep_{ep_id:04d}",
            "training_video": f"raw_videos/droid/{task_short}.mp4",
            "language_instruction": raw_text,
            "n_episodes_available": len(episodes),
        }

    # 4. Write eval_tasks.json (overwrite — small, deterministic)
    with open(EVAL_TASKS_PATH, "w") as f:
        json.dump(eval_tasks, f, indent=2, ensure_ascii=False)
    print(f"wrote {EVAL_TASKS_PATH}")

    # 5. Final report
    print("\n=== FINAL REPORT ===")
    print(f"output root: {OUT_ROOT}")
    print(f"tasks written ({len(eval_tasks)}):")
    for k, v in eval_tasks.items():
        instr = v["language_instruction"]
        print(
            f"  [{k}] eps={v['n_episodes_available']:>2}  "
            f"ep={v['training_episode']}  '{instr[:70]}'"
        )


if __name__ == "__main__":
    main()
