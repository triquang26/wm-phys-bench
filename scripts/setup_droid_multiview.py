#!/usr/bin/env python3
"""Multi-view DROID benchmark setup (idempotent, preserves prior single-view work).

DROID dataset has 3 cameras per episode (320×192, 5 fps):
  - 0.mp4 = exterior_1_left   (front external)
  - 1.mp4 = exterior_2_left   (side external)
  - 2.mp4 = wrist_left        (wrist-mounted)

Each is processed INDEPENDENTLY by WarpDyn (separate null + scoring),
then per-view ratios are cross-view-fused via Cauchy combine.

Migration: if a pre-existing single-view setup exists (raw_videos/droid/{N}.mp4
flat layout, reference/<task>/frame_NNNN.png), MOVE those under exterior_1/
subdir so SAM3 work is preserved. Then ADD exterior_2 + wrist alongside.

## Output structure

paper-physical-droid/
├── eval_tasks.json                          (extended schema with views)
├── raw_videos/droid/<task_short>/
│   ├── exterior_1.mp4
│   ├── exterior_2.mp4
│   └── wrist.mp4
├── conditioning/<task_short>/
│   ├── exterior_1.png                       (frame 0)
│   ├── exterior_2.png
│   ├── wrist.png
│   └── multiview_2x2.png                    (640×384 composite for Cosmos DROID)
├── reference/<task_full>/<view>/            (SAM3 fills, exterior_1 may pre-exist)
└── generated/<task_full>/                   (Cosmos to fill)
"""
from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path("/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination")
BENCH = REPO_ROOT / "paper-physical-droid"
DROID_ROOT = Path("/mnt/data/sftp/data/quangpt3/Ctrl-World/dataset_example/droid_subset_100")
ANNOT_DIR = DROID_ROOT / "annotation" / "train"
VIDEO_DIR = DROID_ROOT / "videos" / "train"

VIEWS = {0: "exterior_1", 1: "exterior_2", 2: "wrist"}
N_TASKS = 5
TASK_FULL_MAX_LEN = 80


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def canonical_text(s: str) -> str:
    return re.sub(r"[^\w\s]", "", s).strip().lower()


def safe_filename(s: str) -> str:
    s = re.sub(r"[^\w\s]", "", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s[:TASK_FULL_MAX_LEN].rstrip()


def first_frame(mp4: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(mp4))
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read first frame: {mp4}")
    return bgr


def make_2x2_composite(view_pngs: dict[str, Path], out_path: Path) -> None:
    """Stitch 4-view composite for Cosmos DROID conditioning (640×384).

    Layout (DROID-checkpoint expects 4-view per nvidia/Cosmos-Predict2 docs):
      top-left:  exterior_1   top-right: exterior_2
      bot-left:  wrist        bot-right: exterior_1 (duplicate filler — only 3 cams in subset)
    """
    imgs = {}
    for v, p in view_pngs.items():
        bgr = cv2.imread(str(p))
        if bgr is None:
            raise FileNotFoundError(p)
        if bgr.shape[:2] != (192, 320):
            bgr = cv2.resize(bgr, (320, 192), interpolation=cv2.INTER_AREA)
        imgs[v] = bgr

    if "exterior_2" not in imgs:
        imgs["exterior_2"] = imgs["exterior_1"]
    if "wrist" not in imgs:
        imgs["wrist"] = imgs["exterior_1"]

    top = np.hstack([imgs["exterior_1"], imgs["exterior_2"]])
    bot = np.hstack([imgs["wrist"],      imgs["exterior_1"]])
    grid = np.vstack([top, bot])
    cv2.imwrite(str(out_path), grid)


def pick_tasks() -> list[dict]:
    """Group episodes by canonical text, sort alphabetically."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for json_path in sorted(ANNOT_DIR.glob("*.json")):
        with open(json_path) as f:
            meta = json.load(f)
        texts = meta.get("texts") or []
        if not texts:
            continue
        text = texts[0].strip()
        if not text:
            continue
        grouped[canonical_text(text)].append(int(meta["episode_id"]))

    tasks_sorted = sorted(grouped.items(), key=lambda kv: kv[0])[:N_TASKS]
    picked = []
    for i, (canon, ep_ids) in enumerate(tasks_sorted):
        ep_id = sorted(ep_ids)[0]
        json_path = ANNOT_DIR / f"{ep_id}.json"
        with open(json_path) as f:
            meta = json.load(f)
        text_raw = meta["texts"][0].strip()
        picked.append({
            "task_short": str(i),
            "task_full": f"{i}_{safe_filename(text_raw)}",
            "language_instruction": text_raw,
            "training_episode_id": ep_id,
            "n_episodes_available": len(ep_ids),
        })
    return picked


# ─────────────────────────────────────────────────────────────────────────────
# Migration helpers
# ─────────────────────────────────────────────────────────────────────────────


def migrate_existing_single_view(picked: list[dict]) -> None:
    """If pre-existing flat single-view layout exists, move under exterior_1/.

    Old layout (from initial setup_droid_bench.py):
      raw_videos/droid/<N>.mp4
      conditioning/<N>_<text>.png
      reference/<task_full>/frame_NNNN.png

    New layout:
      raw_videos/droid/<N>/exterior_1.mp4
      conditioning/<N>/exterior_1.png
      reference/<task_full>/exterior_1/frame_NNNN.png
    """
    raw_dir = BENCH / "raw_videos" / "droid"
    cond_dir = BENCH / "conditioning"
    ref_root = BENCH / "reference"

    for t in picked:
        ts = t["task_short"]
        task_full = t["task_full"]

        # Migrate raw_videos/droid/<N>.mp4 → raw_videos/droid/<N>/exterior_1.mp4
        old_mp4 = raw_dir / f"{ts}.mp4"
        new_dir = raw_dir / ts
        if old_mp4.exists() and old_mp4.is_file():
            new_dir.mkdir(parents=True, exist_ok=True)
            new_mp4 = new_dir / "exterior_1.mp4"
            if not new_mp4.exists():
                shutil.move(str(old_mp4), str(new_mp4))
                print(f"  migrated: {old_mp4.name} → {ts}/exterior_1.mp4")

        # Migrate conditioning/<N>_<text>.png → conditioning/<N>/exterior_1.png
        # Old format used full text in name; search by prefix
        for cond_png in cond_dir.glob(f"{ts}_*.png"):
            if cond_png.is_file() and cond_png.parent == cond_dir:
                tgt_dir = cond_dir / ts
                tgt_dir.mkdir(parents=True, exist_ok=True)
                tgt = tgt_dir / "exterior_1.png"
                if not tgt.exists():
                    shutil.move(str(cond_png), str(tgt))
                    print(f"  migrated: conditioning/{cond_png.name} → {ts}/exterior_1.png")

        # Migrate reference/<task_full>/frame_NNNN.png → reference/<task_full>/exterior_1/frame_NNNN.png
        old_ref_dir = ref_root / task_full
        if old_ref_dir.exists():
            pngs = sorted(old_ref_dir.glob("frame_*.png"))
            if pngs and not (old_ref_dir / "exterior_1").exists():
                ext1_dir = old_ref_dir / "exterior_1"
                ext1_dir.mkdir(parents=True, exist_ok=True)
                for p in pngs:
                    shutil.move(str(p), str(ext1_dir / p.name))
                print(f"  migrated: reference/{task_full}/frame_*.png ({len(pngs)}) → exterior_1/")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    BENCH.mkdir(parents=True, exist_ok=True)
    (BENCH / "raw_videos" / "droid").mkdir(parents=True, exist_ok=True)
    (BENCH / "conditioning").mkdir(parents=True, exist_ok=True)
    (BENCH / "reference").mkdir(parents=True, exist_ok=True)

    tasks = pick_tasks()
    print(f"Picked {len(tasks)} tasks:")
    for t in tasks:
        print(f"  [{t['task_short']}] ep_{t['training_episode_id']:04d}  "
              f"{t['language_instruction'][:60]}")

    # Migrate pre-existing single-view layout (idempotent)
    print(f"\nMigrating any pre-existing single-view artifacts to exterior_1/ …")
    migrate_existing_single_view(tasks)

    # Add multi-view (idempotent skips for files already present)
    print(f"\nAdding multi-view (exterior_1/exterior_2/wrist) …")
    eval_tasks: dict[str, dict] = {}
    for t in tasks:
        ts = t["task_short"]
        ep = t["training_episode_id"]
        task_full = t["task_full"]

        ep_video_dir = VIDEO_DIR / str(ep)
        out_video_dir = BENCH / "raw_videos" / "droid" / ts
        out_cond_dir = BENCH / "conditioning" / ts
        out_ref_dir = BENCH / "reference" / task_full
        out_video_dir.mkdir(parents=True, exist_ok=True)
        out_cond_dir.mkdir(parents=True, exist_ok=True)
        out_ref_dir.mkdir(parents=True, exist_ok=True)

        cond_paths_per_view: dict[str, Path] = {}
        view_video_paths: dict[str, str] = {}

        for src_idx, view_name in VIEWS.items():
            src_mp4 = ep_video_dir / f"{src_idx}.mp4"
            if not src_mp4.exists():
                print(f"    [skip] missing {src_mp4}")
                continue
            dst_mp4 = out_video_dir / f"{view_name}.mp4"
            if not dst_mp4.exists():
                shutil.copy(src_mp4, dst_mp4)
                print(f"    + copy {ts}/{view_name}.mp4")
            (out_ref_dir / view_name).mkdir(parents=True, exist_ok=True)
            view_video_paths[view_name] = str(dst_mp4.relative_to(BENCH))

            cond_png = out_cond_dir / f"{view_name}.png"
            if not cond_png.exists():
                cv2.imwrite(str(cond_png), first_frame(dst_mp4))
                print(f"    + cond {ts}/{view_name}.png")
            cond_paths_per_view[view_name] = cond_png

        # 2×2 composite
        composite_path = out_cond_dir / "multiview_2x2.png"
        if cond_paths_per_view:
            if not composite_path.exists():
                make_2x2_composite(cond_paths_per_view, composite_path)
                print(f"    + composite {ts}/multiview_2x2.png")

        eval_tasks[ts] = {
            "task_full":            task_full,
            "language_instruction": t["language_instruction"],
            "training_episode_id":  ep,
            "n_episodes_available": t["n_episodes_available"],
            "views":                sorted(view_video_paths.keys()),
            "view_videos":          view_video_paths,
            "conditioning_per_view": {
                v: str((out_cond_dir / f"{v}.png").relative_to(BENCH))
                for v in cond_paths_per_view
            },
            "conditioning_2x2":     str(composite_path.relative_to(BENCH)),
        }

    eval_tasks_json = BENCH / "eval_tasks.json"
    eval_tasks_json.write_text(json.dumps(eval_tasks, indent=2))
    print(f"\n→ {eval_tasks_json}")

    # Final tree
    print(f"\nFinal structure:")
    for ts in sorted(eval_tasks.keys()):
        task = eval_tasks[ts]
        ext1_refs = len(list((BENCH / "reference" / task["task_full"] / "exterior_1").glob("*.png")))
        ext2_refs = len(list((BENCH / "reference" / task["task_full"] / "exterior_2").glob("*.png")))
        wrist_refs = len(list((BENCH / "reference" / task["task_full"] / "wrist").glob("*.png")))
        print(f"  task {ts} ({task['task_full'][:50]})")
        print(f"    views: {task['views']}")
        print(f"    refs:  exterior_1={ext1_refs}  exterior_2={ext2_refs}  wrist={wrist_refs}")


if __name__ == "__main__":
    main()
