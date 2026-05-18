"""Build test/ benchmark folder for hallucination detection testing.

Uses the first N gr00t tasks (alphabetically) as the task set:
  - query/high  : random SAMPLES_PER_TASK SAM3-processed frames per task, copied
                  from high_src_dir (default: data/query/high — already processed)
  - query/low   : all frames copied from low_dir for the same tasks
  - reference   : all frames copied from ref_dir for the same tasks

NOTE: query/high source is data/query/high/ (already SAM3-processed gr00t frames).
      Pass --high_src_dir to override.

Usage (groot env):
    python scripts/build_bench_test.py \
        --gr00t_video_dir data/cosmos_synthetic_data/query/gr00t \
        --high_src_dir data/query/high \
        --low_dir  data/query/low \
        --ref_dir  data/reference \
        --out_root test \
        --n_tasks 5 \
        --samples_per_task 20
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def build(
    gr00t_video_dir: Path,
    high_src_dir: Path,
    low_dir: Path,
    ref_dir: Path,
    out_root: Path,
    n_tasks: int,
    samples_per_task: int,
) -> None:
    # --- discover first n_tasks from gr00t video dir (sorted) ---
    task_dirs = sorted(d for d in gr00t_video_dir.iterdir() if d.is_dir())[:n_tasks]
    if not task_dirs:
        print(f"[bench] ERROR: no task dirs found in {gr00t_video_dir}")
        sys.exit(1)
    tasks = [d.name for d in task_dirs]
    print(f"[bench] {len(tasks)} tasks:")
    for t in tasks:
        print(f"  {t[:80]}")

    # --- stage 1: random sample from high_src_dir -> out_root/query/high/ ---
    random.seed(42)
    high_out = out_root / "query" / "high"
    high_out.mkdir(parents=True, exist_ok=True)
    print(f"\n[bench] Stage 1: copy {samples_per_task} random frames/task -> {high_out}")
    for task in tasks:
        src_task = high_src_dir / task
        dst_task = high_out / task
        if not src_task.exists():
            print(f"  WARN: {src_task} not found, skipping")
            continue
        pngs = sorted(src_task.glob("*.png"))
        if not pngs:
            print(f"  WARN: no PNGs in {src_task}, skipping")
            continue
        dst_task.mkdir(parents=True, exist_ok=True)
        sample = random.sample(pngs, min(samples_per_task, len(pngs)))
        for p in sample:
            dst = dst_task / p.name
            if not dst.exists():
                shutil.copy2(p, dst)
        print(f"  {len(sample)} frames  {task[:70]}")

    # --- stage 2: random sample query/low (same count as high) ---
    low_out = out_root / "query" / "low"
    low_out.mkdir(parents=True, exist_ok=True)
    print(f"\n[bench] Stage 2: copy {samples_per_task} random frames/task -> {low_out}")
    for task in tasks:
        src_task = low_dir / task
        dst_task = low_out / task
        if not src_task.exists():
            print(f"  WARN: {src_task} not found, skipping")
            continue
        pngs = sorted(src_task.glob("*.png"))
        if not pngs:
            print(f"  WARN: no PNGs in {src_task}, skipping")
            continue
        dst_task.mkdir(parents=True, exist_ok=True)
        sample = random.sample(pngs, min(samples_per_task, len(pngs)))
        for p in sample:
            dst = dst_task / p.name
            if not dst.exists():
                shutil.copy2(p, dst)
        print(f"  {len(sample)} frames  {task[:70]}")

    # --- stage 3: copy reference (all frames) ---
    for split_src, split_name in [(ref_dir, "reference")]:
        split_dst = out_root / split_name
        split_dst.mkdir(parents=True, exist_ok=True)
        print(f"\n[bench] Stage 3: copy {split_name} -> {split_dst}")
        for task in tasks:
            src = split_src / task
            dst = split_dst / task
            if not src.exists():
                print(f"  WARN: {src} not found, skipping")
                continue
            if dst.exists():
                print(f"  skip (exists): {task[:70]}")
                continue
            shutil.copytree(src, dst)
            n = len(list(dst.glob("*.png")))
            print(f"  {n} PNGs copied  {task[:70]}")

    # --- summary ---
    n_high = len(list((out_root / "query" / "high").rglob("*.png")))
    n_low  = len(list((out_root / "query"  / "low").rglob("*.png")))
    n_ref  = len(list((out_root / "reference").rglob("*.png")))
    print(f"\n[bench] Done.")
    print(f"  query/high : {n_high} PNGs  ({n_high // max(len(tasks), 1)} per task avg)")
    print(f"  query/low  : {n_low} PNGs")
    print(f"  reference  : {n_ref} PNGs")
    print(f"  output     : {out_root.resolve()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gr00t_video_dir", type=Path,
                    default=REPO_ROOT / "data/cosmos_synthetic_data/query/gr00t")
    ap.add_argument("--high_src_dir", type=Path,
                    default=REPO_ROOT / "data/query/high",
                    help="Source of SAM3-processed query/high frames (default: data/query/high)")
    ap.add_argument("--low_dir",  type=Path, default=REPO_ROOT / "data/query/low")
    ap.add_argument("--ref_dir",  type=Path, default=REPO_ROOT / "data/reference")
    ap.add_argument("--out_root", type=Path, default=REPO_ROOT / "test")
    ap.add_argument("--n_tasks",  type=int, default=5)
    ap.add_argument("--samples_per_task", type=int, default=20)
    args = ap.parse_args()

    build(
        gr00t_video_dir=args.gr00t_video_dir,
        high_src_dir=args.high_src_dir,
        low_dir=args.low_dir,
        ref_dir=args.ref_dir,
        out_root=args.out_root,
        n_tasks=args.n_tasks,
        samples_per_task=args.samples_per_task,
    )


if __name__ == "__main__":
    main()
