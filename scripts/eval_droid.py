#!/usr/bin/env python3
"""DROID benchmark entrypoint — thin wrapper around eval_per_task_dense_null.py.

Runs the WarpDyn fusion pipeline on the DROID benchmark (5 tasks).
Forwards any extra CLI args (e.g. --no-use_knn, --out_suffix) through.

Usage:
    python scripts/eval_droid.py [--no-use_knn] [--out_suffix _foo]
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def main() -> None:
    cmd = [
        sys.executable,
        str(BASE / "scripts" / "eval_per_task_dense_null.py"),
        "--bench", "paper-physical-droid",
        "--tasks_json", "paper-physical-droid/eval_tasks.json",
        "--raw_video_subdir", "raw_videos/droid",
    ] + sys.argv[1:]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
