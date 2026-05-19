#!/usr/bin/env python3
"""DROID benchmark entrypoint — thin wrapper around extract_refs_dense.py.

Extracts 182 SAM3-segmented reference frames per DROID task from training mp4s.
Forwards any extra CLI args (e.g. unused) to the generic script.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def main() -> None:
    cmd = [
        sys.executable,
        str(BASE / "scripts" / "extract_refs_dense.py"),
        "--bench", "paper-physical-droid",
        "--tasks_json", "paper-physical-droid/eval_tasks.json",
        "--raw_video_subdir", "raw_videos/droid",
    ] + sys.argv[1:]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
