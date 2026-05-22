#!/usr/bin/env python3
"""Sweep 4 task-1 query videos through benchmark_one_task with pad-to-square code."""
import subprocess
import sys
from pathlib import Path

REPO = Path("/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination")
BENCH_SCRIPT = REPO / ".claude/worktrees/feat-knn-pool-gr1/scripts/benchmark_one_task.py"
PYTHON = sys.executable

VIDEOS = [
    ("1_pad_gr1tuned",         REPO / "paper-physical-gr1/generated/1_Use the right hand to pick up green bok choy from tan table right side to bottom level of wire basket./v0000.mp4"),
    ("1_pad_base_v2000",       REPO / "paper-physical-gr1/generated_base_cosmos/_keep_v2000_orig.mp4"),
    ("1_pad_base_v2001",       REPO / "paper-physical-gr1/generated_base_cosmos/1_Use the right hand to pick up green bok choy from tan table right side to bottom level of wire basket./v0001.mp4"),
    ("1_pad_base_v2002",       REPO / "paper-physical-gr1/generated_base_cosmos/1_Use the right hand to pick up green bok choy from tan table right side to bottom level of wire basket./v0002.mp4"),
]

OUT_BASE = REPO / ".claude/worktrees/feat-knn-pool-gr1/outputs/benchmark_demo"

for name, vid in VIDEOS:
    out = OUT_BASE / name
    print(f"\n{'='*70}\nRunning bench: {name}\n  query: {vid}\n  out:   {out}\n{'='*70}")
    if not vid.exists():
        print(f"  [skip] missing: {vid}")
        continue
    rc = subprocess.run([PYTHON, str(BENCH_SCRIPT),
                         "--query", str(vid),
                         "--out_dir", str(out)],
                        check=False).returncode
    print(f"  exit code: {rc}")
