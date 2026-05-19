#!/usr/bin/env python3
"""Generate Cosmos-Predict2 videos for the DROID benchmark.

Pipeline:
  1. Read paper-physical-droid/eval_tasks.json (task_short → language_instruction)
  2. Build a prompts.json in the format dreamgen_data/generate.py expects:
       [{"task": "<task_full>", "prompt": "<language_instruction>"}, ...]
  3. Invoke dreamgen_data/generate.py with --profile droid using the cosmos venv
     python (the only env with cosmos-predict2 installed).
  4. Outputs land in paper-physical-droid/generated/<task_full>/v0000.mp4 ...

Conditioning images: paper-physical-droid/conditioning/<task_full>.png (already
populated by setup_droid_bench.py — generate.py's fuzzy resolver picks them up
exactly).

Checkpoint: nvidia/Cosmos-Predict2-14B-Sample-GR00T-Dreams-DROID
            (downloaded by dreamgen_data/setup.sh into
             dreamgen_data/checkpoints/nvidia/).

Default: 5 videos per task × 5 tasks = 25 videos. At ~3 min/video (30 steps,
~480p) that's ~75-90 min on 1×H100. Resume-safe: existing v*.mp4 are skipped.

Usage:
    python scripts/generate_droid_cosmos.py [--n_per_task 5] [--num_steps 30]
                                            [--tasks "0_Close the drawer"]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
REPO_ROOT = Path("/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination")
BENCH = REPO_ROOT / "paper-physical-droid"
EVAL_TASKS_JSON = BENCH / "eval_tasks.json"
COND_DIR = BENCH / "conditioning"
SAVE_DIR = BENCH / "generated"

DREAMGEN_DIR = REPO_ROOT / "dreamgen_data"
GENERATE_PY = DREAMGEN_DIR / "generate.py"
COSMOS_VENV_PY = DREAMGEN_DIR / "cosmos-predict2" / ".venv" / "bin" / "python"
CKPT_ROOT = DREAMGEN_DIR / "checkpoints"
DROID_CKPT = CKPT_ROOT / "nvidia" / "Cosmos-Predict2-14B-Sample-GR00T-Dreams-DROID"


def load_eval_tasks() -> list[dict]:
    """Return list of {task, prompt} dicts in the schema generate.py expects."""
    raw = json.loads(EVAL_TASKS_JSON.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{EVAL_TASKS_JSON}: expected top-level dict")
    items = []
    for short, meta in sorted(raw.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0):
        task_full = meta["task_full"]
        prompt = meta.get("language_instruction") or meta.get("prompt") or task_full
        items.append({"task": task_full, "prompt": prompt})
    return items


def build_prompts_json(items: list[dict]) -> Path:
    """Write prompts.json into a tmpdir; return its path."""
    tmp = Path(tempfile.mkdtemp(prefix="droid_prompts_"))
    out = tmp / "prompts_droid.json"
    out.write_text(json.dumps(items, indent=2))
    return out


def preflight() -> None:
    missing = []
    if not EVAL_TASKS_JSON.exists():
        missing.append(f"  eval_tasks.json: {EVAL_TASKS_JSON}")
    if not GENERATE_PY.exists():
        missing.append(f"  generate.py: {GENERATE_PY}")
    if not COSMOS_VENV_PY.exists():
        missing.append(f"  cosmos venv python: {COSMOS_VENV_PY}")
    if not DROID_CKPT.exists():
        missing.append(f"  DROID checkpoint dir: {DROID_CKPT}")
    if not COND_DIR.exists() or not any(COND_DIR.glob("*.png")):
        missing.append(f"  conditioning PNGs in: {COND_DIR}")
    if missing:
        msg = "Missing prerequisites:\n" + "\n".join(missing)
        raise FileNotFoundError(msg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n_per_task", type=int, default=5,
                    help="Videos per task (default 5 → 25 total for 5 tasks)")
    ap.add_argument("--num_steps", type=int, default=30,
                    help="Cosmos denoising steps (default 30, ~3min/video)")
    ap.add_argument("--seed_offset", type=int, default=None,
                    help="Override profile.base_seed (DROID default: 4000)")
    ap.add_argument("--tasks", type=str, default=None,
                    help="Comma-separated subset of task names")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print the subprocess command and exit without running it")
    args = ap.parse_args()

    preflight()
    items = load_eval_tasks()
    print(f"[droid-gen] loaded {len(items)} tasks from {EVAL_TASKS_JSON}")
    for it in items:
        print(f"  - {it['task'][:70]}")

    prompts_json = build_prompts_json(items)
    print(f"[droid-gen] wrote prompts json: {prompts_json}")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(COSMOS_VENV_PY),
        str(GENERATE_PY),
        "--prompts", str(prompts_json),
        "--save_dir", str(SAVE_DIR),
        "--input_dir", str(COND_DIR),
        "--ckpt_root", str(CKPT_ROOT),
        "--profile", "droid",
        "--n_per_task", str(args.n_per_task),
        "--num_steps", str(args.num_steps),
    ]
    if args.seed_offset is not None:
        cmd += ["--seed_offset", str(args.seed_offset)]
    if args.tasks:
        cmd += ["--tasks", args.tasks]

    print(f"[droid-gen] running:\n  {' '.join(cmd)}")
    if args.dry_run:
        print("[droid-gen] --dry_run — exiting without invoking generate.py")
        return

    # generate.py runs heavy GPU work; pass env through so HF cache + token are visible.
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    subprocess.run(cmd, check=True, env=env, cwd=str(DREAMGEN_DIR))
    print("[droid-gen] done.")


if __name__ == "__main__":
    main()
