"""Apply SAM3 background removal to all PNGs in src_root/{high,low}/<task>/*.png
→ out_root/{high,low}/<task>/*.png.

Skips frames that already exist (resume-safe).
Uses the SAM3Segmenter from feepe/sam3.py.

Usage:
    python run_bg_removal_batch.py \
        --src_root cosmos_frames_raw \
        --out_root image_no_bg \
        --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


BG_COLOR = (127, 127, 127)  # matches sam3.py convention


def process_split(seg, src_split: Path, out_split: Path, prompts: list[str],
                  threshold: float = 0.3) -> tuple[int, int, int]:
    """Process one split dir. Returns (saved, skipped, errors)."""
    saved = skipped = errors = 0

    task_dirs = sorted(d for d in src_split.iterdir() if d.is_dir())
    for task_dir in task_dirs:
        pngs = sorted(task_dir.glob("*.png"))
        out_task = out_split / task_dir.name
        out_task.mkdir(parents=True, exist_ok=True)

        for png in tqdm(pngs, desc=f"  {task_dir.name[:50]}", leave=False):
            out_path = out_task / png.name
            if out_path.exists():
                skipped += 1
                continue
            try:
                alpha = seg.segment_multi_prompt(png, prompts, threshold)
                if alpha is None:
                    # No robot detected — write plain frame with bg=(127,127,127)
                    # as a fallback so downstream pipeline still has data
                    img = Image.open(png).convert("RGB")
                    bg = Image.new("RGB", img.size, BG_COLOR)
                    bg.save(out_path)
                    errors += 1
                    continue
                result = seg.remove_background(png, alpha)
                result.save(out_path)
                saved += 1
            except Exception as e:
                print(f"\n  ERROR {png.name}: {e}")
                errors += 1

    return saved, skipped, errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", required=True,
                    help="Dir with high/ and low/ subdirs of raw PNGs")
    ap.add_argument("--out_root", required=True,
                    help="Output dir (e.g. image_no_bg)")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    src = Path(args.src_root).resolve()
    out = Path(args.out_root).resolve()

    # sam3.py lives in the repo root (feature_matching_eval_hallucination/)
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from sam3 import SAM3Segmenter, PROMPTS

    print(f"Loading SAM3 model on {args.device}…")
    seg = SAM3Segmenter()

    total_saved = total_skipped = total_errors = 0
    for split in ("high", "low"):
        src_split = src / split
        if not src_split.exists():
            print(f"[skip] {src_split} not found")
            continue
        out_split = out / split
        print(f"\n[{split}] processing {src_split}")
        s, sk, e = process_split(seg, src_split, out_split, PROMPTS, args.threshold)
        print(f"  saved={s}  skipped={sk}  errors={e}")
        total_saved += s
        total_skipped += sk
        total_errors += e

    print(f"\n✓ Done.  saved={total_saved}  skipped={total_skipped}  errors={total_errors}")
    print(f"Output → {out}")


if __name__ == "__main__":
    main()
