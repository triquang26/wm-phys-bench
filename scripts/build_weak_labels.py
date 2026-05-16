"""Build labels.csv from harvested image_no_bg directories.

Scans high/<task>/*.png  → label=0  (clean reference)
Scans low/<task>/*.png   → label=1  (potentially hallucinated)

Output CSV columns:
    frame  : relative path from repo root
             e.g. ../image_no_bg/high/0_Open_the_box/frame_0001.png
    task   : parent directory name (task identifier)
    label  : 0 or 1
    split  : "high" or "low"

Usage:
    python scripts/build_weak_labels.py \
        --high_dir ../image_no_bg/high \
        --low_dir  ../image_no_bg/low \
        --out labels.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def collect(root: Path, label: int, split: str) -> list[dict]:
    """Return a list of row dicts for all PNGs under root."""
    rows = []
    for png in sorted(root.glob("**/*.png")):
        task = png.parent.name
        rows.append(
            {
                "frame": str(png),
                "task": task,
                "label": label,
                "split": split,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build weak labels CSV from image_no_bg/{high,low} directories."
    )
    parser.add_argument(
        "--high_dir",
        required=True,
        help="Path to image_no_bg/high (label=0, clean references)",
    )
    parser.add_argument(
        "--low_dir",
        required=True,
        help="Path to image_no_bg/low (label=1, hallucination candidates)",
    )
    parser.add_argument(
        "--out",
        default="labels.csv",
        help="Output CSV path (default: labels.csv)",
    )
    args = parser.parse_args()

    high_dir = Path(args.high_dir)
    low_dir = Path(args.low_dir)
    out_path = Path(args.out)

    if not high_dir.exists():
        print(f"ERROR: high_dir not found: {high_dir}", file=sys.stderr)
        sys.exit(1)
    if not low_dir.exists():
        print(f"ERROR: low_dir not found: {low_dir}", file=sys.stderr)
        sys.exit(1)

    rows = collect(high_dir, label=0, split="high") + collect(low_dir, label=1, split="low")

    if not rows:
        print("WARNING: no PNG files found under high_dir or low_dir.", file=sys.stderr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["frame", "task", "label", "split"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_high = sum(1 for r in rows if r["split"] == "high")
    n_low = sum(1 for r in rows if r["split"] == "low")
    n_tasks = len({r["task"] for r in rows})
    print(
        f"Wrote {len(rows)} rows to {out_path}  "
        f"(high={n_high}, low={n_low}, tasks={n_tasks})"
    )


if __name__ == "__main__":
    main()
