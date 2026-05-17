"""Build labels.csv from harvested query directories.

Scans query_high_dir/<task>/*.png  → label=0  (clean held-out query)
Scans query_low_dir/<task>/*.png   → label=1  (potentially hallucinated)

Output CSV columns:
    task   : parent directory name (task identifier)
    frame  : absolute path to the PNG
    split  : "high" or "low"
    label  : 0 or 1

Usage:
    python scripts/build_weak_labels.py \
        --query_high_dir data/query/high \
        --query_low_dir  data/query/low \
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
                "task": task,
                "frame": str(png),
                "split": split,
                "label": label,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build weak labels CSV from data/query/{high,low} directories."
    )
    parser.add_argument(
        "--query_high_dir",
        required=True,
        help="Path to data/query/high (label=0, clean held-out queries)",
    )
    parser.add_argument(
        "--query_low_dir",
        required=True,
        help="Path to data/query/low (label=1, hallucination candidates)",
    )
    parser.add_argument(
        "--out",
        default="labels.csv",
        help="Output CSV path (default: labels.csv)",
    )
    args = parser.parse_args()

    query_high_dir = Path(args.query_high_dir)
    query_low_dir = Path(args.query_low_dir)
    out_path = Path(args.out)

    if not query_high_dir.exists():
        print(f"ERROR: query_high_dir not found: {query_high_dir}", file=sys.stderr)
        sys.exit(1)
    if not query_low_dir.exists():
        print(f"ERROR: query_low_dir not found: {query_low_dir}", file=sys.stderr)
        sys.exit(1)

    rows = collect(query_high_dir, label=0, split="high") + collect(
        query_low_dir, label=1, split="low"
    )

    if not rows:
        print(
            "WARNING: no PNG files found under query_high_dir or query_low_dir.",
            file=sys.stderr,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["task", "frame", "split", "label"]
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
