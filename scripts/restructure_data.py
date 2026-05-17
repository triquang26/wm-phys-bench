#!/usr/bin/env python3
"""Restructure data layout for feature_matching_eval_hallucination.

Moves the three outer data dirs (cosmos_synthetic_data, cosmos_frames_raw,
image_no_bg) from the parent feepe/ directory into this repo's data/ folder
and renames the `high`/`low` split dirs into the new
reference / query/{high,low} layout used downstream.

Idempotent. Default mode is --dry-run; pass --apply to execute.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTER_FEEPE = REPO_ROOT.parent  # /mnt/.../feepe


# Symlinks created by a prior session that bridge into the outer feepe dirs.
SYMLINKS_TO_REMOVE = [
    DATA_DIR / "cosmos_synthetic_data",
    DATA_DIR / "cosmos_frames_raw",
    DATA_DIR / "image_no_bg",
]


@dataclass(frozen=True)
class MovePlan:
    src: Path
    dst: Path

    def describe(self) -> str:
        return f"{self.src}  ->  {self.dst}"


# (outer_src_relative_to_feepe, dst_relative_to_repo_data)
MOVES: list[tuple[str, str]] = [
    ("cosmos_synthetic_data/high", "cosmos_synthetic_data/reference"),
    ("cosmos_synthetic_data/low", "cosmos_synthetic_data/query/low"),
    ("cosmos_frames_raw/high", "cosmos_frames_raw/reference"),
    ("cosmos_frames_raw/low", "cosmos_frames_raw/query/low"),
    ("image_no_bg/high", "reference"),
    ("image_no_bg/low", "query/low"),
]

# Empty placeholder dirs to create after the moves.
PLACEHOLDERS: list[str] = [
    "cosmos_synthetic_data/query/high",
    "cosmos_frames_raw/query/high",
    "query/high",
]

# Outer feepe top-level dirs that should be rmdir'd if empty after moves.
OUTER_SHELLS = ["cosmos_synthetic_data", "cosmos_frames_raw", "image_no_bg"]


def count_files(path: Path, patterns: tuple[str, ...] = ("*",)) -> int:
    if not path.exists():
        return 0
    if path.is_symlink():
        # Resolve through symlink for counting
        path = path.resolve()
        if not path.exists():
            return 0
    total = 0
    for pat in patterns:
        total += sum(1 for _ in path.rglob(pat) if _.is_file())
    return total


class DataRestructurer:
    def __init__(self, repo_root: Path, outer_feepe: Path, apply: bool):
        self.repo_root = repo_root
        self.data_dir = repo_root / "data"
        self.outer_feepe = outer_feepe
        self.apply = apply
        self.errors: list[str] = []

    # ---------- planning ----------

    def plan(self) -> list[MovePlan]:
        plans: list[MovePlan] = []
        for src_rel, dst_rel in MOVES:
            src = self.outer_feepe / src_rel
            dst = self.data_dir / dst_rel
            plans.append(MovePlan(src=src, dst=dst))
        return plans

    # ---------- safety ----------

    def verify_safe_to_move(self, plans: list[MovePlan]) -> bool:
        """Each plan must have either:
        - src exists AND dst doesn't exist  (fresh move), OR
        - src doesn't exist AND dst exists  (already moved; idempotent skip), OR
        - src doesn't exist AND dst doesn't exist (nothing to do — flag),
        We refuse if BOTH src and dst exist with content (would overwrite).
        """
        ok = True
        for p in plans:
            src_exists = p.src.exists() and not p.src.is_symlink()
            dst_exists = p.dst.exists()
            if src_exists and dst_exists:
                dst_nonempty = any(p.dst.iterdir()) if p.dst.is_dir() else True
                if dst_nonempty:
                    self.errors.append(
                        f"REFUSE: both src and dst exist and dst is non-empty: "
                        f"{p.src} -> {p.dst}"
                    )
                    ok = False
            elif not src_exists and not dst_exists:
                # Neither — log but don't fail.
                print(f"  WARN: neither src nor dst exists: {p.describe()}")
        return ok

    # ---------- pre/post inventory ----------

    def inventory(self, label: str) -> None:
        print(f"\n=== Inventory: {label} ===")
        targets = [
            ("data/cosmos_synthetic_data (link or dir)", self.data_dir / "cosmos_synthetic_data"),
            ("data/cosmos_frames_raw (link or dir)", self.data_dir / "cosmos_frames_raw"),
            ("data/image_no_bg (link or dir)", self.data_dir / "image_no_bg"),
            ("data/reference", self.data_dir / "reference"),
            ("data/query/low", self.data_dir / "query" / "low"),
            ("data/query/high", self.data_dir / "query" / "high"),
            ("data/cosmos_synthetic_data/reference", self.data_dir / "cosmos_synthetic_data" / "reference"),
            ("data/cosmos_synthetic_data/query/low", self.data_dir / "cosmos_synthetic_data" / "query" / "low"),
            ("data/cosmos_synthetic_data/query/high", self.data_dir / "cosmos_synthetic_data" / "query" / "high"),
            ("data/cosmos_frames_raw/reference", self.data_dir / "cosmos_frames_raw" / "reference"),
            ("data/cosmos_frames_raw/query/low", self.data_dir / "cosmos_frames_raw" / "query" / "low"),
            ("data/cosmos_frames_raw/query/high", self.data_dir / "cosmos_frames_raw" / "query" / "high"),
            ("outer feepe/cosmos_synthetic_data", self.outer_feepe / "cosmos_synthetic_data"),
            ("outer feepe/cosmos_frames_raw", self.outer_feepe / "cosmos_frames_raw"),
            ("outer feepe/image_no_bg", self.outer_feepe / "image_no_bg"),
        ]
        for name, p in targets:
            if p.is_symlink():
                tgt = os.readlink(p)
                print(f"  [symlink] {name:55s} -> {tgt}")
            elif p.exists():
                n_all = count_files(p)
                n_png = count_files(p, ("*.png",))
                n_mp4 = count_files(p, ("*.mp4",))
                print(f"  [exists ] {name:55s} files={n_all:5d} png={n_png:5d} mp4={n_mp4:3d}")
            else:
                print(f"  [absent ] {name}")

    # ---------- ops ----------

    def remove_symlinks(self) -> None:
        for link in SYMLINKS_TO_REMOVE:
            if link.is_symlink():
                print(f"  unlink: {link}")
                if self.apply:
                    link.unlink()
            elif link.exists():
                print(f"  (not a symlink, leaving) {link}")
            else:
                print(f"  (absent, ok) {link}")

    def do_moves(self, plans: list[MovePlan]) -> None:
        for p in plans:
            src_exists = p.src.exists() and not p.src.is_symlink()
            dst_exists = p.dst.exists()
            if not src_exists and dst_exists:
                print(f"  SKIP (already moved): {p.describe()}")
                continue
            if not src_exists and not dst_exists:
                print(f"  SKIP (src missing, nothing to do): {p.describe()}")
                continue
            # src exists, dst missing -> move
            print(f"  MV: {p.describe()}")
            if self.apply:
                p.dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(p.src), str(p.dst))

    def create_placeholders(self) -> None:
        for rel in PLACEHOLDERS:
            d = self.data_dir / rel
            if d.exists():
                print(f"  placeholder exists: {d}")
                continue
            print(f"  mkdir placeholder: {d}")
            if self.apply:
                d.mkdir(parents=True, exist_ok=True)

    def cleanup_outer_shells(self) -> None:
        for name in OUTER_SHELLS:
            shell = self.outer_feepe / name
            if not shell.exists():
                print(f"  outer shell already gone: {shell}")
                continue
            try:
                remaining = list(shell.iterdir())
            except NotADirectoryError:
                print(f"  outer shell is not a dir, skip: {shell}")
                continue
            if remaining:
                print(f"  outer shell NOT empty, leaving: {shell} "
                      f"(contains: {[p.name for p in remaining]})")
                continue
            print(f"  rmdir: {shell}")
            if self.apply:
                shell.rmdir()

    # ---------- driver ----------

    def apply_all(self) -> None:
        plans = self.plan()
        print("\n=== Planned moves ===")
        for p in plans:
            print(f"  {p.describe()}")

        print("\n=== Safety check ===")
        if not self.verify_safe_to_move(plans):
            for e in self.errors:
                print(f"  ERROR: {e}")
            raise SystemExit(2)
        print("  ok")

        print("\n=== Step 1: remove bridge symlinks ===")
        self.remove_symlinks()

        print("\n=== Step 2: move outer dirs into data/ ===")
        self.do_moves(plans)

        print("\n=== Step 3: create empty placeholders ===")
        self.create_placeholders()

        print("\n=== Step 4: cleanup empty outer shells ===")
        self.cleanup_outer_shells()

    def verify_post(self) -> None:
        print("\n=== Post-move verification ===")
        checks = [
            ("data/reference png count (>=1219)",
             self.data_dir / "reference", "*.png", 1219),
            ("data/query/low png count (>=1219)",
             self.data_dir / "query" / "low", "*.png", 1219),
            ("data/cosmos_synthetic_data/reference mp4 (==23)",
             self.data_dir / "cosmos_synthetic_data" / "reference", "*.mp4", 23),
            ("data/cosmos_synthetic_data/query/low mp4 (==23)",
             self.data_dir / "cosmos_synthetic_data" / "query" / "low", "*.mp4", 23),
            ("data/cosmos_frames_raw/reference png (>=1219)",
             self.data_dir / "cosmos_frames_raw" / "reference", "*.png", 1219),
            ("data/cosmos_frames_raw/query/low png (>=1219)",
             self.data_dir / "cosmos_frames_raw" / "query" / "low", "*.png", 1219),
        ]
        all_ok = True
        for label, path, pat, threshold in checks:
            n = count_files(path, (pat,))
            ok = n >= threshold
            mark = "OK " if ok else "FAIL"
            print(f"  [{mark}] {label}: got {n}")
            if not ok:
                all_ok = False
        # symlink check
        symlinks_left = [p for p in self.data_dir.iterdir() if p.is_symlink()]
        if symlinks_left:
            print(f"  [FAIL] symlinks still in data/: {symlinks_left}")
            all_ok = False
        else:
            print(f"  [OK ] no symlinks remain in data/")
        if not all_ok:
            print("\n  Post-verification reported failures.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--apply", action="store_true",
                   help="Actually perform moves (default is dry-run).")
    g.add_argument("--dry-run", action="store_true",
                   help="Print plan only (default).")
    args = ap.parse_args()

    apply = args.apply  # dry-run is the default

    print(f"repo_root   = {REPO_ROOT}")
    print(f"data_dir    = {DATA_DIR}")
    print(f"outer_feepe = {OUTER_FEEPE}")
    print(f"mode        = {'APPLY' if apply else 'DRY-RUN'}")

    r = DataRestructurer(REPO_ROOT, OUTER_FEEPE, apply=apply)
    r.inventory("BEFORE")
    r.apply_all()
    r.inventory("AFTER")
    if apply:
        r.verify_post()
    else:
        print("\n(dry-run) skipping post-verification — re-run with --apply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
