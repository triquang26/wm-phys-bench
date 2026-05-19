#!/usr/bin/env python3
"""GR1 benchmark with CROSS-TASK reference pool + DINOv2 k-NN.

Approach:
  1. Pool all reference frames from 5 tasks into a single "POOL" task (~200 frames).
  2. Symlink all query frames (held-out real + Cosmos-generated) under POOL too,
     prefixing filenames with the original task slug to preserve identity.
  3. Run warp_score calibrate + detect with adaptive_ref_selector=True, k=50.
     -> For each query, DINOv2 picks top-50 NN from the entire pool (cross-task).
  4. Eval: macro AUROC over POOL, plus per-original-task breakdown.

Why cross-task pool fixes the v2 inversion (AUROC=0.43):
  Cosmos-generated frames are conditioned on same-task refs -> they look
  like same-task training data -> per-task k-NN gives them low H_score.
  Cross-task pool removes this conditioning advantage: generated frames
  must compete against the BEST visual match anywhere in the pool, and
  subtle generation artifacts will show up as higher warp residuals.

Usage (groot env):
    python scripts/run_gr1_pool_benchmark.py [--k 50] [--skip_calib] [--skip_detect]
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
# Data lives in the main repo workdir, not the worktree; resolve via env override or hardcoded path.
REPO_ROOT = Path("/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination")
BENCH = REPO_ROOT / "paper-physical-gr1"
POOL = BENCH / "pool"

TASKS = [
    "1_Use the right hand to pick up green bok choy from tan table right side to bottom level of wire basket.",
    "2_Use the right hand to pick up rubik's cube from top level of the shelf to bottom level of the shelf.",
    "3_Use the right hand to pick up banana from teal plate to wooden table.",
    "4_Use the left hand to pick up dragonfruit from pink plate to teal plate.",
    "6_Use the right hand to pick up orange from middle of table to bottom white shelf.",
]
TASK_NUM = {t: t.split("_")[0] for t in TASKS}

POOL_TASK = "POOL"


# ── Step 1: build symlink layout ─────────────────────────────────────────────

def build_pool_layout() -> Path:
    """Symlink all refs + queries into POOL/, preserving original-task tag in filename."""
    if POOL.exists():
        shutil.rmtree(POOL)

    ref_dir = POOL / "reference" / POOL_TASK
    qh_dir = POOL / "query_high" / POOL_TASK
    ql_dir = POOL / "query_low" / POOL_TASK
    for d in (ref_dir, qh_dir, ql_dir):
        d.mkdir(parents=True, exist_ok=True)

    mapping: dict[str, str] = {}  # frame_stem -> original_task

    n_ref = n_qh = n_ql = 0
    for task in TASKS:
        slug = TASK_NUM[task]
        for p in sorted((BENCH / "reference_calib" / task).glob("*.png")):
            new_name = f"t{slug}__{p.name}"
            (ref_dir / new_name).symlink_to(p.resolve())
            mapping[Path(new_name).stem] = task
            n_ref += 1
        for p in sorted((BENCH / "reference_heldout" / task).glob("*.png")):
            new_name = f"t{slug}__{p.name}"
            (qh_dir / new_name).symlink_to(p.resolve())
            mapping[Path(new_name).stem] = task
            n_qh += 1
        for p in sorted((BENCH / "query_frames" / task).glob("*.png")):
            new_name = f"t{slug}__{p.name}"
            (ql_dir / new_name).symlink_to(p.resolve())
            mapping[Path(new_name).stem] = task
            n_ql += 1

    print(f"Pool layout: refs={n_ref}  query_high={n_qh}  query_low={n_ql}")
    (POOL / "frame_to_task.json").write_text(json.dumps(mapping, indent=2))
    return POOL


# ── Step 2: write pool-level labels ──────────────────────────────────────────

def write_pool_labels() -> Path:
    """Labels for the unified POOL task: high=0, low=1."""
    labels_path = POOL / "results" / "labels.csv"
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in sorted((POOL / "query_high" / POOL_TASK).glob("*.png")):
        rows.append({"task": POOL_TASK, "split": "high", "frame": p.stem, "label": 0})
    for p in sorted((POOL / "query_low" / POOL_TASK).glob("*.png")):
        rows.append({"task": POOL_TASK, "split": "low", "frame": p.stem, "label": 1})

    with open(labels_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "split", "frame", "label"])
        w.writeheader()
        w.writerows(rows)
    n0 = sum(1 for r in rows if r["label"] == 0)
    n1 = sum(1 for r in rows if r["label"] == 1)
    print(f"Pool labels: {n0} real + {n1} generated -> {labels_path}")
    return labels_path


# ── Step 3: build config ─────────────────────────────────────────────────────

def build_pool_config(k: int) -> Path:
    """Generate a pool-specific YAML config (does not overwrite paper_gr1.yaml)."""
    cfg_path = BASE / "warp_score" / "configs" / "paper_gr1_pool.yaml"
    cfg = {
        "signal_names": ["ivar_maha", "peak_maha"],
        "fuser": "cauchy",
        "per_pixel_calibration": True,
        "use_precision": True,
        "bidirectional": True,
        "n_min_refs": 10,
        "adaptive_ref_selector": True,
        "k_per_frame": k,
        "dino_model": "dinov2_vits14",
        "reference_dir": str(POOL / "reference"),
        "query_high_dir": str(POOL / "query_high"),
        "query_low_dir": str(POOL / "query_low"),
        "artifacts_dir": str(POOL / "results"),
    }
    import yaml
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    print(f"Pool config -> {cfg_path}")
    return cfg_path


# ── Step 4: run warp_score subcommands ───────────────────────────────────────

def run_warp_score(subcmd: str, extra: list[str], cfg: Path) -> None:
    cmd = [sys.executable, "-m", "warp_score", "--config", str(cfg), subcmd] + extra
    print(f"$ python -m warp_score --config {cfg.name} {subcmd} {' '.join(extra)}")
    subprocess.run(cmd, check=True, cwd=BASE)


# ── Step 5: per-original-task eval ───────────────────────────────────────────

def eval_per_task(summary_csv: Path, mapping_path: Path) -> dict:
    """Compute per-original-task AUROC from pool predictions."""
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score

    mapping = json.loads(mapping_path.read_text())
    rows = list(csv.DictReader(open(summary_csv)))

    # Group by original task
    by_task: dict[str, list[tuple[int, float]]] = {}
    for r in rows:
        orig_task = mapping.get(r["frame"])
        if orig_task is None:
            continue
        label = 0 if r["split"] == "high" else 1
        by_task.setdefault(orig_task, []).append((label, float(r["H_score"])))

    overall_y, overall_s = [], []
    per_task = {}
    for task, pairs in by_task.items():
        y = np.array([p[0] for p in pairs])
        s = np.array([p[1] for p in pairs])
        overall_y.extend(y.tolist())
        overall_s.extend(s.tolist())
        if len(set(y.tolist())) < 2:
            continue
        per_task[task] = {
            "n": len(y),
            "n_pos": int(y.sum()),
            "auroc": float(roc_auc_score(y, s)),
            "ap": float(average_precision_score(y, s)),
            "mean_score_real": float(s[y == 0].mean()),
            "mean_score_gen":  float(s[y == 1].mean()),
        }

    overall_y = np.array(overall_y)
    overall_s = np.array(overall_s)
    report = {
        "overall": {
            "n": len(overall_y),
            "n_pos": int(overall_y.sum()),
            "auroc": float(roc_auc_score(overall_y, overall_s)),
            "ap": float(average_precision_score(overall_y, overall_s)),
            "mean_score_real": float(overall_s[overall_y == 0].mean()),
            "mean_score_gen":  float(overall_s[overall_y == 1].mean()),
        },
        "per_task": per_task,
    }
    return report


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=50, help="k_per_frame for k-NN")
    ap.add_argument("--skip_layout", action="store_true")
    ap.add_argument("--skip_calib", action="store_true")
    ap.add_argument("--skip_detect", action="store_true")
    args = ap.parse_args()

    print("\n=== Step 1: build pool symlink layout ===")
    if not args.skip_layout:
        build_pool_layout()
    else:
        print("[skip] layout")

    print("\n=== Step 2: write pool labels ===")
    labels_path = write_pool_labels()

    print(f"\n=== Step 3: build config (k={args.k}) ===")
    cfg = build_pool_config(args.k)

    if not args.skip_calib:
        print("\n=== Step 4: calibrate (LOO over pool) ===")
        run_warp_score("calibrate", [], cfg)
    else:
        print("[skip] calibrate")

    if not args.skip_detect:
        print("\n=== Step 5: detect (290 queries) ===")
        run_warp_score("detect", [], cfg)
    else:
        print("[skip] detect")

    print("\n=== Step 6: pool-level eval ===")
    summary_csv = POOL / "results" / "summary.csv"
    eval_report = POOL / "results" / "eval_report.json"
    if summary_csv.exists():
        run_warp_score("eval", [
            "--labels", str(labels_path),
            "--pred",   str(summary_csv),
            "--out",    str(eval_report),
        ], cfg)
        print(json.dumps(json.loads(eval_report.read_text()), indent=2))

    print("\n=== Step 7: per-original-task eval ===")
    if summary_csv.exists():
        mapping_path = POOL / "frame_to_task.json"
        report = eval_per_task(summary_csv, mapping_path)
        out = POOL / "results" / "eval_per_task.json"
        out.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
