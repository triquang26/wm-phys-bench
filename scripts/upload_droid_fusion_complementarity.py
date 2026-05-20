#!/usr/bin/env python3
"""Upload DROID cycle-only vs knn-only catches to HF (twanghcmut/wmbench).

Mirror of upload_fusion_complementarity.py but for DROID. Uses
paper-physical-droid/per_task_dense_eval/per_task_ratio_table.csv as input
and uploads to twanghcmut/wmbench/droid/fusion_analysis/.

These are gens where the cycle and k-NN modalities DISAGREE — concrete
evidence the two signals are orthogonal and the Cauchy fusion adds value.
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path("/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination")
BENCH = REPO_ROOT / "paper-physical-droid"
HF_REPO = "twanghcmut/wmbench"
HF_PREFIX = "droid/fusion_analysis"


def load_task_map() -> dict[str, str]:
    """Return {task_short: task_full} from eval_tasks.json."""
    raw = json.loads((BENCH / "eval_tasks.json").read_text())
    return {short: meta["task_full"] for short, meta in raw.items()}


def main() -> None:
    from huggingface_hub import HfApi
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if not token_path.exists():
        sys.exit(f"HF token not found at {token_path}. Run `huggingface-cli login` first.")
    token = token_path.read_text().strip()
    api = HfApi(token=token)

    task_full = load_task_map()
    print(f"[droid-fusion] {len(task_full)} tasks mapped")

    csv_path = BENCH / "per_task_dense_eval" / "per_task_ratio_table.csv"
    if not csv_path.exists():
        sys.exit(f"missing per_task_ratio_table.csv: {csv_path}\n"
                 f"Run eval_droid.py + compute_per_task_ratio.py first.")
    rows = list(csv.DictReader(open(csv_path)))
    gens = [r for r in rows if r["type"] == "GEN"]
    knn_only = [r for r in gens
                if float(r["ratio_cycle"]) <= 1.0 and float(r["ratio_knn"]) > 1.0]
    cycle_only = [r for r in gens
                  if float(r["ratio_cycle"]) > 1.0 and float(r["ratio_knn"]) <= 1.0]
    both = [r for r in gens
            if float(r["ratio_cycle"]) > 1.0 and float(r["ratio_knn"]) > 1.0]
    neither = [r for r in gens
               if float(r["ratio_cycle"]) <= 1.0 and float(r["ratio_knn"]) <= 1.0]

    print(f"[droid-fusion] gens={len(gens)}: both={len(both)} "
          f"cycle_only={len(cycle_only)} knn_only={len(knn_only)} neither={len(neither)}")

    def upload(group: str, items: list[dict]) -> None:
        for r in items:
            task_s = r["task"]
            task_f = task_full.get(task_s)
            if task_f is None:
                print(f"  [skip] unknown task_short={task_s}")
                continue
            src = BENCH / "generated" / task_f / r["video"]
            if not src.exists():
                print(f"  [skip] missing {src}")
                continue
            dst = f"{HF_PREFIX}/{group}/task_{task_s}__{r['video']}"
            api.upload_file(path_or_fileobj=str(src), path_in_repo=dst,
                            repo_id=HF_REPO, repo_type="dataset")
            print(f"  → {dst}   cycle={float(r['ratio_cycle']):.3f}  "
                  f"knn={float(r['ratio_knn']):.3f}  "
                  f"fused={float(r['ratio_fused']):.3f}")

    print(f"\n[droid-fusion] uploading {len(cycle_only)} cycle-only catches …")
    upload("cycle_only", cycle_only)
    print(f"\n[droid-fusion] uploading {len(knn_only)} knn-only catches …")
    upload("knn_only", knn_only)

    # ── README ─────────────────────────────────────────────────────────
    def fmt_table(items: list[dict]) -> str:
        out = "| task | video | ratio_cycle | ratio_knn | ratio_fused |\n"
        out += "|---|---|---|---|---|\n"
        for r in items:
            out += (f"| {r['task']} | {r['video']} | "
                    f"{float(r['ratio_cycle']):.3f} | "
                    f"{float(r['ratio_knn']):.3f} | "
                    f"**{float(r['ratio_fused']):.3f}** |\n")
        return out

    readme = f"""# Fusion Complementarity Analysis (DROID)

Concrete evidence that cycle (temporal self-consistency) and k-NN (appearance
match to training distribution) are **orthogonal anomaly signals** on the
DROID benchmark — each catches gens the other misses, justifying their
Cauchy fusion.

## Selection criterion (from per_task_ratio_table.csv)

- **cycle_only/**  ratio_cycle > 1.0  AND  ratio_knn ≤ 1.0   (cycle flags, knn says clean)
- **knn_only/**    ratio_cycle ≤ 1.0  AND  ratio_knn > 1.0   (cycle says clean, knn flags)

Filename pattern: `task_<N>__v<MMMM>.mp4`.

## Cycle-only catches ({len(cycle_only)} videos)

Temporal artifacts (physics violation, teleportation, arm jitter between
frames) but rendered appearance still looks task-like.

{fmt_table(cycle_only)}

## k-NN-only catches ({len(knn_only)} videos)

Out-of-distribution relative to training (wrong scene, wrong object,
unusual robot pose) but the temporal evolution within the video is smooth.

{fmt_table(knn_only)}

## Complementarity matrix (DROID gens, n={len(gens)})

| group | count |
|---|---|
| caught by both | {len(both)} |
| cycle-only | {len(cycle_only)} |
| knn-only | {len(knn_only)} |
| neither | {len(neither)} |

Per-task ratio preserves FPR=0% on real training videos for all 3 modalities.
"""
    tmp = Path(tempfile.mkdtemp(prefix="droid_fusion_"))
    (tmp / "README.md").write_text(readme)
    api.upload_file(path_or_fileobj=str(tmp / "README.md"),
                    path_in_repo=f"{HF_PREFIX}/README.md",
                    repo_id=HF_REPO, repo_type="dataset")
    print(f"\n→ {HF_PREFIX}/README.md")

    print("\n=== Done ===")
    print(f"https://huggingface.co/datasets/{HF_REPO}/tree/main/{HF_PREFIX}")


if __name__ == "__main__":
    main()
