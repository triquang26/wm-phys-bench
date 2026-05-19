#!/usr/bin/env python3
"""Upload the DROID benchmark assets + results to twanghcmut/wmbench.

Mirrors upload_wmbench.py but writes under the wmbench/droid/ prefix.

Layout uploaded:

    twanghcmut/wmbench/droid/
      README.md
      training/<task_short>.mp4
      generated/<task_full>/v0000.mp4 ...
      reference/<task_full>/frame_NNNN.png
      null_per_task/<task_short>.npz
      results/
        per_task_dense_table.csv
        per_task_ratio_table.csv
        per_task_ratio_ranking.md
        viz/                 (optional, uploaded only if exists)
      method.md              (copy of WARPDYN_METHOD.md)

Usage (env with huggingface_hub installed):
    python scripts/upload_droid_wmbench.py
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path("/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination")
BENCH_NAME = "paper-physical-droid"
BENCH = REPO_ROOT / BENCH_NAME
RAW_VIDEO_SUBDIR = "raw_videos/droid"

HF_REPO = "twanghcmut/wmbench"
HF_PREFIX = "droid"


def task_short_id(task_full: str) -> str:
    return f"task_{task_full.split('_')[0]}"


def load_eval_tasks() -> list[dict]:
    """Return ordered list of {short, task_full, language_instruction}."""
    raw = json.loads((BENCH / "eval_tasks.json").read_text())
    items = []
    for short, meta in sorted(raw.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0):
        items.append({
            "short": short,
            "task_full": meta["task_full"],
            "language_instruction": meta.get("language_instruction", meta["task_full"]),
        })
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_create", action="store_true",
                    help="Skip create_repo (set if repo already exists)")
    ap.add_argument("--skip_assets", action="store_true",
                    help="Skip uploading mp4/png assets — only update README + results")
    args = ap.parse_args()

    from huggingface_hub import HfApi
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if not token_path.exists():
        sys.exit(f"HF token not found at {token_path}. Run `huggingface-cli login` first.")
    token = token_path.read_text().strip()
    api = HfApi(token=token)

    eval_tasks = load_eval_tasks()
    print(f"[droid-upload] {len(eval_tasks)} tasks loaded from eval_tasks.json")

    if not args.skip_create:
        try:
            api.create_repo(repo_id=HF_REPO, repo_type="dataset",
                            private=False, exist_ok=True)
            print(f"[droid-upload] repo ready: https://huggingface.co/datasets/{HF_REPO}")
        except Exception as e:
            print(f"[droid-upload] create_repo: {e}")

    # ── DROID README ────────────────────────────────────────────────────
    tasks_md = "\n".join(
        f"| {t['short']} | {t['task_full'][2:]} |" for t in eval_tasks
    )
    droid_readme = f"""# DROID dataset

Stanford DROID robot manipulation episodes plus Cosmos-Predict2 generated
videos (`Cosmos-Predict2-14B-Sample-GR00T-Dreams-DROID` fine-tune).

## Layout

```
droid/
├── training/                          {len(eval_tasks)} real .mp4 (one per task, 320×192 @ 5fps)
├── generated/                         Cosmos gens, grouped by task
│   ├── <task_full>/v0000.mp4 ...
│   └── ...
├── reference/                         SAM3-segmented PNG frames per task
│   └── <task_full>/frame_NNNN.png
├── null_per_task/                     Cached cycle + k-NN null distributions
│   └── task_<N>.npz
├── results/
│   ├── per_task_dense_table.csv       Raw H scores per video
│   ├── per_task_ratio_table.csv       Ranked by ratio
│   ├── per_task_ratio_ranking.md      Markdown report
│   └── viz/                           Visualizations (if generated)
└── method.md                          WarpDyn method documentation
```

## Tasks ({len(eval_tasks)} evaluated)

| ID | Description |
|---|---|
{tasks_md}

## Method: WarpDyn

Same as the GR-1 benchmark — pure feature-matching anomaly detector using
RoMa cycle composition error, DINOv2 k-NN appearance match, per-task
multi-lag null distribution, and Cauchy fusion. See [`method.md`](method.md).
"""
    tmp = Path(tempfile.mkdtemp(prefix="droid_wmbench_"))
    (tmp / "README.md").write_text(droid_readme)
    api.upload_file(
        path_or_fileobj=str(tmp / "README.md"),
        path_in_repo=f"{HF_PREFIX}/README.md",
        repo_id=HF_REPO, repo_type="dataset",
    )
    print(f"[droid-upload] {HF_PREFIX}/README.md")

    # ── Method doc (copy from this worktree) ───────────────────────────
    method_src = REPO_ROOT / "WARPDYN_METHOD.md"
    if method_src.exists():
        api.upload_file(
            path_or_fileobj=str(method_src),
            path_in_repo=f"{HF_PREFIX}/method.md",
            repo_id=HF_REPO, repo_type="dataset",
        )
        print(f"[droid-upload] {HF_PREFIX}/method.md")

    if not args.skip_assets:
        # ── Training videos ─────────────────────────────────────────
        print("\n[droid-upload] training videos …")
        for t in eval_tasks:
            src = BENCH / RAW_VIDEO_SUBDIR / f"{t['short']}.mp4"
            dst = f"{HF_PREFIX}/training/{task_short_id(t['task_full'])}.mp4"
            if src.exists():
                api.upload_file(path_or_fileobj=str(src), path_in_repo=dst,
                                repo_id=HF_REPO, repo_type="dataset")
                print(f"  {dst}")
            else:
                print(f"  [skip] missing {src}")

        # ── Generated videos ────────────────────────────────────────
        gen_root = BENCH / "generated"
        if gen_root.exists():
            print("\n[droid-upload] generated videos …")
            for t in eval_tasks:
                gen_dir = gen_root / t["task_full"]
                if not gen_dir.exists():
                    print(f"  [skip] no gens for {t['task_full'][:50]}")
                    continue
                for mp4 in sorted(gen_dir.glob("v*.mp4")):
                    dst = f"{HF_PREFIX}/generated/{task_short_id(t['task_full'])}/{mp4.name}"
                    api.upload_file(path_or_fileobj=str(mp4), path_in_repo=dst,
                                    repo_id=HF_REPO, repo_type="dataset")
                    print(f"  {dst}")
        else:
            print(f"\n[skip] no generated/ dir: {gen_root}")

        # ── Reference SAM3-segmented frames ─────────────────────────
        print("\n[droid-upload] reference frames …")
        for t in eval_tasks:
            ref_dir = BENCH / "reference" / t["task_full"]
            if not ref_dir.exists() or not any(ref_dir.glob("*.png")):
                print(f"  [skip] empty/missing {ref_dir}")
                continue
            ts = task_short_id(t["task_full"])
            api.upload_folder(
                folder_path=str(ref_dir),
                path_in_repo=f"{HF_PREFIX}/reference/{ts}",
                repo_id=HF_REPO, repo_type="dataset",
            )
            print(f"  {HF_PREFIX}/reference/{ts}/  ({len(list(ref_dir.glob('*.png')))} PNGs)")

        # ── null_per_task .npz cache ────────────────────────────────
        null_dir = BENCH / "null_per_task"
        if null_dir.exists():
            print("\n[droid-upload] null_per_task …")
            for npz in sorted(null_dir.glob("task_*.npz")):
                dst = f"{HF_PREFIX}/null_per_task/{npz.name}"
                api.upload_file(path_or_fileobj=str(npz), path_in_repo=dst,
                                repo_id=HF_REPO, repo_type="dataset")
                print(f"  {dst}")

    # ── Results: CSVs + markdown ────────────────────────────────────────
    print("\n[droid-upload] results …")
    eval_dir = BENCH / "per_task_dense_eval"
    if eval_dir.exists():
        for name in ["per_task_dense_table.csv",
                     "per_task_ratio_table.csv",
                     "per_task_ratio_ranking.md"]:
            src = eval_dir / name
            if src.exists():
                api.upload_file(path_or_fileobj=str(src),
                                path_in_repo=f"{HF_PREFIX}/results/{name}",
                                repo_id=HF_REPO, repo_type="dataset")
                print(f"  results/{name}")
    else:
        print(f"  [skip] no eval dir: {eval_dir}")

    # ── Viz gallery (optional) ──────────────────────────────────────────
    viz_dir = BENCH / "viz"
    if viz_dir.exists():
        print("\n[droid-upload] viz gallery …")
        api.upload_folder(
            folder_path=str(viz_dir),
            path_in_repo=f"{HF_PREFIX}/results/viz",
            repo_id=HF_REPO, repo_type="dataset",
        )
        n_png = sum(1 for _ in viz_dir.rglob("*.png"))
        n_md = sum(1 for _ in viz_dir.rglob("*.md"))
        print(f"  results/viz/  ({n_png} PNG, {n_md} MD)")
    else:
        print(f"\n[skip] no viz dir: {viz_dir}")

    print("\n=== Done ===")
    print(f"DROID: https://huggingface.co/datasets/{HF_REPO}/tree/main/{HF_PREFIX}")


if __name__ == "__main__":
    main()
