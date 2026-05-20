#!/usr/bin/env python3
"""Upload the DROID multi-view benchmark assets + results to twanghcmut/wmbench.

Layout uploaded under `twanghcmut/wmbench/droid/`:

    droid/
    ├── README.md                                  multi-view DROID overview
    ├── method.md                                  copy of WARPDYN_METHOD.md
    ├── training/<task_short>/<view>.mp4           per-view raw training mp4s
    ├── generated/<task_full>/v0000.mp4            Cosmos 2×2 composites
    ├── generated/<task_full>/v0000_views/<view>.mp4  per-view demuxed mp4s
    ├── reference/<task_full>/<view>/frame_*.png   SAM3 refs per view
    ├── null_per_task/<task_short>__<view>.npz     per-(task,view) null caches
    ├── conditioning/<task_short>/<view>.png
    ├── conditioning/<task_short>/multiview_2x2.png
    └── results/
        ├── per_task_dense_table.csv
        ├── per_task_ratio_table.csv
        ├── multiview_ratio_ranking.md
        └── viz/
            ├── multiview_hallu/<task>_<vid>.png
            ├── single_view_only/<task>_<vid>.png
            └── multiview_borderline/<task>_<vid>.png

Usage (env with huggingface_hub installed):
    python scripts/upload_droid_wmbench.py
    python scripts/upload_droid_wmbench.py --skip_assets   # only refresh README + results
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

DEFAULT_VIEWS = ["exterior_1", "exterior_2", "wrist"]


def load_eval_tasks() -> list[dict]:
    raw = json.loads((BENCH / "eval_tasks.json").read_text())
    items = []
    for short, meta in sorted(raw.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0):
        items.append({
            "short": short,
            "task_full": meta["task_full"],
            "language_instruction": meta.get("language_instruction", meta["task_full"]),
            "views": meta.get("views", DEFAULT_VIEWS),
        })
    return items


def _maybe_upload_file(api, src: Path, dst: str, label: str | None = None) -> bool:
    if not src.exists():
        return False
    api.upload_file(
        path_or_fileobj=str(src),
        path_in_repo=dst,
        repo_id=HF_REPO,
        repo_type="dataset",
    )
    print(f"  {label or dst}")
    return True


def _maybe_upload_folder(api, src: Path, dst_prefix: str, label: str | None = None) -> bool:
    if not src.exists() or not any(src.rglob("*")):
        return False
    api.upload_folder(
        folder_path=str(src),
        path_in_repo=dst_prefix,
        repo_id=HF_REPO,
        repo_type="dataset",
    )
    print(f"  {label or dst_prefix}")
    return True


def build_readme(eval_tasks: list[dict]) -> str:
    tasks_md = "\n".join(
        f"| {t['short']} | {t['task_full'][len(t['short'])+1:]} | "
        f"{', '.join(t['views'])} |"
        for t in eval_tasks
    )
    return f"""# DROID — multi-view WarpDyn benchmark

Stanford DROID robot manipulation episodes plus Cosmos-Predict2-14B
DROID-finetune generated videos. Five training episodes were selected from
`droid_subset_100`; each is a single-episode unique-text task. Each episode
provides three synchronised camera views (`exterior_1`, `exterior_2`,
`wrist`) at 320×192, 5 fps.

## Layout

```
droid/
├── training/<task_short>/<view>.mp4          per-view real mp4s ({len(eval_tasks)} tasks × 3 views)
├── generated/<task_full>/v0000.mp4           Cosmos 2×2 composite gen
├── generated/<task_full>/v0000_views/        per-view demuxed mp4s
│   └── <view>.mp4
├── reference/<task_full>/<view>/frame_*.png  SAM3-segmented refs (per view)
├── null_per_task/<task_short>__<view>.npz    cycle + k-NN null caches per (task, view)
├── conditioning/<task_short>/<view>.png      first-frame conditioning per view
├── conditioning/<task_short>/multiview_2x2.png  Cosmos input composite
└── results/
    ├── per_task_dense_table.csv              raw per-view H scores + multiview fused
    ├── per_task_ratio_table.csv              ratios + verdicts (per view + multiview)
    ├── multiview_ratio_ranking.md            human-readable ranking + complementarity
    └── viz/                                  per-catch visualizations
```

## Tasks ({len(eval_tasks)} evaluated)

| ID | Description | Views |
|---|---|---|
{tasks_md}

## Method

Each of the 3 camera views is processed **independently** by the same
single-view WarpDyn pipeline as the GR-1 benchmark:

  1. Multi-lag cycle null (RoMa forward/backward composition) using the
     per-view SAM3 reference frames, lags ∈ {{1, 2, 5, 10}}.
  2. DINOv2 ViT-S/14 k-NN nearest-neighbour pool over the same refs, with
     leave-one-out null (k=15).
  3. Within-view fusion: `H_fused_v = 1 − ACAT-Cauchy(p_cycle_v, p_knn_v)`
     at the video level.

Then the per-view fused H scores are **cross-view fused** via a second
Cauchy combine:

  `p_v        = 1 − clip(H_fused_v, ε, 1−ε)`
  `p_mv       = ACAT-Cauchy(p_exterior_1, p_exterior_2, p_wrist)`
  `H_mv       = 1 − p_mv`
  `ratio_mv   = H_mv / H_train_mv`

By construction the per-task training video sits at `ratio = 1.0`, so the
multi-view decision rule `ratio_multiview > 1.0 ⇒ HALLU` has FPR=0% on
training videos.

See [`method.md`](method.md) for the single-view derivation.
"""


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Upload the DROID multi-view benchmark to twanghcmut/wmbench."
    )
    ap.add_argument("--skip_create", action="store_true",
                    help="skip create_repo (set if repo already exists)")
    ap.add_argument("--skip_assets", action="store_true",
                    help="only refresh README + results, leave mp4/png assets alone")
    ap.add_argument("--views", nargs="+", default=DEFAULT_VIEWS,
                    help="views to upload (default: 3 DROID views)")
    args = ap.parse_args()

    from huggingface_hub import HfApi
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if not token_path.exists():
        sys.exit(f"HF token not found at {token_path}. Run `huggingface-cli login` first.")
    token = token_path.read_text().strip()
    api = HfApi(token=token)

    eval_tasks = load_eval_tasks()
    print(f"[droid-upload] {len(eval_tasks)} tasks loaded from eval_tasks.json")
    print(f"[droid-upload] views: {args.views}")

    if not args.skip_create:
        try:
            api.create_repo(repo_id=HF_REPO, repo_type="dataset",
                            private=False, exist_ok=True)
            print(f"[droid-upload] repo ready: https://huggingface.co/datasets/{HF_REPO}")
        except Exception as e:
            print(f"[droid-upload] create_repo: {e}")

    # ── README + method.md ──────────────────────────────────────────────
    tmp = Path(tempfile.mkdtemp(prefix="droid_wmbench_"))
    (tmp / "README.md").write_text(build_readme(eval_tasks))
    _maybe_upload_file(api, tmp / "README.md", f"{HF_PREFIX}/README.md")

    method_src = REPO_ROOT / "WARPDYN_METHOD.md"
    _maybe_upload_file(api, method_src, f"{HF_PREFIX}/method.md")

    if not args.skip_assets:
        # ── Per-view training videos ────────────────────────────────────
        print("\n[droid-upload] training videos (per view) …")
        for t in eval_tasks:
            for v in args.views:
                src = BENCH / RAW_VIDEO_SUBDIR / t["short"] / f"{v}.mp4"
                dst = f"{HF_PREFIX}/training/{t['short']}/{v}.mp4"
                if not _maybe_upload_file(api, src, dst):
                    print(f"  [skip] missing {src}")

        # ── Generated videos (composite + demuxed) ──────────────────────
        gen_root = BENCH / "generated"
        if gen_root.exists():
            print("\n[droid-upload] generated videos …")
            for t in eval_tasks:
                gen_dir = gen_root / t["task_full"]
                if not gen_dir.exists():
                    print(f"  [skip] no gens for {t['task_full'][:50]}")
                    continue
                # Composite
                for mp4 in sorted(gen_dir.glob("v*.mp4")):
                    dst = f"{HF_PREFIX}/generated/{t['task_full']}/{mp4.name}"
                    _maybe_upload_file(api, mp4, dst)
                # Demuxed per-view (subdirs <stem>_views/)
                for views_dir in sorted(gen_dir.glob("v*_views")):
                    if not views_dir.is_dir():
                        continue
                    for mp4 in sorted(views_dir.glob("*.mp4")):
                        dst = (f"{HF_PREFIX}/generated/{t['task_full']}/"
                               f"{views_dir.name}/{mp4.name}")
                        _maybe_upload_file(api, mp4, dst)
        else:
            print(f"\n[skip] no generated/ dir: {gen_root}")

        # ── Reference SAM3 frames per (task, view) ──────────────────────
        print("\n[droid-upload] reference frames …")
        for t in eval_tasks:
            for v in args.views:
                ref_dir = BENCH / "reference" / t["task_full"] / v
                if not ref_dir.exists() or not any(ref_dir.glob("*.png")):
                    print(f"  [skip] empty/missing {ref_dir}")
                    continue
                _maybe_upload_folder(
                    api, ref_dir,
                    f"{HF_PREFIX}/reference/{t['task_full']}/{v}",
                    label=f"reference/{t['task_full']}/{v}/ "
                          f"({len(list(ref_dir.glob('*.png')))} PNGs)",
                )

        # ── null_per_task .npz (per-(task,view)) ────────────────────────
        null_dir = BENCH / "null_per_task"
        if null_dir.exists():
            print("\n[droid-upload] null_per_task …")
            for npz in sorted(null_dir.glob("*.npz")):
                dst = f"{HF_PREFIX}/null_per_task/{npz.name}"
                _maybe_upload_file(api, npz, dst)

        # ── Conditioning ────────────────────────────────────────────────
        cond_root = BENCH / "conditioning"
        if cond_root.exists():
            print("\n[droid-upload] conditioning …")
            for t in eval_tasks:
                cond_task_dir = cond_root / t["short"]
                if not cond_task_dir.exists():
                    continue
                for png in sorted(cond_task_dir.glob("*.png")):
                    dst = f"{HF_PREFIX}/conditioning/{t['short']}/{png.name}"
                    _maybe_upload_file(api, png, dst)

    # ── Results CSVs + markdown ─────────────────────────────────────────
    print("\n[droid-upload] results …")
    eval_dir = BENCH / "per_task_dense_eval"
    if eval_dir.exists():
        # The compute script emits per_task_ratio_ranking.md; expose it under
        # the more-descriptive multiview_ratio_ranking.md filename on HF.
        rename_map = {
            "per_task_dense_table.csv":   "per_task_dense_table.csv",
            "per_task_ratio_table.csv":   "per_task_ratio_table.csv",
            "per_task_ratio_ranking.md":  "multiview_ratio_ranking.md",
        }
        for src_name, dst_name in rename_map.items():
            src = eval_dir / src_name
            dst = f"{HF_PREFIX}/results/{dst_name}"
            if not _maybe_upload_file(api, src, dst):
                print(f"  [skip] missing {src}")
    else:
        print(f"  [skip] no eval dir: {eval_dir}")

    # ── Viz gallery ─────────────────────────────────────────────────────
    viz_dir = BENCH / "viz"
    if viz_dir.exists():
        print("\n[droid-upload] viz gallery …")
        for sub in ["multiview_hallu", "single_view_only", "multiview_borderline"]:
            sub_dir = viz_dir / sub
            if not sub_dir.exists():
                continue
            _maybe_upload_folder(
                api, sub_dir,
                f"{HF_PREFIX}/results/viz/{sub}",
                label=f"results/viz/{sub}/ "
                      f"({sum(1 for _ in sub_dir.rglob('*.png'))} PNG)",
            )
    else:
        print(f"\n[skip] no viz dir: {viz_dir}")

    print("\n=== Done ===")
    print(f"DROID: https://huggingface.co/datasets/{HF_REPO}/tree/main/{HF_PREFIX}")


if __name__ == "__main__":
    main()
