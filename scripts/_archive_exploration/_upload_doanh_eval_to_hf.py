"""Upload doanh-eval benchmark results to HF wmbench/doanh_eval175/."""
import json
import tempfile
from pathlib import Path
from huggingface_hub import HfApi

REPO = Path("/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination")
BENCH = REPO / "paper-doanh-eval"
DEMO = REPO / ".claude/worktrees/feat-knn-pool-gr1/outputs/doanh_eval_demo"
HF_REPO = "twanghcmut/wmbench"
PREFIX = "doanh_eval175"

token = (Path.home() / ".cache/huggingface/token").read_text().strip()
api = HfApi(token=token)

eval_tasks = json.loads((BENCH / "eval_tasks.json").read_text())

# README
readme = """# doanh × EVAL-175 — WarpDyn benchmark on high vs low quality generated videos

Setup:
- **Reference videos**: [doanh25032004/cosmos_synthetic_data](https://huggingface.co/datasets/doanh25032004/cosmos_synthetic_data) `high/<task>.mp4` (high-quality Cosmos generation)
- **Query videos**: same dataset, `low/<task>.mp4` (low-quality Cosmos generation)
- **Task mapping**: [nvidia/PhysicalAI-Robotics-GR00T-Eval](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Eval) — 23 doanh tasks ↔ 23 EVAL-175 entries (direct filename match across gr1_object/gr1_env/gr1_behavior subfolders)

## Pipeline (per task)

```
OFFLINE (~50s per task):
  1. Sample 50 frames from doanh high.mp4 + SAM3 segment
  2. Cycle null: 50 refs × 4 lags = 182 multi-lag pairs
  3. DINOv2 pool: 50 × 384 features
  4. kNN LOO null: 50 samples, k=15
  5. H_train = score high.mp4 against own null

ONLINE (~17s per task):
  Sample 10 frames from doanh low.mp4 + SAM3 →
  cycle 9 pairs + kNN 10 frames + Cauchy fuse →
  ratio = H_low / H_train_high
```

→ ratio > 1.0 means **low version more anomalous than high** (the expected/desired outcome).

## Sigmoid normalization (NEW)

Per-task σ_baseline estimated via bootstrap (200 iter) of `H_train_fused` —
resample per-pair + per-frame H with replacement, recompute Cauchy-fused score.

```
score_norm = sigmoid((ratio_fused - 1) / σ_baseline) ∈ [0, 1]
  0.5  = at baseline (ratio = 1.0)
  >0.5 = more anomalous than baseline
  <0.5 = cleaner than baseline
```

Thresholds:
- `score_norm > 0.88` → strongly HALLU (≥ +2σ_baseline)
- `score_norm > 0.73` → weakly HALLU (≥ +1σ)
- `score_norm ≤ 0.50` → clean

## File layout

```
doanh_eval175/
├── README.md
├── eval_tasks.json                              # 23 task entries with Eval folder + prompts
├── per_task_dense_eval/
│   ├── doanh_low_vs_high_ratio.csv              # 23 rows, ratio per task
│   ├── doanh_ratio_bars.png                     # per-task bar chart (cycle/knn/fused)
│   ├── doanh_ratio_histograms.png               # distribution across 23 tasks
│   └── doanh_summary.md                         # ranked task table
├── raw_videos/
│   ├── high/{0,1,...22}_<task>.mp4              # 23 doanh high mp4s (training ref)
│   └── low/{0,1,...22}_<task>.mp4               # 23 doanh low mp4s (queries)
├── conditioning/{0,...22}_<task>.png            # Eval-175 first frame + prompt
└── task_<i>/
    ├── timing.json + timing_summary.md
    ├── offline/ (7 GR1-style viz panels)
    └── online/  (7 GR1-style viz panels)
```

Per-task `task_<i>/` mirrors the GR1 benchmark_demo layout — 14 informative viz panels each.
"""

# Read summary
summary_md_path = BENCH / "per_task_dense_eval" / "doanh_summary.md"
summary_md = summary_md_path.read_text() if summary_md_path.exists() else ""

tmp = Path(tempfile.mkdtemp(prefix="doanh_eval_"))
(tmp / "README.md").write_text(readme + "\n\n---\n\n" + summary_md)
api.upload_file(path_or_fileobj=str(tmp / "README.md"),
                path_in_repo=f"{PREFIX}/README.md",
                repo_id=HF_REPO, repo_type="dataset")
print(f"→ {PREFIX}/README.md")

# eval_tasks.json
api.upload_file(path_or_fileobj=str(BENCH / "eval_tasks.json"),
                path_in_repo=f"{PREFIX}/eval_tasks.json",
                repo_id=HF_REPO, repo_type="dataset")
print(f"→ {PREFIX}/eval_tasks.json")

# Summary CSV + plots
for fn in ["doanh_low_vs_high_ratio.csv", "doanh_ratio_bars.png",
           "doanh_ratio_histograms.png", "doanh_summary.md",
           "doanh_score_norm_bars.png", "doanh_ratio_vs_score_norm.png"]:
    src = BENCH / "per_task_dense_eval" / fn
    if src.exists():
        api.upload_file(path_or_fileobj=str(src),
                        path_in_repo=f"{PREFIX}/per_task_dense_eval/{fn}",
                        repo_id=HF_REPO, repo_type="dataset")
        print(f"→ {PREFIX}/per_task_dense_eval/{fn}")

# Raw videos (high + low)
api.upload_folder(folder_path=str(BENCH / "raw_videos"),
                  path_in_repo=f"{PREFIX}/raw_videos",
                  repo_id=HF_REPO, repo_type="dataset")
print(f"→ {PREFIX}/raw_videos/  (high + low mp4s)")

# Conditioning
api.upload_folder(folder_path=str(BENCH / "conditioning"),
                  path_in_repo=f"{PREFIX}/conditioning",
                  repo_id=HF_REPO, repo_type="dataset")
print(f"→ {PREFIX}/conditioning/")

# Per-task bench folders (offline+online viz + timing)
for ts, task in sorted(eval_tasks.items(), key=lambda x: int(x[0])):
    folder = f"{ts}_{task['task_full']}"[:200]
    src = DEMO / folder
    if not src.exists() or not (src / "timing.json").exists():
        continue
    api.upload_folder(folder_path=str(src),
                      path_in_repo=f"{PREFIX}/task_{ts}",
                      repo_id=HF_REPO, repo_type="dataset",
                      commit_message=f"doanh eval task {ts}")
    print(f"→ {PREFIX}/task_{ts}/")

print(f"\nHF: https://huggingface.co/datasets/{HF_REPO}/tree/main/{PREFIX}")
