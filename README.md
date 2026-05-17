# Hallucination Detection via Warp Variance

A **training-free hallucination detector** for robot video frames. Given a query frame from a VLA policy rollout, the detector measures how inconsistently RoMaV2 dense warp fields across clean reference frames map to that query. A clean frame is well-explained by multiple references (low warp variance); a hallucinated frame forces each reference to invent a different mapping (high warp variance). That certainty-weighted variance is calibrated against a per-task empirical null built from known-clean references; the resulting **H-score ∈ [0, 1]** represents the probability the frame is hallucinated.

---

## Common Scenarios

### 1. Score a single frame (no calibration needed — uses global fallback)

```bash
conda activate groot
python -m warp_score --artifacts_dir artifacts/v1 \
    detect --query data/query/low/0_Open\ the\ box/frame_0000.png
# prints H_score to stdout; appends a row to artifacts/v1/summary.csv
```

### 2. Quick smoke test on one task end-to-end

```bash
ARTS=artifacts/smoke_task
python -m warp_score --artifacts_dir $ARTS calibrate --task "0_Open the box"
python -m warp_score --artifacts_dir $ARTS detect   --task "0_Open the box"
# labels for this one task:
python scripts/build_weak_labels.py \
    --query_high_dir data/query/high --query_low_dir data/query/low \
    --tasks "0_Open the box" --out /tmp/labels_smoke.csv
python -m warp_score --artifacts_dir $ARTS eval --labels /tmp/labels_smoke.csv
```

Calibrate 1 task ≈ 2.5 min, detect ≈ 5 min on H100.

### 3. Benchmark a custom subset of tasks

Create a symlink directory pointing at the tasks you want:

```bash
python3 -c "
import os, sys
from pathlib import Path
REPO = Path('.')
tasks = ['0_Open the box', '2_Use the right hand to close the black drawer']
for root, dest in [
    ('data/reference',   '/tmp/myref'),
    ('data/query/high',  '/tmp/myq_high'),
    ('data/query/low',   '/tmp/myq_low'),
]:
    os.makedirs(dest, exist_ok=True)
    for t in tasks:
        src = (REPO / root / t).resolve()
        link = f'{dest}/{t}'
        if not os.path.lexists(link): os.symlink(src, link)
"

ARTS=artifacts/my_subset
python -m warp_score --ref_dir /tmp/myref --artifacts_dir $ARTS calibrate
python -m warp_score --ref_dir /tmp/myref \
    --query_high_dir /tmp/myq_high --query_low_dir /tmp/myq_low \
    --artifacts_dir  $ARTS detect
# build labels for those tasks and eval:
python scripts/build_weak_labels.py \
    --query_high_dir /tmp/myq_high --query_low_dir /tmp/myq_low \
    --out /tmp/labels_subset.csv
python -m warp_score --artifacts_dir $ARTS eval --labels /tmp/labels_subset.csv
```

> **Note:** `--ref_dir`, `--query_high_dir`, `--query_low_dir` all follow symlinked directories correctly.

### 4. Score a new video from scratch (inference only, no eval)

```bash
# 1. Extract frames
python scripts/extract_frames.py \
    --video_root /path/to/new_videos --out_root /tmp/new_frames --n_frames 50

# 2. Background removal (optional but recommended)
python scripts/sam3_process.py \
    --frames_root /tmp/new_frames --out_root /tmp/new_clean

# 3. Score against an existing calibration
python -m warp_score --artifacts_dir artifacts/v1 \
    detect --query_dir /tmp/new_clean
# Results in artifacts/v1/summary.csv — H_score > 0.95 → likely hallucinated
```

### 5. Re-run eval on existing predictions with a new labels CSV

Calibrate/detect outputs are cached in `artifacts_dir/summary.csv`. Re-run eval only:

```bash
python -m warp_score --artifacts_dir artifacts/v1 eval \
    --labels new_labels.csv \
    --pred   artifacts/v1/summary.csv \
    --out    /tmp/eval_new.json
```

### 6. Use a faster matching setting to prototype quickly

```bash
# "fast" is ~2× faster than "turbo" (the default), ~5% AUROC drop
python -m warp_score --setting fast --artifacts_dir artifacts/fast_run calibrate
python -m warp_score --setting fast --artifacts_dir artifacts/fast_run detect
```

### 7. Generate new clean-reference videos (dreamgen pipeline)

```bash
export HF_TOKEN=hf_xxx
cd dreamgen_data
bash setup.sh              # one-time: installs cosmos-predict2, downloads ~250 GB checkpoints
make smoke                 # sanity: 1 video (~3 min)
make gen-query-high        # all 23 tasks, ~3.3 min/video on H100, seed_offset=2000
make postprocess           # extract frames + SAM3 bg-removal → ../data/query/high_v2
```

---

## Understanding H_score

| H_score | Interpretation |
|---|---|
| 0.00 – 0.50 | Almost certainly clean — warp variance matches reference distribution |
| 0.50 – 0.90 | Moderate anomaly — scene may have drifted |
| 0.90 – 0.95 | High anomaly — likely hallucinated |
| > 0.95 | Very likely hallucinated (default threshold `fpr_alpha=0.05`) |

H_score is a **calibrated p-value complement**: `H_score = 1 - p`, where `p` is the probability that a clean reference frame would show equal or greater warp variance. Each task has its own null distribution (LOO calibration); tasks with no calibration fall back to the global distribution.

---

## Environment Overview

Two separate Python environments are used:

| Component | Environment | Python | PyTorch | CUDA |
|---|---|---|---|---|
| `warp_score` (detector + eval) | `groot` conda env | 3.10 | 2.5.1+cu121 | 12.1 |
| `dreamgen_data` (cosmos-predict2 generation) | `.venv` inside `dreamgen_data/cosmos-predict2/` | 3.10 | 2.6.0+cu126 | 12.6 |

The two envs are **independent** — warp_score never imports cosmos-predict2 and vice versa.

---

## Quickstart (existing data)

If `data/` is already populated (reference + query frames), skip to step 3.

```bash
# 1. Activate the warp-score environment
conda activate groot          # or: conda env create -f environment.yml && conda activate warp-score

# 2. Install the package (editable)
pip install -e .

# 3. Add RoMaV2 to PYTHONPATH (required — no pip package)
export PYTHONPATH=$(pwd)/third_party/RoMaV2/src:$PYTHONPATH

# 4. Build per-task null distributions from reference frames
python -m warp_score --artifacts_dir artifacts/v1 calibrate

# 5. Score query/{high,low} frames
python -m warp_score --artifacts_dir artifacts/v1 detect

# 6. Build labels CSV (if not present)
python scripts/build_weak_labels.py \
    --query_high_dir data/query/high \
    --query_low_dir  data/query/low \
    --out labels.csv

# 7. Evaluate AUROC / AP / FPR@95TPR
python -m warp_score --artifacts_dir artifacts/v1 eval --labels labels.csv
```

---

## Full Reproduce from Scratch

### Prerequisites

- NVIDIA GPU (tested on H100 80 GB)
- CUDA 12.1+ driver
- Conda (for warp_score env) + `uv` (for dreamgen env, installed by `setup.sh`)
- Hugging Face account with access to:
  - `nvidia/Cosmos-Predict2-14B-Video2World` (~170 GB)
  - `google-t5/t5-11b` (~85 GB)
- ~300 GB free disk for checkpoints + data

### Step 1 — Set up the warp_score environment

```bash
# Option A: use the existing groot env on this machine (already has correct torch/CUDA)
conda activate groot
pip install -e .

# Option B: create a fresh env from spec
conda env create -f environment.yml
conda activate warp-score
# Install torch matching your CUDA version — example for CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

### Step 2 — Set up RoMaV2

RoMaV2 source is bundled in `third_party/RoMaV2/`. Model weights download automatically on first use.

```bash
export PYTHONPATH=$(pwd)/third_party/RoMaV2/src:$PYTHONPATH
# Verify:
python -c "from romav2.model import RoMa; print('RoMaV2 OK')"
```

### Step 3 — Set up cosmos-predict2 (for generating query/high data)

```bash
export HF_TOKEN=<your_huggingface_token>   # needs read access to nvidia/Cosmos-Predict2-14B-Video2World

cd dreamgen_data
bash setup.sh        # clones cosmos-predict2, creates .venv, installs deps, downloads checkpoints
make smoke           # sanity: 1 prompt → 1 video (~3 min on H100)
```

`setup.sh` is idempotent — safe to re-run. Logs to `dreamgen_data/setup.log`.

**Checkpoint paths after setup:**
```
dreamgen_data/checkpoints/
├── nvidia/Cosmos-Predict2-14B-Video2World/   # ~170 GB
└── google-t5/t5-11b/                         # ~85 GB
```

### Step 4 — Generate synthetic query videos (requires `dreamgen_data/.venv` and `HF_TOKEN`)

```bash
export HF_TOKEN=hf_xxx
cd dreamgen_data
source cosmos-predict2/.venv/bin/activate

# generate.py skips any <task>.mp4 that already exists — safe to resume
python generate.py \
    --prompts prompts.json \
    --save_dir ../data/cosmos_synthetic_data/query/high

# Or equivalently via make (same underlying command):
make gen-query-high   # 23 tasks × 1 video = 23 MP4s, ~3.3 min/video on H100
                      # output: ../data/cosmos_synthetic_data/query/high/<task>.mp4
```

Requires per-task conditioning images under `data/cosmos_inputs/<task>.png` (first frames of reference MP4s, already extracted if you have the reference data).

### Step 5 — Extract frames + background removal

```bash
# postprocess.py: extract 50 uniform frames per video + SAM3 bg removal (skips existing output)
cd dreamgen_data
source cosmos-predict2/.venv/bin/activate

python postprocess.py \
    --video_root ../data/cosmos_synthetic_data/query/high \
    --out_root   ../data/query/high

# Or equivalently via make:
make postprocess   # ../data/cosmos_synthetic_data/query/high → ../data/query/high

# Extract reference MP4s → PNG frames (uses groot env, not dreamgen venv)
conda activate groot
python scripts/extract_frames.py \
    --video_root data/cosmos_synthetic_data/reference \
    --out_root   data/cosmos_frames_raw/reference \
    --n_frames   50

# Background removal → final reference frames
python scripts/sam3_process.py \
    --frames_root data/cosmos_frames_raw/reference \
    --out_root    data/reference

# Same for query/low (pre-existing hallucinated frames)
python scripts/sam3_process.py \
    --frames_root data/cosmos_frames_raw/query/low \
    --out_root    data/query/low
```

### Step 6 — Build labels + run full eval

```bash
python scripts/build_weak_labels.py \
    --query_high_dir data/query/high \
    --query_low_dir  data/query/low \
    --out labels.csv

ARTIFACTS=artifacts/v_reproduce
python -m warp_score --artifacts_dir $ARTIFACTS calibrate
python -m warp_score --artifacts_dir $ARTIFACTS detect
python -m warp_score --artifacts_dir $ARTIFACTS eval --labels labels.csv
```

---

## Data Layout

```
data/
├── reference/<task>/*.png          # calibration refs — LOO null (50 frames/task, 23 tasks)
├── query/
│   ├── high/<task>/*.png           # clean held-out (cosmos-predict2 14B, label=0)
│   └── low/<task>/*.png            # hallucinated queries (label=1)
├── cosmos_synthetic_data/          # source MP4s
│   ├── reference/<task>.mp4
│   └── query/
│       ├── high/<task>.mp4
│       └── low/<task>.mp4
└── cosmos_frames_raw/              # raw extracted frames (before SAM3)
    ├── reference/<task>/*.png
    └── query/{high,low}/<task>/*.png
```

Background pixels are exactly `(127, 127, 127)` — the foreground mask relies on this value.

---

## Package Structure

| Module | What it does |
|---|---|
| `warp_score/config.py` | `WarpScoreConfig` dataclass — all hyperparameters. Load from YAML or override via CLI. |
| `warp_score/matcher.py` | `RoMaMatcher` — thin OOP wrapper around RoMaV2. Returns `(warp_HW2, cert_HW)` with background certainty zeroed. |
| `warp_score/mask.py` | `ForegroundMask` (gray-pixel exclusion) and `InteriorMask` (morphological erosion to drop boundary artifacts). |
| `warp_score/statistics.py` | `CertWeightedStatistics` — certainty-weighted warp mean and variance maps across a stack of references. |
| `warp_score/calibrator.py` | `EmpiricalNullCalibrator` — builds per-task sorted null distributions from reference frames (LOO). |
| `warp_score/signals.py` | `IvarSignal` (integrated variance), `PeakSignal` (99th-percentile variance), `CertSignal` (mean certainty). Each emits a scalar p-value. |
| `warp_score/fuser.py` | `StoufferFuser` / `FisherFuser` / `MaxFuser` — combine per-signal p-values into a single H-score. |
| `warp_score/detector.py` | `WarpVarianceDetector` — end-to-end: load query, run matcher, compute signals, fuse → `HallucinationResult`. |
| `warp_score/evaluator.py` | `Evaluator` — joins labels CSV with predictions, computes AUROC / AP / FPR@95TPR. |
| `warp_score/cli.py` | `python -m warp_score` entrypoint — `calibrate`, `detect`, `eval` subcommands. |
| `dreamgen_data/` | cosmos-predict2 generation pipeline — see `dreamgen_data/README.md` or `make help` there. |
| `scripts/build_weak_labels.py` | Build `labels.csv` from query/high + query/low directories. |

---

## Utility Scripts

| Script | Purpose |
|---|---|
| `scripts/viz_matching.py` | Visualize RoMa dense correspondences between a query frame and its references. Useful for debugging matcher quality. |
| `scripts/run_pipeline.sh` | Shell wrapper that runs calibrate → detect → eval in sequence for a given artifacts dir. |

Example — visualize matches for one task:

```bash
python scripts/viz_matching.py \
    --task "0_Open the box" \
    --frames frame_0000,frame_0001 \
    --n_refs 3 --n_kpts 150 \
    --out_dir /tmp/viz_matching
```

Example — run the full pipeline end-to-end:

```bash
conda activate groot
bash scripts/run_pipeline.sh   # skips already-completed steps automatically
```

---

## Configuration

All behaviour is controlled by `warp_score/configs/default.yaml`:

```yaml
# Paths (repo-relative; override with CLI flags)
# reference_dir:  data/reference
# query_high_dir: data/query/high
# query_low_dir:  data/query/low
# artifacts_dir:  artifacts/v9_dreamgen

# RoMaV2 matching
setting:         turbo      # turbo | fast | base | precise
vis_size:        224
bidirectional:   false
device:          cuda

# Signals + fusion
signal_names:    [ivar, peak, cert]
fuser:           stouffer
stouffer_weights:
  ivar: 2.0
  peak: 1.0
  cert: 1.0

# Decision threshold
fpr_alpha:       0.05       # H_score > 0.95 → hallucination
```

Override on CLI: `--ref_dir`, `--query_high_dir`, `--query_low_dir`, `--artifacts_dir`, `--device`, `--setting`.

---

## CLI Reference

```
python -m warp_score [--config YAML] [--artifacts_dir DIR]
                     [--ref_dir DIR] [--query_high_dir DIR] [--query_low_dir DIR]
                     [--device DEVICE] [--setting SETTING]
                     {calibrate,detect,eval} ...
```

**`calibrate`** — Scan `reference_dir`, run all-pairs RoMaV2, save null distributions to `artifacts_dir/calibration.npz`.

**`detect`** — Load calibration, score frames, write `artifacts_dir/summary.csv`.

| Flag | Description |
|---|---|
| `--query PATH` | Score a single PNG. |
| `--query_dir DIR` | Score all PNGs under DIR. |
| `--task NAME` | Score `query_high_dir/<task>` + `query_low_dir/<task>`. |
| (none) | Score all frames under both `query_high_dir` and `query_low_dir`. |

**`eval`** — Join labels CSV with predictions, print + save `artifacts_dir/eval_report.json`.

| Flag | Default | Description |
|---|---|---|
| `--labels PATH` | required | CSV with `frame,label` columns. |
| `--pred PATH` | `artifacts_dir/summary.csv` | Predictions CSV from `detect`. |

---

## Tests

```bash
pytest tests/          # all tests pass without a GPU (~10 s)
```

---

## Hardware

Tested on a single **NVIDIA H100 80 GB** (CUDA 12.1 / 12.6).

| Step | Approx. time (H100, turbo) |
|---|---|
| Calibrate 1 task (53 refs, LOO) | ~2.5 min |
| Calibrate 23 tasks | ~60 min |
| Detect 1 task (50 high + 53 low frames) | ~5 min |
| Detect 23 tasks (2369 frames total) | ~2 h |
| cosmos-predict2 14B generation (1 video, with refiner) | ~3.5 min |
| cosmos-predict2 14B generation (23 videos) | ~1.3 h |
| Postprocess SAM3 (1 task × 50 frames) | ~1 min |

Per-frame detect time is ~3.2 s/frame (3s RoMa match × N_refs). Use `--setting fast` to halve that at ~5% AUROC cost.
