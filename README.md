# Hallucination Detection via Warp Variance

A **training-free hallucination detector** for robot video frames. Given a query frame from a VLA policy rollout, the detector measures how inconsistently RoMaV2 dense warp fields across clean reference frames map to that query. A clean frame is well-explained by multiple references (low warp variance); a hallucinated frame forces each reference to invent a different mapping (high warp variance). That certainty-weighted variance is calibrated against a per-task empirical null built from known-clean references; the resulting **H-score ∈ [0, 1]** represents the probability the frame is hallucinated.

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

### Step 4 — Generate query/high videos (cosmos-predict2 14B)

```bash
cd dreamgen_data
make gen-query-high   # 23 tasks × 1 video = 23 MP4s, ~3.3 min/video on H100
                      # output: ../data/cosmos_synthetic_data/query/high/<task>.mp4
```

Requires per-task conditioning images under `data/cosmos_inputs/<task>.png` (first frames of reference MP4s, already extracted if you have the reference data).

### Step 5 — Extract frames + background removal

```bash
# Extract reference MP4s → PNG frames
python scripts/extract_frames.py \
    --video_root data/cosmos_synthetic_data/reference \
    --out_root   data/cosmos_frames_raw/reference \
    --n_frames   50

# Background removal → final reference frames
python scripts/sam3_process.py \
    --frames_root data/cosmos_frames_raw/reference \
    --out_root    data/reference

# Same for query/high (via dreamgen postprocess)
cd dreamgen_data
make postprocess   # ../data/cosmos_synthetic_data/query/high → ../data/query/high

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

| Step | Approx. time |
|---|---|
| Calibrate (23 tasks × 50 refs, turbo) | ~4 h |
| Detect (23 tasks × 50 queries each split) | ~1–2 h |
| cosmos-predict2 14B generation (23 videos) | ~1.3 h (~3.3 min/video) |
| Postprocess (SAM3 bg-removal, 23 tasks × 50 frames) | ~20 min |

Use `setting: fast` in the config to trade ~5% accuracy for ~2× speed on calibrate/detect.
