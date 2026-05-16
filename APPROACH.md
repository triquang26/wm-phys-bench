# Hallucination Detection via Warp Variance — Full Technical Writeup

> **Implementation note (2026-05)**: This document describes the algorithm and methodology.
> The current implementation lives in the `warp_score/` Python package.
> Entry point: `python -m warp_score calibrate/detect/eval`.
> `warp_variance_vis.py` is now a backward-compat shim.

## 1. Problem Context

### What is a hallucination here?

A **hallucination** is a video frame produced by a robot policy (VLA / world model) where the
rendered scene is **inconsistent** with what the robot is actually doing — the object or robot arm
appears in a physically impossible or semantically wrong position/state.

Concretely, the dataset has:

| Split | Quality | Description |
|-------|---------|-------------|
| `image_no_bg/high/` | Reference (clean) | Frames captured during genuine, successful robot execution. No hallucination by construction. |
| `image_no_bg/low/` | Query (test) | Frames from a policy rollout. Some are correct; some are hallucinated. |

Both splits are organized by **task** (directory name = task description string, e.g.
`0_Open the box`, `4_Use the right hand to close waffle maker`, etc.).  
Each task has exactly **6 high frames** and **6 low frames**.

The background has already been removed by SAM3: background pixels are exactly `(127, 127, 127)`.
Only the robot arm and object pixels are meaningful.

### Goal

Given a single low-quality frame (possibly just one frame, no batch context), output:

```
H ∈ {0, 1}     0 = clean,  1 = hallucination
```

with the ability to generalize across tasks **without retraining** and **without needing an entire
video sequence** at inference time.

---

## 2. Core Intuition: Warp Variance as Hallucination Signal

The detector uses **RoMaV2** (a dense warp field matcher) to match the query frame against each
reference frame independently.

For a pixel `p` in the query:

- RoMaV2 outputs a **warp field** `(u, v)` — where in the reference this pixel corresponds to.
- It also outputs a **certainty** `cert ∈ [0, 1]` — how confident the match is.

### Key insight

| Scenario | What refs "think" | Warp variance |
|----------|-------------------|---------------|
| **Clean frame** — robot is in a position that refs recognize | All refs agree on where `p` maps → low spread of `(u, v)` across refs | **Low** |
| **Hallucinated frame** — robot is in a position refs cannot explain | Each ref "invents" a different target → high spread of `(u, v)` across refs | **High** |

The certainty-weighted warp variance is the primary feature:

```
mean_coord[p] = Σ_r  cert_r[p] · warp_r[p]  /  Σ_r cert_r[p]
warp_var[p]   = Σ_r  cert_r[p] · ‖warp_r[p] − mean_coord[p]‖²  /  Σ_r cert_r[p]
```

Background pixels have `cert = 0` (zeroed out inside `match_dense`), so they contribute zero
weight.

---

## 3. Context Assumptions

1. **Background is `(127,127,127)`** — the `foreground_mask()` function uses this exact pixel value
   to separate fg from bg. Any change to the segmenter output format breaks this.

2. **Tasks are the parent directory name** — `Path(frame).parent.name` is the task ID. Task names
   must match exactly between `high/` and `low/` splits for task-specific calibration to work.

3. **Each task has ≥ 2 high frames** — calibration requires at least one ref per query (so ≥ 2
   total). With 6 per task, each calibration query uses 5 refs.

4. **High frames are guaranteed non-hallucinated** — this is the axiomatic assumption behind
   calibration. The baseline `mean_ivar` for each task is the expected warp variance when the
   scene is cleanly matched.

5. **RoMaV2 `turbo` setting + `use_precision`** — the certainty signal uses
   `det(Σ⁻¹)` (precision matrix determinant) rather than `overlap_AB`, which is more
   discriminative for robotics scenes with many repeated textures.

6. **`VIS_SIZE = 224`** — all maps are resized to 224×224 before any computation.

---

## 4. Preprocessing: Interior Mask

Thin robot arm **edges and tips** create high warp variance purely because of boundary effects
(subpixel alignment noise) — not because of hallucination. A 10-pixel erosion removes these:

```python
interior_mask = cv2.erode(
    fg_mask.astype(np.uint8),
    np.ones((erosion_k, erosion_k), dtype=np.uint8),   # default erosion_k=10
    iterations=1,
).astype(bool)
```

All z-score computation and mean ivar are performed **only inside `interior_mask`**, not on the
full foreground. This is critical — without it, frames with thin robot arm tips always appear
anomalous.

---

## 5. Three Hallucination Signals

The frame-level classification uses an OR of three independent signals:

```python
is_hallucination = int(cond_peak OR cond_global_var OR cond_low_cert)
```

### Signal 1 — `cond_peak`: Local anomaly spike

Within the interior, compute a per-pixel z-score relative to the frame's own interior distribution:

```
z[p] = (warp_var[p] − mean(interior)) / std(interior)
```

Then threshold and find connected blobs:

```python
cond_peak = (max_pixel_z > z_thresh) AND (largest_blob_area >= blob_min_area)
```

Defaults: `z_thresh=2.0`, `blob_min_area=50px`.

**What it catches**: a localized region where refs disagree (e.g., one finger/joint in a wrong
pose while the rest of the arm looks OK).

**Limitation**: the z-score is within-frame relative. For tasks with naturally high structural
variance in the interior (e.g., articulated objects with complex textures), the local z-score can
spike even on clean frames. This signal generates the most false positives.

### Signal 2 — `cond_global_var`: Whole-frame elevation

```python
z_ivar = (interior_mean_var − task_mean_ivar) / task_std_ivar
cond_global_var = (z_ivar > global_z_thresh)   # default 2.0
```

Where `task_mean_ivar` and `task_std_ivar` come from the **task-specific calibration** (see §6).

**What it catches**: the entire interior has elevated warp variance — the robot is in a completely
different state than any reference shows.

**This is the most reliable signal** with task-specific calibration. The ratio between clean
and hallucinated frames is typically 50–200×.

### Signal 3 — `cond_low_cert`: Out-of-distribution frame

```python
cond_low_cert = (mean_cert_fg < cert_low_thresh)   # default 0.10
```

Where `mean_cert_fg` is the mean certainty across the interior mask.

**What it catches**: RoMaV2 cannot find any meaningful correspondence from the query to any
reference. The query frame is entirely out of the distribution of ref scenes — often a strongly
hallucinated frame where the robot arm or object is in a position that doesn't appear at all in
the task's reference set.

---

## 6. Task-Specific Calibration

### Why cross-task calibration fails

With 138 refs spanning 23 different tasks, a cross-task calibration baseline has:

- **cert universally low** (0.08–0.16): refs from task B can't match pixels from task A's scene →
  all certainty weights near-uniform → warp variance is dominated by cross-task scene differences.
- **Inflated baseline**: `mean_ivar ≈ 0.35`, `std_ivar ≈ 0.064`.
- **Tiny z-scores**: a hallucinated frame with `ivar = 0.40` gives `z = (0.40 − 0.35) / 0.064 ≈ 0.78` — well below the threshold.

### Task-specific refs approach (`--task_refs`)

For a query from task X, use **only the 6 high frames of task X** as references.

Same-task refs:
- Show the same scene, same objects, same environment → cert is much higher.
- Agree on pixel correspondence for clean poses → warp variance is much lower.
- Disagree clearly for hallucinated poses → warp variance spikes dramatically.

### Calibration process

```
for each task T:
    for each high frame q in T (6 total):
        refs = T's high frames excluding q  (5 refs)
        compute interior_mean_var(q, refs)  → ivar_q
    task_mean_ivar[T] = mean(ivar_q for q in T)
    task_std_ivar[T]  = std(ivar_q for q in T)
```

Saved to `calib_task_specific.json`:

```json
{
  "task_refs": true,
  "tasks": {
    "0_Open the box": {"mean_ivar": 0.0017, "std_ivar": 0.0008, "n": 6},
    "4_Use the right hand to close waffle maker": {"mean_ivar": 0.0168, "std_ivar": 0.0130, "n": 6},
    ...
  },
  "global": {"mean_ivar": 0.0241, "std_ivar": 0.0365, "n": 138}
}
```

### Before vs after: "0_Open the box"

| | Cross-task calib | Task-specific calib |
|--|--|--|
| Calibration mean_ivar | 0.3481 | **0.0017** |
| Calibration std_ivar | 0.0637 | **0.0008** |
| frame_0001 (clean): z_ivar | −0.80 → H=0 ✓ | **1.84 → H=0 ✓** |
| frame_0005 (halluc): z_ivar | +0.75 → H=0 ✗ | **127.9 → H=1 ✓** |

Signal-to-noise ratio improved **~170×**.

---

## 7. Inference Flow (single frame)

```
query_frame.png
       │
       ▼
foreground_mask()          ← pixels == (127,127,127) → background
       │
       ▼
erode fg by erosion_k px   ← interior_mask (no boundary/tips)
       │
       ├── filter high_paths to same task (--task_refs)
       │
       ▼
for each ref in task_refs:
    RoMaV2.match(query, ref)
       ├── warp_AB: (H,W,2)  normalized coords [-1,1]
       └── precision_AB: (H,W,2,2)  → cert = sqrt(det(Σ⁻¹))
       resize to 224×224
       zero cert on background
       │
       ▼
cert-weighted mean warp     → mean_coord[p]
cert-weighted variance      → warp_var[p]       (H,W)
       │
       ├──── interior_mean_var  →  z_ivar (vs calib)  →  cond_global_var
       │
       ├──── within-frame z-score  →  blobs  →  cond_peak
       │
       └──── mean cert in interior  →  cond_low_cert
       │
       ▼
is_hallucination = cond_peak OR cond_global_var OR cond_low_cert
```

---

## 8. Batch Fallback (`batch_reclassify`)

When no `--calib_file` is provided (inference on a full video/directory), a post-hoc batch
z-score is applied per task using a **min-anchored** approach:

```python
ivar_min = min(ivar values in task)        # best (cleanest) frame anchors zero
devs = [ivar - ivar_min for ivar in task]  # deviation from best frame
z = dev / std(devs)                        # z-score of deviation
cond_global_var = (z > global_z_thresh)
```

The min-anchor prevents hallucinated frames from inflating the mean (which would happen with
the standard mean-centered z-score), making the baseline always the task's cleanest frame.

---

## 9. Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `--erosion_k` | 10 | Kernel size for fg mask erosion. Larger → remove more boundary. Too large removes thin objects entirely. |
| `--z_thresh` | 2.0 | Within-frame peak z-score threshold for `cond_peak`. Raise to reduce false positives from structural variance. |
| `--blob_min_area` | 50 | Minimum blob area (px) for `cond_peak`. Filters noise spikes that aren't spatially coherent. |
| `--global_z_thresh` | 2.0 | z_ivar threshold for `cond_global_var`. Primary tuning knob for task-calibrated detection. |
| `--cert_low_thresh` | 0.10 | Mean cert threshold for `cond_low_cert`. Frames with cert < this are flagged as OOD. |
| `--task_refs` | False | Enable task-specific ref filtering. Requires `--calib_file` with per-task stats. |
| `--setting` | turbo | RoMaV2 resolution (turbo=320px). Higher = slower but more accurate. |
| `--use_precision` | False | Use precision matrix determinant as cert instead of overlap probability. More discriminative. |

---

## 10. Current Results (v8, task-specific)

```
Total frames: 138   H=1: 97 (70%)   H=0: 41

Signal breakdown:
  cond_global_var fires (z_ivar > 2.0):    61 frames  ← most reliable
  cond_peak only (no gvar, no cert):        23 frames  ← mixed quality
  cond_low_cert only (no peak, no gvar):     9 frames
```

### Known ground truth verification ("0_Open the box")

| Frame | z_ivar | H_pred | Ground truth |
|-------|--------|--------|--------------|
| frame_0001 | 1.84 | 0 | 0 ✓ |
| frame_0004 | 90.1 | 1 | 1 ✓ |
| frame_0005 | 127.9 | 1 | 1 ✓ |
| frame_0006 | 75.4 | 1 | 1 ✓ |

### Known false positive pattern

**19 frames** are flagged H=1 by `cond_peak` alone with `z_ivar < 1.0` (global ivar is at or
below calibration baseline). These are likely structural false positives where the within-frame
variance distribution has a local spike in a region with inherent high variance (object texture,
articulation joint) but the frame overall is not anomalous.

Tuning options:
- Raise `--z_thresh` to 3.5–4.0 to silence weak local peaks.
- Add a guard: `cond_peak` only fires if `z_ivar > −0.5` (i.e., global ivar is not below baseline).

---

## 11. Run Commands

### Step 1 — Calibration (once per dataset)

```bash
cd feature_matching_eval_hallucination
PYTHONPATH=../../../RoMaV2/src:$PYTHONPATH \
conda run -n groot --no-capture-output python warp_variance_vis.py \
  --calibrate --task_refs \
  --calib_file ../results/warp_variance_v8/calib_task_specific.json \
  --high_dir ../image_no_bg/high \
  --setting turbo --use_precision --device cuda
```

### Step 2 — Inference (full batch)

```bash
PYTHONPATH=../../../RoMaV2/src:$PYTHONPATH \
conda run -n groot --no-capture-output python warp_variance_vis.py \
  --query_dir ../image_no_bg/low \
  --task_refs \
  --calib_file ../results/warp_variance_v8/calib_task_specific.json \
  --global_z_thresh 2.0 \
  --out_dir ../results/warp_variance_v8 \
  --setting turbo --use_precision --device cuda
```

### Step 3 — Single frame inference

```bash
PYTHONPATH=../../../RoMaV2/src:$PYTHONPATH \
conda run -n groot --no-capture-output python warp_variance_vis.py \
  --query "../image_no_bg/low/0_Open the box/frame_0005.png" \
  --task_refs \
  --calib_file ../results/warp_variance_v8/calib_task_specific.json \
  --global_z_thresh 2.0 \
  --out_dir ../results/warp_variance_v8 \
  --setting turbo --use_precision --device cuda
```

---

## 12. File Map

```
feature_matching_eval_hallucination/
├── warp_variance_vis.py          ← main detector (this document describes this)
│     ├── _compute_ivar_for_paths()   helper: ivar for one query vs given refs
│     ├── run_calibration()           task-specific or global calibration
│     ├── process_one()               single frame → H label + heatmaps + CSV row
│     ├── batch_reclassify()          post-hoc batch z-score (no calib fallback)
│     └── main()                      CLI entry point
│
└── results/
    ├── warp_variance_v7/             cross-task calibration baseline (for comparison)
    │   ├── calib_high2high.json      {"mean_ivar": 0.3481, "std_ivar": 0.0637}
    │   └── summary.csv
    └── warp_variance_v8/             task-specific calibration (current best)
        ├── calib_task_specific.json  {"task_refs": true, "tasks": {...}, "global": {...}}
        └── summary.csv
```
