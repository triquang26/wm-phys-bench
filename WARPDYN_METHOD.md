# WarpDyn — phương pháp production

Per-task multi-lag null + cycle composition signal + per-task ratio score.

Đây là doc PHƯƠNG PHÁP DUY NHẤT — step-by-step, OFFLINE và ONLINE. Không có alternative, không có history.

---

## Tổng quan

```
                    ┌─────────────────────────┐
                    │   OFFLINE (per task)     │
                    │   ────────────────────   │
   training mp4 ──► │  1. Sample 50 frames     │
                    │  2. SAM3 segment         │
                    │  3. Build multi-lag      │
                    │     pairs               │
                    │  4. RoMa + cycle signal  │
                    │  5. Compute H_train      │
                    │  6. Save .npz cache      │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                       null_per_task/<T>.npz
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   ONLINE (per query)     │
                    │   ────────────────────   │
   query mp4   ──► │  1. Sample 10 frames     │
                    │  2. SAM3 segment         │
                    │  3. RoMa per pair        │
                    │  4. Cycle signal         │
                    │  5. Empirical p + Cauchy │
                    │  6. p80 aggregator       │
   ratio       ◄── │  7. ratio = H/H_train    │
                    └─────────────────────────┘
```

---

## OFFLINE — per task T (chạy 1 lần)

Input: 1 file `training.mp4` của task T.
Output: 1 file `null_per_task/<T>.npz` (~5 KB).
Cost: ~8 phút trên 1 GPU.

### Step O1 — Sample 50 reference frames

```
Đọc training.mp4 → tổng số frames N
indices = np.linspace(0, N-1, 50, dtype=int)
ref_bgrs = [đọc frame N tại i for i in indices]
```

→ 50 ảnh BGR.

### Step O2 — SAM3 segment 50 frames

```
sam = VideoFrameSegmenter()
for bgr in ref_bgrs:
    seg = sam.segment_frame(bgr)
    # text prompts: ["robot arm", "robotic hand", "gripper",
    #                "mechanical finger"]
    # foreground giữ nguyên, background → (127, 127, 127)
    save: reference/<T>/frame_NNNN.png
```

→ 50 ảnh PNG đã SAM3-segmented.

### Step O3 — Build multi-lag pairs

```
NULL_LAGS = [1, 2, 5, 10]
pairs = []
for lag in NULL_LAGS:
    for i in range(50 - lag):
        pairs.append( (ref[i], ref[i+lag]) )

Số pairs:
  lag=1  → 49 pairs
  lag=2  → 48 pairs
  lag=5  → 45 pairs
  lag=10 → 40 pairs
  ──────────────────
  Total: 182 pairs
```

### Step O4 — RoMa + cycle signal cho mỗi pair

```
matcher = RoMaMatcher(setting="turbo", device="cuda",
                       use_precision=True, vis_size=224)
cycle   = CycleSignal(cert_floor=0.1)

null_means = []
null_peaks = []

for (frame_a, frame_b) in pairs:
    fwd = matcher.match(frame_a, frame_b)   # warp_AB + cert
    bwd = matcher.match(frame_b, frame_a)   # warp_BA + cert
    sig = cycle.compute(fwd, bwd)
    null_means.append(sig.mean)
    null_peaks.append(sig.peak)

null_mean = np.sort(np.array(null_means))   # ascending
null_peak = np.sort(np.array(null_peaks))
```

**CycleSignal công thức:**

```
For each pixel (x, y) in frame_a:
  (u, v)   = pixel_coord(warp_fwd[y, x])              # land in frame_b
  (x', y') = bilinear_sample(warp_bwd, (u, v))        # back in frame_a
  drift(y, x) = ||(x', y') - (x, y)||                 # in pixel units

valid = (cert_fwd > 0.1) AND interior_mask

mean = (Σ drift × cert × valid) / (Σ cert × valid)    # cert-weighted
peak = percentile(drift[valid], 99)                   # tail anomaly
```

### Step O5 — Compute H_train baseline

Score chính training video qua online pipeline (Step N1-N6 ở dưới) chống null vừa build:

```
sample 10 frames từ training.mp4
SAM3 segment 10 frames
score 9 consecutive pairs với null_mean / null_peak
H_train = np.percentile(H_pairs, 80)
```

→ scalar `H_train ∈ [0, 1]` = anomaly level của chính training video.

### Step O6 — Save cache cho task T

```
np.savez(f"null_per_task/{T}.npz",
         null_mean       = null_mean,        # 182 sorted floats
         null_peak       = null_peak,        # 182 sorted floats
         H_train         = H_train,          # scalar baseline
         null_lags       = NULL_LAGS,
         n_ref_frames    = 50,
         training_video  = "training.mp4")
```

---

## ONLINE — score 1 query video

Input: `query.mp4` + `task_T` (cần biết task).
Output: `H_video ∈ [0, 1]` + `ratio = H_video / H_train`.
Cost: ~10 sec / video.

### Step N1 — Identify task

```
task_T = ...   # user input HOẶC DINOv2 auto-classify
null = load(f"null_per_task/{task_T}.npz")
```

### Step N2 — Sample 10 frames đều

```
Đọc query.mp4 → tổng số frames N_query
indices = np.linspace(0, N_query - 1, 10, dtype=int)
query_bgrs = [đọc frame tại i for i in indices]
```

### Step N3 — SAM3 segment

```
sam = VideoFrameSegmenter()
query_seg = [sam.segment_frame(b) for b in query_bgrs]
```

→ 10 frames đã segmented, cùng style với null pool.

### Step N4 — RoMa + cycle signal cho 9 consecutive pairs

```
H_pairs = []

for t in range(9):
    fwd = matcher.match(query_seg[t],   query_seg[t+1])
    bwd = matcher.match(query_seg[t+1], query_seg[t])
    sig = cycle.compute(fwd, bwd)

    p_mean = empirical_p(sig.mean, null["null_mean"])
    p_peak = empirical_p(sig.peak, null["null_peak"])
    p_pair = cauchy_combine([p_mean, p_peak])

    H_pair = 1.0 - p_pair
    H_pairs.append(H_pair)
```

**empirical_p:**
```
def empirical_p(value, sorted_null):
    n = len(sorted_null)
    rank = np.searchsorted(sorted_null, value, side="right")
    p = (n - rank + 0.5) / (n + 1)
    return clip(p, 1/(n+1), 1 - 1/(n+1))
```

**cauchy_combine (ACAT):**
```
def cauchy_combine(p_list):
    T = mean(tan(pi * (0.5 - p)) for p in p_list if 0 < p < 1)
    return 0.5 - arctan(T) / pi
```

### Step N5 — Aggregate per-video

```
H_video = np.percentile(H_pairs, 80)   # 80th percentile, robust peak
```

Tại sao p80:
- max sensitive với 1 noisy frame (motion blur, SAM3 fail)
- mean dilute tín hiệu nếu chỉ vài frames hallu
- p80: catch ≥ 1-2 hallu pairs, resistant với 1-frame noise

### Step N6 — Per-task ratio + verdict

```
ratio = H_video / null["H_train"]

if   ratio >= 1.00:  verdict = "HALLU"
elif ratio >= 0.95:  verdict = "borderline"
else:                verdict = "clean"
```

Continuous output: `H_video` ∈ [0, 1] và `ratio` ∈ [0, ∞).

---

## Output mỗi query

```
{
  "task":        "1_Use the right hand to pick up green bok choy ...",
  "H_video":     0.877,
  "H_train":     0.728,
  "ratio":       1.205,
  "verdict":     "HALLU",
  "per_frame": [
    {"idx": 0,  "H_pair": 0.91},
    {"idx": 1,  "H_pair": 0.73},
    {"idx": 2,  "H_pair": 0.65},
    ...
    {"idx": 8,  "H_pair": 0.88}
  ]
}
```

---

## Tham số mặc định

```python
# OFFLINE
NULL_LAGS         = [1, 2, 5, 10]   # 182 null pairs per task
N_REFERENCE_FRAMES = 50              # frames extracted per task
SAM3_PROMPTS      = ["robot arm", "robotic hand", "gripper",
                      "mechanical finger"]

# CYCLE SIGNAL
CERT_FLOOR        = 0.1              # drop low-cert pixels (uniform texture)
PEAK_PERCENTILE   = 99               # tail aggregation

# ROMA
ROMA_SETTING      = "turbo"          # speed / accuracy preset
ROMA_VIS_SIZE     = 224              # warp output resolution

# ONLINE
N_QUERY_FRAMES    = 10               # frames sampled per query
VIDEO_AGGREGATOR  = "p80"            # percentile over 9 H_pairs

# DECISION
RATIO_THRESHOLD   = 1.0              # ratio > 1.0 → HALLU
BORDERLINE_BAND   = 0.05             # 0.95-1.0 = borderline
```

---

## Commands

```bash
conda activate groot
cd /mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination

# ─── Run OFFLINE + ONLINE end-to-end cho 5 GR1 tasks ───
python scripts/eval_per_task_dense_null.py
# → paper-physical-gr1/per_task_dense_eval/per_task_dense_table.csv

# ─── Compute per-task ratio + ranking ───
python scripts/compute_per_task_ratio.py
# → paper-physical-gr1/per_task_dense_eval/per_task_ratio_table.csv
# → paper-physical-gr1/per_task_dense_eval/per_task_ratio_ranking.md
```

---

## Files quan trọng

| File | Purpose |
|---|---|
| `warp_score/temporal_signals.py` | `CycleSignal` class, `empirical_p_value`, helpers |
| `warp_score/matcher.py` | `RoMaMatcher.match()` |
| `warp_score/sam_segmenter.py` | `VideoFrameSegmenter` |
| `scripts/eval_per_task_dense_null.py` | End-to-end OFFLINE + ONLINE pipeline |
| `scripts/compute_per_task_ratio.py` | Per-task ratio scoring |

---

## Yêu cầu input

| Yêu cầu | Bắt buộc | Note |
|---|---|---|
| Real training video per task | ✓ | 1 video / task tối thiểu, càng nhiều càng tốt |
| Task ID khi query | ✓ | User truyền vào HOẶC auto-classify DINOv2 |
| SAM3 weights | ✓ | `facebook/sam3` HF hub |
| RoMa weights | ✓ | bundled trong `third_party/RoMaV2/` |
| Query video format | ✓ | `.mp4` ≥ 2 sec |

---

## Kết quả tham chiếu (GR1, 5 tasks)

| Metric | Value |
|---|---|
| Real videos H_peak | 0.62 – 0.83 |
| Real max H_peak | **0.832** |
| Gen videos H_peak | 0.31 – 0.99 |
| Gen catch ratio > 1.0 | **18/24 (75%)** |
| Gen catch global T = 0.832 | 16/24 (67%) |
| AUROC | 0.77 |
| FPR trên 5 real | **0% by construction** |
| Time / training video (offline) | ~8 phút |
| Time / query video (online) | ~10 sec |
