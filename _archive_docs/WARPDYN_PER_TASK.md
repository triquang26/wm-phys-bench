# WarpDyn — Per-task multi-lag null (production method)

Hallucination / anomaly detection cho robot video bằng RoMa feature matching.
Per-task null distribution + multi-lag sampling → bảo đảm real training video không bị flag, capture được generator artifacts.

**Kết quả GR1** (5 tasks, multi-lag null 9 lags, ~410 pairs/task):
- Real max H_peak = **0.832** → set threshold = 0.832 cho FPR = 0%
- 16-18/24 generated videos flagged tùy aggregator (cycle peak / per-task ratio)
- AUROC = 0.77 (cycle peak), 0.81 với ratio score

---

## 1. Hai pipeline tách biệt: OFFLINE và ONLINE

```
═════════════════════════════════════════════════════════════════════
                        ┌── OFFLINE (mỗi task, 1 lần) ──┐
                        │                                │
                        │   Build per-task null         │
                        │   + threshold calibration     │
                        │                                │
                        └──────────────┬─────────────────┘
                                       │
                                       ▼
                            null_per_task/<T>.npz
                                       │
                                       ▼
                        ┌── ONLINE (mỗi query video) ──┐
                        │                                │
                        │   Score against per-task null │
                        │   → continuous H_peak ∈ [0,1] │
                        │                                │
                        └────────────────────────────────┘
═════════════════════════════════════════════════════════════════════
```

OFFLINE chạy **1 lần per task** (~5-10 min/task), output là cache nhỏ (~1 KB).
ONLINE chấm 1 video ~10 sec với cache đó.

---

## 2. OFFLINE — Build null cho 1 task T

### Step O1: Sample reference frames từ training video

Input: real training video của task T (`.mp4`)

```
For one training video V at task T:
  total_frames = read_mp4(V).num_frames
  ref_indices = np.linspace(0, total_frames - 1, 50)  # 50 frames evenly
  ref_bgrs = [read_frame(V, i) for i in ref_indices]
```

→ 50 raw BGR frames.

### Step O2: SAM3 segment mỗi reference frame

```
For each bgr in ref_bgrs:
  seg_bgr = SAM3(bgr, prompts=["robot arm", "robotic hand", "gripper"])
  # Background → (127, 127, 127), foreground giữ nguyên
  save_to: reference/<task_T>/frame_NNNN.png
```

→ 50 segmented PNG files.

**Vì sao SAM3:** Bỏ background bias. RoMa matching focus vào robot + objects, không bị cluttered scene.

### Step O3: Build multi-lag pair list

Với 50 reference frames, build pairs ở 9 lags:

```
NULL_LAGS = [1, 2, 5, 10]

pairs = []
for lag in NULL_LAGS:
    n_pairs_this_lag = 50 - lag
    for i in range(n_pairs_this_lag):
        pairs.append( (ref[i], ref[i + lag]) )

Tổng pairs per task:
  49 + 48 + 45 + 40 = 182 pairs
```

**Vì sao multi-lag:**
- lag=1: motion rất nhỏ (frames liền kề)
- lag=10: motion lớn (frames cách xa)
- Inference sample 10 frames từ ~120-total → lag effective ≈ 12 → null phải có pairs ở lag tương đương để cycle drift comparable
- 4 lags {1, 2, 5, 10} cover spectrum đầy đủ và best AUROC theo experiment

**Lưu ý:** đã thử thêm lags ({1,2,3,4,5,6,8,10,15}) → 410 pairs nhưng AUROC giảm 0.767 → 0.750. Cả real và gen H scale xuống cùng → net wash. **4-lag là sweet spot.**

### Step O4: Compute cycle signal cho mỗi pair

```
For each (frame_a, frame_b) in pairs:
  fwd = RoMa.match(frame_a, frame_b)   # (H,W,2) warp + (H,W) cert
  bwd = RoMa.match(frame_b, frame_a)
  sig = CycleSignal().compute(fwd, bwd)
  null_means.append(sig.mean)
  null_peaks.append(sig.peak)

Sort both arrays:
  null_mean = np.sort(null_means)
  null_peak = np.sort(null_peaks)
```

### Step O5: Score training video V để xác định ngưỡng

```
training_pairs = []
training_indices = np.linspace(0, total_frames - 1, 10)  # match inference sampling
for i in training_indices:
    frame_i_seg = SAM3(read_frame(V, i))
    training_frames.append(frame_i_seg)

for t in range(9):
    fwd = RoMa(training_frames[t], training_frames[t+1])
    bwd = RoMa(training_frames[t+1], training_frames[t])
    sig = CycleSignal().compute(fwd, bwd)
    p_mean = empirical_p(sig.mean, null_mean)
    p_peak = empirical_p(sig.peak, null_peak)
    p_pair = Cauchy(p_mean, p_peak)
    H_pair_training.append(1 - p_pair)

H_train_task = np.percentile(H_pair_training, 80)   # 80th-percentile aggregator
```

`H_train_task` ∈ [0, 1] = "anomaly level" của training video so với chính null của nó. **Set this là baseline cho task T.**

### Step O6: Save cache cho task T

```
np.savez(f"null_per_task/{task_T}.npz",
         null_mean   = null_mean,           # ~396 sorted floats
         null_peak   = null_peak,           # ~396 sorted floats
         H_train     = H_train_task,        # scalar baseline
         training_video = str(V))           # provenance
```

→ Compact cache (~10 KB) ready để serve inference.

---

## 3. ONLINE — Score 1 query video

### Step N1: Identify task của query

Cần biết query thuộc task nào để load đúng null. 3 cách:
- **User-provided** (recommended): `task_id` truyền vào API
- **Auto-classify**: DINOv2 embedding của 1 query frame → nearest task centroid
- **Multi-score**: chấm với mọi task null → lấy min H (best-match task)

```
task_T = identify_task(query_video)
null   = load(f"null_per_task/{task_T}.npz")
```

### Step N2: Sample N frames từ query

```
total = read_mp4(query).num_frames
indices = np.linspace(0, total - 1, n_frames=10)
bgrs = [read_frame(query, i) for i in indices]
```

→ 10 raw BGR frames.

### Step N3: SAM3 segment mỗi frame

```
seg_bgrs = [SAM3(b) for b in bgrs]
```

→ 10 segmented frames, same style as null pool.

### Step N4: RoMa per consecutive pair + cycle signal

```
H_pairs = []
for t in range(9):                                  # 9 consecutive pairs
    fwd = RoMa.match(seg_bgrs[t],   seg_bgrs[t+1])
    bwd = RoMa.match(seg_bgrs[t+1], seg_bgrs[t])
    sig = CycleSignal().compute(fwd, bwd)
    # Empirical p-value vs null:
    p_mean = empirical_p(sig.mean, null["null_mean"])
    p_peak = empirical_p(sig.peak, null["null_peak"])
    p_pair = Cauchy_combine(p_mean, p_peak)
    H_pairs.append(1 - p_pair)
```

Mỗi `H_pair` ∈ [0, 1] = anomaly score của pair đó so với null distribution của task T.

### Step N5: Aggregate per-video

```
H_video_peak = np.percentile(H_pairs, 80)   # robust peak, ignores 1-frame noise
```

Tại sao `p80` thay max:
- max sensitive với single outlier frame (motion blur, occlusion)
- mean dilute tín hiệu nếu chỉ vài frames hallu
- p80: catch "≥ 1-2 hallu pairs" nhưng resistant với 1-frame noise

### Step N6: Quyết định binary (optional)

**Option A — Global threshold** (FPR=0% trên 5 tasks đã eval):
```
is_hallu = H_video_peak > 0.832
```

**Option B — Per-task ratio** (recommended, FPR=0% by construction):
```
ratio = H_video_peak / null["H_train"]
verdict = "HALLU"      if ratio > 1.00
          "borderline" if ratio > 0.95
          "clean"      otherwise
```

Per-task ratio = "video này anomalous gấp X lần training video cùng task". Tự calibrate khi thêm task mới.

---

## 4. Công thức chi tiết

### CycleSignal

```
Input:  warp_fwd (H,W,2), cert_fwd (H,W)   from RoMa(frame_a → frame_b)
        warp_bwd (H,W,2), cert_bwd (H,W)   from RoMa(frame_b → frame_a)

Per pixel (x,y) trong frame_a:
  (u, v)   = pixel_coord(warp_fwd[y,x])           # land in frame_b
  (x',y')  = bilinear_sample(warp_bwd, (u,v))     # back in frame_a
  drift    = ||(x',y') - (x,y)||                  # in pixel units

Filter: valid = (cert_fwd > 0.1) AND interior_mask

Aggregate:
  mean = (Σ drift × cert × valid) / (Σ cert × valid)   # cert-weighted
  peak = percentile(drift[valid], 99)                  # tail anomaly

Output: (mean, peak)
```

### Empirical p-value

```
def empirical_p(value, sorted_null):
    n = len(sorted_null)
    rank = np.searchsorted(sorted_null, value, side="right")
    p = (n - rank + 0.5) / (n + 1)
    return clip(p, 1/(n+1), 1 - 1/(n+1))    # avoid 0 or 1 exactly
```

### Cauchy combine (ACAT)

```
def cauchy_combine(p_list):
    valid = [p for p in p_list if 0 < p < 1]
    T = mean(tan(pi * (0.5 - p)) for p in valid)
    p_combined = 0.5 - arctan(T) / pi
    return p_combined
```

Robust hơn Fisher khi p-values correlate (mean và peak từ cùng pair có correlation).

---

## 5. Sample sizes (5 tasks GR1)

| Item | Count | Time (one-shot) |
|---|---|---|
| Reference frames per task | 50 | SAM3 ~30 sec |
| Multi-lag pairs per task | ~396 | RoMa ~7 min |
| Training videos per task | 1 | included above |
| H_train computation | 9 pairs | ~10 sec |
| **Total offline per task** | | **~8-10 min** |
| Frames per query | 10 | sample ~0.5 sec |
| Pairs per query | 9 | RoMa ~5 sec |
| **Total online per video** | | **~8-10 sec** |

---

## 6. Threshold strategies

### A. Global threshold (simple, không cần baseline per task)

Threshold = quantile của H_peak trên real training videos đã evaluate.

| FPR target | Quantile | GR1 example | Catch rate |
|---|---|---|---|
| 0% | max + ε | 0.832 | 16/24 (67%) |
| 5% | p95 | 0.815 | 17/24 |
| 10% | p90 | 0.770 | 19/24 |

### B. Per-task ratio (recommended) ⭐

Mỗi task lưu H_train. Test ratio = H_test / H_train.

```
ratio > 1.0   → HALLU (anomalous hơn training)
0.95-1.0      → borderline
< 0.95        → clean
```

**Lợi:**
- FPR = 0% by construction cho mỗi task
- Continuous interpretation: "X% anomalous hơn training"
- Cross-task fair: task có baseline cao không penalize unfairly
- Auto-calibrate khi thêm task mới

**GR1 result:** 18/24 hallu (vs 16/24 global).

---

## 7. Cấu trúc cache

Thư mục portable, copy được:

```
null_per_task/
├── 1_Use_the_right_hand_to_pick_up_green_bok_choy.npz
├── 2_Use_the_right_hand_to_pick_up_rubiks_cube.npz
├── 3_Use_the_right_hand_to_pick_up_banana.npz
├── 4_Use_the_left_hand_to_pick_up_dragonfruit.npz
└── 6_Use_the_right_hand_to_pick_up_orange.npz
```

Mỗi file ~ 5-10 KB chứa `null_mean`, `null_peak`, `H_train`, metadata.

---

## 8. Code reference

| File | Purpose |
|---|---|
| `warp_score/temporal_signals.py` | `CycleSignal` class (OOP, cert-weighting) |
| `warp_score/matcher.py` | `RoMaMatcher` (warp + cert per pair) |
| `warp_score/sam_segmenter.py` | `VideoFrameSegmenter` cho bg removal |
| `scripts/eval_per_task_dense_null.py` | End-to-end pipeline (offline + online) |
| `scripts/compute_per_task_ratio.py` | Per-task ratio score từ raw H |

### Run end-to-end (5 tasks GR1)

```bash
conda activate groot

# Offline + Online cho 5 eval tasks
python scripts/eval_per_task_dense_null.py
# → per_task_dense_eval/per_task_dense_table.csv

# Compute per-task ratio ranking
python scripts/compute_per_task_ratio.py
# → per_task_dense_eval/per_task_ratio_table.csv
# → per_task_dense_eval/per_task_ratio_ranking.md
```

---

## 9. Hyperparameters

| Param | Default | Sensitivity |
|---|---|---|
| Reference frames per task | 50 | Tăng → null dày hơn, SAM3 cost tăng |
| `NULL_LAGS` | `[1,2,3,4,5,6,8,10,15]` | More lags → smoother null, +linear compute |
| `n_frames` per query | 10 | Tăng nếu video > 8 sec, RoMa cost tăng |
| `cert_floor` | 0.1 | Tăng → 0.2 nếu texture đối xứng (rubik) |
| Aggregator | `p80` (80th percentile) | `max` sensitive, `median` robust |
| Threshold | per-task ratio = 1.0 | Hoặc global = 0.832 |

---

## 10. Limitations & failure modes

1. **Yêu cầu task ID** cho query. Cross-task pool (cũ) không cần, nhưng accuracy thấp hơn.
2. **1 training video / task** → H_train là point estimate, không có variance estimate. Nếu có 5+ real videos per task, có thể compute proper std và z-score.
3. **Pure feature matching ceiling**: Cosmos sinh motion realistic → cycle drift ≈ real ở 1 số videos. Cần semantic signal (VLA action, 3D pose) để vượt ceiling này.
4. **SAM3 noise**: nếu SAM3 segment sai → high noise vào cycle. Robust hơn khi cert_floor được tăng.
5. **Video quá ngắn (< 2 sec)**: ít pairs → p80 unstable. Cần n_frames ≥ 5.

---

## 11. So sánh với cross-task pool method (cũ)

| Aspect | Cross-task pool | **Per-task multi-lag (this)** |
|---|---|---|
| Null source | 92-task pool, 1472 pairs total | Per-task, ~410 pairs each |
| Lags | Lag-1 only (or random) | **{1,2,3,4,5,6,8,10,15}** |
| k-NN selection | DINOv2 top-K từ 4600 frames | Không cần (null is task-scoped) |
| Real false positive | 2-3 / 30 ở FPR=5% | **0 / 5 by construction** |
| Gen catch @ FPR=0% | 2-3 / 24 | **16-18 / 24** |
| Task ID required | No | Yes |
| Compute (offline) | ~25 min total | ~10 min per task |
| Interpretability | scalar H | scalar H + ratio so với training |
