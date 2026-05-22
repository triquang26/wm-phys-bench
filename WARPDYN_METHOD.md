# WARPDYN — Hallucination Detection Method

Production pipeline cho phát hiện hallucination ở video world-model (Cosmos, OpenSora, …).
Hai phase: **OFFLINE** (build null distribution per task, chạy 1 lần) và **ONLINE** (score
1 query video, ~30s). Không cần labels; chỉ cần 1 training video per task.

---

## Pipeline overview

```
                ┌──────────────────── OFFLINE (per task) ────────────────────┐
training.mp4 ──►│  1. Sample 120 frames evenly                                │
                │  2. SAM3 segment → background → gray (127,127,127)          │
                │  3. CYCLE NULL: 462 pairs × lags [1,2,5,10]                 │──► null_per_task/T.npz
                │     RoMa fwd+bwd → CycleSignal → (mean, peak)              │
                │     → sorted null arrays                                    │
                │  4. KNN POOL: DINOv2 ViT-S/14 CLS features (120×384)       │
                │  5. KNN LOO NULL: for each ref_i, score vs top-k others     │
                │     → (null_ivar, null_peak) + CV-routing decision          │
                │  6. H_TRAIN: run ONLINE on training video → baseline        │
                └─────────────────────────────────────────────────────────────┘

                ┌──────────────────── ONLINE (per query) ────────────────────┐
query.mp4   ──► │  1. Sample 10 frames + SAM3                                 │
                │  2. CYCLE branch: 9 pairs → H_pair each → p80 → H_cycle    │
                │  3. KNN branch: 10 frames → H_frame each → p80 → H_knn    │──► ratio + verdict
                │  4. FUSE: Cauchy([1−H_cycle, 1−H_knn]) → H_fused           │
                │  5. ratio = H_fused / H_train_fused                         │
                │  6. HALLU if ratio > 1.0 (borderline: 0.95–1.05)           │
                └─────────────────────────────────────────────────────────────┘
```

---

## Notation

| Symbol | Meaning |
|---|---|
| `p` | right-tail p-value ∈ (0,1); small p = anomalous |
| `H = 1 − p` | anomaly score ∈ (0,1); high H = anomalous |
| `N` | number of reference frames (120) |
| `K` | number of nearest refs used per kNN query (15) |
| `n_pairs` | number of null cycle pairs (≈462 for N=120) |
| `n_query` | number of online query frames (10) |

---

## OFFLINE — build null per task

Chạy **1 lần** mỗi task. Output: `null_per_task/task_<T>.npz`.

### Step 1 — Sample 120 reference frames

```python
idx = np.linspace(0, total_frames - 1, 120, dtype=int)
refs = [read_frame(training_mp4, i) for i in idx]
```

Sample đều (không random) → reproducible, cover đều trajectory.

### Step 2 — SAM3 segment

```python
seg = VideoFrameSegmenter(prompts=["robot arm", "robotic hand",
                                   "gripper", "mechanical finger"])
refs_seg = [seg.segment_frame(f) for f in refs]   # bg → (127,127,127) gray
```

Gray fill (127,127,127) là neutral: không tạo edge giả cho RoMa, và là background
convention mà `fg_mask_from_seg()` nhận biết (`~np.all(bgr == [127,127,127], axis=-1)`).

### Step 3 — Cycle null (multi-lag pairs)

Build null distribution cho `CycleSignal` bằng cách pair refs trong training ở nhiều lag.

```python
# All pairs across lags [1, 2, 5, 10]: 119 + 118 + 115 + 110 = 462 pairs
for lag in [1, 2, 5, 10]:
    for i in range(N - lag):
        fwd = matcher.match(refs[i], refs[i+lag])   # RoMa turbo
        bwd = matcher.match(refs[i+lag], refs[i])

        # Per-pixel cycle error (pixel units):
        #   err(x,y) = || bwd(fwd(x,y)) - (x,y) ||_2
        sig = CycleSignal(cert_floor=0.1).compute(fwd, bwd)

        # Two summary stats per pair:
        null_cycle_mean.append(sig.mean)   # cert-weighted mean of err_map on FG
        null_cycle_peak.append(sig.peak)   # 99th-percentile of err_map on FG
```

`null_cycle_mean` và `null_cycle_peak` là 2 sorted arrays (~462 values each) saved to .npz.

**CycleSignal aggregation** (trong `TemporalSignal._aggregate`):

```
valid pixels = interior_mask & (cert_fwd > 0.1)
weights      = cert_fwd(x,y) on valid pixels

mean = Σ_{valid} err(x,y) × cert(x,y) / Σ cert(x,y)   [cert-weighted mean]
peak = percentile(err[valid], 99)                        [robust max, not true max]
```

### Step 4 — DINOv2 pool

```python
dino = DinoFeatureExtractor("dinov2_vits14")  # ViT-S/14, D=384
feats = dino.extract(ref_pngs)                # (N, 384) L2-normalized CLS tokens

# Preprocessing before DINOv2:
#   1. Pad-to-square with gray (127,127,127)   → preserve aspect ratio
#   2. Resize to 224×224
#   3. ImageNet normalize: (x - μ)/σ  where μ=[0.485,0.456,0.406], σ=[0.229,0.224,0.225]
```

Cached per task ở `ref_cache/<task_slug>.npz` với SHA-256 key. Re-built nếu paths thay đổi.

### Step 5 — k-NN LOO null + CV-routing

**LOO null**: với mỗi ref_i, score nó như 1 query frame chống lại top-K của {pool \ i}.
Đây là cùng pipeline với ONLINE → null statistically consistent (cùng df, cùng K).

```python
for i in range(N):
    cand = [j for j in range(N) if j != i]
    sims = feats[cand] @ feats[i]                    # cosine similarity (L2-normed)
    top_k = cand[argtopk(sims, K)]                   # K=15 nearest refs
    k_paths = [ref_paths[j] for j in top_k]

    # RoMa batch match: ref_i vs each of 15 refs
    matches = matcher.match_batch(ref_paths[i], k_paths)
    warps = stack([m.warp for m in matches])         # (K, H, W, 2)
    precs = stack([m.precision for m in matches])    # (K, H, W, 2, 2)

    # Cochran deviance D(x,y) — Mahalanobis inter-ref disagreement
    D_map, _, _ = MahalanobisStatistics.ivar_per_pixel(warps, precs)

    # Two summary stats:
    ivar = mean(D_map[fg_mask])                      # interior mean of D
    peak = max_zscore(D_map, fg_mask)                # max z-score within fg

    null_ivar.append(ivar)
    null_peak.append(peak)
```

**Cochran deviance** (precision-weighted inter-ref disagreement):

```
Λ(x,y)   = Σ_r Σ⁻¹_r(x,y)                              # total precision (K, H, W, 2×2 → H, W, 2×2)
μ̂(x,y)  = Λ⁻¹ · Σ_r Σ⁻¹_r(x,y) warp_r(x,y)            # BLUE consensus warp
D(x,y)   = Σ_r (warp_r − μ̂)ᵀ Σ⁻¹_r (warp_r − μ̂)       # ~ χ²(2(K-1)) under H₀
```

High D = refs disagree, weighted by their own confidence. Phân biệt "real pose variation"
(low confidence khi uncertain) vs "generated artifacts" (high confidence nhưng inconsistent).

**CV-routing** — quyết định dùng `ivar` hay `peak` làm signal chính cho task này:

```python
cv = std(null_ivar) / mean(null_ivar)     # coefficient of variation
route = "peak" if cv < 0.50 else "ivar"
```

Lý do: nếu `ivar` null quá flat (CV thấp → ít discriminative), switch sang `peak` (max z-score,
robust hơn với domain where average D is uniformly shifted). Decision là per-task, offline.

### Step 6 — H_train baseline

Chạy **ONLINE pipeline** trên chính training video → baseline score mà query so sánh với.

```python
H_train_cycle  = run_online_cycle(training_mp4, null_cycle_mean, null_cycle_peak)
H_train_knn    = run_online_knn(training_mp4, pool, null_ivar, null_peak, route)
H_train_fused  = cauchy_combine_video(H_train_cycle, H_train_knn)
```

Self-normalization: H_train absorbs task difficulty. "Pour" khó hơn "Pick" nên H_train cao hơn,
nhưng ratio vẫn compare apples-to-apples.

---

## ONLINE — score 1 query video

Load `null_per_task/task_<T>.npz`. Chạy ~30s trên H100.

### Step 1 — Sample 10 frames + SAM3

```python
idx_q = np.linspace(0, total_frames - 1, 10, dtype=int)
frames = [read_frame(query_mp4, i) for i in idx_q]
seg_frames = [seg.segment_frame(f) for f in frames]
```

### Step 2 — CYCLE branch

```python
H_pairs = []
for t in range(9):                                      # 9 consecutive pairs
    fwd = matcher.match(seg[t], seg[t+1])
    bwd = matcher.match(seg[t+1], seg[t])
    sig = CycleSignal(cert_floor=0.1).compute(fwd, bwd)

    # Per pair: convert both stats to p-values, Cauchy-combine → H_pair
    p_mean = empirical_p(sig.mean, null_cycle_mean)     # right-tail
    p_peak = empirical_p(sig.peak, null_cycle_peak)
    H_pair = 1 - cauchy_combine([p_mean, p_peak])
    H_pairs.append(H_pair)

H_cycle = np.percentile(H_pairs, 80)   # "cycle_peak" column in CSV
```

p80 thay vì max → robust với 1–2 pair noise, sensitive với hallu kéo dài ≥ 2 pairs.

### Step 3 — KNN branch

```python
H_frames = []
for i in range(10):
    q_feat = dino.extract([seg[i]])[0]                  # (384,) L2-normed
    sims = pool_feats @ q_feat                           # cosine sim với all refs
    top_k = argtopk(sims, K=15)

    matches = matcher.match_batch(seg[i], [pool_paths[j] for j in top_k])
    warps = stack([m.warp for m in matches])             # (15, H, W, 2)
    precs = stack([m.precision for m in matches])        # (15, H, W, 2, 2)

    D_map, _, _ = MahalanobisStatistics.ivar_per_pixel(warps, precs)
    ivar = mean(D_map[fg_mask])
    peak = max_zscore(D_map, fg_mask)

    # Use routed signal (decided offline):
    sig_val = peak if route == "peak" else ivar
    null    = null_peak if route == "peak" else null_ivar
    p_frame = empirical_p(sig_val, null)
    H_frame = 1 - p_frame
    H_frames.append(H_frame)

H_knn = np.percentile(H_frames, 80)    # "knn_peak" column in CSV
```

### Step 4 — Fusion

```python
# Video-level Cauchy combine (ACAT):
p_cycle = 1 - H_cycle
p_knn   = 1 - H_knn
p_fused = cauchy_combine([p_cycle, p_knn])
H_fused = 1 - p_fused           # "fused_peak" column in CSV
```

### Step 5 — Ratio vs H_train

```python
ratio_cycle  = H_cycle  / H_train_cycle    # H_train from Step 6 OFFLINE
ratio_knn    = H_knn    / H_train_knn
ratio_fused  = H_fused  / H_train_fused    # ← primary decision metric
```

### Step 6 — Verdict

```python
if   ratio_fused > 1.05:  verdict = "HALLU"
elif ratio_fused > 0.95:  verdict = "borderline"
else:                      verdict = "clean"
```

Band 0.95–1.05 = sampling noise zone. Outside band = dứt khoát.

---

## Score computation — exact formulas

### empirical_p_value (right-tail, Laplace-smoothed)

```python
def empirical_p_value(value: float, sorted_null: np.ndarray) -> float:
    n = len(sorted_null)
    rank = np.searchsorted(sorted_null, value, side="right")   # # nulls < value
    p = (n - rank + 0.5) / (n + 1.0)                          # Laplace smooth
    return np.clip(p, 1/(n+1), 1 - 1/(n+1))
```

- `rank = 0` (value below all null) → `p ≈ 1.0` (very normal)
- `rank = n` (value above all null) → `p ≈ 1/(n+1)` (extremely anomalous)
- Laplace smoothing `+0.5` avoids `p=0` or `p=1`

### cauchy_combine (ACAT)

```python
def cauchy_combine(ps: list[float]) -> float:
    # Filter out None and boundary values
    ps = [p for p in ps if 0 < p < 1]
    t = mean([tan(π × (0.5 − p)) for p in ps])
    return 0.5 − arctan(t) / π
```

- Mỗi p → `tan(π(0.5−p))`: map [0,1] → ℝ, heavy tails
- Average trong Cauchy space → map ngược về [0,1]
- **Key property**: nếu một p rất nhỏ (anomalous) → `tan()` rất lớn → combined p nhỏ, kể cả khi p kia bình thường
- Không assume independence (khác Fisher), không bị dominated bởi p lớn (khác max)

### cauchy_combine_video

```python
def cauchy_combine_video(h_cycle: float, h_knn: float) -> float:
    p_cycle = 1 - clip(h_cycle, 1e-6, 1-1e-6)
    p_knn   = 1 - clip(h_knn, 1e-6, 1-1e-6)
    return 1 - cauchy_combine([p_cycle, p_knn])
```

### H aggregation trong ONLINE

```
H_cycle = percentile([H_pair_0, ..., H_pair_8], 80)    # p80 across 9 pairs
H_knn   = percentile([H_frame_0, ..., H_frame_9], 80)  # p80 across 10 frames
```

---

## Bootstrap (BaselineNormalizer)

`BaselineNormalizer` trong `warp_score/fusion.py` ước lượng **độ ổn định** của H_train
qua bootstrap resampling. Output là `σ` (std của H_train_fused dưới sampling noise)
và sigmoid normalization `score ∈ [0,1]`.

**Fit** (chạy offline sau khi có H_train):

```python
bn = BaselineNormalizer(n_boot=200, pct=80, seed=42)
bn.fit(h_pairs_train, h_frames_train)
# h_pairs_train: list of H_pair from cycle branch of training video
# h_frames_train: list of H_frame from kNN branch of training video
```

Bên trong `fit()`:

```python
for b in range(200):                    # 200 bootstrap iterations
    hp_b = rng.choice(hp, n_p, replace=True)    # resample cycle H_pair
    hf_b = rng.choice(hf, n_f, replace=True)    # resample kNN H_frame
    H_b  = cauchy_combine_video(
               percentile(hp_b, 80),            # H_cycle resampled
               percentile(hf_b, 80)             # H_knn resampled
           )
    boot[b] = H_b                               # H_fused of resampled draw

sigma = std(boot)                       # variability of H_train under noise
alpha = 1.0 / max(sigma, 1e-6)         # sharpness
```

**Normalize** (optional, maps ratio → calibrated score):

```python
z     = alpha × (ratio - 1.0)
score = sigmoid(z) = 1 / (1 + exp(-z))
```

- `ratio = 1.0` → `z = 0` → `score = 0.5`
- `ratio > 1.0` (HALLU) → `score > 0.5`
- `ratio < 1.0` (clean) → `score < 0.5`
- σ nhỏ → α lớn → steep sigmoid → sharp decision
- σ lớn → α nhỏ → flat sigmoid → conservative (uncertain baseline)

**Note**: main eval (`eval_per_task_dense_null.py`) dùng **raw ratio threshold** (>1.0)
chứ không phải sigmoid score. `BaselineNormalizer` là utility cho downstream calibration.

---

## Parameters

| Group | Param | Value | Why |
|---|---|---|---|
| Refs | `N_REFERENCE_FRAMES` | 120 | max from 120-frame training videos |
| Refs | `SAM3_PROMPTS` | robot arm, robotic hand, gripper, mechanical finger | covers GR-1 morphology |
| Cycle | `NULL_LAGS` | [1, 2, 5, 10] | short+long range null, 462 total pairs |
| Cycle | `CERT_FLOOR` | 0.1 | mask low-cert pixels (bg, occluded) |
| Cycle | `PEAK_PERCENTILE` | 99 | robust max, not sensitive to single outlier |
| kNN | `K` | 15 | enough diversity, not too slow; same LOO vs inference |
| kNN | `DINO_MODEL` | dinov2_vits14 | D=384, fast, good pose similarity |
| kNN | `CV_THRESHOLD` | 0.50 | empirically separates flat vs discriminative ivar nulls |
| RoMa | `SETTING` | turbo | ~3× faster than `outdoor`, acceptable accuracy |
| RoMa | `VIS_SIZE` | 224 | square output, matches DINOv2 preprocess |
| Online | `N_QUERY_FRAMES` | 10 | 9 pairs, covers ~12-frame stride on 120-frame video |
| Online | `VIDEO_AGGREGATOR` | p80 | `np.percentile(H_pairs, 80)` |
| Decision | `RATIO_THRESHOLD` | 1.0 | by construction FPR=0% on training itself |
| Decision | `BORDERLINE_BAND` | 0.95–1.05 | ±5% sampling noise buffer |
| Bootstrap | `N_BOOT` | 200 | enough for σ estimate, cheap |
| Bootstrap | `BOOT_PCT` | 80 | matches `VIDEO_AGGREGATOR` percentile |

---

## Assumptions

| Assumption | Consequence if violated |
|---|---|
| ≥1 training video per task, ≥30 frames | null too small → unstable p-values |
| SAM3 can isolate robot foreground | cycle + kNN both fail on wrong mask |
| Query is from the same task/robot | ratio invalid if task-mismatch |
| CUDA GPU | RoMa turbo ~30s GPU; CPU would be >5 min |
| DINOv2 reachable via `torch.hub` | auto-downloaded on first run |
| Training video covers ≥half of task trajectory | H_train too low → ratio inflated |
| Background stays gray (127,127,127) after SAM3 | fg_mask downstream correct |

---

## File map

| File | Role |
|---|---|
| `warp_score/temporal_signals.py` | `CycleSignal`, `empirical_p_value`, `cauchy_combine`, `cycle_error_map` |
| `warp_score/matcher.py` | `RoMaMatcher` — `match()`, `match_batch()`, `match_bidir()` |
| `warp_score/sam_segmenter.py` | `VideoFrameSegmenter` (SAM3) |
| `warp_score/adaptive_refs.py` | `DinoFeatureExtractor` (CLS tokens), `AdaptiveRefSelector` (top-k cosine) |
| `warp_score/statistics.py` | `MahalanobisStatistics.ivar_per_pixel()` (Cochran deviance) |
| `warp_score/knn_signal.py` | `KNNFrameSignal` — pool build, LOO calibration, CV-routing, score_frame |
| `warp_score/fusion.py` | `cauchy_combine_video`, `BaselineNormalizer`, `complementarity_report` |
| `scripts/extract_refs_dense.py` | OFFLINE step 1–2: sample 120 frames + SAM3 |
| `scripts/eval_per_task_dense_null.py` | OFFLINE step 3–6 + ONLINE eval → CSV |
| `scripts/compute_per_task_ratio.py` | ratio table + complementarity + markdown report |
| `scripts/benchmark_one_task.py` | end-to-end timing + visualizations for 1 task |

---

## Results (GR-1, 5 tasks, 24 gens)

| Metric | Cycle | kNN | **Fused** |
|---|---|---|---|
| Gen catch (`ratio > 1.0`) | 18/24 | 18/24 | **19/24** |
| AUROC | 0.7750 | 0.7833 | **0.8000** |
| Separation gap (min hallu − max real) | +0.014 | +0.002 | **+0.031** |
| Real flagged (FPR) | 0/5 | 0/5 | **0/5** |
| Complementarity | — | — | 14 both / 4 cycle-only / 4 knn-only / 2 neither |

Fusion tăng separation gap 2× (+0.014 → +0.031) và bắt thêm 1 gen mà không tạo false positive.
