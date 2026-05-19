# WARPDYN — Hallucination Detection Method

Tài liệu mô tả **production flow** của WARPDYN cho việc phát hiện hallucination ở video do world-model sinh ra. Pipeline gồm 2 phase: **OFFLINE** (build null distribution per task, chạy 1 lần) và **ONLINE** (score 1 query video, < 30s).

Default method = **cycle + kNN fusion**. Mọi step dưới đây là production, không phải variant.

---

## Pipeline overview

```
                ┌───────────────── OFFLINE (per task, ~30 min) ─────────────────┐
training.mp4 ──►│  1. Sample 120 frames evenly                                   │
                │  2. SAM3 segment (bg → gray 127)                               │──► null_per_task/<T>.npz
                │  3. Cycle null   (462 multi-lag pairs, lags [1,2,5,10])        │     - null_cycle_{mean,peak}
                │  4. DINOv2 pool  (120 × 384, L2-norm)                          │     - null_knn_{ivar,peak}
                │  5. kNN LOO null + CV routing (ivar vs peak)                   │     - route, H_train_*
                │  6. H_train baseline (run online on training itself)           │
                └─────────────────────────────────────────────────────────────────┘
                                              │
                ┌───────────────── ONLINE (per query, ~30 sec) ─────────────────┐
query.mp4   ──► │  1. Sample 10 frames + SAM3                                    │
                │  2. CYCLE: 9 pairs → Cauchy(p_mean,p_peak) → p80 → H_cycle    │
                │  3. KNN:   10 frames → Cochran D → empirical_p → p80 → H_knn │
                │  4. FUSE:  Cauchy([1-H_cycle, 1-H_knn]) → H_fused             │ ──► ratio + verdict
                │  5. ratio_fused = H_fused / H_train_fused                     │
                │  6. HALLU if ratio_fused > 1.0                                │
                └─────────────────────────────────────────────────────────────────┘
```

Ý tưởng: cycle bắt **temporal inconsistency** (forward-backward không khớp giữa các frame), kNN bắt **appearance drift** (geometry mismatch so với reference pose gần nhất trong training). Fuse 2 p-value qua Cauchy → kết hợp 2 nguồn evidence độc lập, giữ tail behavior.

---

## OFFLINE — build null per task

Chạy 1 lần mỗi task T (Pick-Place-Bowl, Pour, …), output là `null_per_task/<T>.npz`.

### Step 1 — Sample 120 reference frames

Intuition: training video ~120 frames, lấy đều để cover toàn bộ trajectory pose.

```python
N = total_frames(training_mp4)
idx = np.linspace(0, N - 1, 120).astype(int)
refs = [read_frame(training_mp4, i) for i in idx]
```

Sample đều (không random) để null phản ánh distribution của task, reproducible giữa các run.

### Step 2 — SAM3 segment, bg → gray

Intuition: chỉ giữ robot foreground; xóa background nhiễu (lighting, table texture) khỏi RoMa matching.

```python
seg = VideoFrameSegmenter(prompts=["robot arm", "robotic hand",
                                   "gripper", "mechanical finger"])
refs_masked = [seg.segment(f, bg_fill=(127, 127, 127)) for f in refs]
```

Gray fill (127,127,127) là neutral cho DINOv2 & RoMa — không tạo edge giả ngoài silhouette robot.

### Step 3 — Cycle null (multi-lag pairs)

Intuition: build null cho `CycleSignal` bằng cách pair các ref **trong cùng training** ở nhiều lag → đại diện "what normal cycle error looks like".

```python
pairs = []
for lag in [1, 2, 5, 10]:
    for i in range(120 - lag):
        pairs.append((i, i + lag))  # ~462 pairs total

null_mean, null_peak = [], []
for i, j in pairs:
    fwd, bwd = roma.match_bidir(refs_masked[i], refs_masked[j])
    s = CycleSignal.compute(fwd, bwd, cert_floor=0.1, peak_pct=99)
    null_mean.append(s.mean); null_peak.append(s.peak)
```

Multi-lag → null cover cả short-range (lag 1) và long-range drift (lag 10), không bias về 1 scale.

### Step 4 — DINOv2 pool (120 × 384)

Intuition: build pool feature để online query có thể tìm "frame nào trong training giống tôi nhất".

```python
dino = DinoFeatureExtractor("dinov2_vits14")  # ViT-S/14, CLS token
pool = np.stack([dino.extract(f) for f in refs_masked])  # (120, 384)
pool = pool / np.linalg.norm(pool, axis=1, keepdims=True)  # L2-norm
```

CLS feature đủ discriminative cho pose-level retrieval, rẻ hơn patch features ~50×.

### Step 5 — kNN LOO null + CV routing

Intuition: với mỗi ref i, giả vờ nó là query → tìm top-15 nearest từ pool (loại i) → compute Cochran deviance → đây là "null khi query thực sự là in-distribution".

```python
null_ivar, null_peak_knn = [], []
for i in range(120):
    sims = pool @ pool[i]; sims[i] = -1
    top_k = np.argsort(sims)[-15:]
    matches = roma.match_batch(refs_masked[i], [refs_masked[j] for j in top_k])
    ivar, peak = MahalanobisStatistics.cochran_deviance(matches)
    null_ivar.append(ivar); null_peak_knn.append(peak)

cv = np.std(null_ivar) / np.mean(null_ivar)
route = "peak" if cv < 0.50 else "ivar"
```

CV routing: nếu `ivar` quá flat (CV thấp) thì nó không discriminative cho task này → switch sang `peak`. Decision per-task, không global.

### Step 6 — H_train baseline

Intuition: chạy đúng pipeline ONLINE trên chính training video → có baseline H để chia ratio. Nếu không có baseline thì H absolute không có ý nghĩa cross-task.

```python
H_train_cycle = run_online_cycle(training_mp4, null_cycle_*)
H_train_knn   = run_online_knn(training_mp4, pool, null_knn_*, route)
p_train       = cauchy_combine([1 - H_train_cycle, 1 - H_train_knn])
H_train_fused = 1 - p_train

np.savez(f"null_per_task/{T}.npz",
         null_cycle_mean=..., null_cycle_peak=...,
         null_knn_ivar=..., null_knn_peak=...,
         route=route, pool=pool,
         H_train_cycle=H_train_cycle, H_train_knn=H_train_knn,
         H_train_fused=H_train_fused)
```

Self-normalization: H_train absorbs task-specific difficulty (Pour khó hơn Pick → H_train cao hơn) nên ratio luôn so sánh tương đối.

---

## ONLINE — score 1 query video

Load `null_per_task/<T>.npz`, chạy ~30s trên H100.

### Step 1 — Sample 10 frames + SAM3

Intuition: 10 frame đủ để cover query video 120 frame, mỗi frame distance ~12 → dày hơn pair lag 10.

```python
idx_q = np.linspace(0, len(query) - 1, 10).astype(int)
q_frames = [seg.segment(query[i], bg_fill=(127,127,127)) for i in idx_q]
```

Cùng SAM3 prompts như offline để mask distribution match.

### Step 2 — CYCLE branch (9 pairs → H_cycle)

Intuition: 9 consecutive pairs trên 10 frame, mỗi pair cho 2 signal (mean, peak), fuse qua Cauchy thành H_pair, rồi aggregate p80 qua 9 pairs.

```python
H_pairs = []
for k in range(9):
    fwd, bwd = roma.match_bidir(q_frames[k], q_frames[k+1])
    s = CycleSignal.compute(fwd, bwd, cert_floor=0.1, peak_pct=99)
    p_mean = empirical_p(s.mean, null_cycle_mean)   # right-tail
    p_peak = empirical_p(s.peak, null_cycle_peak)
    p_pair = cauchy_combine([p_mean, p_peak])
    H_pairs.append(1 - p_pair)
H_cycle = np.percentile(H_pairs, 80)
```

p80 thay vì max → robust với 1–2 pair noise, vẫn nhạy với hallu kéo dài ≥ 2 pairs.

### Step 3 — KNN branch (10 frames → H_knn)

Intuition: với mỗi query frame, tìm 15 nearest ref pose trong pool → batch-match RoMa → Cochran D → so với null routed signal.

```python
H_frames = []
for i in range(10):
    feat = dino.extract(q_frames[i])
    sims = pool @ feat
    top_k = np.argsort(sims)[-15:]
    matches = roma.match_batch(q_frames[i], [refs_masked[j] for j in top_k])
    ivar, peak = MahalanobisStatistics.cochran_deviance(matches)
    sig  = peak if route == "peak" else ivar
    null = null_knn_peak if route == "peak" else null_knn_ivar
    p_f  = empirical_p(sig, null)
    H_frames.append(1 - p_f)
H_knn = np.percentile(H_frames, 80)
```

`match_batch` chạy 1 forward pass cho 15 refs → ~2.5× nhanh hơn loop (xem commit a18b03b).

### Step 4 — Fuse Cauchy

Intuition: cycle và kNN là 2 evidence độc lập (temporal vs appearance). Cauchy combine = heavy-tail aggregator, không assume independence như Fisher.

```python
p_cycle = 1 - H_cycle
p_knn   = 1 - H_knn
p_fused = cauchy_combine([p_cycle, p_knn])
H_fused = 1 - p_fused
```

Cauchy rất sensitive khi **một** branch nhỏ (p tiến về 0) → bắt cả hallu chỉ-temporal lẫn chỉ-appearance.

### Step 5 — Ratio vs H_train

Intuition: H_fused absolute không cross-task được; chia cho baseline self-test → unit-free score.

```python
ratio_fused = H_fused / H_train_fused
```

Ví dụ Pour task: `H_train_fused = 0.42`, query `H_fused = 0.51` → `ratio = 1.21` → flag.

### Step 6 — Verdict

```python
if ratio_fused > 1.05:     verdict = "HALLU"
elif ratio_fused > 0.95:   verdict = "borderline"
else:                       verdict = "clean"
```

Band 0.95–1.05 = uncertainty zone (sampling noise); ngoài band là decision dứt khoát.

---

## Parameters

| Group | Param | Value |
|---|---|---|
| Refs | `N_REFERENCE_FRAMES` | 120 (max from 120-frame training videos) |
| Refs | `SAM3_PROMPTS` | robot arm, robotic hand, gripper, mechanical finger |
| Cycle | `NULL_LAGS` | [1, 2, 5, 10] |
| Cycle | `CERT_FLOOR` | 0.1 |
| Cycle | `PEAK_PERCENTILE` | 99 |
| kNN | `KNN_K` | 15 |
| kNN | `DINO_MODEL` | dinov2_vits14 |
| kNN | `CV_THRESHOLD` | 0.50 |
| RoMa | `SETTING` | turbo |
| RoMa | `VIS_SIZE` | 224 |
| Online | `N_QUERY_FRAMES` | 10 |
| Online | `VIDEO_AGGREGATOR` | p80 |
| Decision | `RATIO_THRESHOLD` | 1.0 |
| Decision | `BORDERLINE_BAND` | 0.95–1.05 |

---

## Success metrics (Tier 1–4)

| Tier | Metric | Target |
|---|---|---|
| 1 | FPR=0% on real (ratio_fused ≤ 1.0 cho tất cả real) | **REQUIRED** |
| 1 | Cycle-only `--no_knn` reproduces 18/24 catch | **REQUIRED** |
| 2 | Complementarity: ≥ 3 cycle-only catches + ≥ 3 knn-only catches | TARGET |
| 2 | Separation gap (min hallu − max real) > cycle-only gap | TARGET |
| 2 | Borderline count (0.95–1.05) < cycle-only count | TARGET |
| 3 | AUROC video-level > 0.81 (cycle alone: 0.77) | INFO |
| 4 | Manual spot-check 3 hallu + 3 clean gens | OPTIONAL |

---

## Reference results (GR-1, 5 tasks, fusion eval)

| Metric | Cycle | kNN | **Fused** |
|---|---|---|---|
| Gen catch (ratio > 1.0) | 18/24 | 18/24 | **19/24** |
| AUROC | 0.7750 | 0.7833 | **0.8000** |
| Separation gap | +0.014 | +0.002 | **+0.031** |
| Real flagged | 0/5 | 0/5 | **0/5** |
| Complementarity | — | — | 14 both / 4 cycle-only / 4 knn-only / 2 neither |

Fusion vừa tăng catch (+1) vừa double separation gap (+0.014 → +0.031) mà không sinh false positive trên real.

---

## Files

| File | Purpose |
|---|---|
| `warp_score/temporal_signals.py` | `CycleSignal` class, `empirical_p_value`, `cauchy_combine` |
| `warp_score/matcher.py` | `RoMaMatcher` (with `match_batch` for k-ref scoring) |
| `warp_score/sam_segmenter.py` | `VideoFrameSegmenter` (SAM3) |
| `warp_score/adaptive_refs.py` | `DinoFeatureExtractor`, `AdaptiveRefSelector` (k-NN) |
| `warp_score/statistics.py` | `MahalanobisStatistics` (Cochran deviance) |
| `warp_score/knn_signal.py` | `KNNFrameSignal` (pool + LOO + routing + `score_frame`) |
| `warp_score/fusion.py` | `cauchy_combine_video` + `complementarity_report` |
| `scripts/extract_refs_dense.py` | OFFLINE step 1–2: dense SAM3 refs |
| `scripts/eval_per_task_dense_null.py` | OFFLINE step 3–6 + ONLINE eval |
| `scripts/compute_per_task_ratio.py` | Ratio table + complementarity + markdown |

---

## Commands cheatsheet

```bash
conda activate groot
cd /mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination

# Step 1: extract dense SAM3 refs (~30 min for 5 tasks)
python scripts/extract_refs_dense.py

# Step 2: full fusion eval (~95 min for 5 tasks)
python scripts/eval_per_task_dense_null.py

# Cycle-only regression mode (Tier 1 sanity check)
python scripts/eval_per_task_dense_null.py --no_knn

# Step 3: per-task ratio + ranking + markdown report
python scripts/compute_per_task_ratio.py
```
