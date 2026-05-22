# WarpDyn — Robot Video Hallucination Detection

Feature-matching anomaly detector cho video sinh ra bởi world-model (Cosmos, OpenSora, …)
so với training distribution của robot. Không cần labels; chỉ cần 1 training video per task.

**Method docs (chi tiết):** [WARPDYN_METHOD.md](WARPDYN_METHOD.md)

---

## How to run

```bash
conda activate groot
cd /path/to/feature_matching_eval_hallucination

# 1. Extract dense SAM3 reference frames (~30 min, 5 tasks)
python scripts/extract_refs_dense.py

# 2. Build null + run fusion eval (~95 min, 5 tasks)
python scripts/eval_per_task_dense_null.py

# 2b. Cycle-only mode (sanity check / regression)
python scripts/eval_per_task_dense_null.py --no_knn

# 3. Compute per-task ratio table + markdown report
python scripts/compute_per_task_ratio.py

# Benchmark 1 task with full timing + visualizations
python scripts/benchmark_one_task.py [--task <task_name>] [--query <video.mp4>]
```

Outputs land in `null_per_task/<task>.npz` (offline) and `per_task_dense_table.csv` (eval).

---

## Pipeline

```
                ┌─────────────── OFFLINE (per task, ~30 min) ───────────────┐
training.mp4 ──►│ 1. Sample 120 frames evenly                                │
                │ 2. SAM3 segment → background → gray (127,127,127)          │──► null_per_task/<T>.npz
                │ 3. Cycle null   (462 multi-lag pairs, lags 1,2,5,10)       │
                │ 4. DINOv2 pool  (120 × 384-d CLS, L2-norm)                 │
                │ 5. k-NN LOO null + CV-routing (ivar vs peak)               │
                │ 6. H_train baseline (run ONLINE on training video itself)  │
                └────────────────────────────────────────────────────────────┘

                ┌─────────────── ONLINE (per query, ~30 sec) ───────────────┐
query.mp4   ──► │ 1. Sample 10 frames + SAM3                                 │
                │ 2. CYCLE: 9 consecutive pairs → Cauchy(p_mean,p_peak)      │
                │          → p80 across pairs → H_cycle                      │
                │ 3. KNN:   per-frame top-15 DINOv2 refs → RoMa batch       │
                │          → Cochran D → routed p → p80 across frames        │ ──► ratio + verdict
                │          → H_knn                                           │
                │ 4. FUSE:  Cauchy([1−H_cycle, 1−H_knn]) → H_fused          │
                │ 5. ratio  = H_fused / H_train_fused                        │
                │ 6. HALLU if ratio > 1.0  (borderline: 0.95–1.05)          │
                └────────────────────────────────────────────────────────────┘
```

**Core idea:** two independent signals fused via Cauchy (ACAT).
- **Cycle** catches temporal self-inconsistency (fwd-bwd warp drift).
- **kNN** catches appearance drift vs nearest training pose.
- **Ratio** normalizes absolute H per-task so the threshold (1.0) is universal.

---

## Algorithm details

### Cycle signal

RoMa forward + backward match between consecutive frame pairs.
Cycle error = `||fwd(p) + bwd(fwd(p)) - p||` weighted by `cert_fwd × cert_bwd`.
Two summary stats: `mean` (interior FG mean) and `peak` (99th percentile).

Multi-lag null (lags 1,2,5,10 across 120 refs → 462 pairs) ensures the null
covers both short-range and long-range drift, not biased to one temporal scale.

Per pair: `p_mean, p_peak` via right-tail empirical CDF vs null, then
`cauchy_combine([p_mean, p_peak])` → `H_pair`. Video-level: `p80` across 9 pairs.

### kNN signal (DINOv2 Cochran deviance)

For each frame, DINOv2 ViT-S/14 CLS feature → cosine-sim to pool → top 15 refs.
RoMa `match_batch(query, refs_15)` → 15 warp fields + precision matrices.

Cochran deviance D(p) = per-pixel weighted variance across the 15 warp fields
(via Mahalanobis with per-match precision). Summarized as:
- `ivar_maha` — FG interior mean of D(p)
- `peak_maha` — FG max z-score of D(p)

**CV-routing (per task offline):** if null ivar has CV < 0.50 (too flat, not
discriminative), route to `peak`; else use `ivar`. Decision is task-specific
from LOO calibration.

LOO null: for each ref_i, score it as a pseudo-query vs top-k of remaining
refs → same df, statistically consistent.

### Fusion

```
p_fused = cauchy_combine([1 − H_cycle, 1 − H_knn])
H_fused = 1 − p_fused
ratio   = H_fused / H_train_fused   # H_train built offline on training video
```

Cauchy is tail-heavy → fires if *either* branch detects anomaly strongly,
without requiring joint signal (unlike Fisher).

---

## Assumptions

| Assumption | Where it matters |
|---|---|
| 1 training video per task (≥120 frames) | Null quality; fewer frames → unstable null |
| SAM3 can segment robot foreground | Masking quality for both signals |
| Query is from the same task/robot | Ratio is task-relative; cross-task ratio invalid |
| CUDA GPU available | RoMa turbo ~30s online is GPU-measured |
| DINOv2 ViT-S/14 accessible via torch.hub | Auto-downloaded on first run |
| Training and query videos ≥ 10 frames | p80 aggregator needs ≥ 5 valid samples |

---

## Key parameters

| Group | Param | Value |
|---|---|---|
| Refs | N_REFERENCE_FRAMES | 120 |
| Refs | SAM3_PROMPTS | robot arm, robotic hand, gripper, mechanical finger |
| Cycle | NULL_LAGS | [1, 2, 5, 10] |
| Cycle | CERT_FLOOR | 0.1 |
| Cycle | PEAK_PERCENTILE | 99 |
| kNN | K | 15 |
| kNN | DINO_MODEL | dinov2_vits14 |
| kNN | CV_THRESHOLD | 0.50 |
| RoMa | SETTING | turbo, VIS_SIZE=224 |
| Online | N_QUERY_FRAMES | 10 |
| Online | VIDEO_AGGREGATOR | p80 |
| Decision | RATIO_THRESHOLD | 1.0 (borderline band: 0.95–1.05) |

---

## Results (GR-1, 5 tasks)

| Metric | Cycle-only | kNN-only | **Fused** |
|---|---|---|---|
| Gen catch (ratio > 1.0) | 18/24 | 18/24 | **19/24** |
| AUROC | 0.7750 | 0.7833 | **0.8000** |
| Separation gap | +0.014 | +0.002 | **+0.031** |
| Real flagged (FPR) | 0/5 | 0/5 | **0/5** |
| Complementarity | — | — | 14 both / 4 cycle-only / 4 knn-only / 2 neither |

---

## File map

| File | Purpose |
|---|---|
| `warp_score/temporal_signals.py` | `CycleSignal`, `empirical_p_value`, `cauchy_combine` |
| `warp_score/matcher.py` | `RoMaMatcher` (with `match_batch`) |
| `warp_score/sam_segmenter.py` | `VideoFrameSegmenter` (SAM3) |
| `warp_score/adaptive_refs.py` | `DinoFeatureExtractor`, `AdaptiveRefSelector` |
| `warp_score/statistics.py` | `MahalanobisStatistics` (Cochran deviance) |
| `warp_score/knn_signal.py` | `KNNFrameSignal` — pool + LOO + route + score |
| `warp_score/fusion.py` | `cauchy_combine_video`, `complementarity_report` |
| `scripts/extract_refs_dense.py` | OFFLINE step 1–2: sample + SAM3 |
| `scripts/eval_per_task_dense_null.py` | OFFLINE step 3–6 + ONLINE eval |
| `scripts/compute_per_task_ratio.py` | Ratio table + complementarity + report |
| `scripts/benchmark_one_task.py` | End-to-end timing + visualizations for 1 task |
| `WARPDYN_METHOD.md` | Full method documentation (step-by-step with code) |
| `legacy/` | Old pipeline (coreset, graph, stage1–4) — superseded |
| `scripts/_archive_exploration/` | Exploration scripts (k-NN variants, ALN, calibration) |
| `_archive_docs/` | Earlier method writeups |
