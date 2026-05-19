# WarpDyn — Robust anomaly detection on robot videos via pure feature matching

End-to-end pipeline để phát hiện video **bất thường** so với robot training distribution. Output là **continuous anomaly score [0, 1]** — không cần optical flow model, chỉ dùng RoMa dense warp + DINOv2 features.

## ⚠ Framing đúng (REVISED)

**Generated videos KHÔNG phải = hallucinated.** Cosmos / generator chỉ sinh video, có thể plausible hoặc không. Việc dùng "gen = positive label" là sai về phương pháp luận.

**Mục tiêu thực sự:** Real training videos phải **KHÔNG bị flag là anomaly** (FPR thấp).

→ Calibrate threshold **chỉ từ real training data** (conformal style).
→ Score continuous của gen video là **informative**, không phải metric đánh giá.

**Production metric (GR1, 30 real videos production-realistic):**
- Real H_peak distribution: range [0.31, 0.96], mean **0.69**, p95 = **0.93**
- Threshold = max(real) + epsilon = **0.959** → **FPR = 0%** trên real training
- Tại threshold đó: 2/24 gen videos được flag là "anomalous so với real distribution" (informative)

---

## 1. Tổng quan kiến trúc

```
                    ┌─── ReferenceCache (build offline, 1 lần) ───┐
                    │                                              │
                    │   pool_feats.npy       (DINOv2 embeddings)   │
                    │   pool_paths.txt       (200 SAM3 ref frames) │
                    │   cycle_null.npz       (1472 null pairs)     │
                    │   nn_jaccard_null.npy  (1472 null pairs)     │
                    │   threshold.json       (Youden's J = 0.924)  │
                    └──────────────────────────────────────────────┘
                                       │
                                       ▼
                              ┌────────────────┐
   query.mp4 ──── SAM3 ────►  │  VideoScorer   │ ────► H_score + decision
                              │  (10 frames)   │
                              └────────────────┘
                                       │
                              4 signals per frame:
                              ─────────────────────
                              S1  cross-pool k-NN RoMa warp residual
                              S2  fwd/bwd cycle composition drift   (cert-weighted)
                              S4  DINOv2 NN-set Jaccard distance
                              ─────────────────────
                              Cauchy fuse → H_frame
                              80%-percentile peak across frames → H_video
                              H_video > threshold → HALLUCINATION
```

**Không cần task label cho query.** k-NN tự tìm 50 ref phù hợp nhất từ pool cross-task.

---

## 2. Offline setup — calibrate 1 lần, dùng mãi

### Bước A: chuẩn bị reference frames

Cần dataset video robot **thật** (training distribution của model generator). Mỗi video phải:

- Resolution ổn định (224×224 hoặc bội số)
- **SAM3-segment background** thành xám `(127, 127, 127)` để loại bias background
- Lưu mỗi frame một file PNG dưới `<root>/<task_name>/<frame_NNNN>.png`

Repo có sẵn dataset GR1: `paper-physical-gr1/reference/` (92 tasks × 50 frames). Để dùng dataset khác:

```bash
# Pseudo-code: extract + SAM3 segment your videos
for video in your_dataset/*.mp4:
    extract 50 frames at np.linspace(0, total-1, 50)
    for each frame: SAM3 segment (text prompts: ["robot arm", "robotic hand", "gripper"])
    save to ref_root/<task_name>/frame_NNNN.png
```

Tham khảo `scripts/postprocess_gr1_generated.py` cho luồng chuẩn.

### Bước B: build reference pool

Pool = symlink tất cả ref frames vào một thư mục `POOL` duy nhất với prefix task slug. Đây là cái fix vấn đề "Cosmos conditioning trên same-task ref" — query phải compete cross-task.

Chỉnh `scripts/run_gr1_pool_benchmark.py` đổi `TASKS` list, rồi:

```bash
conda activate groot
python scripts/run_gr1_pool_benchmark.py --k 50
# → paper-physical-gr1/pool/reference/POOL/  (200-500 ref symlinks)
```

Pool size khuyến nghị: **200-500 frames cross-task**.
- Quá nhỏ (<100): k-NN không có lựa chọn → false positives
- Quá lớn (>2000): calibration LOO chậm; không cải thiện đáng kể

### Bước C: build cycle null + jaccard null

Null = empirical distribution của signal trên các cặp frame **thật**.

```bash
# Cycle null trên 92 tasks × 16 pairs/task = 1472 samples (~3 min)
python scripts/expand_cycle_null.py --pairs_per_task 16

# Build NN-Jaccard null đồng thời + recompute query signals (~20 min)
python scripts/run_warpdyn_v2.py
# Outputs:
#   paper-physical-gr1/pool/results_warpdyn_v2/cycle_null_v2.npz
#   paper-physical-gr1/pool/results_warpdyn_v2/nn_jaccard_null.npy
#   paper-physical-gr1/pool/results_warpdyn_v2/raw_signals_v2.csv
#   paper-physical-gr1/pool/results_warpdyn_v2/final_report.json
```

**Sampling guidance:**
- **`pairs_per_task = 16`** cho ~1500 null pairs — đủ robust cho empirical p-value tail ở p99
- Nếu < 50 task: tăng `pairs_per_task` lên 25-30
- Nếu pool refs có > 100 frames mỗi task: tăng lên 30-40

### Bước D: compile cache + calibrate threshold

```bash
# Compile mọi thứ vào portable cache
python scripts/build_reference_cache.py
# → paper-physical-gr1/ref_cache/  (~5 MB)
```

**Chọn 1 trong 2 cách calibrate threshold:**

#### Cách 1 — Youden's J (cần labeled batch: cả real + generated)

```bash
python scripts/calibrate_threshold.py
# → threshold.json  (mode="youden_j", threshold=0.924 cho GR1)
```

Tối ưu (TPR - FPR). Cần đã có ≥ 5 real + ≥ 10 gen videos đã gắn nhãn.

#### Cách 2 — Conformal (CHỈ cần real training videos, không cần gen)

```bash
python scripts/calibrate_threshold_unlabeled.py \
    --frame_dirs paper-physical-gr1/reference/ \
    --target_fpr 0.05 \
    --max_videos 50
# → threshold.json (mode="conformal_unlabeled", FPR ≤ 5% guarantee)
```

Statistical guarantee: **FPR ≤ target_fpr** by construction. Không quan tâm phân bố của gen. Phù hợp khi:
- Chưa có generated video nào để label
- Cần work cho generator mới chưa thấy (model robustness)
- Muốn conservative threshold (low false alarm)

Trade-off: không biết TPR — nếu real/gen overlap quá nhiều thì TPR có thể thấp. Nhưng FPR control chặt.

**Cả 2 đều ghi vào cùng `threshold.json` trong cache** — VideoScorer auto-load.

Sau bước này, **ref_cache là deployable**. Copy folder này đến machine production là dùng được.

---

## 3. Inference — chấm video mới

### Cách 1: Python API

```python
from warp_score.video_scorer import VideoScorer

scorer = VideoScorer.from_cache("paper-physical-gr1/ref_cache")
# threshold tự load từ threshold.json trong cache

result = scorer.score("path/to/my_video.mp4", n_frames=10)

print(result.video_h_score)        # robust trimmed mean (0..1)
print(result.video_h_peak)         # 80%-percentile peak (decision metric)
print(result.is_hallucination)     # True / False
print(result.aggregate_breakdown)  # mean, median, p80, max...
for f in result.per_frame:
    print(f.frame_idx, f.h_score, f.p_cycle, f.p_jaccard)
```

### Cách 2: CLI

```bash
python scripts/score_video.py my_video.mp4 \
    --cache_dir paper-physical-gr1/ref_cache \
    --n_frames 10 \
    --out result.json
```

Output stdout:

```
=== Video: my_video.mp4 ===
  Frames processed:   10
  H_score (robust):   0.7919
  H_score (p80 peak): 0.8514
  Decision (>0.924):  CLEAN

  Aggregator breakdown:
    mean           0.7919
    trimmed_mean   0.7919
    median         0.8053
    p80            0.8514
    max            0.9720
```

### Có cần biết task của video không?

**KHÔNG.** DINOv2 k-NN tự tìm 50 reference frames phù hợp nhất trong pool cross-task.

- Pool hiện tại có 92 tasks (200 frames). Nếu video query thuộc 1 trong 92 tasks → k-NN ưu tiên pick frames same-task.
- Nếu video query là task **mới** (không có trong pool) → k-NN pick frames có visual similarity cao (similar robot, similar workspace, similar object category).
- Generated video Cosmos thường được conditioning trên same-task ref, nhưng pool cross-task buộc nó match với frames không liên quan → expose artifacts.

**Khi nào cần biết task:**
- Để **kiểm tra chất lượng pool**: nếu task quá khác (ví dụ outdoor video vs indoor robot pool) → k-NN không tìm được match tốt → H_score cao cho mọi frame → false positive
- Best practice: **expand pool** thay vì gắn task label

---

## 4. Sampling — chọn bao nhiêu frame mỗi video?

| `n_frames` | Use case | Compute / video |
|---|---|---|
| 5 | Quick triage | ~3 sec |
| **10** (default) | **Production** | **~6 sec** |
| 20 | High-stakes verification | ~12 sec |
| 30+ | Diminishing returns | > 18 sec |

Lý do `n_frames = 10`:
- 5 cặp consecutive = 5 cycle + 5 jaccard signals
- 80%-percentile peak có ý nghĩa (cần ≥ 5 mẫu)
- Đủ để cover 5 sec video ở 2 fps

**Sampling pattern:** `np.linspace(0, total_frames-1, n_frames)` — trải đều theo thời gian. Cosmos thường có artifact ở các frame giữa video (autoregressive drift), uniform sampling sẽ chạm vào.

---

## 5. Hyperparameters chính

| Param | Mặc định | Khi nào chỉnh |
|---|---|---|
| `n_frames` | 10 | Tăng nếu video dài > 10 sec |
| `k_per_frame` | 50 | Tăng nếu pool > 500 (k=100), giảm nếu pool < 100 (k=20) |
| `cert_floor` | 0.1 | Tăng → 0.2 nếu nhiều scene uniform-texture |
| `threshold` | 0.924 (auto) | Auto-load từ `threshold.json`. Tăng nếu cần FPR thấp hơn nữa |
| `sam_segment` | True | False **chỉ khi** frames đã SAM3-segmented từ trước |

---

## 6. Cấu trúc reference cache

Một ref_cache portable hoàn chỉnh:

```
paper-physical-gr1/ref_cache/
├── pool_feats.npy        # (200, 384) float32 — DINOv2 ViT-S/14 L2-normalized
├── pool_paths.txt        # 200 absolute paths to ref PNG files
├── cycle_null.npz        # cycle_mean[1472], cycle_peak[1472] sorted
├── nn_jaccard_null.npy   # (1472,) sorted jaccard distances
└── threshold.json        # {threshold, config, aggregator, tpr/fpr metadata}
```

**Lưu ý:** `pool_paths.txt` chứa **absolute paths**. Khi copy cache sang machine khác, phải:
- Copy luôn thư mục ref frames giữ đúng cấu trúc, hoặc
- Symlink lại `pool_paths.txt` về path mới

Để tránh issue path tuyệt đối, có thể serialize ref frames trực tiếp vào cache (chưa implement, dễ thêm).

---

## 7. Khi nào WarpDyn fail (limitations)

| Failure mode | Lý do | Fix |
|---|---|---|
| Texture-uniform object (Rubik cube) | RoMa cert thấp → cycle noise cao cả real lẫn gen | Tăng `cert_floor` lên 0.2 |
| Video < 2 fps | Quá ít cặp consecutive | n_frames ≥ 8 |
| Background completely khác pool | DINOv2 NN không tìm được match | Expand pool |
| Static video (no robot motion) | Cycle = 0 cho cả real | S4 jaccard vẫn work; S2 sẽ noisy |
| Generator quá perfect (chưa từng thấy) | S2/S4 fail; chỉ S1 còn | Expand training distribution của null |

---

## 8. Quy trình debug khi H_score sai

1. **Check SAM3** — đầu vào sau segment có giống style pool không (bg = (127,127,127))?
   ```python
   scorer.score(video, n_frames=5)  # check intermediate tmp/ dir
   ```

2. **Check k-NN refs** — pool top-K có visually similar không?
   ```python
   q_feat = dino.extract([frame])[0]
   sims = scorer.cache.pool_feats @ q_feat
   top = np.argpartition(sims, -10)[-10:]
   for i in top:
       print(sims[i], scorer.cache.pool_paths[i])
   ```

3. **Check raw signals** — `result.per_frame[i].p_cycle` và `p_jaccard` có quá gần 0 (extreme tail) không?

4. **Visualize heatmap** — dùng `warp_score/visualizer.py` để xem region nào high cycle drift.

---

## 9. Roadmap nâng cấp robust hơn

| Cải thiện | Effort | Expected gain |
|---|---|---|
| Lag-matched null (rebuild ở lag 10 thay vì lag 1) | 0.5 ngày | +0.05 AUROC |
| Larger eval set (10 real + 50 gen) | 1 ngày | Narrow CI ±0.15 → ±0.07 |
| Cross-generator test (OpenSora, CogVideo) | 1 ngày | Verify generalization |
| Per-pixel cycle p-value heatmap → fuse | 0.5 ngày | Better viz, possibly +0.03 |
| Lag-normalized cycle (divide by Δt) | 0.5 ngày | Less sensitive to sampling rate |
| Augment pool to 5000 frames (download more GR1 ep.) | 1 ngày + 4h GPU | Tighter NN, +0.05 |

---

## 10. Tóm tắt commands

```bash
# === ONE-TIME SETUP (~30 min total) ===
conda activate groot

# Build pool layout
python scripts/run_gr1_pool_benchmark.py --k 50

# Build all nulls + signals + per-task report
python scripts/run_warpdyn_v2.py

# Compile cache + calibrate threshold
python scripts/build_reference_cache.py
python scripts/calibrate_threshold.py

# === INFERENCE (~6 sec per video) ===
python scripts/score_video.py my_video.mp4 --n_frames 10

# === PYTHON ===
from warp_score.video_scorer import VideoScorer
scorer = VideoScorer.from_cache("paper-physical-gr1/ref_cache")
result = scorer.score("my_video.mp4", n_frames=10)
print(result.is_hallucination, result.video_h_peak)
```

---

## 11. Files được dùng / sửa

```
warp_score/
├── temporal_signals.py    # cycle_signal + trajectory_accel + cert-weight
├── nn_consistency.py      # S4 NN-set jaccard
├── video_scorer.py        # VideoScorer + ReferenceCache (production API)
├── adaptive_refs.py       # DINOv2 k-NN selector (existing)
├── matcher.py             # RoMa wrapper (existing)
└── sam_segmenter.py       # SAM3 segmenter (existing)

scripts/
├── run_gr1_pool_benchmark.py    # build cross-task pool layout
├── run_warpdyn_v2.py             # full benchmark + null builder
├── expand_cycle_null.py          # standalone null builder
├── eval_warpdyn_robust.py        # eval with bootstrap CI
├── calibrate_threshold.py        # Youden's J threshold
├── build_reference_cache.py      # compile portable cache
└── score_video.py                # production CLI
```
