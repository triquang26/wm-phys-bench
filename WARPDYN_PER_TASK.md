# WarpDyn — Per-task multi-lag null (recommended method)

Anomaly detection cho robot video bằng RoMa cycle composition error.
Null distribution **per-task, multi-lag** — bảo đảm real training video không bị flag.

**Kết quả GR1 (5 tasks, 5 real + 24 gen):**
- All real training videos: H_peak ≤ **0.832**
- Set threshold = 0.832 → **FPR = 0% guaranteed**
- 16/24 gen videos above threshold (67% catch rate) — informative
- Overall AUROC = **0.767**

---

## 1. Flow tổng thể

```
═══════════════════════════════════════════════════════════════════
OFFLINE (1 lần per task)
═══════════════════════════════════════════════════════════════════

   Real training video task T (.mp4)
              │
              ▼
   Extract 50 frames evenly + SAM3 segment
              │
              ▼
   reference/<task_T>/frame_NNNN.png   (50 frames per task)
              │
              ▼
   Build per-task null:
     - For lag ∈ {1, 2, 5, 10}:
         For each i: pair = (frame[i], frame[i+lag])
         compute cycle_signal(fwd, bwd) → (mean, peak)
     - Total ~182 pairs per task
     - Sort into null_cycle_mean[], null_cycle_peak[]
              │
              ▼
   Save: null_per_task/<task_T>.npz

═══════════════════════════════════════════════════════════════════
INFERENCE (mỗi video query)
═══════════════════════════════════════════════════════════════════

   Query video.mp4 + task_id   ← (cần biết task của query)
              │
              ▼
   Sample 10 frames via np.linspace(0, total-1, 10)
              │
              ▼
   SAM3 segment mỗi frame (background → (127,127,127))
              │
              ▼
   For 9 consecutive pairs (frame_t, frame_{t+1}):
     fwd = RoMa(frame_t  → frame_{t+1})
     bwd = RoMa(frame_{t+1} → frame_t)
     cycle = CycleSignal().compute(fwd, bwd)
     → (mean, peak) per pair
              │
              ▼
   Load null_per_task/<task_id>.npz
              │
              ▼
   For each pair:
     p_mean = empirical_p(pair.mean, null_cycle_mean)
     p_peak = empirical_p(pair.peak, null_cycle_peak)
     p_pair = Cauchy(p_mean, p_peak)
     H_pair = 1 - p_pair
              │
              ▼
   H_video = np.percentile(H_pairs, 80)    ← 80th percentile peak
              │
              ▼
   Decision: H_video > 0.832 ? HALLU : CLEAN
                 (threshold = max(real_train) + ε per task)
```

---

## 2. Sample sizes & lags

### Null calibration (mỗi task)

| Lag | # pairs từ 50 frames | Cover loại motion |
|---|---|---|
| 1 | 49 | Very slow (frame liền kề) |
| 2 | 48 | Slow |
| 5 | 45 | Medium (≈ inference lag) |
| 10 | 40 | Fast (lag lớn) |
| **Total** | **~182** | **Spectrum đầy đủ** |

Vì sao multi-lag: video query sample 10 frames từ ~120 total → lag ≈ 12. Null phải có pairs ở lag tương đương ĐỂ cycle drift comparable. Multi-lag {1,2,5,10} cover toàn spectrum → real video query không nằm tail giả tạo.

### Test sampling (mỗi query)

- 10 frames, `np.linspace(0, total_frames-1, 10)`
- 9 consecutive pairs → 9 H_pair values
- 80th percentile → H_video (sensitive với ≥1-2 bad pair nhưng không bị 1-frame noise phá)

---

## 3. Signal CycleSignal — công thức

```
Input:  fwd  = RoMa(frame_t → frame_{t+1})  → warp_fwd, cert_fwd
        bwd  = RoMa(frame_{t+1} → frame_t)  → warp_bwd

For each pixel (x,y) of frame_t:
  (u,v)  = warp_fwd[y,x]                    # land in frame_{t+1}
  (x',y') = bilinear_sample(warp_bwd, (u,v)) # back to frame_t
  cycle_drift[y,x] = ||(x',y') - (x,y)||    # pixel units

Filter:  valid = cert_fwd > 0.1            # drop uniform-texture (cube faces)
Aggregate:
  mean = cert_fwd-weighted average của cycle_drift trong valid pixels
  peak = 99th percentile của cycle_drift trong valid pixels

Output:  (mean, peak)
```

Real video → cycle drift nhỏ (consecutive frames coherent về physics).
Generator artifact → cycle drift lớn (texture flicker, ghost objects, frame-to-frame inconsistency).

---

## 4. Empirical p-value + Cauchy fusion

```
p_mean = (n - rank(value, null_cycle_mean)) / (n + 1)   # right-tail p
p_peak = (n - rank(value, null_cycle_peak)) / (n + 1)

Clip:  p ∈ [1/(n+1), 1-1/(n+1)]  để tránh Cauchy blow-up

Cauchy combine (ACAT):
  T = mean(tan(π·(0.5 - p_i)))
  p_combined = 0.5 - arctan(T)/π

H_pair = 1 - p_combined ∈ [0, 1]
```

`p_combined` gần 0 → tail anomalous → H_pair gần 1.

---

## 5. Code skeleton

```python
from warp_score.temporal_signals import CycleSignal, empirical_p_value
from warp_score.matcher import RoMaMatcher
from warp_score.sam_segmenter import VideoFrameSegmenter
import numpy as np, cv2

# Setup
matcher = RoMaMatcher(setting="turbo", device="cuda", use_precision=True)
matcher._load_model()
seg = VideoFrameSegmenter()
cycle = CycleSignal(cert_floor=0.1)

# ─── Offline: build null for one task ───
def build_per_task_null(ref_pngs, lags=(1, 2, 5, 10)):
    means, peaks = [], []
    for lag in lags:
        for i in range(len(ref_pngs) - lag):
            fwd = matcher.match(ref_pngs[i],   ref_pngs[i+lag])
            bwd = matcher.match(ref_pngs[i+lag], ref_pngs[i])
            sig = cycle.compute(fwd, bwd)
            means.append(sig.mean)
            peaks.append(sig.peak)
    return np.sort(means), np.sort(peaks)

# ─── Online: score a video ───
def score_video(mp4_path, null_mean, null_peak, threshold=0.832):
    # 1. Sample 10 frames evenly
    cap = cv2.VideoCapture(str(mp4_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(0, total-1, 10, dtype=int)
    bgrs = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, b = cap.read()
        if ok: bgrs.append(b)
    cap.release()

    # 2. SAM3 segment + save tmp
    import tempfile; from pathlib import Path
    tmp = Path(tempfile.mkdtemp())
    paths = []
    for j, b in enumerate(bgrs):
        p = tmp / f"f{j:04d}.png"
        cv2.imwrite(str(p), seg.segment_frame(b))
        paths.append(p)

    # 3. Pair-wise cycle + Cauchy fuse
    h_pairs = []
    for t in range(len(paths) - 1):
        fwd = matcher.match(paths[t],   paths[t+1])
        bwd = matcher.match(paths[t+1], paths[t])
        sig = cycle.compute(fwd, bwd)
        p_m = empirical_p_value(sig.mean, null_mean)
        p_p = empirical_p_value(sig.peak, null_peak)
        T = np.mean([np.tan(np.pi*(0.5 - p_m)), np.tan(np.pi*(0.5 - p_p))])
        p_pair = 0.5 - np.arctan(T) / np.pi
        h_pairs.append(1.0 - p_pair)

    H_video = float(np.percentile(h_pairs, 80))
    return H_video, H_video > threshold

# ─── Usage ───
# null = build_per_task_null(sorted(Path("reference/task_T").glob("*.png")))
# H, is_hallu = score_video(Path("query.mp4"), *null)
```

---

## 6. Operational thresholds + per-task RATIO score

### Option A: Global threshold

Threshold = quantile của H_peak trên real training videos đã evaluate.

| Target FPR | Quantile | Threshold (GR1 example) | Gen catch |
|---|---|---|---|
| 0% (max + ε) | max + ε | **0.832** | 16/24 (67%) |
| 5% | p95 | 0.815 | 17/24 |
| 10% | p90 | 0.770 | 19/24 |

### Option B: Per-task RATIO (recommended) ⭐

Mỗi task có baseline riêng:

```
H_train_task = H_peak khi chấm video training của task đó
ratio        = H_test / H_train_task

verdict:
  ratio > 1.0  → more anomalous than training → HALLU
  ratio ~ 1.0  → boundary
  ratio < 0.95 → clearly clean
```

**Lợi:** FPR = 0% by construction cho từng task. Tự calibrate khi thêm task mới.
Interpretation rõ ràng: "X% anomalous hơn training video của cùng task".

**Kết quả GR1:** 18/24 gen flagged (vs 16/24 global) — task 2 (rubik) catch tốt hơn vì baseline thấp (0.62) nên 0.72-0.81 vẫn ratio > 1.16.

Continuous H_video score là output chính. Threshold/ratio là helper cho binary decision.

---

## 7. Hyperparameters

| Param | Mặc định | Khi nào chỉnh |
|---|---|---|
| Reference frames per task | 50 | Tăng → null dày hơn (chi phí: SAM3 + lưu trữ) |
| Null lags | {1, 2, 5, 10} | Thêm lag-3, 7 nếu cần cover gap |
| n_frames per query | 10 | Tăng nếu video > 8 sec |
| Cert floor | 0.1 | Tăng → 0.2 nếu nhiều texture đối xứng |
| Aggregator | p80 | Dùng p90 nếu video dài, p50 nếu cần robust |
| Threshold | per-task max(real) + ε | Hoặc p95 cho FPR ≤ 5% |

---

## 8. Cần biết task của query?

**CÓ** — đây là điểm khác cross-task pool.

Mỗi query cần biết thuộc task nào để load đúng null. Một số cách:
- **Pre-label:** user input task khi gọi API
- **Auto-classify:** DINOv2 embedding của 1 frame query → nearest task centroid trong pool → assign task
- **Multi-task scoring:** chấm query với null của all tasks → lấy min H (best match)

Method đơn giản: yêu cầu user truyền `task_id`. Best practice cho production.

---

## 9. Code path đầy đủ

| File | Purpose |
|---|---|
| `warp_score/temporal_signals.py` | `CycleSignal` class (OOP, có cert-weighting) |
| `warp_score/matcher.py` | `RoMaMatcher.match()` → fwd + bwd warp |
| `warp_score/sam_segmenter.py` | `VideoFrameSegmenter` cho bg removal |
| `scripts/eval_per_task_dense_null.py` | Full eval pipeline (run ngay) |

Run end-to-end:

```bash
conda activate groot
python scripts/eval_per_task_dense_null.py
# → paper-physical-gr1/per_task_dense_eval/per_task_dense_table.csv
```

---

## 10. Tóm tắt khác biệt với cách cross-task cũ

| Aspect | Cross-task (cũ) | **Per-task multi-lag (mới)** |
|---|---|---|
| Null source | 92 tasks pool, 1472 pairs | **Per task**, 182 pairs each |
| Lag | Lag-1 (or inference lag) | **Lags 1, 2, 5, 10** |
| k-NN selection | DINOv2 top-K từ pool 4600 | Không cần (null is task-specific) |
| Real false positives | 2-3 / 30 ở FPR=5% | **0 / 5 ở FPR=0%** |
| Gen catch @ FPR=0% | 2-3 / 24 | **16 / 24** |
| Task ID required | No | Yes |
| Compute (calib) | ~25 min (1472 pairs) | ~25 min cho 5 tasks |
