# WarpDyn — robot video hallucination / anomaly detection

Pure feature-matching anomaly detector cho video sinh ra bởi generator (Cosmos, OpenSora, ...) vs robot training distribution.

→ **Method docs**: [WARPDYN_METHOD.md](WARPDYN_METHOD.md)

## TL;DR

```bash
conda activate groot

# OFFLINE (per task, ~8 min): build per-task null + H_train baseline
python scripts/eval_per_task_dense_null.py

# ONLINE included in same script — produces per_task_dense_table.csv

# Compute per-task ratio + ranking
python scripts/compute_per_task_ratio.py
```

## Phương pháp

Per-task multi-lag null + cycle composition signal + per-task ratio scoring.

```
OFFLINE per task:
  50 reference frames extracted + SAM3-segmented
  Build 182 multi-lag pairs (lags 1, 2, 5, 10)
  RoMa fwd+bwd → cycle drift mean/peak per pair
  Sort into null distribution + compute H_train baseline

ONLINE per query:
  10 frames sampled + SAM3-segmented
  RoMa per 9 consecutive pairs → cycle signal
  Empirical p-value vs per-task null + Cauchy combine
  H_video = p80 percentile of 9 H_pair values
  ratio = H_video / H_train  →  > 1.0 = HALLU
```

## Kết quả tham chiếu (GR1, 5 tasks)

| Metric | Value |
|---|---|
| Real max H_peak | 0.832 |
| FPR @ ratio = 1.0 | **0% by construction** |
| Gen catch (ratio > 1.0) | **18/24 (75%)** |
| AUROC | 0.77 |
| Time / training video (offline) | ~8 min |
| Time / query (online) | ~10 sec |

## Files

| File | Purpose |
|---|---|
| `WARPDYN_METHOD.md` | **Method documentation** (step-by-step) |
| `warp_score/temporal_signals.py` | `CycleSignal` class + helpers |
| `warp_score/matcher.py` | `RoMaMatcher` wrapper |
| `warp_score/sam_segmenter.py` | `VideoFrameSegmenter` |
| `scripts/eval_per_task_dense_null.py` | End-to-end OFFLINE + ONLINE eval |
| `scripts/compute_per_task_ratio.py` | Per-task ratio scoring + ranking |

## Archive

- `scripts/_archive_exploration/` — exploratory scripts (cross-task pool, k-NN, voting, ALN, ...)
- `_archive_docs/` — earlier doc iterations và alternative method writeups
