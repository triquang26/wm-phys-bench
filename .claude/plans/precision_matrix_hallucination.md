# Precision-matrix hallucination detector — full plan

> Math derivation lives separately in [`../docs/math_derivation.md`](../docs/math_derivation.md). This file is the implementation plan + orchestration contract.
> **Branch**: `feat/precision-matrix-hallu`.
> **Orchestration**: 4 git worktrees, one Implementation agent each, running in parallel from this shared design contract.

---

## 1. Context

Current `warp_score/` detector (v8) uses 3 signals fused with Stouffer:

| Signal | Status | Issue |
|---|---|---|
| `ivar` (cert-weighted interior warp variance) | **Principled core, keep** | Uses *scalar* `cert = √det Σ⁻¹` as weight → discards anisotropy of the precision matrix |
| `peak` (within-frame z-max of warp variance) | Heuristic, drop | Within-frame standardization, not vs refs |
| `cert` (mean cert + threshold 0.10) | Heuristic, drop | Ad-hoc threshold on a scalar projection of richer information |

**Under-used asset**: `preds["precision_AB"]: (H, W, 2, 2)` from RoMaV2 is the full 2×2 precision matrix `Σ⁻¹_r(p)` per ref per pixel. `matcher.py:117-130` collapses it to a scalar. This is the lever for both tightening `ivar` and adding two orthogonal principled signals — at **zero extra matching cost**.

**Goal**: feed image → produce (a) per-pixel hallucination heatmap, (b) frame-level rankable H-score. Theoretically grounded, no heuristic, reference-based, no within-frame standardization.

For the assumptions and full math derivation (Gauss-Markov fusion, Cochran's theorem, log-evidence, Cauchy combination), see [`../docs/math_derivation.md`](../docs/math_derivation.md).

---

## 2. Summary of the new pipeline

Three new quantities computed from the full per-pixel precision matrix `Σ⁻¹_r(p)`:

```
Λ(p)        =  Σ_r Σ⁻¹_r(p)                                                (total precision)
μ̂(p)        =  Λ(p)⁻¹ · Σ_r Σ⁻¹_r(p) · warp_r(p)                          (BLUE consensus)
D(p)        =  Σ_r (warp_r(p) − μ̂(p))ᵀ Σ⁻¹_r(p) (warp_r(p) − μ̂(p))         ~ χ²(2(K−1)) under H0
e(p)        =  − log det Λ(p)                                              (high = matcher gave up)
```

Frame-level signals:

```
s_ivar      =  mean_{p ∈ interior}  D(p)                  ← tightened ivar (drop-in replacement)
s_evidence  =  mean_{p ∈ interior}  e(p)                  ← replaces cert, orthogonal to s_ivar
s_cycle     =  mean_{p ∈ interior, r}  cert_r(p) · ε_r(p) ← optional, default off (2× match cost)
```

Per-signal p-value via LOO empirical null (per task). Fusion with **Cauchy combination** (Liu & Xie 2020) — closed-form, valid under arbitrary dependence between p_values (which is needed because `s_ivar` and `s_evidence` both depend on the same `Σ⁻¹` field → correlated).

```
T_cauchy   =  Σ_i  w_i · tan( π · (½ − p_i) )                     (default w_i = 1/k)
p_combined =  ½ − arctan(T_cauchy) / π
H_score    =  1 − p_combined
```

Heatmap: per-pixel empirical p-value `1 − F_emp(D(p) ; T_null[:, p])` where `T_null:(N,H,W)` comes from LOO clean refs (mirror of existing `per_pixel_var` infra).

---

## 3. What changes in code

### 3.1 Files & responsibilities

| File | Change |
|---|---|
| `warp_score/matcher.py:20-25, 88-130` | `MatchResult` gains `precision: (H,W,2,2)`. Resize precision with same bilinear interp as warp/cert. Keep scalar `cert = √det Σ⁻¹` derived for viz. Optional `bidirectional` path for `s_cycle`. |
| `warp_score/statistics.py` | Add `MahalanobisStatistics` with: `ivar_per_pixel(warps, precisions) → (D_map, logdetΛ_map, μ̂_map)`, `interior_mean(...)`. Keep `CertWeightedStatistics` for ablation only. |
| `warp_score/signals.py` | `IvarSignal.raw_for(stats)` now reads `D_map`; add `EvidenceSignal`, optional `CycleSignal`; demote `PeakSignal`/`CertSignal` to ablation-only registry. |
| `warp_score/calibrator.py:215-294` | `_stats_for_query` stacks precisions, computes `D_map`, `logdetΛ_map` (+ cycle if on). Stores per-task `ivar_dist`, `evidence_dist`, `T_null:(N,H,W)`. `TaskCalibration` gains `evidence_dist`. |
| `warp_score/fuser.py:32-86` | Add `CauchyFuser`; register in `build_fuser`. |
| `warp_score/detector.py:122-145` | Wire new signals; heatmap from `D_map` vs `T_null`. |
| `warp_score/config.py` | Default `signals=("ivar","evidence")`, `fuser="cauchy"`, `cycle_consistency: bool = False`. |
| `warp_score/configs/maha.yaml` (new) | Activates new pipeline; leave `default.yaml` untouched. |

### 3.2 Interface contract (agents read this verbatim)

- `MatchResult.precision: np.ndarray  # (H, W, 2, 2) float32, positive semi-definite, bg pixels zeroed`
- `MahalanobisStatistics.ivar_per_pixel(warps: (K,H,W,2), precisions: (K,H,W,2,2)) -> tuple[(H,W), (H,W), (H,W,2)]` returning `(D_map, logdetΛ_map, μ̂_map)`.
- `TaskCalibration` gains: `evidence_dist: np.ndarray  # (N,) sorted asc`, and the existing `per_pixel_var` is renamed to `T_null` (or aliased) for the new pipeline.
- `CauchyFuser(SignalFuser)` — same `fuse(p_values: dict[str,float]) -> float` interface as existing fusers.
- CSV columns: keep old `raw_ivar`, `p_ivar`, `raw_peak`, `p_peak`, `raw_cert`, `p_cert` for ablation; add `raw_ivar_maha`, `p_ivar_maha`, `raw_evidence`, `p_evidence`, `H_score_maha`.

### 3.3 Backward compatibility

- Old signals + Stouffer kept, gated by config (`legacy_ivar: true`). CSV writes both old and new columns. AUROC report compares them side-by-side.
- A CLI flag `--legacy_ivar` reproduces v8 behavior exactly.

---

## 4. Branch, worktree, multi-agent orchestration

### 4.1 Branch + worktree layout

```
feature_matching_eval_hallucination/        ← main worktree (planning + integration)
  └── (branch: feat/precision-matrix-hallu)

../wt-maha-statistics/                       ← Agent S worktree
../wt-maha-calibrator/                       ← Agent C worktree
../wt-maha-detector/                         ← Agent D worktree
../wt-maha-docs/                             ← Agent W worktree
```

Each worktree shares `.git` (one branch) but holds an isolated checkout so agents can edit without stepping on each other.

### 4.2 Per-agent prompts

| Agent | Worktree | Files it owns | Done when |
|---|---|---|---|
| **S — Statistics** | `wt-maha-statistics` | `matcher.py` (return precision), `statistics.py` (add `MahalanobisStatistics`), `signals.py` (add Evidence, Cycle classes), unit tests under `tests/test_mahalanobis.py` | Tests for: 2-ref synthetic case `D ~ χ²(2)`, identity-precision case matching old ivar, anisotropic case differing as expected. |
| **C — Calibrator** | `wt-maha-calibrator` | `calibrator.py` (LOO with new stats + `T_null`), `config.py` (defaults), `configs/maha.yaml` (new), unit test for new `TaskCalibration` save/load roundtrip | New calib file roundtrips; smoke test on 1 task with 2 refs. |
| **D — Detector + Fuser** | `wt-maha-detector` | `detector.py` (wire new signals + heatmap), `fuser.py` (add `CauchyFuser`), `evaluator.py` (AUROC of both old + new pipelines side-by-side), `cli.py` (flag `--legacy_ivar`) | Detect on `data/query/low/0_Open the box/frame_0005.png` returns sensible H-score with new pipeline; legacy flag reproduces v8 numbers within float tolerance. |
| **W — Writer** | `wt-maha-docs` | `.claude/plans/precision_matrix_hallucination.md`, `.claude/docs/math_derivation.md`, README.md update, APPROACH.md update with §13 "v9 Precision Matrix" | Docs render cleanly; math derivation matches `math_derivation.md`; references all match file:line in the new code. |

Dependency: **S must finish first** (it pins the interface signatures used by C and D). Then C, D, W run in parallel.

### 4.3 Integration sequence (main worktree)

1. **Pre-flight**: snapshot v8 baseline AUROC to `results/v8_baseline_summary.csv` (read-only verification).
2. **Create branch + worktrees**:
   ```bash
   git checkout -b feat/precision-matrix-hallu
   for w in statistics calibrator detector docs; do
     git worktree add ../wt-maha-$w feat/precision-matrix-hallu
   done
   ```
3. **Spawn S** (foreground, ~10 min). On finish, S commits its changes to `feat/precision-matrix-hallu`; main worktree picks them up with `git pull` from the worktree's branch (same branch, FF).
4. **Spawn C, D, W in parallel** (background) once S has merged interfaces.
5. **Per agent**: each commits to the shared branch from its own worktree (or pushes a sub-branch and FF-merges).
6. **Integration & ablation in main worktree**: run calibration; run AUROC on all 138 frames for v8 vs v9-A (ivar_new only) vs v9-B (ivar_new + evidence + Cauchy); commit summary CSV.
7. **PR**: open one PR `feat/precision-matrix-hallu → main` with all changes + ablation report.

### 4.4 Tracking

Use `TaskCreate` upon orchestration start to enumerate the 7 steps above. Mark each agent's completion explicitly so a stale worktree never lingers (`git worktree remove ../wt-maha-*` once merged).

---

## 5. Verification plan (post-merge into integration branch)

| Step | Check | Pass criterion |
|---|---|---|
| 1 — calib sanity | Recompute on `data/reference/`; print `mean(s_ivar_clean)` per task | Within ±50% of `2(K−1)` for most tasks. If far off, parametric χ² is unreliable but empirical-null still valid (note in calib metadata). |
| 2 — AUROC ablation | Run on full bench (138 frames), variants: v8 baseline, v9-A (ivar_new alone), v9-B (ivar_new+evidence+Cauchy), v9-C (+cycle) | v9-B AUROC ≥ v8 AUROC, and ideally drops false-positive rate on the 19-frame peak-only set without losing the 97 true positives. |
| 3 — Heatmap visual | Render `1 − F_emp(D(p))` on frame_0004/5/6 of `0_Open the box` | Hot zones cover the hallucinated robot region, not foreground boundary; bg masked out. |
| 4 — Known-positive | frame_0001 (clean) and frame_0004/5/6 (hallu) in `0_Open the box` keep correct H labels under v9-B | identical to v8 ground truth column. |
| 5 — Ranking | Sort 138 frames by H_score; top-20 ⊆ labeled `low`, bottom-20 ⊆ labeled `high` | Ranking matches existing labels with no more than 2 swaps. |
| 6 — Legacy reproducibility | Run with `--legacy_ivar` | v8 numbers reproduced within float tolerance. |

---

## 6. Run commands (post-implementation)

```bash
cd feature_matching_eval_hallucination
PYTHONPATH=../../../RoMaV2/src:$PYTHONPATH conda run -n groot --no-capture-output \
  python -m warp_score calibrate \
    --config warp_score/configs/maha.yaml \
    --ref_dir data/reference \
    --calib_file results/warp_variance_v9/calib.npz

PYTHONPATH=../../../RoMaV2/src:$PYTHONPATH conda run -n groot --no-capture-output \
  python -m warp_score detect \
    --config warp_score/configs/maha.yaml \
    --calib_file results/warp_variance_v9/calib.npz \
    --query_dir data/query \
    --out_dir results/warp_variance_v9

PYTHONPATH=../../../RoMaV2/src:$PYTHONPATH conda run -n groot --no-capture-output \
  python -m warp_score eval \
    --summary results/warp_variance_v9/summary.csv \
    --labels labels.csv \
    --compare results/warp_variance_v8/summary.csv
```
