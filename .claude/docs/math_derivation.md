# Mathematical derivation — precision-matrix hallucination detector

> Companion to [`../plans/precision_matrix_hallucination.md`](../plans/precision_matrix_hallucination.md).
> This document walks from assumptions → expected behavior → test statistic. Read top-down.

---

## 1. Assumptions

### A1 — Gaussian correspondence model
For each (query Q, ref R, pixel p), RoMaV2 outputs `(warp_r(p), Σ⁻¹_r(p))` with `Σ⁻¹_r(p) ∈ R^{2×2}` positive semi-definite. We interpret these as a Gaussian posterior over the true target coordinate:

```
c_r(p)  ~  N( warp_r(p),  Σ_r(p) )
```

This is the model RoMa is trained under (NLL-style loss on the precision). We rely on the *parametric form* of the output, not on the precision being perfectly calibrated — the empirical-null calibration absorbs miscalibration.

### A2 — Same-task refs share the scene
Under H0 (query is clean), the K refs are clean frames of the same task. There exists a single ground-truth correspondence `c*(p)` such that

```
c_r(p)  ⫫  c_{r'}(p) | c*(p)        for r ≠ r'
c_r(p)  ~ N( c*(p),  Σ_r(p) )
```

### A3 — Hallucination = no consistent c*(p) exists
Under H1, the query depicts content with no real correspondence to the refs. Either
- (a) refs are individually confident in different `warp_r(p)` → disagreement, or
- (b) refs collectively give up → low total precision everywhere.

### A4 — Background already zeroed
`cert = 0` on `(127,127,127)` pixels (SAM3 background), so `Σ⁻¹ = 0` effectively → bg pixels do not contribute. The 10-px erosion removes boundary noise.

### A5 — LOO refs ≡ clean null
Calibration takes each task's clean refs, leaves one out as a query, matches against the rest. The resulting LOO statistics are an exchangeable sample from the null distribution per task → empirical p-values are finite-sample valid.

---

## 2. Optimal consensus correspondence (Gauss-Markov / Kalman fusion)

Under A1+A2, the K observations are independent Gaussians with known precision. The maximum-likelihood estimate of `c*(p)` is the weighted least-squares solution:

```
μ̂(p)  =  argmin_c   Σ_r  (warp_r(p) − c)ᵀ  Σ⁻¹_r(p)  (warp_r(p) − c)
      =  Λ(p)⁻¹  ·  Σ_r  Σ⁻¹_r(p) · warp_r(p)

where    Λ(p)  =  Σ_r  Σ⁻¹_r(p)   ∈ R^{2×2}    (total precision)
```

This is the **BLUE / Kalman / inverse-variance-weighted estimator** — optimal under Gauss-Markov among linear unbiased estimators. Its posterior covariance is `Λ(p)⁻¹`.

For 2×2 matrices `Λ⁻¹` is the closed-form `(1/det Λ) · [[Λ₂₂, −Λ₁₂], [−Λ₂₁, Λ₁₁]]` — vectorize over `(H,W)`.

---

## 3. Disagreement deviance — the tightened `ivar`

The minimum value of the weighted SSE is the deviance:

```
D(p)  =  Σ_r  (warp_r(p) − μ̂(p))ᵀ  Σ⁻¹_r(p)  (warp_r(p) − μ̂(p))
```

### Expected behavior

**Under H0** (A1+A2), by Cochran's theorem (K independent 2-D Gaussians minus 2 d.f. for estimating `c*`):

```
D(p)  ~  χ²( 2(K−1) )
E[D(p)]    =  2(K−1)
Var[D(p)]  =  4(K−1)
```

At K=6 refs per task: `E[D] = 10`, `Var = 20`.

**Under H1**, refs disagree → `(warp_r − μ̂)` is large in the precision metric → `D(p) ≫ 2(K−1)`.

### Why `D` is the right "tightened ivar"

`D(p)` has the same physical meaning as the old `ivar` ("how much do refs disagree at pixel p?"), with three upgrades:

| Property | `ivar_old` | `ivar_new = D` |
|---|---|---|
| Metric on residuals | isotropic, scalar weight `cert_r` | anisotropic `Σ⁻¹_r` (uses directional confidence) |
| Mean estimator | cert-weighted scalar mean | BLUE / Gauss-Markov optimal |
| Null distribution | empirical only | empirical **and** parametric (χ²) |
| Failure mode "confident but wrong" | partially cancels with normalization | quadratic penalty, as it should |

**Strict generalization**: when `Σ⁻¹_r = c_r · I`, the two coincide up to a normalization. The assumption you trust is preserved; only the metric is upgraded.

### Frame-level statistic

```
s_ivar  =  mean_{p ∈ interior}  D(p)
```

Calibrated per task via LOO clean refs → empirical p-value `p_ivar`. (Same machinery as the existing `IvarSignal`.)

---

## 4. Information signal — replacing `cert`

`D(p)` measures *disagreement given confidence*. It does **not** catch the failure mode where refs uniformly give up (`Σ⁻¹_r ≈ 0` everywhere) — because the Mahalanobis residual scales with `Σ⁻¹`, both terms vanish and `D(p) → 0`. This is exactly when the old `cert` threshold was meant to fire.

The principled quantity is the **log marginal likelihood** of the refs under H0 (up to constants):

```
log p( {warp_r}_r | H0 ) (p)  ∝  ½ log det Λ(p)  −  ½ D(p)
```

The `½ log det Λ` term is the total *information* the refs collectively provide at `p`. Low total information ≡ matcher gave up.

Define (with sign flipped so high = anomalous):

```
e(p)        =  − log det Λ(p)
s_evidence  =  mean_{p ∈ interior}  e(p)
```

### Expected behavior

- Clean query, well-textured: `Λ(p)` large → `log det Λ ≫ 0` → `e(p) ≪ 0` → `s_evidence` very negative.
- OOD / heavily hallucinated query (matcher cannot match anything): `Λ(p) → 0` → `log det Λ → −∞` → `e(p)` large positive → `s_evidence` large positive.

### Orthogonality to `s_ivar`

| Failure mode | `s_ivar` | `s_evidence` |
|---|---|---|
| Refs disagree confidently (textbook hallu) | high | medium |
| Refs all give up (OOD frame) | low (false negative!) | high |
| Refs agree weakly | low | medium-high |
| Clean frame | low | low |

This is exactly why `evidence` is needed alongside `ivar`. Calibrated per task via LOO → `p_evidence`.

---

## 5. (Optional) Cycle-consistency signal

Both `s_ivar` and `s_evidence` operate purely on forward (query→ref) Gaussians. They can be fooled if each ref *individually* hallucinates a self-consistent and mutually-consistent warp. A geometric soundness check catches this.

For each ref `r`, also compute the reverse warp `warp_{r→q}`. Compose at each query pixel:

```
p_cycled(p)  =  warp_{r→q}( warp_{q→r}(p) )
ε_r(p)       =  ‖ p_cycled(p) − p ‖₂
```

Under a valid correspondence, the inverse exists up to subpixel noise: `E[ε_r(p)] = O(1 px)`. Under hallucination, the forward and reverse warps disagree.

Frame-level:

```
s_cycle  =  mean_{p ∈ interior, r}  cert_r(p) · ε_r(p)
```

Cost: doubles the matching budget. Default off; turn on only if AUROC shows residual misses after `s_ivar + s_evidence` deployment.

---

## 6. Fusion — handling correlation between signals

`s_ivar` and `s_evidence` both depend on the same `Σ⁻¹` field → their p-values are correlated under H0. Stouffer's Z-method assumes independence; using it here over-rejects.

The **Cauchy combination test** (Liu & Xie, 2020) gives a closed-form combined p-value valid under **arbitrary dependence**:

```
T_cauchy    =  Σ_i  w_i · tan( π · (½ − p_i) )         (default w_i = 1/k)
p_combined  =  ½  −  arctan(T_cauchy) / π
H_score     =  1  −  p_combined
```

Under H0, `T_cauchy` is approximately Cauchy(0,1) for very general dependence structures (this is the main theorem of Liu & Xie). Under H1 it skews positive (multiple p-values become small simultaneously).

One-liner addition to `fuser.py`.

---

## 7. Per-pixel heatmap (free byproduct)

`D(p) = ivar_new(p)` is itself the per-pixel hallucination map. Two routes:

### Empirical-null (primary, robust to RoMa precision miscalibration)
- LOO build per-pixel sorted null `T_null: (N, H, W)` — mirror of the existing `per_pixel_var` infra in `calibrator.py:243`.
- Heatmap: `heat(p) = 1 − F_emp( D(p) ; T_null[:, p] )`.

### Parametric (sanity / diagnostic, written alongside)
- `heat_chi2(p) = 1 − F_χ²( D(p); df = 2(K−1) )`.
- If parametric and empirical agree on most pixels → RoMa precision is well-calibrated.
- If they diverge → empirical is the safe path; record the divergence in calib metadata.

---

## 8. Summary table — why each piece is theoretically grounded (not heuristic)

| Quantity | Theory |
|---|---|
| `μ̂(p)` | Gauss-Markov BLUE estimator; Kalman fusion of Gaussian observations |
| `D(p)` | Chi-squared goodness-of-fit statistic for K Gaussian observations sharing a mean (Cochran) |
| `e(p)` | Negative log marginal likelihood (Bayesian evidence) of the Gaussian product |
| `s_cycle` | Forward-backward consistency check, standard in optical-flow / SfM literature (Brox & Malik 2010, Sundaram et al. 2010) |
| Empirical-null p-values | Exchangeable LOO refs → finite-sample valid p-values (no asymptotics, no independence assumption) |
| Cauchy combination | Closed-form meta-analysis under arbitrary dependence (Liu & Xie 2020) |

No within-frame standardization. No tunable thresholds (except optionally α for cluster-based diagnostics, which is the standard type-I rate). The pipeline is fully reference-based via LOO calibration on clean refs.

---

## References

- Hotelling, H. (1931). *The generalization of Student's ratio.* Ann. Math. Stat.
- Cochran, W. G. (1934). *The distribution of quadratic forms in a normal system.* Math. Proc. Camb. Philos. Soc.
- Kalman, R. E. (1960). *A new approach to linear filtering and prediction problems.* Trans. ASME.
- Brox, T., Malik, J. (2010). *Object segmentation by long-term analysis of point trajectories.* ECCV.
- Sundaram, N., Brox, T., Keutzer, K. (2010). *Dense point trajectories by GPU-accelerated large displacement optical flow.* ECCV.
- Liu, Y., Xie, J. (2020). *Cauchy combination test: A powerful test with analytic p-value calculation under arbitrary dependency structures.* JASA.
- Vovk, V., Gammerman, A., Shafer, G. (2005). *Algorithmic Learning in a Random World.* Springer. (Conformal/exchangeable p-values.)
