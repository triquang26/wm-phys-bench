"""Cert-weighted warp statistics and Mahalanobis precision-matrix statistics.

Given K (warp, cert) pairs from matching one query against K refs, compute:
- per-pixel cert-weighted mean warp coord
- per-pixel cert-weighted variance of warp coord (i.e. how much do refs disagree?)
- scalar interior mean variance (the primary "ivar" signal)
- per-pixel within-frame z-score of variance (the "peak" signal source)

v9 addition — MahalanobisStatistics:
- BLUE (Best Linear Unbiased Estimator) consensus warp via precision-matrix weighting
- Cochran's deviance D(p) ~ χ²(2(K-1)) per pixel as a principled discrepancy measure
- log-det evidence signal e(p) = -log det Λ(p) (high = matcher had low information)
"""
from __future__ import annotations

import numpy as np


class CertWeightedStatistics:
    """Pure-function utilities, kept as static methods for clarity."""

    @staticmethod
    def variance_per_pixel(
        warps: np.ndarray, certs: np.ndarray, eps: float = 1e-6,
    ) -> np.ndarray:
        """
        warps: (K, H, W, 2)   normalized coords
        certs: (K, H, W)      cert ∈ [0, 1] (bg zeroed)

        Returns: (H, W) cert-weighted variance summed over (x, y).
        """
        if warps.ndim != 4 or certs.ndim != 3:
            raise ValueError(
                f"Expected warps (K,H,W,2) and certs (K,H,W); got {warps.shape}, {certs.shape}"
            )
        w = certs / (certs.sum(axis=0, keepdims=True) + eps)
        mean_coord = (warps * w[..., None]).sum(axis=0)
        diff_sq = ((warps - mean_coord[None]) ** 2 * w[..., None]).sum(axis=0)
        return diff_sq.sum(axis=-1).astype(np.float32)

    @staticmethod
    def interior_mean(var_map: np.ndarray, interior_mask: np.ndarray) -> float:
        vals = var_map[interior_mask]
        if vals.size == 0:
            return 0.0
        return float(vals.mean())

    @staticmethod
    def within_frame_zscore(
        var_map: np.ndarray, interior_mask: np.ndarray,
    ) -> np.ndarray:
        """Return per-pixel z-score of var_map within interior; pixels outside
        interior are set to 0. Returns same shape as var_map."""
        vals = var_map[interior_mask]
        if vals.size < 2 or vals.std() < 1e-8:
            return np.zeros_like(var_map)
        z = (var_map - vals.mean()) / vals.std()
        z = z * interior_mask  # zero outside
        return z.astype(np.float32)

    @staticmethod
    def peak_max_z(var_map: np.ndarray, interior_mask: np.ndarray) -> float:
        """The 'peak' signal: max within-frame z-score on interior."""
        z = CertWeightedStatistics.within_frame_zscore(var_map, interior_mask)
        return float(z.max()) if z.size > 0 else 0.0

    @staticmethod
    def mean_cert_interior(certs: np.ndarray, interior_mask: np.ndarray) -> float:
        """Mean of the per-ref-averaged cert map, restricted to interior."""
        mean_cert = certs.mean(axis=0)  # (H, W)
        vals = mean_cert[interior_mask]
        if vals.size == 0:
            return 0.0
        return float(vals.mean())


# =============================================================================
# v9: Precision-matrix (Mahalanobis) statistics
# =============================================================================

class MahalanobisStatistics:
    """Precision-matrix weighted statistics for the v9 hallucination detector.

    These use the full per-pixel 2×2 precision matrix Σ⁻¹_r(p) returned by
    RoMaV2 ("precision_AB") rather than collapsing it to a scalar cert.

    Key quantities:
      Λ(p)  = Σ_r Σ⁻¹_r(p)                           total precision
      μ̂(p) = Λ(p)⁻¹ · Σ_r Σ⁻¹_r(p) warp_r(p)        BLUE consensus
      D(p)  = Σ_r (warp_r−μ̂)ᵀ Σ⁻¹_r (warp_r−μ̂)     Cochran deviance ~ χ²(2(K-1))
      e(p)  = −log det Λ(p)                            low-information evidence
    """

    # ------------------------------------------------------------------
    # 2×2 closed-form inverse
    # ------------------------------------------------------------------
    @staticmethod
    def _inv2x2(m: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Batch-invert (..., 2, 2) matrices via closed form.

        Background pixels (det ≈ 0) are returned as zero matrices.
        """
        a = m[..., 0, 0]
        b = m[..., 0, 1]
        c = m[..., 1, 0]
        d = m[..., 1, 1]
        det = a * d - b * c                  # (...,)
        valid = np.abs(det) > eps
        det_safe = np.where(valid, det, 1.0)  # avoid div-by-zero

        inv = np.stack([
            np.stack([ d, -b], axis=-1),
            np.stack([-c,  a], axis=-1),
        ], axis=-2) / det_safe[..., None, None]  # (..., 2, 2)

        # Zero out background pixels where det is too small
        inv = inv * valid[..., None, None].astype(inv.dtype)
        return inv

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    @staticmethod
    def ivar_per_pixel(
        warps: np.ndarray,       # (K, H, W, 2)
        precisions: np.ndarray,  # (K, H, W, 2, 2)
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute per-pixel Cochran deviance, log-det evidence, and BLUE mean.

        Parameters
        ----------
        warps : (K, H, W, 2) float32
        precisions : (K, H, W, 2, 2) float32  — Σ⁻¹_r per pixel

        Returns
        -------
        D_map : (H, W) float32
            Cochran deviance.  Under H₀ (consensus), D(p) ~ χ²(2(K-1)).
            High D = refs disagree in the metric defined by their own uncertainty.
        logdetΛ_map : (H, W) float32
            log det Λ(p) = log det (Σ_r Σ⁻¹_r).  Floored at -30 for background
            pixels (all Σ⁻¹_r = 0) where det < 1e-12.
            The *evidence* signal uses -logdetΛ (high = low information).
        mu_hat : (H, W, 2) float32
            BLUE consensus warp estimate.
        """
        if warps.ndim != 4 or precisions.ndim != 5:
            raise ValueError(
                f"Expected warps (K,H,W,2) and precisions (K,H,W,2,2); "
                f"got {warps.shape} and {precisions.shape}"
            )
        K, H, W, _ = warps.shape

        # Total precision: Λ(p) = Σ_r Σ⁻¹_r(p),  shape (H, W, 2, 2)
        Lambda = precisions.sum(axis=0)

        # BLUE numerator: Σ_r Σ⁻¹_r(p) warp_r(p),  shape (H, W, 2)
        # einsum: k,h,w,i,j × k,h,w,j  →  h,w,i
        sum_prec_warp = np.einsum("khwij,khwj->hwi", precisions, warps)

        # Λ⁻¹  (closed-form, zeroed for bg),  shape (H, W, 2, 2)
        Lambda_inv = MahalanobisStatistics._inv2x2(Lambda)

        # BLUE consensus: μ̂(p) = Λ⁻¹ · (Σ_r Σ⁻¹_r warp_r),  shape (H, W, 2)
        mu_hat = np.einsum("hwij,hwj->hwi", Lambda_inv, sum_prec_warp)

        # Residuals: (K, H, W, 2)
        residuals = warps - mu_hat[None]  # broadcast K

        # Precision-weighted residuals: Σ⁻¹_r · residual_r,  shape (K, H, W, 2)
        prec_resid = np.einsum("khwij,khwj->khwi", precisions, residuals)

        # Cochran deviance D(p) = Σ_r residual_rᵀ Σ⁻¹_r residual_r,  shape (H, W)
        D_map = (residuals * prec_resid).sum(axis=-1).sum(axis=0).astype(np.float32)

        # Log-det Λ(p),  shape (H, W).  Floor at -30 for background.
        _LOG_DET_FLOOR = -30.0
        _DET_EPS = 1e-12
        lam_a = Lambda[..., 0, 0]
        lam_b = Lambda[..., 0, 1]
        lam_c = Lambda[..., 1, 0]
        lam_d = Lambda[..., 1, 1]
        det_Lambda = lam_a * lam_d - lam_b * lam_c  # (H, W)
        valid_det = det_Lambda > _DET_EPS
        logdetΛ_map = np.where(
            valid_det,
            np.log(np.where(valid_det, det_Lambda, 1.0)),
            _LOG_DET_FLOOR,
        ).astype(np.float32)

        return D_map, logdetΛ_map, mu_hat.astype(np.float32)

    # ------------------------------------------------------------------
    # Scalar summaries
    # ------------------------------------------------------------------
    @staticmethod
    def interior_mean(map_2d: np.ndarray, interior_mask: np.ndarray) -> float:
        """Mean of map_2d over interior_mask pixels."""
        vals = map_2d[interior_mask]
        return float(vals.mean()) if vals.size > 0 else 0.0

    @staticmethod
    def peak_max_z(D_map: np.ndarray, interior_mask: np.ndarray) -> float:
        """Peak Z-score of D_map within interior.

        Unlike interior_mean (which is diluted by background-shifted pixels),
        max Z-score is invariant to a uniform additive shift in D_map — making
        it robust to domain gaps where all frames show elevated D_map.
        """
        vals = D_map[interior_mask]
        if vals.size < 2 or vals.std() < 1e-8:
            return 0.0
        z = (D_map - vals.mean()) / (vals.std() + 1e-8)
        return float(z[interior_mask].max())
