"""Cert-weighted warp statistics.

Given K (warp, cert) pairs from matching one query against K refs, compute:
- per-pixel cert-weighted mean warp coord
- per-pixel cert-weighted variance of warp coord (i.e. how much do refs disagree?)
- scalar interior mean variance (the primary "ivar" signal)
- per-pixel within-frame z-score of variance (the "peak" signal source)
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
