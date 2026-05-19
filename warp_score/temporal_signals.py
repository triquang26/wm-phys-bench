"""WarpDyn temporal signals — pure feature-matching intra-video anomaly.

Class hierarchy:

    TemporalSignal (abstract)
    ├── CycleSignal              — fwd/bwd warp composition drift (S2)
    ├── PrecisionAnomalySignal   — cycle drift × RoMa precision (S2-PA, motion-robust)
    └── TrajectoryAccelSignal    — multi-lag grid acceleration (S3, currently weak)

All operate on `MatchResult` objects (warp + cert + precision) so signals
can use the FULL match output, not just the scalar cert. PrecisionAnomalySignal
uses RoMa's precision matrix determinant — high precision regions count
more, which discriminates "real high-motion drift" (low precision in
moving region) from "generator artifacts" (high precision but inconsistent).

Calibration null distributions are fit from REAL training-video pairs.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Warp-field coordinate conventions
#
# RoMa returns warp_AB of shape (H, W, 2) in normalized grid_sample coords:
#   warp[y, x] = (u, v) where u, v ∈ [-1, 1] are the normalized coords
#   in image B that pixel (x, y) of image A maps to.
# ─────────────────────────────────────────────────────────────────────────────


def _norm_to_pixel(warp_norm: np.ndarray) -> np.ndarray:
    """[-1, 1] normalized → pixel coords. warp_norm: (H, W, 2)."""
    H, W = warp_norm.shape[:2]
    px = (warp_norm[..., 0] + 1.0) * (W - 1) / 2.0
    py = (warp_norm[..., 1] + 1.0) * (H - 1) / 2.0
    return np.stack([px, py], axis=-1).astype(np.float32)


def _sample_warp_at(warp_field: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    """Bilinear-sample a (H, W, 2) warp field at pixel coords (..., 2)."""
    field_x = warp_field[..., 0].astype(np.float32)
    field_y = warp_field[..., 1].astype(np.float32)
    map_x = pts_xy[..., 0].astype(np.float32)
    map_y = pts_xy[..., 1].astype(np.float32)
    sx = cv2.remap(field_x, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    sy = cv2.remap(field_y, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return np.stack([sx, sy], axis=-1)


def cycle_error_map(warp_fwd: np.ndarray, warp_bwd: np.ndarray) -> np.ndarray:
    """Per-pixel cycle drift between forward and backward warps (pixel units)."""
    H, W = warp_fwd.shape[:2]
    fwd_px = _norm_to_pixel(warp_fwd)
    bwd_at_fwd = _sample_warp_at(warp_bwd, fwd_px)
    back_px = _norm_to_pixel(bwd_at_fwd)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    orig_px = np.stack([xx, yy], axis=-1).astype(np.float32)
    return np.linalg.norm(back_px - orig_px, axis=-1).astype(np.float32)


def precision_det(precision: np.ndarray) -> np.ndarray:
    """det of (H, W, 2, 2) precision matrix → (H, W) confidence per pixel.

    Higher = RoMa is more confident in the warp at this pixel.
    sqrt(det) keeps it in linear scale with cert (which is sqrt(det)/max).
    """
    p = precision
    det = p[..., 0, 0] * p[..., 1, 1] - p[..., 0, 1] * p[..., 1, 0]
    return np.clip(det, 0, None).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Signal result + abstract base
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SignalResult:
    """Per-pair temporal-signal output."""
    name:           str
    mean:           float
    peak:           float
    pixel_map:      Optional[np.ndarray] = None   # (H, W) per-pixel score (heatmap)
    metadata:       dict[str, Any] = field(default_factory=dict)


class TemporalSignal(ABC):
    """Abstract base — compute a scalar+pixel-map score from a frame pair."""

    name: str = "abstract"

    def __init__(self, cert_floor: float = 0.1, peak_percentile: float = 99.0):
        self.cert_floor = cert_floor
        self.peak_percentile = peak_percentile

    @abstractmethod
    def _per_pixel_score(self,
                         match_fwd,                 # MatchResult
                         match_bwd,                 # MatchResult
                         ) -> np.ndarray:
        """Compute per-pixel signal map (H, W). Subclasses override."""
        ...

    def _validity_mask(self,
                       match_fwd,
                       interior_mask: Optional[np.ndarray],
                       ) -> np.ndarray:
        cert = match_fwd.cert.astype(np.float32)
        H, W = cert.shape
        interior = (interior_mask if interior_mask is not None
                    else np.ones((H, W), dtype=bool))
        return interior & (cert > self.cert_floor)

    def _aggregate(self,
                   score_map: np.ndarray,
                   weights:   np.ndarray,
                   valid:     np.ndarray,
                   ) -> tuple[float, float]:
        """Cert-weighted mean and high-percentile peak over valid pixels."""
        if not valid.any():
            return 0.0, 0.0
        w_sum = float(weights[valid].sum())
        if w_sum < 1e-6:
            mean_val = float(score_map[valid].mean())
        else:
            mean_val = float((score_map[valid] * weights[valid]).sum() / w_sum)
        vals = score_map[valid]
        if vals.size < 100:
            peak_val = float(vals.max())
        else:
            peak_val = float(np.percentile(vals, self.peak_percentile))
        return mean_val, peak_val

    def compute(self,
                match_fwd,                          # MatchResult (warp, cert, precision)
                match_bwd,                          # MatchResult
                interior_mask: Optional[np.ndarray] = None,
                ) -> SignalResult:
        score_map = self._per_pixel_score(match_fwd, match_bwd)
        valid = self._validity_mask(match_fwd, interior_mask)
        weights = match_fwd.cert.astype(np.float32) * valid.astype(np.float32)
        mean_val, peak_val = self._aggregate(score_map, weights, valid)
        return SignalResult(
            name=self.name,
            mean=mean_val,
            peak=peak_val,
            pixel_map=score_map,
            metadata={"cert_floor": self.cert_floor, "peak_q": self.peak_percentile},
        )


# ─────────────────────────────────────────────────────────────────────────────
# CycleSignal — original S2
# ─────────────────────────────────────────────────────────────────────────────


class CycleSignal(TemporalSignal):
    """Forward-backward warp composition drift (cert-weighted)."""
    name = "cycle"

    def _per_pixel_score(self, match_fwd, match_bwd) -> np.ndarray:
        return cycle_error_map(match_fwd.warp, match_bwd.warp)


# ─────────────────────────────────────────────────────────────────────────────
# PrecisionAnomalySignal — NEW S2-PA
# ─────────────────────────────────────────────────────────────────────────────


class PrecisionAnomalySignal(TemporalSignal):
    """Cycle drift weighted by RoMa precision determinant — motion-robust.

    Motivation:
        Real videos with fast motion produce LARGE cycle drift in moving
        regions, but RoMa is also UNCERTAIN in those regions (low precision).
        Generated-video artifacts (texture flicker, ghosting) produce HIGH
        cycle drift WHILE RoMa stays falsely confident (mid-to-high precision).
        Per-pixel score = cycle × sqrt(precision_det) suppresses real motion
        (low precision compensates) and amplifies generator artifacts.
    """
    name = "precision_anomaly"

    def _per_pixel_score(self, match_fwd, match_bwd) -> np.ndarray:
        cycle = cycle_error_map(match_fwd.warp, match_bwd.warp)
        if match_fwd.precision is None:
            conf = match_fwd.cert.astype(np.float32)
        else:
            det = precision_det(match_fwd.precision)
            conf = np.sqrt(det).astype(np.float32)
            ref = float(np.percentile(conf, 99.0))
            if ref > 1e-8:
                conf = np.clip(conf / ref, 0, 1)
        return cycle * conf


# ─────────────────────────────────────────────────────────────────────────────
# LostPixelSignal — "if a point can't be found, it's likely hallu"
# ─────────────────────────────────────────────────────────────────────────────


class LostPixelSignal(TemporalSignal):
    """Fraction of interior pixels where the match is unreliable ("lost").

    Per-pixel a pixel is "lost" if RoMa cannot find a sharp correspondence
    in the next frame — operationalized via:
        cert(x,y) < cert_thresh   OR   sqrt(det(precision(x,y))) < prec_thresh

    Motivation (user's intuition):
        For a real consecutive pair, even with motion most pixels can be
        tracked into the next frame (background stays, the moving robot
        arm still has a unique destination). For a generated pair with
        textural artifacts or object morphing, many pixels have NO clean
        correspondence — RoMa returns diffuse / low-confidence matches.

    Outputs:
        mean = global fraction of lost pixels (in interior)
        peak = max 16×16 block density of lost pixels (clustered failure)
    """
    name = "lost_pixel"

    def __init__(self,
                 cert_thresh: float = 0.3,
                 prec_thresh: float = 0.10,
                 block_size: int = 16):
        # cert_floor=0 because we *want* the low-cert pixels here
        super().__init__(cert_floor=0.0)
        self.cert_thresh = cert_thresh
        self.prec_thresh = prec_thresh
        self.block_size = block_size

    def _per_pixel_score(self, match_fwd, match_bwd) -> np.ndarray:
        cert = match_fwd.cert.astype(np.float32)
        lost = cert < self.cert_thresh
        if match_fwd.precision is not None:
            det = precision_det(match_fwd.precision)
            prec_norm = np.sqrt(det).astype(np.float32)
            ref = float(np.percentile(prec_norm, 99.0))
            if ref > 1e-8:
                prec_norm = prec_norm / ref
            lost |= (prec_norm < self.prec_thresh)
        return lost.astype(np.float32)

    def _validity_mask(self, match_fwd, interior_mask):
        H, W = match_fwd.cert.shape
        return (interior_mask if interior_mask is not None
                else np.ones((H, W), dtype=bool))

    def _aggregate(self, score_map, weights, valid):
        if not valid.any():
            return 0.0, 0.0
        rate = float(score_map[valid].mean())
        # Local clustering peak: max-mean over BxB blocks
        H, W = score_map.shape
        b = self.block_size
        hb, wb = H // b, W // b
        if hb > 0 and wb > 0:
            cropped = (score_map * valid.astype(np.float32))[:hb * b, :wb * b]
            blocks = cropped.reshape(hb, b, wb, b).mean(axis=(1, 3))
            peak = float(blocks.max())
        else:
            peak = rate
        return rate, peak


# ─────────────────────────────────────────────────────────────────────────────
# BidirectionalCertSignal — fwd cert high but bwd cert at destination low
# ─────────────────────────────────────────────────────────────────────────────


class BidirectionalCertSignal(TemporalSignal):
    """Asymmetric match confidence: forward high, backward at destination low.

    Per-pixel score:
        s(x,y) = max(0, cert_fwd(x,y) - cert_bwd(warp_fwd(x,y)))

    Real bidirectional matches: cert_fwd ≈ cert_bwd at destination → s ≈ 0.
    Generator artifacts often break this symmetry — forward looks plausible
    but the destination pixel doesn't think it maps back to (x,y).
    """
    name = "biduri_cert"

    def _per_pixel_score(self, match_fwd, match_bwd) -> np.ndarray:
        fwd_px = _norm_to_pixel(match_fwd.warp)
        bwd_cert_at_dest = cv2.remap(
            match_bwd.cert.astype(np.float32),
            fwd_px[..., 0].astype(np.float32),
            fwd_px[..., 1].astype(np.float32),
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
        )
        discrepancy = np.clip(match_fwd.cert.astype(np.float32) - bwd_cert_at_dest, 0, 1)
        return discrepancy


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory acceleration — original S3 (kept for completeness)
# ─────────────────────────────────────────────────────────────────────────────


class TrajectoryAccelSignal:
    """Multi-lag grid acceleration. Standalone — not a TemporalSignal subclass
    because it needs THREE consecutive warps, not a pair."""
    name = "trajectory_accel"

    def __init__(self, grid_size: int = 16):
        self.grid_size = grid_size

    def compute(self, warp_01: np.ndarray, warp_12: np.ndarray) -> SignalResult:
        H, W = warp_01.shape[:2]
        ys = np.linspace(0, H - 1, self.grid_size).astype(np.float32)
        xs = np.linspace(0, W - 1, self.grid_size).astype(np.float32)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        p0 = np.stack([xx, yy], axis=-1).astype(np.float32)
        warp_01_px = _norm_to_pixel(warp_01)
        warp_12_px = _norm_to_pixel(warp_12)
        p1 = _sample_warp_at(warp_01_px, p0)
        p2 = _sample_warp_at(warp_12_px, p1)
        accel_vec = p2 - 2.0 * p1 + p0
        accel_map = np.linalg.norm(accel_vec, axis=-1).astype(np.float32)
        return SignalResult(
            name=self.name,
            mean=float(accel_map.mean()),
            peak=float(accel_map.max()),
            pixel_map=accel_map,
            metadata={"grid_size": self.grid_size},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Backwards-compatible function wrappers (so old scripts keep working)
# ─────────────────────────────────────────────────────────────────────────────


def cycle_signal(warp_fwd: np.ndarray,
                 warp_bwd: np.ndarray,
                 cert_fwd: Optional[np.ndarray] = None,
                 interior_mask: Optional[np.ndarray] = None,
                 cert_floor: float = 0.1,
                 ) -> dict:
    """Legacy function — wraps CycleSignal class. Returns dict for old callers."""

    class _FakeMatch:
        def __init__(self, warp, cert):
            self.warp = warp
            self.cert = cert if cert is not None else np.ones(warp.shape[:2], dtype=np.float32)
            self.precision = None

    fwd = _FakeMatch(warp_fwd, cert_fwd)
    bwd = _FakeMatch(warp_bwd, None)
    sig = CycleSignal(cert_floor=cert_floor).compute(fwd, bwd, interior_mask)
    return {"mean": sig.mean, "peak": sig.peak, "err_map": sig.pixel_map}


def trajectory_accel(warp_01: np.ndarray,
                     warp_12: np.ndarray,
                     grid_size: int = 16,
                     ) -> dict:
    """Legacy function — wraps TrajectoryAccelSignal class."""
    sig = TrajectoryAccelSignal(grid_size=grid_size).compute(warp_01, warp_12)
    return {"mean": sig.mean, "peak": sig.peak, "accel_map": sig.pixel_map}


def empirical_p_value(value: float, sorted_null: np.ndarray) -> float:
    """Right-tail empirical p-value: P(null ≥ value), smoothed."""
    n = sorted_null.size
    if n == 0:
        return 0.5
    rank = int(np.searchsorted(sorted_null, value, side="right"))
    p = (n - rank + 0.5) / (n + 1.0)
    return float(np.clip(p, 1.0 / (n + 1), 1.0 - 1.0 / (n + 1)))
