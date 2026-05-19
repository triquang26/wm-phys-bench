"""WarpDyn temporal signals — pure feature-matching intra-video anomaly.

Two signals computed from RoMa dense warp fields (no optical-flow model):

  S2  cycle_error:    fwd-bwd composition drift between consecutive frames.
                      Real video pairs ≈ identity composition; generated
                      frame pairs accumulate drift due to diffusion artifacts.

  S3  traj_accel:     multi-lag grid-point acceleration. Track a uniform
                      grid through W(f_t -> f_{t+1}) and W(f_{t+1} -> f_{t+2});
                      real motion is smooth (low accel), generated motion
                      shows stationary stutter or jerk.

Both signals return scalar per query frame + per-pixel heatmaps for viz.
Calibration null distributions are fit from REAL training-video pairs.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Warp-field coordinate conventions
#
# RoMa returns warp_AB of shape (H, W, 2) in normalized grid_sample coords:
#   warp[y, x] = (u, v) where u, v ∈ [-1, 1] are the normalized coords
#   in image B that pixel (x, y) of image A maps to.
#
# Helpers below convert between normalized and pixel-coord systems and
# bilinear-sample warp fields.
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


# ─────────────────────────────────────────────────────────────────────────────
# S2: forward-backward cycle composition error
# ─────────────────────────────────────────────────────────────────────────────


def cycle_error_map(
    warp_fwd: np.ndarray,
    warp_bwd: np.ndarray,
) -> np.ndarray:
    """Per-pixel cycle drift between forward and backward warps.

    Args:
        warp_fwd: (H, W, 2) RoMa-normalized warp A → B.
        warp_bwd: (H, W, 2) RoMa-normalized warp B → A.

    Returns:
        err_map: (H, W) cycle drift in pixel units (||p − bwd(fwd(p))||).
    """
    H, W = warp_fwd.shape[:2]

    fwd_px = _norm_to_pixel(warp_fwd)            # (H, W, 2) coords in B
    bwd_at_fwd = _sample_warp_at(warp_bwd, fwd_px)  # (H, W, 2) still normalized
    back_px = _norm_to_pixel(bwd_at_fwd)         # (H, W, 2) coords in A

    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    orig_px = np.stack([xx, yy], axis=-1).astype(np.float32)

    err = np.linalg.norm(back_px - orig_px, axis=-1).astype(np.float32)
    return err


def cycle_signal(
    warp_fwd: np.ndarray,
    warp_bwd: np.ndarray,
    cert_fwd: Optional[np.ndarray] = None,
    interior_mask: Optional[np.ndarray] = None,
) -> dict:
    """Cycle composition signal — mean and peak drift, cert-weighted.

    Returns dict with:
        mean      cert-weighted interior mean cycle drift
        peak      99th percentile drift (for tail anomaly)
        err_map   (H, W) per-pixel drift (for heatmap viz)
    """
    err_map = cycle_error_map(warp_fwd, warp_bwd)
    H, W = err_map.shape

    if interior_mask is None:
        interior_mask = np.ones((H, W), dtype=bool)

    weights = cert_fwd if cert_fwd is not None else np.ones((H, W), dtype=np.float32)
    weights = weights.astype(np.float32) * interior_mask.astype(np.float32)
    w_sum = weights.sum()
    if w_sum < 1e-6:
        mean_val = float(err_map[interior_mask].mean()) if interior_mask.any() else 0.0
    else:
        mean_val = float((err_map * weights).sum() / w_sum)

    vals = err_map[interior_mask]
    peak_val = float(np.percentile(vals, 99.0)) if vals.size > 0 else 0.0

    return {"mean": mean_val, "peak": peak_val, "err_map": err_map}


# ─────────────────────────────────────────────────────────────────────────────
# S3: multi-lag trajectory acceleration
# ─────────────────────────────────────────────────────────────────────────────


def trajectory_accel(
    warp_01: np.ndarray,
    warp_12: np.ndarray,
    grid_size: int = 16,
) -> dict:
    """Grid-point trajectory acceleration across 3 consecutive frames.

    Track a `grid_size × grid_size` uniform lattice from frame f_0 forward:
        p_0 → p_1 via warp_01     (warp_01 = RoMa(f_0 → f_1))
        p_1 → p_2 via warp_12     (warp_12 = RoMa(f_1 → f_2))
    Then acceleration = ||p_2 - 2 p_1 + p_0|| (discrete second derivative).
    Real motion is smooth → small accel; generated stutter/jerk → large accel.

    Returns dict with:
        mean      mean trajectory acceleration over the grid
        peak      max grid-point acceleration
        accel_map (grid_size, grid_size) per-point acceleration
    """
    H, W = warp_01.shape[:2]
    ys = np.linspace(0, H - 1, grid_size).astype(np.float32)
    xs = np.linspace(0, W - 1, grid_size).astype(np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    p0 = np.stack([xx, yy], axis=-1).astype(np.float32)   # (G, G, 2)

    # p_1 = where p_0 ends up in f_1 (via warp_01)
    warp_01_px_field = _norm_to_pixel(warp_01)             # (H, W, 2)
    p1 = _sample_warp_at(warp_01_px_field, p0)             # (G, G, 2)

    # p_2 = where p_1 ends up in f_2 (via warp_12, sampled at p_1)
    warp_12_px_field = _norm_to_pixel(warp_12)
    p2 = _sample_warp_at(warp_12_px_field, p1)             # (G, G, 2)

    accel_vec = p2 - 2.0 * p1 + p0
    accel_map = np.linalg.norm(accel_vec, axis=-1).astype(np.float32)  # (G, G)

    return {
        "mean": float(accel_map.mean()),
        "peak": float(accel_map.max()),
        "accel_map": accel_map,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Empirical p-value (shared with appearance signal logic)
# ─────────────────────────────────────────────────────────────────────────────


def empirical_p_value(value: float, sorted_null: np.ndarray) -> float:
    """Right-tail empirical p-value: P(null ≥ value).

    Smoothed so p ∈ (0, 1) strictly (avoids Cauchy ±∞ blow-up).
    """
    n = sorted_null.size
    if n == 0:
        return 0.5
    rank = int(np.searchsorted(sorted_null, value, side="right"))
    p = (n - rank + 0.5) / (n + 1.0)
    return float(np.clip(p, 1.0 / (n + 1), 1.0 - 1.0 / (n + 1)))
