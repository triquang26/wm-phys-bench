"""KNNFrameSignal — frame-level hallucination signal via DINOv2 k-NN + Cochran deviance.

For each query frame:
  1. DINOv2 (ViT-S/14) → L2-normalized CLS feature (384-dim)
  2. Cosine-sim with task's reference pool → top k=15 nearest refs
  3. RoMa match_batch query ↔ each of the k refs → k (warp, precision) pairs
  4. Cochran deviance D(p) = Σ_r (warp_r(p) − μ̂(p))ᵀ Σ⁻¹_r (warp_r(p) − μ̂(p))
     summary: ivar_maha (interior mean), peak_maha (interior z-score max)
  5. p_ivar, p_peak via empirical_p vs task's LOO null
  6. Route ivar vs peak per task (offline, from CV + training-ivar position)
  7. H_frame = 1 - p_routed

Per-task null built via leave-one-out: for each ref_i, score it as a pseudo-query
against the top-k nearest of the remaining refs. Same k as inference → same χ²(2(k-1))
df → statistically consistent.

Designed to complement CycleSignal (temporal self-consistency); fusion done by
warp_score.fusion.cauchy_combine_video at video level.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from warp_score.adaptive_refs import (
    AdaptiveRefSelector,
    DinoFeatureExtractor,
)
from warp_score.statistics import MahalanobisStatistics
from warp_score.temporal_signals import empirical_p_value


BG_GRAY = np.array([127, 127, 127], dtype=np.uint8)


def fg_mask_from_seg(bgr: np.ndarray) -> np.ndarray:
    """SAM3 convention: background pixels are (127,127,127). FG = everything else."""
    return ~np.all(bgr == BG_GRAY[None, None, :], axis=-1)


def pad_to_square_gray(bgr: np.ndarray) -> np.ndarray:
    """Pad BGR image with gray (127,127,127) to make square, preserving aspect.

    Matches RoMaMatcher._load_tensor padding so fg_mask is in the same
    coordinate system as the matched warp grid.
    """
    H, W = bgr.shape[:2]
    if H == W:
        return bgr
    side = max(H, W)
    pad_h, pad_w = side - H, side - W
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    return cv2.copyMakeBorder(bgr, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=(127, 127, 127))


def fg_mask_at_size(png_path: Path, size: int) -> np.ndarray:
    bgr = cv2.imread(str(png_path))
    if bgr is None:
        raise FileNotFoundError(png_path)
    bgr = pad_to_square_gray(bgr)
    bgr_r = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_NEAREST)
    return fg_mask_from_seg(bgr_r)


class KNNFrameSignal:
    """Frame-level k-NN Cochran deviance signal.

    Construction is cheap; DINOv2 is lazy-loaded on first feature extract.
    Reuses the caller-supplied RoMaMatcher instance (no extra RoMa load).
    """

    def __init__(
        self,
        matcher,                                # RoMaMatcher (already-instantiable)
        k: int = 15,
        dino_model: str = "dinov2_vits14",
        cv_threshold: float = 0.50,
        ivar_inversion_threshold: float = 0.5,  # if training_ivar percentile < this, suspect inversion
    ) -> None:
        self.matcher = matcher
        self.k = k
        self.cv_threshold = cv_threshold
        self.ivar_inversion_threshold = ivar_inversion_threshold
        self.dino = DinoFeatureExtractor(dino_model)
        self.selector = AdaptiveRefSelector(self.dino)

    # ─────────────────────────────────────────────────────────────────────
    # Pool building
    # ─────────────────────────────────────────────────────────────────────

    def build_pool(
        self,
        task: str,
        ref_pngs: list[Path],
        cache_dir: Path,
    ) -> dict:
        """Cache DINOv2 features for all refs. Returns dict with paths + feats."""
        feats = self.selector.build_cache(task, ref_pngs, cache_dir)
        return {"paths": ref_pngs, "feats": feats}

    # ─────────────────────────────────────────────────────────────────────
    # Single-frame scoring core
    # ─────────────────────────────────────────────────────────────────────

    def _score_one(
        self,
        query_path: Path,
        k_ref_paths: list[Path],
        query_fg_mask: np.ndarray,           # bool (H, W) at matcher.vis_size
    ) -> tuple[float, float]:
        """Match query against k refs → (ivar_maha, peak_maha)."""
        match_results = self.matcher.match_batch(
            query_path, k_ref_paths, fg_mask=query_fg_mask
        )
        warps = np.stack([m.warp for m in match_results], axis=0)
        precisions = np.stack([m.precision for m in match_results], axis=0)
        D_map, _, _ = MahalanobisStatistics.ivar_per_pixel(warps, precisions)
        ivar = MahalanobisStatistics.interior_mean(D_map, query_fg_mask)
        peak = MahalanobisStatistics.peak_max_z(D_map, query_fg_mask)
        return float(ivar), float(peak)

    # ─────────────────────────────────────────────────────────────────────
    # LOO calibration + routing decision
    # ─────────────────────────────────────────────────────────────────────

    def calibrate_loo(
        self,
        pool: dict,
        verbose: bool = True,
    ) -> dict:
        """Leave-one-out over the ref pool → null_ivar, null_peak + routing.

        For each ref_i:
            - pick top-k from {pool - i}
            - score ref_i as query → (ivar, peak)

        Returns dict with sorted null arrays and route ∈ {'ivar', 'peak'}.
        """
        ref_paths = pool["paths"]
        feats = pool["feats"]
        N = len(ref_paths)
        if N < self.k + 2:
            raise ValueError(f"Need ≥{self.k+2} refs, got {N}")

        nulls_ivar: list[float] = []
        nulls_peak: list[float] = []
        t0 = time.time()
        for i in range(N):
            cand_idx = [j for j in range(N) if j != i]
            top_k_local = self.selector.select_for_query(
                feats[i], feats[cand_idx], self.k
            )
            k_paths = [ref_paths[cand_idx[j]] for j in top_k_local]
            fg = fg_mask_at_size(ref_paths[i], self.matcher.vis_size)
            ivar, peak = self._score_one(ref_paths[i], k_paths, fg)
            nulls_ivar.append(ivar)
            nulls_peak.append(peak)
            if verbose and (i + 1) % 20 == 0:
                print(f"    LOO {i+1:3d}/{N}  ({time.time()-t0:.0f}s)")

        null_ivar = np.sort(np.asarray(nulls_ivar, dtype=np.float32))
        null_peak = np.sort(np.asarray(nulls_peak, dtype=np.float32))

        # ── Routing decision (offline) ─────────────────────────────────
        # CV: how dispersed is ivar null. Low CV → ivar is flat → prefer peak.
        cv = float(null_ivar.std() / max(null_ivar.mean(), 1e-8))
        # Note: ivar inversion (training looks anomalous vs null) is detected
        # later when scoring the training video; for now CV gates the route.
        route = "peak" if cv < self.cv_threshold else "ivar"

        return {
            "null_ivar": null_ivar,
            "null_peak": null_peak,
            "route": route,
            "cv": cv,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Inference scoring
    # ─────────────────────────────────────────────────────────────────────

    def score_frame(
        self,
        query_path: Path,
        pool: dict,
        null: dict,
        query_fg_mask: Optional[np.ndarray] = None,
    ) -> dict:
        """Per-frame H ∈ [0,1] via routed signal.

        Returns dict with:
            H        : 1 - empirical_p(routed_signal vs routed_null)
            ivar     : raw ivar_maha
            peak     : raw peak_maha
            p_ivar   : right-tail p-value for ivar
            p_peak   : right-tail p-value for peak
            route    : which signal was used
        """
        q_feat = self.dino.extract([query_path])[0]
        top_k = self.selector.select_for_query(q_feat, pool["feats"], self.k)
        k_refs = [pool["paths"][i] for i in top_k]

        if query_fg_mask is None:
            query_fg_mask = fg_mask_at_size(query_path, self.matcher.vis_size)

        ivar, peak = self._score_one(query_path, k_refs, query_fg_mask)
        p_ivar = empirical_p_value(ivar, null["null_ivar"])
        p_peak = empirical_p_value(peak, null["null_peak"])

        p_routed = p_peak if null["route"] == "peak" else p_ivar
        return {
            "H": 1.0 - p_routed,
            "ivar": float(ivar),
            "peak": float(peak),
            "p_ivar": float(p_ivar),
            "p_peak": float(p_peak),
            "route": null["route"],
        }
