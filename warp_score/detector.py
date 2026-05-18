"""WarpVarianceDetector — orchestrate matching → signals → fusion → H_score.

Single-frame inference contract:
    detector.detect(query_path)  →  HallucinationResult
        H_score ∈ [0, 1]
        is_hallucination = H_score > config.decision_threshold
        per-signal raw + p-value
        optional per-pixel heatmap (probability of hallucination at each pixel)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .calibrator import CalibrationArtifact, TaskCalibration
from .config import WarpScoreConfig
from .fuser import SignalFuser
from .mask import ForegroundMask, InteriorMask
from .matcher import RoMaMatcher
from .signals import Signal, per_pixel_p_value
from .statistics import CertWeightedStatistics, MahalanobisStatistics

try:
    from .adaptive_refs import AdaptiveRefSelector, DinoFeatureExtractor
    _ADAPTIVE_AVAILABLE = True
except ImportError:
    _ADAPTIVE_AVAILABLE = False


@dataclass
class HallucinationResult:
    task: str
    frame: str
    n_refs: int
    H_score: float
    is_hallucination: bool
    p_combined: float
    p_per_signal: dict[str, float]
    raw_per_signal: dict[str, float]
    heatmap: Optional[np.ndarray] = None      # (H, W) ∈ [0, 1]; None if disabled
    auto_detected_task: str = ""
    split: str = ""  # "high" or "low" depending on which query/ subtree the frame came from

    # Diagnostics (kept for backward-compat CSV columns)
    interior_mean_var: float = 0.0
    frame_score_max: float = 0.0
    mean_cert_fg: float = 0.0
    fg_pixel_count: int = 0

    # Per-stage timing in milliseconds
    t_dino_ms:   float = 0.0   # DINOv2 embed + k-NN ref selection
    t_roma_ms:   float = 0.0   # RoMa matching (all refs combined)
    t_signal_ms: float = 0.0   # Signal + p-value computation
    t_total_ms:  float = 0.0   # End-to-end for this frame

    def to_csv_row(self) -> dict:
        row = {
            "task": self.task,
            "frame": self.frame,
            "split": self.split,
            "n_refs": self.n_refs,
            "H_score": round(self.H_score, 6),
            "is_hallucination": int(self.is_hallucination),
            "p_combined": round(self.p_combined, 6),
            "interior_mean_var": round(self.interior_mean_var, 6),
            "frame_score_max": round(self.frame_score_max, 6),
            "mean_cert_fg": round(self.mean_cert_fg, 6),
            "fg_pixel_count": self.fg_pixel_count,
            "t_dino_ms":   round(self.t_dino_ms,   1),
            "t_roma_ms":   round(self.t_roma_ms,   1),
            "t_signal_ms": round(self.t_signal_ms, 1),
            "t_total_ms":  round(self.t_total_ms,  1),
            "auto_detected_task": self.auto_detected_task,
        }
        for name, p in self.p_per_signal.items():
            row[f"p_{name}"] = round(p, 6)
        for name, raw in self.raw_per_signal.items():
            row[f"raw_{name}"] = round(raw, 6)
        # v9 maha columns (present only when precision matrices were available)
        if "ivar_maha" in self.raw_per_signal:
            row["raw_ivar_maha"] = round(self.raw_per_signal["ivar_maha"], 6)
            row["p_ivar_maha"] = round(self.p_per_signal.get("ivar_maha", 1.0), 6)
        if "evidence" in self.raw_per_signal:
            row["raw_evidence"] = round(self.raw_per_signal["evidence"], 6)
            row["p_evidence"] = round(self.p_per_signal.get("evidence", 1.0), 6)
        # Maha-specific H_score if both maha signals present
        if "ivar_maha" in self.p_per_signal and "evidence" in self.p_per_signal:
            from .fuser import CauchyFuser
            maha_p = CauchyFuser().fuse({
                k: v for k, v in self.p_per_signal.items()
                if k in ("ivar_maha", "evidence")
            })
            row["H_score_maha"] = round(1.0 - maha_p, 6)
        if "ivar_px" in self.raw_per_signal:
            row["raw_ivar_px"] = round(self.raw_per_signal["ivar_px"], 6)
        if "peak_maha" in self.raw_per_signal:
            row["raw_peak_maha"] = round(self.raw_per_signal["peak_maha"], 6)
        return row


# =============================================================================
# Detector
# =============================================================================

class WarpVarianceDetector:
    def __init__(
        self,
        config: WarpScoreConfig,
        matcher: RoMaMatcher,
        calib: CalibrationArtifact,
        fuser: SignalFuser,
        signals: list[Signal],
        interior_mask: InteriorMask,
    ) -> None:
        self.config = config
        self.matcher = matcher
        self.calib = calib
        self.fuser = fuser
        self.signals = signals
        self.interior = interior_mask
        self._adaptive_selector: "Optional[AdaptiveRefSelector]" = None
        self._ref_feats_cache: dict[str, np.ndarray] = {}

    def detect(
        self,
        query_path: Path,
        task: Optional[str] = None,
        refs: Optional[list[Path]] = None,
    ) -> HallucinationResult:
        """Run detection on one query frame.

        task: if provided, must be a key in calibration. Otherwise inferred from
              parent directory name.
        refs: explicit ref paths; defaults to ones discovered from
              config.reference_dir / <task> / *.png.
        """
        _t0 = time.perf_counter()
        query_path = Path(query_path)
        task = task or query_path.parent.name
        task_calib = self._resolve_task_calib(task)
        refs = refs if refs is not None else self._discover_refs(task)
        if not refs:
            raise RuntimeError(f"No refs found for task '{task}'")

        # ── Adaptive k-NN ref selection (speed-invariant task-state conditioning) ─
        _t_dino_ms = 0.0
        selector = self._get_adaptive_selector()
        if selector is not None:
            _t_dino0 = time.perf_counter()
            ref_feats = self._get_ref_feats(task, refs)
            query_feat = selector.extractor.extract([query_path])[0]
            top_k_idx = selector.select_for_query(query_feat, ref_feats, self.config.k_per_frame)
            refs = [refs[i] for i in top_k_idx]
            _t_dino_ms = (time.perf_counter() - _t_dino0) * 1000.0

        # ── Load query + masks ────────────────────────────────────────────
        img_bgr, fg_mask, interior_mask = self._load_query(query_path)

        # ── Match against refs ────────────────────────────────────────────
        _t_roma0 = time.perf_counter()
        warps, certs, precisions, ok_refs = self._match_all(query_path, refs, fg_mask)
        _t_roma_ms = (time.perf_counter() - _t_roma0) * 1000.0
        if not warps:
            raise RuntimeError(
                f"All {len(refs)} refs failed to match for {query_path} "
                f"(0/{len(refs)} succeeded)"
            )

        warps_a = np.stack(warps)
        certs_a = np.stack(certs)

        # ── Compute raw signals ───────────────────────────────────────────
        _t_sig0 = time.perf_counter()
        var_map = CertWeightedStatistics.variance_per_pixel(warps_a, certs_a)
        raw: dict[str, float] = {
            "ivar": CertWeightedStatistics.interior_mean(var_map, interior_mask),
            "peak": CertWeightedStatistics.peak_max_z(var_map, interior_mask),
            "cert": CertWeightedStatistics.mean_cert_interior(certs_a, interior_mask),
        }

        # v9: Mahalanobis signals (only when all refs have precision matrices)
        D_map: Optional[np.ndarray] = None
        if precisions and len(precisions) == len(warps):
            precisions_a = np.stack(precisions)  # (K, H, W, 2, 2)
            D_map, logdetΛ_map, _ = MahalanobisStatistics.ivar_per_pixel(warps_a, precisions_a)
            raw["ivar_maha"] = MahalanobisStatistics.interior_mean(D_map, interior_mask)
            raw["evidence"] = MahalanobisStatistics.interior_mean(-logdetΛ_map, interior_mask)
            raw["peak_maha"] = MahalanobisStatistics.peak_max_z(D_map, interior_mask)
            # ivar_px: mean per-pixel empirical p-value using T_null — captures local anomalies
            if task_calib.T_null is not None:
                p_px = per_pixel_p_value(D_map, task_calib.T_null, interior_mask)
                vals = (1.0 - p_px)[interior_mask]
                raw["ivar_px"] = float(vals.mean()) if vals.size > 0 else 0.5

        # ── Empirical p-values via signals ────────────────────────────────
        p_per_signal: dict[str, float] = {}
        for sig in self.signals:
            p_per_signal[sig.name] = sig.p_value(raw[sig.name], task_calib)

        # ── Fuse ──────────────────────────────────────────────────────────
        p_combined = self.fuser.fuse(p_per_signal)
        _t_sig_ms = (time.perf_counter() - _t_sig0) * 1000.0
        H_score = 1.0 - p_combined
        is_h = H_score > self.config.decision_threshold

        # ── Per-pixel heatmap (optional) ──────────────────────────────────
        heatmap = self._compute_heatmap(var_map, interior_mask, task_calib, D_map=D_map)

        return HallucinationResult(
            task=task,
            frame=query_path.stem,
            n_refs=len(ok_refs),
            H_score=float(H_score),
            is_hallucination=bool(is_h),
            p_combined=float(p_combined),
            p_per_signal=p_per_signal,
            raw_per_signal=raw,
            heatmap=heatmap,
            interior_mean_var=raw["ivar"],
            frame_score_max=raw["peak"],
            mean_cert_fg=raw["cert"],
            fg_pixel_count=int(fg_mask.sum()),
            t_dino_ms   = _t_dino_ms,
            t_roma_ms   = _t_roma_ms,
            t_signal_ms = _t_sig_ms,
            t_total_ms  = (time.perf_counter() - _t0) * 1000.0,
        )

    # ─────────────────────────────────────────────────────────────────────────

    def _resolve_task_calib(self, task: str) -> TaskCalibration:
        if task in self.calib.tasks:
            return self.calib.tasks[task]
        print(f"[detect] task '{task}' not in calibration; falling back to global")
        return self.calib.global_

    def _discover_refs(self, task: str) -> list[Path]:
        task_dir = self.config.reference_dir / task
        if not task_dir.exists():
            return []
        return sorted(task_dir.glob("*.png"))

    def _get_adaptive_selector(self) -> "Optional[AdaptiveRefSelector]":
        if not getattr(self.config, "adaptive_ref_selector", False):
            return None
        if not _ADAPTIVE_AVAILABLE:
            raise ImportError(
                "adaptive_ref_selector=True but warp_score.adaptive_refs not available. "
                "Check that torch and DINOv2 are installed."
            )
        if self._adaptive_selector is None:
            self._adaptive_selector = AdaptiveRefSelector(
                DinoFeatureExtractor(getattr(self.config, "dino_model", "dinov2_vits14"))
            )
        return self._adaptive_selector

    def _get_ref_feats(self, task: str, refs: list[Path]) -> np.ndarray:
        if task not in self._ref_feats_cache:
            selector = self._get_adaptive_selector()
            dino_cache_dir = getattr(self.config, "dino_cache_dir", None) or (
                self.config.artifacts_dir / "dino_cache"
            )
            feats = selector.load_cache(task, refs, dino_cache_dir)
            if feats is None:
                feats = selector.build_cache(task, refs, dino_cache_dir)
            self._ref_feats_cache[task] = feats
        return self._ref_feats_cache[task]

    def _load_query(
        self, query_path: Path,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        img_bgr = cv2.imread(str(query_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read query: {query_path}")
        img_bgr = cv2.resize(
            img_bgr,
            (self.config.vis_size, self.config.vis_size),
            interpolation=cv2.INTER_NEAREST,
        )
        fg_mask = ForegroundMask.from_image(img_bgr)
        if not fg_mask.any():
            raise RuntimeError(f"Empty foreground for {query_path}")
        interior_mask = self.interior.apply(fg_mask)
        if not interior_mask.any():
            raise RuntimeError(f"Empty interior after erosion for {query_path}")
        return img_bgr, fg_mask, interior_mask

    def _match_all(
        self, query_path: Path, refs: list[Path], fg_mask: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[Path]]:
        warps: list[np.ndarray] = []
        certs: list[np.ndarray] = []
        precisions: list[np.ndarray] = []
        ok_refs: list[Path] = []
        n_failed = 0
        for ref_path in refs:
            try:
                m = self.matcher.match(query_path, ref_path, fg_mask=fg_mask)
                warps.append(m.warp)
                certs.append(m.cert)
                if m.precision is not None:
                    precisions.append(m.precision)
                ok_refs.append(ref_path)
            except Exception as e:
                n_failed += 1
                print(f"[detect] match failed {Path(ref_path).name}: {e}")
        if n_failed:
            print(
                f"[detect] {len(ok_refs)}/{len(refs)} refs matched successfully "
                f"({n_failed} failed) for {query_path.name}"
            )
        return warps, certs, precisions, ok_refs

    def _compute_heatmap(
        self,
        var_map: np.ndarray,
        interior_mask: np.ndarray,
        task_calib: TaskCalibration,
        D_map: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        if not self.config.save_heatmaps:
            return None
        # v9: prefer D_map-based per-pixel calibration when available
        if D_map is not None and task_calib.T_null is not None:
            p_map = per_pixel_p_value(D_map, task_calib.T_null, interior_mask)
            return (1.0 - p_map).astype(np.float32) * interior_mask.astype(np.float32)
        if task_calib.per_pixel_var is None:
            # Fallback: within-frame z normalized via sigmoid as a "heatmap proxy"
            z = CertWeightedStatistics.within_frame_zscore(var_map, interior_mask)
            return _sigmoid(z) * interior_mask.astype(np.float32)
        # Per-pixel empirical: 1 - p(pixel)
        p_map = per_pixel_p_value(var_map, task_calib.per_pixel_var, interior_mask)
        return (1.0 - p_map).astype(np.float32) * interior_mask.astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
