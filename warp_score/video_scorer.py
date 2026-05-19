"""VideoScorer — production-style API: throw a video in, get a robust H-score.

Usage:
    scorer = VideoScorer.from_cache(cache_dir="paper-physical-gr1/ref_cache")
    result = scorer.score("path/to/query.mp4", n_frames=10)

    print(result.video_h_score)            # robust aggregate over frames
    print(result.per_frame)                # list of {frame_idx, h_score, sig_breakdown}

The reference cache is built ONCE offline (see ReferenceCacheBuilder) and
contains:
    pool_feats.npy         (M, D)   DINOv2 CLS embeddings, L2-normalized
    pool_paths.txt                  one path per line, matching pool_feats rows
    cycle_null.npz                  sorted null arrays (cycle_mean, cycle_peak)
    nn_jaccard_null.npy             sorted null array of S4 jaccard distances
    appearance_calib.npz            existing CalibrationArtifact dump

The scorer loads the cache, samples frames from the input video, computes
appearance + cycle + NN-jaccard signals, Cauchy-fuses to per-frame H-scores,
then aggregates robustly (trimmed mean + 80th percentile) to video level.

No optical-flow model used — RoMa + DINOv2 only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class FrameScore:
    frame_idx: int
    h_score: float
    p_appear: Optional[float] = None
    p_cycle:  Optional[float] = None
    p_jaccard: Optional[float] = None


@dataclass
class VideoResult:
    video_path: str
    n_frames: int
    video_h_score: float           # robust aggregate (trimmed mean)
    video_h_peak: float            # 80th percentile (sensitive to single bad frame)
    is_hallucination: bool
    threshold: float
    per_frame: list[FrameScore] = field(default_factory=list)
    aggregate_breakdown: dict[str, float] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Math utils
# ─────────────────────────────────────────────────────────────────────────────


def cauchy_combine(ps: list[float]) -> float:
    ps = [p for p in ps if p is not None and 0 < p < 1]
    if not ps:
        return 0.5
    t = float(np.mean([np.tan(np.pi * (0.5 - p)) for p in ps]))
    return float(0.5 - np.arctan(t) / np.pi)


def empirical_p(value: float, sorted_null: np.ndarray) -> float:
    n = sorted_null.size
    if n == 0:
        return 0.5
    rank = int(np.searchsorted(sorted_null, value, side="right"))
    p = (n - rank + 0.5) / (n + 1.0)
    return float(np.clip(p, 1.0 / (n + 1), 1.0 - 1.0 / (n + 1)))


def trimmed_mean(values: np.ndarray, trim: float = 0.1) -> float:
    """Trim `trim` fraction from each tail before averaging."""
    if values.size == 0:
        return 0.0
    sorted_v = np.sort(values)
    n = len(sorted_v)
    k = int(n * trim)
    return float(sorted_v[k : n - k].mean()) if n - 2 * k > 0 else float(sorted_v.mean())


# ─────────────────────────────────────────────────────────────────────────────
# Reference cache schema
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ReferenceCache:
    pool_paths:        list[Path]
    pool_feats:        np.ndarray            # (M, D) L2-normalized
    cycle_null_mean:   np.ndarray            # (N1,) sorted
    cycle_null_peak:   np.ndarray            # (N1,) sorted
    jaccard_null:      Optional[np.ndarray]  # (N2,) sorted — may be None initially
    appearance_null:   Optional[dict[str, Any]] = None  # raw signals from previous pool benchmark

    @classmethod
    def load(cls, cache_dir: Path) -> "ReferenceCache":
        cache_dir = Path(cache_dir)
        feats = np.load(cache_dir / "pool_feats.npy")
        paths = [Path(line.strip()) for line in (cache_dir / "pool_paths.txt").read_text().splitlines()]
        cycle = np.load(cache_dir / "cycle_null.npz")
        jaccard_path = cache_dir / "nn_jaccard_null.npy"
        jaccard = np.load(jaccard_path) if jaccard_path.exists() else None
        return cls(
            pool_paths=paths,
            pool_feats=feats,
            cycle_null_mean=np.sort(cycle["cycle_mean"].astype(np.float32)),
            cycle_null_peak=np.sort(cycle["cycle_peak"].astype(np.float32)),
            jaccard_null=np.sort(jaccard.astype(np.float32)) if jaccard is not None else None,
        )

    def save(self, cache_dir: Path) -> None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_dir / "pool_feats.npy", self.pool_feats.astype(np.float32))
        (cache_dir / "pool_paths.txt").write_text("\n".join(str(p) for p in self.pool_paths))
        np.savez(cache_dir / "cycle_null.npz",
                 cycle_mean=self.cycle_null_mean,
                 cycle_peak=self.cycle_null_peak)
        if self.jaccard_null is not None:
            np.save(cache_dir / "nn_jaccard_null.npy", self.jaccard_null.astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# VideoScorer
# ─────────────────────────────────────────────────────────────────────────────


class VideoScorer:
    """Throw a video in, get a robust hallucination score out."""

    def __init__(
        self,
        cache: ReferenceCache,
        device: str = "cuda",
        k_per_frame: int = 50,
        threshold: float = 0.5,
        cert_floor: float = 0.1,
        sam_segment: bool = True,
    ) -> None:
        """
        sam_segment: when True, SAM3-segments each frame before scoring so the
                     test frames match the reference pool's segmented style
                     (bg = (127, 127, 127)). The reference pool was SAM3-built
                     so leaving this False produces unreliable DINOv2 NN.
        """
        self.cache = cache
        self.device = device
        self.k_per_frame = k_per_frame
        self.threshold = threshold
        self.cert_floor = cert_floor
        self.sam_segment = sam_segment
        self._matcher = None
        self._dino = None
        self._sam = None

    @classmethod
    def from_cache(cls, cache_dir: Path | str, **kwargs) -> "VideoScorer":
        return cls(cache=ReferenceCache.load(Path(cache_dir)), **kwargs)

    # ─────────────────────────────────────────────────────────────────────

    def _get_matcher(self):
        if self._matcher is None:
            from .matcher import RoMaMatcher
            self._matcher = RoMaMatcher(
                setting="turbo", device=self.device, use_precision=True, vis_size=224,
            )
            self._matcher._load_model()
        return self._matcher

    def _get_dino(self):
        if self._dino is None:
            from .adaptive_refs import DinoFeatureExtractor
            self._dino = DinoFeatureExtractor("dinov2_vits14")
            self._dino._load()
        return self._dino

    def _get_sam(self):
        if self._sam is None and self.sam_segment:
            from .sam_segmenter import VideoFrameSegmenter
            self._sam = VideoFrameSegmenter()
        return self._sam

    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_spread_frames(mp4: Path, n: int) -> list[tuple[int, np.ndarray]]:
        cap = cv2.VideoCapture(str(mp4))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            cap.release()
            return []
        indices = np.linspace(0, total - 1, min(n, total), dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, bgr = cap.read()
            if ok:
                frames.append((int(idx), bgr))
        cap.release()
        return frames

    # ─────────────────────────────────────────────────────────────────────

    def score(
        self,
        video_path: Path | str,
        n_frames: int = 10,
        frame_dir: Optional[Path] = None,
    ) -> VideoResult:
        """Sample frames from `video_path`, compute all signals, aggregate.

        Args:
            video_path: input .mp4
            n_frames:   how many frames to sample (np.linspace across the video)
            frame_dir:  optional override — if given, read .png files from here
                        instead of decoding the mp4 (useful when frames are
                        already SAM3-segmented).

        Returns:
            VideoResult with per-frame H + video-level robust aggregate.
        """
        from .matcher import RoMaMatcher  # noqa: F401
        from .temporal_signals import cycle_signal
        from .nn_consistency import nn_set_jaccard_distance

        video_path = Path(video_path)

        # ── Frame collection
        if frame_dir is not None:
            pngs = sorted(Path(frame_dir).glob("*.png"))[:n_frames]
            frames = []
            for i, p in enumerate(pngs):
                bgr = cv2.imread(str(p))
                if bgr is not None:
                    frames.append((i, bgr, p))
        else:
            raw = self._extract_spread_frames(video_path, n_frames)
            frames = [(idx, bgr, None) for idx, bgr in raw]

        if len(frames) < 2:
            raise RuntimeError(f"Need at least 2 frames; got {len(frames)} from {video_path}")

        # ── Optional SAM3 segmentation so test frames match pool style
        if self.sam_segment and frame_dir is None:
            sam = self._get_sam()
            frames = [(idx, sam.segment_frame(bgr), src) for (idx, bgr, src) in frames]

        # ── Cache frames to tmp .png so RoMa can load them
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="warpdyn_"))
        frame_paths: list[Path] = []
        for i, (idx, bgr, src) in enumerate(frames):
            if src is not None and not self.sam_segment:
                # Already on disk and we're not modifying — point to it
                frame_paths.append(src)
            else:
                p = tmp / f"frame_{i:04d}.png"
                cv2.imwrite(str(p), bgr)
                frame_paths.append(p)

        # ── DINOv2 embeddings (for k-NN selection AND for S4 jaccard)
        dino = self._get_dino()
        q_feats = dino.extract(frame_paths)  # (T, D) L2-normalized

        # ── Per-frame signal computation
        matcher = self._get_matcher()
        per_frame_results: list[FrameScore] = []

        for t in range(len(frames)):
            cur_path = frame_paths[t]
            nxt_path = frame_paths[t + 1] if t + 1 < len(frames) else None

            # S1: appearance — k-NN warp residual against pool top-K
            sims = self.cache.pool_feats @ q_feats[t]
            k = min(self.k_per_frame, len(self.cache.pool_feats))
            top_idx = np.argpartition(sims, -k)[-k:]
            top_refs = [self.cache.pool_paths[i] for i in top_idx]
            # Lightweight appearance signal: median warp magnitude across refs
            match_results = matcher.match_batch(cur_path, top_refs)
            warp_mags = []
            for mr in match_results:
                w = mr.warp
                # Rough magnitude of normalized warp away from identity grid
                # warp ∈ [-1, 1]; identity would be a linspace grid → measure
                # mean displacement from identity normalized to pixel units.
                H, W = w.shape[:2]
                yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
                ident_x = (xx / max(W - 1, 1)) * 2 - 1
                ident_y = (yy / max(H - 1, 1)) * 2 - 1
                disp = np.sqrt((w[..., 0] - ident_x) ** 2 + (w[..., 1] - ident_y) ** 2)
                warp_mags.append(float(disp.mean()))
            appear_raw = float(np.median(warp_mags))
            # No calibrated appearance null in this lightweight path, so we
            # convert via a rank-based proxy: scale appearance_raw by cycle
            # null heuristically. Better: load real appearance null from
            # cache. For now, p_appear stays None and we rely on S2+S4.
            p_appear = None

            # S2: cycle (only when there's a next frame)
            p_cycle = None
            if nxt_path is not None:
                fwd = matcher.match(cur_path, nxt_path)
                bwd = matcher.match(nxt_path, cur_path)
                sig = cycle_signal(fwd.warp, bwd.warp, cert_fwd=fwd.cert,
                                   cert_floor=self.cert_floor)
                p_cm = empirical_p(sig["mean"], self.cache.cycle_null_mean)
                p_cp = empirical_p(sig["peak"], self.cache.cycle_null_peak)
                p_cycle = cauchy_combine([p_cm, p_cp])

            # S4: NN-set jaccard
            p_jaccard = None
            if nxt_path is not None and self.cache.jaccard_null is not None:
                jacc = nn_set_jaccard_distance(
                    q_feats[t], q_feats[t + 1], self.cache.pool_feats,
                    k=self.k_per_frame,
                )
                p_jaccard = empirical_p(jacc, self.cache.jaccard_null)

            p_combined = cauchy_combine([p_cycle, p_jaccard])
            h_score = 1.0 - p_combined
            per_frame_results.append(FrameScore(
                frame_idx=frames[t][0],
                h_score=h_score,
                p_appear=p_appear,
                p_cycle=p_cycle,
                p_jaccard=p_jaccard,
            ))

        # ── Robust video-level aggregate
        h_values = np.array([f.h_score for f in per_frame_results], dtype=np.float32)
        video_h_robust = trimmed_mean(h_values, trim=0.1)
        video_h_peak = float(np.percentile(h_values, 80))
        # Final decision: use robust mean
        is_hallu = video_h_robust > self.threshold

        return VideoResult(
            video_path=str(video_path),
            n_frames=len(frames),
            video_h_score=video_h_robust,
            video_h_peak=video_h_peak,
            is_hallucination=bool(is_hallu),
            threshold=self.threshold,
            per_frame=per_frame_results,
            aggregate_breakdown={
                "mean":         float(h_values.mean()),
                "trimmed_mean": video_h_robust,
                "median":       float(np.median(h_values)),
                "p80":          video_h_peak,
                "max":          float(h_values.max()),
            },
        )
