"""VideoDetector — video-level hallucination detection via Cauchy frame aggregation.

Usage::

    detector = VideoDetector.from_config(cfg, matcher, calib, fuser, signals, interior)
    result   = detector.detect_video(video_path, fps=2.0)
    print(result.H_video, result.decision)

Aggregation strategy:
    Per-frame p_combined values are aggregated with the Cauchy combination test
    (Liu & Xie 2020), which is valid under arbitrary temporal dependence — frames
    from the same video are highly correlated, making Fisher/Stouffer inappropriate.

Cache:
    Results are written to <artifacts_dir>/video_cache/<sha256_key[:16]>.json.
    Cache key includes video_path, config path+mtime, calibration path+mtime, and fps.
    Any change in config or calibration automatically invalidates the cache.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .calibrator import CalibrationArtifact
from .config import WarpScoreConfig
from .detector import HallucinationResult, WarpVarianceDetector
from .fuser import CauchyFuser, SignalFuser
from .mask import InteriorMask
from .matcher import RoMaMatcher
from .sam_segmenter import VideoFrameSegmenter
from .signals import Signal


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class FrameResult:
    """Lightweight per-frame summary stored in the cache JSON."""
    seq_idx:    int
    frame_idx:  int
    ts_s:       float
    H_score:    float
    p_combined: float
    n_refs:     int
    t_dino_ms:  float
    t_roma_ms:  float
    t_signal_ms: float
    t_total_ms: float
    raw:        dict[str, float] = field(default_factory=dict)
    p_per:      dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "seq_idx":    self.seq_idx,
            "frame_idx":  self.frame_idx,
            "ts_s":       round(self.ts_s, 3),
            "H_score":    round(self.H_score, 6),
            "p_combined": round(self.p_combined, 6),
            "n_refs":     self.n_refs,
            "t_dino_ms":  round(self.t_dino_ms, 1),
            "t_roma_ms":  round(self.t_roma_ms, 1),
            "t_signal_ms": round(self.t_signal_ms, 1),
            "t_total_ms": round(self.t_total_ms, 1),
            "raw":        {k: round(v, 6) for k, v in self.raw.items()},
            "p_per":      {k: round(v, 6) for k, v in self.p_per.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FrameResult":
        return cls(
            seq_idx=d["seq_idx"],
            frame_idx=d["frame_idx"],
            ts_s=d["ts_s"],
            H_score=d["H_score"],
            p_combined=d["p_combined"],
            n_refs=d["n_refs"],
            t_dino_ms=d["t_dino_ms"],
            t_roma_ms=d["t_roma_ms"],
            t_signal_ms=d["t_signal_ms"],
            t_total_ms=d["t_total_ms"],
            raw=d.get("raw", {}),
            p_per=d.get("p_per", {}),
        )


@dataclass
class VideoResult:
    """Video-level hallucination detection result."""
    video_path:    Path
    task:          str
    n_frames:      int
    fps_sampled:   float
    frames:        list[FrameResult]
    H_video:       float
    p_video:       float
    decision:      str          # "hallucinated" | "clean" | "uncertain"
    threshold:     float
    t_total_s:     float        # wall time for this video (excluding startup)
    cache_key:     str
    from_cache:    bool = False

    @property
    def mean_H(self) -> float:
        return float(np.mean([f.H_score for f in self.frames])) if self.frames else 0.5

    @property
    def mean_roma_ms(self) -> float:
        return float(np.mean([f.t_roma_ms for f in self.frames])) if self.frames else 0.0

    @property
    def mean_total_ms(self) -> float:
        return float(np.mean([f.t_total_ms for f in self.frames])) if self.frames else 0.0

    def to_dict(self) -> dict:
        return {
            "video_path":  str(self.video_path),
            "task":        self.task,
            "sampling":    {"fps": self.fps_sampled, "n_frames_extracted": self.n_frames},
            "frames":      [f.to_dict() for f in self.frames],
            "video": {
                "H_video":   round(self.H_video, 6),
                "p_video":   round(self.p_video, 6),
                "decision":  self.decision,
                "threshold": self.threshold,
                "n_frames":  self.n_frames,
                "mean_H":    round(self.mean_H, 6),
            },
            "timings": {
                "t_total_s":      round(self.t_total_s, 2),
                "mean_frame_ms":  round(self.mean_total_ms, 1),
                "mean_roma_ms":   round(self.mean_roma_ms, 1),
            },
            "cache_key":  self.cache_key,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict, cache_key: str) -> "VideoResult":
        frames = [FrameResult.from_dict(f) for f in d["frames"]]
        v = d["video"]
        t = d.get("timings", {})
        return cls(
            video_path=Path(d["video_path"]),
            task=d["task"],
            n_frames=v["n_frames"],
            fps_sampled=d["sampling"]["fps"],
            frames=frames,
            H_video=v["H_video"],
            p_video=v["p_video"],
            decision=v["decision"],
            threshold=v["threshold"],
            t_total_s=t.get("t_total_s", 0.0),
            cache_key=cache_key,
            from_cache=True,
        )


# ── Main class ────────────────────────────────────────────────────────────────

class VideoDetector:
    """Detect video-level hallucination by aggregating per-frame scores.

    Wraps WarpVarianceDetector to process individual frames, then combines
    their p_combined values using the Cauchy combination test (valid under
    temporal dependence between consecutive frames).
    """

    _DECISION_THRESHOLD = 0.5   # H_video > this → hallucinated

    def __init__(
        self,
        frame_detector: WarpVarianceDetector,
        config: WarpScoreConfig,
        calib_path: Path,
        segmenter: Optional[VideoFrameSegmenter] = None,
    ) -> None:
        self.frame_detector = frame_detector
        self.config = config
        self.calib_path = Path(calib_path)
        self._cauchy = CauchyFuser()
        self._segmenter = segmenter

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        config: WarpScoreConfig,
        matcher: RoMaMatcher,
        calib: CalibrationArtifact,
        fuser: SignalFuser,
        signals: list[Signal],
        interior: InteriorMask,
        segmenter: Optional[VideoFrameSegmenter] = None,
    ) -> "VideoDetector":
        frame_det = WarpVarianceDetector(
            config=config,
            matcher=matcher,
            calib=calib,
            fuser=fuser,
            signals=signals,
            interior_mask=interior,
        )
        return cls(
            frame_detector=frame_det,
            config=config,
            calib_path=config.calib_path,
            segmenter=segmenter,
        )

    @classmethod
    def load_segmenter(
        cls,
        model_id: str = "facebook/sam3",
        threshold: float = 0.3,
    ) -> VideoFrameSegmenter:
        """Convenience factory: create a VideoFrameSegmenter with defaults."""
        return VideoFrameSegmenter(model_id=model_id, threshold=threshold)

    # ── Public API ────────────────────────────────────────────────────────────

    def detect_video(
        self,
        video_path: Path | str,
        fps: float = 2.0,
        n_frames: Optional[int] = None,
        task: Optional[str] = None,
        use_cache: bool = True,
    ) -> VideoResult:
        """Detect hallucination in a video file.

        Args:
            video_path: path to .mp4 (or any cv2-readable video)
            fps: sample rate in frames per second (ignored if n_frames given)
            n_frames: if set, uniformly sample exactly this many frames
            task: override task name (defaults to parent dir, then stem fallback)
            use_cache: read/write JSON cache
        """
        video_path = Path(video_path).expanduser().resolve()
        task = task or self._infer_task(video_path)
        cache_key = self._make_cache_key(video_path, fps if n_frames is None else None, n_frames, task)

        if use_cache:
            cached = self._load_cache(cache_key)
            if cached is not None:
                return cached

        # Early exit if task has no reference images — avoids wasting SAM3 time
        refs = self.frame_detector._discover_refs(task)
        if not refs:
            print(f"  [skip] task '{task}': no reference images found")
            t_empty = time.perf_counter()
            result = VideoResult(
                video_path=video_path,
                task=task,
                n_frames=0,
                fps_sampled=fps if n_frames is None else 0.0,
                frames=[],
                H_video=0.5,
                p_video=0.5,
                decision="uncertain",
                threshold=self._DECISION_THRESHOLD,
                t_total_s=time.perf_counter() - t_empty,
                cache_key=cache_key,
            )
            if use_cache:
                self._write_cache(cache_key, result)
            return result

        t_start = time.perf_counter()
        raw_frames = self._extract_frames(video_path, fps=fps, n_frames=n_frames)
        fps_actual = fps if n_frames is None else len(raw_frames) / max(1, len(raw_frames))

        frame_results: list[FrameResult] = []
        tmp_dir = Path("/tmp/_video_detector")
        tmp_dir.mkdir(exist_ok=True)

        for seq_i, (frame_idx, ts_s, bgr) in enumerate(raw_frames):
            if self._segmenter is not None:
                bgr = self._segmenter.segment_frame(bgr)
            tmp_png = tmp_dir / f"frame_{seq_i:04d}.png"
            ok = cv2.imwrite(str(tmp_png), bgr)
            if not ok or not tmp_png.exists():
                print(f"  [skip] frame {frame_idx} ({ts_s:.2f}s): imwrite failed")
                continue

            try:
                det: HallucinationResult = self.frame_detector.detect(
                    query_path=tmp_png,
                    task=task,
                )
            except Exception as e:
                print(f"  [skip] frame {frame_idx} ({ts_s:.2f}s): {type(e).__name__}: {e}")
                continue

            frame_results.append(FrameResult(
                seq_idx=seq_i,
                frame_idx=frame_idx,
                ts_s=ts_s,
                H_score=det.H_score,
                p_combined=det.p_combined,
                n_refs=det.n_refs,
                t_dino_ms=det.t_dino_ms,
                t_roma_ms=det.t_roma_ms,
                t_signal_ms=det.t_signal_ms,
                t_total_ms=det.t_total_ms,
                raw=det.raw_per_signal,
                p_per=det.p_per_signal,
            ))

            self._print_frame_row(seq_i + 1, ts_s, det)

        t_total_s = time.perf_counter() - t_start

        # Aggregate with Cauchy combination
        H_video, p_video = self._aggregate(frame_results)
        decision = self._decide(H_video)

        result = VideoResult(
            video_path=video_path,
            task=task,
            n_frames=len(frame_results),
            fps_sampled=fps_actual,
            frames=frame_results,
            H_video=H_video,
            p_video=p_video,
            decision=decision,
            threshold=self._DECISION_THRESHOLD,
            t_total_s=t_total_s,
            cache_key=cache_key,
        )

        if use_cache:
            self._write_cache(cache_key, result)

        return result

    # ── Frame extraction ──────────────────────────────────────────────────────

    def _extract_frames(
        self,
        video_path: Path,
        fps: float,
        n_frames: Optional[int],
    ) -> list[tuple[int, float, np.ndarray]]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        src_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if n_frames is not None:
            # Uniform sampling of exactly n_frames indices
            indices = set(np.linspace(0, n_total - 1, min(n_frames, n_total), dtype=int))
        else:
            step = max(1, round(src_fps / fps))
            indices = set(range(0, n_total, step))

        collected: list[tuple[int, float, np.ndarray]] = []
        idx = 0
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            if idx in indices:
                ts_s = idx / src_fps
                collected.append((idx, ts_s, bgr.copy()))
            idx += 1
        cap.release()
        return collected

    # ── Aggregation ───────────────────────────────────────────────────────────

    def _aggregate(self, frames: list[FrameResult]) -> tuple[float, float]:
        """Cauchy-combine per-frame p_combined → (H_video, p_video)."""
        if not frames:
            return 0.5, 0.5
        if len(frames) == 1:
            p = frames[0].p_combined
            return 1.0 - p, p

        p_dict = {str(i): f.p_combined for i, f in enumerate(frames)}
        p_video = self._cauchy.fuse(p_dict)
        return 1.0 - p_video, p_video

    def _decide(self, H_video: float) -> str:
        if H_video > self._DECISION_THRESHOLD + 0.2:
            return "hallucinated"
        if H_video < self._DECISION_THRESHOLD - 0.2:
            return "clean"
        return "uncertain"

    # ── Task inference ────────────────────────────────────────────────────────

    def _infer_task(self, video_path: Path) -> str:
        """Infer task from directory hierarchy; fall back to stem for flat layouts."""
        candidate = video_path.parent.name
        if self.frame_detector._discover_refs(candidate):
            return candidate
        # Flat layout: video/low/0_Open the box.mp4  → stem = "0_Open the box"
        stem_candidate = video_path.stem
        if self.frame_detector._discover_refs(stem_candidate):
            return stem_candidate
        # Last resort: parent name (detector will use global calibration)
        return candidate

    # ── Cache I/O ─────────────────────────────────────────────────────────────

    def _cache_dir(self) -> Path:
        d = self.config.artifacts_dir / "video_cache"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _make_cache_key(
        self,
        video_path: Path,
        fps: Optional[float],
        n_frames: Optional[int],
        task: str = "",
    ) -> str:
        calib_mtime = self.calib_path.stat().st_mtime if self.calib_path.exists() else 0
        parts = [
            str(video_path),
            task,
            str(fps) if fps is not None else f"n{n_frames}",
            str(calib_mtime),
            "sam" if self._segmenter is not None else "nosam",
        ]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir() / f"{key}.json"

    def _load_cache(self, key: str) -> Optional[VideoResult]:
        p = self._cache_path(key)
        if not p.exists():
            return None
        try:
            with open(p) as f:
                d = json.load(f)
            return VideoResult.from_dict(d, key)
        except Exception as e:
            print(f"  [cache] corrupted ({e}), recomputing")
            return None

    def _write_cache(self, key: str, result: VideoResult) -> None:
        p = self._cache_path(key)
        with open(p, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    # ── Console output ────────────────────────────────────────────────────────

    @staticmethod
    def _print_frame_row(seq_i: int, ts_s: float, det: HallucinationResult) -> None:
        flag = "H" if det.is_hallucination else " "
        print(
            f"  [{flag}] frame {seq_i:>3}  ts={ts_s:>5.2f}s  "
            f"H={det.H_score:.3f}  "
            f"dino={det.t_dino_ms:>6.0f}ms  "
            f"roma={det.t_roma_ms:>6.0f}ms  "
            f"sig={det.t_signal_ms:>5.0f}ms  "
            f"total={det.t_total_ms:>6.0f}ms  "
            f"refs={det.n_refs}"
        )

    @staticmethod
    def print_video_summary(result: VideoResult) -> None:
        bar_width = 40
        fill = int(result.H_video * bar_width)
        bar = "█" * fill + "░" * (bar_width - fill)
        symbol = "⚠" if result.decision == "hallucinated" else ("✓" if result.decision == "clean" else "?")
        print(f"\n  {symbol}  H_video={result.H_video:.4f}  [{bar}]  → {result.decision.upper()}")
        print(f"     frames={result.n_frames}  "
              f"mean_H={result.mean_H:.3f}  "
              f"total={result.t_total_s:.1f}s  "
              f"({result.mean_total_ms:.0f}ms/frame avg)")
        if result.from_cache:
            print(f"     [loaded from cache: {result.cache_key}]")
        else:
            print(f"     [cache written: {result.cache_key}]")
