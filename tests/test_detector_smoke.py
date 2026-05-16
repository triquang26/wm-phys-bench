"""Smoke tests for WarpVarianceDetector — mocks RoMaMatcher to avoid GPU."""
from __future__ import annotations

import numpy as np
import pytest
import cv2
from pathlib import Path
from unittest.mock import MagicMock, patch

from warp_score.calibrator import CalibrationArtifact, TaskCalibration
from warp_score.config import WarpScoreConfig
from warp_score.detector import WarpVarianceDetector, HallucinationResult
from warp_score.fuser import build_fuser
from warp_score.mask import InteriorMask
from warp_score.matcher import MatchResult
from warp_score.signals import build_signals


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

VIS = 32       # small vis_size to keep tests fast
EROSION_K = 3  # small erosion so interior survives on small images


def _make_query_png(tmp_path: Path, name: str = "query.png") -> Path:
    """Create a 32×32 BGR image: center 20×20 block is white foreground,
    border is (127,127,127) background — matches ForegroundMask convention."""
    img = np.full((VIS, VIS, 3), 127, dtype=np.uint8)
    # Large enough foreground so erosion_k=3 leaves a non-empty interior
    img[6:26, 6:26] = 200
    path = tmp_path / name
    cv2.imwrite(str(path), img)
    return path


def _make_task_calib(name: str, n: int = 10) -> TaskCalibration:
    rng = np.random.default_rng(0)
    return TaskCalibration(
        task=name,
        n_refs=n,
        ivar_dist=np.sort(rng.uniform(0, 0.01, n).astype(np.float32)),
        peak_dist=np.sort(rng.uniform(0, 3.0, n).astype(np.float32)),
        cert_dist=np.sort(rng.uniform(0.5, 1.0, n).astype(np.float32)),
    )


def _make_global_calib(n: int = 10) -> TaskCalibration:
    rng = np.random.default_rng(1)
    return TaskCalibration(
        task="__global__",
        n_refs=n,
        ivar_dist=np.sort(rng.uniform(0, 0.01, n).astype(np.float32)),
        peak_dist=np.sort(rng.uniform(0, 3.0, n).astype(np.float32)),
        cert_dist=np.sort(rng.uniform(0.5, 1.0, n).astype(np.float32)),
    )


def _make_artifact(task_name: str) -> CalibrationArtifact:
    tc = _make_task_calib(task_name)
    return CalibrationArtifact(
        tasks={task_name: tc},
        global_=_make_global_calib(),
        config_snapshot={},
        created_at="2025-01-01T00:00:00",
    )


def _make_fixed_match_result(seed: int = 0) -> MatchResult:
    """Warp + cert arrays at vis_size × vis_size."""
    rng = np.random.default_rng(seed)
    warp = rng.uniform(-0.1, 0.1, (VIS, VIS, 2)).astype(np.float32)
    cert = rng.uniform(0.6, 1.0, (VIS, VIS)).astype(np.float32)
    return MatchResult(warp=warp, cert=cert)


def _make_detector(matcher_mock, calib: CalibrationArtifact,
                   save_heatmaps: bool = False) -> WarpVarianceDetector:
    cfg = WarpScoreConfig(vis_size=VIS, save_heatmaps=save_heatmaps,
                          erosion_k=EROSION_K, device="cpu")
    return WarpVarianceDetector(
        config=cfg,
        matcher=matcher_mock,
        calib=calib,
        fuser=build_fuser("stouffer"),
        signals=build_signals(("ivar", "peak", "cert")),
        interior_mask=InteriorMask(erosion_k=EROSION_K),
    )


# ---------------------------------------------------------------------------
# Test 1: detector creates a HallucinationResult with H_score in [0,1]
# ---------------------------------------------------------------------------

def test_detector_creates_hallucination_result(tmp_path):
    """detect() returns a HallucinationResult with H_score ∈ [0, 1]."""
    task_name = "demo_task"
    query_path = _make_query_png(tmp_path)

    # Create 3 ref PNGs on disk (detector does _discover_refs from high_dir)
    high_dir = tmp_path / "high" / task_name
    high_dir.mkdir(parents=True)
    ref_paths = []
    for i in range(3):
        p = _make_query_png(high_dir, f"ref_{i:02d}.png")
        ref_paths.append(p)

    # Mock matcher: always returns the same small, low-variance warp
    matcher = MagicMock()
    matcher.match.return_value = _make_fixed_match_result(seed=0)

    calib = _make_artifact(task_name)
    cfg = WarpScoreConfig(
        vis_size=VIS,
        save_heatmaps=False,
        erosion_k=EROSION_K,
        device="cpu",
        high_dir=tmp_path / "high",
    )
    detector = WarpVarianceDetector(
        config=cfg,
        matcher=matcher,
        calib=calib,
        fuser=build_fuser("stouffer"),
        signals=build_signals(("ivar", "peak", "cert")),
        interior_mask=InteriorMask(erosion_k=EROSION_K),
    )

    result = detector.detect(query_path, task=task_name, refs=ref_paths)

    assert isinstance(result, HallucinationResult)
    assert 0.0 <= result.H_score <= 1.0
    assert result.task == task_name


# ---------------------------------------------------------------------------
# Test 2: H_score is high when warp variance is extreme
# ---------------------------------------------------------------------------

def test_detector_h_score_high_when_warp_var_extreme(tmp_path):
    """Highly variable warps across refs should produce H_score > calib threshold."""
    task_name = "extreme_task"
    query_path = _make_query_png(tmp_path)

    high_dir = tmp_path / "high" / task_name
    high_dir.mkdir(parents=True)
    ref_paths = [_make_query_png(high_dir, f"ref_{i:02d}.png") for i in range(4)]

    call_count = 0

    def variable_match(*args, **kwargs):
        nonlocal call_count
        rng = np.random.default_rng(call_count * 1000)
        call_count += 1
        # Each call returns wildly different warps spanning the full [-1, 1] range
        warp = rng.uniform(-1.0, 1.0, (VIS, VIS, 2)).astype(np.float32)
        # Small but nonzero cert so variance is computed; also anomalous in
        # the "low" direction vs calibration cert_dist ∈ [0.9, 1.0]
        cert = np.full((VIS, VIS), 0.1, dtype=np.float32)
        return MatchResult(warp=warp, cert=cert)

    matcher = MagicMock()
    matcher.match.side_effect = variable_match

    # Build calibration with very tight (low-variance) null distributions so
    # any real variance looks extreme
    rng = np.random.default_rng(99)
    n = 20
    tc = TaskCalibration(
        task=task_name,
        n_refs=n,
        ivar_dist=np.sort(rng.uniform(0, 1e-4, n).astype(np.float32)),
        peak_dist=np.sort(rng.uniform(0, 0.5, n).astype(np.float32)),
        cert_dist=np.sort(rng.uniform(0.9, 1.0, n).astype(np.float32)),
    )
    calib = CalibrationArtifact(
        tasks={task_name: tc},
        global_=_make_global_calib(),
        config_snapshot={},
        created_at="2025-01-01T00:00:00",
    )

    cfg = WarpScoreConfig(
        vis_size=VIS, save_heatmaps=False, erosion_k=EROSION_K, device="cpu",
        high_dir=tmp_path / "high",
    )
    detector = WarpVarianceDetector(
        config=cfg,
        matcher=matcher,
        calib=calib,
        fuser=build_fuser("stouffer"),
        signals=build_signals(("ivar", "peak", "cert")),
        interior_mask=InteriorMask(erosion_k=EROSION_K),
    )

    result = detector.detect(query_path, task=task_name, refs=ref_paths)
    assert result.H_score > cfg.decision_threshold, (
        f"Expected H_score > {cfg.decision_threshold}, got {result.H_score}"
    )


# ---------------------------------------------------------------------------
# Test 3: to_csv_row has the expected keys
# ---------------------------------------------------------------------------

def test_detector_csv_row_keys(tmp_path):
    """to_csv_row() contains the required columns including per-signal p_ keys."""
    task_name = "csv_task"
    query_path = _make_query_png(tmp_path)

    high_dir = tmp_path / "high" / task_name
    high_dir.mkdir(parents=True)
    ref_paths = [_make_query_png(high_dir, f"ref_{i:02d}.png") for i in range(2)]

    matcher = MagicMock()
    matcher.match.return_value = _make_fixed_match_result(seed=7)

    calib = _make_artifact(task_name)
    cfg = WarpScoreConfig(
        vis_size=VIS, save_heatmaps=False, erosion_k=EROSION_K, device="cpu",
        high_dir=tmp_path / "high",
    )
    detector = WarpVarianceDetector(
        config=cfg,
        matcher=matcher,
        calib=calib,
        fuser=build_fuser("stouffer"),
        signals=build_signals(("ivar", "peak", "cert")),
        interior_mask=InteriorMask(erosion_k=EROSION_K),
    )

    result = detector.detect(query_path, task=task_name, refs=ref_paths)
    row = result.to_csv_row()

    expected_keys = {"frame", "task", "H_score", "is_hallucination",
                     "p_ivar", "p_peak", "p_cert"}
    assert expected_keys.issubset(row.keys()), (
        f"Missing keys: {expected_keys - row.keys()}"
    )
