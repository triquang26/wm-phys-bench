"""Tests for HeatmapPlotter in warp_score/visualizer.py.

Uses the Agg backend to avoid display issues in headless environments.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # must be set before any other matplotlib import

import numpy as np
import cv2
import pytest
from pathlib import Path

from warp_score.visualizer import HeatmapPlotter
from warp_score.detector import HallucinationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(heatmap=None) -> HallucinationResult:
    """Construct a minimal HallucinationResult for testing."""
    return HallucinationResult(
        task="test_task",
        frame="frame_0001",
        n_refs=3,
        H_score=0.42,
        is_hallucination=False,
        p_combined=0.58,
        p_per_signal={"ivar": 0.6, "peak": 0.7, "cert": 0.5},
        raw_per_signal={"ivar": 0.003, "peak": 1.2, "cert": 0.85},
        heatmap=heatmap,
    )


def _make_query_png(tmp_path: Path, name: str = "query.png") -> Path:
    """Write a synthetic 224×224 RGB image to disk and return its path."""
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img[50:174, 50:174] = [200, 100, 50]  # coloured foreground block
    path = tmp_path / name
    cv2.imwrite(str(path), img)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_heatmap_plotter_saves_figure(tmp_path):
    """plot() with a valid heatmap saves a non-empty PNG file."""
    img_path = _make_query_png(tmp_path)
    heatmap = np.zeros((224, 224), dtype=np.float32)
    result = _make_result(heatmap=heatmap)

    plotter = HeatmapPlotter(vis_size=224)
    out_path = tmp_path / "out.png"
    plotter.plot(img_path, result, save_to=out_path)

    assert out_path.exists(), "Output PNG was not created"
    assert out_path.stat().st_size > 0, "Output PNG is empty"


def test_heatmap_plotter_no_heatmap(tmp_path):
    """plot() with heatmap=None should still save without error."""
    img_path = _make_query_png(tmp_path)
    result = _make_result(heatmap=None)

    plotter = HeatmapPlotter(vis_size=224)
    out_path = tmp_path / "out_no_heatmap.png"
    plotter.plot(img_path, result, save_to=out_path)

    assert out_path.exists(), "Output PNG was not created when heatmap is None"
    assert out_path.stat().st_size > 0, "Output PNG is empty when heatmap is None"


def test_heatmap_plotter_no_save(tmp_path):
    """plot() without save_to should not raise (plt.show() is a no-op under Agg)."""
    import matplotlib.pyplot as plt
    # Patch plt.show so it doesn't block or error in headless mode
    original_show = plt.show

    shown = []

    def _fake_show(*args, **kwargs):
        shown.append(True)

    plt.show = _fake_show
    try:
        img_path = _make_query_png(tmp_path)
        result = _make_result(heatmap=None)
        plotter = HeatmapPlotter(vis_size=224)
        # Should not raise; save_to=None triggers plt.show()
        plotter.plot(img_path, result, save_to=None)
    finally:
        plt.show = original_show
