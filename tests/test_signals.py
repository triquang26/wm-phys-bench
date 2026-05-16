"""Unit tests for signals + fuser. Run with: pytest tests/"""
from __future__ import annotations

import numpy as np
import pytest

from warp_score.calibrator import TaskCalibration
from warp_score.fuser import FisherFuser, MaxFuser, StoufferFuser
from warp_score.signals import (
    CertSignal,
    IvarSignal,
    PeakSignal,
    _empirical_p,
    build_signals,
    per_pixel_p_value,
)


def _make_calib(ivar_vals, peak_vals, cert_vals) -> TaskCalibration:
    return TaskCalibration(
        task="test",
        n_refs=len(ivar_vals),
        ivar_dist=np.sort(np.array(ivar_vals, dtype=np.float32)),
        peak_dist=np.sort(np.array(peak_vals, dtype=np.float32)),
        cert_dist=np.sort(np.array(cert_vals, dtype=np.float32)),
    )


# ────────────────────────────────────────────────────────────────────────────
# _empirical_p
# ────────────────────────────────────────────────────────────────────────────

def test_empirical_p_high_direction_extreme():
    dist = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    # Value above all → only Laplace smoothing contributes
    assert _empirical_p(10.0, dist, "high") == pytest.approx(1 / 6)


def test_empirical_p_high_direction_median():
    dist = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    # Value at exact median: 3 elements >= 0.3 → (1+3)/6
    assert _empirical_p(0.3, dist, "high") == pytest.approx(4 / 6)


def test_empirical_p_low_direction():
    dist = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    # Value below all → only Laplace
    assert _empirical_p(-1.0, dist, "low") == pytest.approx(1 / 6)
    # Value at max → all elements <= → (1+5)/6
    assert _empirical_p(0.5, dist, "low") == pytest.approx(6 / 6)


# ────────────────────────────────────────────────────────────────────────────
# Signal classes
# ────────────────────────────────────────────────────────────────────────────

def test_ivar_signal_high_value_is_anomalous():
    calib = _make_calib(
        ivar_vals=[0.001, 0.002, 0.003, 0.004, 0.005],
        peak_vals=[1.0, 2.0, 3.0, 4.0, 5.0],
        cert_vals=[0.1, 0.2, 0.3, 0.4, 0.5],
    )
    sig = IvarSignal()
    p_clean = sig.p_value(0.001, calib)   # at min → high p (clean)
    p_halluc = sig.p_value(10.0, calib)   # way above → tiny p
    assert p_halluc < p_clean


def test_cert_signal_low_value_is_anomalous():
    calib = _make_calib([0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                        [0.1, 0.2, 0.3, 0.4, 0.5])
    sig = CertSignal()
    p_normal = sig.p_value(0.5, calib)
    p_halluc = sig.p_value(0.01, calib)
    assert p_halluc < p_normal


def test_build_signals_registry():
    sigs = build_signals(("ivar", "peak", "cert"))
    assert [s.name for s in sigs] == ["ivar", "peak", "cert"]


# ────────────────────────────────────────────────────────────────────────────
# Fusers
# ────────────────────────────────────────────────────────────────────────────

def test_stouffer_uniform_weights():
    fuser = StoufferFuser(weights={"a": 1.0, "b": 1.0, "c": 1.0})
    # All p = 0.5 → Z_i = 0 → Z = 0 → p_comb = 0.5
    p = fuser.fuse({"a": 0.5, "b": 0.5, "c": 0.5})
    assert p == pytest.approx(0.5, abs=1e-6)


def test_stouffer_extreme_signals_lower_p():
    fuser = StoufferFuser(weights={"a": 1.0, "b": 1.0, "c": 1.0})
    p_neutral = fuser.fuse({"a": 0.5, "b": 0.5, "c": 0.5})
    p_extreme = fuser.fuse({"a": 0.01, "b": 0.01, "c": 0.01})
    assert p_extreme < p_neutral


def test_stouffer_weights_emphasize_first_signal():
    fuser = StoufferFuser(weights={"a": 10.0, "b": 1.0, "c": 1.0})
    p_weighted = fuser.fuse({"a": 0.01, "b": 0.5, "c": 0.5})
    fuser_uniform = StoufferFuser(weights={"a": 1.0, "b": 1.0, "c": 1.0})
    p_uniform = fuser_uniform.fuse({"a": 0.01, "b": 0.5, "c": 0.5})
    # Heavier weight on the extreme signal → smaller combined p
    assert p_weighted < p_uniform


def test_fisher_extreme_signals_lower_p():
    fuser = FisherFuser()
    p_neutral = fuser.fuse({"a": 0.5, "b": 0.5})
    p_extreme = fuser.fuse({"a": 0.01, "b": 0.01})
    assert p_extreme < p_neutral


def test_max_fuser_takes_min_p():
    fuser = MaxFuser()
    assert fuser.fuse({"a": 0.3, "b": 0.5, "c": 0.8}) == 0.3
    assert fuser.fuse_to_h_score({"a": 0.3, "b": 0.5}) == pytest.approx(0.7)


# ────────────────────────────────────────────────────────────────────────────
# Per-pixel p-value
# ────────────────────────────────────────────────────────────────────────────

def test_per_pixel_p_value_shape():
    # N=5 refs, 4x4 image
    per_pixel = np.zeros((5, 4, 4), dtype=np.float32)
    per_pixel[:] = np.arange(5)[:, None, None]
    var_map = np.full((4, 4), 2.5, dtype=np.float32)
    interior = np.ones((4, 4), dtype=bool)
    p = per_pixel_p_value(var_map, per_pixel, interior)
    assert p.shape == (4, 4)
    # Pixel value 2.5: 3 of 5 refs (>=2.5: values 2,3,4 = 3 refs); but 2 < 2.5, so >= 2.5 → 3 refs
    # Actually values are 0,1,2,3,4. >= 2.5 → values 3,4 = 2 refs.
    # p = (1 + 2) / (5 + 1) = 0.5
    assert p[0, 0] == pytest.approx(0.5, abs=1e-6)


def test_per_pixel_p_value_outside_interior_is_one():
    per_pixel = np.zeros((5, 4, 4), dtype=np.float32)
    var_map = np.full((4, 4), 100.0, dtype=np.float32)
    interior = np.zeros((4, 4), dtype=bool)
    interior[0, 0] = True
    p = per_pixel_p_value(var_map, per_pixel, interior)
    assert p[0, 0] != 1.0
    assert p[1, 1] == 1.0
