"""Unit tests for CertWeightedStatistics."""
from __future__ import annotations

import numpy as np
import pytest

from warp_score.statistics import CertWeightedStatistics as CWS


def test_variance_per_pixel_agree_when_warps_identical():
    K, H, W = 5, 8, 8
    warps = np.ones((K, H, W, 2), dtype=np.float32)
    certs = np.ones((K, H, W), dtype=np.float32)
    var = CWS.variance_per_pixel(warps, certs)
    assert var.shape == (H, W)
    assert np.allclose(var, 0.0)


def test_variance_per_pixel_disagree_when_warps_differ():
    K, H, W = 3, 4, 4
    warps = np.zeros((K, H, W, 2), dtype=np.float32)
    warps[0] = -1.0
    warps[1] = 0.0
    warps[2] = 1.0
    certs = np.ones((K, H, W), dtype=np.float32)
    var = CWS.variance_per_pixel(warps, certs)
    # With equal weights, mean = 0, variance per pixel per dim = (1+0+1)/3 = 2/3
    # Summed over 2 dims = 4/3 ≈ 1.333
    assert np.allclose(var, 4.0 / 3.0, atol=1e-5)


def test_interior_mean():
    var = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    mask = np.array([[True, True, False], [False, True, True]])
    assert CWS.interior_mean(var, mask) == pytest.approx((1 + 2 + 5 + 6) / 4.0)


def test_within_frame_zscore_zero_var():
    var = np.zeros((4, 4), dtype=np.float32)
    mask = np.ones((4, 4), dtype=bool)
    z = CWS.within_frame_zscore(var, mask)
    assert np.allclose(z, 0.0)


def test_within_frame_zscore_normalized():
    var = np.array([[0, 1], [2, 3]], dtype=np.float32)
    mask = np.ones((2, 2), dtype=bool)
    z = CWS.within_frame_zscore(var, mask)
    # mean=1.5, std ≈ 1.118, so z = (var - 1.5)/std
    assert z.mean() == pytest.approx(0.0, abs=1e-5)
