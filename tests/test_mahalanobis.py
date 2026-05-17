"""Tests for the v9 precision-matrix Mahalanobis hallucination detector.

All tests are pure-numpy (no GPU, no model loading).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the warp_score package is importable from the worktree root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from warp_score.statistics import CertWeightedStatistics, MahalanobisStatistics


# =============================================================================
# Test 1 — identity precision, D ~ χ²(2)
# =============================================================================

def test_identity_precision_chi2():
    """With identity precision and true warp = 0, D should be ~ χ²(2(K-1)).
    E[D] = 2*(K-1).  We use K=2 → E[D]=2; sample over many pixels for accuracy."""
    K, H, W = 2, 8, 8
    np.random.seed(42)
    # true consensus at origin; each ref observes N(0, I)
    warps = np.random.randn(K, H, W, 2).astype(np.float32)
    precisions = np.broadcast_to(np.eye(2, dtype=np.float32), (K, H, W, 2, 2)).copy()

    D_map, logdetΛ_map, mu_hat = MahalanobisStatistics.ivar_per_pixel(warps, precisions)

    assert D_map.shape == (H, W), f"D_map shape mismatch: {D_map.shape}"
    assert logdetΛ_map.shape == (H, W), f"logdetΛ shape mismatch: {logdetΛ_map.shape}"
    assert mu_hat.shape == (H, W, 2), f"mu_hat shape mismatch: {mu_hat.shape}"

    # D should be non-negative
    assert np.all(D_map >= -1e-5), f"Negative D values: min={D_map.min()}"

    # E[D] = 2*(K-1) = 2 for K=2; with 64 pixels, sample mean should be in [0.5, 5.0]
    mean_D = float(D_map.mean())
    assert 0.5 < mean_D < 5.0, (
        f"D_map mean = {mean_D:.3f}, expected ~2 (χ²(2) = 2*(K-1))"
    )

    # log det Λ = log det (K * I) = K * log(1) + log(K) for 2D identity summed K times
    # Λ = K * I → det Λ = K^2 → log det Λ = 2 * log(K)
    expected_logdet = 2.0 * np.log(K)  # for K=2: ~1.386
    np.testing.assert_allclose(logdetΛ_map.mean(), expected_logdet, atol=0.1)


# =============================================================================
# Test 2 — identity precision matches old cert-weighted ivar (directional)
# =============================================================================

def test_identity_precision_matches_old_ivar():
    """When Σ⁻¹_r = c * I (uniform, isotropic), the Mahalanobis deviance D
    should be proportional to the cert-weighted variance (direction of agreement)."""
    K, H, W = 3, 16, 16
    np.random.seed(7)
    warps = np.random.randn(K, H, W, 2).astype(np.float32)
    c = 0.5
    eye2 = np.eye(2, dtype=np.float32)
    precisions = np.broadcast_to(c * eye2, (K, H, W, 2, 2)).copy()

    D_map, _, _ = MahalanobisStatistics.ivar_per_pixel(warps, precisions)

    # Old cert-weighted ivar with uniform cert = c
    certs = np.full((K, H, W), c, dtype=np.float32)
    var_old = CertWeightedStatistics.variance_per_pixel(warps, certs)

    # D should be non-negative everywhere
    assert np.all(D_map >= -1e-5), f"Negative D: min={D_map.min()}"

    # Where var_old is large, D should also be large (positive correlation)
    corr = np.corrcoef(var_old.ravel(), D_map.ravel())[0, 1]
    assert corr > 0.9, (
        f"D_map and cert-weighted var should be highly correlated, got r={corr:.3f}"
    )


# =============================================================================
# Test 3 — anisotropic precision amplifies disagreement along high-prec axis
# =============================================================================

def test_anisotropic_precision():
    """Disagreement along the high-precision axis should produce much larger D
    than isotropic precision with the same magnitude of disagreement."""
    K, H, W = 2, 4, 4

    warps = np.zeros((K, H, W, 2), dtype=np.float32)
    warps[0, ..., 0] = 1.0   # ref0 points right (+x)
    warps[1, ..., 0] = -1.0  # ref1 points left  (-x)
    # Both refs agree on y (zero displacement)

    # Anisotropic: very confident along x (prec=10), uncertain along y (prec=0.1)
    prec_aniso = np.array([[10.0, 0.0], [0.0, 0.1]], dtype=np.float32)
    prec_iso = np.eye(2, dtype=np.float32)

    precisions_aniso = np.broadcast_to(prec_aniso, (K, H, W, 2, 2)).copy()
    precisions_iso = np.broadcast_to(prec_iso, (K, H, W, 2, 2)).copy()

    D_aniso, _, _ = MahalanobisStatistics.ivar_per_pixel(warps, precisions_aniso)
    D_iso, _, _ = MahalanobisStatistics.ivar_per_pixel(warps, precisions_iso)

    # Anisotropic: disagreement along high-precision axis → D >> isotropic D
    # x-contribution to D_aniso ≈ 10 * (disagreement)^2 / 2 ≫ D_iso ≈ 1 * same
    assert D_aniso.mean() > D_iso.mean() * 5, (
        f"Anisotropic D={D_aniso.mean():.2f} should be >> isotropic D={D_iso.mean():.2f}"
    )


# =============================================================================
# Test 4 — background pixels (all-zero precision) yield D=0 and logdetΛ=-30
# =============================================================================

def test_background_pixels_floored():
    """Background pixels (Σ⁻¹_r = 0) should produce D=0 and logdetΛ at floor."""
    K, H, W = 3, 8, 8
    np.random.seed(0)
    warps = np.random.randn(K, H, W, 2).astype(np.float32)

    # Mix: foreground (top half) has eye(2), background (bottom half) is zero
    precisions = np.zeros((K, H, W, 2, 2), dtype=np.float32)
    fg_rows = H // 2
    precisions[:, :fg_rows, :] = np.eye(2, dtype=np.float32)

    D_map, logdetΛ_map, _ = MahalanobisStatistics.ivar_per_pixel(warps, precisions)

    # Background rows should have D = 0 (no precision → no contribution)
    bg_D = D_map[fg_rows:, :]
    assert np.allclose(bg_D, 0.0, atol=1e-5), (
        f"Background D should be 0, got max={bg_D.max():.6f}"
    )

    # Background rows should have logdetΛ at floor (-30)
    bg_logdet = logdetΛ_map[fg_rows:, :]
    assert np.allclose(bg_logdet, -30.0, atol=1e-5), (
        f"Background logdetΛ should be -30 (floor), got {bg_logdet.mean():.3f}"
    )

    # Foreground should have positive D values
    fg_D = D_map[:fg_rows, :]
    assert fg_D.mean() > 0.1, f"Foreground D should be positive, got {fg_D.mean():.4f}"


# =============================================================================
# Test 5 — TaskCalibration roundtrip save/load with v9 fields
# =============================================================================

def test_task_calibration_roundtrip(tmp_path):
    """TaskCalibration with v9 maha fields should survive a save/load roundtrip."""
    import time
    from warp_score.calibrator import CalibrationArtifact, TaskCalibration

    np.random.seed(99)
    tc = TaskCalibration(
        task="test_task",
        n_refs=5,
        ivar_dist=np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
        peak_dist=np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        cert_dist=np.array([0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32),
        ivar_maha_dist=np.array([5.0, 8.0, 10.0, 12.0, 15.0], dtype=np.float32),
        evidence_dist=np.array([-3.0, -2.0, -1.0, 0.0, 1.0], dtype=np.float32),
        T_null=np.random.rand(5, 16, 16).astype(np.float32),
    )
    artifact = CalibrationArtifact(
        tasks={"test_task": tc},
        global_=tc,
        config_snapshot={},
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    save_path = tmp_path / "calib.npz"
    artifact.save(save_path)

    loaded = CalibrationArtifact.load(save_path)
    lt = loaded.tasks["test_task"]

    np.testing.assert_array_almost_equal(lt.ivar_maha_dist, tc.ivar_maha_dist)
    np.testing.assert_array_almost_equal(lt.evidence_dist, tc.evidence_dist)
    np.testing.assert_array_almost_equal(lt.T_null, tc.T_null)

    # Original v8 fields must survive too
    np.testing.assert_array_almost_equal(lt.ivar_dist, tc.ivar_dist)
    np.testing.assert_array_almost_equal(lt.peak_dist, tc.peak_dist)
    np.testing.assert_array_almost_equal(lt.cert_dist, tc.cert_dist)

    # Global should also have maha dists (pooled)
    lg = loaded.global_
    assert lg.ivar_maha_dist is not None, "Global should have ivar_maha_dist"
    assert lg.evidence_dist is not None, "Global should have evidence_dist"


# =============================================================================
# Test 6 — CauchyFuser basic properties
# =============================================================================

def test_cauchy_fuser_properties():
    """CauchyFuser should return valid p-values and be sensitive to small p-values."""
    from warp_score.fuser import CauchyFuser

    fuser = CauchyFuser()

    # All p=0.5 (no signal) → combined p should be near 0.5
    p_null = fuser.fuse({"a": 0.5, "b": 0.5})
    assert 0.4 < p_null < 0.6, f"Null p should be ~0.5, got {p_null:.4f}"

    # Small p-values → small combined p (anomalous)
    p_signal = fuser.fuse({"a": 0.01, "b": 0.01})
    assert p_signal < 0.05, f"Signal p should be small, got {p_signal:.4f}"

    # Extreme p → still a valid float in (0, 1)
    p_extreme = fuser.fuse({"a": 1e-10, "b": 1e-10})
    assert 0.0 < p_extreme < 0.01, f"Extreme p={p_extreme:.2e}"

    # Returns 1.0 for empty input
    assert fuser.fuse({}) == 1.0


# =============================================================================
# Test 7 — IvarMahaSignal and EvidenceSignal registry lookup
# =============================================================================

def test_signal_registry():
    """New signals should be reachable via build_signals and require maha calib."""
    from warp_score.signals import build_signals, IvarMahaSignal, EvidenceSignal
    from warp_score.calibrator import TaskCalibration
    import numpy as np

    sigs = build_signals(("ivar_maha", "evidence"))
    assert len(sigs) == 2
    assert isinstance(sigs[0], IvarMahaSignal)
    assert isinstance(sigs[1], EvidenceSignal)

    # Calling p_value without the required dist should raise RuntimeError
    calib_no_maha = TaskCalibration(
        task="t", n_refs=1,
        ivar_dist=np.array([0.1]),
        peak_dist=np.array([0.1]),
        cert_dist=np.array([0.1]),
    )
    with pytest.raises(RuntimeError, match="ivar_maha_dist"):
        sigs[0].p_value(1.0, calib_no_maha)

    with pytest.raises(RuntimeError, match="evidence_dist"):
        sigs[1].p_value(1.0, calib_no_maha)
