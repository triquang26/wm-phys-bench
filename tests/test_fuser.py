"""Unit tests for SignalFuser implementations."""
import pytest
from warp_score.fuser import StoufferFuser, FisherFuser, MaxFuser, build_fuser


# ---------------------------------------------------------------------------
# StoufferFuser
# ---------------------------------------------------------------------------

def test_stouffer_equal_p_values_equal_weights():
    """All p=0.5 → Z=0 → p_combined=0.5 → H_score=0.5."""
    fuser = StoufferFuser()
    p = {"ivar": 0.5, "peak": 0.5, "cert": 0.5}
    assert fuser.fuse_to_h_score(p) == pytest.approx(0.5, abs=1e-6)


def test_stouffer_small_p_extreme_h_score():
    """All p=0.01 → large combined Z → H_score > 0.99."""
    fuser = StoufferFuser()
    p = {"ivar": 0.01, "peak": 0.01, "cert": 0.01}
    assert fuser.fuse_to_h_score(p) > 0.99


def test_stouffer_weights_change_score():
    """Doubling the weight on the extreme signal (p=0.01) raises H_score."""
    p = {"ivar": 0.01, "peak": 0.5}
    unweighted = StoufferFuser(weights={"ivar": 1.0, "peak": 1.0})
    weighted   = StoufferFuser(weights={"ivar": 2.0, "peak": 1.0})
    assert weighted.fuse_to_h_score(p) > unweighted.fuse_to_h_score(p)


def test_stouffer_empty_returns_h_zero():
    """Empty p_values dict → fuse returns 1.0 → H_score = 0.0."""
    fuser = StoufferFuser()
    assert fuser.fuse({}) == pytest.approx(1.0)
    assert fuser.fuse_to_h_score({}) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# FisherFuser
# ---------------------------------------------------------------------------

def test_fisher_small_p_extreme():
    """All p=0.01 → large χ² → H_score > 0.99."""
    fuser = FisherFuser()
    p = {"ivar": 0.01, "peak": 0.01, "cert": 0.01}
    assert fuser.fuse_to_h_score(p) > 0.99


def test_fisher_large_p_low_score():
    """All p=0.5 with 3 signals → H_score analytically ≈ 0.458; verify < 0.6."""
    fuser = FisherFuser()
    p = {"ivar": 0.5, "peak": 0.5, "cert": 0.5}
    assert fuser.fuse_to_h_score(p) < 0.6


# ---------------------------------------------------------------------------
# MaxFuser
# ---------------------------------------------------------------------------

def test_max_fuser_returns_1_minus_min_p():
    """MaxFuser.fuse_to_h_score == 1 - min(p_values)."""
    fuser = MaxFuser()
    p = {"a": 0.2, "b": 0.8}
    assert fuser.fuse_to_h_score(p) == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# build_fuser factory
# ---------------------------------------------------------------------------

def test_build_fuser_stouffer():
    assert isinstance(build_fuser("stouffer"), StoufferFuser)


def test_build_fuser_invalid():
    with pytest.raises(ValueError, match="unknown"):
        build_fuser("unknown")
