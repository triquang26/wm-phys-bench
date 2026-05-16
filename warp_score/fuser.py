"""Signal fusion — combine multiple p-values into one calibrated score.

Methods implemented:
  - StoufferFuser : Z-fusion (with optional weights)
  - FisherFuser   : -2 * Σ log(p_i) ~ χ²(2k)
  - MaxFuser      : 1 - min(p_i)  — preserves OR semantics in continuous form

All return a combined p-value ∈ (0, 1]. The final hallucination score is
H_score = 1 - p_combined.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy import stats


_P_EPS = 1e-12  # clamp p-values to avoid log(0) or Φ⁻¹(1)


class SignalFuser(ABC):
    @abstractmethod
    def fuse(self, p_values: dict[str, float]) -> float:
        """Return combined p-value in (0, 1]. Smaller = more anomalous."""

    def fuse_to_h_score(self, p_values: dict[str, float]) -> float:
        """Return 1 - p_combined ∈ [0, 1)."""
        return float(1.0 - self.fuse(p_values))


class StoufferFuser(SignalFuser):
    """Stouffer's Z-method:
        Z = Σ w_i Z_i / sqrt(Σ w_i²)        where Z_i = Φ⁻¹(1 - p_i)
        p_combined = 1 - Φ(Z)
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or {}

    def fuse(self, p_values: dict[str, float]) -> float:
        if not p_values:
            return 1.0
        names = list(p_values.keys())
        ps = np.clip(np.array([p_values[n] for n in names]), _P_EPS, 1.0 - _P_EPS)
        ws = np.array([self.weights.get(n, 1.0) for n in names], dtype=np.float64)
        zs = stats.norm.isf(ps)  # Φ⁻¹(1 - p)
        Z = float((ws * zs).sum() / np.sqrt((ws ** 2).sum()))
        return float(stats.norm.sf(Z))  # 1 - Φ(Z)


class FisherFuser(SignalFuser):
    """Fisher's combined test:
        χ² = -2 * Σ log p_i  ~  χ²(2k) under independent uniform null
    """

    def fuse(self, p_values: dict[str, float]) -> float:
        if not p_values:
            return 1.0
        ps = np.clip(np.array(list(p_values.values())), _P_EPS, 1.0)
        chi2 = float(-2.0 * np.log(ps).sum())
        df = 2 * len(ps)
        return float(stats.chi2.sf(chi2, df=df))


class MaxFuser(SignalFuser):
    """Combined p = min(p_i). Preserves the OR-decision semantics of the
    original 3-signal detector, but as a continuous score for ranking."""

    def fuse(self, p_values: dict[str, float]) -> float:
        if not p_values:
            return 1.0
        return float(min(p_values.values()))


def build_fuser(
    name: str, stouffer_weights: dict[str, float] | None = None,
) -> SignalFuser:
    if name == "stouffer":
        return StoufferFuser(weights=stouffer_weights)
    if name == "fisher":
        return FisherFuser()
    if name == "max":
        return MaxFuser()
    raise ValueError(f"Unknown fuser '{name}'. Choices: stouffer, fisher, max.")
