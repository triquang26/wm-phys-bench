"""Signal classes — each maps raw statistics to an empirical p-value
against a TaskCalibration's stored distribution.

Direction convention:
- "high" direction: large raw value = anomalous (e.g. ivar, peak)
- "low"  direction: small raw value = anomalous (e.g. mean cert)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from .calibrator import TaskCalibration


@dataclass
class SignalResult:
    name: str
    raw: float
    p_value: float


class Signal(ABC):
    name: str
    direction: Literal["high", "low"]

    @abstractmethod
    def get_distribution(self, calib: "TaskCalibration") -> np.ndarray:
        """Return the sorted-ascending null distribution for this signal."""

    def p_value(self, raw: float, calib: "TaskCalibration") -> float:
        """Empirical one-sided p-value with Laplace smoothing (1+...)/(n+1)."""
        dist = self.get_distribution(calib)
        return _empirical_p(raw, dist, self.direction)

    def result(self, raw: float, calib: "TaskCalibration") -> SignalResult:
        return SignalResult(
            name=self.name,
            raw=raw,
            p_value=self.p_value(raw, calib),
        )


class IvarSignal(Signal):
    name = "ivar"
    direction = "high"

    def get_distribution(self, calib: "TaskCalibration") -> np.ndarray:
        return calib.ivar_dist


class PeakSignal(Signal):
    name = "peak"
    direction = "high"

    def get_distribution(self, calib: "TaskCalibration") -> np.ndarray:
        return calib.peak_dist


class CertSignal(Signal):
    name = "cert"
    direction = "low"  # low cert = anomalous

    def get_distribution(self, calib: "TaskCalibration") -> np.ndarray:
        return calib.cert_dist


_REGISTRY: dict[str, type[Signal]] = {
    "ivar": IvarSignal,
    "peak": PeakSignal,
    "cert": CertSignal,
}


def build_signals(names: tuple[str, ...]) -> list[Signal]:
    """Instantiate Signal objects by name."""
    return [_REGISTRY[n]() for n in names]


# =============================================================================
# Pure functional helper (also used for per-pixel p-values)
# =============================================================================

def _empirical_p(value: float, dist_sorted: np.ndarray, direction: str) -> float:
    """Empirical one-sided p-value. dist_sorted assumed ascending.

    p_high = (1 + #{x in dist | x >= value}) / (n + 1)
    p_low  = (1 + #{x in dist | x <= value}) / (n + 1)
    """
    n = len(dist_sorted)
    if n == 0:
        return 1.0
    if direction == "high":
        # number of dist values >= value = n - searchsorted(value, side='left')
        k = n - int(np.searchsorted(dist_sorted, value, side="left"))
    elif direction == "low":
        k = int(np.searchsorted(dist_sorted, value, side="right"))
    else:
        raise ValueError(f"Unknown direction '{direction}'")
    return (1 + k) / (n + 1)


def per_pixel_p_value(
    var_map: np.ndarray,
    per_pixel_sorted: np.ndarray,
    interior_mask: np.ndarray,
) -> np.ndarray:
    """Per-pixel empirical p-value (direction = high).

    var_map: (H, W)
    per_pixel_sorted: (N, H, W) sorted ascending along axis 0
    interior_mask: (H, W) bool

    Returns: (H, W) of p-values in [0, 1]; pixels outside interior = 1.0.
    """
    H, W = var_map.shape
    n = per_pixel_sorted.shape[0]
    p = np.ones((H, W), dtype=np.float32)
    ys, xs = np.where(interior_mask)
    if ys.size == 0:
        return p

    # Vectorized: for each interior pixel, compute # of refs >= var_map[p]
    interior_var = var_map[ys, xs]                    # (M,)
    interior_dist = per_pixel_sorted[:, ys, xs]       # (N, M) — already sorted per col
    # searchsorted along axis 0 isn't directly supported; loop is O(M) which is fine for M~50k
    # but vectorized via broadcast:
    #   k = N - sum(interior_dist < interior_var, axis=0) - sum(interior_dist == interior_var)
    # We use side='left' semantics: # of values >= v = N - argfirst(v in sorted asc)
    # Simpler: use rank via sum of boolean
    ge = (interior_dist >= interior_var[None, :])
    k = ge.sum(axis=0)
    p_interior = (1 + k) / (n + 1)
    p[ys, xs] = p_interior.astype(np.float32)
    return p
