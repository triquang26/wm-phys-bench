"""Video-level fusion of cycle (pair-based) + k-NN (frame-based) anomaly signals.

Both signals already aggregate to per-video H ∈ [0, 1] (1 - p_value).
Fuse via Cauchy combine (ACAT) at the video level — symmetric with the per-pair
Cauchy used inside cycle/knn for their own internal mean+peak fusion.
"""
from __future__ import annotations

import numpy as np

_CLIP_EPS = 1e-6


def cauchy_combine(ps: list[float]) -> float:
    """ACAT Cauchy p-value combine. Returns combined p ∈ (0, 1)."""
    ps = [p for p in ps if p is not None and 0.0 < p < 1.0]
    if not ps:
        return 0.5
    t = float(np.mean([np.tan(np.pi * (0.5 - p)) for p in ps]))
    return float(0.5 - np.arctan(t) / np.pi)


def cauchy_combine_video(h_cycle: float, h_knn: float) -> float:
    """Fuse two video-level H scores via Cauchy combine of their p-values."""
    p_cycle = 1.0 - float(np.clip(h_cycle, _CLIP_EPS, 1.0 - _CLIP_EPS))
    p_knn = 1.0 - float(np.clip(h_knn, _CLIP_EPS, 1.0 - _CLIP_EPS))
    return 1.0 - cauchy_combine([p_cycle, p_knn])


def complementarity_report(rows: list[dict]) -> dict:
    """Count cycle-only / knn-only / both / neither catches among gens.

    rows: list of dicts with keys 'type', 'ratio_cycle', 'ratio_knn', 'ratio_fused'.
    """
    gens = [r for r in rows if r.get("type") == "GEN"]
    n = len(gens)
    cycle_catch = sum(1 for r in gens if r["ratio_cycle"] > 1.0)
    knn_catch = sum(1 for r in gens if r["ratio_knn"] > 1.0)
    both = sum(1 for r in gens if r["ratio_cycle"] > 1.0 and r["ratio_knn"] > 1.0)
    fused_catch = sum(1 for r in gens if r["ratio_fused"] > 1.0)
    cycle_only = cycle_catch - both
    knn_only = knn_catch - both
    neither = n - (cycle_catch + knn_catch - both)
    return {
        "n_gens": n,
        "cycle_catch": cycle_catch,
        "knn_catch": knn_catch,
        "both": both,
        "cycle_only": cycle_only,
        "knn_only": knn_only,
        "neither": neither,
        "fused_catch": fused_catch,
    }


def separation_gap(rows: list[dict], key: str = "ratio_fused") -> float:
    """Decision-boundary gap: min(ratio over hallu-flagged gens) - max(ratio over real).

    Higher = cleaner separation. NaN if no gens flagged.
    """
    gens_hallu = [r[key] for r in rows if r.get("type") == "GEN" and r[key] > 1.0]
    reals = [r[key] for r in rows if r.get("type") == "REAL"]
    if not gens_hallu or not reals:
        return float("nan")
    return float(min(gens_hallu) - max(reals))


def borderline_count(rows: list[dict], key: str = "ratio_fused",
                     lo: float = 0.95, hi: float = 1.05) -> int:
    """Count rows in the ambiguous borderline zone."""
    return sum(1 for r in rows if lo <= r[key] <= hi)
