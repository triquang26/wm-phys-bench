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


class BaselineNormalizer:
    """Sigmoid-normalize per-task ratios via bootstrap-estimated baseline σ.

    Maps `ratio_fused = H_test / H_train ∈ [0, ∞)` → `score ∈ [0, 1]`:
        ratio = 1.0  → score = 0.5  (at baseline)
        ratio > 1.0  → score > 0.5  (more anomalous)
        ratio < 1.0  → score < 0.5  (cleaner)

    Steepness α = 1/σ is derived from bootstrapping H_train_fused under
    aggregation noise (resampling per-pair / per-frame H values with
    replacement). σ small → sharp baseline → steep sigmoid; σ large →
    uncertain baseline → flatter sigmoid → more conservative.

    Per-task: fit on each task's H_train values; α is task-specific.
    """

    def __init__(self, n_boot: int = 200, pct: int = 80, seed: int = 42):
        self.n_boot = n_boot
        self.pct = pct
        self.seed = seed
        self.sigma: float | None = None
        self.alpha: float | None = None
        self.bootstrap_dist: np.ndarray | None = None

    def fit(self, h_pairs_train, h_frames_train) -> "BaselineNormalizer":
        """Estimate σ via bootstrap. Returns self for chaining."""
        rng = np.random.default_rng(self.seed)
        hp = np.asarray(h_pairs_train, dtype=np.float64)
        hf = np.asarray(h_frames_train, dtype=np.float64)
        if len(hp) == 0 or len(hf) == 0:
            raise ValueError("h_pairs_train / h_frames_train must be non-empty")
        n_p, n_f = len(hp), len(hf)
        boot = np.empty(self.n_boot, dtype=np.float64)
        for b in range(self.n_boot):
            hp_r = rng.choice(hp, n_p, replace=True)
            hf_r = rng.choice(hf, n_f, replace=True)
            cp = float(np.percentile(hp_r, self.pct))
            kp = float(np.percentile(hf_r, self.pct))
            boot[b] = cauchy_combine_video(cp, kp)
        self.bootstrap_dist = boot
        self.sigma = float(np.std(boot))
        self.alpha = 1.0 / max(self.sigma, 1e-6)
        return self

    def normalize(self, ratio: float) -> float:
        """Map a ratio → score ∈ [0, 1]. Requires .fit() first."""
        if self.sigma is None or self.alpha is None:
            raise RuntimeError("Call .fit() before .normalize()")
        z = self.alpha * (float(ratio) - 1.0)
        return float(1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0))))

    def to_dict(self) -> dict:
        return {
            "sigma": float(self.sigma) if self.sigma is not None else None,
            "alpha": float(self.alpha) if self.alpha is not None else None,
            "n_boot": self.n_boot,
            "pct": self.pct,
            "seed": self.seed,
        }


def bootstrap_baseline_sigma(h_pairs_train, h_frames_train,
                             n_boot: int = 200, pct: int = 80,
                             seed: int = 42) -> tuple[float, np.ndarray]:
    """Functional wrapper around BaselineNormalizer.fit() — returns (σ, dist)."""
    bn = BaselineNormalizer(n_boot=n_boot, pct=pct, seed=seed).fit(
        h_pairs_train, h_frames_train)
    return bn.sigma, bn.bootstrap_dist


def sigmoid_normalize_ratio(ratio: float, sigma_baseline: float) -> float:
    """Functional wrapper — sigmoid mapping with pre-computed σ."""
    alpha = 1.0 / max(sigma_baseline, 1e-6)
    z = alpha * (float(ratio) - 1.0)
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0))))


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
