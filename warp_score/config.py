"""WarpScoreConfig — single source of truth for hyperparameters.

Load from YAML for experiments, or override on CLI.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Literal, Optional

# Compute repo root relative to this file: warp_score/config.py → repo root is one level up
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_ROOT / "data"


@dataclass
class WarpScoreConfig:
    # ── Paths ──────────────────────────────────────────────────────────────
    root_dir: Path = field(default_factory=lambda: _REPO_ROOT)
    reference_dir: Path = field(default_factory=lambda: _DATA_DIR / "reference")
    query_high_dir: Path = field(default_factory=lambda: _DATA_DIR / "query" / "high")
    query_low_dir: Path = field(default_factory=lambda: _DATA_DIR / "query" / "low")
    artifacts_dir: Path = field(default_factory=lambda: _REPO_ROOT / "artifacts" / "v9_dreamgen")

    # ── RoMaV2 ─────────────────────────────────────────────────────────────
    setting: Literal["turbo", "fast", "base", "precise"] = "turbo"
    use_precision: bool = True
    bidirectional: bool = False
    vis_size: int = 224
    device: str = "cuda"

    # ── Foreground / interior mask ─────────────────────────────────────────
    erosion_k: int = 10

    # ── Calibration ────────────────────────────────────────────────────────
    per_pixel_calibration: bool = True
    gpd_tail: bool = False
    n_min_refs: int = 30
    rng_seed: int = 42

    # ── Signals + fusion ───────────────────────────────────────────────────
    signal_names: tuple[str, ...] = ("ivar", "peak", "cert")
    fuser: Literal["stouffer", "fisher", "max", "cauchy"] = "stouffer"
    stouffer_weights: dict[str, float] = field(default_factory=lambda: {
        "ivar": 2.0, "peak": 1.0, "cert": 1.0,
    })

    # ── v9 precision-matrix flags ──────────────────────────────────────────
    cycle_consistency: bool = False   # reserved for future cycle-consistent matching
    legacy_ivar: bool = False          # if True, use old ivar+peak+cert+Stouffer pipeline

    # ── Decision threshold ─────────────────────────────────────────────────
    fpr_alpha: float = 0.05  # H_score > (1 - alpha) → label=1

    # ── Visualization ──────────────────────────────────────────────────────
    save_heatmaps: bool = True

    # ── Derived (set in __post_init__) ─────────────────────────────────────
    calib_path: Path = field(init=False)
    summary_csv: Path = field(init=False)
    run_log: Path = field(init=False)

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        self.reference_dir = Path(self.reference_dir)
        self.query_high_dir = Path(self.query_high_dir)
        self.query_low_dir = Path(self.query_low_dir)
        self.artifacts_dir = Path(self.artifacts_dir)
        self.calib_path = self.artifacts_dir / "calibration.npz"
        self.summary_csv = self.artifacts_dir / "summary.csv"
        self.run_log = self.artifacts_dir / "run.log"

    # ── Decision threshold helper ──────────────────────────────────────────
    @property
    def decision_threshold(self) -> float:
        """H_score above this → labeled hallucination."""
        return 1.0 - self.fpr_alpha

    # ── Loaders ────────────────────────────────────────────────────────────
    @classmethod
    def from_yaml(cls, path: Path | str) -> "WarpScoreConfig":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "WarpScoreConfig":
        valid = {f.name for f in fields(cls) if f.init}
        filtered = {k: v for k, v in data.items() if k in valid}
        # Convert string paths to Path
        for k in ("root_dir", "reference_dir", "query_high_dir", "query_low_dir", "artifacts_dir"):
            if k in filtered:
                filtered[k] = Path(filtered[k])
        # signal_names: list → tuple
        if "signal_names" in filtered and isinstance(filtered["signal_names"], list):
            filtered["signal_names"] = tuple(filtered["signal_names"])
        return cls(**filtered)

    def to_dict(self) -> dict:
        out: dict = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, Path):
                v = str(v)
            elif isinstance(v, tuple):
                v = list(v)
            out[f.name] = v
        return out

    def merge(self, **overrides) -> "WarpScoreConfig":
        """Return a new config with the given attributes replaced."""
        from dataclasses import replace
        # Filter to init=True fields only
        valid = {f.name for f in fields(self) if f.init}
        kept = {k: v for k, v in overrides.items() if k in valid and v is not None}
        return replace(self, **kept)
