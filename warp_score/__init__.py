"""warp_score — continuous, calibrated hallucination scoring via RoMaV2 warp variance.

Pipeline:
    query frame  ─┬─→ RoMaMatcher   →  warp + cert maps (vs N=50 task refs)
                  ├─→ ForegroundMask + InteriorMask
                  └─→ Signals (Ivar / Peak / Cert)
                         │  raw values → empirical p-value via TaskCalibration
                         ▼
                     SignalFuser (Stouffer / Fisher / Max)
                         │
                         ▼
                  HallucinationResult  ─→  H_score ∈ [0,1] + per-pixel heatmap

Entry points:
    python -m warp_score calibrate   ...
    python -m warp_score detect      ...
    python -m warp_score eval        ...
"""

from .config import WarpScoreConfig
from .detector import HallucinationResult, WarpVarianceDetector
from .calibrator import CalibrationArtifact, EmpiricalNullCalibrator, TaskCalibration
from .fuser import FisherFuser, MaxFuser, SignalFuser, StoufferFuser
from .signals import CertSignal, IvarSignal, PeakSignal, Signal

__all__ = [
    "WarpScoreConfig",
    "HallucinationResult",
    "WarpVarianceDetector",
    "CalibrationArtifact",
    "EmpiricalNullCalibrator",
    "TaskCalibration",
    "Signal",
    "IvarSignal",
    "PeakSignal",
    "CertSignal",
    "SignalFuser",
    "StoufferFuser",
    "FisherFuser",
    "MaxFuser",
]
