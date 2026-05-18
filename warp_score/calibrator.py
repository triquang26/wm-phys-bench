"""Empirical-null calibration via leave-one-out matching on per-task clean refs.

For each task T with N>=2 clean refs:
  for each ref q in T:
      refs = T minus {q}
      compute (ivar, peak, cert) for q vs refs (cert-weighted)
  → 3 distributions of size N (sorted): F_T_ivar, F_T_peak, F_T_cert
  → optionally per-pixel sorted distribution (N, H, W) for per-pixel heatmap

Saved as npz (arrays) + json (metadata).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from .config import WarpScoreConfig
from .mask import ForegroundMask, InteriorMask
from .matcher import RoMaMatcher
from .statistics import CertWeightedStatistics, MahalanobisStatistics


@dataclass
class TaskCalibration:
    task: str
    n_refs: int
    ivar_dist: np.ndarray              # (N,) sorted ascending
    peak_dist: np.ndarray              # (N,) sorted ascending
    cert_dist: np.ndarray              # (N,) sorted ascending
    per_pixel_var: Optional[np.ndarray] = None    # (N, H, W) sorted ascending along axis 0
    # v9 precision-matrix fields
    ivar_maha_dist: Optional[np.ndarray] = None   # (N,) sorted ascending
    evidence_dist: Optional[np.ndarray] = None    # (N,) sorted ascending
    peak_maha_dist: Optional[np.ndarray] = None   # (N,) sorted ascending — peak Z-score of D_map
    T_null: Optional[np.ndarray] = None           # (N, H, W) D_map null, sorted ascending along axis 0

    @property
    def mean_ivar(self) -> float:
        return float(self.ivar_dist.mean())

    @property
    def std_ivar(self) -> float:
        return float(self.ivar_dist.std(ddof=1)) if self.n_refs > 1 else 0.0


@dataclass
class CalibrationArtifact:
    tasks: dict[str, TaskCalibration]
    global_: TaskCalibration
    config_snapshot: dict
    created_at: str
    n_tasks: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_tasks = len(self.tasks)

    # ──────────────────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        npz_data: dict[str, np.ndarray] = {}
        task_meta: dict[str, dict] = {}

        for task, tc in self.tasks.items():
            slug = _slug(task)
            npz_data[f"{slug}__ivar"] = tc.ivar_dist
            npz_data[f"{slug}__peak"] = tc.peak_dist
            npz_data[f"{slug}__cert"] = tc.cert_dist
            task_meta[task] = {
                "slug": slug,
                "n_refs": tc.n_refs,
                "mean_ivar": tc.mean_ivar,
                "std_ivar": tc.std_ivar,
                "has_per_pixel": tc.per_pixel_var is not None,
                "has_ivar_maha": tc.ivar_maha_dist is not None,
                "has_evidence": tc.evidence_dist is not None,
                "has_peak_maha": tc.peak_maha_dist is not None,
                "has_T_null": tc.T_null is not None,
            }
            if tc.per_pixel_var is not None:
                npz_data[f"{slug}__per_pixel_var"] = tc.per_pixel_var.astype(np.float32)
            if tc.ivar_maha_dist is not None:
                npz_data[f"{slug}__ivar_maha"] = tc.ivar_maha_dist.astype(np.float32)
            if tc.evidence_dist is not None:
                npz_data[f"{slug}__evidence"] = tc.evidence_dist.astype(np.float32)
            if tc.peak_maha_dist is not None:
                npz_data[f"{slug}__peak_maha"] = tc.peak_maha_dist.astype(np.float32)
            if tc.T_null is not None:
                npz_data[f"{slug}__T_null"] = tc.T_null.astype(np.float32)

        # Global
        g = self.global_
        npz_data["__global__ivar"] = g.ivar_dist
        npz_data["__global__peak"] = g.peak_dist
        npz_data["__global__cert"] = g.cert_dist
        if g.ivar_maha_dist is not None:
            npz_data["__global__ivar_maha"] = g.ivar_maha_dist.astype(np.float32)
        if g.evidence_dist is not None:
            npz_data["__global__evidence"] = g.evidence_dist.astype(np.float32)
        if g.peak_maha_dist is not None:
            npz_data["__global__peak_maha"] = g.peak_maha_dist.astype(np.float32)

        np.savez_compressed(path, **npz_data)

        meta = {
            "n_tasks": self.n_tasks,
            "tasks": task_meta,
            "global": {
                "n_refs": g.n_refs,
                "mean_ivar": g.mean_ivar,
                "std_ivar": g.std_ivar,
                "has_ivar_maha": g.ivar_maha_dist is not None,
                "has_evidence": g.evidence_dist is not None,
                "has_peak_maha": g.peak_maha_dist is not None,
            },
            "config_snapshot": self.config_snapshot,
            "created_at": self.created_at,
        }
        meta_path = path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        print(f"[calibrate] saved arrays   → {path}")
        print(f"[calibrate] saved metadata → {meta_path}")

    @classmethod
    def load(cls, path: Path) -> "CalibrationArtifact":
        path = Path(path)
        meta_path = path.with_suffix(".json")
        with open(meta_path) as f:
            meta = json.load(f)
        npz = np.load(path)

        tasks: dict[str, TaskCalibration] = {}
        for task_name, task_meta in meta["tasks"].items():
            slug = task_meta["slug"]
            per_pixel = (
                npz[f"{slug}__per_pixel_var"]
                if task_meta.get("has_per_pixel") and f"{slug}__per_pixel_var" in npz.files
                else None
            )
            # v9 precision-matrix fields
            ivar_maha_dist = (
                npz[f"{slug}__ivar_maha"]
                if task_meta.get("has_ivar_maha") and f"{slug}__ivar_maha" in npz.files
                else None
            )
            evidence_dist = (
                npz[f"{slug}__evidence"]
                if task_meta.get("has_evidence") and f"{slug}__evidence" in npz.files
                else None
            )
            peak_maha_dist = (
                npz[f"{slug}__peak_maha"]
                if task_meta.get("has_peak_maha") and f"{slug}__peak_maha" in npz.files
                else None
            )
            T_null = (
                npz[f"{slug}__T_null"]
                if task_meta.get("has_T_null") and f"{slug}__T_null" in npz.files
                else None
            )
            try:
                ivar_dist = npz[f"{slug}__ivar"]
                peak_dist = npz[f"{slug}__peak"]
                cert_dist = npz[f"{slug}__cert"]
            except KeyError as e:
                raise RuntimeError(
                    f"Calibration file missing key {e} for task '{task_name}'. "
                    f"File may be corrupted."
                ) from e
            tasks[task_name] = TaskCalibration(
                task=task_name,
                n_refs=task_meta["n_refs"],
                ivar_dist=ivar_dist,
                peak_dist=peak_dist,
                cert_dist=cert_dist,
                per_pixel_var=per_pixel,
                ivar_maha_dist=ivar_maha_dist,
                evidence_dist=evidence_dist,
                peak_maha_dist=peak_maha_dist,
                T_null=T_null,
            )

        global_meta = meta.get("global", {})
        global_ivar_maha = (
            npz["__global__ivar_maha"]
            if global_meta.get("has_ivar_maha") and "__global__ivar_maha" in npz.files
            else None
        )
        global_evidence = (
            npz["__global__evidence"]
            if global_meta.get("has_evidence") and "__global__evidence" in npz.files
            else None
        )
        global_peak_maha = (
            npz["__global__peak_maha"]
            if global_meta.get("has_peak_maha") and "__global__peak_maha" in npz.files
            else None
        )
        global_ = TaskCalibration(
            task="__global__",
            n_refs=int(global_meta["n_refs"]),
            ivar_dist=npz["__global__ivar"],
            peak_dist=npz["__global__peak"],
            cert_dist=npz["__global__cert"],
            per_pixel_var=None,
            ivar_maha_dist=global_ivar_maha,
            evidence_dist=global_evidence,
            peak_maha_dist=global_peak_maha,
            T_null=None,
        )

        return cls(
            tasks=tasks,
            global_=global_,
            config_snapshot=meta.get("config_snapshot", {}),
            created_at=meta.get("created_at", ""),
        )


# =============================================================================
# Calibrator
# =============================================================================

class EmpiricalNullCalibrator:
    """Compute per-task empirical null distributions via leave-one-out matching."""

    def __init__(
        self,
        matcher: RoMaMatcher,
        interior_mask: InteriorMask,
        config: WarpScoreConfig,
    ) -> None:
        self.matcher = matcher
        self.interior = interior_mask
        self.config = config

    def calibrate(
        self, high_refs_by_task: dict[str, list[Path]],
    ) -> CalibrationArtifact:
        per_task: dict[str, TaskCalibration] = {}
        for task, paths in tqdm(
            high_refs_by_task.items(), desc="calibrate tasks", unit="task",
        ):
            if len(paths) < 2:
                print(f"[calibrate] skip '{task}': only {len(paths)} ref(s)")
                continue
            if len(paths) < self.config.n_min_refs:
                print(
                    f"[calibrate] WARN '{task}': n={len(paths)} < n_min_refs="
                    f"{self.config.n_min_refs}; p-value resolution will be coarse."
                )
            tc = self._calibrate_one_task(task, paths)
            if tc is not None:
                per_task[task] = tc

        if not per_task:
            raise RuntimeError("Calibration produced no tasks (all skipped).")

        global_ = self._build_global(per_task)
        return CalibrationArtifact(
            tasks=per_task,
            global_=global_,
            config_snapshot=self.config.to_dict(),
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

    # ─────────────────────────────────────────────────────────────────────────

    def _calibrate_one_task(
        self, task: str, paths: list[Path],
    ) -> Optional[TaskCalibration]:
        n = len(paths)
        ivars: list[float] = []
        peaks: list[float] = []
        certs: list[float] = []
        ivar_mahas: list[float] = []
        evidences: list[float] = []
        peak_mahas: list[float] = []
        per_pixel_maps: list[np.ndarray] = []
        T_null_maps: list[np.ndarray] = []

        for q_idx, query_path in enumerate(
            tqdm(paths, desc=f"  {task[:40]}", leave=False, unit="frame"),
        ):
            refs = [p for j, p in enumerate(paths) if j != q_idx]
            stats = self._stats_for_query(query_path, refs)
            if stats is None:
                continue
            ivars.append(stats["ivar"])
            peaks.append(stats["peak"])
            certs.append(stats["cert"])
            if self.config.per_pixel_calibration:
                per_pixel_maps.append(stats["var_map"])
            # v9 maha fields (only present when all refs returned precision)
            if "ivar_maha" in stats:
                ivar_mahas.append(stats["ivar_maha"])
                evidences.append(stats["evidence"])
                peak_mahas.append(stats["peak_maha"])
                if self.config.per_pixel_calibration and "D_map" in stats:
                    T_null_maps.append(stats["D_map"])

        if not ivars:
            print(f"[calibrate] '{task}' produced no valid LOO stats")
            return None

        ivar_arr = np.sort(np.asarray(ivars, dtype=np.float32))
        peak_arr = np.sort(np.asarray(peaks, dtype=np.float32))
        cert_arr = np.sort(np.asarray(certs, dtype=np.float32))

        per_pixel_arr: Optional[np.ndarray] = None
        if per_pixel_maps:
            # Stack then sort along axis 0 for per-pixel empirical CDF lookup
            stacked = np.stack(per_pixel_maps).astype(np.float32)  # (N, H, W)
            per_pixel_arr = np.sort(stacked, axis=0)

        # v9: precision-matrix distributions
        ivar_maha_arr: Optional[np.ndarray] = None
        evidence_arr: Optional[np.ndarray] = None
        peak_maha_arr: Optional[np.ndarray] = None
        T_null_arr: Optional[np.ndarray] = None
        if ivar_mahas:
            ivar_maha_arr = np.sort(np.asarray(ivar_mahas, dtype=np.float32))
            evidence_arr = np.sort(np.asarray(evidences, dtype=np.float32))
            peak_maha_arr = np.sort(np.asarray(peak_mahas, dtype=np.float32))
        if T_null_maps:
            T_null_arr = np.sort(np.stack(T_null_maps).astype(np.float32), axis=0)

        return TaskCalibration(
            task=task,
            n_refs=len(ivars),
            ivar_dist=ivar_arr,
            peak_dist=peak_arr,
            cert_dist=cert_arr,
            per_pixel_var=per_pixel_arr,
            ivar_maha_dist=ivar_maha_arr,
            evidence_dist=evidence_arr,
            peak_maha_dist=peak_maha_arr,
            T_null=T_null_arr,
        )

    # ─────────────────────────────────────────────────────────────────────────

    def _stats_for_query(
        self, query_path: Path, refs: list[Path],
    ) -> Optional[dict]:
        img_bgr = cv2.imread(str(query_path))
        if img_bgr is None:
            return None
        img_bgr = cv2.resize(
            img_bgr, (self.config.vis_size, self.config.vis_size),
            interpolation=cv2.INTER_NEAREST,
        )
        fg_mask = ForegroundMask.from_image(img_bgr)
        if not fg_mask.any():
            return None
        interior_mask = self.interior.apply(fg_mask)
        if not interior_mask.any():
            return None

        warps: list[np.ndarray] = []
        cert_maps: list[np.ndarray] = []
        precisions: list[np.ndarray] = []
        for ref_path in refs:
            try:
                m = self.matcher.match(query_path, ref_path, fg_mask=fg_mask)
                warps.append(m.warp)
                cert_maps.append(m.cert)
                if m.precision is not None:
                    precisions.append(m.precision)
            except Exception as e:
                print(f"[calibrate] match failed {Path(ref_path).name}: {e}")

        if not warps:
            return None

        warps_a = np.stack(warps)        # (K, H, W, 2)
        certs_a = np.stack(cert_maps)    # (K, H, W)
        var_map = CertWeightedStatistics.variance_per_pixel(warps_a, certs_a)
        result: dict = {
            "ivar": CertWeightedStatistics.interior_mean(var_map, interior_mask),
            "peak": CertWeightedStatistics.peak_max_z(var_map, interior_mask),
            "cert": CertWeightedStatistics.mean_cert_interior(certs_a, interior_mask),
            "var_map": var_map,
        }

        # v9: Mahalanobis statistics (only if all refs returned precision matrices)
        if len(precisions) == len(warps):
            precisions_a = np.stack(precisions)  # (K, H, W, 2, 2)
            D_map, logdetΛ_map, _ = MahalanobisStatistics.ivar_per_pixel(warps_a, precisions_a)
            result["ivar_maha"] = MahalanobisStatistics.interior_mean(D_map, interior_mask)
            result["evidence"] = MahalanobisStatistics.interior_mean(-logdetΛ_map, interior_mask)
            result["peak_maha"] = MahalanobisStatistics.peak_max_z(D_map, interior_mask)
            result["D_map"] = D_map

        return result

    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_global(per_task: dict[str, TaskCalibration]) -> TaskCalibration:
        """Pool ivar/peak/cert (and v9 maha fields) across all tasks → fallback distribution."""
        ivars = np.concatenate([tc.ivar_dist for tc in per_task.values()])
        peaks = np.concatenate([tc.peak_dist for tc in per_task.values()])
        certs = np.concatenate([tc.cert_dist for tc in per_task.values()])

        # v9: pool maha distributions if any task has them
        ivar_maha_parts = [
            tc.ivar_maha_dist for tc in per_task.values() if tc.ivar_maha_dist is not None
        ]
        evidence_parts = [
            tc.evidence_dist for tc in per_task.values() if tc.evidence_dist is not None
        ]
        peak_maha_parts = [
            tc.peak_maha_dist for tc in per_task.values() if tc.peak_maha_dist is not None
        ]
        ivar_maha_arr: Optional[np.ndarray] = None
        evidence_arr: Optional[np.ndarray] = None
        peak_maha_arr: Optional[np.ndarray] = None
        if ivar_maha_parts:
            ivar_maha_arr = np.sort(np.concatenate(ivar_maha_parts))
        if evidence_parts:
            evidence_arr = np.sort(np.concatenate(evidence_parts))
        if peak_maha_parts:
            peak_maha_arr = np.sort(np.concatenate(peak_maha_parts))

        return TaskCalibration(
            task="__global__",
            n_refs=len(ivars),
            ivar_dist=np.sort(ivars),
            peak_dist=np.sort(peaks),
            cert_dist=np.sort(certs),
            per_pixel_var=None,
            ivar_maha_dist=ivar_maha_arr,
            evidence_dist=evidence_arr,
            peak_maha_dist=peak_maha_arr,
            T_null=None,
        )


# =============================================================================
# Helpers
# =============================================================================

def _slug(task: str, max_len: int = 80) -> str:
    """Filesystem-safe slug for a task name."""
    safe = (
        task.replace("/", "_")
            .replace("\\", "_")
            .replace("'", "")
            .replace('"', "")
            .replace(" ", "_")
            .replace(".", "")
    )
    return safe[:max_len]
