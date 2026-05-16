"""Evaluator — compute AUROC, AP, FPR@95TPR from labels + predictions.

Expected labels CSV format:
    image_id,task,frame,split,label
    0_Open_the_box__frame_0001,0_Open the box,frame_0001,low,1
    ...

Predictions CSV: as written by WarpVarianceDetector → summary.csv.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


@dataclass
class EvalReport:
    auroc: float
    ap: float
    fpr_at_95_tpr: float
    optimal_threshold: float
    n_total: int
    n_positive: int
    per_task: dict[str, dict[str, float]] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "auroc": self.auroc,
                    "ap": self.ap,
                    "fpr_at_95_tpr": self.fpr_at_95_tpr,
                    "optimal_threshold": self.optimal_threshold,
                    "n_total": self.n_total,
                    "n_positive": self.n_positive,
                    "per_task": self.per_task,
                },
                f,
                indent=2,
            )

    def print_summary(self) -> None:
        print(
            f"AUROC={self.auroc:.4f}  AP={self.ap:.4f}  "
            f"FPR@95TPR={self.fpr_at_95_tpr:.4f}  "
            f"τ*={self.optimal_threshold:.4f}  "
            f"n={self.n_total} ({self.n_positive} pos)"
        )


class Evaluator:
    def __init__(self, labels_csv: Path, predictions_csv: Path) -> None:
        self.labels_csv = Path(labels_csv)
        self.predictions_csv = Path(predictions_csv)

    def evaluate(self) -> EvalReport:
        labels = self._load_labels()
        preds = self._load_predictions()

        # Inner-join on (task, frame)
        keys = set(labels.keys()) & set(preds.keys())
        if not keys:
            raise RuntimeError(
                "No overlap between labels and predictions on (task, frame)."
            )

        y_true = np.array([labels[k] for k in keys], dtype=np.int32)
        y_score = np.array([preds[k] for k in keys], dtype=np.float64)

        return EvalReport(
            auroc=float(roc_auc_score(y_true, y_score)),
            ap=float(average_precision_score(y_true, y_score)),
            fpr_at_95_tpr=self._fpr_at_tpr(y_true, y_score, target_tpr=0.95),
            optimal_threshold=self._optimal_threshold(y_true, y_score),
            n_total=len(y_true),
            n_positive=int(y_true.sum()),
            per_task=self._per_task_metrics(labels, preds, keys),
        )

    # ─────────────────────────────────────────────────────────────────────────

    def _load_labels(self) -> dict[tuple[str, str], int]:
        out: dict[tuple[str, str], int] = {}
        with open(self.labels_csv, "r") as f:
            for row in csv.DictReader(f):
                key = (row["task"], row["frame"])
                out[key] = int(row["label"])
        return out

    def _load_predictions(self) -> dict[tuple[str, str], float]:
        out: dict[tuple[str, str], float] = {}
        with open(self.predictions_csv, "r") as f:
            for row in csv.DictReader(f):
                key = (row["task"], row["frame"])
                out[key] = float(row["H_score"])
        return out

    @staticmethod
    def _fpr_at_tpr(y_true, y_score, target_tpr: float = 0.95) -> float:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        # First threshold where TPR >= target
        idx = np.argmax(tpr >= target_tpr)
        if tpr[idx] < target_tpr:
            return 1.0
        return float(fpr[idx])

    @staticmethod
    def _optimal_threshold(y_true, y_score) -> float:
        precision, recall, thresh = precision_recall_curve(y_true, y_score)
        f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
        if len(thresh) == 0:
            return 0.5
        # thresh has len(precision)-1 entries
        f1 = f1[:-1]
        return float(thresh[int(np.argmax(f1))])

    @staticmethod
    def _per_task_metrics(labels, preds, keys) -> dict[str, dict[str, float]]:
        by_task: dict[str, list[tuple[int, float]]] = {}
        for k in keys:
            by_task.setdefault(k[0], []).append((labels[k], preds[k]))
        out: dict[str, dict[str, float]] = {}
        for task, pairs in by_task.items():
            y = np.array([p[0] for p in pairs])
            s = np.array([p[1] for p in pairs])
            metrics: dict[str, float] = {"n": float(len(y)), "n_pos": float(y.sum())}
            if y.sum() > 0 and y.sum() < len(y):
                metrics["auroc"] = float(roc_auc_score(y, s))
                metrics["ap"] = float(average_precision_score(y, s))
            out[task] = metrics
        return out
