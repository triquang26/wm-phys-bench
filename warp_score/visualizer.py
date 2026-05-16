"""HeatmapPlotter — render diagnostic figure for one HallucinationResult."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .detector import HallucinationResult


class HeatmapPlotter:
    def __init__(self, vis_size: int = 224) -> None:
        self.vis_size = vis_size

    def plot(
        self,
        query_path: Path,
        result: HallucinationResult,
        save_to: Optional[Path] = None,
    ) -> None:
        img_bgr = cv2.imread(str(query_path))
        img_bgr = cv2.resize(
            img_bgr, (self.vis_size, self.vis_size), interpolation=cv2.INTER_NEAREST,
        )
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        flag = "HALLU" if result.is_hallucination else "CLEAN"
        title = (
            f"{result.task} / {result.frame}  ─ {flag} ─  "
            f"H={result.H_score:.3f}  "
            f"p_comb={result.p_combined:.4f}  refs={result.n_refs}"
        )
        fig.suptitle(title, fontsize=10, fontweight="bold")

        # Panel 1: query image
        axes[0].imshow(img_rgb)
        axes[0].set_title("Query", fontsize=9)
        axes[0].axis("off")

        # Panel 2: heatmap overlay
        if result.heatmap is not None:
            overlay = img_rgb.copy()
            h_n = self._norm01(result.heatmap)
            overlay[:, :, 0] = np.clip(overlay[:, :, 0] * 0.4 + h_n * 0.9, 0, 1)
            overlay[:, :, 1] *= 0.4
            overlay[:, :, 2] *= 0.4
            axes[1].imshow(overlay)
            axes[1].set_title("Heatmap overlay (red = hallu prob)", fontsize=9)
        else:
            axes[1].text(0.5, 0.5, "(no heatmap)", ha="center", va="center")
        axes[1].axis("off")

        # Panel 3: signal table
        axes[2].axis("off")
        lines = ["Per-signal p-values + raw:", ""]
        for name in result.p_per_signal:
            p = result.p_per_signal[name]
            r = result.raw_per_signal.get(name, float("nan"))
            lines.append(f"  {name:6s}  p={p:.4f}  raw={r:.4f}")
        lines.append("")
        lines.append(f"  combined  p={result.p_combined:.4f}")
        lines.append(f"  H_score      ={result.H_score:.4f}")
        lines.append(f"  threshold    >{1.0 - result.p_combined:.4f}")
        axes[2].text(
            0.0, 0.95, "\n".join(lines),
            ha="left", va="top",
            family="monospace", fontsize=9, transform=axes[2].transAxes,
        )

        if save_to is None:
            plt.show()
        else:
            save_to = Path(save_to)
            save_to.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_to, dpi=100, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _norm01(x: np.ndarray) -> np.ndarray:
        lo, hi = float(x.min()), float(x.max())
        if hi - lo < 1e-8:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)
