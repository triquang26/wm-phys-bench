"""Foreground / interior mask utilities.

Background convention: SAM3 segmenter sets background pixels to exactly (127,127,127).
Interior mask: foreground eroded by `erosion_k` pixels to strip boundary noise
where thin edges contribute spurious warp variance.
"""
from __future__ import annotations

import cv2
import numpy as np


class ForegroundMask:
    BG_VALUE: tuple[int, int, int] = (127, 127, 127)

    @classmethod
    def from_image(cls, img_bgr: np.ndarray) -> np.ndarray:
        """Return (H, W) bool array: True = foreground."""
        return ~np.all(img_bgr == cls.BG_VALUE, axis=-1)


class InteriorMask:
    """Erode a foreground mask by `erosion_k` pixels."""

    def __init__(self, erosion_k: int = 10) -> None:
        if erosion_k < 1:
            raise ValueError(f"erosion_k must be >= 1, got {erosion_k}")
        self.erosion_k = erosion_k
        self._kernel = np.ones((erosion_k, erosion_k), dtype=np.uint8)

    def apply(self, fg_mask: np.ndarray) -> np.ndarray:
        """Return (H, W) bool: foreground after erosion."""
        return cv2.erode(
            fg_mask.astype(np.uint8),
            self._kernel,
            iterations=1,
        ).astype(bool)
