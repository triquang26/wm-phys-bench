"""Unit tests for ForegroundMask + InteriorMask."""
from __future__ import annotations

import numpy as np
import pytest

from warp_score.mask import ForegroundMask, InteriorMask


def test_foreground_mask_detects_non_bg_pixels():
    img = np.full((10, 10, 3), 127, dtype=np.uint8)
    img[3:7, 3:7] = 200
    fg = ForegroundMask.from_image(img)
    assert fg.shape == (10, 10)
    assert fg.sum() == 16  # 4×4 region of non-(127,127,127)


def test_foreground_mask_partial_match_is_foreground():
    img = np.full((4, 4, 3), 127, dtype=np.uint8)
    img[0, 0] = [127, 127, 128]   # one channel differs
    fg = ForegroundMask.from_image(img)
    assert fg[0, 0]


def test_interior_mask_erodes_boundary():
    fg = np.ones((20, 20), dtype=bool)
    interior = InteriorMask(erosion_k=5).apply(fg)
    # 20x20 eroded by 5 → 16x16 interior
    assert interior.sum() == 16 * 16


def test_interior_mask_rejects_invalid_erosion():
    with pytest.raises(ValueError):
        InteriorMask(erosion_k=0)
