"""SAM3-based foreground segmentation for video frames.

Segments robot arm + task-relevant objects from each frame using SAM3 text
prompts. Background pixels are set to (127, 127, 127) — the convention used
by ForegroundMask in warp_score/mask.py.

Construction:
    - prompts:        Override the full prompt list (default = robot-only).
    - extra_prompts:  Additive prompts UNIONED with the default robot set
                      (use for task-aware objects: ["drawer", "cup", ...]).
    - fallback:       What to return when SAM3 finds no mask:
        * "keep"  → original frame (legacy GR-1 behaviour; bg leaks as fg)
        * "gray"  → all-(127,127,127) frame (zero signal — safe default for DROID)
        * "none"  → return None (caller decides; e.g. drop the frame)
"""
from __future__ import annotations

from typing import Literal, Optional

import cv2
import numpy as np


_ROBOT_PROMPTS: list[str] = [
    "robot arm",
    "robotic hand",
    "gripper",
    "mechanical finger",
]

BG_COLOR: tuple[int, int, int] = (127, 127, 127)

Fallback = Literal["keep", "gray", "none"]


class VideoFrameSegmenter:
    """SAM3-based segmenter for single BGR video frames.

    Model is lazy-loaded on first call to :meth:`segment_frame`.
    """

    def __init__(
        self,
        model_id: str = "facebook/sam3",
        prompts: Optional[list[str]] = None,
        extra_prompts: Optional[list[str]] = None,
        threshold: float = 0.3,
        fallback: Fallback = "keep",
    ) -> None:
        self._model_id = model_id
        base = list(prompts) if prompts is not None else list(_ROBOT_PROMPTS)
        if extra_prompts:
            for p in extra_prompts:
                if p and p not in base:
                    base.append(p)
        self._prompts = base
        self._threshold = threshold
        self._fallback: Fallback = fallback

        self._processor = None
        self._model = None
        self._device: Optional[str] = None

    @property
    def prompts(self) -> list[str]:
        return list(self._prompts)

    # ── Lazy loader ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import Sam3Model, Sam3Processor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = Sam3Processor.from_pretrained(self._model_id)
        self._model = Sam3Model.from_pretrained(self._model_id).to(device)
        self._model.eval()
        self._device = device

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _segment_one(self, pil_image, prompt: str) -> Optional[np.ndarray]:
        """Run SAM3 for a single prompt.

        Returns a boolean mask (H, W) if any instance was detected, else None.
        """
        import torch

        inputs = self._processor(
            images=pil_image,
            text=prompt,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        result = self._processor.post_process_instance_segmentation(
            outputs,
            threshold=self._threshold,
            mask_threshold=self._threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        masks = result["masks"]
        if len(masks) == 0:
            return None

        return masks.any(dim=0).cpu().numpy()  # (H, W) bool

    def _build_union_mask(self, pil_image) -> Optional[np.ndarray]:
        """Union masks from all prompts. Returns uint8 alpha (0/255) or None."""
        h, w = pil_image.size[1], pil_image.size[0]
        combined = np.zeros((h, w), dtype=bool)

        for prompt in self._prompts:
            mask = self._segment_one(pil_image, prompt)
            if mask is not None:
                combined |= mask

        if not combined.any():
            return None

        return combined.astype(np.uint8) * 255

    # ── Public API ────────────────────────────────────────────────────────────

    def segment_frame(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        """Segment a BGR frame and replace background with (127, 127, 127).

        Returns:
            HxWx3 uint8 BGR with bg masked to BG_COLOR. When SAM3 finds nothing
            the return depends on `self._fallback`:
              * "keep" → original frame (caution: bg leaks as fg)
              * "gray" → all-gray frame (no signal, safer for DROID)
              * "none" → None
        """
        from PIL import Image

        self._load()

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        alpha = self._build_union_mask(pil_image)
        if alpha is None:
            if self._fallback == "keep":
                return bgr
            if self._fallback == "gray":
                return np.full_like(bgr, BG_COLOR, dtype=np.uint8)
            if self._fallback == "none":
                return None
            raise ValueError(f"unknown fallback mode: {self._fallback}")

        bg = np.full_like(bgr, BG_COLOR, dtype=np.uint8)
        fg_mask = alpha > 0  # (H, W)
        result = bg.copy()
        result[fg_mask] = bgr[fg_mask]
        return result
