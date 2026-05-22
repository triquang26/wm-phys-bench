"""Per-frame content-based adaptive reference selection using DINOv2 k-NN.

Speed-invariant hallucination detection null model:
  For each test frame, pick the top-k most visually similar reference frames
  (via DINOv2 CLS-token cosine similarity) and compute D_map only against those.

  This gives P(D_map | task_state) instead of the marginal P(D_map), making the
  null model aware of where the robot is in the task regardless of execution speed.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ImageNet normalization constants
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_DINO_SIZE = 224


class DinoFeatureExtractor:
    """DINOv2 ViT-S/14 — extracts L2-normalized CLS tokens.

    Loaded lazily on first call to keep import cost zero when adaptive
    ref selection is disabled.
    """

    def __init__(self, model: str = "dinov2_vits14") -> None:
        self.model_name = model
        self._model = None
        self._device: Optional[str] = None

    # ------------------------------------------------------------------

    def _load(self) -> None:
        import torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = torch.hub.load(
            "facebookresearch/dinov2", self.model_name, verbose=False
        )
        self._model.eval().to(self._device)

    @staticmethod
    def _preprocess(img_bgr: np.ndarray) -> "torch.Tensor":
        """Pad-to-square (gray 127) BEFORE resize — matches RoMaMatcher/fg_mask
        coordinate system, preserves aspect ratio."""
        import torch
        H, W = img_bgr.shape[:2]
        if H != W:
            side = max(H, W)
            pad_h, pad_w = side - H, side - W
            top, bottom = pad_h // 2, pad_h - pad_h // 2
            left, right = pad_w // 2, pad_w - pad_w // 2
            img_bgr = cv2.copyMakeBorder(img_bgr, top, bottom, left, right,
                                         cv2.BORDER_CONSTANT, value=(127, 127, 127))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (_DINO_SIZE, _DINO_SIZE), interpolation=cv2.INTER_LINEAR)
        x = img_rgb.astype(np.float32) / 255.0
        x = (x - _IMAGENET_MEAN) / _IMAGENET_STD          # HWC
        x = torch.from_numpy(x.transpose(2, 0, 1))         # CHW
        return x

    def extract(self, frames: list[Path], batch_size: int = 16) -> np.ndarray:
        """Extract CLS tokens for a list of frame paths.

        Returns:
            feats: (N, D) float32 array, L2-normalized (cosine-ready).
        """
        import torch

        if self._model is None:
            self._load()

        imgs = []
        for p in frames:
            bgr = cv2.imread(str(p))
            if bgr is None:
                raise FileNotFoundError(f"DinoFeatureExtractor: cannot read {p}")
            imgs.append(self._preprocess(bgr))

        all_feats: list[np.ndarray] = []
        for i in range(0, len(imgs), batch_size):
            batch = torch.stack(imgs[i : i + batch_size]).to(self._device)
            with torch.no_grad():
                feats = self._model(batch)                  # (B, D)
            feats = feats.float().cpu().numpy()
            # L2-normalize
            norms = np.linalg.norm(feats, axis=1, keepdims=True).clip(min=1e-8)
            all_feats.append((feats / norms).astype(np.float32))

        return np.concatenate(all_feats, axis=0)            # (N, D)


# ─────────────────────────────────────────────────────────────────────────────


class AdaptiveRefSelector:
    """Speed-invariant per-frame reference selection via DINOv2 k-NN.

    Usage (inference):
        selector = AdaptiveRefSelector(DinoFeatureExtractor())
        ref_feats = selector.build_cache(task, ref_paths, cache_dir)
        query_feat = selector.extractor.extract([query_path])[0]
        top_k_idx = selector.select_for_query(query_feat, ref_feats, k=15)
        active_refs = [ref_paths[i] for i in top_k_idx]

    Usage (calibration LOO):
        ref_feats = selector.build_cache(task, paths, cache_dir)  # (N, D)
        for q_idx, query_path in enumerate(paths):
            cand_idx = [j for j in range(n) if j != q_idx]
            top_k = selector.select_for_query(
                ref_feats[q_idx], ref_feats[cand_idx], k
            )
            refs = [paths[cand_idx[i]] for i in top_k]
    """

    def __init__(self, extractor: DinoFeatureExtractor) -> None:
        self.extractor = extractor

    # ------------------------------------------------------------------

    def build_cache(
        self,
        task: str,
        ref_paths: list[Path],
        cache_dir: Path,
    ) -> np.ndarray:
        """Compute and save DINOv2 features for ref_paths.

        Returns feats (N, D) float32 L2-normalized.
        Re-uses cached features when cache_key matches.
        """
        cache_dir.mkdir(parents=True, exist_ok=True)
        slug = _task_slug(task)
        cache_path = cache_dir / f"{slug}.npz"
        key = _cache_key(ref_paths, self.extractor.model_name)

        if cache_path.exists():
            data = np.load(cache_path, allow_pickle=True)
            if str(data["cache_key"]) == key:
                return data["feats"].astype(np.float32)

        print(f"[dino-cache] building features for task '{task[:50]}' ({len(ref_paths)} refs)…")
        feats = self.extractor.extract(ref_paths)
        np.savez(
            cache_path,
            paths=np.array([str(p) for p in ref_paths], dtype=object),
            feats=feats,
            cache_key=np.str_(key),
            dino_model=np.str_(self.extractor.model_name),
        )
        return feats

    def load_cache(
        self,
        task: str,
        ref_paths: list[Path],
        cache_dir: Path,
    ) -> Optional[np.ndarray]:
        """Return cached feats if valid, else None."""
        slug = _task_slug(task)
        cache_path = cache_dir / f"{slug}.npz"
        if not cache_path.exists():
            return None
        data = np.load(cache_path, allow_pickle=True)
        if str(data["cache_key"]) != _cache_key(ref_paths, self.extractor.model_name):
            return None
        return data["feats"].astype(np.float32)

    def select_for_query(
        self,
        query_feat: np.ndarray,       # (D,) L2-normalized
        ref_feats: np.ndarray,        # (M, D) L2-normalized
        k: int,
    ) -> list[int]:
        """Return indices of top-k refs by cosine similarity (highest first).

        Since both vectors are L2-normalized, cosine sim = dot product.
        k is clamped to min(k, M).
        """
        k = min(k, len(ref_feats))
        sims = ref_feats @ query_feat                           # (M,)
        # argpartition is O(M) — faster than full sort for large M
        top_k = np.argpartition(sims, -k)[-k:]
        top_k = top_k[np.argsort(sims[top_k])[::-1]]          # sort desc
        return top_k.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers


def _task_slug(task: str) -> str:
    return re.sub(r"[^\w]", "_", task)[:80]


def _cache_key(paths: list[Path], model_name: str) -> str:
    key_str = model_name + "|" + "|".join(sorted(str(p) for p in paths))
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]
