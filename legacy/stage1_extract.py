"""Stage 1: Extract patch features from reference images and stream to HDF5."""
from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import Config

logger = logging.getLogger("HallucinationEval")

PATCHES_PER_IMAGE = 256   # 16×16 with 224×224 input, patch_size=14
PATCH_DIM = 2048          # 2× 1024 (ViT-L/14 hidden dim)


@torch.no_grad()
def extract_patches_batch(
    images: torch.Tensor,
    backbone,
    layers: tuple[int, int] = (8, 16),
    p: int = 3,
) -> torch.Tensor:
    """
    images: (B, 3, 224, 224) on GPU
    Returns: (B*256, 2048) FP32 on CPU
    """
    feats = backbone.get_intermediate_layers(
        images, n=list(layers), reshape=True, return_class_token=False
    )
    fmap_lo, fmap_hi = feats  # each (B, 1024, 16, 16)

    agg_lo = F.avg_pool2d(fmap_lo, kernel_size=p, stride=1, padding=p // 2)
    agg_hi = F.avg_pool2d(fmap_hi, kernel_size=p, stride=1, padding=p // 2)

    if agg_hi.shape[-2:] != agg_lo.shape[-2:]:
        agg_hi = F.interpolate(
            agg_hi, size=agg_lo.shape[-2:], mode="bilinear", align_corners=False
        )

    fmap = torch.cat([agg_lo, agg_hi], dim=1)   # (B, 2048, 16, 16)
    B, D, h, w = fmap.shape
    patches = fmap.permute(0, 2, 3, 1).reshape(B * h * w, D)
    return patches.cpu()


class PatchExtractor:
    """Extracts DINOv2 patch features from all reference images and writes to HDF5."""

    def __init__(self, backbone, cfg: Config) -> None:
        self._backbone = backbone
        self._cfg = cfg

    def run(self, dataloader) -> None:
        if self._cfg.h5_path.exists():
            logger.info(f"[Stage 1] HDF5 exists at {self._cfg.h5_path} — skipping.")
            return
        logger.info("[Stage 1] Extracting patch features → HDF5 …")
        self._extract_to_hdf5(dataloader)
        self._verify()
        logger.info(f"[Stage 1] Done → {self._cfg.h5_path}")

    def _extract_to_hdf5(self, dataloader) -> None:
        N_total = len(dataloader.dataset)
        N_patches = N_total * PATCHES_PER_IMAGE
        device = next(self._backbone.parameters()).device

        with h5py.File(self._cfg.h5_path, "w") as f:
            dset = f.create_dataset(
                "M",
                shape=(N_patches, PATCH_DIM),
                dtype=np.float16,
                chunks=(PATCHES_PER_IMAGE * 32, PATCH_DIM),
            )
            idx_map = f.create_dataset(
                "img_idx", shape=(N_patches,), dtype=np.int32
            )

            offset = 0
            for images, indices in tqdm(dataloader, desc="Stage 1"):
                images = images.to(device, non_blocking=True)
                patches = extract_patches_batch(
                    images, self._backbone,
                    layers=self._cfg.dino_layers,
                    p=self._cfg.patch_agg_kernel,
                )
                patches_np = patches.half().numpy()
                n = patches_np.shape[0]
                dset[offset:offset + n] = patches_np

                B = images.shape[0]
                for b, img_idx in enumerate(indices):
                    s = offset + b * PATCHES_PER_IMAGE
                    idx_map[s:s + PATCHES_PER_IMAGE] = int(img_idx)

                offset += n

            f.attrs["total_patches"] = offset

    def _verify(self) -> None:
        with h5py.File(self._cfg.h5_path, "r") as f:
            sample = f["M"][:1000].astype(np.float32)
        assert not np.isnan(sample).any(), "NaN in patch features"
        assert sample.std() > 0.01, "Degenerate patch features (low variance)"
        logger.info(f"[Stage 1] Verify OK — std={sample.std():.4f}")
