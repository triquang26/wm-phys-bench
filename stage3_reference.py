"""Stage 3: Extract global CLS embeddings and build stratified reference pool."""
from __future__ import annotations

import logging

import faiss
import numpy as np
import torch
from sklearn.cluster import KMeans
from tqdm import tqdm

from config import Config
from dataset import ImageDataset

logger = logging.getLogger("HallucinationEval")


class ReferencePoolBuilder:
    """Clusters CLS tokens from reference images and selects medoid + boundary samples."""

    def __init__(self, backbone, dataset: ImageDataset, cfg: Config) -> None:
        self._backbone = backbone
        self._dataset = dataset
        self._cfg = cfg

    def run(self, dataloader) -> None:
        if self._cfg.ref_indices_path.exists():
            logger.info("[Stage 3] Reference pool exists — skipping.")
            return
        logger.info("[Stage 3] Building reference pool …")

        global_feats = self._extract_global_feats(dataloader)
        np.save(self._cfg.global_feats_path, global_feats)
        logger.info(f"[Stage 3] Global feats: {global_feats.shape}")

        n_images = len(self._dataset.high_images)
        n_clusters = self._cfg.n_clusters if self._cfg.n_clusters > 0 \
            else len(self._dataset.unique_tasks("high"))
        # sklearn KMeans requires at least n_clusters samples
        n_clusters = min(n_clusters, n_images)
        logger.info(f"[Stage 3] K-means clustering: {n_images} → {n_clusters} clusters …")

        ref_indices = self._cluster_and_select(global_feats, n_clusters)
        np.save(self._cfg.ref_indices_path, ref_indices)
        logger.info(f"[Stage 3] {len(ref_indices)} reference images selected.")

        ref_feats = global_feats[ref_indices].copy().astype(np.float32)
        faiss.normalize_L2(ref_feats)
        np.save(self._cfg.ref_feats_path, ref_feats)
        logger.info(f"[Stage 3] Done → {self._cfg.ref_indices_path}")

    @torch.no_grad()
    def _extract_global_feats(self, dataloader) -> np.ndarray:
        device = next(self._backbone.parameters()).device
        all_cls: list[np.ndarray] = []
        for images, _ in tqdm(dataloader, desc="Stage 3 CLS"):
            images = images.to(device, non_blocking=True)
            out = self._backbone.forward_features(images)
            cls = out["x_norm_clstoken"].cpu().numpy()
            all_cls.append(cls)
        return np.concatenate(all_cls, axis=0).astype(np.float32)

    def _cluster_and_select(
        self, global_feats: np.ndarray, n_clusters: int
    ) -> np.ndarray:
        # sklearn KMeans: works on any GPU/CPU, no architecture constraints
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = km.fit_predict(global_feats)
        centroids = km.cluster_centers_

        ref_indices: list[int] = []
        for c in range(n_clusters):
            c_idx = np.where(labels == c)[0]
            if len(c_idx) == 0:
                continue
            c_feats = global_feats[c_idx]
            dists = np.linalg.norm(c_feats - centroids[c], axis=1)

            # Medoid (closest to centroid)
            ref_indices.append(int(c_idx[np.argmin(dists)]))

            # Boundary samples (farthest from centroid)
            n_bound = min(self._cfg.n_boundary_per_cluster, len(c_idx) - 1)
            if n_bound > 0:
                boundary_local = np.argsort(dists)[-n_bound:]
                ref_indices.extend(c_idx[boundary_local].tolist())

        return np.unique(ref_indices)
