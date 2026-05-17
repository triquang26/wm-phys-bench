"""Stage 2: Build greedy coreset from patch features and index with FAISS."""
from __future__ import annotations

import logging

import faiss
import h5py
import numpy as np
import torch
from tqdm import tqdm

from config import Config

logger = logging.getLogger("HallucinationEval")


def greedy_coreset_gpu(
    M: np.ndarray,
    target_size: int,
    proj_dim: int = 128,
    seed: int = 42,
) -> np.ndarray:
    """PatchCore Algorithm 1 greedy coreset on GPU.

    M: (N, D) float32
    Returns: (target_size, D) float32
    """
    rng = np.random.default_rng(seed)
    N, D = M.shape
    target_size = min(target_size, N)

    W = rng.normal(0, 1.0 / np.sqrt(proj_dim), (D, proj_dim)).astype(np.float32)
    M_t = torch.from_numpy(M).cuda()
    W_t = torch.from_numpy(W).cuda()
    M_proj = M_t @ W_t                              # (N, proj_dim)

    selected: list[int] = []
    min_dists = torch.full((N,), float("inf"), device="cuda")

    first = int(rng.integers(0, N))
    selected.append(first)
    diff = M_proj - M_proj[first]
    min_dists = torch.minimum(min_dists, (diff ** 2).sum(dim=1))
    min_dists[first] = 0.0

    for _ in tqdm(range(1, target_size), desc="Greedy coreset", leave=False):
        idx = int(torch.argmax(min_dists).item())
        selected.append(idx)
        diff = M_proj - M_proj[idx]
        new_dists = (diff ** 2).sum(dim=1)
        min_dists = torch.minimum(min_dists, new_dists)
        min_dists[idx] = 0.0

    return M[np.array(selected)]


class CoresetBuilder:
    """Builds memory-bank coreset from Stage 1 HDF5, then indexes with FAISS."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg

    def run(self) -> None:
        if self._cfg.faiss_path.exists() and self._cfg.mc_path.exists():
            logger.info("[Stage 2] Coreset + FAISS exist — skipping.")
            return
        logger.info("[Stage 2] Building coreset …")
        MC = self._build_coreset()
        np.save(self._cfg.mc_path, MC.astype(np.float32))
        logger.info(f"[Stage 2] Coreset saved: {MC.shape}")

        index = self._build_faiss(MC.astype(np.float32))
        faiss.write_index(index, str(self._cfg.faiss_path))
        logger.info(f"[Stage 2] FAISS index saved → {self._cfg.faiss_path}")

    def _build_coreset(self) -> np.ndarray:
        with h5py.File(self._cfg.h5_path, "r") as f:
            N_total = int(f.attrs["total_patches"])
            D = f["M"].shape[1]
            cfg = self._cfg

            if N_total > cfg.skip_kmeans_below:
                # Two-stage: K-means then greedy coreset
                rng = np.random.default_rng(42)
                sub_n = min(5_000_000, N_total)
                sub_idx = np.sort(rng.choice(N_total, sub_n, replace=False))
                logger.info(f"[Stage 2] Loading {sub_n} patches for K-means …")
                M_sub = f["M"][sub_idx].astype(np.float32)

                logger.info(
                    f"[Stage 2] K-means: {sub_n} → {cfg.coreset_intermediate_size} …"
                )
                km = faiss.Kmeans(
                    D, cfg.coreset_intermediate_size,
                    niter=20, gpu=True, seed=42, verbose=False,
                )
                km.train(M_sub)
                centroids = km.centroids.astype(np.float32)
                del M_sub

                logger.info(
                    f"[Stage 2] Greedy coreset: "
                    f"{cfg.coreset_intermediate_size} → {cfg.coreset_final_size} …"
                )
                return greedy_coreset_gpu(
                    centroids, cfg.coreset_final_size, cfg.coreset_proj_dim
                )
            else:
                # Small dataset — load entirely and run greedy coreset directly
                logger.info(
                    f"[Stage 2] Loading {N_total} patches for direct coreset …"
                )
                M = f["M"][:N_total].astype(np.float32)

            final_size = min(cfg.coreset_final_size, N_total)
            logger.info(
                f"[Stage 2] Greedy coreset: {N_total} → {final_size} …"
            )
            return greedy_coreset_gpu(M, final_size, cfg.coreset_proj_dim)

    def _build_faiss(self, MC: np.ndarray) -> faiss.Index:
        D = MC.shape[1]
        if len(MC) < 100_000:
            # Exact NN — fast and reliable for small coreset
            index = faiss.IndexFlatL2(D)
            index.add(MC)
            return index
        nlist = max(32, int(np.sqrt(len(MC))))
        quantizer = faiss.IndexFlatL2(D)
        index = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_L2)
        index.train(MC)
        index.add(MC)
        index.nprobe = 16
        return index
