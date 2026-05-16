"""Stage 4: Build reference consistency graph via pairwise RoMa matching."""
from __future__ import annotations

import logging
import os

import faiss
import numpy as np
from tqdm import tqdm

from config import Config
from dataset import ImageDataset
from roma_utils import roma_match_score

logger = logging.getLogger("HallucinationEval")


class GraphBuilder:
    """Computes pairwise RoMa match scores between reference images.

    The graph S_ref[i,j] captures how geometrically consistent two reference
    images are. Per-node mean/std (mu_ref, sigma_ref) are used later for
    z-score normalization in the online detector.
    """

    def __init__(self, matcher, dataset: ImageDataset, cfg: Config) -> None:
        self._matcher = matcher
        self._dataset = dataset
        self._cfg = cfg

    def run(self) -> None:
        done = (
            self._cfg.s_ref_path.exists()
            and self._cfg.mu_ref_path.exists()
            and self._cfg.sigma_ref_path.exists()
        )
        if done:
            logger.info("[Stage 4] Reference graph exists — skipping.")
            return
        logger.info("[Stage 4] Building reference consistency graph …")

        ref_indices = np.load(self._cfg.ref_indices_path)
        ref_feats = np.load(self._cfg.ref_feats_path)  # already L2-normalized
        ref_paths = [str(self._dataset.high_images[i].path) for i in ref_indices]

        S_ref, mu_ref, sigma_ref = self._build_graph(ref_indices, ref_feats, ref_paths)

        np.save(self._cfg.s_ref_path, S_ref)
        np.save(self._cfg.mu_ref_path, mu_ref)
        np.save(self._cfg.sigma_ref_path, sigma_ref)
        logger.info(f"[Stage 4] Done → {self._cfg.s_ref_path}")

    def _build_graph(
        self,
        ref_indices: np.ndarray,
        ref_feats: np.ndarray,
        ref_paths: list[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        N_refs = len(ref_indices)
        cfg = self._cfg

        # FAISS neighbor search on reference embeddings
        ref_index = faiss.IndexFlatIP(ref_feats.shape[1])
        ref_index.add(ref_feats)
        _, neighbors = ref_index.search(ref_feats, cfg.top_k_neighbors + 1)

        # Resume from checkpoint if available
        ckpt_path = cfg.artifacts_dir / "S_ref_checkpoint.npz"
        if ckpt_path.exists():
            ckpt = np.load(ckpt_path)
            S_ref = ckpt["S_ref"]
            computed_pairs: set[tuple[int, int]] = set(map(tuple, ckpt["pairs"].tolist()))
            logger.info(f"[Stage 4] Resuming from {len(computed_pairs)} computed pairs.")
        else:
            S_ref = np.zeros((N_refs, N_refs), dtype=np.float32)
            computed_pairs = set()

        # Collect unique undirected pairs
        all_pairs: list[tuple[int, int]] = []
        for i in range(N_refs):
            for j in neighbors[i, 1:]:  # skip self (index 0)
                pair = (min(i, int(j)), max(i, int(j)))
                if pair not in computed_pairs:
                    all_pairs.append(pair)

        logger.info(f"[Stage 4] {len(all_pairs)} pairs to process …")
        for k, (i, j) in enumerate(tqdm(all_pairs, desc="Stage 4")):
            score = roma_match_score(
                ref_paths[i], ref_paths[j], self._matcher,
                num_samples=cfg.roma_num_samples,
            )
            S_ref[i, j] = score
            S_ref[j, i] = score
            computed_pairs.add((i, j))

            if (k + 1) % cfg.checkpoint_every == 0:
                np.savez(
                    ckpt_path,
                    S_ref=S_ref,
                    pairs=np.array(list(computed_pairs), dtype=np.int32),
                )

        # Per-node statistics over neighbor scores
        mu_ref = np.zeros(N_refs, dtype=np.float32)
        sigma_ref = np.zeros(N_refs, dtype=np.float32)
        for i in range(N_refs):
            nb_scores = S_ref[i, neighbors[i, 1:]]
            nb_scores = nb_scores[nb_scores > 0]
            if len(nb_scores) > 1:
                mu_ref[i] = float(nb_scores.mean())
                sigma_ref[i] = float(nb_scores.std())
            else:
                mu_ref[i] = 0.5
                sigma_ref[i] = 0.15

        # Clean up checkpoint
        if ckpt_path.exists():
            os.remove(ckpt_path)

        return S_ref, mu_ref, sigma_ref
