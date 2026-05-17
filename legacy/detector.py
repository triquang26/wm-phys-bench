"""Online hallucination detector: PatchCore + Reference Graph voting."""
from __future__ import annotations

import logging

import cv2
import faiss
import numpy as np
import torch
from tqdm import tqdm

from config import Config
from dataset import ImageDataset, ImageRecord, RobotDataset
from roma_utils import roma_match_score
from stage1_extract import extract_patches_batch

logger = logging.getLogger("HallucinationEval")


class HallucinationDetector:
    """Fuses PatchCore anomaly score with Reference Graph voting.

    Offline artifacts must exist before constructing this class (run Stages 1-4).
    Call calibrate() once on nominal (high) images before calling detect().

    Scoring:
      score_patchcore = s_pc_raw / tau_raw   (linear ratio; > 1.0 means anomalous)
      score_graph     = sigmoid(-z_bar)       (> 0.5 means anomalous)
      score           = alpha * clip(S_pc,1) + (1-alpha) * S_graph
      is_hallucination = s_pc_raw > tau_raw
    """

    def __init__(
        self,
        backbone,
        matcher,
        dataset: ImageDataset,
        cfg: Config,
    ) -> None:
        self._backbone = backbone
        self._matcher = matcher
        self._dataset = dataset
        self._cfg = cfg
        self._device = next(backbone.parameters()).device

        # PatchCore memory bank
        self._MC = np.load(cfg.mc_path)
        self._faiss_index = faiss.read_index(str(cfg.faiss_path))

        # Reference pool
        self._ref_indices = np.load(cfg.ref_indices_path)
        self._ref_feats = np.load(cfg.ref_feats_path)    # L2-normalized
        self._mu_ref = np.load(cfg.mu_ref_path)
        self._sigma_ref = np.load(cfg.sigma_ref_path)
        self._ref_paths = [
            str(dataset.high_images[i].path) for i in self._ref_indices
        ]

        self._ref_index = faiss.IndexFlatIP(self._ref_feats.shape[1])
        self._ref_index.add(self._ref_feats)

        # Calibration threshold — raw PatchCore scale
        self._tau_raw: float | None = None

        if cfg.calibration_path.exists():
            cal = np.load(cfg.calibration_path)
            self._tau_raw = float(cal["tau_raw"])
            logger.info(f"[Detector] Calibration loaded: tau_raw={self._tau_raw:.4f}")

    # ── Calibration ───────────────────────────────────────────────────────────

    def calibrate(self, high_dataloader) -> None:
        """Single-pass calibration on nominal (high) images.

        tau_raw = (1-fpr) percentile of raw PatchCore scores on high images.
        score_patchcore = s_pc_raw / tau_raw  (linear, can exceed 1.0).
        is_hallucination = s_pc_raw > tau_raw.
        """
        if self._cfg.calibration_path.exists():
            logger.info("[Detector] Calibration file exists — skipping.")
            cal = np.load(self._cfg.calibration_path)
            self._tau_raw = float(cal["tau_raw"])
            logger.info(f"[Detector] tau_raw={self._tau_raw:.4f}")
            return

        logger.info("[Detector] Calibrating raw PatchCore scores on high images …")
        pc_raws: list[float] = []
        device = self._device
        cfg = self._cfg

        with torch.no_grad():
            for images, _ in tqdm(high_dataloader, desc="Calibrate"):
                images = images.to(device, non_blocking=True)
                for b in range(images.shape[0]):
                    pc_raws.append(self._patchcore_raw(images[b:b + 1]))

        self._tau_raw = float(np.percentile(pc_raws, 100 * (1 - cfg.target_fpr)))
        logger.info(
            f"[Detector] tau_raw={self._tau_raw:.4f}  "
            f"mean={np.mean(pc_raws):.4f}  std={np.std(pc_raws):.4f}  "
            f"n={len(pc_raws)}"
        )
        np.savez(cfg.calibration_path, tau_raw=self._tau_raw)

    # ── Detection ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def detect(self, image_tensor: torch.Tensor, image_path: str) -> dict:
        """
        image_tensor: (1, 3, 224, 224) normalized, on device
        image_path:   str path for RoMa matching

        Returns dict with: is_hallucination, score, score_patchcore (ratio, can
        exceed 1.0), score_graph, heatmap (numpy 224×224).
        """
        assert self._tau_raw is not None, "Call calibrate() first."
        cfg = self._cfg

        # ═══ PatchCore score ════════════════════════════════════════════════
        s_pc_raw, heatmap = self._patchcore_score(image_tensor)
        # Linear ratio: 1.0 = at calibration threshold, > 1.0 = anomalous
        S_pc = s_pc_raw / (self._tau_raw + 1e-8)

        # ═══ Reference Graph voting ══════════════════════════════════════════
        out = self._backbone.forward_features(image_tensor)
        q_cls = out["x_norm_clstoken"].cpu().numpy().astype(np.float32)
        faiss.normalize_L2(q_cls)
        _, active_idx = self._ref_index.search(q_cls, cfg.K_refs)
        active_idx = active_idx[0]

        raw_scores = np.array([
            roma_match_score(
                image_path,
                self._ref_paths[i],
                self._matcher,
                num_samples=cfg.roma_num_samples,
            )
            for i in active_idx
        ])

        mu_k = self._mu_ref[active_idx]
        sigma_k = self._sigma_ref[active_idx]
        z_scores = (raw_scores - mu_k) / (sigma_k + 1e-8)

        # Inverse variance weighting
        weights = 1.0 / (sigma_k ** 2 + 1e-4)
        weights = weights / weights.sum()
        z_bar = float(np.dot(z_scores, weights))
        S_graph = float(1.0 / (1.0 + np.exp(z_bar)))

        # ═══ Fusion ══════════════════════════════════════════════════════════
        S_pc_clipped = min(1.0, S_pc)
        S_final = cfg.alpha * S_pc_clipped + (1.0 - cfg.alpha) * S_graph

        return {
            "is_hallucination": bool(s_pc_raw > self._tau_raw),
            "score": float(S_final),
            "score_patchcore": float(S_pc),   # linear ratio; > 1.0 = anomalous
            "score_graph": float(S_graph),
            "heatmap": heatmap,
        }

    def detect_image(self, record: ImageRecord) -> dict:
        """Load, preprocess, detect. Adds record metadata to output dict."""
        img_tensor = RobotDataset([record])[0][0].unsqueeze(0).to(self._device)
        result = self.detect(img_tensor, str(record.path))
        result.update({
            "image_id": record.image_id,
            "task": record.task,
            "frame": record.frame,
            "split": record.split,
        })
        return result

    # ── Internal helpers ──────────────────────────────────────────────────────

    @torch.no_grad()
    def _patchcore_raw(self, image_tensor: torch.Tensor) -> float:
        """Return raw (unnormalized) PatchCore score s_pc_raw."""
        patches = extract_patches_batch(
            image_tensor, self._backbone,
            layers=self._cfg.dino_layers,
            p=self._cfg.patch_agg_kernel,
        )
        patches_np = patches.float().numpy()

        sq_dists, nn_idx = self._faiss_index.search(patches_np, k=1)
        l2_dists = np.sqrt(np.maximum(sq_dists[:, 0], 0))

        worst = int(np.argmax(l2_dists))
        s_star = float(l2_dists[worst])
        m_test_star = patches_np[worst]
        m_star = self._MC[int(nn_idx[worst, 0])]

        # Neighborhood reweighting (PatchCore Eq. 7)
        b = self._cfg.b_neighbors
        _, nb_idx = self._faiss_index.search(m_star.reshape(1, -1), k=b + 1)
        nb_features = self._MC[nb_idx[0, 1:]]
        dists_to_nbs = np.linalg.norm(m_test_star[None, :] - nb_features, axis=1)
        denom = np.sum(np.exp(np.clip(dists_to_nbs, None, 80)))
        exp_s = np.exp(min(s_star, 80))
        w = 1.0 - exp_s / (denom + 1e-8)
        w = max(0.0, w)
        return float(w * s_star)

    @torch.no_grad()
    def _patchcore_score(
        self, image_tensor: torch.Tensor
    ) -> tuple[float, np.ndarray]:
        """Return (s_pc_raw, spatial_heatmap 224×224)."""
        patches = extract_patches_batch(
            image_tensor, self._backbone,
            layers=self._cfg.dino_layers,
            p=self._cfg.patch_agg_kernel,
        )
        patches_np = patches.float().numpy()

        sq_dists, nn_idx = self._faiss_index.search(patches_np, k=1)
        l2_dists = np.sqrt(np.maximum(sq_dists[:, 0], 0))

        worst = int(np.argmax(l2_dists))
        s_star = float(l2_dists[worst])
        m_test_star = patches_np[worst]
        m_star = self._MC[int(nn_idx[worst, 0])]

        b = self._cfg.b_neighbors
        _, nb_idx = self._faiss_index.search(m_star.reshape(1, -1), k=b + 1)
        nb_features = self._MC[nb_idx[0, 1:]]
        dists_to_nbs = np.linalg.norm(m_test_star[None, :] - nb_features, axis=1)
        denom = np.sum(np.exp(np.clip(dists_to_nbs, None, 80)))
        exp_s = np.exp(min(s_star, 80))
        w = max(0.0, 1.0 - exp_s / (denom + 1e-8))
        s_pc_raw = float(w * s_star)

        heatmap = l2_dists.reshape(16, 16).astype(np.float32)
        heatmap = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_LINEAR)
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=4.0)

        return s_pc_raw, heatmap
