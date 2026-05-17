"""Load RoMaV2 and compute the cycle-consistent match score used by Stage 4 + detector."""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import KDTree


def load_roma(device: str):
    """Return a ready-to-use RoMaV2 model."""
    try:
        from romav2 import RoMaV2
    except ImportError as e:
        raise ImportError(
            "romav2 not found. Install with: pip install romav2 (or romatch)"
        ) from e
    model = RoMaV2()
    model = model.to(device)
    model.balanced_sampling = True
    model.eval()
    return model


def roma_match_score(
    path_a: str,
    path_b: str,
    model,
    num_samples: int = 1000,
    cycle_thresh_px: float = 5.0,
) -> float:
    """Confidence-weighted cycle-consistent inlier ratio in [0, 1].

    High score ≈ good geometric correspondence (same scene, nominal frame).
    Low score ≈ poor / hallucinated correspondence.

    Uses the actual romav2 API:
      preds = model.match(path_a, path_b)
      matches, conf, _, _ = model.sample(preds, n)
      kpts_a, kpts_b = model.to_pixel_coordinates(matches, h_a, w_a, h_b, w_b)
    """
    import cv2

    img_a = cv2.imread(path_a)
    img_b = cv2.imread(path_b)
    if img_a is None or img_b is None:
        return 0.0
    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]

    # Forward A→B
    preds_ab = model.match(path_a, path_b)
    matches_ab, conf_ab, _, _ = model.sample(preds_ab, num_samples)
    kpts_a, kpts_b = model.to_pixel_coordinates(matches_ab, h_a, w_a, h_b, w_b)
    kpts_a_np = kpts_a.cpu().numpy()
    kpts_b_np = kpts_b.cpu().numpy()
    conf_np = conf_ab.cpu().numpy()

    if len(kpts_a_np) == 0:
        return 0.0

    # Backward B→A
    preds_ba = model.match(path_b, path_a)
    matches_ba, _, _, _ = model.sample(preds_ba, num_samples)
    kpts_b2, kpts_a2 = model.to_pixel_coordinates(matches_ba, h_b, w_b, h_a, w_a)
    kpts_b2_np = kpts_b2.cpu().numpy()
    kpts_a2_np = kpts_a2.cpu().numpy()

    if len(kpts_b2_np) == 0:
        return 0.0

    # Cycle error: A→B→A
    tree = KDTree(kpts_b2_np)
    _, idxs = tree.query(kpts_b_np, k=1)
    kpts_a_reconstructed = kpts_a2_np[idxs.ravel()]
    cycle_errors = np.linalg.norm(kpts_a_np - kpts_a_reconstructed, axis=1)

    inlier_mask = (cycle_errors < cycle_thresh_px).astype(np.float32)
    score = float((inlier_mask * conf_np).sum() / (conf_np.sum() + 1e-8))
    return score
