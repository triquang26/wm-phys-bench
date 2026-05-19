"""S4: NN-set Jaccard signal — intra-video DINOv2 NN consistency.

Hypothesis:
  For a real test frame f_t, its top-K nearest neighbors in the reference
  pool form a stable, semantically coherent subset (e.g., adjacent training
  frames from the same task). The NN set of f_{t+1} should overlap heavily
  with that of f_t — both because the scene barely changed and because
  reference frames themselves are temporally clustered.

  For a generated test frame from Cosmos, even though each individual
  frame may be visually plausible, the *trajectory* through DINOv2 feature
  space exhibits jumps: NN sets between consecutive generated frames
  jitter between unrelated training videos.

Signal:
  jaccard_distance(NN_K(f_t), NN_K(f_{t+1})) = 1 - |∩|/|∪|

  Real pairs:  small distance (near 0)
  Gen pairs:   larger distance (jumpy NN sets)

Why this is novel for hallu detection:
  Most OOD detectors look at where ONE frame lands in feature space.
  This measures how consecutive frames *move together* through that
  space — a temporal-coherence prior on the reference manifold itself,
  not on raw image content.
"""
from __future__ import annotations

import numpy as np


def nn_set_jaccard_distance(
    query_feat_t:   np.ndarray,    # (D,) L2-normalized
    query_feat_tp1: np.ndarray,    # (D,) L2-normalized
    pool_feats:     np.ndarray,    # (M, D) L2-normalized
    k: int = 50,
) -> float:
    """Jaccard distance between top-k NN sets of two consecutive frames.

    Args:
        query_feat_t:   DINOv2 CLS embedding of frame t.
        query_feat_tp1: DINOv2 CLS embedding of frame t+1.
        pool_feats:     (M, D) embeddings of all reference pool frames.
        k:              size of NN set to compare.

    Returns:
        Jaccard distance in [0, 1].
        0.0 means the two frames have identical NN sets (perfect coherence).
        1.0 means disjoint NN sets (severe trajectory jump).
    """
    sims_t   = pool_feats @ query_feat_t
    sims_tp1 = pool_feats @ query_feat_tp1
    k = min(k, len(pool_feats))
    top_t   = set(np.argpartition(sims_t,   -k)[-k:].tolist())
    top_tp1 = set(np.argpartition(sims_tp1, -k)[-k:].tolist())
    inter = len(top_t & top_tp1)
    union = len(top_t | top_tp1)
    if union == 0:
        return 0.0
    return float(1.0 - inter / union)


def nn_traj_jump(
    query_feats_seq: np.ndarray,   # (T, D) L2-normalized, sequential
    pool_feats:      np.ndarray,   # (M, D)
    k: int = 50,
) -> dict:
    """Per-frame NN-jaccard signal for a sequence of frames.

    For each frame t (t = 0 .. T-2) returns the Jaccard distance between
    its NN set and the next frame's NN set.

    Returns dict with:
        per_frame_jaccard  list[float] length T-1 (one entry per pair)
        mean               mean jaccard distance over the video
        peak               max jaccard distance over the video
    """
    T = len(query_feats_seq)
    if T < 2:
        return {"per_frame_jaccard": [], "mean": 0.0, "peak": 0.0}

    dists = []
    for t in range(T - 1):
        d = nn_set_jaccard_distance(
            query_feats_seq[t], query_feats_seq[t + 1], pool_feats, k=k,
        )
        dists.append(d)
    arr = np.asarray(dists, dtype=np.float32)
    return {
        "per_frame_jaccard": dists,
        "mean": float(arr.mean()),
        "peak": float(arr.max()),
    }
