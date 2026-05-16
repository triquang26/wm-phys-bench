"""
warp_variance_vis.py

Standalone visualization: given a single "low" query image, match it against
all "high" reference images via RoMaV2 dense warp, then visualize:

  Row 0: Query image | Mean certainty | Min certainty | Cert std
  Row 1: Warp var (weighted) | Warp var (raw) | Warp var overlay | Combined signal
  Row 2: 8 individual ref certainty maps (4 best + 4 worst by mean cert)

Key insight being tested:
  - Nominal pixel  → refs AGREE on correspondence even if certainty is low → low warp variance
  - Hallucination  → refs DISAGREE (each "invents" a different target) → high warp variance

Usage:
  PYTHONPATH=/mnt/data/sftp/data/quangpt3/gcvwm/calibration/RoMaV2/src:$PYTHONPATH \\
  python warp_variance_vis.py \\
    --query ../image_no_bg/low/0_Open\\ the\\ box/frame_0001.png \\
    --setting turbo --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from glob import glob

import csv
import json
import math
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.measure import label as cc_label, regionprops


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VIS_SIZE = 224  # all maps resized to this before stacking / visualizing


def load_model(device: str, setting: str, bidirectional: bool = False,
               h_lr: int = 0, w_lr: int = 0):
    try:
        from romav2 import RoMaV2
    except ImportError:
        romav2_src = Path(__file__).parent.parent.parent.parent / "RoMaV2" / "src"
        sys.path.insert(0, str(romav2_src))
        from romav2 import RoMaV2
    model = RoMaV2()
    model.apply_setting(setting)
    model.balanced_sampling = True
    # Override bidirectional — works on top of any setting
    if bidirectional:
        model.bidirectional = True
    # Override resolution if specified
    if h_lr > 0:
        model.H_lr = h_lr
    if w_lr > 0:
        model.W_lr = w_lr
    model.eval()
    model = model.to(device)
    return model


def _cert_from_preds(preds: dict, use_precision: bool) -> torch.Tensor:
    """Extract (H, W) certainty from preds.

    overlap_AB: sigmoid of first confidence channel — probability of match.
    precision mode: det(Σ⁻¹) = product of eigenvalues of 2×2 precision matrix.
      High determinant → tight match (low uncertainty in both directions).
    """
    if use_precision and preds.get("precision_AB") is not None:
        prec = preds["precision_AB"][0]  # (H, W, 2, 2)
        # det of 2×2: ad - bc
        det = prec[..., 0, 0] * prec[..., 1, 1] - prec[..., 0, 1] * prec[..., 1, 0]
        # clamp negatives (numerical noise), then sqrt for scale stability
        cert = det.clamp(min=0).sqrt()
        # normalize to [0,1] range for consistent visualization
        cert = cert / (cert.amax() + 1e-8)
    else:
        cert = preds["overlap_AB"][0]
        if cert.dim() == 3:
            cert = cert.squeeze(-1)
    return cert


def match_dense(model, query_path: str, ref_path: str, device: str,
                use_precision: bool = False, fg_mask: np.ndarray | None = None):
    """Return (warp_HW2, cert_HW) resized to VIS_SIZE, as float32 numpy.

    fg_mask: (VIS_SIZE, VIS_SIZE) bool — if given, background cert pixels are zeroed,
    so background pixels contribute zero weight to the weighted warp variance.
    """
    preds = model.match(query_path, ref_path)
    warp = preds["warp_AB"][0]       # (H, W, 2)  normalized [-1,1]
    cert = _cert_from_preds(preds, use_precision)   # (H, W)

    # Resize: warp (1,2,H,W) and cert (1,1,H,W)
    warp_t = warp.permute(2, 0, 1).unsqueeze(0).float()    # (1,2,H,W)
    cert_t = cert.unsqueeze(0).unsqueeze(0).float()         # (1,1,H,W)

    warp_r = F.interpolate(warp_t, size=(VIS_SIZE, VIS_SIZE),
                           mode="bilinear", align_corners=False)
    cert_r = F.interpolate(cert_t, size=(VIS_SIZE, VIS_SIZE),
                           mode="bilinear", align_corners=False)

    warp_np = warp_r[0].permute(1, 2, 0).cpu().numpy()     # (224,224,2)
    cert_np = cert_r[0, 0].cpu().numpy()                    # (224,224)

    # Zero background — prevents textureless gray region from dominating variance
    if fg_mask is not None:
        cert_np[~fg_mask] = 0.0

    return warp_np, cert_np


def foreground_mask(img_bgr: np.ndarray) -> np.ndarray:
    """True = foreground. Background is exactly (127,127,127) from SAM3 segmenter."""
    return ~np.all(img_bgr == 127, axis=-1)   # (H, W) bool


def load_query_vis(query_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load query image. Returns (img_rgb float32 [0,1], fg_mask bool), both (224,224,*)."""
    img = cv2.imread(query_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {query_path}")
    img = cv2.resize(img, (VIS_SIZE, VIS_SIZE), interpolation=cv2.INTER_NEAREST)
    fg = foreground_mask(img)                              # (224,224) bool, computed on uint8
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img_rgb, fg


def norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Calibration (high→high matching to establish clean ivar baseline)
# ---------------------------------------------------------------------------

def _compute_ivar_for_paths(query_path, refs, model, device_str, use_precision, erosion_k):
    """Compute interior mean warp variance for one query vs given refs. Returns float or None."""
    query_img, fg_mask = load_query_vis(query_path)
    if not fg_mask.any():
        return None, "empty fg"
    interior_mask = cv2.erode(
        fg_mask.astype(np.uint8),
        np.ones((erosion_k, erosion_k), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    if not interior_mask.any():
        return None, "interior empty after erosion"
    all_warps, all_certs = [], []
    for ref_path in refs:
        try:
            warp_np, cert_np = match_dense(model, query_path, ref_path, device_str,
                                           use_precision=use_precision, fg_mask=fg_mask)
            all_warps.append(warp_np)
            all_certs.append(cert_np)
        except Exception:
            pass
    if not all_warps:
        return None, "no refs matched"
    warps = np.stack(all_warps)
    certs = np.stack(all_certs)
    w = certs / (certs.sum(0, keepdims=True) + 1e-6)
    mean_coord = (warps * w[..., None]).sum(0)
    diff_sq = ((warps - mean_coord[None]) ** 2 * w[..., None]).sum(0)
    interior_vals = (diff_sq.sum(-1) * interior_mask)[interior_mask]
    ivar = float(interior_vals.mean()) if len(interior_vals) > 0 else 0.0
    return ivar, "ok"


def run_calibration(
    high_paths: list,
    model,
    device: str,
    use_precision: bool,
    erosion_k: int,
    calib_k: int,
    calib_file: Path,
    task_specific_refs: bool = False,
) -> dict:
    """Match high frames vs same-task (or all) high refs → per-task (or global) ivar baseline.

    task_specific_refs=True: for each task, calibrate each high frame against only that
    task's other high frames. Produces per-task mean_ivar/std_ivar that match inference.
    task_specific_refs=False: sample calib_k random high frames vs ALL other high refs.
    """
    device_str = next(model.parameters()).device.type

    if task_specific_refs:
        task_groups: dict[str, list] = defaultdict(list)
        for p in high_paths:
            task_groups[Path(p).parent.name].append(p)

        per_task_stats: dict[str, dict] = {}
        all_ivars: list[float] = []

        for task_name, task_paths in sorted(task_groups.items()):
            print(f"\n  [task] {task_name}  ({len(task_paths)} frames)")
            ivars = []
            for qi, query_path in enumerate(task_paths):
                refs = [p for p in task_paths if p != query_path]
                print(f"    [{qi+1}/{len(task_paths)}] {Path(query_path).name}", end="  ", flush=True)
                ivar, status = _compute_ivar_for_paths(
                    query_path, refs, model, device_str, use_precision, erosion_k)
                if ivar is None:
                    print(f"skip ({status})")
                    continue
                ivars.append(ivar)
                all_ivars.append(ivar)
                print(f"ivar={ivar:.4f}")
            if ivars:
                mu  = sum(ivars) / len(ivars)
                std = math.sqrt(sum((v - mu) ** 2 for v in ivars) / max(len(ivars) - 1, 1))
                per_task_stats[task_name] = {"mean_ivar": mu, "std_ivar": std, "n": len(ivars)}
                print(f"    → mean={mu:.4f}  std={std:.4f}")
            else:
                print(f"    → no valid frames, skipping")

        if not all_ivars:
            raise RuntimeError("Calibration failed: no valid ivar computed.")
        g_mu  = sum(all_ivars) / len(all_ivars)
        g_std = math.sqrt(sum((v - g_mu) ** 2 for v in all_ivars) / max(len(all_ivars) - 1, 1))
        calib_data = {
            "task_refs": True,
            "tasks": per_task_stats,
            "global": {"mean_ivar": g_mu, "std_ivar": g_std, "n": len(all_ivars)},
            "erosion_k": erosion_k,
        }
        print(f"\n[Calibration task-specific] global mean={g_mu:.4f}  std={g_std:.4f}  "
              f"tasks={len(per_task_stats)}")
    else:
        random.seed(42)
        queries = random.sample(high_paths, min(calib_k, len(high_paths)))
        ivars = []
        for qi, query_path in enumerate(queries):
            print(f"  [calib {qi+1}/{len(queries)}] {Path(query_path).name}", end="  ", flush=True)
            refs = [p for p in high_paths if p != query_path]
            ivar, status = _compute_ivar_for_paths(
                query_path, refs, model, device_str, use_precision, erosion_k)
            if ivar is None:
                print(f"skip ({status})")
                continue
            ivars.append(ivar)
            print(f"ivar={ivar:.4f}")
        if not ivars:
            raise RuntimeError("Calibration failed: no high frames produced valid ivar.")
        mu  = sum(ivars) / len(ivars)
        std = math.sqrt(sum((v - mu) ** 2 for v in ivars) / max(len(ivars) - 1, 1))
        calib_data = {
            "task_refs": False,
            "mean_ivar": mu,
            "std_ivar":  std,
            "n_calib":   len(ivars),
            "erosion_k": erosion_k,
            "ivars":     ivars,
        }
        print(f"\n[Calibration] mean_ivar={mu:.4f}  std_ivar={std:.4f}  n={len(ivars)}")

    calib_file = Path(calib_file)
    calib_file.parent.mkdir(parents=True, exist_ok=True)
    with open(calib_file, "w") as f:
        json.dump(calib_data, f, indent=2)
    print(f"Saved: {calib_file}")
    return calib_data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _ivar_from_subset(
    warps: np.ndarray,
    certs: np.ndarray,
    interior_mask: np.ndarray,
) -> float:
    """Cert-weighted interior mean warp variance from a subset of already-matched warps/certs."""
    if warps.shape[0] == 0:
        return 0.0
    w = certs / (certs.sum(0, keepdims=True) + 1e-6)
    mean_coord = (warps * w[..., None]).sum(0)
    diff_sq = ((warps - mean_coord[None]) ** 2 * w[..., None]).sum(0)
    interior_vals = (diff_sq.sum(-1) * interior_mask)[interior_mask]
    return float(interior_vals.mean()) if len(interior_vals) > 0 else 0.0


def process_one(
    query_path: Path,
    high_paths: list,
    model,
    out_dir: Path,
    use_precision: bool,
    bidir: bool,
    setting: str,
    cert_label: str,
    blob_min_area: int = 50,
    z_thresh: float = 2.0,
    erosion_k: int = 10,
    cert_low_thresh: float = 0.10,
    calib_stats: dict | None = None,
    global_z_thresh: float = 2.0,
    task_specific_refs: bool = False,
    auto_task: bool = False,
) -> dict | None:
    """Process a single query image. Returns summary dict or None on skip/error."""
    task_out = out_dir / query_path.parent.name
    task_out.mkdir(parents=True, exist_ok=True)

    stem = query_path.stem
    png_path = task_out / f"{stem}_heatmaps.png"
    npz_path = task_out / f"{stem}_stats.npz"

    if png_path.exists() and npz_path.exists():
        print(f"  [skip] {query_path.parent.name}/{query_path.name} (already done)")
        return None

    # ------------------------------------------------------------------
    # Task-specific ref filtering: match only against same-task high frames
    # ------------------------------------------------------------------
    query_task = query_path.parent.name
    if task_specific_refs:
        active_refs = [p for p in high_paths if Path(p).parent.name == query_task]
        if not active_refs:
            print(f"  [warn] no task refs found for '{query_task}', falling back to all refs")
            active_refs = high_paths
    else:
        active_refs = high_paths

    # ------------------------------------------------------------------
    # Foreground mask — computed once, passed into every match_dense call
    # ------------------------------------------------------------------
    query_img, fg_mask = load_query_vis(str(query_path))  # (224,224,3), (224,224) bool
    fg_pixel_count = int(fg_mask.sum())

    # Interior mask: erode fg to strip boundary/edge pixels.
    # Thin tips and narrow edges are removed; wide object bodies remain.
    # This prevents edge-artifact variance from being mistaken for hallucination.
    _ek = erosion_k
    interior_mask = cv2.erode(
        fg_mask.astype(np.uint8),
        np.ones((_ek, _ek), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    interior_pixel_count = int(interior_mask.sum())

    # ------------------------------------------------------------------
    # Match against active refs (bg cert zeroed inside match_dense)
    # ------------------------------------------------------------------
    all_warps, all_certs, ref_names = [], [], []
    device_str = next(model.parameters()).device.type
    K = len(active_refs)
    for i, ref_path in enumerate(active_refs):
        print(f"    ref [{i+1:3d}/{K}] {Path(ref_path).name}", end="\r", flush=True)
        try:
            warp_np, cert_np = match_dense(model, str(query_path), str(ref_path),
                                           device_str,
                                           use_precision=use_precision,
                                           fg_mask=fg_mask)
            all_warps.append(warp_np)
            all_certs.append(cert_np)
            ref_names.append(Path(ref_path).stem)
        except Exception as e:
            print(f"\n    Warning: failed on {Path(ref_path).name}: {e}")

    if len(all_warps) == 0:
        print(f"\n  [error] No refs matched for {query_path.name}, skipping.")
        return None

    warps = np.stack(all_warps)   # (K, 224, 224, 2)
    certs = np.stack(all_certs)   # (K, 224, 224) — bg already zeroed

    # ------------------------------------------------------------------
    # Statistics (all bg-zeroed via certs)
    # ------------------------------------------------------------------
    mean_cert = certs.mean(0)
    cert_std  = certs.std(0)

    w = certs / (certs.sum(0, keepdims=True) + 1e-6)
    mean_coord = (warps * w[..., None]).sum(0)
    diff_sq = ((warps - mean_coord[None]) ** 2 * w[..., None]).sum(0)
    warp_var_weighted = diff_sq.sum(-1)              # (224,224)
    warp_var_raw = warps.var(0).sum(-1)

    # ------------------------------------------------------------------
    # Z-score signal computed on INTERIOR only (boundary pixels excluded)
    # ------------------------------------------------------------------
    warp_var_fg = warp_var_weighted * fg_mask         # full fg — kept for display only
    warp_var_interior = warp_var_weighted * interior_mask
    interior_vals = warp_var_interior[interior_mask]
    if len(interior_vals) > 1 and interior_vals.std() > 1e-8:
        z = (warp_var_interior - interior_vals.mean()) / interior_vals.std()
    else:
        z = np.zeros_like(warp_var_interior)
    signal = np.clip(z, 0, None) * interior_mask      # positive z-score, interior only

    # Absolute z-score threshold — only interior pixels truly anomalous
    tau = z_thresh
    spatial_label = (signal > tau) & interior_mask

    # Frame-level scores
    blobs = regionprops(cc_label(spatial_label.astype(np.uint8)))
    max_blob_area = max((r.area for r in blobs), default=0)
    frame_score_blob = float(max_blob_area)
    frame_score_max  = float(signal.max())
    blob_signal_mass = float(signal[spatial_label].sum()) if spatial_label.any() else 0.0
    interior_mean_var = float(interior_vals.mean()) if len(interior_vals) > 0 else 0.0

    mean_cert_fg = float(mean_cert[interior_mask].mean()) if interior_mask.any() else 0.0

    cond_peak     = frame_score_max > z_thresh and max_blob_area >= blob_min_area
    cond_low_cert = mean_cert_fg < cert_low_thresh

    # Global-var signal: z-score ivar against high→high calibration baseline.
    # Captures whole-frame hallucination (entire interior disagrees with refs).
    auto_detected_task = ""
    if auto_task and calib_stats is not None and calib_stats.get("task_refs"):
        # Cert-based task identification: group all refs by task, pick task with highest mean cert.
        # Recompute ivar using only T*'s warps/certs (already in memory — no extra RoMaV2 calls).
        # This allows classification with no folder structure / task label.
        task_ref_indices: dict[str, list[int]] = defaultdict(list)
        for idx, ref_path in enumerate(active_refs):
            task_ref_indices[Path(ref_path).parent.name].append(idx)
        # Per-task z-score: compute ivar for each task's refs, then normalize by
        # that task's calibration baseline. T* = argmin(z_T) — the task where the
        # query frame deviates least from the clean baseline.
        # Unifies task identification and hallucination detection: a clean frame from
        # task X has z_X ≈ 0 (minimum) while z_Y >> 0 for wrong tasks. A hallucinated
        # frame has elevated z across all tasks; even z_T* > threshold → H=1.
        tasks_calib = calib_stats.get("tasks", {})
        task_z: dict[str, float] = {}
        task_ivar: dict[str, float] = {}
        for t, idxs in task_ref_indices.items():
            t_idx_arr = np.array(idxs)
            ivar_t = _ivar_from_subset(warps[t_idx_arr], certs[t_idx_arr], interior_mask)
            task_ivar[t] = ivar_t
            t_stat = tasks_calib.get(t) or calib_stats.get("global", {})
            if t_stat:
                task_z[t] = (ivar_t - t_stat["mean_ivar"]) / (t_stat["std_ivar"] + 1e-8)
            else:
                task_z[t] = 0.0
        # argmin(raw_ivar) — task whose refs produce tightest agreement on the query.
        # argmin(z_T) was biased: tasks with high calib std absorb cross-task queries cheaply.
        # Raw ivar is scene-specific: same-scene refs agree (low var), cross-scene refs disagree (high var).
        auto_detected_task = min(task_ivar, key=task_ivar.get)
        t_idxs = np.array(task_ref_indices[auto_detected_task])
        # Recompute interior_mean_var and signal using only T*'s refs
        interior_mean_var = _ivar_from_subset(warps[t_idxs], certs[t_idxs], interior_mask)
        # Recompute per-pixel warp_var_interior for visualization using T* subset
        w_t = certs[t_idxs] / (certs[t_idxs].sum(0, keepdims=True) + 1e-6)
        mean_coord_t = (warps[t_idxs] * w_t[..., None]).sum(0)
        diff_sq_t = ((warps[t_idxs] - mean_coord_t[None]) ** 2 * w_t[..., None]).sum(0)
        warp_var_interior = diff_sq_t.sum(-1) * interior_mask
        interior_vals_t = warp_var_interior[interior_mask]
        if len(interior_vals_t) > 1 and interior_vals_t.std() > 1e-8:
            z = (warp_var_interior - interior_vals_t.mean()) / interior_vals_t.std()
        else:
            z = np.zeros_like(warp_var_interior)
        signal = np.clip(z, 0, None) * interior_mask
        spatial_label = (signal > z_thresh) & interior_mask
        blobs = regionprops(cc_label(spatial_label.astype(np.uint8)))
        max_blob_area = max((r.area for r in blobs), default=0)
        frame_score_blob = float(max_blob_area)
        frame_score_max  = float(signal.max())
        blob_signal_mass = float(signal[spatial_label].sum()) if spatial_label.any() else 0.0
        cond_peak = frame_score_max > z_thresh and max_blob_area >= blob_min_area
        # z_ivar against T*'s per-task calib
        task_stat = calib_stats["tasks"].get(auto_detected_task) or calib_stats["global"]
        z_ivar = (interior_mean_var - task_stat["mean_ivar"]) / (task_stat["std_ivar"] + 1e-8)
        cond_global_var = z_ivar > global_z_thresh
        sorted_ivars = sorted(task_ivar.values())
        second_ivar = sorted_ivars[1] if len(sorted_ivars) > 1 else float("inf")
        ratio = second_ivar / (task_ivar[auto_detected_task] + 1e-8)
        print(f"  [auto_task] T*={auto_detected_task[:45]}  "
              f"ivar_T*={task_ivar[auto_detected_task]:.4f}(2nd={second_ivar:.4f}, ratio={ratio:.1f}x)  final_z={z_ivar:.2f}")
    elif calib_stats is not None:
        if calib_stats.get("task_refs"):
            task_stat = calib_stats.get("tasks", {}).get(query_task) or calib_stats.get("global")
        else:
            task_stat = calib_stats
        z_ivar = (interior_mean_var - task_stat["mean_ivar"]) / (task_stat["std_ivar"] + 1e-8)
        cond_global_var = z_ivar > global_z_thresh
    else:
        z_ivar = float("nan")
        cond_global_var = False  # batch_reclassify will fill this in

    is_hallucination = int(cond_peak or cond_global_var or cond_low_cert)

    np.savez(npz_path,
             mean_cert=mean_cert, cert_std=cert_std,
             warp_var_weighted=warp_var_weighted, warp_var_raw=warp_var_raw,
             warp_var_fg=warp_var_fg, warp_var_interior=warp_var_interior,
             signal=signal, spatial_label=spatial_label,
             fg_mask=fg_mask, interior_mask=interior_mask,
             frame_score_blob=frame_score_blob,
             frame_score_max=frame_score_max,
             blob_signal_mass=blob_signal_mass,
             interior_mean_var=interior_mean_var,
             z_thresh=z_thresh, erosion_k=erosion_k)

    # ------------------------------------------------------------------
    # Visualization (3 rows × 4 cols)
    # ------------------------------------------------------------------
    per_ref_cert = certs.mean(axis=(1, 2))
    sorted_idx = np.argsort(per_ref_cert)
    example_idxs = sorted_idx[-4:][::-1].tolist() + sorted_idx[:4].tolist()

    fig = plt.figure(figsize=(20, 14))
    z_ivar_str = f"{z_ivar:.2f}" if not math.isnan(z_ivar) else "n/a"
    flags = f"peak={'Y' if cond_peak else 'N'}  gvar={'Y' if cond_global_var else 'N'}  lcert={'Y' if cond_low_cert else 'N'}"
    h_label = f"H={is_hallucination}  [{flags}]  score={frame_score_max:.2f}  ivar={interior_mean_var:.3f}(z={z_ivar_str})  cert={mean_cert_fg:.3f}"
    fig.suptitle(
        f"Warp Variance — {query_path.parent.name}/{query_path.name}  "
        f"({len(all_warps)} refs, {setting}{'+bidir' if bidir else ''}, cert={cert_label})\n"
        f"{h_label}",
        fontsize=10, fontweight="bold",
    )
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.35)

    def add_heatmap(ax, data, title, cmap="viridis", vmin=None, vmax=None):
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 0: query | fg overlay | mean cert | cert std
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(query_img)
    ax0.set_title(f"Query (low)\nfg={fg_pixel_count}px", fontsize=9)
    ax0.axis("off")

    # Show interior mask: fg=dark gray, boundary-only=light gray, interior=image
    ax_fg = fig.add_subplot(gs[0, 1])
    interior_vis = np.full((*fg_mask.shape, 3), 0.5, dtype=np.float32)  # all gray
    interior_vis[fg_mask] = 0.25                                          # fg boundary = darker
    interior_vis[interior_mask] = query_img[interior_mask]                # interior = real image
    ax_fg.imshow(interior_vis)
    ax_fg.set_title(f"Interior mask (erode {erosion_k}px)\ninterior={interior_pixel_count}px", fontsize=9)
    ax_fg.axis("off")

    add_heatmap(fig.add_subplot(gs[0, 2]), mean_cert,
                f"Mean cert (interior)\nμ={mean_cert_fg:.3f}")
    add_heatmap(fig.add_subplot(gs[0, 3]), cert_std,
                f"Cert std\nμ={cert_std[interior_mask].mean():.3f}" if interior_mask.any() else "Cert std")

    # Row 1: warp_var_interior | z-score signal | signal overlay | spatial label
    add_heatmap(fig.add_subplot(gs[1, 0]), warp_var_interior,
                f"Warp var (interior)\nmax={warp_var_interior.max():.4f}", cmap="hot")
    add_heatmap(fig.add_subplot(gs[1, 1]), signal,
                f"Z-score signal\nmax={frame_score_max:.2f}", cmap="hot")

    ax_ov = fig.add_subplot(gs[1, 2])
    sig_n = norm01(signal)
    overlay = query_img.copy()
    overlay[:, :, 0] = np.clip(overlay[:, :, 0] * 0.4 + sig_n * 0.9, 0, 1)
    overlay[:, :, 1] = overlay[:, :, 1] * 0.4
    overlay[:, :, 2] = overlay[:, :, 2] * 0.4
    ax_ov.imshow(overlay)
    ax_ov.set_title("Signal overlay\n(red = anomaly)", fontsize=9)
    ax_ov.axis("off")

    ax_lbl = fig.add_subplot(gs[1, 3])
    lbl_vis = np.zeros((*spatial_label.shape, 3), dtype=np.float32)
    lbl_vis[fg_mask] = [0.1, 0.1, 0.1]              # fg boundary = very dark
    lbl_vis[interior_mask] = [0.25, 0.25, 0.25]     # interior = dark gray
    lbl_vis[spatial_label] = [1.0, 0.1, 0.1]        # anomaly blobs = red
    ax_lbl.imshow(lbl_vis)
    ax_lbl.set_title(
        f"Spatial label (z>{tau:.2f})\nblob={max_blob_area}px  ivar={interior_mean_var:.3f}  H={is_hallucination}",
        fontsize=9,
    )
    ax_lbl.axis("off")

    # Row 2: 4 best + 4 worst ref cert maps
    for col, ref_idx in enumerate(example_idxs):
        ax_r = fig.add_subplot(gs[2, col % 4])
        label = "BEST" if col < 4 else "WORST"
        im = ax_r.imshow(certs[ref_idx], cmap="viridis", vmin=0, vmax=certs.max())
        ax_r.set_title(f"[{label}] {ref_names[ref_idx]}\n{per_ref_cert[ref_idx]:.4f}", fontsize=7)
        ax_r.axis("off")
        plt.colorbar(im, ax=ax_r, fraction=0.046, pad=0.04)

    fig.text(0.13, 0.32, "← 4 Best refs", fontsize=8, color="green")
    fig.text(0.63, 0.32, "4 Worst refs →", fontsize=8, color="red")

    fig.savefig(png_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  → {png_path.relative_to(out_dir.parent.parent)}  H={is_hallucination}")

    return {
        "task": query_path.parent.name,
        "frame": query_path.stem,
        "n_refs": len(all_warps),
        "frame_score_max": frame_score_max,
        "frame_score_blob": frame_score_blob,
        "blob_signal_mass": blob_signal_mass,
        "interior_mean_var": interior_mean_var,
        "z_ivar": z_ivar if not math.isnan(z_ivar) else "",
        "is_hallucination": is_hallucination,
        "cond_peak": int(cond_peak),
        "cond_global_var": int(cond_global_var),
        "cond_low_cert": int(cond_low_cert),
        "mean_cert_fg": mean_cert_fg,
        "fg_pixel_count": fg_pixel_count,
        "auto_detected_task": auto_detected_task,
    }


def main():
    parser = argparse.ArgumentParser(description="Warp variance hallucination visualization")
    # Single or batch mode (not required when --calibrate is set)
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--query", help="Single query image path")
    group.add_argument("--query_dir", help="Batch: process all PNGs under this dir (default: image_no_bg/low/)")
    parser.add_argument(
        "--high_dir", default=None,
        help="Root dir with high reference images (default: image_no_bg/high/)",
    )
    parser.add_argument("--out_dir", default=None, help="Output root (default: results/warp_variance/)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--setting", default="turbo",
        choices=["turbo", "fast", "base", "precise"],
        help="RoMaV2 resolution (turbo=320, fast=512, base=640, precise=800+1280)",
    )
    parser.add_argument(
        "--bidirectional", action="store_true",
        help="Bidirectional matching (better quality, ~2x slower). Auto-on for 'precise'.",
    )
    parser.add_argument(
        "--use_precision", action="store_true",
        help="Use det(precision_AB) as certainty instead of overlap_AB.",
    )
    parser.add_argument("--h_lr", type=int, default=0, help="Override LR height")
    parser.add_argument("--w_lr", type=int, default=0, help="Override LR width")
    parser.add_argument("--top_k", type=int, default=0, help="Limit refs to top_k (0=all)")
    parser.add_argument("--blob_min_area", type=int, default=50,
                        help="Min blob area (px) to classify as hallucination (default: 50)")
    parser.add_argument("--z_thresh", type=float, default=2.0,
                        help="Absolute z-score threshold for anomaly pixels (default: 2.0). "
                             "H=1 requires BOTH frame_score_max > z_thresh AND blob >= blob_min_area.")
    parser.add_argument("--erosion_k", type=int, default=10,
                        help="Erosion kernel size (px) to strip fg boundary before z-score (default: 10). "
                             "Removes thin tips/edges where high variance is structural, not hallucination.")
    parser.add_argument("--global_z_thresh", type=float, default=2.0,
                        help="Z-score threshold for global-var hallucination signal (default: 2.0). "
                             "Used with --calib_file (per-frame) or batch_reclassify fallback.")
    parser.add_argument("--cert_low_thresh", type=float, default=0.10,
                        help="Mean certainty threshold (below = hallucination) for low-cert signal (default: 0.10). "
                             "Catches frames that are completely out-of-distribution from refs.")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run calibration mode: match calib_k random high frames vs other high refs, "
                             "save ivar baseline to --calib_file, then exit.")
    parser.add_argument("--calib_file", default=None,
                        help="Path to calibration JSON. Required with --calibrate. "
                             "If provided during inference, enables per-frame global-var scoring.")
    parser.add_argument("--calib_k", type=int, default=20,
                        help="Number of high frames to sample for calibration (default: 20).")
    parser.add_argument("--task_refs", action="store_true",
                        help="Use task-specific refs: match query against only same-task high frames. "
                             "Requires per-task calibration (--calibrate --task_refs). "
                             "Enables per-frame classification without batch context.")
    parser.add_argument("--auto_task", action="store_true",
                        help="Auto-identify task from cert without needing folder structure. "
                             "Matches query vs ALL high refs, groups certs by task, picks T* = "
                             "task with highest mean cert, then recomputes ivar using only T*'s "
                             "warps/certs (already in memory). Requires --calib_file with "
                             "per-task stats (built with --calibrate --task_refs).")
    args = parser.parse_args()
    if not args.calibrate and not args.query and not args.query_dir:
        parser.error("one of --query or --query_dir is required (unless --calibrate)")

    root = Path(__file__).parent.parent
    high_dir = Path(args.high_dir) if args.high_dir else root / "image_no_bg" / "high"
    out_dir  = Path(args.out_dir)  if args.out_dir  else root / "results" / "warp_variance"
    out_dir.mkdir(parents=True, exist_ok=True)

    high_paths = sorted(glob(str(high_dir / "**" / "*.png"), recursive=True))
    if not high_paths:
        raise RuntimeError(f"No high images found under: {high_dir}")
    if args.top_k > 0:
        high_paths = high_paths[: args.top_k]

    # Collect query paths
    if args.query:
        query_paths = [Path(args.query).expanduser().resolve()]
    else:
        qdir = Path(args.query_dir) if args.query_dir else root / "image_no_bg" / "low"
        query_paths = sorted(Path(p) for p in glob(str(qdir / "**" / "*.png"), recursive=True))
    if not query_paths:
        raise RuntimeError("No query images found.")

    bidir = args.bidirectional or args.setting == "precise"
    cert_label = "precision det" if args.use_precision else "overlap"

    print(f"Queries : {len(query_paths)}")
    print(f"Refs    : {len(high_paths)} from {high_dir}")
    print(f"Setting : {args.setting}  bidir={bidir}  precision={args.use_precision}")
    print(f"Thresholds: z>{args.z_thresh}  blob>={args.blob_min_area}px  erosion={args.erosion_k}px  "
          f"global_z>{args.global_z_thresh}  cert<{args.cert_low_thresh}")
    print(f"Output  : {out_dir}")

    print("Loading RoMaV2...")
    model = load_model(args.device, args.setting, bidirectional=bidir,
                       h_lr=args.h_lr, w_lr=args.w_lr or args.h_lr)

    # Calibration mode: match high frames against each other, save ivar baseline, exit
    if args.calibrate:
        if not args.calib_file:
            raise ValueError("--calib_file is required with --calibrate")
        mode_str = "task-specific refs" if args.task_refs else f"{args.calib_k} sampled frames"
        print(f"\nCalibration mode: {mode_str} → high refs")
        run_calibration(high_paths, model, args.device, args.use_precision,
                        args.erosion_k, args.calib_k, Path(args.calib_file),
                        task_specific_refs=args.task_refs)
        return

    # Load calibration stats if provided
    calib_stats = None
    if args.calib_file and Path(args.calib_file).exists():
        with open(args.calib_file) as f:
            calib_stats = json.load(f)
        if calib_stats.get("task_refs"):
            g = calib_stats.get("global", {})
            print(f"Calibration (task-specific): global mean={g.get('mean_ivar',float('nan')):.4f}  "
                  f"std={g.get('std_ivar',float('nan')):.4f}  tasks={len(calib_stats.get('tasks',{}))}")
        else:
            print(f"Calibration: mean_ivar={calib_stats['mean_ivar']:.4f}  "
                  f"std_ivar={calib_stats['std_ivar']:.4f}  n={calib_stats.get('n_calib','?')}")
    elif args.calib_file:
        print(f"Warning: --calib_file {args.calib_file} not found. Run --calibrate first.")

    summary_path = out_dir / "summary.csv"
    csv_fields = ["task", "frame", "n_refs", "frame_score_max",
                  "frame_score_blob", "blob_signal_mass", "interior_mean_var", "z_ivar",
                  "is_hallucination", "cond_peak", "cond_global_var", "cond_low_cert",
                  "mean_cert_fg", "fg_pixel_count", "auto_detected_task"]
    write_header = not summary_path.exists()

    with open(summary_path, "a", newline="") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=csv_fields)
        if write_header:
            writer.writeheader()

        for qi, query_path in enumerate(query_paths):
            print(f"\n[{qi+1}/{len(query_paths)}] {query_path.parent.name}/{query_path.name}")
            result = process_one(query_path, high_paths, model, out_dir,
                                 use_precision=args.use_precision, bidir=bidir,
                                 setting=args.setting, cert_label=cert_label,
                                 blob_min_area=args.blob_min_area,
                                 z_thresh=args.z_thresh,
                                 erosion_k=args.erosion_k,
                                 cert_low_thresh=args.cert_low_thresh,
                                 calib_stats=calib_stats,
                                 global_z_thresh=args.global_z_thresh,
                                 task_specific_refs=args.task_refs,
                                 auto_task=args.auto_task)
            if result is not None:
                writer.writerow(result)
                csvf.flush()

    # Batch reclassify fallback: only runs when no calib_file provided.
    # z-scores ivar within each task group and updates H labels.
    if args.query_dir and calib_stats is None:
        batch_reclassify(summary_path, args.global_z_thresh, args.cert_low_thresh)

    print(f"\nDone. Results in: {out_dir}")
    print(f"Summary CSV: {summary_path}")


def batch_reclassify(summary_path: Path, batch_z_thresh: float, cert_low_thresh: float):
    """Post-process summary.csv: z-score interior_mean_var per task, update H labels."""
    import math
    from collections import defaultdict

    with open(summary_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not rows:
        return

    # Group row indices by task
    task_groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        task_groups[row["task"]].append(i)

    # Min-anchored z-score: anchor at the best (lowest) frame in each task.
    # This avoids the mean being pulled up by hallucinated frames, so z-scores
    # reliably detect elevated ivar without needing external calibration.
    ivar_z = [0.0] * len(rows)
    for task, idxs in task_groups.items():
        vals = [float(rows[i]["interior_mean_var"]) for i in idxs]
        ivar_min = min(vals)
        devs = [v - ivar_min for v in vals]
        dev_mean = sum(devs) / len(devs)
        dev_std = math.sqrt(sum((d - dev_mean) ** 2 for d in devs) / max(len(devs) - 1, 1))
        for k, i in enumerate(idxs):
            ivar_z[i] = devs[k] / (dev_std + 1e-8)

    # Recompute cond_global_var and is_hallucination; store batch z_ivar in CSV
    for i, row in enumerate(rows):
        gvar = ivar_z[i] > batch_z_thresh
        row["z_ivar"] = f"{ivar_z[i]:.4f}"
        row["cond_global_var"] = int(gvar)
        row["is_hallucination"] = int(
            int(row["cond_peak"]) or gvar or int(row["cond_low_cert"])
        )

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_h = sum(int(r["is_hallucination"]) for r in rows)
    print(f"\n[batch_reclassify] batch_z_thresh={batch_z_thresh}  "
          f"{n_h}/{len(rows)} hallucinations")
    for task, idxs in sorted(task_groups.items()):
        task_rows = [rows[i] for i in idxs]
        print(f"  {task}: " + "  ".join(
            f"{r['frame']} H={r['is_hallucination']}(p={r['cond_peak']} g={r['cond_global_var']} c={r['cond_low_cert']})"
            for r in task_rows
        ))


if __name__ == "__main__":
    main()
