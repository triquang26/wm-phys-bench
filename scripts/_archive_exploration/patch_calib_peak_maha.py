"""Patch an existing calibration.npz to add peak_maha_dist.

Extracts peak Z-score of D_map from the stored T_null arrays —
avoids a full re-calibration (~27 min) when adding the new signal.

Usage:
    python scripts/patch_calib_peak_maha.py <calibration.npz>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _peak_max_z(d_map: np.ndarray, interior_mask: np.ndarray) -> float:
    vals = d_map[interior_mask]
    if vals.size < 2 or vals.std() < 1e-8:
        return 0.0
    z = (d_map - vals.mean()) / (vals.std() + 1e-8)
    return float(z[interior_mask].max())


def patch(npz_path: Path) -> None:
    meta_path = npz_path.with_suffix(".json")
    with open(meta_path) as f:
        meta = json.load(f)

    npz = np.load(npz_path)
    new_data = {k: npz[k] for k in npz.files}

    added = []
    for task_name, task_meta in meta["tasks"].items():
        slug = task_meta["slug"]
        t_null_key = f"{slug}__T_null"
        if t_null_key not in npz.files:
            print(f"  [skip] {task_name[:50]} — no T_null")
            continue

        T_null = npz[t_null_key]  # (N, H, W)
        N = T_null.shape[0]

        # Use average non-zero pixels as interior proxy (same as ForegroundMask would give)
        avg_map = T_null.mean(axis=0)
        interior_proxy = avg_map > (avg_map.max() * 0.005)  # 0.5% of peak

        peak_mahas = []
        for i in range(N):
            peak_mahas.append(_peak_max_z(T_null[i], interior_proxy))

        arr = np.sort(np.array(peak_mahas, dtype=np.float32))
        peak_key = f"{slug}__peak_maha"
        new_data[peak_key] = arr
        task_meta["has_peak_maha"] = True
        added.append((task_name, len(peak_mahas)))
        print(f"  [ok]   {task_name[:55]}  N={N}  peak_maha range [{arr.min():.2f}, {arr.max():.2f}]")

    # Pool into global
    peak_parts = []
    for task_name, task_meta in meta["tasks"].items():
        slug = task_meta["slug"]
        key = f"{slug}__peak_maha"
        if key in new_data:
            peak_parts.append(new_data[key])
    if peak_parts:
        global_peak = np.sort(np.concatenate(peak_parts))
        new_data["__global__peak_maha"] = global_peak
        meta["global"]["has_peak_maha"] = True
        print(f"  [ok]   __global__  N={len(global_peak)}  range [{global_peak.min():.2f}, {global_peak.max():.2f}]")

    # Save
    np.savez_compressed(npz_path, **new_data)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"\nPatched {npz_path} with peak_maha for {len(added)} task(s).")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz", help="Path to calibration.npz")
    args = p.parse_args()
    patch(Path(args.npz))


if __name__ == "__main__":
    main()
