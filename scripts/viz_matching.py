"""Visualize RoMa dense correspondences between query frames and reference frames.

For each query, produces a figure with N_REFS panels:
  [ref_img | query_img] with sampled keypoint correspondences drawn as colored lines.

Usage (groot env):
    python scripts/viz_matching.py \
        --query_frames "0_Open the box/v0004_f04.png" \
        --ref_task "0_Open the box" \
        --n_refs 3 --n_kpts 150 \
        --out_dir /tmp/viz_matching
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "third_party" / "RoMaV2" / "src"))
sys.path.insert(0, str(REPO))

from warp_score.matcher import RoMaMatcher
from warp_score.mask import InteriorMask


def load_rgb(path: Path, size: int = 224) -> np.ndarray:
    img = cv2.imread(str(path))
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def draw_correspondences(
    ref_img: np.ndarray,
    qry_img: np.ndarray,
    warp_HW2: np.ndarray,   # (H,W,2) in [-1,1] — for each query pixel, where in ref
    cert_HW: np.ndarray,    # (H,W) in [0,1]
    n_kpts: int = 150,
    title: str = "",
) -> np.ndarray:
    H, W = ref_img.shape[:2]
    canvas = np.concatenate([ref_img, qry_img], axis=1).copy()

    # Sample top-cert foreground pixels
    fg = cert_HW > 0.05
    if fg.sum() < 10:
        return canvas

    ys, xs = np.where(fg)
    certs = cert_HW[ys, xs]
    # take top-N by cert
    idx = np.argsort(certs)[::-1][:n_kpts * 4]
    ys, xs, certs = ys[idx], xs[idx], certs[idx]
    # sub-sample evenly-spaced to avoid clutter
    step = max(1, len(ys) // n_kpts)
    ys, xs, certs = ys[::step][:n_kpts], xs[::step][:n_kpts], certs[::step][:n_kpts]

    # warp[:,2] is (warp_x, warp_y) in [-1,1] → pixel coords in ref
    ref_ys = ((warp_HW2[ys, xs, 1] + 1) / 2 * (H - 1)).astype(int)
    ref_xs = ((warp_HW2[ys, xs, 0] + 1) / 2 * (W - 1)).astype(int)

    # clamp to valid range
    ref_ys = np.clip(ref_ys, 0, H - 1)
    ref_xs = np.clip(ref_xs, 0, W - 1)

    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    for i in range(len(ys)):
        c = float(certs[i])
        color = (int((1-c)*255), int(c*200), 0)  # BGR: red→green by cert
        pt_ref = (int(ref_xs[i]), int(ref_ys[i]))
        pt_qry = (int(xs[i]) + W,  int(ys[i]))   # offset by W for right panel
        cv2.line(canvas_bgr, pt_ref, pt_qry, color, 1, cv2.LINE_AA)
        cv2.circle(canvas_bgr, pt_ref, 2, color, -1)
        cv2.circle(canvas_bgr, pt_qry, 2, color, -1)

    canvas = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)

    # title bar
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.imshow(canvas)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.tight_layout(pad=0.3)

    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    arr = np.frombuffer(buf.read(), dtype=np.uint8)
    out = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query_high_dir", type=Path,
                    default=REPO / "data/query/high")
    ap.add_argument("--ref_dir", type=Path,
                    default=REPO / "data/reference")
    ap.add_argument("--task", required=True,
                    help="Task name (dir under query_high_dir and ref_dir)")
    ap.add_argument("--frames", required=True,
                    help="Comma-separated frame stems, e.g. v0001_f03,v0001_f04")
    ap.add_argument("--n_refs", type=int, default=3,
                    help="Number of reference frames to match against per query")
    ap.add_argument("--n_kpts", type=int, default=120,
                    help="Number of correspondences to draw")
    ap.add_argument("--out_dir", type=Path, default=Path("/tmp/viz_matching"))
    ap.add_argument("--setting", default="turbo")
    ap.add_argument("--vis_size", type=int, default=224)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    matcher = RoMaMatcher(
        setting=args.setting, device="cuda", vis_size=args.vis_size
    )
    interior_mask = InteriorMask(erosion_k=5)

    ref_dir = args.ref_dir / args.task
    refs = sorted(ref_dir.glob("*.png"))
    # pick n_refs spread across available refs
    step = max(1, len(refs) // args.n_refs)
    selected_refs = refs[::step][: args.n_refs]
    print(f"[viz] task={args.task}")
    print(f"[viz] refs ({len(selected_refs)}): {[r.name for r in selected_refs]}")

    qry_dir = args.query_high_dir / args.task
    frame_list = [f.strip() for f in args.frames.split(",")]

    for frame_stem in frame_list:
        qry_path = qry_dir / f"{frame_stem}.png"
        if not qry_path.exists():
            print(f"[viz] SKIP missing: {qry_path}")
            continue

        qry_img = load_rgb(qry_path, args.vis_size)
        panels = []

        for ref_path in selected_refs:
            ref_img = load_rgb(ref_path, args.vis_size)
            result = matcher.match(qry_path, ref_path)
            warp  = result.warp   # (H,W,2) in [-1,1]
            cert  = result.cert   # (H,W)

            mean_cert = float(cert[cert > 0].mean()) if (cert > 0).any() else 0.0
            title = (
                f"ref={ref_path.name}  query={frame_stem}\n"
                f"mean_cert={mean_cert:.3f}  "
                f"(green=high cert, red=low cert)"
            )
            panel = draw_correspondences(ref_img, qry_img, warp, cert,
                                         n_kpts=args.n_kpts, title=title)
            panels.append(panel)
            print(f"  matched {ref_path.name} → cert={mean_cert:.3f}")

        # stack panels vertically into one image
        combined = np.concatenate(panels, axis=0)
        out_path = args.out_dir / f"{frame_stem}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        print(f"[viz] saved → {out_path}")


if __name__ == "__main__":
    main()
