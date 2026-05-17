"""MP4 -> 50 frames -> SAM3 bg removal, writing data/query/high/<task>/*.png.

Usage:
    python postprocess.py \
        --video_root ../data/cosmos_synthetic_data/query/high \
        --frames_root ../data/cosmos_frames_raw/query/high \
        --out_root ../data/query/high

Resume-safe: skips an output PNG if it already exists.

Reuses existing utilities (no logic duplication):
  * scripts/extract_frames.py::extract_uniform
        Uniformly samples N frames from an MP4 (np.linspace(0, total-1, N)).
  * sam3.py::SAM3Segmenter (+ PROMPTS list)
        SAM3 multi-prompt segmenter producing an H*W uint8 alpha mask. We use
        its .segment_multi_prompt() + .remove_background() pair.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Wire reusable utilities onto sys.path before importing.
# - extract_frames.py lives at <repo>/scripts/
# - sam3.py lives at <repo>/../sam3.py (parent of repo, calibration/feepe/)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]              # feature_matching_eval_hallucination/
PARENT    = REPO_ROOT.parent                                  # calibration/feepe/
# Insert in reverse so scripts/ ends up FIRST in sys.path.
# (outer feepe/extract_frames.py is a stale older variant; we want repo/scripts/.)
for p in (PARENT, REPO_ROOT, REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from extract_frames import extract_uniform                    # noqa: E402  (reuse)
from sam3 import SAM3Segmenter, PROMPTS                       # noqa: E402  (reuse)
from tqdm import tqdm                                          # noqa: E402


# ---------------------------------------------------------------------------
# Harvester
# ---------------------------------------------------------------------------
class QueryHighHarvester:
    """video_root/*.mp4  ->  frames_root/<task>/frame_NNNN.png  ->  out_root/<task>/frame_NNNN.png"""

    def __init__(
        self,
        video_root: Path,
        frames_root: Path,
        out_root: Path,
        frames_per_video: int = 50,
        sam3_threshold: float = 0.3,
        prompts: list[str] | None = None,
    ) -> None:
        self.video_root = Path(video_root)
        self.frames_root = Path(frames_root)
        self.out_root = Path(out_root)
        self.frames_per_video = frames_per_video
        self.sam3_threshold = sam3_threshold
        self.prompts = list(prompts) if prompts is not None else list(PROMPTS)

        self.frames_root.mkdir(parents=True, exist_ok=True)
        self.out_root.mkdir(parents=True, exist_ok=True)

    # -- public ---------------------------------------------------------

    def run(self) -> None:
        self._extract_all()
        self._remove_bg_all()

    # -- step 1: video -> raw PNGs -------------------------------------

    def _extract_all(self) -> None:
        mp4s = sorted(self.video_root.glob("*.mp4"))
        if not mp4s:
            print(f"[postprocess] WARN: no MP4s in {self.video_root}")
            return
        total_frames = 0
        print(f"[postprocess] extract: {len(mp4s)} videos -> {self.frames_root}")
        for mp4 in mp4s:
            task = mp4.stem
            task_dir = self.frames_root / task
            n = extract_uniform(mp4, task_dir, self.frames_per_video)
            total_frames += n
            print(f"  {n:3d} frames -> {task[:70]}")
        print(f"[postprocess] extracted {total_frames} frames total")

    # -- step 2: raw PNG -> SAM3 bg-removed PNG -------------------------

    def _remove_bg_all(self) -> None:
        task_dirs = sorted(d for d in self.frames_root.iterdir() if d.is_dir())
        if not task_dirs:
            print(f"[postprocess] WARN: no per-task frame dirs in {self.frames_root}")
            return

        # Lazy: only construct the SAM3 segmenter if we actually have work to do.
        seg: SAM3Segmenter | None = None

        for task_dir in task_dirs:
            out_task = self.out_root / task_dir.name
            out_task.mkdir(parents=True, exist_ok=True)
            pngs = sorted(task_dir.glob("*.png"))
            pending = [p for p in pngs if not (out_task / p.name).exists()]
            if not pending:
                print(f"[postprocess] skip (all done): {task_dir.name}")
                continue
            if seg is None:
                print("[postprocess] loading SAM3...")
                seg = SAM3Segmenter()
            print(f"[postprocess] bg-remove: {task_dir.name}  ({len(pending)} frames)")
            for png in tqdm(pending, desc=task_dir.name[:40], leave=False):
                out_png = out_task / png.name
                try:
                    alpha = seg.segment_multi_prompt(
                        png, self.prompts, threshold=self.sam3_threshold
                    )
                    if alpha is None:
                        # No mask found — copy raw frame so downstream never sees
                        # a missing index. SAM3 caller would otherwise drop it.
                        shutil.copyfile(png, out_png)
                        continue
                    out_img = seg.remove_background(png, alpha)
                    out_img.save(out_png)
                except Exception as e:
                    print(f"  [warn] {png.name}: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="MP4 -> 50 frames -> SAM3 bg removal.")
    ap.add_argument("--video_root", required=True, type=Path,
                    help="Dir containing <task>.mp4 files.")
    ap.add_argument("--frames_root", required=True, type=Path,
                    help="Staging dir for raw extracted PNGs.")
    ap.add_argument("--out_root", required=True, type=Path,
                    help="Final dir: out_root/<task>/frame_NNNN.png (bg removed).")
    ap.add_argument("--frames_per_video", type=int, default=50)
    ap.add_argument("--sam3_threshold", type=float, default=0.3)
    args = ap.parse_args()
    QueryHighHarvester(
        video_root=args.video_root,
        frames_root=args.frames_root,
        out_root=args.out_root,
        frames_per_video=args.frames_per_video,
        sam3_threshold=args.sam3_threshold,
    ).run()


if __name__ == "__main__":
    main()
