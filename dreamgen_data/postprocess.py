"""postprocess.py — Bridge from cosmos-predict2 video output to image_no_bg/ format.

Pipeline:
    output/<profile>/item_XXXX/*.mp4
        → extract N frames from each video
        → SAM3 background removal (bg → (127,127,127))
        → image_no_bg/<split>/<task>/frame_NNNN.png

Mapping:
    profile="high"        → split="high"   (calibration refs, label=0)
    profile="hallucinate" → split="low"    (test set, label=1)

Task name = sanitized GR1 prompt (from metadata.jsonl), prefixed with item idx.

Idempotent: skips a target PNG if it already exists.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal, Optional

import cv2
import numpy as np
from tqdm import tqdm


# ============================================================================
# Frame extraction
# ============================================================================

class VideoFrameExtractor:
    """Extract N frames from an mp4. Strategies: 'uniform' (evenly spaced) or 'tail'."""

    def __init__(
        self,
        frames_per_video: int = 1,
        strategy: Literal["uniform", "tail", "first"] = "uniform",
    ) -> None:
        if frames_per_video < 1:
            raise ValueError(f"frames_per_video must be >= 1, got {frames_per_video}")
        self.frames_per_video = frames_per_video
        self.strategy = strategy

    def extract(self, video_path: Path) -> list[np.ndarray]:
        cap = cv2.VideoCapture(str(video_path))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n_frames <= 0:
            cap.release()
            return []

        target_idxs = self._target_indices(n_frames)
        frames: list[np.ndarray] = []
        for idx in target_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
        cap.release()
        return frames

    def _target_indices(self, n_frames: int) -> list[int]:
        k = min(self.frames_per_video, n_frames)
        if self.strategy == "first":
            return list(range(k))
        if self.strategy == "tail":
            return list(range(n_frames - k, n_frames))
        # uniform
        return [
            int(round((i + 1) * n_frames / (k + 1))) - 1
            for i in range(k)
        ]


# ============================================================================
# Background removal — SAM3
# ============================================================================

class SAM3BackgroundRemover:
    """Set non-foreground pixels to (127,127,127). Wraps existing feepe/sam3.py logic.

    Lazy import: sam3 deps are heavy. Construct + run only when actually needed.
    """

    BG_VALUE = (127, 127, 127)

    def __init__(
        self,
        device: str = "cuda",
        text_prompts: Optional[list[str]] = None,
        score_thresh: float = 0.3,
    ) -> None:
        self.device = device
        self.text_prompts = text_prompts or ["robot arm", "robot gripper", "object"]
        self.score_thresh = score_thresh
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            # Try to reuse the existing SAM3 wrapper in feepe/
            import sys
            sam3_dir = Path(__file__).parent.parent.parent
            if str(sam3_dir) not in sys.path:
                sys.path.insert(0, str(sam3_dir))
            import sam3  # type: ignore
            # The exact API is project-specific; expose just what we need.
            self._model = sam3
        except ImportError as e:
            raise ImportError(
                "Could not import sam3 module. Ensure calibration/feepe/sam3.py is "
                "accessible and SAM3 is installed."
            ) from e

    def remove(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Run SAM3, set bg pixels to (127,127,127).

        If sam3 module fails or fg mask is empty, returns the original frame
        unchanged (caller can decide to drop it).
        """
        self._ensure_model()
        try:
            mask = self._predict_mask(frame_bgr)
        except Exception as e:
            print(f"[postprocess] SAM3 failed: {e}; returning raw frame.")
            return frame_bgr

        if mask is None or not mask.any():
            return frame_bgr

        out = frame_bgr.copy()
        out[~mask] = self.BG_VALUE
        return out

    def _predict_mask(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Adapter — overridable. Default tries common entrypoints on the sam3 module."""
        sam3 = self._model
        # Possible APIs: predict(image, prompts) or run(image)
        for attr in ("predict_mask", "predict", "segment", "run"):
            fn = getattr(sam3, attr, None)
            if callable(fn):
                result = fn(
                    frame_bgr,
                    text_prompts=self.text_prompts,
                    score_thresh=self.score_thresh,
                    device=self.device,
                )
                # Result may be dict {'mask': ndarray} or ndarray directly
                if isinstance(result, dict):
                    return result.get("mask")
                if isinstance(result, np.ndarray):
                    return result.astype(bool)
        raise RuntimeError(
            "SAM3 module has no recognized entrypoint. "
            "Override _predict_mask() with the correct call signature."
        )


# ============================================================================
# Harvester — orchestrator
# ============================================================================

@dataclass
class HarvestResult:
    profile: str
    split: str
    n_videos_seen: int = 0
    n_frames_extracted: int = 0
    n_frames_written: int = 0
    n_skipped_existing: int = 0
    failed_videos: list[str] = field(default_factory=list)


class DreamGenHarvester:
    """video output/<profile>/item_XXXX/*.mp4 → image_no_bg/<split>/<task>/frame_NNNN.png."""

    PROFILE_TO_SPLIT = {"high": "high", "hallucinate": "low"}

    def __init__(
        self,
        extractor: VideoFrameExtractor,
        remover: SAM3BackgroundRemover,
        out_root: Path,
        metadata_jsonl: Path,
        max_task_name_len: int = 80,
    ) -> None:
        self.extractor = extractor
        self.remover = remover
        self.out_root = Path(out_root)
        self.max_task_name_len = max_task_name_len
        self.metadata = self._load_metadata(metadata_jsonl)

    def harvest(self, video_root: Path, profile: str) -> HarvestResult:
        if profile not in self.PROFILE_TO_SPLIT:
            raise ValueError(
                f"Unknown profile '{profile}'. Known: {list(self.PROFILE_TO_SPLIT)}"
            )
        split = self.PROFILE_TO_SPLIT[profile]
        result = HarvestResult(profile=profile, split=split)

        video_root = Path(video_root)
        item_dirs = sorted(d for d in video_root.iterdir() if d.is_dir() and d.name.startswith("item_"))

        for item_dir in tqdm(item_dirs, desc=f"harvest {profile}"):
            item_idx = self._parse_item_idx(item_dir.name)
            task_dir_name = self._task_dir_name(item_idx)
            task_out_dir = self.out_root / split / task_dir_name
            task_out_dir.mkdir(parents=True, exist_ok=True)

            self._harvest_one_item(item_dir, task_out_dir, result)

        print(
            f"[harvest] profile={profile}  split={split}  "
            f"videos={result.n_videos_seen}  frames_written={result.n_frames_written}  "
            f"skipped={result.n_skipped_existing}  failed={len(result.failed_videos)}"
        )
        return result

    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_metadata(metadata_jsonl: Path) -> dict[int, dict]:
        out: dict[int, dict] = {}
        with open(metadata_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                out[int(rec["idx"])] = rec
        return out

    @staticmethod
    def _parse_item_idx(item_dir_name: str) -> int:
        # "item_0023" → 23
        return int(item_dir_name.split("_")[1])

    def _task_dir_name(self, item_idx: int) -> str:
        meta = self.metadata.get(item_idx)
        if meta is None:
            return f"{item_idx}_unknown"
        prompt = meta["prompt"]
        safe = (
            prompt.replace("/", "_")
                  .replace("'", "")
                  .replace("\n", " ")
                  .strip()
        )
        truncated = safe[: self.max_task_name_len]
        return f"{item_idx}_{truncated}"

    def _harvest_one_item(
        self,
        item_dir: Path,
        task_out_dir: Path,
        result: HarvestResult,
    ) -> None:
        videos = sorted(item_dir.glob("*.mp4"))
        for video_path in videos:
            result.n_videos_seen += 1
            frames = self.extractor.extract(video_path)
            if not frames:
                result.failed_videos.append(str(video_path))
                continue
            result.n_frames_extracted += len(frames)

            for offset, frame in enumerate(frames):
                stem = video_path.stem
                if self.extractor.frames_per_video == 1:
                    out_path = task_out_dir / f"{stem}.png"
                else:
                    out_path = task_out_dir / f"{stem}_f{offset:02d}.png"

                if out_path.exists():
                    result.n_skipped_existing += 1
                    continue

                masked = self.remover.remove(frame)
                cv2.imwrite(str(out_path), masked)
                result.n_frames_written += 1


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Harvest cosmos-predict2 videos → image_no_bg/")
    ap.add_argument("--video_root", type=str, required=True,
                    help="output/<profile>/ dir containing item_XXXX subdirs")
    ap.add_argument("--profile", choices=list(DreamGenHarvester.PROFILE_TO_SPLIT), required=True)
    ap.add_argument("--metadata", type=str, default="data/metadata.jsonl")
    ap.add_argument("--out_root", type=str, required=True,
                    help="image_no_bg root (will create <out_root>/<split>/<task>/)")
    ap.add_argument("--frames_per_video", type=int, default=1)
    ap.add_argument("--strategy", choices=["uniform", "tail", "first"], default="uniform")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--no_sam3", action="store_true",
                    help="Skip SAM3 bg removal (debug — keeps raw frames).")
    args = ap.parse_args()

    extractor = VideoFrameExtractor(
        frames_per_video=args.frames_per_video,
        strategy=args.strategy,
    )

    if args.no_sam3:
        class _IdentityRemover:
            def remove(self, f): return f
        remover = _IdentityRemover()  # type: ignore
    else:
        remover = SAM3BackgroundRemover(device=args.device)

    harvester = DreamGenHarvester(
        extractor=extractor,
        remover=remover,  # type: ignore[arg-type]
        out_root=Path(args.out_root),
        metadata_jsonl=Path(args.metadata),
    )
    harvester.harvest(video_root=Path(args.video_root), profile=args.profile)


if __name__ == "__main__":
    main()
