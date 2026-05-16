"""prepare_data.py — Download GR00T-GR1 HF dataset, extract first frames, emit batch JSON.

Output layout:
    <out_dir>/
        frames/0001.png … 0100.png
        metadata.jsonl          (one row per item)
        batch_input.json        (cosmos-predict2-compatible batch file)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm


HF_REPO = "nvidia/PhysicalAI-Robotics-GR00T-GR1"


@dataclass
class PrepareResult:
    n_items: int
    out_dir: Path
    batch_json: Path
    metadata_jsonl: Path
    frames_dir: Path


class DreamGenDataPreparer:
    """Download HF dataset, extract first frames, write batch_input.json."""

    def __init__(
        self,
        out_dir: Path,
        cache_dir: Optional[Path] = None,
        save_path_template: str = "output/{profile}/item_{idx:04d}/video.mp4",
    ) -> None:
        self.out_dir = Path(out_dir)
        self.frames_dir = self.out_dir / "frames"
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.save_path_template = save_path_template

    def prepare(self, max_items: int = 100) -> PrepareResult:
        """Run the full preparation pipeline."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        print(f"[prepare] loading dataset {HF_REPO}…")
        ds = load_dataset(
            HF_REPO,
            split="train",
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
        )
        print(f"[prepare] dataset loaded, {len(ds)} rows")

        n = min(max_items, len(ds))
        metadata: list[dict] = []
        batch: list[dict] = []

        for i in tqdm(range(n), desc="extracting first frames"):
            record = self._process_one(i, ds[i])
            if record is None:
                continue
            metadata.append(record)
            batch.append(self._to_batch_item(record))

        metadata_path = self._write_metadata(metadata)
        batch_path = self._write_batch_input(batch)

        print(f"[prepare] wrote {len(metadata)} items")
        print(f"[prepare] frames:        {self.frames_dir}")
        print(f"[prepare] metadata:      {metadata_path}")
        print(f"[prepare] batch_input:   {batch_path}")

        return PrepareResult(
            n_items=len(metadata),
            out_dir=self.out_dir,
            batch_json=batch_path,
            metadata_jsonl=metadata_path,
            frames_dir=self.frames_dir,
        )

    # ─────────────────────────────────────────────────────────────────────────

    def _process_one(self, i: int, row: dict) -> Optional[dict]:
        prompt = row["text"].strip()
        video_path = self._resolve_video_path(row, i)
        if video_path is None:
            print(f"[prepare] WARN: no video for item {i}", file=sys.stderr)
            return None

        png_path = self.frames_dir / f"{i + 1:04d}.png"
        if not png_path.exists():
            ok = self._extract_first_frame(video_path, png_path)
            if not ok:
                print(
                    f"[prepare] WARN: failed to extract frame from {video_path}",
                    file=sys.stderr,
                )
                return None

        return {
            "idx": i + 1,
            "prompt": prompt,
            "image": str(png_path.resolve()),
            "video_source": str(video_path),
        }

    def _resolve_video_path(self, row: dict, i: int) -> Optional[str]:
        v = row["video"]
        if isinstance(v, dict):
            video_path = v.get("path") or v.get("bytes")
        else:
            video_path = str(v)

        if video_path is None or not os.path.exists(str(video_path)):
            # Fallback: download the mp4 directly via HF Hub.
            video_path = hf_hub_download(
                repo_id=HF_REPO,
                filename=f"{i + 1}.mp4",
                repo_type="dataset",
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
            )
        return str(video_path) if video_path else None

    @staticmethod
    def _extract_first_frame(video_path: str, out_png: Path) -> bool:
        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return False
        cv2.imwrite(str(out_png), frame)
        return True

    def _to_batch_item(self, record: dict) -> dict:
        return {
            "input_video": record["image"],
            "prompt": record["prompt"],
            "output_video": self.save_path_template.format(
                idx=record["idx"],
                profile="{profile}",
            ),
        }

    def _write_metadata(self, metadata: list[dict]) -> Path:
        path = self.out_dir / "metadata.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for row in metadata:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path

    def _write_batch_input(self, batch: list[dict]) -> Path:
        path = self.out_dir / "batch_input.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(batch, f, indent=2, ensure_ascii=False)
        return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out_dir", type=str, default="data",
        help="Where to place frames/, metadata.jsonl, batch_input.json",
    )
    ap.add_argument(
        "--max_items", type=int, default=100,
        help="Limit number of items (the dataset has 100 total)",
    )
    ap.add_argument(
        "--cache_dir", type=str, default=None,
        help="HF cache dir (defaults to ~/.cache/huggingface)",
    )
    ap.add_argument(
        "--save_path_template",
        type=str,
        default="output/{profile}/item_{idx:04d}/video.mp4",
        help="Template for the 'output_video' field in batch_input.json.",
    )
    args = ap.parse_args()

    preparer = DreamGenDataPreparer(
        out_dir=Path(args.out_dir),
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        save_path_template=args.save_path_template,
    )
    preparer.prepare(max_items=args.max_items)


if __name__ == "__main__":
    main()
