from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    split: str       # "high" | "low"
    task: str
    frame: str

    @property
    def image_id(self) -> str:
        slug = (
            self.task
            .replace("/", "_")
            .replace(" ", "_")
            .replace("'", "")
            .replace(".", "")
        )
        return f"{slug[:55]}__{self.frame}"


class ImageDataset:
    """
    Discovers all PNG images under:
        {root}/image_no_bg/high/{task}/frame_XXXX.png
        {root}/image_no_bg/low/{task}/frame_XXXX.png
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._base = self._root / "image_no_bg"
        self.high_images: list[ImageRecord] = []
        self.low_images: list[ImageRecord] = []
        self._discover()

    def _discover(self) -> None:
        for split, store in (("high", self.high_images), ("low", self.low_images)):
            split_dir = self._base / split
            if not split_dir.is_dir():
                raise FileNotFoundError(f"Directory not found: {split_dir}")
            for task_dir in sorted(split_dir.iterdir()):
                if not task_dir.is_dir():
                    continue
                for img_path in sorted(task_dir.glob("*.png")):
                    store.append(ImageRecord(
                        path=img_path,
                        split=split,
                        task=task_dir.name,
                        frame=img_path.stem,
                    ))

    def sample_high_records(
        self,
        k: int,
        rng: random.Random | None = None,
    ) -> list[ImageRecord]:
        """Draw k ImageRecords from high_images at random."""
        if len(self.high_images) < k:
            raise ValueError(
                f"Requested {k} high images but only {len(self.high_images)} available."
            )
        sampler = rng if rng is not None else random
        return sampler.sample(self.high_images, k)

    def iter_low(self) -> Iterator[ImageRecord]:
        yield from self.low_images

    def unique_tasks(self, split: str = "high") -> list[str]:
        src = self.high_images if split == "high" else self.low_images
        return sorted({r.task for r in src})

    def __repr__(self) -> str:
        return (
            f"ImageDataset(root={str(self._root)!r}, "
            f"high={len(self.high_images)}, low={len(self.low_images)})"
        )


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class RobotDataset(Dataset):
    """Wraps a list[ImageRecord] for use with DataLoader.

    Returns (tensor [3,224,224], global_index int) per item.
    """

    SIZE = 224
    _tf = transforms.Compose([
        transforms.Resize(SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(SIZE),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])

    def __init__(self, records: list[ImageRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        img = Image.open(self.records[i].path).convert("RGB")
        return self._tf(img), i
