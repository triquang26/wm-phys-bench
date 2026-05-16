"""Hallucination detection pipeline — 4 offline stages + online detection.

Offline stages are idempotent: each stage skips if its output already exists.

Usage:
    python main.py                                   # uses defaults
    python main.py --root_dir /path/to/feepe         # custom root
    python main.py --artifacts_dir /path/to/artif    # custom artifact dir
"""
from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import ImageDataset, RobotDataset
from detector import HallucinationDetector
from roma_utils import load_roma
from stage1_extract import PatchExtractor
from stage2_coreset import CoresetBuilder
from stage3_reference import ReferencePoolBuilder
from stage4_graph import GraphBuilder


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("HallucinationEval")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ── CSV output ────────────────────────────────────────────────────────────────

_CSV_FIELDS = [
    "image_id", "task", "frame", "split",
    "is_hallucination", "score", "score_patchcore", "score_graph",
]


def _save_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            row_out = dict(row)
            for k in ("score", "score_patchcore", "score_graph"):
                if k in row_out:
                    row_out[k] = f"{row_out[k]:.4f}"
            writer.writerow(row_out)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = Config.from_args()
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(cfg.artifacts_dir / "run.log")
    logger.info("=== Hallucination Detection Pipeline ===")
    logger.info(f"root_dir      : {cfg.root_dir}")
    logger.info(f"artifacts_dir : {cfg.artifacts_dir}")
    logger.info(f"device        : {cfg.device}")

    # ── Dataset + DataLoader ──────────────────────────────────────────────────
    dataset = ImageDataset(cfg.root_dir)
    logger.info(
        f"Dataset: {len(dataset.high_images)} high images, "
        f"{len(dataset.low_images)} low images across "
        f"{len(dataset.unique_tasks('high'))} tasks."
    )

    high_loader = DataLoader(
        RobotDataset(dataset.high_images),
        batch_size=cfg.batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )

    # ── Load backbone (shared across stages 1, 3, calibration) ───────────────
    logger.info("Loading DINOv2 ViT-L/14 …")
    backbone = (
        torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", verbose=False)
        .to(cfg.device)
        .eval()
    )

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 1 — Patch feature extraction → HDF5
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=== STAGE 1: Patch feature extraction ===")
    PatchExtractor(backbone, cfg).run(high_loader)

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2 — Coreset + FAISS index
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=== STAGE 2: Coreset + FAISS ===")
    CoresetBuilder(cfg).run()

    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 3 — Reference pool (CLS clustering)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=== STAGE 3: Reference pool ===")
    ReferencePoolBuilder(backbone, dataset, cfg).run(high_loader)

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 4 — Reference consistency graph
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=== STAGE 4: Reference consistency graph ===")
    matcher = load_roma(cfg.device)
    GraphBuilder(matcher, dataset, cfg).run()

    # ══════════════════════════════════════════════════════════════════════════
    # CALIBRATION — threshold from nominal (high) images
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=== CALIBRATION ===")
    detector = HallucinationDetector(backbone, matcher, dataset, cfg)
    detector.calibrate(high_loader)

    # ══════════════════════════════════════════════════════════════════════════
    # DETECTION — run on low images
    # ══════════════════════════════════════════════════════════════════════════
    logger.info(f"=== DETECTION: {len(dataset.low_images)} low images ===")
    rows: list[dict] = []
    for record in tqdm(dataset.iter_low(), total=len(dataset.low_images), desc="Detecting"):
        try:
            result = detector.detect_image(record)
            rows.append(result)
        except Exception as exc:
            logger.warning(f"Failed on {record.image_id}: {exc}")

    _save_csv(rows, cfg.output_csv)
    n_halluc = sum(1 for r in rows if r.get("is_hallucination"))
    logger.info(
        f"Done. {len(rows)} images evaluated, {n_halluc} hallucinations detected."
    )
    logger.info(f"Results → {cfg.output_csv}")


if __name__ == "__main__":
    main()
