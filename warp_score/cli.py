"""CLI for warp_score.

Subcommands:
    calibrate    Build per-task empirical null distributions from high/ refs.
    detect       Run detection on a query frame or directory.
    eval         Compute AUROC/AP/FPR@TPR from labels CSV + predictions CSV.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path

from .calibrator import CalibrationArtifact, EmpiricalNullCalibrator
from .config import WarpScoreConfig
from .detector import WarpVarianceDetector
from .evaluator import Evaluator
from .fuser import build_fuser
from .mask import InteriorMask
from .matcher import RoMaMatcher
from .signals import build_signals
from .visualizer import HeatmapPlotter


# ── Logging ──────────────────────────────────────────────────────────────────

def _setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("warp_score")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


# ── Subcommand handlers ──────────────────────────────────────────────────────

def calibrate_cmd(args: argparse.Namespace) -> None:
    cfg = _load_config(args)
    logger = _setup_logger(cfg.run_log)
    logger.info("=== Calibrate ===")
    logger.info(f"high_dir={cfg.high_dir}  artifacts={cfg.artifacts_dir}")

    high_refs = _discover_refs_by_task(cfg.high_dir)
    logger.info(
        f"Found {sum(len(v) for v in high_refs.values())} refs across {len(high_refs)} tasks"
    )

    matcher = RoMaMatcher(
        setting=cfg.setting,
        device=cfg.device,
        use_precision=cfg.use_precision,
        bidirectional=cfg.bidirectional,
        vis_size=cfg.vis_size,
    )
    calibrator = EmpiricalNullCalibrator(
        matcher=matcher,
        interior_mask=InteriorMask(cfg.erosion_k),
        config=cfg,
    )

    artifact = calibrator.calibrate(high_refs)
    artifact.save(cfg.calib_path)
    logger.info("Done.")


def detect_cmd(args: argparse.Namespace) -> None:
    cfg = _load_config(args)
    logger = _setup_logger(cfg.run_log)
    logger.info("=== Detect ===")

    if not cfg.calib_path.exists():
        raise FileNotFoundError(
            f"Calibration not found at {cfg.calib_path}. Run `calibrate` first."
        )
    calib = CalibrationArtifact.load(cfg.calib_path)
    logger.info(
        f"Calibration loaded: {calib.n_tasks} tasks, created {calib.created_at}"
    )

    detector = WarpVarianceDetector(
        config=cfg,
        matcher=RoMaMatcher(
            setting=cfg.setting, device=cfg.device,
            use_precision=cfg.use_precision, bidirectional=cfg.bidirectional,
            vis_size=cfg.vis_size,
        ),
        calib=calib,
        fuser=build_fuser(cfg.fuser, stouffer_weights=cfg.stouffer_weights),
        signals=build_signals(cfg.signal_names),
        interior_mask=InteriorMask(cfg.erosion_k),
    )
    plotter = HeatmapPlotter(vis_size=cfg.vis_size) if cfg.save_heatmaps else None

    # Resolve query list
    if args.query:
        queries = [Path(args.query)]
    elif args.query_dir:
        queries = sorted(Path(args.query_dir).glob("**/*.png"))
    else:
        # Default: scan low_dir
        queries = sorted(cfg.low_dir.glob("**/*.png"))
    if not queries:
        raise RuntimeError("No queries found.")
    logger.info(f"Processing {len(queries)} queries → {cfg.summary_csv}")

    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cfg.summary_csv
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as csvf:
        writer: csv.DictWriter | None = None
        for qi, qpath in enumerate(queries):
            logger.info(f"[{qi+1}/{len(queries)}] {qpath.parent.name}/{qpath.name}")
            try:
                result = detector.detect(qpath)
            except Exception as e:
                logger.error(f"detect failed: {e}")
                continue

            row = result.to_csv_row()
            if writer is None:
                writer = csv.DictWriter(csvf, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
            writer.writerow(row)
            csvf.flush()

            if plotter is not None:
                heatmap_dir = cfg.artifacts_dir / "heatmaps" / qpath.parent.name
                plotter.plot(
                    qpath, result, save_to=heatmap_dir / f"{qpath.stem}.png",
                )

    logger.info(f"Wrote: {csv_path}")


def eval_cmd(args: argparse.Namespace) -> None:
    cfg = _load_config(args)
    logger = _setup_logger(cfg.run_log)
    logger.info("=== Eval ===")

    pred_path = Path(args.pred) if args.pred else cfg.summary_csv
    labels_path = Path(args.labels)
    out_path = Path(args.out) if args.out else cfg.artifacts_dir / "eval_report.json"

    logger.info(f"labels={labels_path}  preds={pred_path}")
    report = Evaluator(labels_path, pred_path).evaluate()
    report.print_summary()
    report.save(out_path)
    logger.info(f"Saved report → {out_path}")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _discover_refs_by_task(root: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = defaultdict(list)
    for png in sorted(root.glob("**/*.png")):
        task = png.parent.name
        out[task].append(png)
    return dict(out)


def _load_config(args: argparse.Namespace) -> WarpScoreConfig:
    if args.config and Path(args.config).exists():
        cfg = WarpScoreConfig.from_yaml(args.config)
    else:
        cfg = WarpScoreConfig()
    # CLI overrides
    cfg = cfg.merge(
        high_dir=Path(args.high_dir) if getattr(args, "high_dir", None) else None,
        low_dir=Path(args.low_dir) if getattr(args, "low_dir", None) else None,
        artifacts_dir=Path(args.artifacts_dir) if getattr(args, "artifacts_dir", None) else None,
        device=getattr(args, "device", None),
        setting=getattr(args, "setting", None),
    )
    return cfg


# ── Entrypoint ───────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="warp_score")
    parser.add_argument(
        "--config", default=None,
        help="YAML config (default: warp_score/configs/default.yaml)",
    )
    parser.add_argument("--high_dir", default=None)
    parser.add_argument("--low_dir", default=None)
    parser.add_argument("--artifacts_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--setting", default=None)

    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("calibrate", help="Build per-task empirical null distributions")

    det = sub.add_parser("detect", help="Run detection on a query/dir")
    det.add_argument("--query", default=None, help="Single PNG path")
    det.add_argument("--query_dir", default=None, help="Dir of PNGs (overrides low_dir)")

    ev = sub.add_parser("eval", help="Compute AUROC/AP from labels + preds CSV")
    ev.add_argument("--labels", required=True)
    ev.add_argument("--pred", default=None, help="defaults to artifacts/summary.csv")
    ev.add_argument("--out", default=None, help="defaults to artifacts/eval_report.json")

    if argv is None:
        argv = sys.argv[1:]
    # If user passed config as last arg under subcmd, argparse handles it via the parent parser.
    args = parser.parse_args(argv)

    handlers = {
        "calibrate": calibrate_cmd,
        "detect": detect_cmd,
        "eval": eval_cmd,
    }
    handlers[args.cmd](args)


if __name__ == "__main__":
    main()
