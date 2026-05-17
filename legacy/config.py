from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    root_dir: Path
    artifacts_dir: Path
    output_csv: Path
    device: str

    # Stage 1 — patch extraction
    dino_layers: tuple[int, int] = (8, 16)
    patch_agg_kernel: int = 3
    batch_size: int = 32

    # Stage 2 — coreset
    # K-means pre-step skipped when total patches < skip_kmeans_below
    skip_kmeans_below: int = 500_000
    coreset_intermediate_size: int = 10_000
    coreset_final_size: int = 3_000
    coreset_proj_dim: int = 128

    # Stage 3 — reference pool
    # n_clusters=0 → auto-set to number of unique tasks
    n_clusters: int = 0
    n_boundary_per_cluster: int = 2

    # Stage 4 — consistency graph
    top_k_neighbors: int = 10
    checkpoint_every: int = 100
    roma_num_samples: int = 1000

    # Online detection
    alpha: float = 0.5        # PatchCore weight in fusion
    K_refs: int = 10          # active references per query
    b_neighbors: int = 9      # neighborhood for PatchCore reweighting
    target_fpr: float = 0.05

    # Derived paths (set in __post_init__)
    h5_path: Path = field(init=False)
    mc_path: Path = field(init=False)
    faiss_path: Path = field(init=False)
    global_feats_path: Path = field(init=False)
    ref_indices_path: Path = field(init=False)
    ref_feats_path: Path = field(init=False)
    s_ref_path: Path = field(init=False)
    mu_ref_path: Path = field(init=False)
    sigma_ref_path: Path = field(init=False)
    calibration_path: Path = field(init=False)

    def __post_init__(self) -> None:
        a = self.artifacts_dir
        self.h5_path = a / "memory_bank.h5"
        self.mc_path = a / "MC.npy"
        self.faiss_path = a / "faiss.idx"
        self.global_feats_path = a / "global_feats.npy"
        self.ref_indices_path = a / "ref_indices.npy"
        self.ref_feats_path = a / "ref_feats_normalized.npy"
        self.s_ref_path = a / "S_ref.npy"
        self.mu_ref_path = a / "mu_ref.npy"
        self.sigma_ref_path = a / "sigma_ref.npy"
        self.calibration_path = a / "calibration.npz"

    @classmethod
    def from_args(cls) -> "Config":
        parser = argparse.ArgumentParser(
            description="PatchCore + Reference Graph hallucination detector"
        )
        parser.add_argument(
            "--root_dir",
            default="/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe",
        )
        parser.add_argument("--artifacts_dir", default=None)
        parser.add_argument("--output_csv", default=None)
        parser.add_argument("--device", default="cuda")
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--coreset_final_size", type=int, default=3_000)
        parser.add_argument("--n_clusters", type=int, default=0)
        parser.add_argument("--top_k_neighbors", type=int, default=10)
        parser.add_argument("--alpha", type=float, default=0.5)
        parser.add_argument("--K_refs", type=int, default=10)
        parser.add_argument("--target_fpr", type=float, default=0.05)
        args = parser.parse_args()

        root = Path(args.root_dir)
        artifacts = Path(args.artifacts_dir) if args.artifacts_dir else root / "artifacts"
        output_csv = Path(args.output_csv) if args.output_csv else artifacts / "results.csv"

        return cls(
            root_dir=root,
            artifacts_dir=artifacts,
            output_csv=output_csv,
            device=args.device,
            batch_size=args.batch_size,
            coreset_final_size=args.coreset_final_size,
            n_clusters=args.n_clusters,
            top_k_neighbors=args.top_k_neighbors,
            alpha=args.alpha,
            K_refs=args.K_refs,
            target_fpr=args.target_fpr,
        )
