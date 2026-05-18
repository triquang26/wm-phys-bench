"""RoMaMatcher — wrapper around RoMaV2 dense warp matcher.

match(query, ref) → (warp_HW2 normalized [-1,1], cert_HW [0,1])
both resized to (vis_size, vis_size).

Background certainty zeroed if a foreground mask is provided — prevents
textureless gray pixels from biasing the cert-weighted statistics.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import torch
import torch.nn.functional as F


class MatchResult(NamedTuple):
    warp: np.ndarray       # (H, W, 2)  float32 in [-1, 1]
    cert: np.ndarray       # (H, W)     float32 in [0, 1] (bg zeroed if mask given)
    precision: Optional[np.ndarray] = None  # (H, W, 2, 2) float32, bg zeroed


class RoMaMatcher:
    """Thin OOP wrapper around RoMaV2."""

    def __init__(
        self,
        setting: str = "turbo",
        device: str = "cuda",
        use_precision: bool = True,
        bidirectional: bool = False,
        vis_size: int = 224,
        h_lr: int = 0,
        w_lr: int = 0,
    ) -> None:
        self.setting = setting
        self.device = device
        self.use_precision = use_precision
        self.bidirectional = bidirectional
        self.vis_size = vis_size
        self.h_lr = h_lr
        self.w_lr = w_lr
        self._model = None

    # ─────────────────────────────────────────────────────────────────────────

    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            from romav2 import RoMaV2
        except ImportError:
            # Fallback: locate RoMaV2 source bundled in third_party/
            here = Path(__file__).resolve()
            candidates = [
                here.parents[1] / "third_party" / "RoMaV2" / "src",
                here.parents[3] / "RoMaV2" / "src",
            ]
            for romav2_src in candidates:
                if romav2_src.exists():
                    sys.path.insert(0, str(romav2_src))
                    break
            else:
                raise ImportError(
                    f"Could not import romav2. Tried: {[str(c) for c in candidates]}"
                )
            from romav2 import RoMaV2  # type: ignore

        model = RoMaV2()
        model.apply_setting(self.setting)
        model.balanced_sampling = True
        if self.bidirectional:
            model.bidirectional = True
        if self.h_lr > 0:
            model.H_lr = self.h_lr
        if self.w_lr > 0:
            model.W_lr = self.w_lr
        model.eval()
        model = model.to(self.device)
        self._model = model
        return model

    # ─────────────────────────────────────────────────────────────────────────

    def _load_tensor(self, img_like: "Path | str | np.ndarray", size: tuple[int, int]) -> "torch.Tensor":
        """Load image → (1, 3, H, W) float32 on model device, resized to size."""
        from PIL import Image as PILImage
        if isinstance(img_like, (str, Path)):
            img = PILImage.open(img_like).convert("RGB")
            t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        elif isinstance(img_like, np.ndarray):
            t = torch.from_numpy(img_like[..., ::-1].copy() if img_like.shape[-1] == 3 else img_like).permute(2, 0, 1).float() / 255.0
        else:
            t = img_like.float() / 255.0 if img_like.max() > 1.0 else img_like.float()
        t = t.unsqueeze(0).to(self.device)
        return F.interpolate(t, size=size, mode="bicubic", align_corners=False, antialias=True)

    @torch.no_grad()
    def match_batch(
        self,
        query_path: "Path | str",
        ref_paths: "list[Path | str]",
        fg_mask: Optional[np.ndarray] = None,
    ) -> "list[MatchResult]":
        """Match query against all refs in one GPU forward pass.

        Same semantics as calling match() k times but 3-5x faster because
        RoMaV2's transformer forward is batched natively.
        """
        model = self._load_model()
        k = len(ref_paths)
        size_lr = (model.H_lr, model.W_lr)
        size_out = (self.vis_size, self.vis_size)

        # Load query once, repeat k times → (k, 3, H_lr, W_lr)
        img_q = self._load_tensor(query_path, size_lr)
        img_q_batch = img_q.expand(k, -1, -1, -1)

        # Load and stack all refs → (k, 3, H_lr, W_lr)
        img_refs = torch.cat([self._load_tensor(r, size_lr) for r in ref_paths], dim=0)

        # Single forward pass (no HR stage for turbo/small settings)
        preds = model(img_q_batch, img_refs)

        # Map confidence → overlap (k,H,W,1) and precision (k,H,W,2,2)
        from romav2.romav2 import _map_confidence  # module-level helper
        confidence = preds["confidence_AB"]
        overlap_batch, precision_batch = _map_confidence(
            confidence=confidence, threshold=model.threshold
        )

        results: list[MatchResult] = []
        for i in range(k):
            warp_i = preds["warp_AB"][i]  # (H_lr, W_lr, 2)

            # Resize warp → vis_size
            warp_t = warp_i.permute(2, 0, 1).unsqueeze(0).float()
            warp_r = F.interpolate(warp_t, size=size_out, mode="bilinear", align_corners=False)
            warp_np = warp_r[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)

            # Cert from precision det or overlap
            if self.use_precision:
                prec_i = precision_batch[i]  # (H_lr, W_lr, 2, 2)
                det = prec_i[..., 0, 0] * prec_i[..., 1, 1] - prec_i[..., 0, 1] * prec_i[..., 1, 0]
                cert_raw = det.clamp(min=0).sqrt()
                cert_raw = cert_raw / (cert_raw.amax() + 1e-8)
            else:
                cert_raw = overlap_batch[i, ..., 0]

            cert_t = cert_raw.unsqueeze(0).unsqueeze(0).float()
            cert_r = F.interpolate(cert_t, size=size_out, mode="bilinear", align_corners=False)
            cert_np = cert_r[0, 0].cpu().numpy().astype(np.float32)
            if fg_mask is not None:
                cert_np[~fg_mask] = 0.0

            # Precision matrix → vis_size
            prec_i = precision_batch[i]  # (H_lr, W_lr, 2, 2)
            H0, W0 = prec_i.shape[:2]
            prec_flat = prec_i.reshape(H0, W0, 4).permute(2, 0, 1).unsqueeze(0).float()
            prec_r = F.interpolate(prec_flat, size=size_out, mode="bilinear", align_corners=False)
            precision_np = prec_r[0].permute(1, 2, 0).reshape(
                self.vis_size, self.vis_size, 2, 2
            ).cpu().numpy().astype(np.float32)
            if fg_mask is not None:
                precision_np[~fg_mask] = 0.0

            results.append(MatchResult(warp=warp_np, cert=cert_np, precision=precision_np))

        return results

    @torch.no_grad()
    def match(
        self,
        query_path: Path | str,
        ref_path: Path | str,
        fg_mask: Optional[np.ndarray] = None,
    ) -> MatchResult:
        model = self._load_model()
        preds = model.match(str(query_path), str(ref_path))
        warp = preds["warp_AB"][0]  # (H, W, 2)
        cert = self._cert_from_preds(preds)  # (H, W)

        warp_t = warp.permute(2, 0, 1).unsqueeze(0).float()
        cert_t = cert.unsqueeze(0).unsqueeze(0).float()

        size = (self.vis_size, self.vis_size)
        warp_r = F.interpolate(warp_t, size=size, mode="bilinear", align_corners=False)
        cert_r = F.interpolate(cert_t, size=size, mode="bilinear", align_corners=False)

        warp_np = warp_r[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
        cert_np = cert_r[0, 0].cpu().numpy().astype(np.float32)

        if fg_mask is not None:
            cert_np[~fg_mask] = 0.0

        # ── Extract full precision matrix (H, W, 2, 2) ───────────────────────
        precision_np: Optional[np.ndarray] = None
        raw_prec = preds.get("precision_AB")
        if raw_prec is not None:
            # raw_prec[0]: (H_orig, W_orig, 2, 2) tensor
            prec_orig = raw_prec[0]  # (H, W, 2, 2)
            H_orig, W_orig = prec_orig.shape[:2]
            # Reshape to (1, 4, H, W) for bilinear interpolation
            prec_flat = prec_orig.reshape(H_orig, W_orig, 4).permute(2, 0, 1).unsqueeze(0).float()
            prec_r = F.interpolate(prec_flat, size=size, mode="bilinear", align_corners=False)
            # prec_r: (1, 4, vis_size, vis_size) → (vis_size, vis_size, 2, 2)
            precision_np = prec_r[0].permute(1, 2, 0).reshape(
                self.vis_size, self.vis_size, 2, 2
            ).cpu().numpy().astype(np.float32)
            if fg_mask is not None:
                precision_np[~fg_mask] = 0.0

        return MatchResult(warp=warp_np, cert=cert_np, precision=precision_np)

    # ─────────────────────────────────────────────────────────────────────────

    def _cert_from_preds(self, preds: dict) -> torch.Tensor:
        if self.use_precision and preds.get("precision_AB") is not None:
            prec = preds["precision_AB"][0]
            det = (
                prec[..., 0, 0] * prec[..., 1, 1]
                - prec[..., 0, 1] * prec[..., 1, 0]
            )
            cert = det.clamp(min=0).sqrt()
            cert = cert / (cert.amax() + 1e-8)
        else:
            cert = preds["overlap_AB"][0]
            if cert.dim() == 3:
                cert = cert.squeeze(-1)
        return cert
