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
    warp: np.ndarray  # (H, W, 2)  float32 in [-1, 1]
    cert: np.ndarray  # (H, W)     float32 in [0, 1] (bg zeroed if mask given)


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

        return MatchResult(warp=warp_np, cert=cert_np)

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
