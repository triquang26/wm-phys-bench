"""generate.py — Bulk generate N videos per prompt using cosmos-predict2.

Two profiles (see profiles.py):
  - high         : best-quality config (post-trained GR1 ckpt, CFG=7, refiner on)
  - hallucinate  : config tuned to maximize hallucinations (base ckpt, low CFG,
                   refiner off, random seed + guidance jitter)

Usage (must be run from inside cosmos-predict2 dir OR with PYTHONPATH set):

    cd cosmos-predict2
    torchrun --nproc_per_node=1 ../generate.py \
        --profile high --num_videos 50 \
        --batch_json ../data/batch_input.json \
        --save_dir ../output/high \
        --start_idx 0 --end_idx 23
"""
from __future__ import annotations

import argparse
import json
import os
import random
import socket
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch

from profiles import GenerationProfile, fallback_to_2b


# Cosmos-Predict2 native API. Imported lazily inside PipelineFactory because the
# stack is heavy and not always installed (allows unit tests on profiles).
def _import_cosmos():
    from imaginaire.constants import get_cosmos_predict2_video2world_checkpoint  # noqa: F401
    from imaginaire.utils.io import save_image_or_video
    from cosmos_predict2.configs.base.config_video2world import (
        get_cosmos_predict2_video2world_pipeline,
    )
    from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
    return {
        "save_image_or_video": save_image_or_video,
        "get_pipeline_config": get_cosmos_predict2_video2world_pipeline,
        "Video2WorldPipeline": Video2WorldPipeline,
    }


# =============================================================================
# Pipeline factory
# =============================================================================

class PipelineFactory:
    """Build (and cache) a Video2WorldPipeline given a GenerationProfile."""

    def __init__(self, ckpt_root: Optional[Path] = None) -> None:
        self.ckpt_root = Path(
            ckpt_root or os.environ.get("COSMOS_CKPT_ROOT", "checkpoints")
        )
        self._pipe = None
        self._loaded_profile_key: Optional[tuple] = None

    def get(self, profile: GenerationProfile):
        """Return a pipeline matching `profile`. Caches across calls."""
        key = self._profile_key(profile)
        if self._pipe is None or self._loaded_profile_key != key:
            self._pipe = self._build(profile)
            self._loaded_profile_key = key
        return self._pipe

    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _profile_key(profile: GenerationProfile) -> tuple:
        # Only attributes that affect which weights to load.
        return (profile.model_size, profile.gr00t_variant, profile.resolution, profile.fps)

    def _resolve_ckpt_path(self, profile: GenerationProfile) -> Path:
        if profile.gr00t_variant == "gr1":
            sub = (
                f"nvidia/Cosmos-Predict2-{profile.model_size}-"
                f"Video2World-Sample-GR00T-Dreams-GR1"
            )
        elif profile.gr00t_variant == "droid":
            sub = (
                f"nvidia/Cosmos-Predict2-{profile.model_size}-"
                f"Video2World-Sample-GR00T-Dreams-DROID"
            )
        else:
            sub = f"nvidia/Cosmos-Predict2-{profile.model_size}-Video2World"
        return self.ckpt_root / sub / f"model-{profile.resolution}p-{profile.fps}fps.pt"

    def _build(self, profile: GenerationProfile):
        cosmos = _import_cosmos()
        pipe_cfg = cosmos["get_pipeline_config"](model_size=profile.model_size)
        dit_path = self._resolve_ckpt_path(profile)
        print(f"[generate] loading DiT from {dit_path}")
        pipe = cosmos["Video2WorldPipeline"].from_config(
            config=pipe_cfg,
            dit_path=str(dit_path),
        )

        if profile.disable_guardrail and hasattr(pipe, "guardrail"):
            pipe.guardrail = None
            print("[generate] guardrail disabled")
        if profile.disable_prompt_refiner:
            try:
                pipe.config.prompt_refiner_config.enabled = False
                print("[generate] prompt refiner disabled")
            except AttributeError:
                print("[generate] WARN: could not disable prompt_refiner_config (API drift)")

        return pipe


# =============================================================================
# Inference adapter (handles cosmos-predict2 signature drift)
# =============================================================================

def _call_pipeline(
    pipe,
    *,
    prompt: str,
    negative_prompt: str,
    image_path: str,
    num_conditional_frames: int,
    guidance: float,
    seed: int,
    aspect_ratio: str,
):
    """Documented signature first; fall back to diffusers-style on TypeError."""
    try:
        return pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            input_path=image_path,
            num_conditional_frames=num_conditional_frames,
            guidance=guidance,
            seed=seed,
            aspect_ratio=aspect_ratio,
        )
    except TypeError as e:
        print(f"[generate] primary signature failed ({e}); trying diffusers-style")
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        out = pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        )
        return out.frames[0] if hasattr(out, "frames") else out


# =============================================================================
# Bulk generation
# =============================================================================

class BulkGenerator:
    """Iterate over (prompt × N videos), save to disk with skip-on-exist."""

    def __init__(
        self,
        pipeline,
        profile: GenerationProfile,
        save_root: Path,
        save_image_or_video,
    ) -> None:
        self.pipe = pipeline
        self.profile = profile
        self.save_root = Path(save_root)
        self.save_root.mkdir(parents=True, exist_ok=True)
        self._save_fn = save_image_or_video

    def run(
        self,
        batch: list[dict],
        num_videos: int,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
    ) -> None:
        end_idx = end_idx if end_idx is not None else len(batch)
        work = batch[start_idx:end_idx]
        print(
            f"[generate] profile={self.profile.name}  "
            f"items={len(work)} (idx {start_idx}:{end_idx})  "
            f"num_videos/item={num_videos}"
        )
        self._write_run_meta(start_idx, end_idx, num_videos, n_items=len(batch))

        for k, item in enumerate(work, start=start_idx):
            sub = self.save_root / f"item_{k:04d}"
            print(f"\n[item {k}] prompt: {item['prompt'][:100]}…")
            self._run_one_item(item, sub, num_videos)

    # ─────────────────────────────────────────────────────────────────────────

    def _run_one_item(self, item: dict, save_dir: Path, n: int) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "_config.json", "w") as f:
            json.dump({**asdict(self.profile), **item}, f, indent=2, ensure_ascii=False)

        rng = random.Random(self.profile.base_seed)

        for i in range(n):
            seed = self._next_seed(i, rng)
            guidance = self._next_guidance(rng)

            out_path = save_dir / f"{self.profile.name}_{i:03d}_seed{seed}_g{guidance:.2f}.mp4"
            if out_path.exists():
                print(f"[skip] {out_path.name}")
                continue

            t0 = time.time()
            video = _call_pipeline(
                self.pipe,
                prompt=item["prompt"],
                negative_prompt=self.profile.negative_prompt,
                image_path=item["input_video"],
                num_conditional_frames=self.profile.num_conditional_frames,
                guidance=guidance,
                seed=seed,
                aspect_ratio=self.profile.aspect_ratio,
            )
            self._save_fn(video, str(out_path), fps=self.profile.fps)
            dt = time.time() - t0
            print(
                f"[{i + 1}/{n}] seed={seed} g={guidance:.2f} -> "
                f"{out_path.name} ({dt:.1f}s)"
            )

    def _next_seed(self, i: int, rng: random.Random) -> int:
        if self.profile.seed_strategy == "random":
            return rng.randint(0, 2**31 - 1)
        return self.profile.base_seed + i

    def _next_guidance(self, rng: random.Random) -> float:
        if self.profile.guidance_jitter > 0:
            return max(
                1.0,
                self.profile.guidance
                + rng.uniform(-self.profile.guidance_jitter, self.profile.guidance_jitter),
            )
        return self.profile.guidance

    def _write_run_meta(
        self, start_idx: int, end_idx: int, num_videos: int, n_items: int
    ) -> None:
        meta = {
            "profile": asdict(self.profile),
            "start_idx": start_idx,
            "end_idx": end_idx,
            "num_videos_per_item": num_videos,
            "n_items_total_in_batch": n_items,
            "host": socket.gethostname(),
            "git_sha": _git_sha(),
            "cosmos_git_sha": _cosmos_git_sha(),
            "gpu": _gpu_name(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(self.save_root / "_run_meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)


def _git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def _cosmos_git_sha() -> Optional[str]:
    cosmos = Path(__file__).parent / "cosmos-predict2"
    if not cosmos.exists():
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(cosmos), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def _gpu_name() -> Optional[str]:
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return None


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=GenerationProfile.available(), required=True)
    ap.add_argument("--num_videos", type=int, default=50,
                    help="How many videos per prompt (default: 50)")
    ap.add_argument("--batch_json", type=str, required=True)
    ap.add_argument("--save_dir", type=str, required=True)
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--end_idx", type=int, default=None)
    # Profile overrides
    ap.add_argument("--model_size", choices=["2B", "14B"], default=None)
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--base_seed", type=int, default=0)
    ap.add_argument(
        "--use_base_checkpoint", action="store_true",
        help="Force base Cosmos-Predict2 (no GR00T fine-tune).",
    )
    ap.add_argument(
        "--ckpt_root", type=str, default=None,
        help="Override checkpoint root (defaults to $COSMOS_CKPT_ROOT or ./checkpoints).",
    )
    args = ap.parse_args()

    profile = GenerationProfile.from_name(args.profile)

    # Apply CLI overrides via dataclasses.replace
    from dataclasses import replace
    overrides: dict = {}
    if args.model_size is not None:
        overrides["model_size"] = args.model_size
    if args.model_size == "2B":
        profile = fallback_to_2b(profile)
    if args.guidance is not None:
        overrides["guidance"] = args.guidance
    if args.use_base_checkpoint:
        overrides["gr00t_variant"] = None
    overrides["base_seed"] = args.base_seed
    profile = replace(profile, **overrides)

    # Load batch
    with open(args.batch_json) as f:
        batch = json.load(f)
    end = args.end_idx if args.end_idx is not None else len(batch)
    print(f"[generate] processing {end - args.start_idx} prompts")

    # Build pipeline once
    cosmos = _import_cosmos()
    factory = PipelineFactory(ckpt_root=Path(args.ckpt_root) if args.ckpt_root else None)
    pipeline = factory.get(profile)

    bulk = BulkGenerator(
        pipeline=pipeline,
        profile=profile,
        save_root=Path(args.save_dir),
        save_image_or_video=cosmos["save_image_or_video"],
    )
    bulk.run(
        batch=batch,
        num_videos=args.num_videos,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )


if __name__ == "__main__":
    main()
