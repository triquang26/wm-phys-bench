"""Generate high-profile videos with cosmos-predict2 for the query/high test set.

Usage:
    cosmos-predict2/.venv/bin/python generate.py \
        --prompts prompts.json \
        --save_dir ../data/cosmos_synthetic_data/query/high \
        --input_dir ../data/cosmos_inputs \
        --seed_offset 1000

Each prompt produces exactly 1 MP4 named "<task>.mp4". Skips if dest already
exists (resume-safe). Writes _run_meta.json with seeds + git SHAs + ckpt revision.

cosmos-predict2 Video2World is *image-conditioned* -- every generation needs an
input conditioning image (or short video). We look up the input file under
`--input_dir` using a fuzzy match on the task name (slugified). If no match is
found we fall back to the single GR1 sample bundled in the repo
(`cosmos-predict2/assets/sample_gr00t_dreams_gr1/*.png`). For the smoke test
this fallback is the canonical path -- it's the only input image that ships
with the repo.

Resolved cosmos-predict2 API (verified against the canonical
`cosmos-predict2/examples/video2world.py`):

    from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
    from cosmos_predict2.configs.base.config_video2world import (
        get_cosmos_predict2_video2world_pipeline,
    )
    from imaginaire.constants import get_cosmos_predict2_video2world_checkpoint
    from imaginaire.utils.io import save_image_or_video

    config = get_cosmos_predict2_video2world_pipeline(
        model_size="2B", resolution="480", fps=16,
    )
    config.guardrail_config.enabled = False           # only if --disable_guardrail
    config.prompt_refiner_config.enabled = False      # only if --disable_prompt_refiner
    dit_path = get_cosmos_predict2_video2world_checkpoint(
        model_size="2B", resolution="480", fps=16, aspect_ratio="16:9",
    )
    pipe = Video2WorldPipeline.from_config(
        config=config, dit_path=dit_path,
        device="cuda", torch_dtype=torch.bfloat16,
        load_prompt_refiner=True,
    )
    video, prompt_used = pipe(
        prompt=...,
        negative_prompt=...,
        aspect_ratio="16:9",
        input_path=<jpg/png/mp4>,
        num_conditional_frames=1,
        guidance=7.0,
        seed=...,
        return_prompt=True,
    )
    save_image_or_video(video, out_path, fps=fps)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from profiles import HIGH, GenerationProfile, from_name


SCRIPT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Cosmos imports (heavy; not available before setup.sh)
# ---------------------------------------------------------------------------
def _import_cosmos():
    """Return the public symbols we use. Raises on missing install."""
    import torch  # noqa: F401  (consumed lazily by callers)

    from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
    from cosmos_predict2.configs.base.config_video2world import (
        get_cosmos_predict2_video2world_pipeline,
    )
    from imaginaire.constants import (
        get_cosmos_predict2_video2world_checkpoint,
        get_cosmos_predict2_gr00t_checkpoint,
    )
    from imaginaire.utils.io import save_image_or_video

    return {
        "Video2WorldPipeline": Video2WorldPipeline,
        "get_pipeline_config": get_cosmos_predict2_video2world_pipeline,
        "get_dit_path": get_cosmos_predict2_video2world_checkpoint,
        "get_gr00t_dit_path": get_cosmos_predict2_gr00t_checkpoint,
        "save_image_or_video": save_image_or_video,
    }


# ---------------------------------------------------------------------------
# Input image resolution
# ---------------------------------------------------------------------------
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(s: str) -> str:
    return _SLUG_RE.sub("_", s.lower()).strip("_")


def resolve_input_path(task: str, input_dirs: list[Path]) -> Optional[Path]:
    """Find the conditioning frame for `task` under any of input_dirs.

    Strategy:
      1. <input_dir>/<task>.{png,jpg,jpeg,mp4} (exact)
      2. <input_dir>/<task>.png after replacing slashes/spaces
      3. fuzzy slug match -- pick the first file whose slug starts with the
         task's slug prefix (the task names are prefixed e.g. "8_..." which is
         the GR1 episode-id ordering).
    """
    task_slug = _slug(task)
    exts = (".png", ".jpg", ".jpeg", ".mp4")

    for d in input_dirs:
        if not d.exists():
            continue
        # 1 + 2: direct names
        for ext in exts:
            for cand in (d / f"{task}{ext}", d / f"{task.replace('/', '_')}{ext}"):
                if cand.exists():
                    return cand
        # 3: fuzzy slug match
        candidates = []
        for f in d.iterdir():
            if not f.is_file() or f.suffix.lower() not in exts:
                continue
            cs = _slug(f.stem)
            if cs == task_slug or cs.startswith(task_slug[:20]) or task_slug.startswith(cs[:20]):
                candidates.append(f)
        if candidates:
            # prefer images over videos for num_conditional_frames=1
            candidates.sort(key=lambda p: (p.suffix.lower() == ".mp4", len(p.name)))
            return candidates[0]
    return None


# ---------------------------------------------------------------------------
# PipelineFactory
# ---------------------------------------------------------------------------
class PipelineFactory:
    """Build + cache a Video2WorldPipeline for a given GenerationProfile."""

    def __init__(self, ckpt_root: Path):
        # cosmos-predict2 reads CHECKPOINTS_DIR from its own argparse, default "checkpoints".
        # We export it via env so a non-cwd run still finds the right dir.
        self.ckpt_root = Path(ckpt_root).resolve()
        os.environ["COSMOS_PREDICT2_ARGS"] = f"--checkpoints {self.ckpt_root}"
        self._pipe = None
        self._cosmos = None
        self._fps_for_save = None

    def get(self, profile: GenerationProfile):
        if self._pipe is None:
            self._pipe = self._build(profile)
        return self._pipe

    @property
    def cosmos(self) -> dict:
        if self._cosmos is None:
            self._cosmos = _import_cosmos()
        return self._cosmos

    @property
    def fps_for_save(self) -> int:
        return self._fps_for_save or 16

    def _build(self, profile: GenerationProfile):
        import torch

        cosmos = self.cosmos
        # 1. build the LazyConfig
        resolution = str(profile.resolution)  # "480"
        fps = int(profile.fps)                # 16
        config = cosmos["get_pipeline_config"](
            model_size=profile.model_size,
            resolution=resolution,
            fps=fps,
        )
        # 2. flip guardrail / refiner if requested
        if profile.disable_guardrail and hasattr(config, "guardrail_config"):
            config.guardrail_config.enabled = False
            print("[generate] guardrail disabled")
        if profile.disable_prompt_refiner and hasattr(config, "prompt_refiner_config"):
            config.prompt_refiner_config.enabled = False
            print("[generate] prompt refiner disabled")

        # 3. resolve the .pt path (GR00T fine-tune or base model)
        if profile.gr00t_variant:
            dit_path = cosmos["get_gr00t_dit_path"](
                gr00t_variant=profile.gr00t_variant,
                model_size=profile.model_size,
                resolution=resolution,
                fps=fps,
                aspect_ratio=profile.aspect_ratio,
            )
            print(f"[generate] GR00T variant={profile.gr00t_variant}, dit={Path(dit_path).name}")
        else:
            dit_path = cosmos["get_dit_path"](
                model_size=profile.model_size,
                resolution=resolution,
                fps=fps,
                aspect_ratio=profile.aspect_ratio,
            )
        if not Path(dit_path).exists():
            raise FileNotFoundError(
                f"DiT checkpoint not found: {dit_path}. Did setup.sh download "
                f"nvidia/Cosmos-Predict2-{profile.model_size}-Video2World into "
                f"{self.ckpt_root}/nvidia/?"
            )

        # 4. instantiate via the documented entry point
        Pipeline = cosmos["Video2WorldPipeline"]
        pipe = Pipeline.from_config(
            config=config,
            dit_path=str(dit_path),
            use_text_encoder=True,
            offload_text_encoder=False,
            downcast_text_encoder=False,
            device="cuda",
            torch_dtype=torch.bfloat16,
            load_ema_to_reg=False,
            load_prompt_refiner=not profile.disable_prompt_refiner,
        )

        # Track fps for save_image_or_video (10 if state_t==16, else 16 per
        # the canonical example).
        try:
            self._fps_for_save = 10 if pipe.config.state_t == 16 else 16
        except AttributeError:
            self._fps_for_save = fps

        return pipe


# ---------------------------------------------------------------------------
# BulkGenerator
# ---------------------------------------------------------------------------
class BulkGenerator:
    """Iterate over prompts (1 video each), save MP4s with skip-on-exist."""

    def __init__(
        self,
        factory: PipelineFactory,
        profile: GenerationProfile,
        save_dir: Path,
        input_dirs: list[Path],
        fallback_input: Optional[Path],
    ):
        self.factory = factory
        self.profile = profile
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.input_dirs = input_dirs
        self.fallback_input = fallback_input

    def run(self, prompts: list[dict], seed_offset: int = 0) -> dict:
        pipe = self.factory.get(self.profile)
        meta = {
            "profile": asdict(self.profile),
            "seed_offset": seed_offset,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "host": socket.gethostname(),
            "git_sha": _git_sha(SCRIPT_DIR.parent),
            "cosmos_sha": _git_sha(SCRIPT_DIR / "cosmos-predict2"),
            "ckpt_revision": _read_text(SCRIPT_DIR / "_ckpt_revision.txt"),
            "items": [],
        }

        for i, item in enumerate(prompts):
            task = item["task"]
            out = self.save_dir / f"{task}.mp4"
            if out.exists():
                print(f"[gen] skip (exists): {task}")
                meta["items"].append({"task": task, "seed": None, "status": "skipped"})
                continue

            seed = seed_offset + i
            input_path = resolve_input_path(task, self.input_dirs) or self.fallback_input
            if input_path is None:
                msg = f"no input image found (input_dirs={self.input_dirs}, no fallback)"
                print(f"[gen] FAIL {task}: {msg}")
                meta["items"].append({"task": task, "seed": seed, "status": f"fail: {msg}"})
                continue

            t0 = time.time()
            try:
                self._gen_one(pipe, item["prompt"], str(input_path), out, seed)
                dt = time.time() - t0
                print(f"[gen] {task} -> {out.name} (input={input_path.name}, seed={seed}, {dt:.1f}s)")
                meta["items"].append({
                    "task": task, "seed": seed, "status": "ok",
                    "duration_s": round(dt, 1),
                    "input_path": str(input_path),
                })
            except Exception as e:
                print(f"[gen] FAIL {task}: {e}")
                meta["items"].append({"task": task, "seed": seed, "status": f"fail: {e}"})

        meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        (self.save_dir / "_run_meta.json").write_text(
            json.dumps(meta, indent=2, default=str)
        )
        return meta

    # ---- internals --------------------------------------------------------

    def _gen_one(self, pipe, prompt: str, input_path: str, out_path: Path, seed: int) -> None:
        full_prompt = self.profile.prompt_prefix + prompt
        result = pipe(
            prompt=full_prompt,
            negative_prompt=self.profile.negative_prompt,
            aspect_ratio=self.profile.aspect_ratio,
            input_path=input_path,
            num_conditional_frames=self.profile.num_conditional_frames,
            guidance=self.profile.guidance,
            seed=seed,
            use_cuda_graphs=False,
            return_prompt=True,
        )
        if isinstance(result, tuple):
            if len(result) != 2:
                raise TypeError(f"Expected (video, prompt) tuple from pipe(), got length {len(result)}")
            video, _prompt_used = result
        else:
            video = result
        if video is None:
            raise RuntimeError("pipe() returned video=None — guardrail may have rejected the prompt")
        # Use the cosmos-bundled saver (handles tensor -> mp4 + pyav)
        save_fn = self.factory.cosmos["save_image_or_video"]
        save_fn(video, str(out_path), fps=self.factory.fps_for_save)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _git_sha(repo_dir: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def _read_text(p: Path) -> Optional[str]:
    try:
        return p.read_text().strip() if p.exists() else None
    except Exception:
        return None


def _default_fallback_input() -> Optional[Path]:
    """The single GR1 sample image bundled in the cosmos-predict2 repo."""
    d = SCRIPT_DIR / "cosmos-predict2" / "assets" / "sample_gr00t_dreams_gr1"
    if d.exists():
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                return f
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Bulk-generate query/high videos with cosmos-predict2.")
    ap.add_argument("--prompts", default="prompts.json", type=Path,
                    help="JSON file: [{task, prompt}, ...] (default: prompts.json)")
    ap.add_argument("--save_dir", required=True, type=Path,
                    help="Output dir for <task>.mp4 files.")
    ap.add_argument("--input_dir", action="append", type=Path, default=None,
                    help="Dir of conditioning images keyed by task name. Repeat for multiple.")
    ap.add_argument("--ckpt_root", default=SCRIPT_DIR / "checkpoints", type=Path,
                    help="Dir holding nvidia/Cosmos-Predict2-* and google-t5/t5-11b.")
    ap.add_argument("--seed_offset", type=int, default=1000,
                    help="seed = seed_offset + prompt_index (default: 1000)")
    ap.add_argument("--profile", default="high", choices=["high", "gr00t"],
                    help="Generation profile: 'high' (base 14B) or 'gr00t' (GR1 fine-tune)")
    args = ap.parse_args()

    prompts = json.loads(args.prompts.read_text())
    if not isinstance(prompts, list) or not all("task" in p and "prompt" in p for p in prompts):
        raise ValueError(f"{args.prompts}: expected list of dicts with 'task' and 'prompt'")
    print(f"[generate] {len(prompts)} prompts; seed_offset={args.seed_offset}; profile={args.profile}")

    input_dirs = list(args.input_dir or [])
    fallback = _default_fallback_input()
    if fallback is not None:
        print(f"[generate] fallback conditioning frame: {fallback}")

    profile = from_name(args.profile)
    factory = PipelineFactory(args.ckpt_root)
    BulkGenerator(factory, profile, args.save_dir, input_dirs, fallback).run(
        prompts, seed_offset=args.seed_offset
    )


if __name__ == "__main__":
    main()
