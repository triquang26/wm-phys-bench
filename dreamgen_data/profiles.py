"""GenerationProfile -- single 'high' preset for cosmos-predict2 video2world.

Scoped down to what we actually use: a clean (query/high) test set generated
with the 2B Video2World checkpoint and seed_offset=1000. No 'low' / 'hallucinate'
profile here -- the original reference set already serves that role.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


_DEFAULT_NEGATIVE_PROMPT = ""  # cosmos-predict2 supplies its own internal default


@dataclass(frozen=True)
class GenerationProfile:
    """All knobs controlling a single cosmos-predict2 generation profile."""

    name: str                                                        # "high"
    model_size: Literal["2B", "14B"] = "14B"
    fps: int = 16
    resolution: int = 480
    num_conditional_frames: int = 1
    aspect_ratio: str = "16:9"
    guidance: float = 7.0
    negative_prompt: str = _DEFAULT_NEGATIVE_PROMPT
    disable_guardrail: bool = True
    disable_prompt_refiner: bool = True
    seed_strategy: Literal["deterministic", "random"] = "deterministic"
    base_seed: int = 2000      # seed_offset: distinct from any original gen
    guidance_jitter: float = 0.0


HIGH = GenerationProfile(name="high")


def from_name(name: str) -> GenerationProfile:
    if name != "high":
        raise ValueError(f"Only 'high' profile is supported, got '{name}'")
    return HIGH
