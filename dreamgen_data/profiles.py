"""GenerationProfile -- generation profiles for cosmos-predict2 video2world.

Supported profiles:
  "high"   -- base Cosmos-Predict2-14B-Video2World, seed_offset=2000
  "gr00t"  -- Cosmos-Predict2-14B-Sample-GR00T-Dreams-GR1 fine-tune,
               adds robot prompt prefix, same 480p/16fps config
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


_DEFAULT_NEGATIVE_PROMPT = ""  # cosmos-predict2 supplies its own internal default

# GR00T prepends this prefix to every prompt (as per the official example script)
_GR00T_PROMPT_PREFIX = "The robot arm is performing a task. "


@dataclass(frozen=True)
class GenerationProfile:
    """All knobs controlling a single cosmos-predict2 generation profile."""

    name: str
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
    base_seed: int = 2000
    guidance_jitter: float = 0.0
    # GR00T fine-tune: set to "gr1" to use Sample-GR00T-Dreams-GR1 checkpoint
    gr00t_variant: Literal["gr1", "droid", ""] = ""
    # Prefix prepended to every prompt (used by GR00T)
    prompt_prefix: str = ""


# Prompt refiner (Cosmos-Reason1-7B) improves short/vague prompts for better motion.
# Download: huggingface-cli download nvidia/Cosmos-Reason1-7B --local-dir checkpoints/nvidia/Cosmos-Reason1-7B
HIGH = GenerationProfile(
    name="high",
    disable_prompt_refiner=False,   # enable Cosmos-Reason1-7B prompt refiner
)

# GR00T fine-tune: prompt refiner must stay OFF — the GR00T checkpoint was not
# trained with the refiner pipeline; enabling it degrades output quality.
# Use detailed prompts directly instead.
GR00T = GenerationProfile(
    name="gr00t",
    gr00t_variant="gr1",
    prompt_prefix=_GR00T_PROMPT_PREFIX,
    base_seed=3000,            # distinct from base-14B seeds (2000+)
    disable_prompt_refiner=True,
)


def from_name(name: str) -> GenerationProfile:
    """Return a pre-defined GenerationProfile by name. Raises ValueError if unknown."""
    if name == "high":
        return HIGH
    if name == "gr00t":
        return GR00T
    raise ValueError(f"Unknown profile '{name}'. Supported: 'high', 'gr00t'")
