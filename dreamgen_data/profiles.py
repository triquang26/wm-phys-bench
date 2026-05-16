"""GenerationProfile — config dataclass for cosmos-predict2 video2world generation.

Tách ra file riêng để override / extend dễ hơn (mỗi profile = một preset).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


_DEFAULT_NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no motion, "
    "motion blur, over-saturation, shaky footage, low resolution, grainy texture, "
    "pixelated images, poorly lit areas, underexposed and overexposed scenes, "
    "poor color balance, washed out colors, choppy sequences, jerky movements, "
    "low frame rate, artifacting, color banding, unnatural transitions, outdated "
    "special effects, fake elements, unconvincing visuals, poorly edited content, "
    "jump cuts, visual noise, and flickering. Overall, the video is of poor quality."
)


@dataclass(frozen=True)
class GenerationProfile:
    """All knobs controlling a single profile of cosmos-predict2 generation."""

    name: str
    # model
    model_size: Literal["2B", "14B"] = "14B"
    gr00t_variant: Optional[Literal["gr1", "droid"]] = "gr1"
    fps: int = 16
    resolution: int = 480
    # conditioning
    num_conditional_frames: int = 1
    aspect_ratio: str = "16:9"
    # sampling
    guidance: float = 7.0
    negative_prompt: str = _DEFAULT_NEGATIVE_PROMPT
    # safety / refiner
    disable_guardrail: bool = True
    disable_prompt_refiner: bool = False
    # seed strategy
    seed_strategy: Literal["deterministic", "random"] = "deterministic"
    base_seed: int = 0
    guidance_jitter: float = 0.0

    @classmethod
    def from_name(cls, name: str) -> "GenerationProfile":
        if name not in _PROFILES:
            raise KeyError(
                f"Unknown profile '{name}'. Available: {sorted(_PROFILES.keys())}"
            )
        return _PROFILES[name]

    @classmethod
    def available(cls) -> list[str]:
        return sorted(_PROFILES.keys())


_PROFILES: dict[str, GenerationProfile] = {
    # Best-quality config: post-trained GR1 ckpt, refiner on, deterministic seeds.
    "high": GenerationProfile(
        name="high",
        gr00t_variant="gr1",
        guidance=7.0,
        disable_prompt_refiner=False,
        seed_strategy="deterministic",
        guidance_jitter=0.0,
    ),
    # Hallucination-amplified: base ckpt (no robot anchor), low CFG, refiner off,
    # random seeds + guidance jitter. Same prompt as `high` → paired test set.
    "hallucinate": GenerationProfile(
        name="hallucinate",
        gr00t_variant=None,
        guidance=2.0,
        negative_prompt="",
        disable_prompt_refiner=True,
        seed_strategy="random",
        guidance_jitter=1.5,
    ),
}


def fallback_to_2b(profile: GenerationProfile) -> GenerationProfile:
    """Return a 2B-version of the given profile.

    2B does not ship with the GR00T-GR1 post-trained variant, so both profiles
    will use the base 2B checkpoint. Profile separation comes from CFG / refiner /
    seed strategy only.
    """
    from dataclasses import replace
    return replace(profile, model_size="2B", gr00t_variant=None)
