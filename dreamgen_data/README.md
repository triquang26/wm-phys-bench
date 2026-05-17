# dreamgen_data -- query/high test set generator

Minimal wrapper around [NVIDIA cosmos-predict2](https://github.com/nvidia-cosmos/cosmos-predict2) that
generates a held-out **`query/high`** clean test set: 23 high-profile videos with new seeds
(`seed_offset=1000`), then post-processes them to per-frame SAM3-segmented PNGs.

Scope is intentionally tight -- this directory does **not** generate the `low` set or any other
profile. The reference set already serves that role.

## Prereqs

- CUDA 12.6+, NVIDIA driver supporting Hopper or newer, **>= 50 GB free disk**
- Python 3.10 / 3.11
- `uv` on `PATH` (auto-installs if missing)
- `HF_TOKEN` env var with access to gated repos (e.g. Llama-Guard-3-8B):
  `export HF_TOKEN=hf_xxx`

## Pipeline

```text
prompts.json (23 entries)
  -> [generate.py via cosmos-predict2 2B Video2World, seed_offset=1000]
  -> data/cosmos_synthetic_data/query/high/<task>.mp4
  -> [postprocess.py: extract_uniform (50 frames) + SAM3 bg removal]
  -> data/query/high/<task>/frame_NNNN.png
```

## Usage

```bash
# 1. One-time setup (clones cosmos-predict2, creates uv venv, downloads 2B ckpt ~50 GB).
export HF_TOKEN=hf_xxx
make setup

# 2. Smoke test: 1 prompt -> 1 video. Verifies cosmos-predict2 API guesses are correct.
make smoke

# 3. Full generation + post-processing.
make gen-query-high
make postprocess
```

## Resume safety

Every step skips work whose output already exists:

- `setup.sh` short-circuits per-step (clone / venv / ckpt).
- `generate.py` skips `<task>.mp4` if it already exists.
- `postprocess.py` skips a frame PNG if the bg-removed output already exists.

So a crashed run can be resumed with the same `make` command -- no flags needed.

## Output layout

| Path | Content |
|---|---|
| `cosmos-predict2/` | upstream clone (read-only) |
| `cosmos-predict2/.venv/` | uv venv with cosmos-predict2 + our wrapper deps |
| `checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/` | downloaded 2B ckpt (~50 GB) |
| `checkpoints/google-t5/t5-11b/` | text encoder (~45 GB, T5-XXL) |
| `checkpoints/nvidia/Cosmos-Guardrail1/` | guardrail (~5 GB, only used if guardrail enabled) |
| `../data/cosmos_synthetic_data/query/high/*.mp4` | 23 generated videos |
| `../data/cosmos_synthetic_data/query/high/_run_meta.json` | seeds + git SHAs + ckpt revision |
| `../data/cosmos_frames_raw/query/high/<task>/*.png` | raw extracted frames |
| `../data/query/high/<task>/*.png` | final SAM3-bg-removed PNGs |

## Image-conditioned generation

cosmos-predict2 Video2World is **image-conditioned** -- every generation needs a
conditioning frame (first frame of the predicted video). `generate.py` resolves
the input per-prompt via `--input_dir`:

```bash
make gen-query-high  # uses default input_dir search
$(VENV_PY) generate.py \
    --prompts prompts.json \
    --save_dir ../data/cosmos_synthetic_data/query/high \
    --input_dir /path/to/gr1/first_frames   # repeat to add fallbacks
```

If no per-task input is found, the script falls back to the single GR1 sample
image bundled at `cosmos-predict2/assets/sample_gr00t_dreams_gr1/*.png`. The
smoke test uses this fallback intentionally.

To get per-task inputs for the full 23-prompt set, download the GR1-100
benchmark first frames:

```bash
huggingface-cli download nvidia/GR1-100 --repo-type dataset \
    --local-dir /tmp/gr1_100 --include "*/frame_0000.png"
# then point --input_dir at the slugified-task-named tree.
```

## Reproducibility

Each generation run leaves three tracking files:

- `_cosmos_sha.txt`   -- git SHA of the cosmos-predict2 clone used.
- `_ckpt_revision.txt` -- HuggingFace model revision SHA of the 2B ckpt.
- `<save_dir>/_run_meta.json` -- per-prompt seeds, timing, host, repo SHA.

## Files

| File | Role |
|---|---|
| `setup.sh`        | Clone cosmos-predict2, create uv venv, install, download 2B ckpt. Logs to `setup.log`. |
| `profiles.py`     | `GenerationProfile` dataclass + `HIGH` preset (single profile only). |
| `prompts.json`    | 23 `{task, prompt}` entries extracted from `data/cosmos_synthetic_data/reference/*.txt`. |
| `generate.py`     | `PipelineFactory` + `BulkGenerator`. Skip-on-exist. |
| `postprocess.py`  | `QueryHighHarvester`. Reuses `scripts/extract_frames.py` and `../sam3.py`. |
| `Makefile`        | `setup`, `smoke`, `gen-query-high`, `postprocess`. |
| `requirements.txt`| Pinned wrapper deps (cosmos-predict2 brings its own torch). |

## Verified cosmos-predict2 API

After install, the resolved API used by `generate.py` is (no more guesses;
see comments in `generate.py` and `ENV.md`):

```python
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from cosmos_predict2.configs.base.config_video2world import (
    get_cosmos_predict2_video2world_pipeline,
)
from imaginaire.constants import get_cosmos_predict2_video2world_checkpoint
from imaginaire.utils.io import save_image_or_video

config  = get_cosmos_predict2_video2world_pipeline(model_size="2B", resolution="480", fps=16)
dit_path = get_cosmos_predict2_video2world_checkpoint(
    model_size="2B", resolution="480", fps=16, aspect_ratio="16:9",
)
pipe = Video2WorldPipeline.from_config(
    config=config, dit_path=dit_path,
    device="cuda", torch_dtype=torch.bfloat16,
    load_prompt_refiner=True,
)
video, prompt_used = pipe(
    prompt=..., negative_prompt=..., aspect_ratio="16:9",
    input_path=<jpg/png/mp4>, num_conditional_frames=1,
    guidance=7.0, seed=..., return_prompt=True,
)
save_image_or_video(video, out_path, fps=fps)
```

`make smoke` is the recommended quick check after install. See `ENV.md` for the
exact toolchain versions used and any known-issue workarounds.
