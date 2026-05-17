# ENV.md — cosmos-predict2 / dreamgen_data setup record

Canonical, copy-pasteable record of the cosmos-predict2 install attempt for the
`dreamgen_data/` pipeline. **Status: blocked** by two host-side issues that need
admin / HF account action; everything below is reproducible from a clean state.

## TL;DR — current state on this host

| Stage | Status |
|---|---|
| Pre-flight: hardware / OS / disk / network probe | done |
| Clone cosmos-predict2 + record SHA | done |
| `uv venv` with Python 3.11 (seeded with pip) | done |
| Install wrapper deps from `requirements.txt` | done |
| Install torch 2.6.0+cu126 + torchvision (pytorch.org index) | done |
| Install cosmos-predict2 1.0.9 from PyPI (no `[cu126]` extras) | done |
| Editable install of cosmos-predict2 clone | done |
| **Install flash-attn / transformer-engine / apex / natten** | **BLOCKED** (Issue 1) |
| **Download `nvidia/Cosmos-Predict2-2B-Video2World` ckpt** | **BLOCKED** (Issue 2) |
| Sanity import `Video2WorldPipeline` | fails on `import transformer_engine as te` |
| Smoke test (1 prompt → 1 MP4) | **NOT RUN** (depends on the above) |

Disk used in `dreamgen_data/`: **6.6 GB** (clone + venv with torch+cu126 deps,
no checkpoints).

---

## Known issues (must be resolved before Agent E can run full generation)

### Issue 1 — `nvidia-cosmos.github.io` is firewalled on this host

cosmos-predict2's `[cu126]` extra pulls `torch`, `torchvision`, `flash-attn`,
`transformer-engine==1.13`, `apex==0.1.0`, and `natten==0.21.0` from a custom
NVIDIA wheel index served via GitHub Pages:

    https://nvidia-cosmos.github.io/cosmos-dependencies/cu126_torch260/simple

That host resolves to GitHub Pages' IPv4 range `185.199.108–111.153`. From
this worker node:

```
$ timeout 5 bash -c 'cat </dev/tcp/185.199.108.153/443'; echo $?
124            # SYN timeout — IP range is blocked
$ curl -sI -m 10 -4 https://nvidia-cosmos.github.io/   # IPv4-forced
                # empty (network unreachable)
$ curl -sI -m 10 https://huggingface.co/                # control
HTTP/2 200      # HF is fine
```

Workarounds attempted, in order:
- IPv4-forced curl: same timeout.
- IPv6 (default ordering): the DNS records 2606:50c0:8001::153 etc. also fail
  ("Errno 101 Network is unreachable" inside the venv's Python).
- HOSTALIASES file: ignored by libc for non-root in this environment.
- Cannot edit `/etc/hosts` (no sudo).

The four blocked packages have **no prebuilt manylinux wheels on PyPI**:
- `transformer-engine==1.13.0` → only `transformer_engine-1.13.0-py3-none-any.whl`
  (a stub that depends on `transformer-engine-cu12==1.13.0` (PyPI: wheel ✅)
  **AND** `transformer-engine-torch==1.13.0` (PyPI: sdist only ❌, needs nvcc 12.6
  to build))
- `flash-attn==2.6.3` → sdist on PyPI; needs nvcc 12.6 + matching torch ABI;
  prebuilt wheels on Dao-AILab GitHub releases stop at torch 2.4.
- `apex==0.1.0` → not the public PyPI `apex`; this is NVIDIA Apex; lives on the
  same blocked index.
- `natten==0.21.0` → sdist on PyPI; needs nvcc 12.6 to build.

System CUDA on this host is **12.2** (driver 535.247.01 → max 12.4 runtime),
so we cannot build any of these from source either.

**Fix**: ask infra/cluster admin to whitelist outbound 443 to GitHub Pages
(`185.199.108.0/22`) **or** install the CUDA 12.6 toolkit system-wide. Once
either is fixed, re-run `bash setup.sh`; the script auto-detects reachability
and uses the primary `[cu126]` install path.

### Issue 2 — `nvidia/Cosmos-Predict2-2B-Video2World` HF gated repo, not yet authorized

After login (HF_TOKEN OK), checkpoint download fails:

```
huggingface_hub.errors.GatedRepoError: 403 Client Error.
Cannot access gated repo for url
https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World/...
Access to model nvidia/Cosmos-Predict2-2B-Video2World is restricted and
you are not in the authorized list.
```

**Fix**: open https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World
in a browser logged in as the same HF account that owns `$HF_TOKEN`, click
"Request access". Approval is typically minutes–hours. While you're there,
also accept:
- https://huggingface.co/meta-llama/Llama-Guard-3-8B (gated, only needed if
  you keep the guardrail enabled — we disable it in `profiles.py`)
- https://huggingface.co/nvidia/Cosmos-Reason1-7B (gated; only needed if you
  keep the prompt refiner — `profiles.py:disable_prompt_refiner=False` so this
  IS required by default; flip it to `True` to skip)

---

## Hardware

```
GPU:    1× NVIDIA H100 80GB HBM3   (driver 535.247.01)
VRAM:   81 559 MiB
CPU RAM: 2.0 TiB total, ~1.4 TiB available at baseline
Disk:   /mnt/data — NFS, 95 TB total, 20 TB free after partial install
```

## OS / kernel

```
Linux worker-0 5.15.0-130-generic #140-Ubuntu SMP Wed Dec 18 17:59:53 UTC 2024
Ubuntu (Antigravity remote-execution host)
glibc / NSS: standard; `nsswitch.conf` "hosts: files dns"
No sudo, no /etc/hosts write, no $HOSTALIASES override.
```

## CUDA toolkit + driver

```
Driver:       535.247.01 (max CUDA 12.4 runtime)
System toolkits in /usr/local/:
  cuda -> cuda-12.2
  cuda-12 -> cuda-12.2
  cuda-12.2  (nvcc: release 12.2.140, built Aug 15 2023)
Driver supports the cu126 prebuilt torch wheels (they ship their own runtime),
so torch 2.6.0+cu126 imports fine. But building anything from source against
torch's cu126 ABI would need a cu126 toolkit, which we don't have.
```

## Python — system

```
$ python --version
Python 3.13.12
$ which python
/mnt/data/sftp/data/quangpt3/miniconda3/bin/python
```

This is Miniconda base; cosmos-predict2 doesn't use it. We make a separate
uv-managed venv on Python 3.11.

## Python — cosmos-predict2 venv

```
$ ./cosmos-predict2/.venv/bin/python --version
Python 3.11.15
```

uv-managed venv at `dreamgen_data/cosmos-predict2/.venv/` (Python 3.11.15,
installed by uv from astral's prebuilt CPython). Seeded with `--seed` so
`pip`/`setuptools`/`wheel` are present (uv's modern default is pip-less).

## `uv`

- Already present at `/home/quangpt3/.local/bin/uv`, version **0.11.4**.
- If missing on a fresh host:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source "$HOME/.local/bin/env"
  ```

## `huggingface-cli`

- The user-local one at `/mnt/data/sftp/data/quangpt3/.local/bin/huggingface-cli`
  is **broken** (its 1.15-style deprecated shim wants `typer`, which is missing).
  Do not use it.
- `setup.sh` installs a fresh one inside the cosmos-predict2 venv via
  `pip install -r requirements.txt` (`huggingface_hub==0.26.1`). It works:
  ```
  $ cosmos-predict2/.venv/bin/huggingface-cli --version
  huggingface-cli 0.26.1
  ```

## cosmos-predict2

| Item | Value |
|---|---|
| Repo | https://github.com/nvidia-cosmos/cosmos-predict2 |
| Clone | `dreamgen_data/cosmos-predict2/` (depth=1) |
| Pinned SHA | **`661da4774b0ca41d082a0ecbeb47550bcf07e03f`** (in `_cosmos_sha.txt`) |
| Package version | 1.0.9 (from PyPI sdist) |
| Install method | `pip install cosmos-predict2==1.0.9 --extra-index-url https://pypi.org/simple` + editable install of the clone for `imaginaire.utils.io` |

## torch / CUDA stack

| Package | Version | Source |
|---|---|---|
| torch | **2.6.0+cu126** | https://download.pytorch.org/whl/cu126 |
| torchvision | **0.21.0+cu126** | same |
| triton | 3.2.0 | PyPI |
| nvidia-cublas-cu12 | 12.6.4.1 | PyPI (torch deps) |
| nvidia-cudnn-cu12 | 9.5.1.17 | PyPI |
| nvidia-cusolver-cu12 | 11.7.1.2 | PyPI |
| (full set of nvidia-*-cu12 12.6.* packages installed by torch) | | |

## flash-attn / transformer-engine / apex / natten

**Not installed** — see Issue 1. cosmos-predict2 source hard-imports
`transformer_engine as te` in `cosmos_predict2/models/text2image_dit.py:25`,
which is reached transitively from `Video2WorldPipeline`, so this is a
**mandatory** dep for inference, not optional.

## HuggingFace checkpoints

Layout cosmos-predict2 expects (root = `dreamgen_data/checkpoints/`,
configurable via `COSMOS_PREDICT2_ARGS="--checkpoints <dir>"`):

```
checkpoints/
├── nvidia/
│   ├── Cosmos-Predict2-2B-Video2World/     # GATED — request access (Issue 2)
│   │   └── model-480p-16fps.pt             # ~5 GB DiT (the 2B model)
│   │   └── tokenizer/tokenizer.pth
│   ├── Cosmos-Guardrail1/                  # ~5 GB, only if guardrail enabled
│   └── Cosmos-Reason1-7B/                  # ~15 GB, only if prompt_refiner enabled (GATED)
├── google-t5/
│   └── t5-11b/                             # ~45 GB text encoder
└── meta-llama/                             # only if guardrail enabled (GATED)
    └── Llama-Guard-3-8B/
```

Pinned revisions (from cosmos-predict2's `scripts/download_checkpoints.py`):

| repo_id | rev |
|---|---|
| `nvidia/Cosmos-Predict2-2B-Video2World` | `f50c09f5d8ab133a90cac3f4886a6471e9ba3f18` |
| `google-t5/t5-11b` | `90f37703b3334dfe9d2b009bfcbfbf1ac9d28ea3` |
| `nvidia/Cosmos-Guardrail1` | `d6d4bfa899a71454a700907664f3e88f503950cf` |
| `nvidia/Cosmos-Reason1-7B` | `8fe96c1fa10db9e666b6fa6a87fea57dd9635649` |

The download is via `huggingface_hub.snapshot_download(repo_id=..., revision=...,
local_dir="checkpoints/<repo_id>", max_workers=4)`.

## Resolved cosmos-predict2 API (verified against the upstream source)

| Item | Value |
|---|---|
| Pipeline import | `from cosmos_predict2.pipelines.video2world import Video2WorldPipeline` |
| Config builder | `from cosmos_predict2.configs.base.config_video2world import get_cosmos_predict2_video2world_pipeline` |
| dit_path resolver | `from imaginaire.constants import get_cosmos_predict2_video2world_checkpoint` |
| Video saver | `from imaginaire.utils.io import save_image_or_video` |
| Constructor | `Video2WorldPipeline.from_config(config=..., dit_path=..., device="cuda", torch_dtype=torch.bfloat16, load_prompt_refiner=...)` — there is **no** `from_pretrained` |
| Inference call | `pipe(prompt=..., negative_prompt=..., aspect_ratio=..., input_path=<jpg/png/mp4>, num_conditional_frames=1, guidance=..., seed=..., return_prompt=True)` returns `(video, prompt_used)` |
| Required arg | `input_path` — Video2World is **image-conditioned**; no text-only mode |
| Output save | `save_image_or_video(video, out_path, fps=fps_for_save)` where `fps_for_save = 10 if pipe.config.state_t == 16 else 16` |

`generate.py` has been rewritten to use this exact API (no more 3-fallback
guessing). `setup.sh`'s sanity-check uses the verified imports too.

The bundled sample frame for the smoke test is at
`cosmos-predict2/assets/sample_gr00t_dreams_gr1/8_Use_the_right_hand_to_pick_up_rubik's_cube_..._wooden_shelf..png`
(the ONLY GR1 sample image shipped with the repo).

## Network

| Host | Reachable | Notes |
|---|---|---|
| huggingface.co | ✅ HTTP/2 200 | required for ckpt download |
| pypi.org | ✅ | base index |
| download.pytorch.org | ✅ | torch+cu126 fallback |
| pythonhosted.org (PyPI CDN) | ✅ | |
| github.com (clone) | ✅ | |
| **nvidia-cosmos.github.io** | ❌ | **Issue 1** — IPv4 range 185.199.108–111.153 blocked |
| github releases (release files) | ⚠ unchecked | flash-attn prebuilt wheels live here (different CDN); only relevant under Issue 1 |

## Disk usage

| Stage | `dreamgen_data/` |
|---|---|
| Baseline (scripts only) | 42 K |
| After partial install (clone + venv + torch+cu126, **no ckpts**) | 6.6 G |
| Projected full install | ~120 G (clone + venv + 2B + T5 + guardrail + reason1) |

## Wall-clock time

| Step | Time |
|---|---|
| Clone cosmos-predict2 | ~30 s |
| Create uv venv (3.11, seeded) | ~5 s |
| Install wrapper deps (requirements.txt) | ~45 s |
| Install torch+cu126 + cosmos-predict2 + deps | ~14 min |
| (failed) HF login + checkpoint snapshot_download | ~5 s before 403 |
| **Total partial install** | **~15 min** |

## Reproduction (clean state, AFTER both blockers are resolved)

```bash
cd /mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination

# 1. HF token (gitignored)
test -f .env.dreamgen || { echo "create .env.dreamgen with HF_TOKEN=hf_xxx (mode 600)"; exit 1; }
set -a && source .env.dreamgen && set +a

# 2. One-shot install
cd dreamgen_data
bash setup.sh 2>&1 | tee setup.log

# 3. Smoke (1 prompt, ~3-5 min on H100)
make smoke 2>&1 | tee smoke.log

# 4. Full generation (Agent E)
make gen-query-high
```

## Files modified by Agent D vs. Agent C's drop

- `setup.sh` — rewritten:
  - `uv venv` now uses `--seed` (modern uv venvs are pip-less by default).
  - The custom `[cu126]` index is **probed for reachability first**; if it's
    unreachable, the script falls back to pytorch.org + base PyPI (current
    case on this host).
  - HF checkpoint download switched from `huggingface-cli download` to a
    pinned `snapshot_download(repo_id, revision, local_dir)` in Python so we
    get reproducible revisions and a proper `{org}/{repo}/` layout under
    `checkpoints/`, matching cosmos-predict2's `CHECKPOINTS_DIR` convention.
  - Sanity import upgraded to also check `get_cosmos_predict2_video2world_pipeline`
    and `imaginaire.utils.io.save_image_or_video`.
- `generate.py` — fully rewritten to the verified API (`from_config`, not
  `from_pretrained`; tuple return; `input_path` arg; `save_image_or_video`).
  Added `--input_dir` flag + per-task fuzzy match and a fallback to the single
  bundled GR1 sample frame.
- `Makefile` — `smoke` now picks the prompt whose task name matches the bundled
  GR1 sample (`8_Use_the_right_hand_to_pick_up_rubik's_cube...`) so the smoke
  test exercises the image-conditioned path without any extra inputs.
- `README.md` — updated checkpoint layout, replaced "Agent D notes" with the
  verified API block, added image-conditioned-generation section.

## Next step for Agent E

1. Have the user (a) request HF access to
   `nvidia/Cosmos-Predict2-2B-Video2World` and (b) ask cluster admin to
   whitelist `185.199.108.0/22:443` (GitHub Pages CDN — needed by the cu126
   wheel index).
2. Re-run `bash setup.sh` — it picks up where it left off (clone, venv, base
   install are idempotent), installs the `[cu126]` extras, downloads
   checkpoints, and runs the sanity import.
3. Run `make smoke` to verify.
4. Then `make gen-query-high`.

If only blocker #2 (HF access) is fixed but #1 (network) remains, generation
will still fail at import-time on `transformer_engine`. There is no
inference-time fallback path that skips TE — it is a hard build dep of the
cosmos-predict2 DiT model code.
