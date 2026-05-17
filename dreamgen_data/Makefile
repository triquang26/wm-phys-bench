# Makefile -- one-command workflow for dreamgen_data/.
#
#   make setup               # install cosmos-predict2 + download checkpoints (one-time)
#   make smoke               # 1 prompt -> 1 video (verify cosmos-predict2 API)
#   make gen-query-high      # 23 prompts -> 23 MP4s (base 14B, seed=2000)
#   make gen-query-gr00t     # 23 prompts -> 23 MP4s (GR1 fine-tune, seed=3000)
#   make postprocess         # MP4s -> 50 frames -> SAM3 bg removal -> data/query/high
#
# Override vars via CLI:
#   make gen-query-high SAVE_DIR=/tmp/foo PROMPTS=prompts.json SEED=2000
#   make postprocess GROOT_PY=/path/to/python

.PHONY: setup smoke gen-query-high gen-query-gr00t postprocess clean-smoke

# Robust SCRIPT_DIR -- works regardless of invocation directory.
SCRIPT_DIR := $(dir $(realpath $(firstword $(MAKEFILE_LIST))))
DATA_DIR   := $(SCRIPT_DIR)../data
VENV_PY    := $(SCRIPT_DIR)cosmos-predict2/.venv/bin/python

# postprocess uses SAM3 (transformers.Sam3Processor) -- lives in the groot/warp-score
# conda env, NOT the cosmos venv. Override GROOT_PY if your env is named differently.
GROOT_PY   ?= $(shell \
    conda run -n groot which python 2>/dev/null || \
    conda run -n warp-score which python 2>/dev/null || \
    echo "python")
GROOT_PY_OK := $(shell [ -x "$(GROOT_PY)" ] && echo yes || echo no)

PROMPTS    ?= $(SCRIPT_DIR)prompts.json
SEED       ?= 2000
SAVE_DIR   ?= $(DATA_DIR)/cosmos_synthetic_data/query/high
SMOKE_DIR  ?= $(SCRIPT_DIR)output/smoke

# --------------------------------------------------------------------------
setup:
	bash $(SCRIPT_DIR)setup.sh

smoke:
	@echo "[smoke] 1 prompt -> 1 video at $(SMOKE_DIR)"
	mkdir -p $(SMOKE_DIR)
	$(VENV_PY) -c "\
import json, sys; \
data = json.load(open('$(PROMPTS)')); \
pick = [p for p in data if p['task'].startswith('8_Use')] or data[:1]; \
json.dump(pick[:1], open('$(SMOKE_DIR)/prompts_smoke.json', 'w')); \
print('[smoke] selected task:', pick[0]['task'][:60])"
	$(VENV_PY) $(SCRIPT_DIR)generate.py \
	    --prompts $(SMOKE_DIR)/prompts_smoke.json \
	    --save_dir $(SMOKE_DIR) \
	    --input_dir $(DATA_DIR)/cosmos_inputs \
	    --seed_offset 9999

# Base 14B model, seed_offset=2000 (distinct from original set)
gen-query-high:
	@echo "[gen-query-high] 23 prompts -> $(SAVE_DIR) (profile=high, seed=$(SEED))"
	$(VENV_PY) $(SCRIPT_DIR)generate.py \
	    --prompts $(PROMPTS) \
	    --save_dir $(SAVE_DIR) \
	    --input_dir $(DATA_DIR)/cosmos_inputs \
	    --profile high \
	    --seed_offset $(SEED)

# GR00T GR1 fine-tune, seed_offset=3000
gen-query-gr00t:
	@echo "[gen-query-gr00t] 23 prompts -> $(SAVE_DIR) (profile=gr00t, seed=$(SEED))"
	$(VENV_PY) $(SCRIPT_DIR)generate.py \
	    --prompts $(PROMPTS) \
	    --save_dir $(SAVE_DIR) \
	    --input_dir $(DATA_DIR)/cosmos_inputs \
	    --profile gr00t \
	    --seed_offset $(SEED)

# Postprocess: MP4 -> frames -> SAM3 bg removal
# Uses GROOT_PY (groot/warp-score conda env) because SAM3 needs transformers.Sam3Processor.
postprocess:
	@echo "[postprocess] MP4 -> 50 frames -> SAM3 bg removal"
	@if [ "$(GROOT_PY_OK)" != "yes" ]; then \
	    echo "[warn] GROOT_PY=$(GROOT_PY) not executable — set GROOT_PY manually"; \
	fi
	$(GROOT_PY) $(SCRIPT_DIR)postprocess.py \
	    --video_root $(DATA_DIR)/cosmos_synthetic_data/query/high \
	    --frames_root $(DATA_DIR)/cosmos_frames_raw/query/high \
	    --out_root $(DATA_DIR)/query/high

clean-smoke:
	rm -rf $(SMOKE_DIR)
