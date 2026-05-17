# Makefile -- minimal one-command workflow for dreamgen_data/.
#
#   make setup           # install cosmos-predict2 + download 2B ckpt (one-time, ~50 GB)
#   make smoke           # 1 prompt -> 1 video (verify cosmos-predict2 API)
#   make gen-query-high  # 23 prompts -> 23 MP4s into data/cosmos_synthetic_data/query/high
#   make postprocess     # MP4s -> 50 frames -> SAM3 bg removal -> data/query/high
#
# Override save_dir / prompts via CLI:
#   make gen-query-high SAVE_DIR=/tmp/foo PROMPTS=prompts.json

.PHONY: setup smoke gen-query-high postprocess clean-smoke

SCRIPT_DIR := $(shell pwd)
DATA_DIR   := $(SCRIPT_DIR)/../data
VENV_PY    := $(SCRIPT_DIR)/cosmos-predict2/.venv/bin/python

PROMPTS    ?= prompts.json
SEED       ?= 1000
SAVE_DIR   ?= $(DATA_DIR)/cosmos_synthetic_data/query/high
SMOKE_DIR  ?= $(SCRIPT_DIR)/output/smoke

setup:
	bash setup.sh

smoke:
	@echo "[smoke] 1 prompt -> 1 video at $(SMOKE_DIR)"
	mkdir -p $(SMOKE_DIR)
	# Pick the one prompt whose name matches the GR1 sample image bundled
	# in the cosmos-predict2 repo (task ID prefix "8_"). This lets the smoke
	# test exercise the *image-conditioned* path without needing extra inputs.
	$(VENV_PY) -c "import json; \
	    data = json.load(open('$(PROMPTS)')); \
	    pick = [p for p in data if p['task'].startswith('8_Use')] or data[:1]; \
	    json.dump(pick[:1], open('$(SMOKE_DIR)/prompts_smoke.json', 'w'))"
	$(VENV_PY) generate.py \
	    --prompts $(SMOKE_DIR)/prompts_smoke.json \
	    --save_dir $(SMOKE_DIR) \
	    --seed_offset 9999

gen-query-high:
	@echo "[gen-query-high] 23 prompts -> $(SAVE_DIR)"
	$(VENV_PY) generate.py \
	    --prompts $(PROMPTS) \
	    --save_dir $(SAVE_DIR) \
	    --seed_offset $(SEED) \
	    --input_dir $(DATA_DIR)/cosmos_inputs

postprocess:
	@echo "[postprocess] MP4 -> 50 frames -> SAM3 bg removal"
	$(VENV_PY) postprocess.py \
	    --video_root $(DATA_DIR)/cosmos_synthetic_data/query/high \
	    --frames_root $(DATA_DIR)/cosmos_frames_raw/query/high \
	    --out_root $(DATA_DIR)/query/high

clean-smoke:
	rm -rf $(SMOKE_DIR)
