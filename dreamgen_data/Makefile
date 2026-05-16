# Makefile — one-command workflow for DreamGen data generation.
#
# Quick usage:
#   make setup             # install cosmos-predict2 + download ckpts (~250GB, one-time)
#   make data              # prepare HF data + extract first frames
#   make smoke             # 1 prompt × 5 videos, high profile — validate single-GPU works
#   make full              # full gen: 50 videos × N_PROMPTS × 2 profiles
#   make harvest           # videos → image_no_bg/{high,low}/<task>/*.png
#   make all               # data → smoke → full → harvest
#
# Tunable variables (override on command line):
#   make full N_VIDEOS=25 N_PROMPTS=10
#   make smoke MODEL_SIZE=2B

ROOT          := $(shell pwd)
COSMOS_DIR    := $(ROOT)/cosmos-predict2
PY            := python
TORCHRUN      := torchrun
N_GPUS        ?= 1
N_VIDEOS      ?= 50
N_PROMPTS     ?= 23
START_IDX     ?= 0
END_IDX       ?= $(N_PROMPTS)
MODEL_SIZE    ?= 2B      # default 2B (single-GPU friendly); set MODEL_SIZE=14B for full quality
IMAGE_NO_BG   := $(ROOT)/../../image_no_bg

# Pass-through extra args
EXTRA         ?=


.PHONY: help setup data smoke smoke-2b \
        full full-high full-halluc \
        harvest harvest-high harvest-halluc \
        clean-output

help:
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | sort | awk \
	  'BEGIN {FS = ":.*?##"} {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup:                          ## install cosmos-predict2 + download checkpoints
	bash setup.sh

data:                           ## download HF dataset + extract first frames
	$(PY) prepare_data.py --out_dir data --max_items 100

smoke:                          ## 1 prompt × 5 videos, high profile (14B by default)
	cd $(COSMOS_DIR) && $(TORCHRUN) --nproc_per_node=$(N_GPUS) $(ROOT)/generate.py \
	    --profile high --num_videos 5 \
	    --batch_json $(ROOT)/data/batch_input.json \
	    --save_dir $(ROOT)/output/smoke_high \
	    --start_idx 0 --end_idx 1 \
	    $(if $(MODEL_SIZE),--model_size $(MODEL_SIZE),) $(EXTRA)

smoke-2b:                       ## 1 prompt × 5 videos, high profile, 2B model
	$(MAKE) smoke MODEL_SIZE=2B

full: full-high full-halluc     ## full generation: 50 videos × N prompts × 2 profiles

full-high:                      ## generate "high" (clean) profile
	cd $(COSMOS_DIR) && $(TORCHRUN) --nproc_per_node=$(N_GPUS) $(ROOT)/generate.py \
	    --profile high --num_videos $(N_VIDEOS) \
	    --batch_json $(ROOT)/data/batch_input.json \
	    --save_dir $(ROOT)/output/high \
	    --start_idx $(START_IDX) --end_idx $(END_IDX) \
	    $(if $(MODEL_SIZE),--model_size $(MODEL_SIZE),) $(EXTRA)

full-halluc:                    ## generate "hallucinate" profile
	cd $(COSMOS_DIR) && $(TORCHRUN) --nproc_per_node=$(N_GPUS) $(ROOT)/generate.py \
	    --profile hallucinate --num_videos $(N_VIDEOS) \
	    --batch_json $(ROOT)/data/batch_input.json \
	    --save_dir $(ROOT)/output/hallucinate \
	    --start_idx $(START_IDX) --end_idx $(END_IDX) \
	    $(if $(MODEL_SIZE),--model_size $(MODEL_SIZE),) $(EXTRA)

harvest: harvest-high harvest-halluc  ## videos → image_no_bg/{high,low}/<task>/

harvest-high:                   ## high MP4s → image_no_bg/high/<task>/*.png
	$(PY) postprocess.py \
	    --video_root output/high --profile high \
	    --metadata data/metadata.jsonl \
	    --out_root $(IMAGE_NO_BG)

harvest-halluc:                 ## hallucinate MP4s → image_no_bg/low/<task>/*.png
	$(PY) postprocess.py \
	    --video_root output/hallucinate --profile hallucinate \
	    --metadata data/metadata.jsonl \
	    --out_root $(IMAGE_NO_BG)

all: data smoke full harvest    ## full pipeline (smoke included for safety)

clean-output:                   ## DANGEROUS — delete all generated MP4s
	rm -rf output/high output/hallucinate output/smoke_high
