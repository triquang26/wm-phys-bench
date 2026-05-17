# Makefile — top-level workflow for warp_score hallucination detection.
#
# Usage:
#   make install
#   make calibrate
#   make detect
#   make eval
#   make all           # calibrate -> detect -> eval
#   make reproduce     # restructure-data -> calibrate -> detect -> eval
#
# Override any variable on the command line:
#   make calibrate ARTIFACTS=artifacts/v10 REF_DIR=/data/reference DEVICE=cuda:1

REPO_ROOT       := $(shell pwd)
ARTIFACTS       ?= $(REPO_ROOT)/artifacts/v9_dreamgen
REF_DIR         ?= $(REPO_ROOT)/data/reference
QUERY_HIGH_DIR  ?= $(REPO_ROOT)/data/query/high
QUERY_LOW_DIR   ?= $(REPO_ROOT)/data/query/low
CONFIG          ?= warp_score/configs/default.yaml
LABELS          ?= labels.csv
DEVICE          ?= cuda

UPLOAD_SCRIPT ?= /mnt/data/sftp/data/quangpt3/.claude/skills/upload-to-hf/upload.py
PYTHON        := python

.PHONY: install calibrate detect eval labels upload-samples restructure-data \
        reproduce dreamgen-smoke dreamgen-full all help

help:	## print available targets
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | sort | \
	  awk 'BEGIN {FS = ":.*?##"} {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

install:	## pip install -e .
	$(PYTHON) -m pip install -e .

calibrate:	## build per-task empirical null distributions from reference/ refs
	$(PYTHON) -m warp_score \
	    --config $(CONFIG) \
	    --ref_dir $(REF_DIR) \
	    --artifacts_dir $(ARTIFACTS) \
	    --device $(DEVICE) \
	    calibrate

detect:	## score all query/{high,low} frames -> artifacts/summary.csv
	$(PYTHON) -m warp_score \
	    --config $(CONFIG) \
	    --ref_dir $(REF_DIR) \
	    --query_high_dir $(QUERY_HIGH_DIR) \
	    --query_low_dir $(QUERY_LOW_DIR) \
	    --artifacts_dir $(ARTIFACTS) \
	    --device $(DEVICE) \
	    detect

eval:	## compute AUROC/AP/FPR@95TPR from labels CSV + summary.csv
	$(PYTHON) -m warp_score \
	    --config $(CONFIG) \
	    --artifacts_dir $(ARTIFACTS) \
	    eval --labels $(LABELS)

labels:	## build labels.csv from query/{high,low} directories
	$(PYTHON) scripts/build_weak_labels.py \
	    --query_high_dir $(QUERY_HIGH_DIR) \
	    --query_low_dir $(QUERY_LOW_DIR) \
	    --out $(LABELS)

upload-samples:	## upload first 5 sample PNGs from QUERY_LOW_DIR to HuggingFace
	$(PYTHON) $(UPLOAD_SCRIPT) \
	    $(shell find $(QUERY_LOW_DIR) -name "*.png" | head -5 | tr '\n' ' ')

restructure-data:	## stub: reorganize data/ into reference/, query/{high,low} (owned by separate script/agent)
	@echo "restructure-data is a stub — perform data layout migration outside this Makefile."

reproduce: restructure-data calibrate detect eval	## end-to-end: restructure-data -> calibrate -> detect -> eval

dreamgen-smoke:	## run DreamGen smoke test (1 prompt x 5 videos)
	cd dreamgen_data && $(MAKE) smoke

dreamgen-full:	## full DreamGen generation + harvest
	cd dreamgen_data && $(MAKE) full-high full-halluc harvest

all: calibrate detect eval	## calibrate -> detect -> eval
