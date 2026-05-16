# Makefile — top-level workflow for warp_score hallucination detection.
#
# Usage:
#   make install
#   make calibrate
#   make detect
#   make eval
#   make all           # calibrate → detect → eval
#
# Override any variable on the command line:
#   make calibrate ARTIFACTS=artifacts/v10 HIGH_DIR=/data/high DEVICE=cuda:1

ARTIFACTS  ?= artifacts/v9_dreamgen
HIGH_DIR   ?= ../image_no_bg/high
LOW_DIR    ?= ../image_no_bg/low
CONFIG     ?= warp_score/configs/default.yaml
LABELS     ?= labels.csv
DEVICE     ?= cuda

UPLOAD_SCRIPT := /mnt/data/sftp/data/quangpt3/.claude/skills/upload-to-hf/upload.py
PYTHON        := python

.PHONY: install calibrate detect eval labels upload-samples \
        dreamgen-smoke dreamgen-full all help

help:	## print available targets
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | sort | \
	  awk 'BEGIN {FS = ":.*?##"} {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

install:	## pip install -e .
	$(PYTHON) -m pip install -e .

calibrate:	## build per-task empirical null distributions from high/ refs
	$(PYTHON) -m warp_score \
	    --config $(CONFIG) \
	    --high_dir $(HIGH_DIR) \
	    --artifacts_dir $(ARTIFACTS) \
	    --device $(DEVICE) \
	    calibrate

detect:	## score all low/ query frames → artifacts/summary.csv
	$(PYTHON) -m warp_score \
	    --config $(CONFIG) \
	    --high_dir $(HIGH_DIR) \
	    --low_dir $(LOW_DIR) \
	    --artifacts_dir $(ARTIFACTS) \
	    --device $(DEVICE) \
	    detect

eval:	## compute AUROC/AP/FPR@95TPR from labels CSV + summary.csv
	$(PYTHON) -m warp_score \
	    --config $(CONFIG) \
	    --artifacts_dir $(ARTIFACTS) \
	    eval --labels $(LABELS)

labels:	## build labels.csv from high/ and low/ directories
	$(PYTHON) scripts/build_weak_labels.py \
	    --high_dir $(HIGH_DIR) \
	    --low_dir $(LOW_DIR) \
	    --out $(LABELS)

upload-samples:	## upload first 5 sample PNGs from LOW_DIR to HuggingFace
	$(PYTHON) $(UPLOAD_SCRIPT) \
	    $(shell find $(LOW_DIR) -name "*.png" | head -5 | tr '\n' ' ')

dreamgen-smoke:	## run DreamGen smoke test (1 prompt × 5 videos)
	cd dreamgen_data && $(MAKE) smoke

dreamgen-full:	## full DreamGen generation + harvest
	cd dreamgen_data && $(MAKE) full-high full-halluc harvest

all: calibrate detect eval	## calibrate → detect → eval
