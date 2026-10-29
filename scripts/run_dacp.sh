#!/usr/bin/env bash
set -euo pipefail

# Script to launch Stage 1: DACP (Domain-Adaptive Continued Pre-training)
# on 1x RTX 6000 Pro (96GB VRAM)

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

CONFIG_PATH="${1:-configs/dacp_config.yaml}"
OUTPUT_DIR="${2:-outputs/qwen3.8-27b-dacp}"

echo "=========================================================="
echo " Launching Stage 1: DACP (Mathlib4 & Theoretical Physics)"
echo " Config: ${CONFIG_PATH}"
echo " Output Directory: ${OUTPUT_DIR}"
echo " Precision: BF16 (No Quantization) + RSLoRA r=128"
echo "=========================================================="

python3 -m src.train.dacp_train \
    --config "${CONFIG_PATH}" \
    --output_dir "${OUTPUT_DIR}"

echo "DACP Run Completed!"
