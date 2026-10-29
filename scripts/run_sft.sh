#!/usr/bin/env bash
set -euo pipefail

# Script to launch Cold-Start SFT on 1x RTX 6000 Pro (96GB VRAM)
# Unquantized BF16 Base Model + RSLoRA (r=128, alpha=256)

export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

CONFIG_PATH="${1:-configs/sft_config.yaml}"
OUTPUT_DIR="${2:-outputs/qwen3.8-27b-lean4-sft}"

echo "=========================================================="
echo " Launching Stage 1: Cold-Start SFT"
echo " Config: ${CONFIG_PATH}"
echo " Output Directory: ${OUTPUT_DIR}"
echo " Precision: BF16 (No Quantization) + RSLoRA r=128"
echo "=========================================================="

python3 -m src.train.sft_train \
    --config "${CONFIG_PATH}" \
    --output_dir "${OUTPUT_DIR}"

echo "SFT Run Completed!"
