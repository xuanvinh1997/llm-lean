#!/usr/bin/env bash
set -euo pipefail

# Script to launch Stage 2: RLVR with GRPO on 1x RTX 6000 Pro (96GB VRAM)
# Colocated vLLM rollout (0.45 utilization) + Backward pass on RSLoRA adapter

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

CONFIG_PATH="${1:-configs/grpo_config.yaml}"
SFT_ADAPTER="${2:-outputs/qwen3.8-27b-lean4-sft/final}"
OUTPUT_DIR="${3:-outputs/qwen3.8-27b-lean4-grpo}"

echo "=========================================================="
echo " Launching Stage 2: RLVR with GRPO"
echo " Config: ${CONFIG_PATH}"
echo " Base SFT Adapter: ${SFT_ADAPTER}"
echo " Output Directory: ${OUTPUT_DIR}"
echo " Group Size K = 4 | Ground Truth Verifier = Lean 4 CPU"
echo "=========================================================="

python3 -m src.train.grpo_train \
    --config "${CONFIG_PATH}" \
    --sft_adapter "${SFT_ADAPTER}" \
    --output_dir "${OUTPUT_DIR}"

echo "GRPO Run Completed!"
