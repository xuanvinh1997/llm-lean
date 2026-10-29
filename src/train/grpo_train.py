"""
Stage 2: RLVR with GRPO for Formal Lean 4 Theorem Proving.
Uses TRL GRPOTrainer, vLLM for high-throughput rollout, and CPU Lean 4 Sandbox Verifier.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel, LoraConfig
from trl import GRPOTrainer, GRPOConfig

from src.data.dataset import load_lean_rl_dataset
from src.verifier.rewards import (
    lean4_reward_func,
    python_sympy_reward_func,
    format_reward_func,
    get_sandbox
)
from src.utils.gpu_mem import print_gpu_vram_summary

def parse_args():
    parser = argparse.ArgumentParser(description="Run Lean 4 GRPO Training")
    parser.add_argument("--config", type=str, default="configs/grpo_config.yaml", help="Path to GRPO YAML config")
    parser.add_argument("--dataset", type=str, default=None, help="Override RL prompts dataset path")
    parser.add_argument("--sft_adapter", type=str, default=None, help="Path to pre-trained SFT adapter")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit training samples for testing")
    return parser.parse_args()

def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    data_cfg = cfg["data"]
    grpo_cfg = cfg["grpo"]
    verifier_cfg = cfg.get("verifier", {})

    if args.dataset:
        data_cfg["dataset_name_or_path"] = args.dataset
    if args.output_dir:
        grpo_cfg["output_dir"] = args.output_dir
    sft_adapter_path = args.sft_adapter or model_cfg.get("sft_adapter_path")

    # Initialize CPU Sandbox Verifier Engine
    get_sandbox(
        max_workers=verifier_cfg.get("max_workers", 16),
        lake_path=verifier_cfg.get("lake_path"),
        repl_bin=verifier_cfg.get("repl_path"),
        lake_dir=verifier_cfg.get("lake_dir"),
        timeout_seconds=verifier_cfg.get("timeout_seconds", 15.0)
    )

    print(f"=== Starting RLVR with GRPO ===")
    print(f"Model ID: {model_cfg['model_id']}")
    print(f"SFT Adapter Path: {sft_adapter_path}")
    print(f"GRPO Group Size (K): {grpo_cfg['num_generations']}")
    print(f"vLLM Co-location Enabled: {grpo_cfg.get('use_vllm', True)} (utilization: {grpo_cfg.get('vllm_gpu_memory_utilization', 0.45)})")
    print_gpu_vram_summary("Pre-GRPO Initialization")

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["model_id"],
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Precision & Model Loading
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    torch_dtype = dtype_map.get(model_cfg.get("torch_dtype", "bfloat16"), torch.bfloat16)

    # 3. Setup GRPOConfig
    training_args = GRPOConfig(
        output_dir=grpo_cfg["output_dir"],
        learning_rate=float(grpo_cfg.get("learning_rate", 3.0e-6)),
        lr_scheduler_type=grpo_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(grpo_cfg.get("warmup_ratio", 0.05)),
        per_device_train_batch_size=int(grpo_cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(grpo_cfg.get("gradient_accumulation_steps", 8)),
        num_generations=int(grpo_cfg.get("num_generations", 4)),
        temperature=float(grpo_cfg.get("temperature", 0.7)),
        top_p=float(grpo_cfg.get("top_p", 0.95)),
        max_prompt_length=int(grpo_cfg.get("max_prompt_length", 2048)),
        max_completion_length=int(grpo_cfg.get("max_completion_length", 6144)),
        beta=float(grpo_cfg.get("beta", 0.02)),
        gradient_checkpointing=grpo_cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs=grpo_cfg.get("gradient_checkpointing_kwargs", {"use_reentrant": False}),
        bf16=grpo_cfg.get("bf16", True),
        fp16=grpo_cfg.get("fp16", False),
        optim=grpo_cfg.get("optim", "paged_adamw_8bit"),
        logging_steps=int(grpo_cfg.get("logging_steps", 2)),
        save_steps=int(grpo_cfg.get("save_steps", 50)),
        save_total_limit=int(grpo_cfg.get("save_total_limit", 3)),
        report_to=grpo_cfg.get("report_to", "none"),
        seed=int(grpo_cfg.get("seed", 42)),
        use_vllm=grpo_cfg.get("use_vllm", True),
        vllm_gpu_memory_utilization=float(grpo_cfg.get("vllm_gpu_memory_utilization", 0.45)),
        vllm_max_model_len=int(grpo_cfg.get("vllm_max_model_len", 8192))
    )

    # 4. Prepare Model / Adapter
    if sft_adapter_path and os.path.exists(sft_adapter_path):
        print(f"Loading base model with pre-trained SFT adapter: {sft_adapter_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_cfg["model_id"],
            torch_dtype=torch_dtype,
            attn_implementation=model_cfg.get("attn_implementation", "flash_attention_2"),
            trust_remote_code=model_cfg.get("trust_remote_code", True),
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, sft_adapter_path, is_trainable=True)
    else:
        print("Initializing new RSLoRA adapter for GRPO training...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_cfg["model_id"],
            torch_dtype=torch_dtype,
            attn_implementation=model_cfg.get("attn_implementation", "flash_attention_2"),
            trust_remote_code=model_cfg.get("trust_remote_code", True),
            device_map="auto"
        )
        peft_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg.get("lora_dropout", 0.05),
            use_rslora=lora_cfg.get("use_rslora", True),
            target_modules=lora_cfg["target_modules"],
            task_type="CAUSAL_LM"
        )
        model = PeftModel(base_model, peft_config)

    print_gpu_vram_summary("Model & Policy Ready")

    # 5. Load RL Dataset
    dataset = load_lean_rl_dataset(
        data_cfg["dataset_name_or_path"],
        tokenizer=tokenizer,
        max_samples=args.max_samples
    )
    print(f"Loaded {len(dataset)} theorem proof prompts.")

    # 6. Initialize GRPOTrainer with multi-environment rewards
    reward_funcs = [
        lean4_reward_func,   # Ground truth: +1.0 proved, +0.1 partial, -0.5 error, -0.3 timeout
        format_reward_func   # Formatting: <think> tags & ```lean4 / ```python code fence
    ]
    if verifier_cfg.get("enable_python_sandbox", True):
        reward_funcs.append(python_sympy_reward_func)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer
    )

    # 7. Run GRPO Optimization
    print("Beginning GRPO RLVR optimization loop...")
    trainer.train()

    # 8. Save Final Model
    final_output = os.path.join(grpo_cfg["output_dir"], "final")
    print(f"Saving final GRPO adapter to {final_output}...")
    trainer.model.save_pretrained(final_output)
    tokenizer.save_pretrained(final_output)

    print_gpu_vram_summary("GRPO Completed")
    print("RLVR Stage Completed Successfully!")

if __name__ == "__main__":
    main()
