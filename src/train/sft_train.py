"""
Stage 1: Cold-Start SFT for Formal Lean 4 Theorem Proving.
Pure BF16 Base Model + RSLoRA (r=128, alpha=256) on 1x RTX 6000 Pro 96GB.
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
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

from src.data.dataset import load_lean_sft_dataset
from src.utils.gpu_mem import print_gpu_vram_summary, clear_gpu_cache

def parse_args():
    parser = argparse.ArgumentParser(description="Run Lean 4 Cold-Start SFT Training")
    parser.add_argument("--config", type=str, default="configs/sft_config.yaml", help="Path to SFT YAML config")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset path")
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
    train_args_cfg = cfg["training_arguments"]

    if args.dataset:
        data_cfg["dataset_name_or_path"] = args.dataset
    if args.output_dir:
        train_args_cfg["output_dir"] = args.output_dir

    print(f"=== Starting Cold-Start SFT ===")
    print(f"Model ID: {model_cfg['model_id']}")
    print(f"Precision: {model_cfg['torch_dtype']} (Unquantized Base)")
    print(f"LoRA: RSLoRA={lora_cfg.get('use_rslora', True)}, r={lora_cfg['r']}, alpha={lora_cfg['lora_alpha']}")
    print_gpu_vram_summary("Pre-Initialization")

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["model_id"],
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Model in 100% BF16 (No Quantization)
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    torch_dtype = dtype_map.get(model_cfg.get("torch_dtype", "bfloat16"), torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_id"],
        torch_dtype=torch_dtype,
        attn_implementation=model_cfg.get("attn_implementation", "flash_attention_2"),
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        device_map="auto"
    )

    print_gpu_vram_summary("Base Model Loaded")

    # 3. Setup High-Rank RSLoRA Adapter
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        use_rslora=lora_cfg.get("use_rslora", True),
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM")
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print_gpu_vram_summary("RSLoRA Attached")

    # 4. Load Dataset
    dataset = load_lean_sft_dataset(
        data_cfg["dataset_name_or_path"],
        tokenizer=tokenizer,
        max_samples=args.max_samples
    )
    print(f"Loaded {len(dataset)} SFT training examples.")

    # 5. Build SFTConfig & SFTTrainer
    training_args = SFTConfig(
        output_dir=train_args_cfg["output_dir"],
        learning_rate=float(train_args_cfg["learning_rate"]),
        lr_scheduler_type=train_args_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(train_args_cfg.get("warmup_ratio", 0.03)),
        per_device_train_batch_size=int(train_args_cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(train_args_cfg.get("gradient_accumulation_steps", 16)),
        gradient_checkpointing=train_args_cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs=train_args_cfg.get("gradient_checkpointing_kwargs", {"use_reentrant": False}),
        max_grad_norm=float(train_args_cfg.get("max_grad_norm", 1.0)),
        bf16=train_args_cfg.get("bf16", True),
        fp16=train_args_cfg.get("fp16", False),
        optim=train_args_cfg.get("optim", "paged_adamw_8bit"),
        logging_steps=int(train_args_cfg.get("logging_steps", 5)),
        save_strategy=train_args_cfg.get("save_strategy", "steps"),
        save_steps=int(train_args_cfg.get("save_steps", 100)),
        save_total_limit=int(train_args_cfg.get("save_total_limit", 3)),
        dataloader_num_workers=int(train_args_cfg.get("dataloader_num_workers", 4)),
        report_to=train_args_cfg.get("report_to", "none"),
        seed=int(train_args_cfg.get("seed", 42)),
        max_seq_length=int(data_cfg.get("max_seq_length", 8192)),
        dataset_text_field="text" if "text" in dataset.column_names else None,
        packing=data_cfg.get("packing", False)
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config
    )

    # 6. Execute Training
    print("Beginning SFT training loop...")
    trainer.train()

    # 7. Save Final Adapter
    final_output = os.path.join(train_args_cfg["output_dir"], "final")
    print(f"Saving final RSLoRA adapter to {final_output}...")
    trainer.model.save_pretrained(final_output)
    tokenizer.save_pretrained(final_output)

    print_gpu_vram_summary("Post-Training Complete")
    print("SFT Stage Completed Successfully!")

if __name__ == "__main__":
    main()
