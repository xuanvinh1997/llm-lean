"""
Stage 1: Domain-Adaptive Continued Pre-training (DACP).
Trains Qwen3.8-27B on Mathlib4, Theoretical Physics & Math Textbooks.
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
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset

from src.utils.gpu_mem import print_gpu_vram_summary

def parse_args():
    parser = argparse.ArgumentParser(description="Run DACP Continued Pre-training")
    parser.add_argument("--config", type=str, default="configs/dacp_config.yaml", help="Path to DACP config")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset path")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples")
    return parser.parse_args()

def main():
    args = parse_args()
    config_path = Path(args.config)
    
    # Fallback default configuration if config file doesn't exist yet
    cfg = {
        "model_id": "Qwen/Qwen3.8-27B",
        "torch_dtype": "bfloat16",
        "output_dir": "./outputs/qwen3.8-27b-dacp",
        "dataset_name": args.dataset or "data/dacp_math_physics_sample.jsonl",
        "learning_rate": 2.0e-5,
        "max_seq_length": 8192,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "save_steps": 100,
        "logging_steps": 5
    }
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f)
            cfg.update(user_cfg)

    if args.output_dir:
        cfg["output_dir"] = args.output_dir

    print(f"=== Starting Stage 1: DACP (Continued Pre-training) ===")
    print(f"Model ID: {cfg['model_id']}")
    print(f"Precision: {cfg['torch_dtype']} (Unquantized Base)")
    print_gpu_vram_summary("Pre-Initialization")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_id"],
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        device_map="auto"
    )

    # Setup RSLoRA for deep domain adaptation
    peft_config = LoraConfig(
        r=128,
        lora_alpha=256,
        lora_dropout=0.05,
        use_rslora=True,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load dataset
    ds_path = Path(cfg["dataset_name"])
    if ds_path.exists():
        dataset = load_dataset("json", data_files=str(ds_path), split="train")
    else:
        # Fallback dummy sample for dry-run
        dataset = Dataset.from_dict({
            "text": [
                "Mathlib4 topology: In a topological space X, a subset U is open iff...",
                "Hamiltonian mechanics: The phase space flow is governed by Hamilton's canonical equations...",
                "Quantum physics: The state vector in Hilbert space evolves unitarily according to the Schroedinger equation..."
            ] * 10
        })

    if args.max_samples:
        dataset = dataset.select(range(min(len(dataset), args.max_samples)))

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=cfg["max_seq_length"],
            padding="max_length"
        )

    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=cfg["logging_steps"],
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        save_total_limit=3,
        dataloader_num_workers=4,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    print("Beginning DACP pre-training loop...")
    trainer.train()

    final_output = os.path.join(cfg["output_dir"], "final")
    print(f"Saving DACP adapted weights to {final_output}...")
    trainer.model.save_pretrained(final_output)
    tokenizer.save_pretrained(final_output)
    print("Stage 1 DACP Completed Successfully!")

if __name__ == "__main__":
    main()
