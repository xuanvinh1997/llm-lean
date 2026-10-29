"""
Comprehensive Dataset Registry and Downloader for the 3-Stage Training Pipeline:
  Stage 1: DACP (Domain-Adaptive Continued Pre-training) - Mathlib4, OpenWebMath, Proof-Pile-2, Physics Textbooks/arXiv
  Stage 2: Agentic & Formal SFT - Tool Calling (Python, SymPy), CoT, Synthetic Proof Search
  Stage 3: RLVR / GRPO - Multi-Environment Ground Truth (Lean 4 REPL, Python Sandbox, Math Solvers)
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Full 3-Stage Dataset Registry
DATASET_REGISTRY = {
    # ==========================================
    # STAGE 1: DACP (Toán - Lý Lý Thuyết & Mathlib4)
    # ==========================================
    "proof-pile-2": {
        "hf_id": "EleutherAI/proof-pile-2",
        "stage": "Stage 1: DACP",
        "description": "55B tokens of ArXiv, OpenWebMath, and AlgebraicStack (Math, Physics, Formal Code).",
        "subset": "arxiv", # or "open-web-math", "algebraic-stack"
        "type": "pretraining"
    },
    "openwebmath": {
        "hf_id": "open-web-math/open-web-math",
        "stage": "Stage 1: DACP",
        "description": "14.7B tokens of high-quality mathematical web documents with full LaTeX math.",
        "type": "pretraining"
    },
    "finemath": {
        "hf_id": "HuggingFaceTB/finemath",
        "stage": "Stage 1: DACP",
        "description": "34B-54B tokens of high-educational-value math pretraining text filtered by LLM.",
        "type": "pretraining"
    },
    "mathlib4-raw": {
        "hf_id": "phanerozoic/Lean4-Mathlib",
        "stage": "Stage 1: DACP & SFT",
        "description": "Full formal statements and proofs from official Mathlib4.",
        "type": "formal_code"
    },
    "physics-corpus": {
        "hf_id": "konsman/physics-corpus",
        "stage": "Stage 1: DACP",
        "description": "Structured arXiv physics preprints (quant-ph, hep-th, gr-qc, math-ph).",
        "type": "physics_theory"
    },

    # ==========================================
    # STAGE 2: High-Quality Agentic & Formal SFT (Tool Calling + CoT)
    # ==========================================
    "numinamath-tir": {
        "hf_id": "AI-MO/NuminaMath-TIR",
        "stage": "Stage 2: Agentic SFT",
        "description": "Tool-Integrated Reasoning (TIR) dataset interleaving natural language CoT with executable Python/SymPy.",
        "type": "tool_calling_sft"
    },
    "numinamath-cot": {
        "hf_id": "AI-MO/NuminaMath-CoT",
        "stage": "Stage 2: Agentic SFT",
        "description": "860K+ Olympiad and competition math solutions with step-by-step Chain of Thought.",
        "type": "cot_sft"
    },
    "mathinstruct": {
        "hf_id": "TIGER-Lab/MathInstruct",
        "stage": "Stage 2: Agentic SFT",
        "description": "260K multi-turn math instructions combining CoT (Chain of Thought) and PoT (Program of Thought).",
        "type": "tool_calling_sft"
    },
    "lean-workbook": {
        "hf_id": "internlm/Lean-Workbook",
        "stage": "Stage 2: Formal SFT & RL",
        "description": "57K - 83K formal Lean 4 Olympiad problems with theorem statements and formal proofs.",
        "type": "formal_sft"
    },

    # ==========================================
    # STAGE 3: RL with Verifiable Reward (RLVR / GRPO)
    # ==========================================
    "minif2f-lean4": {
        "hf_id": "HaimingW/miniF2F-lean4",
        "stage": "Stage 3: RLVR (Lean 4)",
        "description": "Standard benchmark (488 Olympiad math problems) for Lean 4 compiler verification.",
        "type": "rl_verifier"
    },
    "proofnet-lean4": {
        "hf_id": "UDACA/proofnet-lean4",
        "stage": "Stage 3: RLVR (Lean 4)",
        "description": "Undergraduate level math theorems formalized in Lean 4.",
        "type": "rl_verifier"
    },
    "competition-math-rl": {
        "hf_id": "hendrycks/competition_math",
        "stage": "Stage 3: RLVR (Python / Exact Math)",
        "description": "12,500 difficult competition math problems with verifiable final answers.",
        "type": "rl_verifier"
    }
}

def list_registry():
    print("=" * 80)
    print(" COMPREHENSIVE 3-STAGE DATASET REGISTRY (Qwen3.8-27B Pipeline)")
    print("=" * 80)
    
    current_stage = None
    for key, meta in DATASET_REGISTRY.items():
        if meta["stage"] != current_stage:
            current_stage = meta["stage"]
            print(f"\n>>> {current_stage.upper()} <<<")
        print(f"  • Key: '{key}'")
        print(f"    HF Repo:     {meta['hf_id']}")
        print(f"    Description: {meta['description']}")

def download_dataset(dataset_key: str, output_dir: str = "data", max_samples: int = None, split: str = "train"):
    from datasets import load_dataset

    if dataset_key not in DATASET_REGISTRY:
        raise ValueError(f"Unknown key '{dataset_key}'. Choose from: {list(DATASET_REGISTRY.keys())}")

    meta = DATASET_REGISTRY[dataset_key]
    hf_id = meta["hf_id"]
    subset = meta.get("subset")

    print(f"[*] Downloading '{hf_id}' (subset: {subset}, split: {split}) ...")
    if subset:
        ds = load_dataset(hf_id, subset, split=split)
    else:
        ds = load_dataset(hf_id, split=split)

    if max_samples:
        ds = ds.select(range(min(len(ds), max_samples)))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / f"{dataset_key}_{split}.jsonl"

    print(f"[*] Serializing {len(ds)} samples to {out_file}...")
    with open(out_file, "w", encoding="utf-8") as f:
        for item in ds:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[✓] Successfully downloaded and saved to: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3-Stage Pipeline Dataset Downloader")
    parser.add_argument("--list", action="store_true", help="List all datasets across 3 stages")
    parser.add_argument("--download", type=str, default=None, help="Dataset key to download")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples limit")
    parser.add_argument("--split", type=str, default="train", help="Split name")
    args = parser.parse_args()

    if args.list or not args.download:
        list_registry()
    else:
        download_dataset(args.download, output_dir=args.output_dir, max_samples=args.max_samples, split=args.split)
