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

# Held-out evaluation registry, grouped by the capability it measures.
# "split": None -> every split is downloaded. "source": "github" -> clone only, not on HF.
# "gated": True -> needs `huggingface-cli login` and accepting the dataset terms.
EVAL_REGISTRY = {
    # ==========================================
    # REASONING (math / science)
    # ==========================================
    "math-500": {"hf_id": "HuggingFaceH4/MATH-500", "capability": "reasoning", "split": "test",
                 "description": "500 competition problems, sanity check. Do not RL on hendrycks/competition_math test."},
    "aime-2024": {"hf_id": "HuggingFaceH4/aime_2024", "capability": "reasoning", "split": "train",
                  "description": "AIME 2024. Likely contaminated via NuminaMath; use only as reference."},
    "aime-2025": {"hf_id": "MathArena/aime_2025", "capability": "reasoning", "split": "train",
                  "description": "AIME 2025 I, post-cutoff."},
    "aime-2025-ii": {"hf_id": "MathArena/aime_2025_II", "capability": "reasoning", "split": "train",
                     "description": "AIME 2025 II."},
    "aime-2026": {"hf_id": "MathArena/aime_2026", "capability": "reasoning", "split": "train",
                  "description": "AIME 2026, cleanest against contamination."},
    "hmmt-feb-2025": {"hf_id": "MathArena/hmmt_feb_2025", "capability": "reasoning", "split": "train",
                      "description": "HMMT February 2025."},
    "hmmt-nov-2025": {"hf_id": "MathArena/hmmt_nov_2025", "capability": "reasoning", "split": "train",
                      "description": "HMMT November 2025."},
    "hmmt-feb-2026": {"hf_id": "MathArena/hmmt_feb_2026", "capability": "reasoning", "split": "train",
                      "description": "HMMT February 2026."},
    "brumo-2025": {"hf_id": "MathArena/brumo_2025", "capability": "reasoning", "split": "train",
                   "description": "BRUMO 2025."},
    "cmimc-2025": {"hf_id": "MathArena/cmimc_2025", "capability": "reasoning", "split": "train",
                   "description": "CMIMC 2025."},
    "amc23": {"hf_id": "math-ai/amc23", "capability": "reasoning", "split": "test",
              "description": "AMC 2023 (40 problems)."},
    "minerva-math": {"hf_id": "math-ai/minervamath", "capability": "reasoning", "split": "test",
                     "description": "Minerva-style undergraduate STEM word problems."},
    "olympiadbench": {"hf_id": "Hothan/OlympiadBench", "capability": "reasoning", "split": None,
                      "description": "Olympiad-level math and physics, bilingual."},
    "olymmath": {"hf_id": "RUC-AIBOX/OlymMATH", "capability": "reasoning", "split": None,
                 "description": "Olympiad math with easy/hard tiers, EN and ZH."},
    "omni-math": {"hf_id": "KbsdJames/Omni-MATH", "capability": "reasoning", "split": None,
                  "description": "4.4K Olympiad problems labelled by domain and difficulty."},
    "polymath": {"hf_id": "Qwen/PolyMath", "capability": "reasoning", "split": None,
                 "description": "Multilingual math reasoning."},
    "ugmathbench": {"hf_id": "UGMathBench/ugmathbench", "capability": "reasoning", "split": None,
                    "description": "Undergraduate math with randomized versions to detect memorization."},
    "theoremqa": {"hf_id": "TIGER-Lab/TheoremQA", "capability": "reasoning", "split": None,
                  "description": "Applying theorems from math, physics, EE and finance."},
    "gpqa-diamond": {"hf_id": "Idavidrein/gpqa", "capability": "reasoning", "subset": "gpqa_diamond",
                     "split": "train", "gated": True,
                     "description": "Graduate-level physics, chemistry, biology."},
    "mmlu-pro": {"hf_id": "TIGER-Lab/MMLU-Pro", "capability": "reasoning", "split": "test",
                 "description": "10-choice reasoning-heavy MMLU."},
    "supergpqa": {"hf_id": "m-a-p/SuperGPQA", "capability": "reasoning", "split": None,
                  "description": "Graduate-level QA across 285 disciplines."},
    "hle": {"hf_id": "cais/hle", "capability": "reasoning", "split": "test", "gated": True,
            "description": "Humanity's Last Exam, headroom check."},
    "phybench": {"hf_id": "Eureka-Lab/PHYBench", "capability": "reasoning", "split": None,
                 "description": "Physics reasoning with expression-edit-distance scoring."},
    "ugphysics": {"hf_id": "UGPhysics/ugphysics", "capability": "reasoning", "split": None,
                  "description": "Undergraduate physics reasoning."},
    "physreason": {"hf_id": "zhibei1204/PhysReason", "capability": "reasoning", "split": None,
                   "description": "Physics reasoning with theorem-based multi-step solutions."},
    "scibench": {"hf_id": "xw27/scibench", "capability": "reasoning", "split": None,
                 "description": "College-level math, physics, chemistry problems."},
    "olympicarena": {"hf_id": "GAIR/OlympicArena", "capability": "reasoning", "split": None,
                     "description": "Multi-discipline Olympiad problems."},
    "processbench": {"hf_id": "Qwen/ProcessBench", "capability": "reasoning", "split": None,
                     "description": "Locate the first erroneous step in a solution."},

    # ==========================================
    # FORMAL REASONING (Lean 4, compiler-verified)
    # ==========================================
    "minif2f-lean4": {"hf_id": "HaimingW/miniF2F-lean4", "capability": "formal", "split": None,
                      "description": "Olympiad statements. Hold out the test part, do not use it as RL prompts."},
    "proofnet-lean4": {"hf_id": "UDACA/proofnet-lean4", "capability": "formal", "split": None,
                       "description": "Undergraduate theorems in Lean 4."},
    "deepseek-proverbench": {"hf_id": "deepseek-ai/DeepSeek-ProverBench", "capability": "formal", "split": None,
                             "description": "325 problems from recent AIME and textbooks."},
    "formalmath-lite": {"hf_id": "SphereLab/FormalMATH-Lite", "capability": "formal", "split": None,
                        "description": "Subset of FormalMATH, Lean 4 from high school to undergraduate."},
    "formalmath-all": {"hf_id": "SphereLab/FormalMATH-All", "capability": "formal", "split": None,
                       "description": "Full FormalMATH, 5.5K statements."},
    "putnambench": {"source": "github", "repo": "https://github.com/trishullab/PutnamBench",
                    "capability": "formal",
                    "description": "Putnam problems in Lean 4, very hard. Clone only."},
    "formal-conjectures": {"source": "github", "repo": "https://github.com/google-deepmind/formal-conjectures",
                           "capability": "formal",
                           "description": "Open conjectures in Lean, for qualitative exploration only."},

    # ==========================================
    # TOOL USE
    # ==========================================
    "bfcl": {"hf_id": "gorilla-llm/Berkeley-Function-Calling-Leaderboard", "capability": "tool_use",
             "split": None, "description": "Single, parallel, multi-turn function calling and irrelevance detection."},
    "nestful": {"hf_id": "ibm-research/nestful", "capability": "tool_use", "split": None,
                "description": "Nested API calls where one output feeds the next."},
    "nexusraven-api": {"hf_id": "Nexusflow/NexusRaven_API_evaluation", "capability": "tool_use", "split": None,
                       "description": "Function calling over real APIs."},
    "apigen-mt": {"hf_id": "Salesforce/APIGen-MT-5k", "capability": "tool_use", "split": None,
                  "description": "Multi-turn tool trajectories, dev set only."},
    "multichallenge": {"hf_id": "ScaleAI/MultiChallenge", "capability": "tool_use", "split": None,
                       "description": "Multi-turn instruction retention and self-coherence."},
    "scicode": {"hf_id": "SciCode1/SciCode", "capability": "tool_use", "split": None,
                "description": "Scientific computing code from physics, math, chemistry papers."},
    "swe-bench-verified": {"hf_id": "princeton-nlp/SWE-bench_Verified", "capability": "tool_use", "split": "test",
                           "description": "Agentic repo-level fixes. Optional, needs a full harness."},
    "tau-bench": {"source": "github", "repo": "https://github.com/sierra-research/tau-bench",
                  "capability": "tool_use", "description": "Policy-bound multi-turn agent. Clone only."},
    "tau2-bench": {"source": "github", "repo": "https://github.com/sierra-research/tau2-bench",
                   "capability": "tool_use", "description": "tau-bench successor with dual-control domains. Clone only."},
    "mcp-universe": {"source": "github", "repo": "https://github.com/SalesforceAIResearch/MCP-Universe",
                     "capability": "tool_use", "description": "Real MCP server tasks. Clone only."},

    # ==========================================
    # LOGIC-BASED CREATIVITY (search, rule induction, novelty)
    # ==========================================
    "zebralogic": {"hf_id": "WildEval/ZebraLogic", "capability": "logic_creativity", "split": None,
                   "description": "Constraint-satisfaction grid puzzles with scalable difficulty."},
    "enigmata-eval": {"hf_id": "BytedTsinghua-SIA/Enigmata-Eval", "capability": "logic_creativity", "split": None,
                      "description": "Auto-verifiable puzzles (crypto, mazes, grids, arithmetic)."},
    "bbeh": {"hf_id": "BBEH/bbeh", "capability": "logic_creativity", "split": None,
             "description": "BIG-Bench Extra Hard."},
    "bbh": {"hf_id": "lukaemon/bbh", "capability": "logic_creativity", "split": None,
            "description": "BIG-Bench Hard, reference only (saturated)."},
    "folio": {"hf_id": "yale-nlp/FOLIO", "capability": "logic_creativity", "split": None,
              "description": "First-order logic entailment."},
    "countdown": {"hf_id": "Jiayi-Pan/Countdown-Tasks-3to4", "capability": "logic_creativity", "split": None,
                  "description": "Combine numbers with operators to hit a target; needs search."},
    "apex-2025": {"hf_id": "MathArena/apex_2025", "capability": "logic_creativity", "split": "train",
                  "description": "Problems nearly all models fail; need a new idea."},
    "usamo-2025": {"hf_id": "MathArena/usamo_2025", "capability": "logic_creativity", "split": "train",
                   "description": "Proof-based, needs judge grading."},
    "novelty-bench": {"hf_id": "yimingzhang/novelty-bench", "capability": "logic_creativity", "split": None,
                      "description": "Answer diversity. Measures novelty, not logic."},
    "reasoning-gym": {"source": "github", "repo": "https://github.com/open-thought/reasoning-gym",
                      "capability": "logic_creativity",
                      "description": "Procedurally generated tasks with verifiers. `pip install reasoning-gym`."},
    "arc-agi-2": {"source": "github", "repo": "https://github.com/arcprize/ARC-AGI-2",
                  "capability": "logic_creativity",
                  "description": "Abstract rule induction, hard for text-only models. Clone only."},
}

def list_evals():
    print("=" * 80)
    print(" EVAL REGISTRY (held-out benchmarks)")
    print("=" * 80)
    current = None
    for key, meta in EVAL_REGISTRY.items():
        if meta["capability"] != current:
            current = meta["capability"]
            print(f"\n>>> {current.upper()} <<<")
        where = meta.get("hf_id") or meta["repo"]
        gated = " [gated]" if meta.get("gated") else ""
        print(f"  • '{key}'{gated}  {where}")
        print(f"    {meta['description']}")

def download_eval(eval_key: str, output_dir: str = "data/eval", max_samples: int = None):
    from datasets import load_dataset

    if eval_key not in EVAL_REGISTRY:
        raise ValueError(f"Unknown eval '{eval_key}'. Choose from: {list(EVAL_REGISTRY.keys())}")

    meta = EVAL_REGISTRY[eval_key]
    if meta.get("source") == "github":
        print(f"[!] '{eval_key}' is not on HuggingFace. Run: git clone {meta['repo']} {output_dir}/{eval_key}")
        return

    subset, split = meta.get("subset"), meta.get("split")
    print(f"[*] Downloading '{meta['hf_id']}' (subset: {subset}, split: {split or 'all'}) ...")
    ds = load_dataset(meta["hf_id"], subset, split=split) if split else load_dataset(meta["hf_id"], subset)
    parts = {split: ds} if split else dict(ds)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        if max_samples:
            part = part.select(range(min(len(part), max_samples)))
        out_file = output_path / f"{eval_key}_{name}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for item in part:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[✓] {len(part)} samples -> {out_file}")

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
    parser.add_argument("--list_evals", action="store_true", help="List held-out eval benchmarks")
    parser.add_argument("--download_eval", type=str, default=None, help="Eval key to download")
    args = parser.parse_args()

    if args.list_evals:
        list_evals()
    elif args.download_eval:
        download_eval(args.download_eval, max_samples=args.max_samples)
    elif args.list or not args.download:
        list_registry()
    else:
        download_dataset(args.download, output_dir=args.output_dir, max_samples=args.max_samples, split=args.split)
