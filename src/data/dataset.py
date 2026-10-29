"""
Dataset preparation utilities for Lean 4 SFT and GRPO training.
Supports JSONL files and Hugging Face datasets format.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datasets import Dataset, load_dataset
from src.data.prompts import SYSTEM_PROMPT, format_lean4_prompt

def load_lean_sft_dataset(
    file_path_or_name: str,
    tokenizer: Any = None,
    max_samples: Optional[int] = None
) -> Dataset:
    """
    Load SFT dataset.
    Expected record formats:
    1) {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
    OR
    2) {"theorem": "theorem foo (n : Nat) : n + 0 = n := by ...", "proof": "...", "think": "..."}
    """
    path = Path(file_path_or_name)
    if path.exists():
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                if "messages" in data:
                    records.append({"messages": data["messages"]})
                else:
                    # Construct message from theorem + proof
                    imports = data.get("imports", "import Mathlib\n")
                    user_prompt = format_lean4_prompt(data["theorem"], imports=imports)
                    think_cot = data.get("think", "We use standard induction or simplification.")
                    proof_code = data["proof"]
                    assistant_response = f"<think>\n{think_cot}\n</think>\n\n```lean4\n{proof_code}\n```"
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": assistant_response}
                    ]
                    records.append({"messages": messages})
                if max_samples and len(records) >= max_samples:
                    break
        dataset = Dataset.from_list(records)
    else:
        # Load from HuggingFace hub
        dataset = load_dataset(file_path_or_name, split="train")
        if max_samples:
            dataset = dataset.select(range(min(len(dataset), max_samples)))
            
    return dataset

def load_lean_rl_dataset(
    file_path_or_name: str,
    tokenizer: Any = None,
    max_samples: Optional[int] = None
) -> Dataset:
    """
    Load RL dataset containing prompts for theorem proving.
    Expected record formats:
    1) {"prompt": "..."} or {"messages": [...]}
    2) {"theorem": "theorem ...", "imports": "import Mathlib\n"}
    """
    path = Path(file_path_or_name)
    if path.exists():
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                imports = data.get("imports", "import Mathlib\n")
                if "prompt" in data:
                    prompt_str = data["prompt"]
                elif "messages" in data:
                    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
                        prompt_str = tokenizer.apply_chat_template(
                            data["messages"], tokenize=False, add_generation_prompt=True
                        )
                    else:
                        prompt_str = data["messages"][-1]["content"]
                else:
                    user_prompt = format_lean4_prompt(data["theorem"], imports=imports)
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ]
                    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
                        prompt_str = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                    else:
                        prompt_str = f"{SYSTEM_PROMPT}\n\nUser: {user_prompt}\nAssistant: "
                
                records.append({
                    "prompt": prompt_str,
                    "theorem": data.get("theorem", ""),
                    "imports": imports
                })
                if max_samples and len(records) >= max_samples:
                    break
        dataset = Dataset.from_list(records)
    else:
        dataset = load_dataset(file_path_or_name, split="train")
        if max_samples:
            dataset = dataset.select(range(min(len(dataset), max_samples)))
            
    return dataset
