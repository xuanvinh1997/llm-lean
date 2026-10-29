"""
System prompts and formatting helpers for Lean 4 theorem proving.
Encourages long CoT reasoning inside `<think>...</think>` tags followed
by Lean 4 code blocks.
"""

from typing import Dict, Any, List

SYSTEM_PROMPT = """You are an expert mathematician and Lean 4 formalizer.
Your task is to prove the given theorem in Lean 4 (using Mathlib).

Guidelines:
1. First, think carefully about the mathematical argument and proof strategy inside <think>...</think> tags.
2. Outline the necessary lemmas, tactics (e.g. `intro`, `apply`, `rw`, `ring`, `linarith`, `omega`, `induction`), and step-by-step transformations.
3. Finally, provide the complete, self-contained Lean 4 proof enclosed in a ```lean4 code block.
4. Your code must be syntactically valid Lean 4, use correct Mathlib imports, and NEVER use `sorry` or `admit`.
"""

def format_lean4_prompt(theorem_header: str, imports: str = "import Mathlib\n") -> str:
    """
    Formats the user prompt presenting the formal theorem to be proved.
    """
    user_content = (
        f"Please provide a complete formal proof in Lean 4 for the following statement:\n\n"
        f"```lean4\n"
        f"{imports.strip()}\n\n"
        f"{theorem_header.strip()}\n"
        f"```"
    )
    return user_content

def build_chat_messages(theorem_header: str, imports: str = "import Mathlib\n") -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_lean4_prompt(theorem_header, imports)},
    ]
