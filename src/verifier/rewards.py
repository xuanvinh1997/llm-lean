"""
Multi-Environment Reward Engine for RLVR / GRPO.
Supports:
  1. Lean 4 Formal Compiler Sandbox (Ground-truth Lean 4 verification)
  2. Python & SymPy Sandbox (Tool-use & algebraic/numerical correctness)
  3. Format & Thinking Tag Compliance (Chain-of-Thought integrity)
"""

import re
from typing import List, Dict, Any, Optional
from src.verifier.sandbox_manager import SandboxManager
from src.verifier.python_sandbox import PythonSandbox, verify_math_equivalence

# Global singletons
_sandbox: Optional[SandboxManager] = None
_python_sandbox: Optional[PythonSandbox] = None

def get_sandbox(
    max_workers: int = 16,
    lake_path: Optional[str] = None,
    repl_bin: Optional[str] = None,
    lake_dir: Optional[str] = None,
    timeout_seconds: float = 15.0
) -> SandboxManager:
    global _sandbox
    if _sandbox is None:
        _sandbox = SandboxManager(
            max_workers=max_workers,
            lake_path=lake_path,
            repl_bin=repl_bin,
            lake_dir=lake_dir,
            timeout_seconds=timeout_seconds
        )
    return _sandbox

def get_python_sandbox(timeout_seconds: float = 10.0) -> PythonSandbox:
    global _python_sandbox
    if _python_sandbox is None:
        _python_sandbox = PythonSandbox(timeout_seconds=timeout_seconds)
    return _python_sandbox

def lean4_reward_func(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """
    Lean 4 Ground-truth verifier reward:
      +1.0 : Fully verified theorem without sorry (R_compiler)
      +0.1 : Valid tactic sequence, subgoals remain (R_progress)
      -0.5 : Compiler syntax/type crash or contains sorry (R_error)
      -0.3 : Subprocess execution timeout (> 15s)
    """
    sandbox = get_sandbox()
    imports_list = kwargs.get("imports", ["import Mathlib\n"] * len(completions))
    if isinstance(imports_list, str):
        imports_list = [imports_list] * len(completions)

    codes = [sandbox.extract_code(comp) or "" for comp in completions]
    results = sandbox.verify_batch(codes, imports_list=imports_list)
    return [res.reward for res in results]

def python_sympy_reward_func(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """
    Python & SymPy tool execution and math correctness verifier:
      +1.0 : Python code executed cleanly AND final answer is symbolically/numerically equivalent
      +0.2 : Python code executed without error (intermediate tool call valid)
      -0.5 : Python syntax error, uncaught exception
      -0.3 : Execution timeout
      -0.2 : Executed cleanly but wrong final answer
    """
    py_sandbox = get_python_sandbox()
    ground_truths = kwargs.get("answer", kwargs.get("ground_truth", [None] * len(completions)))
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths] * len(completions)

    rewards = []
    for comp, gt in zip(completions, ground_truths):
        code = py_sandbox.extract_python_code(comp)
        if not code:
            rewards.append(-0.2)
            continue

        exec_res = py_sandbox.execute(code)
        if not exec_res.success:
            if "TimeoutExpired" in (exec_res.error or ""):
                rewards.append(-0.3)
            else:
                rewards.append(-0.5)
            continue

        # If ground truth answer is provided, verify mathematical equivalence
        if gt:
            # Check if output matches ground truth
            output = exec_res.output.split("\n")[-1].strip()
            if verify_math_equivalence(output, str(gt)):
                rewards.append(1.0)
            else:
                rewards.append(-0.2)
        else:
            # Intermediate tool-call reward for clean execution
            rewards.append(0.2)

    return rewards

def format_reward_func(completions: List[str], **kwargs) -> List[float]:
    """
    Strict formatting and reasoning structure reward:
      +0.1 : Well-formed <think>...</think> with substantial reasoning (> 20 chars)
      -0.2 : Missing or broken <think> tags
      +0.1 : Code cleanly enclosed in ```lean4 or ```python blocks
      -0.1 : Raw unstructured generation
    """
    rewards = []
    think_pat = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    code_pat = re.compile(r"```(lean4|lean|python|py)\s*(.*?)\s*```", re.DOTALL)

    for comp in completions:
        score = 0.0
        # Check thinking block
        think_match = think_pat.search(comp)
        if think_match and len(think_match.group(1).strip()) > 20:
            score += 0.1
        else:
            score -= 0.2

        # Check code fence
        if code_pat.search(comp):
            score += 0.1
        else:
            score -= 0.1

        rewards.append(round(score, 2))

    return rewards
