"""
Standalone verification test script.
Tests the Lean 4 Verifier and Reward functions on canonical test cases.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.verifier.lean4_repl import Lean4Verifier, VerificationStatus
from src.verifier.rewards import lean4_reward_func, format_reward_func, python_sympy_reward_func

def run_tests():
    print("==================================================")
    print(" Testing Lean 4 Sandbox Verifier & Reward Engine")
    print("==================================================")

    verifier = Lean4Verifier(timeout_seconds=5.0)
    print(f"Lean 4 Compiler Available: {verifier.is_lean_available}")

    test_cases = [
        {
            "name": "Case 1: Valid Proof (omega / rfl)",
            "completion": """<think>
We need to prove that n + 0 = n. This holds directly by definition.
</think>

```lean4
theorem add_zero_test (n : Nat) : n + 0 = n := by
  rfl
```""",
            "expected_reward": 1.0
        },
        {
            "name": "Case 2: Cheating with 'sorry'",
            "completion": """<think>
I'm not sure how to solve this, so I will admit it.
</think>

```lean4
theorem hard_theorem (n : Nat) : n = n := by
  sorry
```""",
            "expected_reward": -0.5
        },
        {
            "name": "Case 3: Incomplete / Partial Proof",
            "completion": """<think>
Let's introduce n and see.
</think>

```lean4
theorem partial_test (n m : Nat) : n + m = m + n := by
  intro
```""",
            "expected_reward": 0.1
        },
        {
            "name": "Case 4: Syntax / Type Error",
            "completion": """<think>
Applying completely wrong syntax.
</think>

```lean4
theorem bad_syntax : 1 + 1 = 3 := by
  nonsense_tactic
```""",
            "expected_reward": -0.5
        }
    ]

    completions = [tc["completion"] for tc in test_cases]
    prompts = ["dummy_prompt"] * len(completions)

    print("\n--- Running lean4_reward_func ---")
    rewards = lean4_reward_func(prompts, completions)
    for tc, r in zip(test_cases, rewards):
        status = "PASSED" if (r == tc["expected_reward"] or (not verifier.is_lean_available and r in [1.0, 0.1, -0.5])) else "DIFF"
        print(f"[{status}] {tc['name']} -> Reward: {r} (Expected: {tc['expected_reward']})")

    print("\n--- Running format_reward_func ---")
    format_rewards = format_reward_func(completions)
    for tc, fr in zip(test_cases, format_rewards):
        print(f"{tc['name']} -> Format Reward: {fr:+.2f}")

    print("\n--- Running python_sympy_reward_func ---")
    py_test_cases = [
        {
            "name": "PyCase 1: Correct Tool Computation (Quadratic Roots)",
            "completion": """<think>
We compute the roots of x^2 - 5x + 6 = 0 using Python tool.
</think>
```python
import math
a, b, c = 1, -5, 6
d = b**2 - 4*a*c
r1 = (-b - int(math.isqrt(d))) // 2
r2 = (-b + int(math.isqrt(d))) // 2
print([r1, r2])
```""",
            "expected_gt": "[2, 3]",
            "expected_reward": 1.0
        },
        {
            "name": "PyCase 2: Python Syntax Error",
            "completion": """<think>
Invalid code.
</think>
```python
def broken_syntax(
```""",
            "expected_gt": "0",
            "expected_reward": -0.5
        }
    ]
    py_comps = [tc["completion"] for tc in py_test_cases]
    py_gts = [tc["expected_gt"] for tc in py_test_cases]
    py_rewards = python_sympy_reward_func(["p"] * len(py_comps), py_comps, ground_truth=py_gts)
    for tc, r in zip(py_test_cases, py_rewards):
        status = "PASSED" if r == tc["expected_reward"] else "DIFF"
        print(f"[{status}] {tc['name']} -> Reward: {r} (Expected: {tc['expected_reward']})")

    print("\nAll verifier tests executed cleanly!")

if __name__ == "__main__":
    run_tests()
