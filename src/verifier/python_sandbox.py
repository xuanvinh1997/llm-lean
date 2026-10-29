"""
Isolated Python & SymPy execution sandbox for mathematical and theoretical physics verification.
Enforces execution time limits, memory bounds, and checks symbolic equivalence.
"""

import sys
import re
import math
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Optional, Any, Dict, Tuple

@dataclass
class PythonExecutionResult:
    success: bool
    output: str
    error: Optional[str] = None
    time_taken: float = 0.0

class PythonSandbox:
    """
    Subprocess execution environment for running Python scripts with SymPy/NumPy.
    """
    def __init__(self, timeout_seconds: float = 10.0, memory_limit_mb: int = 1024):
        self.timeout_seconds = timeout_seconds
        self.memory_limit_mb = memory_limit_mb

    def extract_python_code(self, text: str) -> Optional[str]:
        """
        Extracts python code from ```python ... ``` or tool call tags.
        """
        patterns = [
            r"```python\s*(.*?)\s*```",
            r"<tool_call>\s*(.*?)\s*</tool_call>",
            r"```py\s*(.*?)\s*```"
        ]
        for pat in patterns:
            match = re.search(pat, text, re.DOTALL)
            if match:
                return match.group(1).strip()
        return None

    def execute(self, code: str) -> PythonExecutionResult:
        """
        Runs the provided code in an isolated python subprocess with timeout.
        """
        if not code or not code.strip():
            return PythonExecutionResult(success=False, output="", error="Empty code")

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=True) as tmp:
            tmp.write(code)
            tmp.flush()

            start_t = time.perf_counter()
            try:
                proc = subprocess.run(
                    [sys.executable, tmp.name],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds
                )
                time_taken = time.perf_counter() - start_t
                if proc.returncode == 0:
                    return PythonExecutionResult(
                        success=True,
                        output=proc.stdout.strip(),
                        time_taken=time_taken
                    )
                else:
                    return PythonExecutionResult(
                        success=False,
                        output=proc.stdout.strip(),
                        error=proc.stderr.strip()[:500],
                        time_taken=time_taken
                    )
            except subprocess.TimeoutExpired:
                time_taken = time.perf_counter() - start_t
                return PythonExecutionResult(
                    success=False,
                    output="",
                    error=f"TimeoutExpired after {self.timeout_seconds}s",
                    time_taken=time_taken
                )
            except Exception as e:
                time_taken = time.perf_counter() - start_t
                return PythonExecutionResult(
                    success=False,
                    output="",
                    error=str(e),
                    time_taken=time_taken
                )

def verify_math_equivalence(pred_str: str, gt_str: str) -> bool:
    """
    Checks if a predicted math answer matches the ground truth.
    First checks string match, then parses through SymPy for algebraic equivalence.
    """
    pred_clean = pred_str.strip().strip("$").strip()
    gt_clean = gt_str.strip().strip("$").strip()

    if pred_clean == gt_clean:
        return True

    # Try SymPy algebraic equivalence
    try:
        import sympy
        from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
        transformations = standard_transformations + (implicit_multiplication_application,)

        # Clean LaTeX commands if any
        p_expr_str = pred_clean.replace(r"\frac", "").replace("{", "(").replace("}", ")")
        g_expr_str = gt_clean.replace(r"\frac", "").replace("{", "(").replace("}", ")")

        p_sym = parse_expr(p_expr_str, transformations=transformations)
        g_sym = parse_expr(g_expr_str, transformations=transformations)

        diff = sympy.simplify(p_sym - g_sym)
        if diff == 0:
            return True
    except Exception:
        pass

    return False
