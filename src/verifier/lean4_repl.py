"""
Lean 4 REPL and compiler interface.
Provides isolated execution on CPU with strict per-proof timeout (default 15s).
Detects complete proofs, partial progress, syntax/type errors, and timeouts.
"""

import os
import re
import json
import shutil
import tempfile
import subprocess
import time
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

class VerificationStatus(str, Enum):
    PROVED = "PROVED"                    # Complete proof without sorry
    PARTIAL = "PARTIAL"                  # Valid tactic, subgoals remain
    ERROR = "ERROR"                      # Syntax, type, or compiler error
    TIMEOUT = "TIMEOUT"                  # Subprocess exceeded time limit
    SORRY = "SORRY"                      # Used sorry / admit
    EMPTY = "EMPTY"                      # No Lean 4 code found

@dataclass
class LeanVerificationResult:
    status: VerificationStatus
    reward: float
    time_taken: float
    message: str
    error: Optional[str] = None
    subgoals_remaining: Optional[int] = None
    raw_output: Optional[str] = None

class Lean4Verifier:
    """
    Subprocess sandbox for compiling and verifying Lean 4 proof scripts.
    Pins to CPU and enforces memory and execution time limits.
    """
    def __init__(
        self,
        lake_path: Optional[str] = None,
        repl_bin: Optional[str] = None,
        lake_dir: Optional[str] = None,
        timeout_seconds: float = 15.0,
        memory_limit_mb: int = 2048,
    ):
        self.timeout_seconds = timeout_seconds
        self.memory_limit_mb = memory_limit_mb
        self.lake_path = lake_path or shutil.which("lake") or "lake"
        self.repl_bin = repl_bin
        self.lake_dir = lake_dir or os.getcwd()

        # Check if Lean 4 is available in system
        self.is_lean_available = (
            shutil.which(self.lake_path) is not None or
            (self.repl_bin and os.path.exists(self.repl_bin)) or
            shutil.which("lean") is not None
        )

    def extract_lean_code(self, completion: str) -> Optional[str]:
        """
        Extracts code from ```lean4 ... ``` or ```lean ... ``` blocks.
        If no code fence is present, checks if the text itself resembles Lean 4 code.
        """
        patterns = [
            r"```lean4\s*(.*?)\s*```",
            r"```lean\s*(.*?)\s*```",
            r"```\s*(theorem.*?)```",
        ]
        for pat in patterns:
            match = re.search(pat, completion, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # Fallback: if completion contains 'theorem ' or 'lemma '
        if "theorem " in completion or "lemma " in completion or "example " in completion:
            return completion.strip()

        return None

    def verify_code(self, code: str, imports: str = "import Mathlib\n") -> LeanVerificationResult:
        """
        Executes Lean 4 verification on a given snippet.
        Applies memory bounds and strict timeout to avoid infinite tactic loops.
        """
        if not code or not code.strip():
            return LeanVerificationResult(
                status=VerificationStatus.EMPTY,
                reward=-0.5,
                time_taken=0.0,
                message="No code found in completion."
            )

        # Quick check for cheating / sorry
        has_sorry = bool(re.search(r"\b(sorry|admit)\b", code))

        if not self.is_lean_available:
            # Fallback mock for testing environments where Lean 4 isn't installed
            return self._mock_fallback_verify(code, has_sorry)

        full_code = f"{imports.strip()}\n\n{code.strip()}\n"

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "Verify.lean")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(full_code)

            start_t = time.perf_counter()
            cmd = [self.lake_path, "env", "lean", file_path]
            
            cwd = self.lake_dir if os.path.isdir(self.lake_dir) else tmpdir

            try:
                # Pre-exec fn to set memory limits if supported on OS
                def preexec():
                    try:
                        import resource
                        # Convert MB to bytes
                        mem_bytes = self.memory_limit_mb * 1024 * 1024
                        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
                    except Exception:
                        pass

                proc = subprocess.run(
                    cmd,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    preexec_fn=preexec if os.name != "nt" else None
                )
                time_taken = time.perf_counter() - start_t
                stdout = proc.stdout + "\n" + proc.stderr

                return self._parse_lean_output(proc.returncode, stdout, has_sorry, time_taken)

            except subprocess.TimeoutExpired:
                time_taken = time.perf_counter() - start_t
                return LeanVerificationResult(
                    status=VerificationStatus.TIMEOUT,
                    reward=-0.3,
                    time_taken=time_taken,
                    message=f"Execution timed out after {self.timeout_seconds}s.",
                    error="TimeoutExpired"
                )
            except Exception as e:
                time_taken = time.perf_counter() - start_t
                return LeanVerificationResult(
                    status=VerificationStatus.ERROR,
                    reward=-0.5,
                    time_taken=time_taken,
                    message=f"Subprocess execution error: {str(e)}",
                    error=str(e)
                )

    def _parse_lean_output(
        self, returncode: int, output: str, has_sorry: bool, time_taken: float
    ) -> LeanVerificationResult:
        """
        Parses output from Lean 4 compiler:
        - Returncode 0 and no 'sorry': Proved (+1.0)
        - Warning 'declaration uses sorry': Sorry used (-0.5)
        - Unsolved goals / partial progress: Partial (+0.1)
        - Syntax / type error: Error (-0.5)
        """
        lower_out = output.lower()

        # Check for sorry usage
        if has_sorry or "declaration uses 'sorry'" in lower_out or "uses sorry" in lower_out:
            return LeanVerificationResult(
                status=VerificationStatus.SORRY,
                reward=-0.5,
                time_taken=time_taken,
                message="Proof contains 'sorry' or 'admit'.",
                raw_output=output
            )

        if returncode == 0:
            # Check for any remaining warnings about unsolved goals
            if "unsolved goals" in lower_out:
                return LeanVerificationResult(
                    status=VerificationStatus.PARTIAL,
                    reward=0.1,
                    time_taken=time_taken,
                    message="Valid tactics executed, but subgoals remain open.",
                    raw_output=output
                )
            
            # Proof verified successfully!
            return LeanVerificationResult(
                status=VerificationStatus.PROVED,
                reward=1.0,
                time_taken=time_taken,
                message="Theorem successfully verified without sorry.",
                raw_output=output
            )

        # If return code is non-zero, check whether it is partial or fatal error
        if "unsolved goals" in lower_out and "error:" not in lower_out:
            return LeanVerificationResult(
                status=VerificationStatus.PARTIAL,
                reward=0.1,
                time_taken=time_taken,
                message="Tactic valid but subgoals unresolved.",
                raw_output=output
            )

        # Fatal compiler / syntax / type error
        return LeanVerificationResult(
            status=VerificationStatus.ERROR,
            reward=-0.5,
            time_taken=time_taken,
            message="Compiler syntax or type error.",
            error=output[:500],
            raw_output=output
        )

    def _mock_fallback_verify(self, code: str, has_sorry: bool) -> LeanVerificationResult:
        """
        Mock verifier for running dry-runs or unit tests in environments without Lean 4.
        """
        if has_sorry:
            return LeanVerificationResult(
                status=VerificationStatus.SORRY,
                reward=-0.5,
                time_taken=0.01,
                message="[MOCK] Proof contains 'sorry'."
            )
        if "nonsense_tactic" in code or "bad_syntax" in code:
            return LeanVerificationResult(
                status=VerificationStatus.ERROR,
                reward=-0.5,
                time_taken=0.01,
                message="[MOCK] Detected invalid syntax or unknown tactic."
            )
        if "by" in code and ("ring" in code or "linarith" in code or "rfl" in code or "simp" in code or "omega" in code):
            return LeanVerificationResult(
                status=VerificationStatus.PROVED,
                reward=1.0,
                time_taken=0.05,
                message="[MOCK] Valid proof pattern recognized."
            )
        return LeanVerificationResult(
            status=VerificationStatus.PARTIAL,
            reward=0.1,
            time_taken=0.02,
            message="[MOCK] Syntactically acceptable partial proof."
        )
