"""
Sandbox Manager for parallel verification of Lean 4 completions across CPU cores.
Prevents GPU starvation by utilizing multi-core CPU workers with time-limits.
"""

import os
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.verifier.lean4_repl import Lean4Verifier, LeanVerificationResult

_global_verifier: Optional[Lean4Verifier] = None

def _init_verifier_worker(lake_path: Optional[str], repl_bin: Optional[str], lake_dir: Optional[str], timeout: float, mem_limit: int):
    global _global_verifier
    _global_verifier = Lean4Verifier(
        lake_path=lake_path,
        repl_bin=repl_bin,
        lake_dir=lake_dir,
        timeout_seconds=timeout,
        memory_limit_mb=mem_limit
    )

def _verify_single_task(args) -> LeanVerificationResult:
    global _global_verifier
    code, imports = args
    if _global_verifier is None:
        _global_verifier = Lean4Verifier()
    return _global_verifier.verify_code(code, imports=imports)

class SandboxManager:
    """
    Manages a pool of CPU worker processes to verify Lean 4 theorems in parallel.
    """
    def __init__(
        self,
        max_workers: int = 16,
        lake_path: Optional[str] = None,
        repl_bin: Optional[str] = None,
        lake_dir: Optional[str] = None,
        timeout_seconds: float = 15.0,
        memory_limit_mb: int = 2048,
    ):
        self.max_workers = min(max_workers, max(1, os.cpu_count() or 4))
        self.lake_path = lake_path
        self.repl_bin = repl_bin
        self.lake_dir = lake_dir
        self.timeout_seconds = timeout_seconds
        self.memory_limit_mb = memory_limit_mb
        self.verifier = Lean4Verifier(
            lake_path=lake_path,
            repl_bin=repl_bin,
            lake_dir=lake_dir,
            timeout_seconds=timeout_seconds,
            memory_limit_mb=memory_limit_mb
        )

    def extract_code(self, completion: str) -> Optional[str]:
        return self.verifier.extract_lean_code(completion)

    def verify_batch(
        self,
        codes: List[str],
        imports_list: Optional[List[str]] = None
    ) -> List[LeanVerificationResult]:
        """
        Verify a batch of Lean 4 snippets in parallel across CPU workers.
        """
        if not codes:
            return []

        if imports_list is None:
            imports_list = ["import Mathlib\n"] * len(codes)

        tasks = list(zip(codes, imports_list))

        if len(tasks) == 1:
            return [self.verifier.verify_code(tasks[0][0], tasks[0][1])]

        results = [None] * len(tasks)
        with ProcessPoolExecutor(
            max_workers=self.max_workers,
            initializer=_init_verifier_worker,
            initargs=(self.lake_path, self.repl_bin, self.lake_dir, self.timeout_seconds, self.memory_limit_mb)
        ) as executor:
            future_to_idx = {executor.submit(_verify_single_task, task): idx for idx, task in enumerate(tasks)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    res = future.result()
                except Exception as e:
                    res = LeanVerificationResult(
                        status=False,
                        reward=-0.5,
                        time_taken=0.0,
                        message=f"Worker failure: {str(e)}",
                        error=str(e)
                    )
                results[idx] = res

        return results
