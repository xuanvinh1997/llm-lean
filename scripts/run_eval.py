"""
Held-out evaluation harness. Talks to any OpenAI-compatible server (e.g. `vllm serve`,
with or without a LoRA adapter), so it needs no torch/vllm in this environment.

Tasks:
  math : boxed final answer vs ground truth. --tools on lets the model run ```python blocks
         (TIR loop); compare on vs off to see whether tool use actually helps.
  lean : full proof compiled with Lean 4. --repair_turns N feeds compiler errors back,
         reporting pass@k after 0..N repair rounds.

Examples:
  python scripts/run_eval.py --task math --file data/eval/aime-2026_train.jsonl \
      --endpoint http://localhost:8000/v1 --model qwen-lean -n 8 --tools on
  python scripts/run_eval.py --task lean --file data/eval/minif2f-lean4_test.jsonl \
      --endpoint http://localhost:8000/v1 --model qwen-lean -n 8 --repair_turns 3
  python scripts/run_eval.py --task math --file X.jsonl --backend mock   # plumbing smoke test
"""

import re
import sys
import json
import time
import argparse
import urllib.request
from math import comb
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.prompts import SYSTEM_PROMPT as LEAN_SYSTEM_PROMPT
from src.verifier.lean4_repl import Lean4Verifier, VerificationStatus
from src.verifier.python_sandbox import PythonSandbox

MATH_SYSTEM_PROMPT = (
    "You are an expert mathematician. Think step by step inside <think>...</think>, then give the "
    "final answer as \\boxed{...}."
)
TOOL_HINT = (
    " You may run Python/SymPy by writing a ```python block; its output will be returned to you. "
    "Stop writing code once you are ready to answer."
)


# ---------------------------------------------------------------- metrics
def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator from the Codex paper."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


# ---------------------------------------------------------------- answers
def last_boxed(text: str):
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None
    i, depth, start = idx + len("\\boxed{"), 1, idx + len("\\boxed{")
    while i < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    return text[start:i - 1].strip() if depth == 0 else None


def _latex_to_expr(s: str) -> str:
    s = s.strip().strip("$").replace("\\left", "").replace("\\right", "").replace("\\!", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("^\\circ", "").replace("\\%", "")
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    for _ in range(5):  # nested fractions / roots
        s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
        s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\pi", "pi").replace("^", "**")
    s = s.replace("{", "(").replace("}", ")").replace(",", "") if re.fullmatch(r"[\d,]+", s) is None else s.replace(",", "")
    return s


def answers_match(pred, gt) -> bool:
    """Exact string, then SymPy on a proper LaTeX->expr conversion. Does NOT reuse
    verify_math_equivalence, which strips \\frac and so treats \\frac{1}{2} as 2."""
    if pred is None:
        return False
    norm = lambda x: re.sub(r"\s+", "", str(x)).strip("$")
    if norm(pred) == norm(gt):
        return True
    try:
        import sympy
        from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
        tf = standard_transformations + (implicit_multiplication_application,)
        diff = sympy.simplify(parse_expr(_latex_to_expr(pred), transformations=tf)
                              - parse_expr(_latex_to_expr(str(gt)), transformations=tf))
        return diff == 0
    except Exception:
        return False


# ---------------------------------------------------------------- model backends
class OpenAIChat:
    def __init__(self, endpoint, model, temperature, top_p, max_tokens, api_key="EMPTY", timeout=1800):
        self.url = endpoint.rstrip("/") + "/chat/completions"
        self.body = dict(model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        self.timeout = timeout

    def __call__(self, messages) -> str:
        req = urllib.request.Request(self.url, json.dumps({**self.body, "messages": messages}).encode(), self.headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)["choices"][0]["message"]["content"]


class MockChat:
    """Deterministic fake model for plumbing tests. Its numbers say nothing about any real model."""

    def __call__(self, messages) -> str:
        last = messages[-1]["content"]
        if "```lean4" in messages[0]["content"] + messages[1]["content"]:
            if len(messages) > 3:  # after a repair request, 'fix' it
                return "<think>x</think>\n```lean4\ntheorem t : 1 + 1 = 2 := by\n  rfl\n```"
            return "<think>x</think>\n```lean4\ntheorem t : 1 + 1 = 2 := by\n  nonsense_tactic\n```"
        if "Output:" not in last and "python" in messages[0]["content"]:
            return "```python\nprint(6*7)\n```"
        return "<think>x</think> The answer is \\boxed{42}."


# ---------------------------------------------------------------- episodes
def run_math(row, chat, args, python: PythonSandbox):
    problem = row.get("problem") or row["question"]
    system = MATH_SYSTEM_PROMPT + (TOOL_HINT if args.tools == "on" else "")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": problem}]
    tool_calls = 0
    reply = ""
    for _ in range(args.max_turns if args.tools == "on" else 1):
        reply = chat(messages)
        code = python.extract_python_code(reply) if args.tools == "on" else None
        if not code or last_boxed(reply) is not None:
            break
        res = python.execute(code)
        tool_calls += 1
        messages += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": f"Output:\n{res.output or res.error}"}]
    pred = last_boxed(reply)
    return {"correct": answers_match(pred, row["answer"]), "pred": pred, "tool_calls": tool_calls, "reply": reply}


def statement_preserved(code: str, statement: str) -> bool:
    """The proof must still prove the given statement (whitespace-insensitive prefix match)."""
    stmt = re.sub(r"\s*:=\s*(by)?\s*(sorry)?\s*$", "", statement.strip())
    squash = lambda s: re.sub(r"\s+", " ", s)
    return squash(stmt) in squash(code)


def run_lean(row, chat, args, verifier: Lean4Verifier):
    header, stmt = row.get("header", "import Mathlib\n"), row["formal_statement"]
    user = ("Please provide a complete formal proof in Lean 4 for the following statement:\n\n"
            f"```lean4\n{header.strip()}\n\n{stmt.strip()}\n```")
    messages = [{"role": "system", "content": LEAN_SYSTEM_PROMPT}, {"role": "user", "content": user}]
    solved_at = None
    for turn in range(args.repair_turns + 1):
        reply = chat(messages)
        code = verifier.extract_lean_code(reply) or ""
        if not statement_preserved(code, stmt):
            status, feedback = "STATEMENT_CHANGED", "The theorem statement must not be changed."
        else:
            res = verifier.verify_code(code, imports=header)
            status = res.status.value
            feedback = res.error or res.message
        if status == VerificationStatus.PROVED.value:
            solved_at = turn
            break
        messages += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": f"The Lean compiler rejected this ({status}):\n{feedback}\nPlease fix the proof."}]
    return {"solved_at": solved_at, "correct": solved_at is not None, "turns_used": turn + 1}


# ---------------------------------------------------------------- driver
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=["math", "lean"], required=True)
    p.add_argument("--file", required=True, help="jsonl from download_datasets.py --download_eval")
    p.add_argument("--backend", choices=["openai", "mock"], default="openai")
    p.add_argument("--endpoint", default="http://localhost:8000/v1")
    p.add_argument("--model", default="default")
    p.add_argument("-n", type=int, default=8, help="samples per problem")
    p.add_argument("--k", type=int, nargs="+", default=[1, 4, 8])
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=6144)
    p.add_argument("--tools", choices=["on", "off"], default="off", help="math: allow ```python execution")
    p.add_argument("--max_turns", type=int, default=4, help="math+tools: max model turns")
    p.add_argument("--repair_turns", type=int, default=0, help="lean: extra rounds after compiler feedback")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--lean_dir", default=None, help="lake project dir with Mathlib")
    p.add_argument("--allow_mock_lean", action="store_true",
                   help="allow the repo's fake Lean verifier. Results are meaningless; plumbing tests only.")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    rows = [json.loads(l) for l in open(args.file, encoding="utf-8")][: args.limit]
    chat = MockChat() if args.backend == "mock" else OpenAIChat(
        args.endpoint, args.model, args.temperature, args.top_p, args.max_tokens)

    if args.task == "lean":
        env = Lean4Verifier(lake_dir=args.lean_dir)
        if not env.is_lean_available and not args.allow_mock_lean:
            sys.exit("Lean 4 not found. Install it (scripts/setup_lean4.sh) or pass --allow_mock_lean for a plumbing test.")
        episode = lambda r: run_lean(r, chat, args, env)
    else:
        env = PythonSandbox()
        episode = lambda r: run_math(r, chat, args, env)

    jobs = [(i, s) for i in range(len(rows)) for s in range(args.n)]
    t0 = time.time()
    with ThreadPoolExecutor(args.concurrency) as pool:
        results = list(pool.map(lambda j: (j[0], episode(rows[j[0]])), jobs))
    per_problem = {}
    for i, r in results:
        per_problem.setdefault(i, []).append(r)

    summary = {"task": args.task, "file": args.file, "model": args.model, "n": args.n,
               "problems": len(rows), "tools": args.tools, "repair_turns": args.repair_turns,
               "mock_backend": args.backend == "mock",
               "mock_lean": args.task == "lean" and not env.is_lean_available, "seconds": round(time.time() - t0, 1)}
    for k in [k for k in args.k if k <= args.n]:
        summary[f"pass@{k}"] = round(sum(pass_at_k(args.n, sum(r["correct"] for r in rs), k)
                                         for rs in per_problem.values()) / len(rows), 4)
    if args.task == "math" and args.tools == "on":
        summary["avg_tool_calls"] = round(sum(r["tool_calls"] for _, r in results) / len(results), 2)
    if args.task == "lean":  # pass@1-style rate by repair round
        for t in range(args.repair_turns + 1):
            summary[f"solved_within_{t}_repairs"] = round(
                sum(r["solved_at"] is not None and r["solved_at"] <= t for _, r in results) / len(results), 4)
    if args.task == "lean" and not env.is_lean_available:
        summary["WARNING"] = "Mock Lean verifier used. Numbers are NOT real."

    out = Path(args.output or f"outputs/eval/{Path(args.file).stem}_{args.task}_{args.model.replace('/', '_')}"
                              f"_tools-{args.tools}_rep{args.repair_turns}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "samples": [{"idx": i, **r} for i, r in results]},
                              ensure_ascii=False, indent=1))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[✓] per-sample results: {out}")


if __name__ == "__main__":
    main()
