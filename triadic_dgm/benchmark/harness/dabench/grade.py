"""DABench grading.

DABench answers follow the *format-prompting* schema: the program prints one or
more `@field_name[value]` tokens. The gold is `common_answers` = a list of
[field, value] pairs. A question passes iff every gold field is present in the
agent's stdout with a matching value.

Matching policy (per field):
  1. numeric compare with tolerance (|a-b| <= 0.02, or relative <= 1%), so float
     rounding and benign computation-order differences don't fail a correct answer;
  2. else case-insensitive exact string compare.
Falls back to an LLM judge only when NO @field[value] token is parsed from stdout
(agent ignored the format) but stdout is non-empty.
"""
import re
from typing import Any, Dict, List, Optional

_FIELD_RE = re.compile(r"@([A-Za-z0-9_]+)\s*\[\s*([^\]]*?)\s*\]")


def parse_answer(stdout: str) -> Dict[str, str]:
    return {m.group(1): m.group(2).strip() for m in _FIELD_RE.finditer(stdout or "")}


def _numeric_match(pred: str, gold: str, abs_tol: float = 0.02, rel_tol: float = 0.01) -> Optional[bool]:
    """Return True/False for a numeric comparison, or None if either side is non-numeric."""
    try:
        fp, fg = float(pred), float(gold)
    except (ValueError, TypeError):
        return None
    if abs(fp - fg) <= abs_tol:
        return True
    denom = max(abs(fg), 1e-9)
    return abs(fp - fg) / denom <= rel_tol


def _field_match(pred: str, gold: str) -> bool:
    n = _numeric_match(pred, gold)
    if n is not None:
        return n
    return str(pred).strip().lower() == str(gold).strip().lower()


def _llm_judge(llm_client, question: str, gold, stdout: str) -> bool:
    """Ask the model whether the agent's free-form stdout answers the question correctly
    relative to the gold. Used only when format-prompting parsing finds nothing."""
    gold_str = ", ".join(f"{f}={v}" for f, v in gold)
    prompt = (
        "You are a strict grader for a data-analysis question.\n\n"
        f"Question: {question}\n"
        f"Reference correct answer(s): {gold_str}\n\n"
        f"Agent's printed answer (stdout):\n```\n{(stdout or '').strip()[:2000]}\n```\n\n"
        "Does the agent's answer match the reference (numerically equal within small "
        "rounding tolerance, or semantically the same)? Reply with EXACTLY one word: "
        "'CORRECT' or 'INCORRECT'."
    )
    try:
        out = (llm_client.generate(prompt, temperature=0.0) or "").strip().upper()
    except Exception:  # noqa: BLE001
        return False
    return out.startswith("CORRECT")


def grade_one(result: Dict[str, Any], gold: List[List[str]], llm_client=None) -> Dict[str, Any]:
    stdout = result.get("stdout", "") or ""
    parsed = parse_answer(stdout)
    if (not parsed) and stdout.strip() and gold and llm_client is not None:
        ok = _llm_judge(llm_client, result.get("question", ""), gold, stdout)
        return {"success": bool(ok), "method": "llm_judge", "parsed": {}, "details": []}

    details = []
    all_ok = bool(gold)
    for pair in gold:
        field, gval = pair[0], pair[1]
        pred = parsed.get(field)
        ok = False
        if pred is not None:
            ok = _field_match(pred, gval)
        details.append({"field": field, "gold": gval, "pred": pred, "ok": ok})
        all_ok = all_ok and ok
    return {"success": bool(all_ok), "method": "format_prompt", "parsed": parsed, "details": details}


def grade_results(results: List[Dict[str, Any]], llm_client=None) -> List[Dict[str, Any]]:
    graded = []
    for r in results:
        g = grade_one(r, r.get("gold", []), llm_client=llm_client)
        rr = dict(r)
        rr["grade"] = g
        graded.append(rr)
    return graded


def summarize(graded: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(graded)
    passed = sum(1 for r in graded if r.get("grade", {}).get("success"))
    by_level: Dict[str, Dict[str, int]] = {}
    for r in graded:
        lv = r.get("level", "unknown")
        d = by_level.setdefault(lv, {"total": 0, "passed": 0})
        d["total"] += 1
        if r.get("grade", {}).get("success"):
            d["passed"] += 1
    return {
        "benchmark": "dabench",
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "by_level": {k: {**v, "pass_rate": round(v["passed"]/v["total"], 4) if v["total"] else 0.0}
                     for k, v in by_level.items()},
    }
