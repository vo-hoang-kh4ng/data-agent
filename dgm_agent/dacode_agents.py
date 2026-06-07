"""
Triadic DGM Agents for DA-Code Benchmark
==========================================
Adapts the Triadic DGM pipeline (Planner → Solver → Verifier) for DA-Code.

Triadic Roles (adapted from DGM_lambda.py EvolutionaryScheduler):
- DACodePlanner: [Original Proposer role] Analyzes question + Blackboard context → structured solution plan
  In original Triadic DGM, Proposer GENERATES tasks from Knowledge Graph.
  For DA-Code (fixed benchmark), it ANALYZES existing tasks instead — same triadic position, adapted role.
- DACodeSolver: [Original Solver role] Generates Python code from plan, repairs with RIMRULE rules
- DACodeVerifier: [Original Verifier role] Executes code, validates output, diagnoses errors

Core Triadic DGM components reused:
- core/inspector.py → compute_mdl_epiplexity for code quality scoring
- core/rule_generator.py → RuleLibrary for cross-task debugging rules (RIMRULE)
- core/capacity_manager.py → DynamicCapacityManager for per-task budgets
- dgm_agent/llm.py → get_response_from_llm for all LLM calls

Epiplexity functions:
- estimate_task_epiplexity(): NCD-based complexity estimation BEFORE solving (drives budget)
- compute_ncd_epiplexity(): NCD between question and code (Goldilocks Zone check)
- is_in_goldilocks_zone(): Check if code is in learning zone [0.5, 2.2]
"""

import json
import math
import os
import re
import subprocess
import sys
import traceback
import zlib
from collections import Counter
from typing import Dict, List, Optional, Tuple

# Add project root for core/ imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.inspector import compute_mdl_epiplexity
from dgm_agent.llm import get_response_from_llm


# ── Goldilocks Zone Constants (from core/inspector.py Verifier) ──
GOLDILOCKS_MIN = 0.5   # Below = too easy (code is trivial copy of question)
GOLDILOCKS_MAX = 2.2   # Above = too hard (code is noise/unrelated)


def estimate_task_epiplexity(question: str, context: str = "") -> float:
    """Estimate task complexity from question analysis.

    v2 FIX: NCD(question, full_context) ≈ 2.0 for ALL tasks because
    context (~40K chars) >> question (~100 chars), making NCD meaningless.
    Replaced with multi-signal question complexity score.

    Signals used:
    1. Question length (longer → more complex requirements)
    2. Analytical operation count (t-test, regression, etc.)
    3. Multi-entity indicators (join, merge, multiple files)

    Returns float in [0.3, 1.5] range where:
    - < 0.6: Simple (count, basic filter)
    - 0.6–1.0: Standard (aggregation, groupby)
    - > 1.0: Complex (statistical tests, multi-table, predictions)
    """
    if not question:
        return 0.8

    score = 0.5  # Base

    # Signal 1: Question length — longer questions tend to need more code
    q_len = len(question)
    if q_len < 60:
        score += 0.1   # Short: likely simple lookup
    elif q_len < 120:
        score += 0.2   # Medium: standard aggregation
    elif q_len < 200:
        score += 0.35  # Long: complex multi-step
    else:
        score += 0.5   # Very long: multi-entity / complex analysis

    # Signal 2: Analytical operations (high-value complexity indicator)
    complex_ops = [
        r't-test', r'anova', r'chi.?square', r'regression', r'correlation',
        r'hypothesis', r'significant', r'p.value', r'confidence.interval',
        r'predict', r'classif', r'cluster', r'model', r'forecast',
        r'std|standard.deviation', r'variance', r'distribution',
        r'outlier', r'normalize', r'scale',
    ]
    for pattern in complex_ops:
        if re.search(pattern, question, re.IGNORECASE):
            score += 0.2

    # Signal 3: Multi-entity indicators (need joins/merges)
    multi_indicators = [r'join', r'merge', r'combine', r'both', r'compare.*and',
                        r'relationship between', r'top\s+\d+', r'each\s+\w+']
    for pattern in multi_indicators:
        if re.search(pattern, question, re.IGNORECASE):
            score += 0.1

    # Clamp to [0.3, 1.5]
    score = max(0.3, min(1.5, score))
    return round(score, 4)


def compute_ncd_epiplexity(task_description: str, generated_code: str) -> float:
    """Compute NCD-based epiplexity between task description and generated code.

    Reuses the exact pattern from core/inspector.py Verifier.estimate_epiplexity().
    This is the REAL epiplexity score used for Goldilocks Zone filtering.

    The NCD (Normalized Compression Distance) approximates Kolmogorov complexity:
    - Low NCD: Code is very similar to question (trivial, too easy)
    - Medium NCD: Code adds real problem-solving logic (Goldilocks zone)
    - High NCD: Code is unrelated to question (too hard / noise)
    """
    if not task_description or not generated_code:
        return 0.0

    task_bytes = task_description.encode('utf-8')
    code_bytes = generated_code.encode('utf-8')

    mdl_task = len(zlib.compress(task_bytes, level=9))
    mdl_code = len(zlib.compress(code_bytes, level=9))
    mdl_joint = len(zlib.compress(task_bytes + b"\n" + code_bytes, level=9))

    denom = max(mdl_task, mdl_code)
    if denom == 0:
        return 0.0

    ncd = (mdl_joint - min(mdl_task, mdl_code)) / denom
    epiplexity = ncd * 2.0  # Scale to [0, ~2.2+]

    return round(epiplexity, 4)


def is_in_goldilocks_zone(epiplexity_score: float) -> dict:
    """Check if epiplexity score is in the Goldilocks learning zone.

    From core/inspector.py:
    - [0, 0.5): Too easy — code is trivially similar to question
    - [0.5, 2.2]: Goldilocks — real learning zone
    - (2.2, ∞): Too hard — code is unrelated to question

    Returns
    -------
    dict with keys: epiplexity_score, goldilocks_status ("PASS"/"FAIL"), zone
    """
    is_pass = GOLDILOCKS_MIN <= epiplexity_score <= GOLDILOCKS_MAX
    if epiplexity_score < GOLDILOCKS_MIN:
        zone = "too_easy"
    elif epiplexity_score <= GOLDILOCKS_MAX:
        zone = "goldilocks"
    else:
        zone = "too_hard"

    return {
        "epiplexity_score": epiplexity_score,
        "goldilocks_status": "PASS" if is_pass else "FAIL",
        "zone": zone,
    }


# ── Prompt Constants ──

PROPOSER_SYSTEM = """You are a Proposer Agent in the Triadic DGM system for data science.
Your role is to analyze questions and data context to produce a structured solution plan.
You work methodically: first understand the question, then identify relevant data, then plan the approach."""

SOLVER_SYSTEM = """You are a Solver Agent in the Triadic DGM system for data science.
Your role is to write correct, executable Python code based on a structured plan.
You use the EXACT file paths and column names from the plan.
You handle missing values, type conversions, and edge cases carefully."""

VERIFIER_SYSTEM = """You are a Verifier Agent in the Triadic DGM system for data science.
Your role is to diagnose code errors and suggest precise fixes.
You identify the ROOT CAUSE: wrong file path, wrong column name, type mismatch, logic error, etc.
You provide specific, actionable fix suggestions."""

PLAN_PROMPT = """Analyze this data science question and create a structured solution plan.

QUESTION:
{question}

AVAILABLE FILES (compact summary — names, columns, dtypes only):
{context}

TASK:
1. RELEVANT FILES: Which files should be used? Why?
2. KEY COLUMNS: Which columns contain the data needed?
3. CALCULATION PLAN: Step-by-step approach (joins, filters, aggregations, sorting).
4. OUTPUT FORMAT: What should the final JSON output look like?
5. POTENTIAL ISSUES: Data quality issues (NaN, wrong types, date parsing, encoding).

Provide your analysis as structured text. Do NOT write code yet."""

EXPLORE_PROMPT = """Now examine the actual data to verify and refine your plan.

DETAILED FILE PREVIEWS:
{context}

Based on the actual data:
1. Confirm or revise which files and columns to use.
2. Verify data types and note any cleaning needed.
3. Identify the EXACT column names to use in code.
4. Note specific values, ranges, and patterns relevant to the question.
5. Update your calculation approach if the data differs from expectations.

Provide your refined analysis. Do NOT write code yet."""

CODE_PROMPT = """Now write the complete Python code to solve the question based on your plan and data exploration.

QUESTION:
{question}

INSTRUCTIONS:
- Use the EXACT file paths and column names from your exploration above.
- Import all needed libraries (pandas, numpy, json, etc.).
- Handle missing values (NaN) and type conversions carefully.
- The final answer MUST be printed as JSON. DO NOT wrap in a "main-task" key.
- If the question provides a JSON template like {{"key1": [...], "key2": [...]}}, follow that EXACT format:
  print(json.dumps({{"key1": value1, "key2": value2}}, indent=4))
- String values should NOT be wrapped in lists unless the template shows a list.
- For CSV output tasks, save to "result.csv" or as specified.
- Return ONLY valid, executable Python code inside ```python ``` blocks."""

REPAIR_PROMPT = """The code produced an error when executed:

ERROR:
{error}

DIAGNOSIS:
{diagnosis}

KNOWN DEBUGGING RULES:
{rules}

Fix the code based on the diagnosis and rules. Consider:
1. Is the error due to a wrong file path? Check exact paths from the data.
2. Is it a data type issue? Convert types explicitly.
3. Is it a KeyError? Verify exact column names.
4. Is it a calculation error? Verify the approach.

Return ONLY the fixed Python code inside ```python ``` blocks. Make sure to print the result as JSON."""

DIAGNOSE_PROMPT = """Diagnose this code error for a data science task.

QUESTION:
{question}

CODE:
```python
{code}
```

ERROR:
{error}

Provide a structured diagnosis:
1. ROOT CAUSE: What exactly went wrong? (wrong file path, wrong column, type error, logic error, NaN handling, etc.)
2. FIX SUGGESTION: What specific change should be made?
3. CONFIDENCE: How certain are you? (high/medium/low)

Respond in JSON format:
{{"root_cause": "...", "fix_suggestion": "...", "error_category": "KeyError|FileNotFoundError|ValueError|TypeError|LogicError|NaNError|Other", "confidence": "high|medium|low"}}"""


# ── DACodePlanner ──
# Renamed from DACodeProposer: In original Triadic DGM, Proposer GENERATES tasks.
# For DA-Code (fixed benchmark with given tasks), this role ANALYZES tasks instead.
# Same triadic position in the pipeline, adapted role.

class DACodePlanner:
    """Triadic DGM Planner Agent for DA-Code (adapted Proposer role).

    In original Triadic DGM, the Proposer generates new tasks from Knowledge Graph.
    For DA-Code, the benchmark provides fixed tasks — so Proposer's role shifts to:
    ANALYZING questions and data context to produce structured solution plans.

    Uses 2-step planning: Plan (compact context) → Explore (detailed context).
    """

    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    def plan(self, question: str, compact_context: str) -> Tuple[str, list]:
        """Step 1: Analyze question with compact context (file names + columns only).
        Returns (plan_text, msg_history).
        """
        prompt = PLAN_PROMPT.format(question=question, context=compact_context)
        response, msg_history = get_response_from_llm(
            msg=prompt,
            client=self.client,
            model=self.model,
            system_message=PROPOSER_SYSTEM,
            msg_history=[],
            temperature=0.2,
        )
        return response or "", msg_history

    def explore(self, question: str, detailed_context: str, msg_history: list) -> Tuple[str, list]:
        """Step 2: Refine plan with detailed context (data previews).
        Returns (refined_plan, msg_history).
        """
        prompt = EXPLORE_PROMPT.format(context=detailed_context)
        response, msg_history = get_response_from_llm(
            msg=prompt,
            client=self.client,
            model=self.model,
            system_message=PROPOSER_SYSTEM,
            msg_history=msg_history,
            temperature=0.2,
        )
        return response or "", msg_history


# ── DACodeSolver ──

class DACodeSolver:
    """Triadic DGM Solver Agent for DA-Code.

    Generates Python code from the Proposer's plan and repairs code
    using RIMRULE debugging rules from RuleLibrary.
    """

    def __init__(self, client, model: str, rule_library=None):
        self.client = client
        self.model = model
        self.rule_library = rule_library

    def generate_code(self, question: str, msg_history: list) -> Tuple[Optional[str], list]:
        """Generate Python code based on accumulated plan + exploration.
        Returns (code, msg_history).
        """
        prompt = CODE_PROMPT.format(question=question)
        response, msg_history = get_response_from_llm(
            msg=prompt,
            client=self.client,
            model=self.model,
            system_message=SOLVER_SYSTEM,
            msg_history=msg_history,
            temperature=0.2,
        )
        if not response:
            return None, msg_history
        code = self._extract_python_code(response)
        return code, msg_history

    def repair_code(self, code: str, error: str, diagnosis: str, rules: str = "") -> Optional[str]:
        """Repair code based on error, diagnosis, and RIMRULE rules.
        Returns fixed code or None.
        """
        error_str = str(error)[-3000:] if error else ""
        prompt = REPAIR_PROMPT.format(
            error=error_str,
            diagnosis=diagnosis or "No diagnosis available.",
            rules=rules or "No rules available yet.",
        )
        response, _ = get_response_from_llm(
            msg=prompt,
            client=self.client,
            model=self.model,
            system_message=SOLVER_SYSTEM,
            msg_history=[],
            temperature=0.3,
        )
        if not response:
            return None
        return self._extract_python_code(response)

    @staticmethod
    def _extract_python_code(text: str) -> Optional[str]:
        """Extract Python code from markdown code blocks."""
        if not text:
            return None
        # Try ```python ... ``` blocks
        matches = re.findall(r'```python\s*(.*?)```', text, re.DOTALL)
        if matches:
            return matches[-1].strip()
        # Try ``` ... ``` blocks
        matches = re.findall(r'```\s*(.*?)```', text, re.DOTALL)
        if matches:
            return matches[-1].strip()
        # Fallback: return entire text if it looks like code
        if any(kw in text for kw in ['import ', 'def ', 'print(', 'pd.']):
            return text.strip()
        return None


# ── DACodeVerifier ──

class DACodeVerifier:
    """Triadic DGM Verifier Agent for DA-Code.

    Executes code in sandbox, validates output format, diagnoses errors,
    and computes MDL epiplexity scores for code quality.
    """

    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    def execute_and_validate(self, code: str, task_id: str, sandbox_dir: str,
                              timeout: int = 120) -> Dict:
        """Execute code in sandbox and validate output.

        Returns verdict dict:
        {
            "success": bool,
            "output": str,          # stdout if success
            "error": str,           # stderr if failed
            "format_valid": bool,   # JSON/CSV format check
            "epiplexity": float,    # MDL-based code quality score
        }
        """
        epiplexity = compute_mdl_epiplexity(code) if code else 0.0
        result = {"success": False, "output": "", "error": "", "format_valid": False, "epiplexity": epiplexity}

        if not code or len(code) < 5:
            result["error"] = "Empty or too short code"
            return result

        # Run in sandbox
        task_sandbox = os.path.join(sandbox_dir, task_id)
        os.makedirs(task_sandbox, exist_ok=True)
        script_path = os.path.join(task_sandbox, "_sandbox_run.py")

        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)

        try:
            proc = subprocess.run(
                [sys.executable, script_path],
                capture_output=True, text=True, timeout=timeout,
                encoding="utf-8", errors="replace",
                cwd=task_sandbox,
            )
            if proc.returncode == 0:
                result["success"] = True
                result["output"] = proc.stdout.strip()
                result["format_valid"] = self._check_output_format(proc.stdout)
            else:
                result["error"] = (proc.stderr + "\n" + proc.stdout).strip()
        except subprocess.TimeoutExpired:
            result["error"] = f"TimeoutError: Code ran over {timeout}s"
        except Exception as e:
            result["error"] = str(e)

        return result

    def diagnose_error(self, code: str, error: str, question: str) -> Dict:
        """Use LLM to diagnose code error. Returns structured diagnosis."""
        code_snippet = code[-2000:] if len(code) > 2000 else code
        error_snippet = str(error)[-2000:] if error else "Unknown error"
        prompt = DIAGNOSE_PROMPT.format(question=question, code=code_snippet, error=error_snippet)

        try:
            response, _ = get_response_from_llm(
                msg=prompt,
                client=self.client,
                model=self.model,
                system_message=VERIFIER_SYSTEM,
                msg_history=[],
                temperature=0.1,
            )
            if response:
                # Try to parse JSON from response
                match = re.search(r'\{[^{}]+\}', response, re.DOTALL)
                if match:
                    return json.loads(match.group())
        except Exception as e:
            pass

        # Fallback diagnosis
        return {
            "root_cause": error_snippet[:200],
            "fix_suggestion": "Check file paths, column names, and data types.",
            "error_category": "Other",
            "confidence": "low",
        }

    @staticmethod
    def _check_output_format(output: str) -> bool:
        """Check if output looks like valid JSON or contains CSV info."""
        if not output:
            return False
        # Check for JSON output
        try:
            json.loads(output)
            return True
        except (json.JSONDecodeError, ValueError):
            pass
        # Check for JSON embedded in output
        if re.search(r'\{.*\}', output, re.DOTALL):
            return True
        return False

    @staticmethod
    def find_csv_results(task_id: str, sandbox_dir: str) -> List[str]:
        """Find CSV output files in task sandbox."""
        task_sandbox = os.path.join(sandbox_dir, task_id)
        csv_files = []
        if os.path.exists(task_sandbox):
            for f in os.listdir(task_sandbox):
                if f.endswith(".csv") and not f.startswith("_"):
                    csv_files.append(f)
        return csv_files
