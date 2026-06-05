"""
DA-Code Benchmark Runner for Triadic DGM
==========================================
Adapts dgm_agent/blackboard.py (Triadic DGM) to run DA-Code 91 tasks.
Uses multi-step exploration: Plan → Explore → Code → Execute + Debug.
Outputs in DA-Code format (dabench/result.json) for official evaluation.

Usage:
    python -m dgm_agent.dacode_runner --manifest data/dacode_unified_manifest.jsonl
    python -m dgm_agent.dacode_runner --example_name di-text-001
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "dgm_agent"))

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=True)

from dgm_agent.blackboard import DSBlackboardAgent, build_file_agents, Blackboard, BlackboardRequest
from dgm_agent.llm import create_client, get_response_from_llm


# ── Multi-Step Exploration Prompts ──

SYSTEM_PROMPT_MULTI = """You are an expert Data Scientist solving data science problems using Python.
You work methodically: first understand the question and plan, then explore the data, then write code.
Always reason carefully about which files and columns are relevant before coding."""

DACODE_PLAN_PROMPT = """Analyze this data science question and create a plan.

QUESTION:
{question}

AVAILABLE FILES (compact summary — names, columns, dtypes only):
{compact_context}

TASK:
1. RELEVANT FILES: Which files should be used? Why?
2. KEY COLUMNS: Which columns contain the data needed for the answer?
3. CALCULATION PLAN: Step-by-step approach to compute the answer (joins, filters, aggregations, sorting).
4. OUTPUT FORMAT: What should the final JSON output look like based on the question template?
5. POTENTIAL ISSUES: Data quality issues to watch for (NaN, wrong types, date parsing, encoding).

Provide your analysis as structured text. Do NOT write code yet."""

DACODE_EXPLORE_PROMPT = """Now examine the actual data to verify and refine your plan.

DETAILED FILE PREVIEWS:
{detailed_context}

Based on the actual data:
1. Confirm or revise which files and columns to use.
2. Verify data types and note any cleaning needed (e.g., trailing spaces, date formats, mixed types).
3. Identify the exact column names to use in code.
4. Note specific values, ranges, and patterns relevant to the question.
5. Update your calculation approach if the data differs from what you expected.

Provide your refined analysis as structured text. Do NOT write code yet."""

DACODE_CODE_PROMPT = """Now write the complete Python code to solve the question based on your plan and data exploration.

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

DACODE_DEBUG_PROMPT = """The code produced an error when executed:

ERROR:
{error}

Fix the code. Consider:
1. Is the error due to a wrong file path? Check exact paths from the data.
2. Is it a data type issue? Convert types explicitly.
3. Is it a KeyError? Verify exact column names.
4. Is it a calculation error? Verify the approach.

Return ONLY the fixed Python code inside ```python ``` blocks. Make sure to print the result as JSON."""

# ── Fallback 1-shot prompt (kept for fallback path) ──

DACODE_FALLBACK_PROMPT = """You are an expert Data Scientist. Solve the following data science question using Python.

QUESTION:
{question}

{context}

INSTRUCTIONS:
- Use the EXACT file paths shown above to load the data.
- Use pandas, numpy, or other standard data science libraries.
- Handle missing values (NaN) appropriately.
- The final answer MUST be printed as JSON. DO NOT wrap in a "main-task" key.
- If the question provides a JSON template like {{"key1": [...], "key2": [...]}}, follow that EXACT format:
  print(json.dumps({{"key1": value1, "key2": value2}}, indent=4))
- Always import json at the top of your code.
- String values should NOT be wrapped in lists unless the template shows a list.
- For CSV output tasks, save the result to a CSV file named "result.csv" or as specified.
- Return ONLY valid, executable Python code inside ```python ``` blocks.
"""


class DACodeRunner:
    """Adapts Triadic DGM to DA-Code benchmark format with multi-step exploration."""

    def __init__(self, model: str = "hosted_vllm/Qwen/Qwen3.5-35B-A3B-FP8",
                 sandbox_dir: str = "", max_debug_rounds: int = 3):
        if not sandbox_dir:
            sandbox_dir = os.path.join(PROJECT_ROOT, "data", "dacode_sandbox")
        self.sandbox_dir = sandbox_dir
        self.max_debug_rounds = max_debug_rounds

        # Initialize Triadic DGM agent (for blackboard + file discovery)
        print(f"🚀 Initializing Triadic DGM Agent with Multi-Step Exploration (model={model})")
        self.agent = DSBlackboardAgent(model=model, sandbox_dir=sandbox_dir, max_debug_rounds=max_debug_rounds)

    # ── Context Builders ──

    def _build_compact_context(self, bb: Blackboard) -> str:
        """Build compact context: file names + columns + dtypes only. No previews. ~3-5K chars."""
        lines = ["=== AVAILABLE DATA FILES (compact) ===\n"]
        sorted_resp = sorted(bb.responses, key=lambda r: r.relevance_score, reverse=True)
        for resp in sorted_resp:
            lines.append(f"\n[Cluster: {resp.cluster_name}] (relevance: {resp.relevance_score:.2f})")
            for fc in resp.file_contexts:
                if fc.error:
                    lines.append(f"  {fc.file_path}  -> ERROR: {fc.error}")
                    continue
                lines.append(f"  {fc.file_path}  (type={fc.file_type})")
                if fc.sheets:
                    lines.append(f"    Sheets: {fc.sheets}")
                if fc.columns:
                    lines.append(f"    Columns: {fc.columns}")
                if fc.dtypes:
                    dtype_str = ", ".join(f"{k}: {v}" for k, v in fc.dtypes.items())
                    lines.append(f"    Dtypes: {dtype_str}")
            lines.append("")
        return "\n".join(lines)

    def _build_detailed_context(self, bb: Blackboard) -> str:
        """Build detailed context with file previews for top-3 clusters. ~20-40K chars."""
        sorted_resp = sorted(bb.responses, key=lambda r: r.relevance_score, reverse=True)
        top_responses = sorted_resp[:3]

        lines = ["=== DETAILED DATA PREVIEWS ===\n"]
        for resp in top_responses:
            lines.append(f"\n[Cluster: {resp.cluster_name}]")
            for fc in resp.file_contexts:
                if fc.error:
                    lines.append(f"  File: {fc.file_path}  -> ERROR: {fc.error}")
                    continue
                lines.append(f"  File: {fc.file_path}  (type={fc.file_type})")
                if fc.sheets:
                    lines.append(f"    Sheets: {fc.sheets}")
                if fc.columns:
                    lines.append(f"    Columns: {fc.columns}")
                if fc.dtypes:
                    dtype_str = ", ".join(f"{k}: {v}" for k, v in list(fc.dtypes.items())[:20])
                    lines.append(f"    Dtypes: {dtype_str}")
                if fc.preview:
                    lines.append(f"    Preview:\n{fc.preview}")
            lines.append("")

        full_context = "\n".join(lines)
        if len(full_context) > 40000:
            full_context = full_context[:40000] + "\n...[CONTEXT TRUNCATED DUE TO LENGTH LIMIT]..."
        return full_context

    # ── Multi-Step Generation ──

    def _run_multi_step_generation(self, question: str, bb: Blackboard, trajectory: list) -> Optional[str]:
        """Plan -> Explore -> Code using multi-turn conversation. Returns Python code or None."""
        compact_ctx = self._build_compact_context(bb)
        detailed_ctx = self._build_detailed_context(bb)

        client = self.agent.client
        model_name = self.agent.model_name

        msg_history = []

        # Step 1: Plan (compact context — filenames + columns only)
        trajectory.append({"action": "plan"})
        plan_prompt = DACODE_PLAN_PROMPT.format(question=question, compact_context=compact_ctx)
        print(f"  🧠 Step 1/3: Planning... ({len(plan_prompt)} chars)")

        plan_response, msg_history = get_response_from_llm(
            msg=plan_prompt,
            client=client,
            model=model_name,
            system_message=SYSTEM_PROMPT_MULTI,
            msg_history=msg_history,
            temperature=0.2,
        )
        if not plan_response:
            return None

        # Step 2: Explore (detailed context — data previews)
        trajectory.append({"action": "explore"})
        explore_prompt = DACODE_EXPLORE_PROMPT.format(detailed_context=detailed_ctx)
        print(f"  🔎 Step 2/3: Exploring data... ({len(explore_prompt)} chars)")

        explore_response, msg_history = get_response_from_llm(
            msg=explore_prompt,
            client=client,
            model=model_name,
            system_message=SYSTEM_PROMPT_MULTI,
            msg_history=msg_history,
            temperature=0.2,
        )
        if not explore_response:
            return None

        # Step 3: Code generation (uses accumulated msg_history)
        trajectory.append({"action": "code_generation"})
        code_prompt = DACODE_CODE_PROMPT.format(question=question)
        print(f"  💻 Step 3/3: Generating code...")

        code_response, msg_history = get_response_from_llm(
            msg=code_prompt,
            client=client,
            model=model_name,
            system_message=SYSTEM_PROMPT_MULTI,
            msg_history=msg_history,
            temperature=0.2,
        )
        if not code_response:
            return None

        return self.agent._extract_python_code(code_response)

    # ── Sandbox & IO ──

    def _run_code_sandbox(self, code: str, task_id: str, timeout: int = 120) -> tuple:
        """Run code and return (success, output)."""
        task_sandbox = os.path.join(self.sandbox_dir, task_id)
        os.makedirs(task_sandbox, exist_ok=True)
        script_path = os.path.join(task_sandbox, "_sandbox_run.py")

        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True, text=True, timeout=timeout,
                encoding="utf-8", errors="replace",
                cwd=task_sandbox,
            )
            if result.returncode == 0:
                return True, result.stdout.strip()
            else:
                return False, (result.stderr + "\n" + result.stdout).strip()
        except subprocess.TimeoutExpired:
            return False, f"TimeoutError: Code ran over {timeout}s"
        except Exception as e:
            return False, str(e)

    def _save_result(self, task_id: str, result_data: dict):
        """Save result in DA-Code format."""
        task_sandbox = os.path.join(self.sandbox_dir, task_id)
        dabench_dir = os.path.join(task_sandbox, "dabench")
        os.makedirs(dabench_dir, exist_ok=True)
        result_path = os.path.join(dabench_dir, "result.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

    def _check_existing(self, task_id: str) -> bool:
        """Check if task already has a result."""
        result_path = os.path.join(self.sandbox_dir, task_id, "dabench", "result.json")
        if os.path.exists(result_path):
            try:
                data = json.load(open(result_path))
                return data.get("finished", False)
            except:
                pass
        return False

    # ── Main Task Runner ──

    def run_task(self, task: dict, force: bool = False) -> dict:
        """Run a single DA-Code task using Triadic DGM with multi-step exploration."""
        task_id = task["task_id"]
        question = task["question"]
        data_lake_dir = task.get("data_lake_dir", "")

        if not force and self._check_existing(task_id):
            print(f"  ⏭️ Already done: {task_id}")
            return {"task_id": task_id, "skipped": True}

        t_start = time.time()
        print(f"\n[{task_id}] {question[:80]}...")

        # ═══ Phase 0: Blackboard Discovery ═══
        request = BlackboardRequest(
            task_id=task_id,
            question=question,
            data_lake_dir=data_lake_dir,
        )
        self.agent.blackboard.post_request(request)

        file_agents = build_file_agents(data_lake_dir)
        for fa in file_agents:
            response = fa.handle_request(request)
            self.agent.blackboard.post_response(response)

        # Top-K relevance filtering
        bb = self.agent.blackboard
        if len(bb.responses) > 5:
            bb.responses.sort(key=lambda r: r.relevance_score, reverse=True)
            bb.responses = bb.responses[:5]
            print(f"  🔍 Top-5 filter: kept {len(bb.responses)} most relevant clusters")

        trajectory = [{"action": "blackboard_discovery", "agents": len(file_agents), "clusters_kept": len(bb.responses)}]

        # ═══ Phase 1: Multi-Step Generation (Plan → Explore → Code) ═══
        final_code = None
        multi_step_ok = False
        try:
            final_code = self._run_multi_step_generation(question, bb, trajectory)
            if final_code and len(final_code) > 20:
                multi_step_ok = True
                print(f"  ✅ Multi-step generation succeeded ({len(final_code)} chars)")
        except Exception as e:
            print(f"  ⚠️ Multi-step failed: {str(e)[:120]}")

        # ═══ Fallback: 1-shot generation ═══
        if not multi_step_ok:
            trajectory.append({"action": "fallback_1shot"})
            print(f"  🔄 Falling back to 1-shot generation...")
            context = bb.get_context_summary()
            coding_prompt = DACODE_FALLBACK_PROMPT.format(question=question, context=context)
            raw_response = self.agent._call_llm(coding_prompt)
            final_code = self.agent._extract_python_code(raw_response)

        # ═══ Phase 2: Execute + Debug ═══
        finished = False
        output = ""

        for round_idx in range(self.max_debug_rounds):
            if not final_code or len(final_code) < 5:
                break

            success, output = self._run_code_sandbox(final_code, task_id)
            trajectory.append({
                "action": f"code_execution_round_{round_idx}",
                "success": success,
            })

            if success and len(output) > 0:
                finished = True
                print(f"  ✅ Execution success (round {round_idx})")
                break
            else:
                print(f"  ❌ Round {round_idx}: {str(output)[:100]}")
                if round_idx < self.max_debug_rounds - 1:
                    debug_msg = DACODE_DEBUG_PROMPT.format(error=str(output)[-3000:])
                    raw_fix = self.agent._call_llm(debug_msg)
                    new_code = self.agent._extract_python_code(raw_fix)
                    if new_code and len(new_code) > 10:
                        final_code = new_code

        # Check for CSV output files
        result_files = []
        task_sandbox = os.path.join(self.sandbox_dir, task_id)
        if os.path.exists(task_sandbox):
            for f in os.listdir(task_sandbox):
                if f.endswith(".csv") and not f.startswith("_"):
                    result_files.append(f)

        elapsed = time.time() - t_start
        print(f"  {'✅' if finished else '❌'} {task_id} — {elapsed:.1f}s")

        # Save result
        result_data = {
            "finished": finished,
            "steps": len(trajectory),
            "result": output if finished else "",
            "result_files": result_files,
            "trajectory": trajectory,
        }
        self._save_result(task_id, result_data)

        return {
            "task_id": task_id,
            "finished": finished,
            "elapsed": round(elapsed, 1),
        }


def main():
    parser = argparse.ArgumentParser(description="DA-Code Runner for Triadic DGM")
    parser.add_argument("--manifest", type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "dacode_unified_manifest.jsonl"))
    parser.add_argument("--model", type=str, default=os.environ.get("DACODE_MODEL", "hosted_vllm/Qwen/Qwen3.5-35B-A3B-FP8"))
    parser.add_argument("--sandbox_dir", type=str, default=os.path.join(PROJECT_ROOT, "data", "dacode_sandbox"))
    parser.add_argument("--max_debug_rounds", type=int, default=3)
    parser.add_argument("--example_name", type=str, default="")
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument("--max_retries", type=int, default=1)
    args = parser.parse_args()

    # Load tasks
    tasks = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))

    if args.example_name:
        tasks = [t for t in tasks if args.example_name in t["task_id"]]

    print(f"🚀 Triadic DGM (Multi-Step) — DA-Code Benchmark ({len(tasks)} tasks)")
    print(f"   Model: {args.model}")
    print(f"   Sandbox: {args.sandbox_dir}")

    runner = DACodeRunner(model=args.model, sandbox_dir=args.sandbox_dir, max_debug_rounds=args.max_debug_rounds)

    done = 0
    failed = 0
    for i, task in enumerate(tasks):
        task_id = task["task_id"]
        print(f"\n[{i+1}/{len(tasks)}] {task_id}")

        result = runner.run_task(task, force=args.force_rerun)

        if result.get("skipped"):
            done += 1
            continue

        if result["finished"]:
            done += 1
        else:
            failed += 1
            # Retry
            for retry in range(args.max_retries):
                print(f"  🔄 Retry {retry+1}/{args.max_retries}")
                result = runner.run_task(task, force=True)
                if result["finished"]:
                    done += 1
                    failed -= 1
                    break

        total = done + failed
        if total % 10 == 0 or total == len(tasks):
            print(f"\n📊 Progress: {total}/{len(tasks)} | Done: {done} | Failed: {failed} | Rate: {done/total*100:.1f}%")

    print(f"\n{'='*60}")
    print(f"📊 FINAL: {done}/{len(tasks)} done ({done/len(tasks)*100:.1f}%) | {failed} failed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
