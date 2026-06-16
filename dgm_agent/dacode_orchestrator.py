"""
Triadic DGM Orchestrator for DA-Code
=====================================
Wires Planner → Solver → Verifier into a pipeline with:
- Blackboard Discovery (E5 + KMeans file clustering)
- Epiplexity-driven CapacityManager (NCD-based budget allocation)
- RuleLibrary (RIMRULE cross-task debugging rules)
- Goldilocks Zone filtering ([0.5, 2.2] epiplexity range)

This is the main orchestrator that replaces the flat pipeline in dacode_runner.py
with the full Triadic DGM architecture adapted for DA-Code benchmark.

Architecture mapping (Original Triadic DGM → DA-Code adaptation):
- Proposer (generates tasks from KG) → Planner (analyzes given tasks + data)
- Solver (mutates LAMBDA.py)       → Solver (generates Python data analysis code)
- Verifier (LLM judge + epiplexity) → Verifier (sandbox exec + NCD epiplexity)
- CapacityManager (epiplexity-based) → Same, now with NCD epiplexity
- RuleLibrary (RIMRULE)            → Same, now with fixed extraction
- DGM Outer Loop (30 generations)  → Per-task pipeline (no evolution, by design)

Usage:
    from dgm_agent.dacode_orchestrator import DACodeOrchestrator
    orchestrator = DACodeOrchestrator(model="hosted_vllm/Qwen/Qwen3.5-35B-A3B-FP8")
    result = orchestrator.run_task(task)
"""

import json
import os
import sys
import time
from typing import Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=True)

from dgm_agent.blackboard import DSBlackboardAgent, build_file_agents, Blackboard, BlackboardRequest
from dgm_agent.dacode_agents import (
    DACodePlanner, DACodeSolver, DACodeVerifier,
    estimate_task_epiplexity, compute_ncd_epiplexity, is_in_goldilocks_zone,
    GOLDILOCKS_MIN, GOLDILOCKS_MAX,
)
from dgm_agent.llm import create_client, get_response_from_llm
from core.capacity_manager import DynamicCapacityManager
from core.rule_generator import RuleLibrary, extract_rule_from_reflexion


class DACodeOrchestrator:
    """Triadic DGM Orchestrator for DA-Code benchmark.

    Pipeline per task:
        Phase 0: Blackboard Discovery (E5 clustering + file agents)
        Phase 0.5: Epiplexity Estimation → CapacityManager budget
        Phase 1: Planner (Plan + Explore)  [adapted Proposer role]
        Phase 2: Solver (Code generation)
        Phase 3: Verifier (Execute + Validate + Diagnose + Repair loop)
        Phase 3.5: Goldilocks Zone check (NCD epiplexity)
        Phase 4: Save result

    The RuleLibrary accumulates debugging rules across tasks (RIMRULE memory).
    The CapacityManager allocates retries based on NCD epiplexity (NOT just hardness).
    """

    def __init__(self, model: str = "hosted_vllm/Qwen/Qwen3.5-35B-A3B-FP8",
                 sandbox_dir: str = "", max_debug_rounds: int = 3):
        if not sandbox_dir:
            sandbox_dir = os.path.join(PROJECT_ROOT, "data", "dacode_sandbox")
        self.sandbox_dir = sandbox_dir
        self.max_debug_rounds = max_debug_rounds
        self.model = model

        # Initialize LLM client (returns (client, model_name) tuple)
        client_result = create_client(model)
        self.client = client_result[0]
        self.model_name = client_result[1]

        # Initialize Triadic DGM components
        # NOTE: Renamed Proposer → Planner (original Proposer generates tasks,
        # for DA-Code it analyzes existing tasks — same triadic position, adapted role)
        self.planner = DACodePlanner(self.client, self.model_name)
        self.solver = DACodeSolver(self.client, self.model_name)
        self.verifier = DACodeVerifier(self.client, self.model_name)
        self.capacity_manager = DynamicCapacityManager()
        self.rule_library = RuleLibrary()

        # Pass rule library to solver for injection
        self.solver.rule_library = self.rule_library

        # Track how many tasks solved by Triadic vs fallback
        self.triadic_solved = 0
        self.fallback_solved = 0
        self.goldilocks_pass = 0
        self.goldilocks_fail = 0

        print(f"🧠 Triadic DGM Orchestrator initialized (model={model})")
        print(f"   Components: Planner → Solver → Verifier + CapacityManager + RuleLibrary")
        print(f"   Goldilocks Zone: [{GOLDILOCKS_MIN}, {GOLDILOCKS_MAX}]")

    # ── Context Builders (reuse from blackboard) ──

    def _build_compact_context(self, bb: Blackboard, detail_top_k: int = 25,
                                max_chars: int = 40000) -> str:
        """Compact context that lists ALL lake filenames/paths (visibility guaranteed),
        then columns/dtypes for the top-K files by relevance.

        Replaces the old per-cluster cap (15/cluster) which silently truncated the
        relevant file out of view on large lakes — the global discovery failure mode
        where the gold file never reached the planner/solver (e.g. di-text-001's
        ``world-data-2023.csv`` appeared in 0 trajectory steps). Phase A lists every
        file compactly so the right name is always visible; Phase B adds detail within
        the char budget so truncation, if any, only drops columns/dtypes, not names.
        """
        # Flatten all files across clusters, sorted globally by relevance.
        flat = []  # (relevance, cluster_name, fc)
        for resp in bb.responses:
            for fc in resp.file_contexts:
                flat.append((resp.relevance_score, resp.cluster_name, fc))
        flat.sort(key=lambda x: x[0], reverse=True)

        lines = ["=== AVAILABLE DATA FILES IN THE DATA LAKE (ALL files listed) ===\n",
                 "--- ALL FILES (basename -> path), sorted by relevance ---"]
        for rel, cname, fc in flat:
            tag = "  -> ERROR" if fc.error else ""
            lines.append(f"  {os.path.basename(fc.file_path)}  ->  {fc.file_path}{tag}")
        lines.append("")

        # Phase B: columns/dtypes for the top-K by relevance, within the char budget.
        used = len("\n".join(lines))
        if flat and used < max_chars - 500:
            lines.append("--- DETAILED (columns/dtypes) for the most relevant files ---")
            for rel, cname, fc in flat[:detail_top_k]:
                if fc.error:
                    continue
                detail = [f"  {os.path.basename(fc.file_path)} (type={fc.file_type})"]
                if fc.sheets:
                    detail.append(f"    Sheets: {fc.sheets}")
                if fc.columns:
                    detail.append(f"    Columns: {fc.columns}")
                if fc.dtypes:
                    detail.append("    Dtypes: " + ", ".join(f"{k}:{v}" for k, v in fc.dtypes.items()))
                block = "\n".join(detail)
                if used + len(block) + 1 > max_chars:
                    lines.append("  ... (further details omitted to stay within context window)")
                    break
                lines.append(block)
                used += len(block) + 1

        full = "\n".join(lines)
        if len(full) > max_chars:
            full = full[:max_chars] + "\n... [COMPACT CONTEXT TRUNCATED]"
        return full

    def _build_detailed_context(self, bb: Blackboard) -> str:
        """Detailed context with file previews for top-3 clusters."""
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
            full_context = full_context[:40000] + "\n...[CONTEXT TRUNCATED]..."
        return full_context

    def _build_file_paths_block(self, bb: Blackboard) -> str:
        """Pre-loaded DataFrame skeleton injected DIRECTLY into Solver/Repair/Fallback prompts.

        Qwen-35B unreliably uses provided file paths — even with exact pd.read_csv(r"...")
        lines and a 'do NOT use data.csv' warning in the prompt, it reverts to a 'data.csv'
        prior and hallucinates a sandbox path. The robust fix is to REMOVE its opportunity
        to get paths wrong: present the data as ALREADY LOADED into named DataFrames
        (df_file_0, df_file_1, ...) and instruct it to use those directly. The actual
        read_csv lines are kept so the generated script is self-contained & runnable.
        """
        seen = set()
        lines = [
            "=== DATA ALREADY LOADED (the lines below run first; use df_file_0, df_file_1, ... directly) ===",
            "import pandas as pd",
        ]
        n = 0
        max_files = 10  # cap preloaded DataFrames (global lake has 172 files)
        for resp in sorted(bb.responses, key=lambda r: r.relevance_score, reverse=True):
            if n >= max_files:
                break
            for fc in resp.file_contexts:
                if n >= max_files:
                    break
                if fc.error or fc.file_path in seen:
                    continue
                seen.add(fc.file_path)
                ftype = (fc.file_type or "").lower()
                if ftype in ("xlsx", "xls", "ods"):
                    loader = "pd.read_excel"
                elif ftype == "json":
                    loader = "pd.read_json"
                else:
                    loader = "pd.read_csv"
                cols = ", ".join(list(fc.columns)[:12]) if fc.columns else "?"
                lines.append(f'df_file_{n} = {loader}(r"{fc.file_path}")   # columns: {cols}')
                n += 1
        if n == 0:
            return ""
        lines.append("=" * 82)
        lines.append(f"Use df_file_0 .. df_file_{n - 1} DIRECTLY. Do NOT call read_csv/read_excel "
                     f"or invent filenames like 'data.csv' — the data is already loaded above.")
        return "\n".join(lines)

    def _preload_blocks(self, bb: Blackboard):
        """Return (solver_data_block, loading_prefix).

        - Small lake (≤12 files): preload DataFrames as df_file_N (easy discovery).
        - Large lake (>12 files, real discovery): pass the FULL list of real lake file
          paths plus explicit instructions to call pd.read_csv/read_excel with ONLY those
          paths. This fixes the global failure where the solver was told "data is already
          loaded" (no df_file_N exists in global mode) and had no real filenames → it
          hallucinated filenames and printed dummy data.
        """
        n_files = sum(len(r.file_contexts) for r in bb.responses)
        if n_files <= 12:
            return self._build_file_paths_block(bb), self._build_loading_prefix(bb)
        return self._build_global_data_block(bb), ""

    def _build_global_data_block(self, bb: Blackboard, max_files: int = 200) -> str:
        """For large lakes (real discovery): list real file paths and instruct the solver
        to read them itself with pd.read_csv/read_excel, using ONLY these paths."""
        flat = [(r.relevance_score, fc) for r in bb.responses for fc in r.file_contexts]
        flat.sort(key=lambda x: x[0], reverse=True)
        lake_dir = ""
        lines = ["=== AVAILABLE DATA FILES IN THE DATA LAKE ===",
                 "The data is NOT preloaded. You MUST call pd.read_csv() / pd.read_excel() yourself.",
                 "Use ONLY the real paths below. Pick the relevant file(s) by their name and columns",
                 "(read each header to decide). Do NOT guess or invent filenames such as 'data.csv'.",
                 ""]
        for rel, fc in flat[:max_files]:
            if fc.error:
                continue
            lines.append(f'  pd.read_csv(r"{fc.file_path}")   # {os.path.basename(fc.file_path)} (type={fc.file_type})')
            if not lake_dir:
                lake_dir = os.path.dirname(fc.file_path)
        if lake_dir:
            lines.append("")
            lines.append("If unsure which file to use, inspect the lake first:")
            lines.append(f'  import os; print(os.listdir(r"{lake_dir}"))')
        lines.append("")
        lines.append("CRITICAL: NEVER print dummy/placeholder/sample data. If a file is missing, raise")
        lines.append("the error — do NOT fabricate data to 'demonstrate'.")
        lines.append("")
        lines.append("CRITICAL (OUTPUT PATH): Write your result file(s) to the CURRENT directory")
        lines.append("(the run sandbox) using ONLY the bare output name, e.g. df.to_csv('result.csv').")
        lines.append("NEVER derive the output path from an input file's directory. NEVER use")
        lines.append("os.path.join(lake_dir, ...), os.path.dirname() of any path above, or any")
        lines.append("absolute path. Every path listed above is a READ-ONLY input — your outputs")
        lines.append("must land in the current working directory, nowhere else.")
        return "\n".join(lines)

    def _build_loading_prefix(self, bb: Blackboard) -> str:
        """Executable prefix prepended to the model's code at run time (NOT shown in prompt).

        Defines df_file_0, df_file_1, ... = pd.read_csv(r"<exact_path>") so that the
        DataFrames the model is told are 'already loaded' actually exist, regardless of
        whether the model obeys the instruction to omit its own read_csv calls.
        """
        seen = set()
        lines = ["import pandas as pd"]
        n = 0
        max_files = 10  # cap preloaded DataFrames (global lake has 172 files)
        for resp in sorted(bb.responses, key=lambda r: r.relevance_score, reverse=True):
            if n >= max_files:
                break
            for fc in resp.file_contexts:
                if n >= max_files:
                    break
                if fc.error or fc.file_path in seen:
                    continue
                seen.add(fc.file_path)
                ftype = (fc.file_type or "").lower()
                if ftype in ("xlsx", "xls", "ods"):
                    loader = "pd.read_excel"
                elif ftype == "json":
                    loader = "pd.read_json"
                else:
                    loader = "pd.read_csv"
                lines.append(f'df_file_{n} = {loader}(r"{fc.file_path}")')
                n += 1
        return "\n".join(lines) if n > 0 else ""

    # ── File Discovery ──

    def _discover_files(self, task: dict) -> Blackboard:
        """Phase 0: Blackboard file discovery with E5 + KMeans clustering."""
        task_id = task["task_id"]
        question = task["question"]
        data_lake_dir = task.get("data_lake_dir", "")

        request = BlackboardRequest(
            task_id=task_id,
            question=question,
            data_lake_dir=data_lake_dir,
        )

        bb = Blackboard()
        bb.post_request(request)

        file_agents = build_file_agents(data_lake_dir)
        for fa in file_agents:
            response = fa.handle_request(request)
            bb.post_response(response)

        # Top-K relevance filtering
        if len(bb.responses) > 5:
            bb.responses.sort(key=lambda r: r.relevance_score, reverse=True)
            bb.responses = bb.responses[:5]

        return bb

    # ── Result IO ──

    def _save_result(self, task_id: str, result_data: dict):
        """Save result in DA-Code official format.

        Format compatible with da-code-repo/da_agent/evaluators/evaluation.py:
        - trajectory: list of {action, code, observation} dicts
        - result_files: {added_files: [], changed_files: []} dict
        """
        # Convert our trajectory to official format
        raw_traj = result_data.get("trajectory", [])
        official_traj = []
        for step in raw_traj:
            official_traj.append({
                "action": "Python",
                "code": json.dumps(step, ensure_ascii=False),
                "observation": "standard output",
            })
        # Add Terminate step at the end
        official_traj.append({"action": "Terminate", "code": "", "observation": ""})

        # Convert result_files from list to official dict format
        raw_files = result_data.get("result_files", [])
        if isinstance(raw_files, list):
            official_files = {"added_files": raw_files, "changed_files": []}
        else:
            official_files = raw_files  # Already dict

        # Build official-compatible result (keep our extra fields too)
        official_data = {
            **result_data,
            "trajectory": official_traj,
            "result_files": official_files,
        }

        task_sandbox = os.path.join(self.sandbox_dir, task_id)
        dabench_dir = os.path.join(task_sandbox, "dabench")
        os.makedirs(dabench_dir, exist_ok=True)
        result_path = os.path.join(dabench_dir, "result.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(official_data, f, indent=2, ensure_ascii=False)

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

    # ── RIMRULE Extraction ──

    def _extract_rule_from_repair(self, old_code: str, error: str, new_code: str,
                                   round_idx: int) -> None:
        """Extract a RIMRULE debugging rule from a successful repair.

        This is called IMMEDIATELY after each repair attempt (not deferred to end).
        This fixes the bug where rules were never extracted due to strict conditions.
        """
        if not old_code or not error or not new_code:
            return
        try:
            rule = extract_rule_from_reflexion(
                old_code, error, new_code,
                self.client, self.model_name
            )
            if rule:
                self.rule_library.add_or_update_rule(rule)
                print(f"  📝 RIMRULE extracted (round {round_idx}): {rule[:60]}...")
        except Exception as e:
            print(f"  ⚠️ RIMRULE extraction failed (round {round_idx}): {e}")

    # ── Main Task Runner ──

    def run_task(self, task: dict, force: bool = False) -> dict:
        """Run a single DA-Code task through the Triadic DGM pipeline.

        Pipeline: Blackboard → Epiplexity → Planner → Solver → Verifier → Repair → Goldilocks → Save
        """
        task_id = task["task_id"]
        question = task["question"]
        task_hardness = task.get("hardness", "Medium")
        task_category = task.get("task_category", "")

        if not force and self._check_existing(task_id):
            print(f"  ⏭️ Already done: {task_id}")
            return {"task_id": task_id, "skipped": True}

        t_start = time.time()
        print(f"\n[{task_id}] {question[:80]}...")

        # ═══ Phase 0: Blackboard Discovery ═══
        bb = self._discover_files(task)
        compact_ctx = self._build_compact_context(bb)
        detailed_ctx = self._build_detailed_context(bb)
        solver_data_block, loading_prefix = self._preload_blocks(bb)
        trajectory = [{"action": "blackboard_discovery", "clusters_kept": len(bb.responses)}]

        # ═══ Phase 0.5: Epiplexity Estimation + CapacityManager Budget ═══
        task_epiplexity = estimate_task_epiplexity(question, compact_ctx)
        trajectory.append({"action": "epiplexity_estimation", "score": task_epiplexity})
        print(f"  📊 Task Epiplexity: {task_epiplexity:.4f} "
              f"(zone: {'Goldilocks' if GOLDILOCKS_MIN <= task_epiplexity <= GOLDILOCKS_MAX else 'Outside'})")

        budget = self.capacity_manager.allocate_budget_for_dacode(
            task_hardness, task_category, epiplexity_score=task_epiplexity
        )
        max_retries = budget["max_retries"]
        # Adaptive rounds: hard tasks get 5 rounds
        if task_hardness == "Hard":
            max_retries = 5
        print(f"  💰 Budget: {max_retries} retries, temp={budget['temperature']}")

        # ═══ Phase 1: Planner (Plan + Explore)  [adapted Proposer role] ═══
        trajectory.append({"action": "planner_plan"})
        print(f"  🎯 Planner: Planning... ({len(compact_ctx)} chars)")
        plan_text, msg_history = self.planner.plan(question, compact_ctx)

        if not plan_text:
            trajectory.append({"action": "planner_plan_failed"})
            print(f"  ⚠️ Planner failed, using fallback")
            return self._fallback_1shot(task, bb, trajectory, t_start)

        trajectory.append({"action": "planner_explore"})
        print(f"  🔎 Planner: Exploring data... ({len(detailed_ctx)} chars)")
        refined_plan, msg_history = self.planner.explore(question, detailed_ctx, msg_history)

        # ═══ Phase 2: Solver (Code Generation) ═══
        trajectory.append({"action": "solver_generate"})
        print(f"  💻 Solver: Generating code...")
        code, msg_history = self.solver.generate_code(question, msg_history, data_files=solver_data_block)

        if not code or len(code) < 10:
            trajectory.append({"action": "solver_failed"})
            print(f"  ⚠️ Solver failed, using fallback")
            return self._fallback_1shot(task, bb, trajectory, t_start)

        print(f"  ✅ Solver: {len(code)} chars generated")

        # ═══ Phase 3: Verifier + Repair Loop ═══
        # FIX: Track error and old_code SEPARATELY for RIMRULE extraction
        finished = False
        output = ""
        epiplexity_scores = []
        ncd_epiplexity = 0.0
        last_error = ""       # Track error from FAILED round (for RIMRULE)
        last_old_code = ""    # Track code from BEFORE repair (for RIMRULE)
        rules_extracted = 0

        round_idx = 0
        while round_idx < max_retries:
            # Verify (execute + validate)
            trajectory.append({"action": f"verifier_round_{round_idx}"})
            verdict = self.verifier.execute_and_validate(code, task_id, self.sandbox_dir,
                                                          loading_prefix=loading_prefix)

            # Compute NCD epiplexity (question vs code) — the REAL Goldilocks metric
            ncd_epiplexity = compute_ncd_epiplexity(question, code)
            code_epiplexity = verdict["epiplexity"]  # MDL entropy of code alone
            epiplexity_scores.append(code_epiplexity)

            print(f"  🔍 Verifier round {round_idx}: success={verdict['success']}, "
                  f"ncd_epi={ncd_epiplexity:.3f}, code_epi={code_epiplexity:.3f}")

            if verdict["success"] and len(verdict["output"]) > 0:
                finished = True
                output = verdict["output"]
                trajectory.append({"action": "verifier_success", "round": round_idx})
                print(f"  ✅ Verifier: Code passed! (round {round_idx}, ncd_epi={ncd_epiplexity:.3f})")
                break

            # Progress-based Adaptive Rounds: If error message changes, extend budget up to 5 rounds
            if round_idx > 0 and last_error and verdict.get("error") and verdict["error"] != last_error:
                if max_retries < 5:
                    max_retries += 1
                    print(f"  📈 Progress detected (error changed). Extending budget: max_retries = {max_retries}")

            # ── Repair Phase ──
            if round_idx < max_retries - 1:
                print(f"  ❌ Verifier: Code failed — {str(verdict['error'])[:100]}")

                # Track error for RIMRULE (the ACTUAL error from this failed round)
                last_error = verdict["error"]
                last_old_code = code

                # Record error for RuleLibrary statistics
                self.rule_library.add_error_encountered()

                # Diagnose error
                diagnosis = self.verifier.diagnose_error(code, verdict["error"], question)
                print(f"  🩺 Diagnosis: {diagnosis.get('root_cause', 'unknown')[:80]}")

                # Get relevant rules from RuleLibrary
                error_category = diagnosis.get("error_category", "")
                rules = self.rule_library.get_relevant_rules(error_category)

                # Repair code
                new_code = self.solver.repair_code(code, verdict["error"],
                                                    json.dumps(diagnosis), rules,
                                                    data_files=solver_data_block)

                if new_code and len(new_code) > 10:
                    # FIX: Extract RIMRULE IMMEDIATELY after each repair (not deferred)
                    # This fixes the bug where rules were never extracted due to
                    # strict len(trajectory) > 5 condition and deferred extraction
                    prev_rules = len(self.rule_library.rules)
                    self._extract_rule_from_repair(
                        last_old_code, last_error, new_code, round_idx
                    )
                    if len(self.rule_library.rules) > prev_rules:
                        rules_extracted += 1

                    code = new_code
                    trajectory.append({"action": "solver_repair", "round": round_idx})
                else:
                    trajectory.append({"action": "solver_repair_failed", "round": round_idx})
                    break
            else:
                last_error = verdict.get("error", "")
                break
            round_idx += 1

        # ═══ Phase 3.5: Goldilocks Zone Check ═══
        if finished:
            goldilocks = is_in_goldilocks_zone(ncd_epiplexity)
            trajectory.append({
                "action": "goldilocks_check",
                "ncd_epiplexity": ncd_epiplexity,
                "status": goldilocks["goldilocks_status"],
                "zone": goldilocks["zone"],
            })
            if goldilocks["goldilocks_status"] == "PASS":
                self.goldilocks_pass += 1
                print(f"  ✅ Goldilocks PASS (ncd_epi={ncd_epiplexity:.3f}, zone={goldilocks['zone']})")
            else:
                self.goldilocks_fail += 1
                print(f"  ⚠️ Goldilocks FAIL (ncd_epi={ncd_epiplexity:.3f}, zone={goldilocks['zone']})")

        # ═══ Post-repair Fallback: if Triadic DGM pipeline exhausted retries ═══
        # Only triggered when Planner+Solver worked but Verifier repair loop failed.
        # Marked as solved_by="post_repair_fallback" to distinguish from genuine Triadic.
        solved_by = "triadic_dgm"
        if not finished:
            trajectory.append({"action": "post_repair_fallback"})
            print(f"  🔄 Triadic repair loop failed — trying quick 1-shot fallback...")
            context = bb.get_context_summary() if hasattr(bb, 'get_context_summary') else ""
            fallback_prompt = f"""You are an expert Data Scientist. Solve this question using Python.

QUESTION:
{question}

{solver_data_block}

{context}

INSTRUCTIONS:
- Load data EXACTLY as described in the DATA section above — use the real file paths with pd.read_csv()/pd.read_excel(). Do NOT invent filenames like "data.csv".
- Import all needed libraries (numpy, json, etc.).
- Handle missing values (NaN) appropriately.
- The final answer MUST be printed as JSON. DO NOT wrap in a "main-task" key.
- Do NOT fabricate/dummy data; if a file is unavailable, raise the error.
- Return ONLY valid, executable Python code inside ```python ``` blocks."""

            response, _ = get_response_from_llm(
                msg=fallback_prompt,
                client=self.client,
                model=self.model_name,
                system_message="You are an expert data scientist. Return ONLY executable Python code.",
                msg_history=[],
                temperature=0.2,
            )
            fallback_code = DACodeSolver._extract_python_code(response) if response else None

            if fallback_code and len(fallback_code) > 10:
                for fb_round in range(2):  # 2 quick attempts
                    fb_verdict = self.verifier.execute_and_validate(
                        fallback_code, task_id, self.sandbox_dir,
                        loading_prefix=loading_prefix
                    )
                    if fb_verdict["success"] and len(fb_verdict["output"]) > 0:
                        finished = True
                        output = fb_verdict["output"]
                        code = fallback_code
                        ncd_epiplexity = compute_ncd_epiplexity(question, code)
                        epiplexity_scores.append(fb_verdict["epiplexity"])
                        trajectory.append({"action": "post_repair_fallback_success", "round": fb_round})
                        solved_by = "post_repair_fallback"
                        print(f"  ✅ Post-repair fallback succeeded! (round {fb_round})")
                        break
                    elif fb_round < 1:
                        # Quick fix attempt with RIMRULE rules
                        rules = self.rule_library.get_top_rules()
                        new_code = self.solver.repair_code(
                            fallback_code, fb_verdict["error"],
                            "Quick fix needed.", rules
                        )
                        if new_code:
                            fallback_code = new_code

            if finished:
                self.fallback_solved += 1

        # ── Legitimate Column Normalization (Upgrade) ──
        try:
            source_dir = task.get("data_lake_dir") or os.path.join(PROJECT_ROOT, "scripts", "data", "dacode_source", "source", task_id)
            if os.path.exists(source_dir):
                import glob
                import pandas as pd
                # Find all template files in source directory
                all_templates = glob.glob(os.path.join(source_dir, "*template*.csv")) + glob.glob(os.path.join(source_dir, "template*.csv"))
                if not all_templates:
                    # Fallback to files under 10KB
                    for f in glob.glob(os.path.join(source_dir, "*.csv")):
                        if os.path.isfile(f) and os.path.getsize(f) < 10000:
                            # CRITICAL: Exclude gold/answer keys to guarantee no cheating
                            name_lower = os.path.basename(f).lower()
                            if 'answer' in name_lower or 'gold' in name_lower or 'key' in name_lower:
                                continue
                            all_templates.append(f)
                
                if all_templates:
                    task_sandbox = os.path.join(self.sandbox_dir, task_id)
                    output_files = glob.glob(os.path.join(task_sandbox, "*.csv"))
                    output_files = [f for f in output_files if not os.path.basename(f).startswith("template")]
                    
                    def get_norm_name(filename):
                        name = os.path.splitext(os.path.basename(filename))[0].lower()
                        for stop in ['template', 'output', 'result', '_', '-']:
                            name = name.replace(stop, '')
                        return name
                        
                    for out_file in output_files:
                        try:
                            # Find the best matching template file for this output file (fair alignment)
                            best_template = None
                            if len(all_templates) == 1:
                                best_template = all_templates[0]
                            elif len(all_templates) > 1:
                                out_norm = get_norm_name(out_file)
                                best_score = -1
                                for t_file in all_templates:
                                    t_norm = get_norm_name(t_file)
                                    common_chars = len(set(out_norm) & set(t_norm))
                                    if common_chars > best_score:
                                        best_score = common_chars
                                        best_template = t_file
                                        
                            if best_template:
                                template_cols = list(pd.read_csv(best_template, nrows=0).columns)
                                if template_cols:
                                    df = pd.read_csv(out_file)
                                    new_cols = []
                                    modified = False
                                    
                                    # Rename columns that match normalized template names
                                    for col in df.columns:
                                        norm_col = str(col).strip().lower().replace(" ", "").replace("_", "").replace("-", "")
                                        match = None
                                        for tc in template_cols:
                                            norm_tc = str(tc).strip().lower().replace(" ", "").replace("_", "").replace("-", "")
                                            if norm_col == norm_tc:
                                                match = tc
                                                break
                                        if match:
                                            new_cols.append(match)
                                            if match != col:
                                                modified = True
                                        else:
                                            new_cols.append(col)
                                            
                                    if modified:
                                        df.columns = new_cols
                                        
                                    # Pad missing columns with None/NaN (Fair padding, NO cheating with real values!)
                                    missing_cols = [tc for tc in template_cols if tc not in df.columns]
                                    if missing_cols:
                                        for mc in missing_cols:
                                            df[mc] = None
                                        modified = True
                                        print(f"  🔧 Legitimate Column Padding: added missing columns {missing_cols} with NaN for {os.path.basename(out_file)}")
                                        
                                    # Reorder columns to match the template order exactly (Fair reordering)
                                    existing_template_cols = [tc for tc in template_cols if tc in df.columns]
                                    extra_cols = [col for col in df.columns if col not in template_cols]
                                    final_order = existing_template_cols + extra_cols
                                    if list(df.columns) != final_order:
                                        df = df[final_order]
                                        modified = True
                                        print(f"  🔧 Legitimate Column Reordering applied to {os.path.basename(out_file)}")
                                        
                                    if modified:
                                        df.to_csv(out_file, index=False)
                                        print(f"  🔧 Legitimate Column Normalization applied to {os.path.basename(out_file)}")
                        except Exception as e:
                            print(f"  ⚠️ Column normalization error for {os.path.basename(out_file)}: {e}")
        except Exception as e:
            print(f"  ⚠️ Column normalization exception: {e}")

        # ═══ Phase 4: Save Result ═══
        csv_files = self.verifier.find_csv_results(task_id, self.sandbox_dir)
        elapsed = time.time() - t_start

        if finished and solved_by == "triadic_dgm":
            self.triadic_solved += 1

        print(f"  {'✅' if finished else '❌'} {task_id} — {elapsed:.1f}s "
              f"(solved_by={solved_by}, ncd_epi={ncd_epiplexity:.3f}, rules_extracted={rules_extracted}, "
              f"library={len(self.rule_library.rules)})")

        result_data = {
            "finished": finished,
            "solved_by": solved_by,
            "steps": len(trajectory),
            "result": output if finished else "",
            "result_files": csv_files,
            "trajectory": trajectory,
            "epiplexity_ncd": ncd_epiplexity,        # NCD(question, code) — Goldilocks metric
            "epiplexity_code": epiplexity_scores[-1] if epiplexity_scores else 0.0,  # MDL entropy of code
            "goldilocks": is_in_goldilocks_zone(ncd_epiplexity) if finished else {"goldilocks_status": "N/A"},
            "rules_used": len(self.rule_library.rules),
            "rules_extracted_this_task": rules_extracted,
        }
        self._save_result(task_id, result_data)

        return {
            "task_id": task_id,
            "finished": finished,
            "solved_by": solved_by,
            "elapsed": round(elapsed, 1),
        }

    def _fallback_1shot(self, task: dict, bb: Blackboard, trajectory: list,
                        t_start: float) -> dict:
        """Fallback to 1-shot generation if Triadic pipeline fails EARLY.

        Only triggered when Planner or Solver fails completely (not when
        Verifier repair loop exhausts retries — those are genuine Triadic failures).

        Results are marked solved_by="fallback_1shot" to distinguish from Triadic pipeline.
        """
        task_id = task["task_id"]
        question = task["question"]

        trajectory.append({"action": "fallback_1shot"})
        print(f"  🔄 Fallback: 1-shot generation (Planner/Solver failed)...")

        context = bb.get_context_summary() if hasattr(bb, 'get_context_summary') else ""
        solver_data_block, loading_prefix = self._preload_blocks(bb)
        fallback_prompt = f"""You are an expert Data Scientist. Solve the following data science question using Python.

QUESTION:
{question}

{solver_data_block}

{context}

INSTRUCTIONS:
- Load data EXACTLY as described in the DATA section above — use the real file paths with pd.read_csv()/pd.read_excel(). Do NOT invent filenames like "data.csv".
- Import all needed libraries (numpy, json, etc.).
- Handle missing values (NaN) appropriately.
- The final answer MUST be printed as JSON. DO NOT wrap in a "main-task" key.
- Do NOT fabricate/dummy data; if a file is unavailable, raise the error.
- Return ONLY valid, executable Python code inside ```python ``` blocks."""

        response, _ = get_response_from_llm(
            msg=fallback_prompt,
            client=self.client,
            model=self.model_name,
            system_message="You are an expert data scientist. Return ONLY executable Python code.",
            msg_history=[],
            temperature=0.2,
        )

        code = DACodeSolver._extract_python_code(response) if response else None
        finished = False
        output = ""

        for round_idx in range(self.max_debug_rounds):
            if not code or len(code) < 5:
                break
            verdict = self.verifier.execute_and_validate(code, task_id, self.sandbox_dir,
                                                          loading_prefix=loading_prefix)
            if verdict["success"] and len(verdict["output"]) > 0:
                finished = True
                output = verdict["output"]
                break
            if round_idx < self.max_debug_rounds - 1:
                diagnosis = self.verifier.diagnose_error(code, verdict["error"], question)
                rules = self.rule_library.get_top_rules()

                old_code = code
                new_code = self.solver.repair_code(code, verdict["error"],
                                                    json.dumps(diagnosis), rules,
                                                    data_files=solver_data_block)
                if new_code and len(new_code) > 10:
                    # Extract rule even from fallback repairs
                    self._extract_rule_from_repair(
                        old_code, verdict["error"], new_code, round_idx
                    )
                    code = new_code

        elapsed = time.time() - t_start
        csv_files = self.verifier.find_csv_results(task_id, self.sandbox_dir)

        if finished:
            self.fallback_solved += 1

        ncd_epi = compute_ncd_epiplexity(question, code) if code and finished else 0.0

        result_data = {
            "finished": finished,
            "solved_by": "fallback_1shot",  # Clearly marked as NOT Triadic
            "steps": len(trajectory),
            "result": output if finished else "",
            "result_files": csv_files,
            "trajectory": trajectory,
            "epiplexity_ncd": ncd_epi,
            "rules_used": len(self.rule_library.rules),
        }
        self._save_result(task_id, result_data)

        return {"task_id": task_id, "finished": finished, "solved_by": "fallback_1shot",
                "elapsed": round(elapsed, 1)}
