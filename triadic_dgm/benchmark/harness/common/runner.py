"""Shared runner: run the Triadic DGM orchestrator over a benchmark manifest,
then re-execute each produced program in the task workspace so the answer
(stdout for DABench / output files for SAB) is captured for grading.

Why re-execute: `TriadicDGMOrchestrator.run_task` returns the *code* of the best
candidate, not the computed answer. The inner loop already executes candidates in
the sandbox for verification, but it does not return their stdout. So we run the
final code once more, with cwd = the task workspace, and record stdout + the files
it produced.

Concurrency: each worker thread gets its own orchestrator + sandbox + memory bank
(the orchestrator holds a single sandbox whose cwd is mutated per task, and the
RIMRULE memory bank is not thread-safe). The LLM client is shared. With
num_workers=1 (default) a single memory bank accumulates across tasks, exercising
Triadic DGM's cross-task learning.
"""
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from triadic_dgm.benchmark.DGM_orchestrator import TriadicDGMOrchestrator
from triadic_dgm.benchmark.interfaces.llm_client import ILLMClient
from triadic_dgm.benchmark.harness.common.sandbox import LocalWorkspaceSandbox
from triadic_dgm.memory.rimrule_memory import RimruleMemoryBank


class TriadicRunner:
    def __init__(
        self,
        llm_client: ILLMClient,
        max_inner_retries: int = 3,
        exec_timeout: int = 300,
        num_workers: int = 1,
        solver_instruction: str = "",
        script_dir: str = "./_triadic_eval_scripts",
        memory_archive_dir: str = "./_triadic_eval_memory",
        epi_min: float = 0.0,
        epi_max: float = 10.0,
    ):
        self.llm_client = llm_client
        self.max_inner_retries = max_inner_retries
        self.exec_timeout = exec_timeout
        self.num_workers = max(1, num_workers)
        self.solver_instruction = solver_instruction
        self.script_dir = script_dir
        self.memory_archive_dir = memory_archive_dir
        os.makedirs(self.memory_archive_dir, exist_ok=True)
        self._tls = threading.local()
        self._worker_counter = 0
        self._counter_lock = threading.Lock()

        # The verifier's epiplexity Goldilocks window defaults to [0.5, 1.8], which is
        # calibrated for code-evolution (Polyglot) and rejects *correct* DABench/SAB
        # solutions (a typical DA solution scores ~1.8). For an eval that measures raw
        # task-solving ability, widen the window so any non-degenerate runtime-passing
        # candidate is accepted. Applied to the global HYPERPARAMS singleton before any
        # worker builds its orchestrator.
        from triadic_dgm.benchmark.core.evolution_hyperparams import HYPERPARAMS
        HYPERPARAMS.epiplexity_min = epi_min
        HYPERPARAMS.epiplexity_max = epi_max

    # ---- per-thread orchestrator/sandbox/memory ----
    def _get_worker(self):
        if not hasattr(self._tls, "worker_id"):
            with self._counter_lock:
                self._worker_counter += 1
                wid = self._worker_counter
            self._tls.worker_id = wid
            sb = LocalWorkspaceSandbox(
                script_dir=os.path.join(self.script_dir, f"w{wid}"),
                timeout=self.exec_timeout,
            )
            mem = RimruleMemoryBank(
                archive_path=os.path.join(self.memory_archive_dir, f"rimrule_w{wid}.json")
            )
            orch = TriadicDGMOrchestrator(
                self.llm_client, sb, mem, max_inner_retries=self.max_inner_retries
            )
            self._tls.sandbox = sb
            self._tls.orchestrator = orch
        return self._tls.orchestrator, self._tls.sandbox

    # ---- single task ----
    def run_one(self, task: Dict[str, Any]) -> Dict[str, Any]:
        task_id = task.get("task_id", "task")
        workspace = task.get("workspace", ".")
        description = task.get("description", "")
        orch, sandbox = self._get_worker()
        sandbox.set_workspace(workspace)
        # Reset the blackboard so file agents from a previous task (a different
        # workspace) don't leak into this task's context graph.
        if hasattr(orch, "blackboard"):
            orch.blackboard.agents = []
            orch.blackboard.responses = []

        t0 = time.time()
        result: Dict[str, Any] = {"task_id": task_id}
        try:
            orch_result = orch.run_task(description, workspace, self.solver_instruction)
        except Exception as e:  # noqa: BLE001 - never let one task kill the batch
            orch_result = {"status": "RUNNER_EXCEPTION", "error": f"{e}", "code": ""}
        result["status"] = orch_result.get("status")
        result["epiplexity_score"] = orch_result.get("epiplexity_score")
        result["attempts"] = orch_result.get("attempts")
        code = orch_result.get("code") or ""
        result["code"] = code

        # Re-execute the final code in the workspace to capture the answer.
        baseline = set(os.listdir(workspace)) if os.path.isdir(workspace) else set()
        if code.strip():
            ok, out = sandbox.execute(code, task_id=task_id)
            result["exec_ok"] = bool(ok)
            result["stdout"] = out or ""
            if not ok:
                result["exec_traceback"] = out or ""
        else:
            result["exec_ok"] = False
            result["stdout"] = ""
            result["exec_traceback"] = orch_result.get("error", "no code produced")

        after = set(os.listdir(workspace)) if os.path.isdir(workspace) else set()
        result["produced_files"] = sorted(after - baseline)
        result["elapsed_sec"] = round(time.time() - t0, 1)
        # carry through benchmark-specific fields the grader needs
        for k in ("gold", "question", "output_fname", "level", "domain"):
            if k in task:
                result[k] = task[k]
        return result

    # ---- batch ----
    def run_all(
        self,
        tasks: List[Dict[str, Any]],
        results_dir: str,
        resume: bool = True,
        progress_cb=None,
    ) -> List[Dict[str, Any]]:
        os.makedirs(results_dir, exist_ok=True)
        run_jsonl = os.path.join(results_dir, "run.jsonl")
        done_ids = set()
        if not resume and os.path.exists(run_jsonl):
            # Fresh run: discard any leftover results from a previous (possibly killed) run
            # so the summary isn't polluted with stale/duplicate task records.
            try:
                os.remove(run_jsonl)
            except OSError:
                pass
        if resume and os.path.exists(run_jsonl):
            with open(run_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            done_ids.add(json.loads(line).get("task_id"))
                        except json.JSONDecodeError:
                            pass
        pending = [t for t in tasks if t.get("task_id") not in done_ids]
        if done_ids:
            print(f"[RUNNER] resuming: {len(done_ids)} already done, {len(pending)} pending")

        write_lock = threading.Lock()

        def _append(res: Dict[str, Any]):
            with write_lock:
                with open(run_jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")

        completed = 0
        total = len(pending)
        if total == 0:
            return []

        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            futures = {pool.submit(self.run_one, t): t for t in pending}
            for fut in as_completed(futures):
                t = futures[fut]
                try:
                    res = fut.result()
                except Exception as e:  # noqa: BLE001
                    res = {"task_id": t.get("task_id"), "status": "RUNNER_EXCEPTION",
                           "error": f"{e}", "stdout": "", "exec_ok": False}
                _append(res)
                completed += 1
                if progress_cb:
                    try:
                        progress_cb(completed, total, res)
                    except Exception:  # noqa: BLE001
                        pass
                status = res.get("status")
                print(f"[RUNNER] {completed}/{total}  {res.get('task_id')}  status={status}  "
                      f"exec_ok={res.get('exec_ok')}  {res.get('elapsed_sec')}s")
        return []
