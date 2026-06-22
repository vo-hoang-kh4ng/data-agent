"""Workspace-aware local subprocess sandbox.

`LocalSubprocessSandbox` (the one the orchestrator wires by default) writes each
candidate script into a single shared dir and runs it with the *current* working
directory, so the agent's `pd.read_csv("x.csv")` only works if x.csv happens to be
in that cwd. For DABench / ScienceAgentBench the data files live in a per-task
workspace, so we need an ISandbox that runs each script with `cwd = workspace`.

This module provides `LocalWorkspaceSandbox`, which:
  * runs generated code with cwd set to the task workspace (so relative paths and
    output files resolve correctly),
  * uses a longer timeout (data-science / science code is slow),
  * defensively strips ```python markdown fences the LLM may wrap around code.
"""
import os
import re
import subprocess
import sys
from typing import Tuple

from triadic_dgm.benchmark.interfaces.sandbox import ISandbox

_FENCE_RE = re.compile(r"^\s*```(?:python|py|Python)?\s*\n(.*?)\n```\s*$", re.DOTALL)


def _strip_fences(code: str) -> str:
    """If the model wrapped the whole snippet in a ```python fence, unwrap it."""
    m = _FENCE_RE.match(code.strip())
    return m.group(1) if m else code


class LocalWorkspaceSandbox(ISandbox):
    """Execute Python code in a chosen working directory."""

    def __init__(self, script_dir: str = "./_triadic_eval_scripts", timeout: int = 300):
        # Resolve to an ABSOLUTE path now: the script is written here and later run by a
        # subprocess whose cwd is the per-task workspace. A relative script_dir would be
        # written relative to this process's cwd but resolved by the child relative to the
        # workspace → "can't open file". Absolute path is found regardless of child cwd.
        self.script_dir = os.path.abspath(script_dir)
        self.timeout = timeout
        self.cwd = os.getcwd()
        os.makedirs(self.script_dir, exist_ok=True)

    def set_workspace(self, workspace_dir: str) -> None:
        self.cwd = os.path.abspath(workspace_dir) if workspace_dir else os.getcwd()

    def execute(self, code: str, task_id: str = "eval", timeout: int = None) -> Tuple[bool, str]:
        code = _strip_fences(code or "")
        if not code.strip():
            return False, "Empty code."

        # NOTE: the original LocalSubprocessSandbox skipped execution when the code
        # contained "#include" / "package " / "func main" / "public class" (to defer
        # compiled languages to Docker in the Polyglot benchmark). That heuristic is
        # WRONG here: LLM-written Python routinely contains the word "package" in
        # comments/strings, which silently skipped real Python programs — both in the
        # orchestrator's inner verifier (false runtime-pass) and in this re-execution
        # (no output file). DABench/SAB tasks are always Python, so we just run them.

        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(task_id))[:80]
        script_path = os.path.join(self.script_dir, f"ds_{safe_id}.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)

        os.makedirs(self.cwd, exist_ok=True)
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=timeout if timeout is not None else self.timeout,
                cwd=self.cwd,
                encoding="utf-8",
                errors="replace",
            )
            combined = (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")
            if result.returncode == 0:
                return True, combined.strip()
            return False, ((result.stderr or "") + "\n" + (result.stdout or "")).strip()
        except subprocess.TimeoutExpired:
            return False, f"TimeoutError: execution exceeded the time limit."
        except Exception as e:  # pragma: no cover - defensive
            return False, f"SandboxError: {e}"
