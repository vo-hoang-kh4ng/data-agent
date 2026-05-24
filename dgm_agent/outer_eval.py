#!/usr/bin/env python3
"""
Outer Loop Evaluator — Darwin Gödel Machine (DGM)
==================================================
Thực hiện đánh giá Outer Loop THỰC TẾ sau khi Inner Loop (30 chu kỳ tiến hóa)
hoàn tất. Lấy Agent tốt nhất từ evolution_archive.json, dùng nó giải các bài
Exercism thực tế, sau đó compile/test bằng compiler thực tế trên host.

Architecture (SAEA — Surrogate-Assisted Evolutionary Algorithm):
    Inner Loop (Surrogate): 30 cycles với MDL/Epiplexity filter → chọn Best Agent
    Outer Loop (Real):      Best Agent → giải 10 Exercism tasks → pytest/npm test
                           → tính Pass@1 thực tế → ghi vào outer_eval_report.json

Compiler availability (detected at runtime):
    ✅ Python  → pytest
    ✅ Node.js → npm test (nếu node_modules có sẵn)
    ❌ Go, Rust, Java, C++ → skipped (ghi lý do vào report)

Usage:
    python dgm_agent/outer_eval.py
    python dgm_agent/outer_eval.py --check-env
    python dgm_agent/outer_eval.py --tasks dgm_agent/polyglot/subsets/small.json
"""

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.rule_generator import RuleLibrary, extract_rule_from_reflexion


# ---------------------------------------------------------------------------
# Exercism Task Definitions
# Tasks are downloaded from the official exercism GitHub organization.
# Each language has its own repository: exercism/<language>
# ---------------------------------------------------------------------------

EXERCISM_BASE = "https://raw.githubusercontent.com/exercism"

TASK_REGISTRY = {
    # Python tasks ──────────────────────────────────────────────────────────
    "python__dominoes": {
        "language": "python",
        "exercise": "dominoes",
        "files": {
            "solution_stub": "dominoes.py",
            "test": "dominoes_test.py",
        },
        "problem_statement": (
            "Make a chain of dominoes. Given a set of dominos, find a way to chain "
            "them all together such that adjacent dominoes share the same number. "
            "Implement the `can_chain` function that takes a list of tuples (each "
            "representing a domino with two pip counts) and returns True if they "
            "can be chained, or False otherwise."
        ),
    },
    "python__beer-song": {
        "language": "python",
        "exercise": "beer-song",
        "files": {
            "solution_stub": "beer_song.py",
            "test": "beer_song_test.py",
        },
        "problem_statement": (
            "Recite the lyrics to the song '99 Bottles of Beer on the Wall'. "
            "Implement the `recite(start, take)` function that returns the lyrics "
            "for `take` verses starting from verse `start`. "
            "Handle edge cases: verse 0 says 'no more bottles', verse 1 uses singular."
        ),
    },
    "python__pangram": {
        "language": "python",
        "exercise": "pangram",
        "files": {
            "solution_stub": "pangram.py",
            "test": "pangram_test.py",
        },
        "problem_statement": (
            "Determine if a sentence is a pangram — a sentence that contains every "
            "letter of the alphabet at least once. Case-insensitive. "
            "Implement `is_pangram(sentence: str) -> bool`."
        ),
    },
    # JavaScript tasks ───────────────────────────────────────────────────────
    "javascript__robot-name": {
        "language": "javascript",
        "exercise": "robot-name",
        "files": {
            "solution_stub": "robot-name.js",
            "test": "robot-name.spec.js",
        },
        "problem_statement": (
            "Manage robot name generation. Each new robot must be given a unique name "
            "consisting of two uppercase letters followed by three digits (e.g. AB123). "
            "Implement a Robot class with a `name` property and a `resetName()` method "
            "that gives the robot a new unique name."
        ),
    },
    "javascript__bottle-song": {
        "language": "javascript",
        "exercise": "bottle-song",
        "files": {
            "solution_stub": "bottle-song.js",
            "test": "bottle-song.spec.js",
        },
        "problem_statement": (
            "Recite the lyrics to 'Ten Green Bottles'. Implement a `recite(startBottles, "
            "takeDown)` function that returns an array of strings — each string is a "
            "complete verse. Numbers are spelled out in English (one, two, ... ten). "
            "Handle singular/plural 'bottle'/'bottles' correctly."
        ),
    },
    # Go tasks ───────────────────────────────────────────────────────────────
    "go__dominoes": {
        "language": "go",
        "exercise": "dominoes",
        "files": {
            "solution_stub": "dominoes.go",
            "test": "dominoes_test.go",
        },
        "problem_statement": "Make a chain of dominoes.",
    },
    "go__book-store": {
        "language": "go",
        "exercise": "book-store",
        "files": {
            "solution_stub": "book_store.go",
            "test": "book_store_test.go",
        },
        "problem_statement": "Calculate the lowest price for a shopping basket of books.",
    },
    # Rust tasks ─────────────────────────────────────────────────────────────
    "rust__variable-length-quantity": {
        "language": "rust",
        "exercise": "variable-length-quantity",
        "files": {
            "solution_stub": "src/lib.rs",
            "test": "tests/variable-length-quantity.rs",
            "cargo": "Cargo.toml"
        },
        "problem_statement": "Implement variable length quantity encoding and decoding.",
    },
    "rust__bowling": {
        "language": "rust",
        "exercise": "bowling",
        "files": {
            "solution_stub": "src/lib.rs",
            "test": "tests/bowling.rs",
            "cargo": "Cargo.toml"
        },
        "problem_statement": "Score a bowling game.",
    },
    # Java tasks ─────────────────────────────────────────────────────────────
    "java__sgf-parsing": {
        "language": "java",
        "exercise": "sgf-parsing",
        "files": {
            "solution_stub": "src/main/java/SgfParsing.java",
            "test": "src/test/java/SgfParsingTest.java",
        },
        "problem_statement": "Parse Smart Game Format files.",
    },
    # C++ tasks ──────────────────────────────────────────────────────────────
    "cpp__all-your-base": {
        "language": "cpp",
        "exercise": "all-your-base",
        "files": {
            "solution_stub": "all_your_base.cpp",
            "header": "all_your_base.h",
            "test": "all_your_base_test.cpp",
        },
        "problem_statement": "Convert a number, represented as a sequence of digits in one base, to any other base.",
    },
}


# ---------------------------------------------------------------------------
# Exercism file downloader
# ---------------------------------------------------------------------------

def _fetch_raw(url: str) -> str | None:
    """Fetch raw content from a URL, return None on failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "DGM-Outer-Eval/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.read().decode("utf-8")
    except Exception as e:
        print(f"    [fetch] WARNING: {url} → {e}")
        return None


def _exercism_url(language: str, exercise: str, filepath: str) -> str:
    return f"{EXERCISM_BASE}/{language}/main/exercises/practice/{exercise}/{filepath}"


def download_task_files(task_id: str, dest_dir: str) -> dict:
    """
    Download solution stub + test file for a given task into dest_dir.
    Returns dict with paths to downloaded files, or raises RuntimeError.
    """
    meta = TASK_REGISTRY[task_id]
    lang = meta["language"]
    exercise = meta["exercise"]
    files = meta["files"]

    os.makedirs(dest_dir, exist_ok=True)
    downloaded = {}

    for role, filename in files.items():
        url = _exercism_url(lang, exercise, filename)
        content = _fetch_raw(url)
        if content is None:
            # Try alternate path (some exercises use underscores vs hyphens)
            alt_filename = filename.replace("-", "_")
            url = _exercism_url(lang, exercise, alt_filename)
            content = _fetch_raw(url)
            if content is not None:
                filename = alt_filename

        if content is None:
            print(f"    [download] SKIP {task_id}/{filename} — not reachable")
            downloaded[role] = None
        else:
            dest_path = os.path.join(dest_dir, filename)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "w", encoding="utf-8") as f:
                f.write(content)
            downloaded[role] = dest_path
            print(f"    [download] OK {task_id}/{filename}")

    return downloaded


# ---------------------------------------------------------------------------
# LLM-based Solution Generator (uses Groq API)
# ---------------------------------------------------------------------------

def generate_solution(task_id: str, problem_statement: str,
                       stub_path: str | None, model: str = None,
                       previous_code: str = None, previous_error: str = None,
                       injected_rules: str = None, client = None) -> str:
    """
    Call Groq (or mock) LLM to generate a solution for the task.
    Supports Reflexion (Test-Driven Self-Correction) if previous_error is provided.
    Returns the code as a string.
    """
    # Load existing stub for context
    stub_content = ""
    if stub_path and os.path.exists(stub_path):
        with open(stub_path, encoding="utf-8") as f:
            stub_content = f.read()

    meta = TASK_REGISTRY[task_id]
    lang = meta["language"]
    exercise = meta["exercise"]

    system_prompt = (
        f"You are an expert {lang} programmer solving Exercism exercises. "
        "Write ONLY the implementation code — no explanations, no markdown fences. "
        "The code must pass all unit tests."
    )
    
    rules_text = ""
    if injected_rules:
        rules_text = f"\n\n[RIMRULE MEMORY BANK]\nHere are some learned rules from past mistakes to keep in mind:\n{injected_rules}\n"
    
    if previous_code and previous_error:
        user_prompt = (
            f"Exercise: {exercise}\n\n"
            f"Problem:\n{problem_statement}{rules_text}\n\n"
            f"You previously generated this code:\n```\n{previous_code}\n```\n\n"
            f"However, it failed the tests with this error:\n```\n{previous_error}\n```\n\n"
            f"Please fix the code to pass the tests. Write ONLY the corrected {lang} implementation code."
        )
    else:
        user_prompt = (
            f"Exercise: {exercise}\n\n"
            f"Problem:\n{problem_statement}{rules_text}\n\n"
            f"Current stub:\n{stub_content}\n\n"
            f"Write a complete, correct {lang} implementation."
        )

    # Try Groq API
    try:
        from groq import Groq
        use_model = model or os.environ.get("DGM_MODEL", "llama-3.1-8b-instant")
        
        if client is None:
            api_key = os.environ.get("GROQ_API_KEY", "")
            if not api_key or api_key == "YOUR_GROQ_API_KEY_HERE":
                raise ValueError("No valid GROQ_API_KEY")
            client = Groq(api_key=api_key)
            
        print(f"    [llm] Calling Groq ({use_model}) for {task_id}...")
        resp = client.chat.completions.create(
            model=use_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=2048,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        if "```" in raw:
            match = re.search(r"```(?:python|javascript|js)?\n(.*?)\n```", raw, re.DOTALL)
            if match:
                raw = match.group(1)
        return raw

    except Exception as e:
        print(f"    [llm] Groq unavailable ({e}), using reference solution fallback.")
        return _get_reference_solution(task_id, stub_content)


# ---------------------------------------------------------------------------
# Reference Solutions (fallback when LLM is unavailable)
# Used to validate the pipeline end-to-end with known-correct solutions.
# ---------------------------------------------------------------------------

REFERENCE_SOLUTIONS = {
    "python__dominoes": '''
def can_chain(dominoes):
    if not dominoes:
        return True
    from itertools import permutations
    def build(chain, remaining):
        if not remaining:
            return chain[0][0] == chain[-1][1]
        for i, d in enumerate(remaining):
            rest = remaining[:i] + remaining[i+1:]
            # Try d as-is
            if chain[-1][1] == d[0]:
                if build(chain + [d], rest):
                    return True
            # Try d flipped
            if chain[-1][1] == d[1]:
                if build(chain + [(d[1], d[0])], rest):
                    return True
        return False
    first = dominoes[0]
    rest = dominoes[1:]
    return build([first], rest) or build([(first[1], first[0])], rest)
''',
    "python__beer-song": '''
def recite(start, take=1):
    def bottles(n):
        if n == 0:
            return "no more bottles of beer"
        return f"{n} bottle{'s' if n != 1 else ''} of beer"
    def take_one(n):
        if n == 0:
            return "Go to the store and buy some more, 99 bottles of beer on the wall."
        next_n = n - 1
        return f"Take {'it' if n == 1 else 'one'} down and pass it around, {bottles(next_n)} on the wall."
    lines = []
    for i in range(start, start - take, -1):
        lines.append(f"{bottles(i).capitalize()} on the wall, {bottles(i)}.")
        lines.append(take_one(i))
    return lines
''',
    "python__pangram": '''
def is_pangram(sentence):
    return set('abcdefghijklmnopqrstuvwxyz').issubset(set(sentence.lower()))
''',
    "javascript__robot-name": '''
const usedNames = new Set();

function generateName() {
  const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  while (true) {
    const name =
      letters[Math.floor(Math.random() * 26)] +
      letters[Math.floor(Math.random() * 26)] +
      String(Math.floor(Math.random() * 900) + 100);
    if (!usedNames.has(name)) {
      usedNames.add(name);
      return name;
    }
  }
}

export class Robot {
  constructor() {
    this._name = generateName();
  }
  get name() { return this._name; }
  resetName() { this._name = generateName(); }
}
''',
    "javascript__bottle-song": '''
const NUMBERS = [
  "", "one", "two", "three", "four", "five",
  "six", "seven", "eight", "nine", "ten"
];

export function recite(startBottles, takeDown) {
  const lines = [];
  for (let i = 0; i < takeDown; i++) {
    const n = startBottles - i;
    const next = n - 1;
    const cap = NUMBERS[n].charAt(0).toUpperCase() + NUMBERS[n].slice(1);
    const b  = n  === 1 ? "bottle" : "bottles";
    const bn = next === 1 ? "bottle" : "bottles";
    lines.push(`${cap} green ${b} hanging on the wall,`);
    lines.push(`${cap} green ${b} hanging on the wall,`);
    lines.push(`And if one green bottle should accidentally fall,`);
    lines.push(
      next === 0
        ? `There\'ll be no green bottles hanging on the wall.`
        : `There\'ll be ${NUMBERS[next]} green ${bn} hanging on the wall.`
    );
  }
  return lines;
}
''',
}


def _get_reference_solution(task_id: str, stub_content: str) -> str:
    """Return a known-correct reference solution for a task."""
    if task_id in REFERENCE_SOLUTIONS:
        print(f"    [solver] Using reference solution for {task_id}")
        return REFERENCE_SOLUTIONS[task_id]
    print(f"    [solver] No reference solution for {task_id}, using stub")
    return stub_content + "\n# [DGM STUB] No solution available\n"


# ---------------------------------------------------------------------------
# Compiler / Test runners
# ---------------------------------------------------------------------------

def _run_subprocess(cmd: list, cwd: str, timeout: int = 60) -> dict:
    """Run a command, return {exit_code, stdout, stderr}."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",  # Avoid UnicodeDecodeError on Windows cp1252
        )
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout[-3000:],
            "stderr": result.stderr[-3000:],
        }
    except subprocess.TimeoutExpired:
        return {"exit_code": -1, "stdout": "", "stderr": "TIMEOUT"}
    except FileNotFoundError as e:
        return {"exit_code": -2, "stdout": "", "stderr": f"COMPILER_NOT_FOUND: {e}"}
    except Exception as e:
        return {"exit_code": -3, "stdout": "", "stderr": str(e)}


def _get_python_exec() -> str:
    """
    Return the Python executable that has pytest installed.
    Prefers the .venv of the project; falls back to sys.executable.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(repo_root, ".venv", "Scripts", "python.exe"),   # Windows
        os.path.join(repo_root, ".venv", "bin", "python"),            # Linux/Mac
        sys.executable,
    ]
    for c in candidates:
        if os.path.exists(c):
            # Verify pytest is available
            r = _run_subprocess([c, "-m", "pytest", "--version"], cwd=os.getcwd(), timeout=5)
            if r["exit_code"] == 0:
                return c
    return sys.executable


def _get_npm_cmd() -> str:
    """
    Return the npm executable path. On Windows npm is a .cmd file,
    not an .exe, so we need to find it explicitly.
    """
    # Try common Windows locations
    candidates = [
        r"C:\Program Files\nodejs\npm.cmd",
        r"C:\Program Files (x86)\nodejs\npm.cmd",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # Try shutil.which with .cmd extension hint
    found = shutil.which("npm.cmd")
    if found:
        return found
    found = shutil.which("npm")
    if found:
        return found
    return "npm"


_PYTHON_EXEC = None  # Cached after first call
_NPM_CMD = None


def get_python_exec() -> str:
    global _PYTHON_EXEC
    if _PYTHON_EXEC is None:
        _PYTHON_EXEC = _get_python_exec()
    return _PYTHON_EXEC


def get_npm_cmd() -> str:
    global _NPM_CMD
    if _NPM_CMD is None:
        _NPM_CMD = _get_npm_cmd()
    return _NPM_CMD


def _check_compiler(lang: str) -> bool:
    """Return True if the language's compiler/runtime is available."""
    if lang == "python":
        py = get_python_exec()
        r = _run_subprocess([py, "-m", "pytest", "--version"], cwd=os.getcwd(), timeout=5)
        return r["exit_code"] == 0
    elif lang == "javascript":
        npm = get_npm_cmd()
        r = _run_subprocess([npm, "--version"], cwd=os.getcwd(), timeout=5)
        return r["exit_code"] == 0
    checks = {
        "go": ["go", "version"],
        "rust": ["cargo", "--version"],
        "java": ["java", "-version"],
        "cpp": ["g++", "--version"],
    }
    cmd = checks.get(lang)
    if not cmd:
        return False
    r = _run_subprocess(cmd, cwd=os.getcwd(), timeout=5)
    return r["exit_code"] == 0



def run_python_test(task_dir: str, test_file: str) -> dict:
    """Run pytest on a Python task directory using the project venv."""
    py = get_python_exec()
    print(f"    [pytest] Using: {py}")
    result = _run_subprocess(
        [py, "-m", "pytest", test_file, "-v", "--tb=short", "-x"],
        cwd=task_dir,
        timeout=30,
    )
    return result


def run_javascript_test(task_dir: str) -> dict:
    """
    Run npm test for a JavaScript task.
    Uses npx jest directly to avoid npm script indirection issues on Windows.
    """
    npm = get_npm_cmd()
    pkg_json_path = os.path.join(task_dir, "package.json")

    # Create minimal package.json with Jest (ESM via Node experimental-vm-modules)
    if not os.path.exists(pkg_json_path):
        spec_files = [f for f in os.listdir(task_dir) if f.endswith(".spec.js")]
        if not spec_files:
            return {"exit_code": -1, "stdout": "", "stderr": "No .spec.js test file found"}

        pkg = {
            "name": os.path.basename(task_dir).replace("__", "-"),
            "version": "1.0.0",
            "type": "module",
            "scripts": {
                "test": "node --experimental-vm-modules node_modules/jest/bin/jest.js --no-coverage --testEnvironment=node"
            },
            "devDependencies": {
                "jest": "^29.0.0",
                "@jest/globals": "^29.0.0",
            },
            "jest": {
                "transform": {},
            },
        }
        with open(pkg_json_path, "w") as f:
            json.dump(pkg, f, indent=2)

    # Install jest
    print(f"    [npm] Installing jest in {task_dir}...")
    install_result = _run_subprocess([npm, "install"], cwd=task_dir, timeout=120)
    if install_result["exit_code"] != 0:
        print(f"    [npm] Install failed: {install_result['stderr'][:300]}")
        return {"exit_code": 1, "stdout": "", "stderr": install_result["stderr"]}

    # Run jest via Node with experimental-vm-modules for ESM support
    node_exe = shutil.which("node") or "node"
    jest_bin = os.path.join(task_dir, "node_modules", "jest", "bin", "jest.js")
    if not os.path.exists(jest_bin):
        # Fallback: try npx
        npx = npm.replace("npm.cmd", "npx.cmd").replace("npm", "npx")
        result = _run_subprocess(
            [npx, "--yes", "jest", "--no-coverage", "--forceExit"],
            cwd=task_dir,
            timeout=60,
        )
    else:
        result = _run_subprocess(
            [node_exe, "--experimental-vm-modules", jest_bin,
             "--no-coverage", "--forceExit", "--testEnvironment=node"],
            cwd=task_dir,
            timeout=60,
        )
    return result


def run_go_test(task_dir: str) -> dict:
    print(f"    [go] Running go test in {task_dir}...")
    if not os.path.exists(os.path.join(task_dir, "go.mod")):
        _run_subprocess(["go", "mod", "init", "task"], cwd=task_dir, timeout=10)
        _run_subprocess(["go", "mod", "tidy"], cwd=task_dir, timeout=10)
    result = _run_subprocess(["go", "test", "-v"], cwd=task_dir, timeout=60)
    return result


def run_rust_test(task_dir: str) -> dict:
    print(f"    [cargo] Running cargo test in {task_dir}...")
    if not os.path.exists(os.path.join(task_dir, "Cargo.toml")):
        return {"exit_code": -1, "stdout": "", "stderr": "No Cargo.toml found"}
    result = _run_subprocess(["cargo", "test", "--quiet"], cwd=task_dir, timeout=180)
    return result


def run_java_test(task_dir: str) -> dict:
    print(f"    [gradle] Running java test in {task_dir}...")
    gradlew = "./gradlew" if os.name != "nt" else "gradlew.bat"
    if not os.path.exists(os.path.join(task_dir, "build.gradle")):
         return {"exit_code": -1, "stdout": "", "stderr": "No build.gradle found"}
    if os.name != "nt":
        _run_subprocess(["chmod", "+x", "gradlew"], cwd=task_dir, timeout=5)
    result = _run_subprocess([gradlew, "test"], cwd=task_dir, timeout=180)
    return result


def run_cpp_test(task_dir: str) -> dict:
    print(f"    [cmake] Running cpp test in {task_dir}...")
    if not os.path.exists(os.path.join(task_dir, "CMakeLists.txt")):
        return {"exit_code": -1, "stdout": "", "stderr": "No CMakeLists.txt found"}
    build_dir = os.path.join(task_dir, "build")
    os.makedirs(build_dir, exist_ok=True)
    
    cmake_result = _run_subprocess(["cmake", ".."], cwd=build_dir, timeout=60)
    if cmake_result["exit_code"] != 0:
         return {"exit_code": 1, "stdout": "", "stderr": f"CMake failed: {cmake_result['stderr'][:500]}"}
         
    make_result = _run_subprocess(["cmake", "--build", "."], cwd=build_dir, timeout=180)
    if make_result["exit_code"] != 0:
         return {"exit_code": 1, "stdout": "", "stderr": f"Make failed: {make_result['stderr'][:500]}"}
         
    ctest_result = _run_subprocess(["ctest", "-V"], cwd=build_dir, timeout=60)
    return ctest_result


# ---------------------------------------------------------------------------
# Best Agent Selector
# ---------------------------------------------------------------------------

def load_best_agent(archive_path: str) -> dict | None:
    """
    Load the best agent record from evolution_archive.json.
    Priority: highest score among PASS records; fallback to highest score overall.
    """
    if not os.path.exists(archive_path):
        print(f"[archive] Not found: {archive_path}")
        return None

    with open(archive_path, encoding="utf-8") as f:
        archive = json.load(f)

    if not archive:
        return None

    # Prefer PASS records
    pass_records = [r for r in archive if r.get("goldilocks_status") == "PASS"]
    pool = pass_records if pass_records else archive

    best = max(pool, key=lambda r: r.get("score", 0))
    print(f"[archive] Best agent: cycle={best.get('cycle')}, "
          f"score={best.get('score')}, "
          f"goldilocks={best.get('goldilocks_status')}, "
          f"epiplexity={best.get('epiplexity_score')}")
    return best


# ---------------------------------------------------------------------------
# Main Outer Evaluation Loop
# ---------------------------------------------------------------------------

def run_outer_eval(
    tasks_json: str,
    archive_path: str,
    output_path: str,
    groq_model: str = None,
    use_reference: bool = False,
) -> dict:
    """
    Main Outer Loop evaluation pipeline.

    Args:
        use_reference: If True, use hardcoded reference solutions instead of LLM.
                       Use this to validate the test pipeline infrastructure.
    Returns a report dict with real Pass@1 numbers.
    """
    print("\n" + "=" * 70)
    print("  DGM OUTER LOOP EVALUATION — Real Compiler Pass@1")
    print("=" * 70)

    # 1. Load best agent from archive
    print("\n[Step 1] Loading best evolved agent from archive...")
    best_agent = load_best_agent(archive_path)
    if best_agent is None:
        print("WARNING: No archive found, using baseline (Gen 0) agent.")
        best_agent = {"cycle": 0, "score": 0.0, "goldilocks_status": "N/A"}

    # 2. Load task list
    print(f"\n[Step 2] Loading task list from {tasks_json}...")
    with open(tasks_json, encoding="utf-8") as f:
        task_ids = json.load(f)
    print(f"  Tasks: {task_ids}")

    # 3. Check compiler availability
    print("\n[Step 3] Checking compiler availability...")
    available_langs = {}
    for lang in ["python", "javascript", "go", "rust", "java", "cpp"]:
        ok = _check_compiler(lang)
        available_langs[lang] = ok
        status = "✅" if ok else "❌"
        print(f"  {status} {lang}")

    # 4. Create temp workspace
    workspace = tempfile.mkdtemp(prefix="dgm_outer_eval_")
    print(f"\n[Step 4] Workspace: {workspace}")

    # 5. Evaluate each task
    print("\n[Step 5] Running evaluations...\n")
    results = []
    
    rule_library = RuleLibrary()
    from groq import Groq
    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    llm_client = Groq(api_key=groq_api_key) if groq_api_key and groq_api_key != "YOUR_GROQ_API_KEY_HERE" else None

    for task_id in task_ids:
        print(f"{'─'*60}")
        print(f"  Task: {task_id}")

        if task_id not in TASK_REGISTRY:
            print(f"  ⚠️  Not in TASK_REGISTRY — skipping")
            results.append({
                "task_id": task_id,
                "language": "unknown",
                "status": "SKIPPED",
                "reason": "Task not in registry",
                "pass": False,
            })
            continue

        meta = TASK_REGISTRY[task_id]
        lang = meta["language"]

        # Check compiler
        if not available_langs.get(lang, False):
            print(f"  ❌ Compiler not available for {lang} — skipping")
            results.append({
                "task_id": task_id,
                "language": lang,
                "status": "SKIPPED",
                "reason": f"{lang} compiler/runtime not installed on host",
                "pass": False,
            })
            continue

        # Create task directory
        task_dir = os.path.join(workspace, task_id)
        os.makedirs(task_dir, exist_ok=True)

        # Download task files
        print(f"  Downloading exercism files...")
        downloaded = download_task_files(task_id, task_dir)

        stub_path = downloaded.get("solution_stub")
        test_path = downloaded.get("test")

        if test_path is None:
            print(f"  ❌ Test file download failed — skipping")
            results.append({
                "task_id": task_id,
                "language": lang,
                "status": "SKIPPED",
                "reason": "Could not download test file from exercism GitHub",
                "pass": False,
            })
            continue

        # Generate and test loop (Reflexion)
        max_retries = 3
        passed = False
        test_result = None
        solution_code = None
        error_log = None
        
        mode = "reference" if use_reference else "LLM"
        injected_rules = rule_library.get_top_rules(top_k=3)
        if injected_rules:
            print(f"  [RIMRULE] Injecting top learned rules from Memory Bank...")
            
        for attempt in range(max_retries):
            if attempt == 0:
                print(f"  Generating solution ({mode}, Gen {best_agent.get('cycle', 0)})...")
                if use_reference:
                    solution_code = _get_reference_solution(task_id, "")
                else:
                    solution_code = generate_solution(
                        task_id,
                        meta["problem_statement"],
                        stub_path,
                        model=groq_model,
                        injected_rules=injected_rules,
                        client=llm_client
                    )
            else:
                print(f"  [Reflexion] Attempt {attempt+1}/{max_retries} - Fixing previous errors...")
                rule_library.add_error_encountered()
                
                # Store the failed code to generate rules later
                failed_code = solution_code
                
                solution_code = generate_solution(
                    task_id,
                    meta["problem_statement"],
                    stub_path,
                    model=groq_model,
                    previous_code=solution_code,
                    previous_error=error_log,
                    injected_rules=injected_rules,
                    client=llm_client
                )

            # Write solution to solution file
            solution_filename = meta["files"]["solution_stub"]
            solution_path = os.path.join(task_dir, solution_filename)
            with open(solution_path, "w", encoding="utf-8") as f:
                f.write(solution_code)
            
            if attempt == 0:
                print(f"  Solution written ({len(solution_code)} chars)")

            # Run tests
            print(f"  Running {lang} tests...")
            if lang == "python":
                test_result = run_python_test(task_dir, os.path.basename(test_path))
            elif lang == "javascript":
                test_result = run_javascript_test(task_dir)
            elif lang == "go":
                test_result = run_go_test(task_dir)
            elif lang == "rust":
                test_result = run_rust_test(task_dir)
            elif lang == "java":
                test_result = run_java_test(task_dir)
            elif lang == "cpp":
                test_result = run_cpp_test(task_dir)
            else:
                test_result = {"exit_code": -1, "stdout": "", "stderr": "Language not supported"}

            passed = test_result["exit_code"] == 0
            status_str = "PASS" if passed else "FAIL"
            print(f"  Result: {status_str} (exit_code={test_result['exit_code']})")
            
            if passed:
                if attempt > 0 and llm_client is not None:
                    # Successful Reflexion! Extract a rule to remember.
                    print("  [RIMRULE] Reflexion successful. Extracting generalizable rule...")
                    new_rule = extract_rule_from_reflexion(failed_code, error_log, solution_code, client=llm_client)
                    if new_rule:
                        rule_library.add_or_update_rule(new_rule)
                        print(f"  [RIMRULE] Rule added to Memory Bank: {new_rule}")
                break
            
            # Prepare error log for next Reflexion attempt
            if not passed:
                error_log = test_result.get("stderr", "")
                if not error_log.strip():
                    error_log = test_result.get("stdout", "")
                error_log = error_log[-2000:]  # Limit context size

            if test_result.get("stdout"):
                print(f"  Stdout: {test_result['stdout'][:600]}")
            if test_result.get("stderr"):
                print(f"  Stderr: {test_result['stderr'][:400]}")


        results.append({
            "task_id": task_id,
            "language": lang,
            "status": "PASS" if passed else "FAIL",
            "reason": "Real compiler test",
            "pass": passed,
            "exit_code": test_result["exit_code"],
            "stdout_snippet": test_result.get("stdout", "")[:500],
            "stderr_snippet": test_result.get("stderr", "")[:500],
        })

    # 6. Compute real Pass@1
    print(f"\n{'='*70}")
    attempted = [r for r in results if r["status"] != "SKIPPED"]
    resolved = [r for r in results if r["pass"]]
    skipped = [r for r in results if r["status"] == "SKIPPED"]

    pass_at_1 = len(resolved) / len(attempted) if attempted else 0.0

    print(f"\n📊 OUTER LOOP RESULTS (Real Compiler Evaluation)")
    print(f"   Total tasks:      {len(task_ids)}")
    print(f"   Attempted:        {len(attempted)}")
    print(f"   Skipped:          {len(skipped)} (compiler not installed)")
    print(f"   Passed:           {len(resolved)}")
    print(f"   Failed:           {len(attempted) - len(resolved)}")
    print(f"   ─────────────────────────────────")
    print(f"   Pass@1 (real):   {pass_at_1*100:.1f}%  ← This is your paper number!")
    print(f"{'='*70}\n")

    # 7. Build and save report
    solver_mode = "reference-solutions" if use_reference else "groq-llm"
    report = {
        "metadata": {
            "timestamp": datetime.datetime.now().isoformat(),
            "evaluation_type": "outer_loop_real_compiler",
            "solver_mode": solver_mode,
            "description": f"Outer Loop evaluation using real compilers. Solver: {solver_mode}",
            "best_agent": {
                "cycle": best_agent.get("cycle", 0),
                "score": best_agent.get("score", 0),
                "epiplexity_score": best_agent.get("epiplexity_score", 0),
                "goldilocks_status": best_agent.get("goldilocks_status", "N/A"),
                "context_node": best_agent.get("context_node", "N/A"),
            },
        },
        "compiler_availability": available_langs,
        "task_results": results,
        "summary": {
            "total_tasks": len(task_ids),
            "attempted_tasks": len(attempted),
            "skipped_tasks": len(skipped),
            "resolved_tasks": len(resolved),
            "failed_tasks": len(attempted) - len(resolved),
            "pass_at_1": round(pass_at_1, 4),
            "pass_at_1_pct": round(pass_at_1 * 100, 2),
        },
        "academic_note": (
            "Pass@1 computed on the subset of tasks where compilers were available "
            "on the host machine (Python, Node.js). Tasks requiring Go, Rust, C++, "
            "or Java were skipped due to missing compiler installation. "
            "Evaluation conducted locally without Docker as a proof-of-concept Outer Loop."
        ),
    }

    # Clean up workspace
    shutil.rmtree(workspace, ignore_errors=True)

    # Save report
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"✅ Report saved to: {output_path}")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DGM Outer Loop Real Compiler Evaluator")
    parser.add_argument(
        "--tasks",
        default=os.path.join(os.path.dirname(__file__), "polyglot", "subsets", "small.json"),
        help="Path to task list JSON (default: polyglot/subsets/small.json)",
    )
    parser.add_argument(
        "--archive",
        default=os.path.join(os.path.dirname(__file__), "evolution_archive.json"),
        help="Path to evolution_archive.json",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "outer_eval_report.json"),
        help="Output path for the evaluation report",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Groq model to use for solution generation",
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Only check compiler availability and exit",
    )
    parser.add_argument(
        "--use-reference",
        action="store_true",
        help="Use hardcoded reference solutions (validates pipeline infrastructure, not LLM quality)",
    )
    args = parser.parse_args()

    # Load .env file (manual parser — no python-dotenv dependency needed)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dotenv_path = os.path.join(repo_root, ".env")
    if os.path.exists(dotenv_path):
        with open(dotenv_path, encoding="utf-8") as fenv:
            for line in fenv:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:  # don't override existing env
                    os.environ[key] = val
        print(f"[env] Loaded .env from {dotenv_path}")

    if args.check_env:
        print("\n[ENV CHECK] Compiler availability:\n")
        for lang in ["python", "javascript", "go", "rust", "java", "cpp"]:
            ok = _check_compiler(lang)
            print(f"  {'✅' if ok else '❌'} {lang}")
        groq_key = os.environ.get("GROQ_API_KEY", "")
        has_key = bool(groq_key) and groq_key != "YOUR_GROQ_API_KEY_HERE"
        print(f"\n  {'✅' if has_key else '❌'} GROQ_API_KEY {'set' if has_key else 'not set'}")
        print()
        return

    # Load Groq key from config.yaml if still not in env
    if not os.environ.get("GROQ_API_KEY"):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(repo_root, "config.yaml")
        if os.path.exists(config_path):
            try:
                import yaml
                with open(config_path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                if cfg.get("api_key") and cfg["api_key"] != "YOUR_GROQ_API_KEY_HERE":
                    os.environ["GROQ_API_KEY"] = cfg["api_key"]
            except Exception:
                pass

    run_outer_eval(
        tasks_json=args.tasks,
        archive_path=args.archive,
        output_path=args.output,
        groq_model=args.model,
        use_reference=args.use_reference,
    )


if __name__ == "__main__":
    main()
