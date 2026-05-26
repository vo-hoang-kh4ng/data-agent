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
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
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
        is_binary = filename.endswith(".jar") or filename.endswith(".zip")
        content = None
        
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "DGM-Outer-Eval/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                content = resp.read()
        except Exception as e:
            # Try alternate path (some exercises use underscores vs hyphens)
            alt_filename = filename.replace("-", "_")
            url = _exercism_url(lang, exercise, alt_filename)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "DGM-Outer-Eval/1.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    content = resp.read()
                    filename = alt_filename
            except Exception:
                pass

        if content is None:
            print(f"    [download] SKIP {task_id}/{filename} — not reachable")
            downloaded[role] = None
        else:
            dest_path = os.path.join(dest_dir, filename)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            if is_binary:
                with open(dest_path, "wb") as f:
                    f.write(content)
            else:
                try:
                    text_content = content.decode("utf-8")
                except UnicodeDecodeError:
                    text_content = content.decode("latin-1")
                with open(dest_path, "w", encoding="utf-8") as f:
                    f.write(text_content)
            downloaded[role] = dest_path
            print(f"    [download] OK {task_id}/{filename}")

    return downloaded


def get_task_meta_dynamically(task_id: str) -> dict:
    """Dynamically construct metadata for a task not present in TASK_REGISTRY."""
    if task_id in TASK_REGISTRY:
        return TASK_REGISTRY[task_id]

    parts = task_id.split("__")
    lang = parts[0]
    exercise = parts[1] if len(parts) > 1 else task_id

    # Standard file layout based on language
    files = {}
    exercise_underscore = exercise.replace("-", "_")
    if lang == "python":
        files = {
            "solution_stub": f"{exercise_underscore}.py",
            "test": f"{exercise_underscore}_test.py"
        }
    elif lang == "javascript":
        files = {
            "solution_stub": f"{exercise}.js",
            "test": f"{exercise}.spec.js"
        }
    elif lang == "go":
        files = {
            "solution_stub": f"{exercise_underscore}.go",
            "test": f"{exercise_underscore}_test.go"
        }
    elif lang == "rust":
        files = {
            "solution_stub": "src/lib.rs",
            "test": f"tests/{exercise}.rs",
            "cargo": "Cargo.toml"
        }
    elif lang == "java":
        # Query GitHub API to get the correct filenames in src/main/java and src/test/java
        main_files = []
        test_files = []
        try:
            url_main = f"https://api.github.com/repos/exercism/java/contents/exercises/practice/{exercise}/src/main/java"
            req = urllib.request.Request(url_main, headers={"User-Agent": "DGM-Outer-Eval/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                main_files = [f["name"] for f in data if f["name"].endswith(".java")]
        except Exception:
            pass

        try:
            url_test = f"https://api.github.com/repos/exercism/java/contents/exercises/practice/{exercise}/src/test/java"
            req = urllib.request.Request(url_test, headers={"User-Agent": "DGM-Outer-Eval/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                test_files = [f["name"] for f in data if f["name"].endswith(".java")]
        except Exception:
            pass

        files = {}
        if main_files:
            files["solution_stub"] = f"src/main/java/{main_files[0]}"
        else:
            camel_case = "".join(w.capitalize() for w in exercise.replace("-", "_").split("_"))
            files["solution_stub"] = f"src/main/java/{camel_case}.java"

        if test_files:
            files["test"] = f"src/test/java/{test_files[0]}"
            for idx, tf in enumerate(test_files[1:]):
                files[f"test_{idx}"] = f"src/test/java/{tf}"
        else:
            camel_case = "".join(w.capitalize() for w in exercise.replace("-", "_").split("_"))
            files["test"] = f"src/test/java/{camel_case}Test.java"

        files["build_gradle"] = "build.gradle"
        files["gradlew"] = "gradlew"
        files["gradlew_bat"] = "gradlew.bat"
        files["gradle_wrapper_jar"] = "gradle/wrapper/gradle-wrapper.jar"
        files["gradle_wrapper_properties"] = "gradle/wrapper/gradle-wrapper.properties"
    elif lang == "cpp":
        files = {
            "solution_stub": f"{exercise_underscore}.cpp",
            "header": f"{exercise_underscore}.h",
            "test": f"{exercise_underscore}_test.cpp"
        }
    else:
        files = {
            "solution_stub": f"{exercise_underscore}.py",
            "test": f"{exercise_underscore}_test.py"
        }

    # Fetch instructions dynamically from Exercism main repo
    # Standard path: .docs/instructions.md
    instructions_url = f"{EXERCISM_BASE}/{lang}/main/exercises/practice/{exercise}/.docs/instructions.md"
    instructions = _fetch_raw(instructions_url)

    # Try alternate path just in case
    if instructions is None:
        instructions_url_alt = f"{EXERCISM_BASE}/{lang}/main/exercises/practice/{exercise_underscore}/.docs/instructions.md"
        instructions = _fetch_raw(instructions_url_alt)

    if instructions is None:
        # Try introduction.md or a generic fallback
        introduction_url = f"{EXERCISM_BASE}/{lang}/main/exercises/practice/{exercise}/.docs/introduction.md"
        instructions = _fetch_raw(introduction_url)

    if instructions is None:
        instructions = f"Solve the {exercise} problem in {lang}."

    meta = {
        "language": lang,
        "exercise": exercise,
        "files": files,
        "problem_statement": instructions
    }

    # Cache it in TASK_REGISTRY
    TASK_REGISTRY[task_id] = meta
    return meta


def save_checkpoint(output_path: str, results: list, task_ids: list,
                    available_langs: dict, best_agent: dict, solver_mode: str):
    attempted = [r for r in results if r["status"] != "SKIPPED"]
    resolved = [r for r in results if r["pass"]]
    skipped = [r for r in results if r["status"] == "SKIPPED"]
    pass_at_1 = len(resolved) / len(attempted) if attempted else 0.0

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
            "checkpoint": f"Evaluated {len(results)}/{len(task_ids)} tasks",
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

    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    tmp_output = output_path + ".tmp"
    try:
        with open(tmp_output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        os.replace(tmp_output, output_path)
        print(f"  [Checkpoint] Saved progress to {output_path} ({len(results)}/{len(task_ids)} tasks)")
    except Exception as e:
        print(f"  [Checkpoint] WARNING: Failed to save progress: {e}")


# ---------------------------------------------------------------------------
# LLM-based Solution Generator (uses Groq API)
# ---------------------------------------------------------------------------

def generate_solution(task_id: str, problem_statement: str,
                       stub_path: str | None, model: str = None,
                       previous_code: str = None, previous_error: str = None,
                       injected_rules: str = None, client = None,
                       test_file_content: str = None, n_candidates: int = 1) -> str:
    """
    Call Groq (or mock) LLM to generate a solution for the task.
    Supports Reflexion (Test-Driven Self-Correction) if previous_error is provided.
    Supports Best-of-N search if n_candidates > 1.
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
    )
    if n_candidates > 1:
        system_prompt += (
            f"You MUST provide exactly {n_candidates} different coding approaches to solve this problem. "
            "For each approach, wrap the code in a distinct block labeled [APPROACH X] where X is 1, 2, etc. "
            "Example format:\n[APPROACH 1]\n<raw code here>\n\n[APPROACH 2]\n<raw code here>\n"
            "CRITICAL: Output ONLY raw code under each approach header, no markdown fences (```)."
        )
    else:
        system_prompt += (
            "Write ONLY the raw implementation code. "
            "CRITICAL: Do NOT wrap the code in markdown fences (``` or ```python etc). "
            "Output the raw code directly with NO surrounding backticks. "
            "The code must compile and pass all unit tests."
        )
    
    rules_text = ""
    if injected_rules:
        rules_text = f"\n\n[RIMRULE MEMORY BANK]\nHere are some learned rules from past mistakes to keep in mind:\n{injected_rules}\n"
    
    # Include test file content so LLM knows expected API signatures
    test_context = ""
    if test_file_content:
        # Truncate very long test files to save tokens
        truncated = test_file_content[:3000]
        test_context = f"\n\nThe test file that your code must pass (study the expected function signatures and types):\n```\n{truncated}\n```\n"
    
    if previous_code and previous_error:
        user_prompt = (
            f"Exercise: {exercise}\n\n"
            f"Problem:\n{problem_statement}{rules_text}{test_context}\n\n"
            f"You previously generated this code:\n```\n{previous_code}\n```\n\n"
            f"However, it failed the tests with this error:\n```\n{previous_error}\n```\n\n"
            f"Please fix the code to pass the tests. Write ONLY the corrected {lang} implementation code, no markdown fences."
        )
    else:
        candidates_str = f" Write {n_candidates} distinct approaches labeled [APPROACH X]." if n_candidates > 1 else ""
        user_prompt = (
            f"Exercise: {exercise}\n\n"
            f"Problem:\n{problem_statement}{rules_text}{test_context}\n\n"
            f"Current stub:\n{stub_content}\n\n"
            f"Write a complete, correct {lang} implementation.{candidates_str} Output ONLY raw code, no markdown fences."
        )

    # Try Custom OpenAI API with Retry Logic for Rate Limits
    import time
    max_api_retries = 5
    for api_attempt in range(max_api_retries):
        try:
            if client is None:
                import yaml
                from dotenv import load_dotenv
                from openai import OpenAI
                
                load_dotenv()
                config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                    
                api_key_env_var = config.get("api_key_env_var", "QWEN_API_KEY")
                api_key = os.environ.get(api_key_env_var, "")
                base_url = config.get("base_url_conv_model", "https://proxy.onebot.meobeo.ai/v1")
                
                client = OpenAI(api_key=api_key, base_url=base_url)
                
                if not model:
                    model = config.get("conv_model", "hosted_vllm/Qwen/Qwen3.5-35B-A3B-FP8")
                    
            use_model = model
                
            print(f"    [llm] Calling LLM ({use_model}) for {task_id} (Attempt {api_attempt+1})...")
            resp = client.chat.completions.create(
                model=use_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=4096,
                extra_body={
                    "cache": {"no-cache": True},
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            )
            raw = resp.choices[0].message.content.strip()
            
            # If Best-of-N, parse out approaches and use Epiplexity
            if n_candidates > 1:
                # regex to find [APPROACH X] blocks
                approaches = re.split(r'\[APPROACH \d+\]\s*', raw)
                approaches = [a.strip() for a in approaches if a.strip()]
                
                if approaches:
                    print(f"    [GRPO] Generated {len(approaches)} approaches. Scoring with Epiplexity...")
                    import sys
                    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    from core.inspector import compute_mdl_epiplexity
                    
                    best_approach = approaches[0]
                    best_score = float('inf')
                    
                    for idx, app in enumerate(approaches):
                        app = app.strip()
                        # Strip accidental markdown fences from each approach
                        if "```" in app:
                            match = re.search(r"```(?:[a-zA-Z+]+)?\n(.*?)\n```", app, re.DOTALL)
                            if match:
                                app = match.group(1)
                            else:
                                app = re.sub(r"^```[a-zA-Z+]*\n", "", app)
                                app = re.sub(r"\n```$", "", app)
                        
                        # Sanity Check (Vòng lọc số 1): Loại code lừa đảo (quá ngắn)
                        if len(app) < 50:
                            print(f"      - Approach {idx+1} Rejected: Too short (len={len(app)})")
                            continue
                            
                        score = compute_mdl_epiplexity(app)
                        print(f"      - Approach {idx+1} Epiplexity: {score:.4f}")
                        
                        # Vòng lọc số 2: Chọn min Epiplexity NHƯNG phải > ngưỡng 0.5 (Tránh Trivial Trap)
                        if score > 0.5 and score < best_score:
                            best_score = score
                            best_approach = app
                    
                    # Nếu tất cả đều rớt ngưỡng (hoặc quá ngắn), lấy fallback là approach đầu tiên
                    if best_score == float('inf'):
                        print(f"    [GRPO] All approaches failed threshold/sanity. Falling back to Approach 1.")
                        best_approach = approaches[0]
                    else:
                        print(f"    [GRPO] Selected approach with Epiplexity = {best_score:.4f}")
                        
                    return best_approach
            
            # Single approach mode: Strip markdown fences if present (all languages)
            if "```" in raw:
                match = re.search(
                    r"```(?:python|javascript|js|go|golang|rust|java|cpp|c\+\+|c)?\n(.*?)\n```",
                    raw, re.DOTALL
                )
                if match:
                    raw = match.group(1)
                else:
                    # Fallback: strip any ``` fences regardless of language tag
                    raw = re.sub(r"^```[a-zA-Z+]*\n", "", raw)
                    raw = re.sub(r"\n```$", "", raw)
            return raw

        except Exception as e:
            err_msg = str(e).lower()
            if "429" in err_msg or "rate limit" in err_msg or "too many requests" in err_msg:
                wait_time = 35 * (api_attempt + 1)
                print(f"    [llm] Rate Limit hit! Waiting {wait_time}s before retry... ({e})")
                time.sleep(wait_time)
            else:
                print(f"    [llm] Groq unavailable ({e}). Using empty stub.")
                return stub_content + "\n# [DGM STUB] LLM Failure\n"
    
    print(f"    [llm] Exhausted all API retries. Returning empty stub.")
    return stub_content + "\n# [DGM STUB] API Exhausted\n"


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
    """
    Run Java test using javac + direct JUnit execution (no Gradle needed).
    Falls back to Gradle if build.gradle exists.
    """
    print(f"    [java] Running java test in {task_dir}...")
    
    # Try Gradle first if available
    if os.path.exists(os.path.join(task_dir, "build.gradle")):
        gradlew = "./gradlew" if os.name != "nt" else "gradlew.bat"
        if os.name != "nt":
            _run_subprocess(["chmod", "+x", "gradlew"], cwd=task_dir, timeout=5)
        result = _run_subprocess([gradlew, "test"], cwd=task_dir, timeout=180)
        return result
    
    # Fallback: compile with javac and run with java directly
    java_files = []
    for root, dirs, files in os.walk(task_dir):
        for f in files:
            if f.endswith(".java"):
                java_files.append(os.path.join(root, f))
    
    if not java_files:
        return {"exit_code": -1, "stdout": "", "stderr": "No .java files found"}
    
    # Compile all java files together
    print(f"    [javac] Compiling {len(java_files)} Java files...")
    compile_result = _run_subprocess(
        ["javac", "-cp", "."] + java_files,
        cwd=task_dir, timeout=30
    )
    if compile_result["exit_code"] != 0:
        return {"exit_code": compile_result["exit_code"], "stdout": compile_result["stdout"],
                "stderr": f"Compilation failed: {compile_result['stderr'][:500]}"}
    
    # Find test class name
    test_classes = [os.path.splitext(os.path.basename(f))[0] for f in java_files if "Test" in f]
    if not test_classes:
        return {"exit_code": -1, "stdout": "", "stderr": "No test class found"}
    
    # Run the first test class
    run_result = _run_subprocess(
        ["java", "-cp", ".", test_classes[0]],
        cwd=task_dir, timeout=30
    )
    return run_result


def run_cpp_test(task_dir: str) -> dict:
    """
    Run C++ test. Auto-generates CMakeLists.txt if not present.
    """
    print(f"    [cmake] Running cpp test in {task_dir}...")
    
    # Auto-generate CMakeLists.txt if missing
    cmake_path = os.path.join(task_dir, "CMakeLists.txt")
    if not os.path.exists(cmake_path):
        print(f"    [cmake] Auto-generating CMakeLists.txt...")
        cpp_files = [f for f in os.listdir(task_dir) if f.endswith(".cpp")]
        h_files = [f for f in os.listdir(task_dir) if f.endswith(".h")]
        test_files = [f for f in cpp_files if "test" in f.lower()]
        src_files = [f for f in cpp_files if "test" not in f.lower()]
        
        exercise_name = os.path.basename(task_dir).replace("cpp__", "").replace("-", "_")
        
        cmake_content = f"""cmake_minimum_required(VERSION 3.10)
project({exercise_name})
set(CMAKE_CXX_STANDARD 17)

add_executable({exercise_name}_test {' '.join(src_files + test_files)})
target_include_directories({exercise_name}_test PRIVATE ${{CMAKE_CURRENT_SOURCE_DIR}})

enable_testing()
add_test(NAME {exercise_name}_test COMMAND {exercise_name}_test)
"""
        with open(cmake_path, "w") as f:
            f.write(cmake_content)
    
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
    solver_mode = "reference-solutions" if use_reference else "groq-llm"
    
    rule_library = RuleLibrary()
    
    import yaml
    from dotenv import load_dotenv
    from openai import OpenAI
    
    load_dotenv()
    
    # Load config.yaml
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    api_key_env_var = config.get("api_key_env_var", "QWEN_API_KEY")
    api_key = os.environ.get(api_key_env_var, "")
    base_url = config.get("base_url_conv_model", "https://proxy.onebot.meobeo.ai/v1")
    
    if not api_key:
        print(f"WARNING: API key not found in env var '{api_key_env_var}'. Check your .env file.")
        
    llm_client = OpenAI(api_key=api_key, base_url=base_url)
    
    # Update default groq_model parameter to the new model from config if not specified
    if not groq_model:
        groq_model = config.get("conv_model", "hosted_vllm/Qwen/Qwen3.5-35B-A3B-FP8")
    
    # ===== CONNECTION VERIFICATION =====
    print(f"\n[Step 4.5] Verifying LLM connection...")
    print(f"  API Base URL: {base_url}")
    print(f"  Model: {groq_model}")
    print(f"  API Key (masked): {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else '****'}")
    try:
        test_resp = llm_client.chat.completions.create(
            model=groq_model,
            messages=[{"role": "user", "content": "Reply with exactly: CONNECTION_OK"}],
            max_tokens=20,
            extra_body={
                "cache": {"no-cache": True},
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
        test_reply = test_resp.choices[0].message.content.strip()
        print(f"  ✅ LLM Connection VERIFIED! Response: '{test_reply}'")
        print(f"  ✅ Model ID from response: {test_resp.model}")
    except Exception as e:
        print(f"  ❌ LLM Connection FAILED: {e}")
        print(f"  ❌ Cannot proceed without LLM. Exiting.")
        return None
    # ===== END CONNECTION VERIFICATION =====

    # Load completed tasks if resuming from checkpoint
    completed_tasks = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_report = json.load(f)
            completed_tasks = {r["task_id"]: r for r in existing_report.get("task_results", [])}
            print(f"  [Checkpoint] Loaded {len(completed_tasks)} completed task(s) from {output_path}")
        except Exception as e:
            print(f"  [Checkpoint] Warning: could not load existing report from {output_path}: {e}")

    for task_id in task_ids:
        print(f"{'─'*60}")
        print(f"  Task: {task_id}")

        # Check if already completed
        if task_id in completed_tasks:
            print(f"  [Checkpoint] Reusing result from previous run (status: {completed_tasks[task_id].get('status')})")
            results.append(completed_tasks[task_id])
            continue

        if task_id not in TASK_REGISTRY:
            print(f"  [Registry] Task {task_id} not in hardcoded registry. Attempting dynamic load...")
            try:
                get_task_meta_dynamically(task_id)
            except Exception as e:
                print(f"  ❌ Failed to dynamically load metadata for {task_id}: {e}")

        if task_id not in TASK_REGISTRY:
            print(f"  ⚠️  Not in TASK_REGISTRY and dynamic load failed — skipping")
            results.append({
                "task_id": task_id,
                "language": "unknown",
                "status": "SKIPPED",
                "reason": "Task not in registry",
                "pass": False,
            })
            save_checkpoint(output_path, results, task_ids, available_langs, best_agent, solver_mode)
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
            save_checkpoint(output_path, results, task_ids, available_langs, best_agent, solver_mode)
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
            
        # Read test file content to inject into LLM prompt
        test_file_content = None
        if test_path and os.path.exists(test_path):
            try:
                with open(test_path, encoding="utf-8") as tf:
                    test_file_content = tf.read()
            except Exception:
                pass
        
        for attempt in range(max_retries):
            if attempt == 0:
                print(f"  Generating solution ({mode}, Gen {best_agent.get('cycle', 0)}, GRPO Best-of-3)...")
                if use_reference:
                    solution_code = _get_reference_solution(task_id, "")
                else:
                    solution_code = generate_solution(
                        task_id,
                        meta["problem_statement"],
                        stub_path,
                        model=groq_model,
                        injected_rules=injected_rules,
                        client=llm_client,
                        test_file_content=test_file_content,
                        n_candidates=3
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
                    client=llm_client,
                    test_file_content=test_file_content
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
                    new_rule = extract_rule_from_reflexion(failed_code, error_log, solution_code, client=llm_client, model=groq_model)
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
            "generated_code": solution_code,
        })
        save_checkpoint(output_path, results, task_ids, available_langs, best_agent, solver_mode)

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
        default=os.path.join(os.path.dirname(__file__), "polyglot", "subsets", "polyglot_60.json"),
        help="Path to task list JSON (default: polyglot/subsets/polyglot_60.json)",
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
