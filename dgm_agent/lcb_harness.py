"""
LiveCodeBench (LCB) Harness — Darwin Gödel Machine (DGM)
=========================================================
Đánh giá Outer Loop siêu nhẹ cho các bài toán thuật toán thuần Python.
Không cần Docker, chạy subprocess + pytest trực tiếp trên host.

Ưu điểm so với SWE-bench:
  - Timeout 5 giây thay vì 1 tiếng
  - Không cần build Docker image
  - Chạy 100 bài trong < 5 phút
  - Bắt log lỗi pytest cho Reflexion

Usage:
    python dgm_agent/lcb_harness.py
    python dgm_agent/lcb_harness.py --tasks dgm_agent/lcb_tasks/tasks.json
    python dgm_agent/lcb_harness.py --model hosted_vllm/Qwen/Qwen3.5-35B-A3B-FP8
"""

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Load .env for API keys
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    # Manual fallback
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(dotenv_path):
        with open(dotenv_path, encoding="utf-8") as fenv:
            for line in fenv:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class LCBHarness:
    """Lightweight sandbox for LiveCodeBench evaluation."""

    def __init__(self, workspace_dir="./dgm_agent/lcb_sandbox"):
        """
        Khởi tạo môi trường Sandbox siêu nhẹ cho LiveCodeBench.
        Không cần build Docker image mới, tái sử dụng Python env hiện tại.
        """
        self.workspace = workspace_dir
        os.makedirs(self.workspace, exist_ok=True)

    def extract_python_code(self, llm_response):
        """Trích xuất mã nguồn Python từ câu trả lời của Qwen/Llama."""
        match = re.search(r'```python\n(.*?)\n```', llm_response, re.DOTALL)
        return match.group(1).strip() if match else llm_response.strip()

    def generate_lcb_prompt(self, problem_description):
        """Đóng gói đề bài LCB thành Prompt tiêu chuẩn cho Solver Agent."""
        return f"""You are an expert Data & Algorithm Agent.
Solve the following programming problem.
You must return ONLY valid, executable Python code inside ```python ``` blocks.
Do not add any explanations or print statements.

<problem_description>
{problem_description}
</problem_description>
"""

    def extract_example_tests(self, problem_description):
        """
        Trích xuất các ví dụ Input/Output từ đề bài và tạo assert test để
        dùng cho vòng Self-Debug trước khi chạy hidden test case.

        Chiến lược: tìm các block Input/Output trong đề bài dạng:
            Input: X
            Output: Y
        Sau đó build pytest assertions kiểm tra stdout của solution.
        """
        # Tìm tất cả cặp Input/Output trong đề bài
        pattern = r'[Ii]nput[^\n]*\n([^\n]+)\n[Oo]utput[^\n]*\n([^\n]+)'
        matches = re.findall(pattern, problem_description)
        if not matches:
            return None

        test_lines = [
            "import subprocess, sys",
            "",
            "def run_solution(stdin_input):",
            "    import subprocess, sys",
            "    result = subprocess.run(",
            "        [sys.executable, 'solution_example.py'],",
            "        input=stdin_input, capture_output=True, text=True, timeout=5",
            "    )",
            "    return result.stdout.strip()",
            "",
        ]
        for idx, (inp, out) in enumerate(matches[:3]):  # Chỉ lấy 3 ví dụ đầu
            inp = inp.strip().strip('`')
            out = out.strip().strip('`')
            test_lines.append(f"def test_example_{idx}():")
            test_lines.append(f"    assert run_solution({repr(inp)}) == {repr(out)}")
            test_lines.append("")

        return "\n".join(test_lines)

    def run_code_on_example(self, solution_code, problem_description, timeout=5):
        """
        Chạy thử code trên example inputs từ đề bài, trả về (passed, error_trace).
        Dùng cho vòng Self-Debug trước hidden tests.
        """
        # Thử extract example tests
        # Cách đơn giản nhất: compile + exec code, kiểm tra không có lỗi syntax
        example_dir = os.path.join(self.workspace, "example_debug")
        os.makedirs(example_dir, exist_ok=True)

        sol_file = os.path.join(example_dir, "solution_example.py")
        with open(sol_file, "w", encoding="utf-8") as f:
            f.write(solution_code)

        # Bước 1: Kiểm tra syntax bằng py_compile
        try:
            result = subprocess.run(
                [sys.executable, "-c", f"import py_compile; py_compile.compile(r'{sol_file}', doraise=True)"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                error = result.stderr.strip()
                return False, f"SyntaxError: {error}"
        except Exception as e:
            return False, f"Compile check failed: {e}"

        # Bước 2: Thử extract và chạy example test nếu có
        example_test_code = self.extract_example_tests(problem_description)
        if example_test_code is None:
            # Không có ví dụ → chỉ cần pass syntax check
            return True, "Syntax OK (no examples found)"

        test_file = os.path.join(example_dir, "test_example.py")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(example_test_code)

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "test_example.py", "--disable-warnings", "-q"],
                capture_output=True, text=True, timeout=timeout,
                cwd=example_dir,
            )
            passed = result.returncode == 0
            error_trace = (result.stdout + "\n" + result.stderr).strip()
            return passed, error_trace
        except subprocess.TimeoutExpired:
            return False, "Example test timeout"
        except Exception as e:
            return False, str(e)

    def evaluate_task(self, task_id, problem_description, hidden_test_cases, llm_response, timeout=5):
        """
        Vòng lặp Outer Loop thực chiến: Trích xuất code, ghi file và chạy Pytest.

        Args:
            task_id: Mã bài toán (vd: lcb_001_two_sum)
            problem_description: Đề bài gốc
            hidden_test_cases: Chuỗi test case ẩn dạng pytest (vd: assert f(2) == 4)
            llm_response: Câu trả lời thô từ LLM
            timeout: Giới hạn thời gian chạy pytest (giây)

        Returns:
            (is_pass, logs): Tuple (bool, str)
        """
        print(f"\n🚀 Đang đánh giá Task: [ {task_id} ] trên LiveCodeBench...")

        # 1. Trích xuất mã nguồn từ Agent
        solution_code = self.extract_python_code(llm_response)

        # 2. Ghi mã nguồn của Agent vào file thực thi
        solution_file = os.path.join(self.workspace, f"solution_{task_id}.py")
        with open(solution_file, "w", encoding="utf-8") as f:
            f.write(solution_code)

        # 3. Ghi các Test Cases ẩn (Private Tests) vào file test riêng
        test_file = os.path.join(self.workspace, f"test_{task_id}.py")
        test_content = hidden_test_cases  # Các test case do LCB cung cấp

        with open(test_file, "w", encoding="utf-8") as f:
            f.write(test_content)

        # 4. Thực thi Pytest siêu tốc (Bắt lỗi và Timeout)
        try:
            test_basename = os.path.basename(test_file)
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_basename, "--disable-warnings", "-q"],
                capture_output=True, text=True, timeout=timeout,
                cwd=self.workspace,
            )

            if result.returncode == 0:
                print(f"✅ STATUS: [ RESOLVED ] - {task_id} vượt qua 100% test case ẩn!")
                return True, result.stdout
            else:
                print(f"❌ STATUS: [ FAILED ] - {task_id} sai logic hoặc không biên dịch được.")
                return False, result.stdout + "\n" + result.stderr

        except subprocess.TimeoutExpired:
            print(f"⏱️ STATUS: [ TIMEOUT ] - {task_id} quá thời gian thực thi (>{timeout}s).")
            return False, f"Execution Timeout (>{timeout}s)"


def call_llm_for_task(task, model_name_or_path, temperature=0.7):
    """
    Gọi LLM (Qwen / Claude / OpenAI) để giải một bài toán LCB.
    Tái sử dụng cơ sở hạ tầng LLM đã có trong dgm_agent/llm.py.
    """
    from llm import create_client, get_response_from_llm

    client, model = create_client(model_name_or_path)

    harness = LCBHarness()
    prompt = harness.generate_lcb_prompt(task["problem_description"])

    system_msg = (
        "You are an expert competitive programmer. "
        "Write clean, correct Python code that solves the given problem. "
        "Return ONLY the solution inside ```python ``` code blocks."
    )

    try:
        response, msg_history = get_response_from_llm(
            msg=prompt,
            client=client,
            model=model,
            system_message=system_msg,
            msg_history=[],
            temperature=temperature
        )
    except TypeError:
        response, msg_history = get_response_from_llm(
            msg=prompt,
            client=client,
            model=model,
            system_message=system_msg,
            msg_history=[],
        )

    return response


def call_llm_debug_fix(task, model_name_or_path, current_code, error_trace, round_num):
    """
    Gọi LLM với error trace từ lần chạy trước để yêu cầu sửa code.
    Đây là trái tim của cơ chế Self-Debug / Reflexion.
    """
    from llm import create_client, get_response_from_llm

    client, model = create_client(model_name_or_path)

    system_msg = (
        "You are an expert competitive programmer doing code review and debugging. "
        "You will be given a buggy Python solution and its error trace. "
        "Fix the code so it passes all test cases. "
        "Return ONLY the corrected solution inside ```python ``` code blocks."
    )

    debug_prompt = f"""The following Python solution has a bug:

```python
{current_code}
```

When run against the test cases, it produced this error:
```
{error_trace[:3000]}
```

Problem description (for context):
<problem_description>
{task['problem_description']}
</problem_description>

Please analyze the error carefully and provide a CORRECTED Python solution.
Return ONLY the corrected code inside ```python ``` code blocks."""

    print(f"  🔧 [Self-Debug Round {round_num}] Gửi error trace tới LLM để sửa...")
    try:
        response, _ = get_response_from_llm(
            msg=debug_prompt,
            client=client,
            model=model,
            system_message=system_msg,
            msg_history=[],
            temperature=0.3,  # Thấp hơn để sửa bug chính xác hơn
        )
    except TypeError:
        response, _ = get_response_from_llm(
            msg=debug_prompt,
            client=client,
            model=model,
            system_message=system_msg,
            msg_history=[],
        )

    return response


def call_llm_with_self_debug(task, model_name_or_path, temperature=0.7, max_debug_rounds=3):
    """
    Gọi LLM + Self-Debug Loop: Sinh code → Chạy example tests → Nếu sai → Gọi LLM sửa.
    Lặp lại tối đa max_debug_rounds lần.

    Returns:
        (final_response, debug_rounds_used, final_passed_example)
    """
    harness = LCBHarness()

    # Round 0: Sinh code lần đầu
    llm_response = call_llm_for_task(task, model_name_or_path, temperature)

    if not llm_response or not llm_response.strip():
        return llm_response, 0, False

    current_code = harness.extract_python_code(llm_response)
    current_response = llm_response

    for round_num in range(1, max_debug_rounds + 1):
        # Chạy thử trên example tests
        passed_example, error_trace = harness.run_code_on_example(
            current_code, task["problem_description"]
        )

        if passed_example:
            print(f"  ✅ [Self-Debug] Code vượt qua example tests (round {round_num - 1})")
            return current_response, round_num - 1, True

        # Chưa pass → Gọi LLM sửa nếu còn round
        if round_num <= max_debug_rounds:
            print(f"  ❌ [Self-Debug Round {round_num}/{max_debug_rounds}] Lỗi: {error_trace[:200]}...")
            fixed_response = call_llm_debug_fix(
                task, model_name_or_path, current_code, error_trace, round_num
            )
            if fixed_response and fixed_response.strip():
                current_response = fixed_response
                current_code = harness.extract_python_code(fixed_response)

    print(f"  ⚠️ [Self-Debug] Hết {max_debug_rounds} rounds, trả về code tốt nhất.")
    return current_response, max_debug_rounds, False

import math

def calculate_pass_at_k(n, c, k):
    """Calculate pass@k metric.
    n: total samples generated per task
    c: number of correct samples
    k: k in pass@k
    """
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def run_lcb_evaluation(
    tasks,
    model_name_or_path="hosted_vllm/Qwen/Qwen3.5-35B-A3B-FP8",
    max_workers=5,
    timeout=5,
    pred_dname="dgm_agent/lcb_predictions",
    num_samples=1,
    max_debug_rounds=3,
):
    """
    Chạy đánh giá LCB song song trên danh sách tasks.

    Args:
        tasks: List of task dicts, mỗi dict có keys:
               task_id, problem_description, hidden_test_cases
        model_name_or_path: Tên model LLM
        max_workers: Số luồng chạy song song
        timeout: Giới hạn chạy pytest (giây)
        pred_dname: Thư mục lưu kết quả

    Returns:
        report dict
    """
    os.makedirs(pred_dname, exist_ok=True)
    harness = LCBHarness(workspace_dir=os.path.join(pred_dname, "sandbox"))

    results = []
    resolved_ids = []
    failed_ids = []
    error_ids = []
    empty_ids = []

    start_time = time.time()

    task_pass_rates = []

    for i, task in enumerate(tasks):
        task_id = task["task_id"]
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(tasks)}] Processing: {task_id} ({num_samples} samples)")
        print(f"{'='*60}")
        
        samples_passed = 0
        samples_failed = 0
        samples_error = 0
        samples_empty = 0
        task_predictions = []

        for sample_idx in range(num_samples):
            # 1. Gọi LLM để sinh code + Self-Debug Loop
            temperature = 0.0 if num_samples == 1 else 0.7
            try:
                llm_response, debug_rounds, passed_example = call_llm_with_self_debug(
                    task, model_name_or_path, temperature,
                    max_debug_rounds=max_debug_rounds
                )
            except Exception as e:
                print(f"❌ Sample {sample_idx+1}: LLM Error for {task_id}: {e}")
                samples_error += 1
                task_predictions.append({
                    "sample_idx": sample_idx,
                    "status": "error",
                    "error": str(e),
                })
                continue

            if not llm_response or not llm_response.strip():
                print(f"⚠️ Sample {sample_idx+1}: Empty response for {task_id}")
                samples_empty += 1
                task_predictions.append({
                    "sample_idx": sample_idx,
                    "status": "empty",
                })
                continue

            # 2. Chạy hidden test cases (pytest)
            sample_task_id = f"{task_id}_s{sample_idx}"
            is_pass, logs = harness.evaluate_task(
                task_id=sample_task_id,
                problem_description=task["problem_description"],
                hidden_test_cases=task["hidden_test_cases"],
                llm_response=llm_response,
                timeout=timeout,
            )

            if is_pass:
                samples_passed += 1
            else:
                samples_failed += 1

            # 3. Lưu prediction cho sample này
            task_predictions.append({
                "sample_idx": sample_idx,
                "status": "resolved" if is_pass else "failed",
                "llm_response": llm_response,
                "solution_code": harness.extract_python_code(llm_response),
                "test_logs": logs[:2000],
                "debug_rounds": debug_rounds,
                "passed_example": passed_example,
            })

        # Tổng hợp kết quả cho task này
        pred_file = os.path.join(pred_dname, f"{task_id}.json")
        with open(pred_file, "w", encoding="utf-8") as f:
            json.dump({
                "task_id": task_id,
                "model_name_or_path": model_name_or_path,
                "num_samples": num_samples,
                "passed": samples_passed,
                "predictions": task_predictions
            }, f, indent=2, ensure_ascii=False)

        task_pass_rates.append(samples_passed)
        if samples_passed > 0:
            resolved_ids.append(task_id)
        else:
            failed_ids.append(task_id)

    elapsed = time.time() - start_time
    
    # Tính pass@k tổng quan
    n = num_samples
    pass_at_1_sum = sum(calculate_pass_at_k(n, c, 1) for c in task_pass_rates)
    pass_at_5_sum = sum(calculate_pass_at_k(n, c, 5) for c in task_pass_rates) if n >= 5 else 0
    pass_at_10_sum = sum(calculate_pass_at_k(n, c, 10) for c in task_pass_rates) if n >= 10 else 0

    total = max(len(tasks), 1)
    pass_at_1 = round((pass_at_1_sum / total) * 100, 2)
    pass_at_5 = round((pass_at_5_sum / total) * 100, 2) if n >= 5 else None
    pass_at_10 = round((pass_at_10_sum / total) * 100, 2) if n >= 10 else None

    # 4. Tạo report tổng hợp
    report = {
        "model_name_or_path": model_name_or_path,
        "num_samples": num_samples,
        "total_tasks": len(tasks),
        "resolved_at_least_once": len(resolved_ids),
        "failed_all": len(failed_ids),
        "pass_at_1": pass_at_1,
        "pass_at_5": pass_at_5,
        "pass_at_10": pass_at_10,
        "elapsed_seconds": round(elapsed, 1),
        "resolved_ids": resolved_ids,
        "failed_ids": failed_ids,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    report_file = os.path.join(pred_dname, "lcb_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"📊 LCB EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Model:       {model_name_or_path}")
    print(f"  Total Tasks: {len(tasks)}")
    print(f"  Samples/Task:{num_samples}")
    print(f"  ✅ Passed >=1:{len(resolved_ids)}")
    print(f"  ❌ Failed All:{len(failed_ids)}")
    print(f"  📈 Pass@1:   {pass_at_1}%")
    if pass_at_5 is not None:
        print(f"  📈 Pass@5:   {pass_at_5}%")
    if pass_at_10 is not None:
        print(f"  📈 Pass@10:  {pass_at_10}%")
    print(f"  ⏱️  Time:     {report['elapsed_seconds']}s")
    print(f"  📁 Report:   {report_file}")
    print(f"{'='*60}")

    return report


# ==========================================
# BUILT-IN DEMO TASKS (để test harness)
# ==========================================
DEMO_TASKS = [
    {
        "task_id": "lcb_001_two_sum",
        "problem_description": (
            "Given an array of integers nums and an integer target, "
            "return indices of the two numbers such that they add up to target. "
            "Name your function `twoSum`."
        ),
        "hidden_test_cases": """
def test_twoSum_basic():
    assert twoSum([2, 7, 11, 15], 9) == [0, 1]

def test_twoSum_middle():
    assert twoSum([3, 2, 4], 6) == [1, 2]

def test_twoSum_same():
    assert twoSum([3, 3], 6) == [0, 1]
""",
    },
    {
        "task_id": "lcb_002_palindrome",
        "problem_description": (
            "Given an integer x, return True if x is a palindrome, "
            "and False otherwise. Name your function `isPalindrome`."
        ),
        "hidden_test_cases": """
def test_palindrome_positive():
    assert isPalindrome(121) == True

def test_palindrome_negative():
    assert isPalindrome(-121) == False

def test_palindrome_ten():
    assert isPalindrome(10) == False

def test_palindrome_zero():
    assert isPalindrome(0) == True
""",
    },
    {
        "task_id": "lcb_003_fizzbuzz",
        "problem_description": (
            "Given an integer n, return a list of strings from 1 to n where: "
            "multiples of 3 are 'Fizz', multiples of 5 are 'Buzz', "
            "multiples of both are 'FizzBuzz', otherwise the number as a string. "
            "Name your function `fizzBuzz`."
        ),
        "hidden_test_cases": """
def test_fizzbuzz_15():
    result = fizzBuzz(15)
    assert result[0] == '1'
    assert result[2] == 'Fizz'
    assert result[4] == 'Buzz'
    assert result[14] == 'FizzBuzz'
    assert len(result) == 15

def test_fizzbuzz_1():
    assert fizzBuzz(1) == ['1']
""",
    },
]


def main():
    parser = argparse.ArgumentParser(description="LiveCodeBench Harness for DGM")
    parser.add_argument(
        "--tasks", type=str, default=None,
        help="Path to tasks JSON file. If not provided, uses built-in demo tasks.",
    )
    parser.add_argument(
        "--model", type=str, default="hosted_vllm/Qwen/Qwen3.5-35B-A3B-FP8",
        help="Model name or path.",
    )
    parser.add_argument(
        "--max_workers", type=int, default=5,
        help="Number of parallel workers.",
    )
    parser.add_argument(
        "--num_samples", type=int, default=1,
        help="Number of candidate solutions to generate per task for pass@k evaluation.",
    )
    parser.add_argument(
        "--timeout", type=int, default=5,
        help="Pytest execution timeout in seconds.",
    )
    parser.add_argument(
        "--max_debug_rounds", type=int, default=3,
        help="Max Self-Debug / Reflexion rounds per sample (0 = disable).",
    )
    parser.add_argument(
        "--pred_dname", type=str, default="dgm_agent/lcb_predictions",
        help="Output directory for predictions.",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run with built-in demo tasks (no LLM call, uses hardcoded solutions).",
    )
    args = parser.parse_args()

    if args.demo:
        # Quick self-test mode: dùng solution hardcode, không gọi LLM
        print("🧪 Running DEMO mode (hardcoded solutions, no LLM)...")
        harness = LCBHarness()

        demo_solutions = {
            "lcb_001_two_sum": """```python
def twoSum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        diff = target - num
        if diff in seen:
            return [seen[diff], i]
        seen[num] = i
```""",
            "lcb_002_palindrome": """```python
def isPalindrome(x):
    if x < 0:
        return False
    return str(x) == str(x)[::-1]
```""",
            "lcb_003_fizzbuzz": """```python
def fizzBuzz(n):
    result = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            result.append('FizzBuzz')
        elif i % 3 == 0:
            result.append('Fizz')
        elif i % 5 == 0:
            result.append('Buzz')
        else:
            result.append(str(i))
    return result
```""",
        }

        passed = 0
        total = len(DEMO_TASKS)
        for task in DEMO_TASKS:
            tid = task["task_id"]
            is_pass, logs = harness.evaluate_task(
                task_id=tid,
                problem_description=task["problem_description"],
                hidden_test_cases=task["hidden_test_cases"],
                llm_response=demo_solutions.get(tid, ""),
            )
            if is_pass:
                passed += 1

        print(f"\n🏁 Demo kết quả: {passed}/{total} PASSED")
        return

    # Load tasks
    if args.tasks:
        with open(args.tasks, "r", encoding="utf-8") as f:
            tasks = json.load(f)
    else:
        tasks = DEMO_TASKS

    # Run evaluation
    run_lcb_evaluation(
        tasks=tasks,
        model_name_or_path=args.model,
        max_workers=args.max_workers,
        timeout=args.timeout,
        pred_dname=args.pred_dname,
        num_samples=args.num_samples,
        max_debug_rounds=args.max_debug_rounds,
    )


if __name__ == "__main__":
    main()
