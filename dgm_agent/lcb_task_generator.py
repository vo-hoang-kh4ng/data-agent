import os
import json
import zlib
import pickle
import base64
import random

def extract_lcb_v1():
    print("Reading LCB v1 release data from dgm_agent/lcb_dataset/test.jsonl...")
    
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lcb_dataset", "test.jsonl")
    tasks = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            
            task_id = data.get("question_id", "lcb_unknown")
            desc = data.get("question_content", data.get("description", ""))
            
            metadata = data.get("metadata", {})
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            func_name = metadata.get("func_name", "main")
            
            raw_cases = data.get("private_test_cases", [])
            cases = []
            if isinstance(raw_cases, str):
                try:
                    cases = json.loads(raw_cases)
                except:
                    try:
                        cases = json.loads(
                            pickle.loads(
                                zlib.decompress(base64.b64decode(raw_cases.encode("utf-8")))
                            )
                        )
                    except Exception:
                        continue
            else:
                cases = raw_cases
                
            # Safely generate the test code.
            # We determine the module name dynamically from __name__.
            # We mock stdin BEFORE any execution or import to prevent hanging on top-level input()
            
            test_code = f"""
def test_evaluation_loop():
    import json
    import ast
    import sys
    import runpy
    import importlib
    from io import StringIO
    
    cases = {repr(cases)}
    module_name = __name__.replace('test_', 'solution_')
    
    for idx, case in enumerate(cases):
        inp_str = case.get('input', '')
        out_str = case.get('output', '')
        testtype = case.get('testtype', 'functional')
        
        if testtype == 'stdin':
            # Mock sys.stdin and sys.stdout
            original_stdin = sys.stdin
            original_stdout = sys.stdout
            sys.stdin = StringIO(inp_str)
            fake_out = StringIO()
            sys.stdout = fake_out
            
            try:
                # Run module directly as script
                runpy.run_module(module_name, run_name="__main__")
            except SystemExit:
                pass
            finally:
                sys.stdin = original_stdin
                sys.stdout = original_stdout
            
            result = fake_out.getvalue().strip()
            expected = out_str.strip()
            assert result == expected, f"Stdin case {{idx}} failed"
            
        else:
            # Functional test
            try:
                expected = ast.literal_eval(out_str)
            except:
                expected = out_str.strip()
                
            if not inp_str.strip():
                continue
            
            # Since functional tests usually don't have top-level blocking input(), we can import them safely.
            sol_mod = importlib.import_module(module_name)
            func = getattr(sol_mod, "{func_name}")
            
            # Evaluate the arguments string to a tuple of arguments
            args = ast.literal_eval(f"({{inp_str}},)")
            if isinstance(args, tuple) and len(args) == 1 and not inp_str.strip().endswith(","):
                # If ast.literal_eval parses it as a single value rather than a tuple, we wrap it
                args = (args,)
            elif isinstance(args, tuple) and len(args) >= 1:
                pass
            else:
                args = (args,)
                
            result = func(*args[0]) if len(args) == 1 and type(args[0]) == tuple else func(*args)
            
            # Allow flexible matching for output
            assert str(result).strip() == str(expected).strip() or result == expected, f"Functional case {{idx}} failed"
"""
            
            tasks.append({
                "task_id": str(task_id).replace("/", "_").replace("-", "_"),
                "problem_description": desc,
                "hidden_test_cases": test_code,
                "starter_code": data.get("starter_code", "")
            })
            
    print(f"Total tasks processed: {len(tasks)}")
    random.seed(42)
    random.shuffle(tasks)
    val_tasks = tasks[:100]
    
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lcb_tasks")
    val_path = os.path.join(out_dir, "tasks_100_val.json")
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_tasks, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    extract_lcb_v1()
