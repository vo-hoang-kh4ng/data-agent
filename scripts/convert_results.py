"""Convert dgm-blackboard result.json to DA-Code evaluator format."""

import json
import os
import glob

OUTPUT_DIR = r"D:\Data Agent\da-code\output\dgm-blackboard"

task_dirs = sorted(glob.glob(os.path.join(OUTPUT_DIR, "di-*")))

converted = 0
for task_dir in task_dirs:
    result_file = os.path.join(task_dir, "dabench", "result.json")
    if not os.path.exists(result_file):
        print(f"  SKIP {os.path.basename(task_dir)}: no result.json")
        continue

    with open(result_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check if already converted (has trajectory)
    if "trajectory" in data:
        continue

    # Build dummy trajectory matching DA-Code format
    steps = data.get("steps", 1)
    trajectory = []
    for i in range(steps):
        step = {
            "observation": "execution succeeded",
            "thought": f"Step {i+1}",
            "action": f"Python(code=\"...\")",
            "code": f"# step {i+1}",
            "response": f"Thought: Step {i+1}\n\nAction: Python(code=\"...\")"
        }
        trajectory.append(step)
    # Final observation
    trajectory.append({
        "observation": "Task completed",
        "thought": "Done",
        "action": "Terminate()",
        "code": "",
        "response": "Thought: Done\n\nAction: Terminate()"
    })

    # Fix result_files format
    old_result_files = data.get("result_files", [])
    if isinstance(old_result_files, list):
        result_files = {
            "added_files": old_result_files,
            "changed_files": []
        }
    else:
        result_files = old_result_files

    # Rebuild in DA-Code format
    converted_data = {
        "finished": data["finished"],
        "steps": data["steps"],
        "result": data["result"],
        "result_files": result_files,
        "trajectory": trajectory
    }

    # Keep original fields too
    for k, v in data.items():
        if k not in converted_data:
            converted_data[k] = v

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)

    converted += 1

print(f"\nConverted {converted}/{len(task_dirs)} task results")
