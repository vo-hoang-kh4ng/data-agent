import json
import os

report_path = r"d:\kh4ng\Data_agent\dgm_agent\outer_eval_report.json"
if os.path.exists(report_path):
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    results = data.get("task_results", [])
    lang_stats = {}
    for r in results:
        lang = r.get("language", "unknown")
        is_pass = r.get("pass", False)
        status = r.get("status", "FAIL")
        
        if lang not in lang_stats:
            lang_stats[lang] = {"total": 0, "pass": 0, "skip": 0}
        
        if status == "SKIP":
            lang_stats[lang]["skip"] += 1
        else:
            lang_stats[lang]["total"] += 1
            if is_pass:
                lang_stats[lang]["pass"] += 1
                
    print("Language Stats:")
    for lang, stats in lang_stats.items():
        total = stats["total"]
        passed = stats["pass"]
        skipped = stats["skip"]
        rate = (passed / total * 100) if total > 0 else 0
        print(f"  {lang}: {passed}/{total} passed ({rate:.1f}%), {skipped} skipped")
else:
    print("Report not found")
