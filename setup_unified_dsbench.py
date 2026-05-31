"""
Setup Unified DS-Bench Experiment
===================================
Script này chuẩn bị dữ liệu thực nghiệm theo đúng setup của bài báo 2510.01285v2:
1. Unroll (trải phẳng) các câu hỏi trong DS-Bench thành từng bài độc lập.
2. Lọc câu hỏi dựa trên danh sách ID từ Appendix G.
3. Gộp TẤT CẢ các file dữ liệu của các bài được chọn vào chung MỘT Unified Data Lake.
   (Xử lý trùng tên thông minh: chỉ đổi tên nếu file bị trùng).
4. Sinh ra `unified_manifest.jsonl` trỏ tất cả câu hỏi về chung Unified Data Lake.

Usage:
    python setup_unified_dsbench.py
"""

import json
import os
import shutil
import sys
from pathlib import Path

# Danh sách ID đã được lọc (Filtered IDs) từ Appendix G của bài báo
DSBENCH_FILTERED_IDS = [
    "00000001_question17", "00000001_question7", "00000001_question16", "00000001_question13",
    "00000001_question12", "00000001_question9", "00000001_question10", "00000001_question11",
    "00000001_question18", "00000001_question15", "00000004_question50", "00000004_question45",
    "00000004_question42", "00000004_question48", "00000004_question47", "00000004_question49",
    "00000004_question44", "00000004_question41", "00000004_question43", "00000005_question21",
    "00000005_question29", "00000005_question28", "00000005_question24", "00000005_question25",
    "00000005_question23", "00000005_question26", "00000005_question20", "00000005_question27",
    "00000006_question17", "00000006_question22", "00000006_question21", "00000006_question18",
    "00000006_question24", "00000006_question25", "00000006_question19", "00000006_question23",
    "00000006_question26", "00000006_question20", "00000007_question17", "00000007_question22",
    "00000007_question7",  "00000007_question16", "00000007_question6",  "00000007_question5",
    "00000007_question13", "00000007_question12", "00000007_question21", "00000007_question9",
    "00000007_question10", "00000007_question11", "00000007_question3",  "00000007_question18",
    "00000007_question4",  "00000007_question19", "00000007_question23", "00000007_question14",
    "00000007_question2",  "00000007_question15", "00000007_question20", "00000007_question1",
]


def setup_unified_experiment(
    cache_dir: str = "data/local_test/_hf_cache/data_analysis",
    output_lake_dir: str = "data/unified_data_lake",
    output_manifest: str = "data/unified_manifest.jsonl"
):
    cache_path = Path(cache_dir)
    data_json_path = cache_path / "data.json"
    extract_dir = cache_path / "extracted"

    if not data_json_path.exists():
        print(f"Lỗi: Không tìm thấy {data_json_path}. Vui lòng chạy download_dsbench.py trước.")
        sys.exit(1)

    lake_path = Path(output_lake_dir)
    lake_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Đọc data.json (chứa các tasks, mỗi task có nhiều questions)
    tasks_data = []
    with open(data_json_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks_data.append(json.loads(line))
    
    print(f"Đã đọc {len(tasks_data)} tasks gốc từ data.json")

    # 2. Lọc và trải phẳng (unroll)
    filtered_instances = []
    task_id_to_files = {}  # Map task_id -> list of file paths in extract_dir

    for row in tasks_data:
        task_id = str(row.get("id"))
        task_name = row.get("name", task_id)
        questions = row.get("questions", [])
        answers = row.get("answers", [])
        base_text = row.get("txt", "")

        # Find files for this task
        if extract_dir.exists():
            candidates = []
            for search_name in [task_name, task_id, task_name.replace("-", "_")]:
                found = list(extract_dir.rglob(f"*{search_name}*"))
                candidates.extend([f for f in found if f.is_file()])
            for d in extract_dir.rglob("*"):
                if d.is_dir() and (task_name in d.name or task_id in d.name):
                    for f in d.iterdir():
                        if f.is_file():
                            candidates.append(f)
            
            # Deduplicate
            seen = set()
            unique_files = []
            for f in candidates:
                if f not in seen:
                    seen.add(f)
                    unique_files.append(f)
            task_id_to_files[task_id] = unique_files

        # Unroll questions
        for idx, q_name in enumerate(questions):
            instance_id = f"{task_id}_{q_name}"
            if instance_id in DSBENCH_FILTERED_IDS:
                # Tìm được câu hỏi hợp lệ!
                question_text = f"{base_text}\n\nQuestion: {q_name}"
                answer_val = answers[idx] if idx < len(answers) else ""
                
                filtered_instances.append({
                    "task_id": instance_id,
                    "parent_task_id": task_id,
                    "question": question_text.strip(),
                    "answer": answer_val,
                    "original_files": task_id_to_files.get(task_id, [])
                })

    print(f"Đã lọc được {len(filtered_instances)} câu hỏi dựa trên Appendix G.")
    if len(filtered_instances) == 0:
        print("Cảnh báo: Không tìm thấy câu hỏi nào. Có thể ID không khớp định dạng.")
        sys.exit(1)

    # 3. Gộp files vào Unified Data Lake (Xử lý trùng tên)
    manifest_entries = []
    copied_filenames = set()
    total_files_copied = 0

    print(f"Đang gom file vào {lake_path}...")
    for inst in filtered_instances:
        # Copy file của instance này vào lake
        for src_file in inst["original_files"]:
            fname = src_file.name
            dst_file = lake_path / fname
            
            # Nếu file đã tồn tại trong lake
            if dst_file.exists():
                # So sánh kích thước để xem có phải cùng 1 file không
                if dst_file.stat().st_size != src_file.stat().st_size:
                    # Trùng tên nhưng khác nội dung -> Đổi tên
                    fname = f"{inst['parent_task_id']}_{fname}"
                    dst_file = lake_path / fname

            if fname not in copied_filenames:
                shutil.copy2(src_file, dst_file)
                copied_filenames.add(fname)
                total_files_copied += 1

        # Tạo entry cho manifest
        entry = {
            "task_id": inst["task_id"],
            "question": inst["question"],
            "answer": inst["answer"],
            "task_dir": str(lake_path.resolve())  # Trỏ TẤT CẢ về hồ chung!
        }
        manifest_entries.append(entry)

    print(f"Đã copy {total_files_copied} files độc lập vào Unified Data Lake.")

    # 4. Ghi manifest
    manifest_path = Path(output_manifest)
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Đã ghi manifest tại: {manifest_path}")
    print(f"Sẵn sàng benchmark với Unified Data Lake!")

if __name__ == "__main__":
    setup_unified_experiment()
