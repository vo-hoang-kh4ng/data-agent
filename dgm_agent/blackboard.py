"""
Blackboard Architecture for Data Science Tasks
================================================
Triển khai cơ chế Blackboard phân tán theo kiến trúc đề xuất trong bài báo
'LLM-based Multi-Agent Blackboard System for Information Discovery in Data Science'
(2510.01285v2), tích hợp vào framework DGM-Agent.

Luồng hoạt động:
  1. MainAgent post Request lên Blackboard ("Cần tìm file chứa cột 'Age' và 'APP-Z'").
  2. FileAgent quét Data Lake → đọc header + 20 dòng đầu → post Response.
  3. MainAgent tổng hợp Context → gọi LLM sinh code pandas.
  4. Kết quả code được chạy qua Sandbox (subprocess).

Usage:
    from dgm_agent.blackboard import Blackboard, BlackboardAgent, run_ds_pipeline
"""

import json
import os
import re
import subprocess
import sys
import time
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(dotenv_path):
        with open(dotenv_path, encoding="utf-8") as fenv:
            for line in fenv:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# ────────────────────────────── Data Classes ──────────────────────────────── #

@dataclass
class BlackboardRequest:
    """Yêu cầu được đăng lên Bảng Đen bởi Main Agent."""
    task_id: str
    question: str
    data_lake_dir: str          # Thư mục chứa file dữ liệu của bài toán


@dataclass
class FileContext:
    """Ngữ cảnh đọc được từ một file, do FileAgent tạo ra."""
    file_path: str
    file_type: str              # 'csv', 'xlsx', 'json', 'txt', ...
    columns: List[str] = field(default_factory=list)
    dtypes: Dict[str, str] = field(default_factory=dict)
    sheets: List[str] = field(default_factory=list)
    preview: str = ""           # 20 dòng đầu dưới dạng text
    error: Optional[str] = None


@dataclass
class BlackboardResponse:
    """Phản hồi được đăng lên Bảng Đen bởi File Agent."""
    agent_id: str
    cluster_name: str
    file_contexts: List[FileContext] = field(default_factory=list)
    relevance_score: float = 0.0    # Độ liên quan với Request (0.0 – 1.0)


# ─────────────────────────── Blackboard (Hub) ─────────────────────────────── #

class Blackboard:
    """
    Bảng Đen trung tâm: nơi Main Agent đăng yêu cầu và
    File Agent đăng phản hồi theo cơ chế pub/sub phi tập trung.
    """

    def __init__(self):
        self.request: Optional[BlackboardRequest] = None
        self.responses: List[BlackboardResponse] = []

    def post_request(self, request: BlackboardRequest):
        """Main Agent đăng yêu cầu lên bảng."""
        self.request = request
        self.responses = []
        print(f"\n📋 [Blackboard] Request đăng: '{request.question[:80]}...'")

    def post_response(self, response: BlackboardResponse):
        """File Agent đăng phản hồi lên bảng."""
        self.responses.append(response)

    def get_context_summary(self) -> str:
        """
        Tổng hợp toàn bộ ngữ cảnh file từ tất cả File Agent
        thành một chuỗi văn bản để đưa vào Prompt của Coding Agent.
        """
        if not self.responses:
            return "Không tìm thấy file dữ liệu liên quan trong Data Lake."

        # Sắp xếp theo độ liên quan giảm dần
        sorted_responses = sorted(self.responses, key=lambda r: r.relevance_score, reverse=True)

        lines = ["=== DATA LAKE CONTEXT (Blackboard Summary) ===\n"]
        for resp in sorted_responses:
            lines.append(f"[Cluster: {resp.cluster_name}]")
            for fc in resp.file_contexts:
                if fc.error:
                    lines.append(f"  File: {fc.file_path}  → ⚠️ Error: {fc.error}")
                    continue
                lines.append(f"  File: {fc.file_path}  (type={fc.file_type})")
                if fc.sheets:
                    lines.append(f"    Sheets: {fc.sheets}")
                if fc.columns:
                    lines.append(f"    Columns: {fc.columns}")
                if fc.dtypes:
                    dtype_str = ", ".join(f"{k}: {v}" for k, v in list(fc.dtypes.items())[:15])
                    lines.append(f"    Dtypes: {dtype_str}")
                if fc.preview:
                    lines.append(f"    Preview (20 rows):\n{fc.preview}")
            lines.append("")
            
        full_context = "\n".join(lines)
        # Context limit — keep under 80K chars to leave room for question + code
        if len(full_context) > 80000:
            full_context = full_context[:80000] + "\n...[CONTEXT TRUNCATED DUE TO LENGTH LIMIT]..."
            
        return full_context


# ─────────────────────────── File Agent ───────────────────────────────────── #

class FileAgent:
    """
    Agent quản lý một cụm (cluster) file dữ liệu.
    Tự động đọc header + 20 dòng đầu và đánh giá độ liên quan với Request.
    """

    def __init__(self, agent_id: str, cluster_name: str, file_paths: List[str]):
        self.agent_id = agent_id
        self.cluster_name = cluster_name
        self.file_paths = file_paths

    # ── Đọc file ── #

    def _prune_dataframe(self, df: pd.DataFrame, goal_hint: str) -> str:
        """Return header + first 20 rows as text. No NCD pruning — keep raw samples."""
        if df.empty:
            return "(empty dataframe)"
        return df.head(20).to_string(index=False, max_rows=20, max_cols=15)

    def _prune_text(self, lines: List[str], goal_hint: str) -> str:
        """Return first 20 lines. No NCD pruning — keep raw samples."""
        if not lines:
            return ""
        return "".join(lines[:20])[:2000]

    def _read_csv(self, fpath: str, goal_hint: str) -> FileContext:
        try:
            df = pd.read_csv(fpath, nrows=1000, encoding="utf-8", on_bad_lines="skip")
            preview = self._prune_dataframe(df, goal_hint)
            return FileContext(
                file_path=fpath,
                file_type="csv",
                columns=list(df.columns),
                dtypes={col: str(df[col].dtype) for col in df.columns},
                preview=preview,
            )
        except Exception as e:
            return FileContext(file_path=fpath, file_type="csv", error=str(e))

    def _read_xlsx(self, fpath: str, goal_hint: str) -> FileContext:
        try:
            xl = pd.ExcelFile(fpath)
            sheets = xl.sheet_names
            # Đọc sheet đầu tiên để lấy cột + preview
            df = pd.read_excel(fpath, sheet_name=sheets[0], nrows=1000)
            preview = self._prune_dataframe(df, goal_hint)
            return FileContext(
                file_path=fpath,
                file_type="xlsx",
                columns=list(df.columns),
                dtypes={col: str(df[col].dtype) for col in df.columns},
                sheets=sheets,
                preview=preview,
            )
        except Exception as e:
            return FileContext(file_path=fpath, file_type="xlsx", error=str(e))

    def _read_json(self, fpath: str, goal_hint: str) -> FileContext:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            preview = self._prune_text(lines, goal_hint)
            
            try:
                obj = json.loads("".join(lines))
                if isinstance(obj, list) and obj:
                    cols = list(obj[0].keys()) if isinstance(obj[0], dict) else []
                else:
                    cols = []
            except Exception:
                cols = []
            return FileContext(file_path=fpath, file_type="json", columns=cols, preview=preview)
        except Exception as e:
            return FileContext(file_path=fpath, file_type="json", error=str(e))

    def _read_generic(self, fpath: str, goal_hint: str) -> FileContext:
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            preview = self._prune_text(lines, goal_hint)
            return FileContext(file_path=fpath, file_type="txt", preview=preview)
        except Exception as e:
            return FileContext(file_path=fpath, file_type="other", error=str(e))

    def read_file(self, fpath: str, goal_hint: str) -> FileContext:
        ext = Path(fpath).suffix.lower()
        if ext == ".csv":
            return self._read_csv(fpath, goal_hint)
        elif ext in (".xlsx", ".xls", ".ods"):
            return self._read_xlsx(fpath, goal_hint)
        elif ext == ".json":
            return self._read_json(fpath, goal_hint)
        else:
            return self._read_generic(fpath, goal_hint)

    # ── Đánh giá độ liên quan ── #

    def _compute_relevance(self, question: str, file_contexts: List[FileContext]) -> float:
        """
        Tính độ liên quan dựa trên keyword matching giữa câu hỏi
        và tên cột + tên file. Đơn giản nhưng hiệu quả với DS-Bench.
        """
        question_lower = question.lower()
        # Tokenize thành từ, loại bỏ stop words
        keywords = set(re.findall(r"\b\w{3,}\b", question_lower))
        stop_words = {"the", "and", "for", "with", "from", "this", "that", "are", "have",
                      "what", "find", "which", "data", "calculate", "compute", "using", "each"}
        keywords -= stop_words

        if not keywords:
            return 0.1

        match_count = 0
        total_signals = 0

        for fc in file_contexts:
            if fc.error:
                continue
            # Kiểm tra tên cột
            for col in fc.columns:
                col_lower = col.lower().replace("_", " ").replace("-", " ")
                col_tokens = set(re.findall(r"\b\w{3,}\b", col_lower))
                if col_tokens & keywords:
                    match_count += len(col_tokens & keywords)
                    total_signals += 1
            # Kiểm tra tên file
            fname_lower = Path(fc.file_path).stem.lower()
            fname_tokens = set(re.findall(r"\b\w{3,}\b", fname_lower))
            if fname_tokens & keywords:
                match_count += 1

        return min(1.0, match_count / max(len(keywords), 1) * 0.8) if total_signals > 0 else 0.05

    # ── Trigger khi có request ── #

    def handle_request(self, request: BlackboardRequest) -> BlackboardResponse:
        """Đọc các file trong cụm và trả về FileContexts."""
        print(f"  🤖 FileAgent [{self.agent_id}] đang quét cụm '{self.cluster_name}' ({len(self.file_paths)} files)...")
        file_contexts = [self.read_file(fp, request.question) for fp in self.file_paths]
        relevance = self._compute_relevance(request.question, file_contexts)
        return BlackboardResponse(
            agent_id=self.agent_id,
            cluster_name=self.cluster_name,
            file_contexts=file_contexts,
            relevance_score=relevance,
        )


# ─────────────────────────── Cluster Factory ──────────────────────────────── #

# Solution-methodology hints / gold-template files that must NEVER be ingested into the
# agent's context. The data-discovery benchmark is only valid if the agent sees raw data
# files — never answer methodology, category mappings, gold outputs, or filename hints.
# (Benchmark-integrity guardrail; see DA-Code-README verification.)
HINT_EXACT_NAMES = {
    "README.md", ".DS_Store",
    "result.csv", "sample_result.csv",            # gold output templates
    "tips.txt", "tips.md", "guidance.txt", "step.md", "workflow.md",
    "data_standard.md", "weight_class.md",
    "playerposition.txt", "BMI.txt", "age.txt", "iqr.txt",
    "relevant_avocado_categories.txt", "relevant_olive_oil_categories.txt",
    "relevant_sourdough_categories.txt",
}
# Substrings (matched against the file STEM, lowercased) flagging task-prefixed hint
# variants produced by the dedup collision path (e.g. "dm-csv-020_guidance.txt").
HINT_STEM_SUBSTRINGS = ("tips", "guidance", "step", "workflow", "data_standard",
                        "playerposition", "relevant_", "weight_class")


def _is_hint_or_gold_file(path) -> bool:
    """True if `path` is a solution hint / gold template that must be excluded from the
    agent's view of the data lake (answer leakage prevention)."""
    name = path.name
    if name in HINT_EXACT_NAMES or name.startswith('.'):
        return True
    stem = path.stem.lower()
    return any(s in stem for s in HINT_STEM_SUBSTRINGS)


def build_file_agents(data_lake_dir: str, max_files_per_cluster: int = 8, use_semantic: bool = True) -> List[FileAgent]:
    """
    Tự động nhóm các file trong data_lake_dir thành các cụm (cluster).
    Ưu tiên E5-Large semantic clustering, fallback sang prefix clustering.
    """
    data_root = Path(data_lake_dir)
    if not data_root.exists():
        print(f"  ⚠️ Data Lake không tồn tại: {data_lake_dir}")
        return []

    DATA_EXTS = {".csv", ".xlsx", ".xls", ".ods", ".json", ".txt", ".tsv"}
    all_files = [
        str(f) for f in data_root.rglob("*")
        if f.is_file() and f.suffix.lower() in DATA_EXTS and not _is_hint_or_gold_file(f)
    ]
    _rejected = sum(
        1 for f in data_root.rglob("*")
        if f.is_file() and f.suffix.lower() in DATA_EXTS and _is_hint_or_gold_file(f)
    )
    if _rejected:
        print(f"  🛡️  Excluded {_rejected} hint/gold-template file(s) from agent context (data-discovery integrity).")

    if not all_files:
        print(f"  ⚠️ Không tìm thấy file dữ liệu trong: {data_lake_dir}")
        return []

    # ── Method 0b: Hierarchical LLM-based filename clustering ── #
    # TDGM_FROZEN_CLUSTER_HOOK_V1
    _tdgm_frozen_clusters = os.environ.get('TDGM_FROZEN_CLUSTER_FILE', '')
    if _tdgm_frozen_clusters:
        from dgm_agent.evolution.frozen_clustering import load_frozen_clusters
        _tdgm_materialized = load_frozen_clusters(
            Path(_tdgm_frozen_clusters), data_root, expected_k=26
        )
        agents = [
            FileAgent(agent_id=f'file_agent_{idx:02d}', cluster_name=item['name'], file_paths=item['file_paths'])
            for idx, item in enumerate(_tdgm_materialized)
        ]
        print(f'  Frozen Hierarchical Clustering: {len(agents)} clusters')
        return agents

    clustering_method = os.environ.get("DACODE_CLUSTERING_METHOD", "kmeans").lower()
    if clustering_method == "llm_hierarchical":
        try:
            from llm import create_client, get_response_from_llm, extract_json_between_markers
            
            llm_model = os.environ.get("DACODE_LLM_CLUSTERING_MODEL", "deepseek-chat")
            print(f"  🌐 Using Hierarchical LLM-based filename clustering with model: {llm_model}...")
            
            # Read K cluster limit for LLM-based clustering
            K_LIMIT = os.environ.get("DACODE_LLM_CLUSTERING_K", "26")
            if K_LIMIT.lower() == "auto":
                K_VAL = None
                print(f"  🎯 Target count: AUTOMATIC (Natural) clusters for {len(all_files)} files...")
            else:
                K_VAL = int(K_LIMIT) if K_LIMIT.isdigit() else 26
                K_VAL = max(1, min(K_VAL, len(all_files)))
                print(f"  🎯 Target count: K={K_VAL} clusters for {len(all_files)} files...")
            
            # Helper to form file addresses input list
            def get_rel_paths(files_list):
                res = []
                for fpath in files_list:
                    rel = str(Path(fpath).relative_to(data_root))
                    res.append(rel)
                return res
            
            # Helper function for calling Vertex AI
            def call_llm_for_clustering(prompt: str, system_instruction: str = None) -> str:
                if llm_model == "google/gemini-2.5-pro":
                    import google.auth
                    from google.auth.transport.requests import Request
                    import requests
                    
                    credentials, project_id = google.auth.default(
                        scopes=["https://www.googleapis.com/auth/cloud-platform"]
                    )
                    credentials.refresh(Request())
                    proj = project_id or "gen-lang-client-0072409547"
                    url = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{proj}/locations/us-central1/publishers/google/models/gemini-2.5-pro:generateContent"
                    
                    headers = {
                        "Authorization": f"Bearer {credentials.token}",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "contents": [{
                            "role": "user",
                            "parts": [{"text": prompt}]
                        }],
                        "generationConfig": {
                            "temperature": 0.1,
                            "maxOutputTokens": 8192,
                            "thinkingConfig": {
                                "thinkingBudget": 1024
                            }
                        }
                    }
                    if system_instruction:
                        payload["systemInstruction"] = {
                            "parts": [{"text": system_instruction}]
                        }
                    
                    response = requests.post(url, headers=headers, json=payload, timeout=120)
                    if response.status_code == 200:
                        res_json = response.json()
                        return res_json["candidates"][0]["content"]["parts"][0]["text"]
                    else:
                        raise ValueError(f"Vertex API returned status code {response.status_code}: {response.text}")
                else:
                    client, client_model = create_client(llm_model)
                    response_text, _ = get_response_from_llm(
                        msg=prompt,
                        client=client,
                        model=client_model,
                        system_message=system_instruction or "You are a helpful assistant.",
                        temperature=0.0
                    )
                    return response_text

            # --- CACHING SETUP ---
            import hashlib
            all_files_sorted = sorted(all_files)
            hash_input = f"{llm_model}_{clustering_method}_{len(all_files_sorted)}_{''.join(all_files_sorted)}_K_{K_LIMIT}"
            config_hash = hashlib.md5(hash_input.encode('utf-8')).hexdigest()
            
            cache_dir = Path(data_root).parent / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"llm_clusters_{config_hash}.json"
            
            final_clusters = defaultdict(list)
            
            if cache_path.exists():
                print(f"  ♻️ Found cached Hierarchical LLM clustering results at {cache_path}. Loading...")
                with open(cache_path, "r", encoding="utf-8") as fcache:
                    cached_data = json.load(fcache)
                for cname, rel_paths in cached_data.items():
                    for rp in rel_paths:
                        final_clusters[cname].append(str(Path(data_root) / rp))
                print(f"  ✅ Loaded {len(final_clusters)} clusters from cache.")
            else:
                # If target K is close to or larger than file count, return single-file clusters
                if K_VAL is not None and K_VAL >= len(all_files):
                    final_clusters = {f"file_cluster_{idx:02d}": [f] for idx, f in enumerate(all_files)}
                else:
                    # --- STAGE 1: High-level category clustering ---
                    rel_all_files = get_rel_paths(all_files)
                    stage1_prompt = f"""You are an expert in classifying files and directories. You are given a list of file addresses.
Your task is to classify them into a set of 4 to 8 major, high-level clusters based on their names and directories.
Group similar files together (e.g., files belonging to the same category, topic, or sub-project).

# Your input:
    - file addresses: a list of file addresses and names.

# Your output:
You should generate a valid json object in ```json ``` block with the following structure:
    - "clusters": a list of valid json objects each containing:
        - "name": the name of the cluster
        - "files": a list of file addresses and names that belong to this cluster.
        - "description": a short description of the cluster

# file addresses:
{json.dumps(rel_all_files, indent=2)}"""
                    
                    print("  🌐 Calling Stage 1 high-level category clustering...")
                    response1 = call_llm_for_clustering(
                        prompt=stage1_prompt,
                        system_instruction="You are an expert in classifying files into related categories."
                    )
                    
                    parsed1 = extract_json_between_markers(response1)
                    if not parsed1 or not isinstance(parsed1, dict) or "clusters" not in parsed1:
                        raise ValueError(f"Stage 1 failed to parse valid JSON with 'clusters' key. Response preview: {response1[:300]}")
                    
                    clusters_list1 = parsed1["clusters"]
                    if not isinstance(clusters_list1, list):
                        raise ValueError("Stage 1 'clusters' is not a list in the parsed JSON.")
                    
                    # Map LLM output files back to absolute paths
                    stage1_clusters = defaultdict(list)
                    assigned_files = set()
                    cluster_descriptions = {}
                    
                    for idx, c_obj in enumerate(clusters_list1):
                        if not isinstance(c_obj, dict):
                            continue
                        cluster_name = c_obj.get("name", f"category_{idx:02d}")
                        cluster_name = re.sub(r'[\\s/\\\\:]+', '_', cluster_name.strip())
                        if not cluster_name:
                            cluster_name = f"category_{idx:02d}"
                            
                        cluster_descriptions[cluster_name] = c_obj.get("description", "")
                        files_in_c = c_obj.get("files", [])
                        if not isinstance(files_in_c, list):
                            continue
                            
                        for f_entry in files_in_c:
                            if not isinstance(f_entry, str):
                                continue
                            f_entry = f_entry.strip()
                            if not f_entry:
                                continue
                                
                            matched = False
                            # Exact match
                            for fpath in all_files:
                                rel = str(Path(fpath).relative_to(data_root))
                                if f_entry == rel or Path(f_entry).name == Path(fpath).name:
                                    stage1_clusters[cluster_name].append(fpath)
                                    assigned_files.add(fpath)
                                    matched = True
                                    break
                            if matched:
                                continue
                            # Substring match
                            for fpath in all_files:
                                rel = str(Path(fpath).relative_to(data_root))
                                if f_entry in rel or rel in f_entry:
                                    stage1_clusters[cluster_name].append(fpath)
                                    assigned_files.add(fpath)
                                    matched = True
                                    break
                                    
                    unassigned_files = [f for f in all_files if f not in assigned_files]
                    if unassigned_files:
                        print(f"  ⚠️ Stage 1: LLM omitted {len(unassigned_files)} files. Adding to 'misc' category...")
                        stage1_clusters["misc"].extend(unassigned_files)
                        cluster_descriptions["misc"] = "Miscellaneous unassigned files"
                    
                    # Filter out empty categories
                    active_categories = {k: v for k, v in stage1_clusters.items() if v}
                    num_cats = len(active_categories)
                    print(f"  Stage 1 completed: formed {num_cats} major categories.")
                    
                    # --- ALLOCATION STAGE ---
                    if K_VAL is not None:
                        allocations = {cat: 1 for cat in active_categories.keys()}
                        if num_cats >= K_VAL:
                            pass
                        else:
                            # Greedily allocate remaining sub-cluster slots
                            while sum(allocations.values()) < K_VAL:
                                best_cat = None
                                best_ratio = -1.0
                                for cat, files in active_categories.items():
                                    sub_k = allocations[cat]
                                    if sub_k < len(files):
                                        ratio = len(files) / sub_k
                                        if ratio > best_ratio:
                                            best_ratio = ratio
                                            best_cat = cat
                                if best_cat is None:
                                    break
                                allocations[best_cat] += 1
                        print(f"  Target sub-cluster allocations: {allocations}")
                    else:
                        allocations = None

                    # --- STAGE 2: Semantic Sub-clustering ---
                    for cat, files in active_categories.items():
                        if K_VAL is not None:
                            sub_k = allocations[cat]
                            if sub_k == 1 or len(files) <= 1:
                                final_clusters[cat] = files
                                continue
                            
                            rel_files = get_rel_paths(files)
                            desc = cluster_descriptions.get(cat, "")
                            stage2_prompt = f"""You are an expert in classifying files.
We have a group of files belonging to the category '{cat}' (description: {desc}).
Your task is to subdivide this category into exactly {sub_k} sub-clusters based on their names. Do not create more or fewer than {sub_k} sub-clusters.

# Your input files:
{json.dumps(rel_files, indent=2)}

# Your output:
You should generate a valid json object in ```json ``` block with the following structure:
    - "clusters": a list of valid json objects each containing:
        - "name": the name of the sub-cluster
        - "files": a list of file addresses that belong to this sub-cluster.

Each input file MUST be assigned to exactly one of the {sub_k} sub-clusters. Do not leave any file out. All files in the input list must appear in the files list of the clusters."""
                        else:
                            # Automatic natural subdivision
                            if len(files) <= 2:
                                final_clusters[cat] = files
                                continue
                            rel_files = get_rel_paths(files)
                            desc = cluster_descriptions.get(cat, "")
                            stage2_prompt = f"""You are an expert in classifying files.
We have a group of files belonging to the category '{cat}' (description: {desc}).
Your task is to subdivide this category into related sub-clusters based on their names. Decide the number of sub-clusters naturally based on the files.

# Your input files:
{json.dumps(rel_files, indent=2)}

# Your output:
You should generate a valid json object in ```json ``` block with the following structure:
    - "clusters": a list of valid json objects each containing:
        - "name": the name of the sub-cluster
        - "files": a list of file addresses that belong to this sub-cluster.

Each input file MUST be assigned to exactly one of the sub-clusters. Do not leave any file out. All files in the input list must appear in the files list of the clusters."""

                        try:
                            if K_VAL is not None:
                                print(f"  🌐 Calling Stage 2 sub-clustering for category '{cat}' (K={sub_k})...")
                            else:
                                print(f"  🌐 Calling Stage 2 natural sub-clustering for category '{cat}'...")
                            response2 = call_llm_for_clustering(
                                prompt=stage2_prompt,
                                system_instruction="You are an expert in subdividing files into sub-categories."
                            )
                            parsed2 = extract_json_between_markers(response2)
                            if not parsed2 or not isinstance(parsed2, dict) or "clusters" not in parsed2:
                                raise ValueError(f"Failed to parse valid JSON for sub-clustering. Response preview: {response2[:200]}")
                            
                            sub_clusters_list = parsed2["clusters"]
                            if not isinstance(sub_clusters_list, list):
                                raise ValueError("'clusters' key is not a list in sub-clustering parsed JSON.")
                                
                            sub_assigned = set()
                            for sub_idx, sub_obj in enumerate(sub_clusters_list):
                                if not isinstance(sub_obj, dict):
                                    continue
                                sub_name = sub_obj.get("name", f"sub_{sub_idx:02d}")
                                sub_name = re.sub(r'[\\s/\\\\:]+', '_', sub_name.strip())
                                if not sub_name:
                                    sub_name = f"sub_{sub_idx:02d}"
                                full_sub_name = f"{cat}_{sub_name}"
                                
                                sub_files_in = sub_obj.get("files", [])
                                if not isinstance(sub_files_in, list):
                                    continue
                                    
                                for f_entry in sub_files_in:
                                    if not isinstance(f_entry, str):
                                        continue
                                    f_entry = f_entry.strip()
                                    for fpath in files:
                                        rel = str(Path(fpath).relative_to(data_root))
                                        if f_entry == rel or Path(f_entry).name == Path(fpath).name:
                                            if full_sub_name not in final_clusters:
                                                final_clusters[full_sub_name] = []
                                            final_clusters[full_sub_name].append(fpath)
                                            sub_assigned.add(fpath)
                                            break
                                            
                            sub_unassigned = [f for f in files if f not in sub_assigned]
                            if sub_unassigned:
                                first_sub = next((k for k in final_clusters.keys() if k.startswith(f"{cat}_")), None)
                                if first_sub:
                                    final_clusters[first_sub].extend(sub_unassigned)
                                else:
                                    final_clusters[f"{cat}_sub_misc"] = sub_unassigned
                                    
                        except Exception as sub_e:
                            print(f"  ⚠️ Sub-clustering category '{cat}' failed: {sub_e}. Applying mechanical fallback split...")
                            chunk_size = (len(files) + sub_k - 1) // sub_k
                            for s_idx in range(sub_k):
                                sub_files = files[s_idx * chunk_size : (s_idx + 1) * chunk_size]
                                if sub_files:
                                    final_clusters[f"{cat}_sub_{s_idx:02d}"] = sub_files
                                    
                    # --- POST-PROCESSING SAFETY NET ---
                    for k in list(final_clusters.keys()):
                        if not final_clusters[k]:
                            del final_clusters[k]
                            
                    if K_VAL is not None:
                        while len(final_clusters) > K_VAL:
                            sorted_keys = sorted(final_clusters.keys(), key=lambda k: len(final_clusters[k]))
                            key_from = sorted_keys[0]
                            key_to = sorted_keys[1]
                            final_clusters[key_to].extend(final_clusters[key_from])
                            del final_clusters[key_from]
                            
                        while len(final_clusters) < K_VAL:
                            sorted_keys = sorted(final_clusters.keys(), key=lambda k: len(final_clusters[k]), reverse=True)
                            largest_key = sorted_keys[0]
                            largest_files = final_clusters[largest_key]
                            if len(largest_files) <= 1:
                                break
                            mid = len(largest_files) // 2
                            final_clusters[largest_key] = largest_files[:mid]
                            new_key = f"{largest_key}_split_{len(final_clusters)}"
                            final_clusters[new_key] = largest_files[mid:]

                    # --- CACHE SAVING ---
                    cached_data = {}
                    for cname, files in final_clusters.items():
                        cached_data[cname] = [str(Path(f).relative_to(data_root)) for f in files]
                    with open(cache_path, "w", encoding="utf-8") as fcache:
                        json.dump(cached_data, fcache, indent=2, ensure_ascii=False)
                    print(f"  💾 Saved hierarchical clustering results cache to: {cache_path}")
                    
            agents = []
            for idx, (cluster_name, files) in enumerate(final_clusters.items()):
                if not files:
                    continue
                agents.append(FileAgent(
                    agent_id=f"file_agent_{idx:02d}",
                    cluster_name=cluster_name,
                    file_paths=files,
                ))
            print(f"  ✅ Hierarchical LLM Clustering: {len(agents)} clusters for {len(all_files)} files")
            return agents
            
        except Exception as e:
            print(f"  ❌ Hierarchical LLM Clustering failed: {e}. Falling back to KMeans...")

    # ── Method 1: E5-Large + KMeans semantic clustering ── #
    if use_semantic:
        try:
            import numpy as _np
            from sklearn.cluster import KMeans as _KMeans
            from openai import OpenAI as _OAI

            # KMeans K is configurable via DACODE_KMEANS_K. Default 8 = the best config on the
            # paper-faithful clean 147-file lake (K-sweep: K=8 wins both μ_generation and
            # μ_retrieval F1; the v1 default of 2 was tuned for the buggy 172-file legacy lake).
            # max(1, min(K, len//2)) guards per-subtask lakes with few files: KMeans needs
            # n_clusters>=1 and n_clusters<=n_samples.
            K_CLUSTER = int(os.environ.get("DACODE_KMEANS_K", "8"))
            N_CLUSTERS = max(1, min(K_CLUSTER, len(all_files) // 2))

            # Build file descriptions: filename + first 200 chars
            descriptions = []
            for fpath in all_files:
                fname = Path(fpath).stem
                preview = ""
                try:
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        preview = f.read(200)
                except:
                    pass
                descriptions.append(f"File: {fname}. {preview[:150]}")

            # Embed via API
            embed_key = os.environ.get("EMBED_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
            embed_base = os.environ.get("EMBED_BASE_URL", "https://proxy.onebot.meobeo.ai/v1")
            embed_model = os.environ.get("EMBED_MODEL", "hosted_vllm/intfloat/multilingual-e5-large")

            if embed_key and len(all_files) >= N_CLUSTERS:
                embed_client = _OAI(api_key=embed_key, base_url=embed_base, timeout=60.0, max_retries=1)

                # Batch embed
                all_embeddings = []
                batch_size = 32
                for i in range(0, len(descriptions), batch_size):
                    batch = descriptions[i:i+batch_size]
                    resp = embed_client.embeddings.create(model=embed_model, input=batch)
                    for item in resp.data:
                        all_embeddings.append(item.embedding)

                embeddings = _np.array(all_embeddings)
                print(f"  📊 E5 embeddings: {embeddings.shape}")

                kmeans = _KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings)

                # Group by cluster
                cluster_groups: Dict[int, List[str]] = defaultdict(list)
                for fpath, label in zip(all_files, labels):
                    cluster_groups[label].append(fpath)

                # Auto-name clusters from top file stems
                final_clusters: Dict[str, List[str]] = {}
                for label, files in cluster_groups.items():
                    top = [Path(f).stem[:15] for f in files[:3]]
                    name = '_'.join(top[:2]) if top else f"cluster_{label:02d}"
                    final_clusters[name] = files

                agents = []
                for idx, (cluster_name, files) in enumerate(final_clusters.items()):
                    agents.append(FileAgent(
                        agent_id=f"file_agent_{idx:02d}",
                        cluster_name=cluster_name,
                        file_paths=files,
                    ))
                print(f"  ✅ E5-Large + KMeans: {len(agents)} clusters for {len(all_files)} files")
                return agents
        except Exception as e:
            print(f"  ⚠️ E5 clustering failed: {str(e)[:120]}, falling back to prefix...")

    # ── Fallback: prefix clustering ── #
    clusters: Dict[str, List[str]] = defaultdict(list)
    for fpath in all_files:
        rel = Path(fpath).relative_to(data_root)
        cluster_key = str(rel.parent) if str(rel.parent) != "." else "_root_"
        clusters[cluster_key].append(fpath)

    final_clusters: Dict[str, List[str]] = {}
    for cluster_name, files in clusters.items():
        if len(files) <= max_files_per_cluster:
            final_clusters[cluster_name] = files
        else:
            sub: Dict[str, List[str]] = defaultdict(list)
            for f in files:
                prefix = Path(f).stem[:3].lower()
                sub[prefix].append(f)
            for sub_key, sub_files in sub.items():
                final_clusters[f"{cluster_name}_{sub_key}"] = sub_files

    agents = []
    for idx, (cluster_name, files) in enumerate(final_clusters.items()):
        agents.append(FileAgent(
            agent_id=f"file_agent_{idx:02d}",
            cluster_name=cluster_name,
            file_paths=files,
        ))
    print(f"  📂 Prefix clustering: {len(agents)} clusters for {len(all_files)} files")
    return agents


# ─────────────────────────── Main DS Agent ────────────────────────────────── #

class DSBlackboardAgent:
    """
    Agent điều phối chính (Main Agent) kết hợp Blackboard + DGM Coding Agent.
    """

    def __init__(self, model: str = "hosted_vllm/Qwen/Qwen3.5-35B-A3B-FP8",
                 sandbox_dir: str = "./dgm_agent/lcb_sandbox",
                 max_debug_rounds: int = 3):
        self.model = model
        self.sandbox_dir = sandbox_dir
        self.max_debug_rounds = max_debug_rounds
        self.blackboard = Blackboard()
        os.makedirs(sandbox_dir, exist_ok=True)

        # Khởi tạo LLM client
        from llm import create_client
        self.client, self.model_name = create_client(model)

    def _call_llm(self, user_prompt: str, system_prompt: str = "") -> str:
        """Gọi LLM qua proxy, trả về nội dung text."""
        from llm import get_response_from_llm
        response, _ = get_response_from_llm(
            msg=user_prompt,
            client=self.client,
            model=self.model_name,
            system_message=system_prompt or "You are an expert data scientist. Generate clean, executable Python code.",
            temperature=0.2,
        )
        return response or ""

    def _extract_python_code(self, text: str) -> str:
        match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
        return match.group(1).strip() if match else text.strip()

    def _run_in_sandbox(self, code: str, task_id: str, timeout: int = 30) -> Tuple[bool, str]:
        """Chạy code Python trong subprocess, trả về (success, output/traceback)."""
        script_path = os.path.join(self.sandbox_dir, f"ds_{task_id}.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True, text=True, timeout=timeout,
                encoding="utf-8", errors="replace"
            )
            if result.returncode == 0:
                return True, result.stdout.strip()
            else:
                return False, (result.stderr + "\n" + result.stdout).strip()
        except subprocess.TimeoutExpired:
            return False, f"TimeoutError: Code chạy quá {timeout}s"
        except Exception as e:
            return False, str(e)

    def _build_coding_prompt(self, question: str, context: str) -> str:
        return f"""You are an expert Data Scientist. Solve the following data science question using Python.

QUESTION:
{question}

{context}

INSTRUCTIONS:
- Use the exact file paths shown above to load the data.
- Use pandas, numpy, or other standard data science libraries.
- Handle missing values (NaN) appropriately.
- DO NOT use print() to output results. Instead, define the specific variables requested in the question (e.g., 'avg_salary') and assign the final calculated result to them.
- Return ONLY valid, executable Python code inside ```python ``` blocks.
"""

    def solve(self, task_id: str, question: str, data_lake_dir: str, test_code: str = "") -> Dict[str, Any]:
        """
        Giải một bài toán DA-Code bằng Blackboard + DGM pipeline.
        Returns dict với answer, code, success, traceback.
        """
        print(f"\n{'='*60}")
        print(f"🚀 [DS-Agent] Đang xử lý Task: {task_id}")
        print(f"{'='*60}")
        t_start = time.time()

        # ── Pha 1: Blackboard Discovery ── #
        request = BlackboardRequest(
            task_id=task_id,
            question=question,
            data_lake_dir=data_lake_dir,
        )
        self.blackboard.post_request(request)

        file_agents = build_file_agents(data_lake_dir)
        for agent in file_agents:
            response = agent.handle_request(request)
            self.blackboard.post_response(response)

        context = self.blackboard.get_context_summary()
        print(f"\n📊 [Blackboard] Context tổng hợp ({len(context)} chars)")

        # ── Pha 2: Coding Agent sinh code pandas ── #
        coding_prompt = self._build_coding_prompt(question, context)
        
        print("\n🧠 [Coding Agent] Đang sinh code pandas...")
        raw_response = self._call_llm(coding_prompt)
        code = self._extract_python_code(raw_response)

        # ── Pha 3: Execution + Self-Debug vòng lặp ── #
        success = False
        output = ""
        traceback_info = ""

        for round_idx in range(self.max_debug_rounds):
            print(f"\n⚙️  [Sandbox] Chạy thử (round {round_idx})...")
            
            # Combine generated code with test_code
            combined_code = code + "\n\n# --- TEST CODE ---\n" + test_code
            
            execution_success, output = self._run_in_sandbox(combined_code, f"{task_id}_r{round_idx}")
            
            if execution_success:
                print(f"  ✅ Thực thi thành công (Pass toàn bộ assert)!")
                success = True
                break
            else:
                success = False
                traceback_info = output
                print(f"  ❌ Lỗi Runtime / Assertion (round {round_idx}): {output[:300]}")

            # Nếu fail (runtime hoặc logic) thì debug
            if not success and round_idx < self.max_debug_rounds - 1:
                debug_prompt = f"""The following Python code produced an AssertionError or Runtime Error when tested. Fix it.

ORIGINAL QUESTION:
{question}

{context}

CURRENT CODE:
```python
{code}
```

TEST CODE THAT FAILED:
```python
{test_code}
```

ERROR OR INCORRECT OUTPUT:
{traceback_info[-5000:] if len(traceback_info) > 5000 else traceback_info}

Return ONLY the fixed Python code inside ```python ``` blocks. Do not include the test code in your response, just the main logic."""
                print(f"  🔧 [Self-Debug] Đang sửa lỗi (round {round_idx+1})...")
                raw_fix = self._call_llm(debug_prompt)
                code = self._extract_python_code(raw_fix)

        elapsed = time.time() - t_start
        return {
            "task_id": task_id,
            "question": question,
            "data_lake_dir": data_lake_dir,
            "success": success,
            "code": code,
            "traceback": traceback_info,
            "elapsed_sec": round(elapsed, 2),
            "num_files_discovered": sum(len(r.file_contexts) for r in self.blackboard.responses),
        }


# ─────────────────────────── Public API ───────────────────────────────────── #

def run_ds_pipeline(
    tasks_manifest_path: str,
    model: str = "hosted_vllm/Qwen/Qwen3.5-35B-A3B-FP8",
    output_dir: str = "data/results",
    max_debug_rounds: int = 3,
) -> List[Dict]:
    """
    Chạy toàn bộ DS-Bench pipeline cho một manifest tasks.
    
    Args:
        tasks_manifest_path: Path tới file tasks_manifest.jsonl
                             (được tạo bởi download_dsbench.py).
        model: Model LLM để dùng.
        output_dir: Thư mục lưu kết quả.
        max_debug_rounds: Số vòng Self-Debug tối đa.
    
    Returns:
        Danh sách kết quả cho mỗi bài toán.
    """
    os.makedirs(output_dir, exist_ok=True)
    agent = DSBlackboardAgent(model=model, max_debug_rounds=max_debug_rounds)

    tasks = []
    with open(tasks_manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))

    results = []
    for i, task in enumerate(tasks):
        print(f"\n\n[{i+1}/{len(tasks)}] Processing task: {task['task_id']}")
        test_code = task.get("test_code", "")
        data_lake_dir = task.get("data_lake_dir", task.get("task_dir", ""))
        
        # Đọc nội dung câu hỏi thật từ file txt nếu có (dành cho log DS-Bench cũ)
        q_id = task['task_id'].split('_')[-1] # vd: 'question7'
        q_file_path = os.path.join(data_lake_dir, f"{q_id}.txt")
        actual_question = task["question"]
        if os.path.exists(q_file_path):
            try:
                with open(q_file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        actual_question = content
            except Exception as e:
                print(f"  ⚠️ Lỗi khi đọc câu hỏi thật: {e}")

        result = agent.solve(
            task_id=task["task_id"],
            question=actual_question,
            data_lake_dir=data_lake_dir,
            test_code=test_code,
        )
        results.append(result)

        result_path = os.path.join(output_dir, f"{task['task_id']}.json")
        with open(result_path, "w", encoding="utf-8") as rf:
            json.dump(result, rf, indent=2, ensure_ascii=False)

    # Tổng kết
    passed = sum(1 for r in results if r["success"])
    print(f"\n{'='*60}")
    print(f"📊 DS-BENCH EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total Tasks: {len(results)}")
    print(f"  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {len(results) - passed}")
    print(f"  📈 Pass Rate: {passed/len(results)*100:.1f}%")
    print(f"{'='*60}")

    report_path = os.path.join(output_dir, "ds_report.json")
    with open(report_path, "w", encoding="utf-8") as rep:
        json.dump({"total": len(results), "passed": passed,
                   "pass_rate": passed/len(results) if results else 0,
                   "results": results}, rep, indent=2, ensure_ascii=False)
    print(f"  📁 Report: {report_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run DS-Bench evaluation with Blackboard Agent")
    parser.add_argument("--manifest", type=str, default="data/local_test/tasks_manifest.jsonl",
                        help="Path to tasks manifest file")
    parser.add_argument("--model", type=str, default="hosted_vllm/Qwen/Qwen3.5-35B-A3B-FP8",
                        help="LLM model to use")
    parser.add_argument("--output_dir", type=str, default="data/results",
                        help="Output directory for results")
    parser.add_argument("--max_debug_rounds", type=int, default=3,
                        help="Max Self-Debug rounds")
    args = parser.parse_args()

    run_ds_pipeline(
        tasks_manifest_path=args.manifest,
        model=args.model,
        output_dir=args.output_dir,
        max_debug_rounds=args.max_debug_rounds,
    )
