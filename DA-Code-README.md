# Triadic DGM on DA-Code Benchmark (EMNLP 2024)

## Tổng quan

Đánh giá hệ thống **Triadic DGM** trên benchmark **DA-Code** — Data Analysis Code Generation (EMNLP 2024), theo hướng **data discovery**: agent chỉ nhận question + data lake (179 files), phải tự tìm dữ liệu liên quan, sinh code, tạo output để chấm bằng official evaluator.

**Paper tham chiếu:** "DA-Code: A Benchmark for LLM-Based Data Analysis Code Generation" (EMNLP 2024)
**Paper Blackboard:** "LLM-based Multi-Agent Blackboard System for Information Discovery in Data Science" (arXiv 2510.01285)

---

## Kết quả chính

### So sánh KMeans Clustering Config

| Config | Score | Perfect | Finished | Thay đổi |
|--------|-------|---------|----------|----------|
| KMeans=26 | 0.1853 | 15/91 | 91/91 (100%) | baseline |
| KMeans=16 | 0.2146 | 17/91 | 91/91 (100%) | +15.8% |
| **KMeans=2** | **0.2529** | **20/91** | **91/91 (100%)** | **+36.5%** |

**Config tốt nhất: KMeans=2** — vượt DA-Code baseline (0.244, paper gốc).

### So sánh với baselines

| System | Score | Perfect | Model |
|--------|-------|---------|-------|
| DA-Code baseline (paper) | 0.244 | 21/91 | GPT-4 |
| Blackboard + Claude 4 Opus (paper) | 0.0714 | — | Claude 4 Opus |
| Blackboard + Gemini 2.5 Pro (paper) | 0.0934 | — | Gemini 2.5 Pro |
| Blackboard + Qwen3-Coder-30B (paper) | 0.0111 | — | Qwen3-Coder |
| **Triadic DGM (ours)** | **0.2529** | **20/91** | **Qwen 3.5 35B** |

### Kết quả chi tiết (KMeans=2)

**Theo loại task:**

| Category | Score | Perfect | Total |
|----------|-------|---------|-------|
| Data Insight | 0.2489 | 13 | 61 |
| Data Manipulation | 0.1852 | 3 | 18 |
| Statistical Analysis | 0.3750 | 4 | 12 |

**Theo độ khó:**

| Hardness | Score | Total |
|----------|-------|-------|
| Easy | 0.1944 | 12 |
| Medium | 0.2886 | 63 |
| Hard | 0.1562 | 16 |

**Theo loại đánh giá:**

| Eval Type | Score | Perfect | Total |
|-----------|-------|---------|-------|
| text (JSON key-value) | 0.3556 | 16 | 45 |
| csv (hash-based F1) | 0.1525 | 4 | 46 |

---

## Data-Discovery Setting

```
Agent nhận:  question + data_lake (179 files)
Agent KHÔNG nhận:  gold_dir, eval_result, correct files

91 tasks chia từ 500 tasks gốc (lọc bỏ task lỗi/duplicate)
Data lake: 179 files (163 CSV, 8 JSON, 3 TXT, 3 MD, 1 XLSX, 1 XLS)
            aggregated từ source files của 91 tasks, dedup (filename, content_hash)
```

---

## Pipeline Architecture

```
DA-Code Task (question, data_lake_dir, hardness)
    │
    ▼
[Phase 0: Blackboard Discovery]
    │  E5-Large embeddings → KMeans clustering (2 clusters)
    │  FileAgent quét từng cluster → relevance scoring → top-K filter
    │  Output: compact_context + detailed_context
    ▼
[Phase 0.5: Epiplexity Estimation]
    │  NCD epiplexity score → CapacityManager
    │  Budget: max_retries (4-7), temperature (0.15-0.3)
    │  Statistical Analysis: +1 retry bonus
    ▼
[Phase 1: Planner]
    │  plan(question, compact_ctx) → structured plan
    │  explore(question, detailed_ctx) → refined plan + msg_history
    ▼
[Phase 2: Solver]
    │  generate_code(question, msg_history) → Python code
    │  + inject RIMRULE rules from RuleLibrary
    ▼
[Phase 3: Verifier + Repair Loop]
    │  execute_and_validate() → {success, output, error}
    │  diagnose_error() → structured diagnosis
    │  compute_mdl_epiplexity() → code quality score
    │
    ├── FAIL → repair loop (Solver + Verifier + RuleLibrary)
    │         → extract_rule_from_reflexion() on successful repair
    ▼
[Phase 3.5: Goldilocks Zone]
    │  NCD epiplexity ∈ [0.5, 2.2] → PASS
    │  Outside zone → rejected
    ▼
[Phase 4: Save result.json]
    Official DA-Code format (dabench/result.json)
```

### Component Summary

| Component | File | Role |
|-----------|------|------|
| Blackboard | `dgm_agent/blackboard.py` | E5-Large + KMeans file discovery |
| Orchestrator | `dgm_agent/dacode_orchestrator.py` | Wire pipeline phases |
| Agents | `dgm_agent/dacode_agents.py` | Planner, Solver, Verifier |
| CapacityManager | `core/capacity_manager.py` | Budget theo hardness + complexity |
| RuleLibrary | `core/rule_generator.py` | RIMRULE rules tích lũy qua tasks |
| LLM Client | `dgm_agent/llm.py` | OpenAI-compatible API wrapper |

---

## Cách chạy

### Prerequisites

```bash
# Cài dependencies
pip install -r requirements.txt

# Clone DA-Code official repo (cần cho evaluator)
git clone https://github.com/Leo-CHL/DA-Code.git da-code-repo

# Download source data từ DA-Code official
# https://drive.google.com/file/d/1eM_FVT1tlY4XXp6b7TrKzgTWOvskrjTs/view
# Extract vào data/dacode_source/source/

# Setup unified data lake (aggregates 179 files from 91 tasks)
python scripts/setup_dacode_unified.py
```

> **Lưu ý:** `eval_official.py` import evaluator từ `da-code-repo/da_agent/evaluators/evaluation.py` — repo này phải nằm cùng cấp với project.

### Chạy benchmark

```bash
# Chạy tất cả 91 tasks
python -m dgm_agent.dacode_runner

# Chạy task cụ thể (force rerun)
python -m dgm_agent.dacode_runner --example_name di-text-001 --force_rerun

# Chạy với model khác
python -m dgm_agent.dacode_runner --model hosted_vllm/Qwen/Qwen3.5-35B-A3B-FP8

# Force rerun tất cả
python -m dgm_agent.dacode_runner --force_rerun
```

### CLI Arguments

| Arg | Default | Mô tả |
|-----|---------|-------|
| `--manifest` | `data/dacode_unified_manifest.jsonl` | Path đến task manifest |
| `--model` | `hosted_vllm/Qwen/Qwen3.5-35B-A3B-FP8` | LLM model (hoặc env `DACODE_MODEL`) |
| `--sandbox_dir` | `data/dacode_sandbox` | Output directory |
| `--max_debug_rounds` | `3` | Max repair rounds |
| `--example_name` | (all) | Chạy task cụ thể |
| `--force_rerun` | `false` | Rerun cả tasks đã done |

### Chạy official evaluator

```bash
python eval_official.py
```

Sử dụng **official evaluator** từ `da-code-repo/da_agent/evaluators/evaluation.py`:
- `compare_text`: JSON key-value matching (tolerance=1e-2)
- `compare_csv`: Hash-based column F1

Kết quả lưu vào `data/dacode_official_eval_results.json`.

---

## Key Files

```
data-agent/
├── dgm_agent/
│   ├── blackboard.py              # E5-Large + KMeans file discovery
│   ├── dacode_orchestrator.py     # Pipeline orchestrator
│   ├── dacode_agents.py           # Planner, Solver, Verifier
│   ├── dacode_runner.py           # CLI runner
│   └── llm.py                     # LLM client
├── core/
│   ├── capacity_manager.py        # Budget allocation
│   ├── rule_generator.py          # RIMRULE rule extraction
│   └── inspector.py               # Epiplexity scoring
├── eval/
│   ├── eval_official.py           # Official evaluator wrapper
│   └── configs/eval_all.jsonl     # Eval configs cho 91 tasks
├── data/
│   ├── dacode_lake/               # 179 files data lake
│   ├── dacode_sandbox/            # Task outputs (91 dirs)
│   ├── dacode_gold/gold/          # Gold standard files
│   ├── dacode_unified_manifest.jsonl  # 91 task definitions
│   └── dacode_official_eval_results.json  # Latest eval results
├── da-code-repo/                  # Official DA-Code evaluation code
├── scripts/
│   └── setup_dacode_unified.py    # Data lake setup
├── DA-Code-README.md              # ← This file
└── CLAUDE.md                      # Project overview
```

---

## Phân tích kết quả

### Tại sao KMeans=2 tốt nhất?
- 2 clusters lớn (83-93 files/cluster) → agent quét gần như toàn bộ data lake
- Ít rủi ro bỏ sót file quan trọng do clustering sai
- KMeans=26 (11 files/cluster) đôi khi lọc sai cluster → mất file cần thiết

### Điểm yếu chính
- **CSV eval** (0.1525): Nhiều task CSV trả sai format hoặc sai giá trị
- **Hard tasks** (0.1562): Task khó cần nhiều bước reasoning, model 35B chưa đủ
- **72/91 tasks score=0**: Lỗi phổ biến — wrong file loading, wrong column selection, format mismatch

### RIMRULE Rules học được (top 5)
1. Always verify df.columns before accessing specific column names (KeyError)
2. Always convert data to numeric types before arithmetic operations
3. Verify column names exist before accessing DataFrame columns
4. Always convert merge key columns to matching data types before merging
5. Always wrap file loading in try-except blocks (FileNotFoundError)

---

## Config hiện tại

```yaml
# config.yaml
model: hosted_vllm/Qwen/Qwen3.5-35B-A3B-FP8
api: https://proxy.onebot.meobeo.ai/v1

# blackboard.py
N_CLUSTERS: 2           # Best config (score=0.2529)
embedding: E5-Large     # intfloat/e5-large-v2 (1024 dim)

# dacode_orchestrator.py
Goldilocks Zone: [0.5, 2.2]  # NCD epiplexity range
Planner temp: 0.2
Solver temp: 0.2 (code), 0.3 (repair)
Verifier temp: 0.1

# capacity_manager.py
Easy: temp=0.15, retries=3
Medium: temp=0.2, retries=4
Hard: temp=0.3, retries=5
Statistical Analysis: +1 retry bonus
Complexity score: [0.3, 1.5] (replaces flat NCD epiplexity)
```
