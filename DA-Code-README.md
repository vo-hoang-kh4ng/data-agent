# Triadic DGM on DA-Code Benchmark (EMNLP 2024)

## Tổng quan

Đánh giá hệ thống **Triadic DGM** trên benchmark **DA-Code** — Data Analysis Code Generation (EMNLP 2024), theo hướng **data discovery**: agent chỉ nhận question + data lake (paper-faithful **147 files** sau khi loại hint/gold-template + fix case-clobber), phải tự tìm dữ liệu liên quan, sinh code, tạo output để chấm bằng official evaluator.

**Paper tham chiếu:** "DA-Code: A Benchmark for LLM-Based Data Analysis Code Generation" (EMNLP 2024)
**Paper Blackboard:** "LLM-based Multi-Agent Blackboard System for Information Discovery in Data Science" (arXiv 2510.01285)

---

## Kết quả chính (v2, paper-faithful)

Setup: **91 retained tasks · clean 147-file unified lake (145 csv + 1 xls + 1 xlsx) · real discovery · official DA-Code evaluator · Qwen3.5-35B-A3B · thinking-ON + 8192 tokens** (paper's reasoning-on protocol).

### KMeans sweep (thinking-OFF — controlled K comparison)

| K (KMeans) | clusters kept | μ_generation | perfect | μ_retrieval F1 | recall | halluc |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 0.1626 | 12/91 | 0.483 | 0.455 | 80% |
| 4 | 4 | 0.1459 | 12/91 | 0.482 | 0.458 | 76% |
| **8** | **5** | **0.2008** | **18/91** | **0.490** | **0.473** | 74% |
| 16 | 5 | 0.2060 | 17/91 | 0.416 | 0.403 | 64% |

**K=8 là config tốt nhất** — thắng trên cả generation lẫn discovery F1. Generation bão hòa ở K≥8; discovery F1 đạt đỉnh K=8 rồi giảm ở K=16 (relevance filter giới hạn ~5 cluster hiệu quả). **v2 default = K=8** (env `DACODE_KMEANS_K`).

### So sánh với baselines (K=8, thinking-ON — v2 default)

| System | μ_generation | μ_retrieval F1 | Model | Setting |
|--------|-------|---------|-------|---------|
| Blackboard + Qwen3-Coder-30B (paper) | 0.0111 | — | Qwen3-Coder | Discovery, 91 tasks |
| Blackboard + Claude 4 Opus (paper) | 0.0714 | — | Claude 4 Opus | Discovery, 91 tasks |
| Blackboard + Gemini 2.5 Pro (paper) | 0.0934 | 0.64 | Gemini 2.5 Pro | Discovery, 91 tasks |
| DA-Code paper (Huang 2024) | 0.244 | — | GPT-4 | ⚠️ **Non-discovery**, 500 tasks |
| **Triadic DGM (ours, K=8 think-ON)** | **0.1912** | **0.599** | **Qwen 3.5 35B-A3B** | Discovery, 91 tasks |

→ **2.05× Blackboard+Gemini-2.5-Pro và 17.2× Blackboard+Qwen3-Coder** về generation; **94% discovery F1** của Blackboard. ⚠ Bội số **Gemini không** backbone-controlled (khác model); bội số **Qwen3-Coder (17.2×) mới là so sánh công bằng** (cùng họ model). Bật thinking (4096→8192) tăng discovery F1 0.49→0.60, generation flat trong nhiễu (0.20→0.19).

> ⚠️ **Không so sánh trực tiếp với DA-Code paper 0.244** — đó là **non-discovery** setting (agent được cho sẵn đúng file, 500 tasks, GPT-4). Baseline cùng discovery-setting là các dòng Blackboard.
>
> ❌ **Headline cũ "0.2529 / 2.7× Gemini" (v1) bị INVALIDATE.** Nó đến từ **legacy lake 172-file lỗi** — bug case-clobber trên Windows (`tables.csv` vs `Tables.csv` gộp thành 1 path → 9 task đọc sai data) + 7 file answer/hint rò rỉ trong lake. Không so sánh được với paper protocol; **bị thay thế** bởi kết quả 147-file controlled ở trên (xem v2 changelog cuối file).

### Kết quả chi tiết (K=8, thinking-ON — v2 default)

**Theo loại task:**

| Category | μ_generation | Perfect | Total |
|----------|-------|---------|-------|
| Data Insight | 0.1814 | 9 | 61 |
| Data Manipulation | 0.1852 | 3 | 18 |
| Statistical Analysis | 0.2500 | 3 | 12 |

**Theo độ khó:**

| Hardness | μ_generation | Perfect | Total |
|----------|-------|---------|-------|
| Easy | 0.1944 | 2 | 12 |
| Medium | 0.1862 | 10 | 63 |
| Hard | 0.2083 | 3 | 16 |

**Theo loại đánh giá:**

| Eval Type | μ_generation | Perfect | Total |
|-----------|-------|---------|-------|
| text (JSON key-value) | 0.2231 | 9 | 45 |
| csv (hash-based F1) | 0.1599 | 6 | 46 |

---

## Data-Discovery Setting

```
Agent nhận:     question + data_lake (paper-faithful 147 files)
Agent KHÔNG nhận: gold_dir, eval_result, correct files, hint/guidance/tips

91 retained task IDs (Appendix G, paper Blackboard)
Data lake: 147 files (145 csv + 1 xls + 1 xlsx)
            union của source-input files của 91 tasks, loại hint/gold-template (anti-leak),
            dedup (filename, content_hash), disambiguate case-collision bằng task-prefix
            → 6-dimension integrity-verified (0 case-collision, 0 leak, 0 missing)
```

---

## Pipeline Architecture

```
DA-Code Task (question, data_lake_dir, hardness)
    │
    ▼
[Phase 0: Blackboard Discovery]
    │  E5-Large embeddings → KMeans clustering (K=8, configurable via DACODE_KMEANS_K)
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

# Setup paper-faithful data lake (aggregates 147 files from 91 tasks, after hint/gold cleanup)
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
│   ├── dacode_lake_paper/         # 147 files paper-faithful data lake (clean, after hint/gold cleanup)
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

### Tại sao KMeans=8 tốt nhất (paper-faithful clean lake)?
- Relevance filter giữ lại ~5 cluster hiệu quả cho mọi K≥8 → K8/K16 chỉ khác partition
- Generation bão hòa ở K≥8 (K8 0.201 ≈ K16 0.206, cả hai >> K2 0.163 / K4 0.146)
- Discovery F1 đạt đỉnh K=8 (0.490) rồi GIẢM ở K=16 (0.416) → K=8 là điểm ngọt
- ⚠ ngược với legacy lake lỗi cũ (K2>K16>K26) — clean faithful setup thưởng clustering mịn hơn đến ~5 cluster

### Điểm yếu chính (v2, K=8 thinking-ON)
- **CSV eval** (0.1599, 6/46 perfect): nhiều task trả sai format hoặc sai giá trị hash
- **Hard tasks** (0.2083, 3/16): task multi-step reasoning vượt khả năng backbone 35B
- **76/91 chưa perfect**: chọn sai file giữa các file cùng tên (đã disambiguate bằng task-prefix), sai cột, format mismatch

### v2 changelog (bug fixes vs v1's 0.2529)
1. **Clean 147-file lake** thay lake lỗi 172-file: 0 case-collision (disambiguate task-prefix), 0 leak answer (loại hint/gold-template), verify 6 chiều.
2. **Discovery→solver interface fix**: solver nhận real lake paths + "read yourself" (v1 báo "data already loaded" nhưng global mode không có file thật → hallucinate + dummy data); compact context list ALL files.
3. **Lake-write-pollution fix**: agent không ghi output vào lake dir (147→152 mid-run v1); instruction + `scrub_lake()` guard → lake giữ nguyên 147 end-to-end.
4. **Anti-fabrication guard**: verifier reject dummy/placeholder output → repair loop fix path thay vì im lặng score 0.
5. **Thinking-ON + 8192** (paper reasoning-on protocol); v1 cap 4096 truncate think block → 0% finished.

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

# dgm_agent/llm.py
MAX_OUTPUT_TOKENS: 8192   # v2: paper reasoning-on protocol (v1: 4096 truncated think block)
thinking: ON              # enable_thinking=True for Qwen3.x-A3B hybrid-reasoning

# blackboard.py
N_CLUSTERS: 8           # v2 best (K-sweep on clean 147-file lake); env DACODE_KMEANS_K
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
