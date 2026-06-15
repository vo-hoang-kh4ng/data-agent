# CLAUDE.md — Data Agent Project (Triadic DGM + LAMBDA)

## Tổng quan dự án
Hệ thống **Triadic DGM** — khung tiến hóa mở cho agent sinh mã đa ngôn ngữ, xây dựng trên nền LAMBDA (Large Model Based Data Agent).

## Paper tham chiếu: LAMBDA (arxiv 2407.17535)
Paper eval trên **7 nhóm benchmark** + DGM eval trên **Polyglot Benchmark**.

### Trạng thái đánh giá hoàn chỉnh (2026-05-30)

| Nhóm | Dataset | Status | Kết quả chính |
|------|---------|--------|---------------|
| Group 1: Classification | AIDS, NHANES, Breast Cancer, Wine | ✅ | Tương đương GPT-4 |
| Group 2: Regression | Concrete, Power Plant, Abalone, Airfoil | ✅ (bug) | MSE cao do chưa chuẩn hóa |
| Group 3: Genomic | TCGAmirna, EMTAB386, GSE49997 | ❌ | Context window exceeded |
| Group 4: Missing Data | Framingham, Student, Heart Disease | ✅ | Vượt xa GPT-4 (+20-44%) |
| Group 5: MNIST | CNN, Transformer | ✅ | CNN: 99.11%, Transformer: 11.28% |
| Group 6: SMS Spam | Naive Bayes, BERT | ⚠️ | NB: 95.89%, BERT: skipped |
| Group 7: Knowledge | PAMI, NCM, FPNENN | ✅ | 3/3 đạt điểm tối đa |
| **Polyglot DGM** | 60 tasks, 6 ngôn ngữ | ✅ | **Pass@1: 46.7%** (SOTA) |

## Files quan trọng
- `report.md` — Báo cáo tổng hợp toàn bộ
- `BENCHMARK_RESULTS.md` — Kết quả chi tiết LAMBDA benchmarks
- `RESULTS.md` — Kết quả Polyglot DGM
- `proposal.md` — Đề xuất nghiên cứu Triadic DGM
- `benchmarks/checkpoints/benchmark_state.json` — Checkpoint kết quả

## Cấu hình
- **LLM**: Qwen 3.5 35B via proxy.onebot.meobeo.ai
- **Config**: config.yaml

## Lệnh chạy benchmarks
```bash
python -m benchmarks.run_benchmarks                                    # Tất cả
python -m benchmarks.run_benchmarks --group group4_missing --retry-failed  # Nhóm cụ thể
python -m benchmarks.run_benchmarks --report-only                      # Chỉ generate report
```

## DA-Code Benchmark (EMNLP 2024)
- **Tasks**: 91 retained IDs (Data Insight: 61, Data Manipulation: 18, Statistical Analysis: 12)
- **Setting**: Data Discovery (question + unified data lake, no pre-known correct files). **v2 paper-faithful lake = 147 files** (145 csv + 1 xls + 1 xlsx) — union of all 91 tasks' source inputs, with hint/guidance/gold-template files excluded (answer-leak prevention). v1 used a buggy 172-file lake (Windows case-clobber + leaked answers); see v2 changelog below.
- **LLM**: Qwen 3.5 35B-A3B (proxy.onebot.meobeo.ai), **thinking-ON, max_output_tokens=8192** (the Blackboard paper's reasoning-on protocol; v1's 4096 cap truncated the think block → 0% finished)
- **Pipeline**: Triadic DGM (Blackboard → Planner → Solver → Verifier + RIMRULE + Goldilocks)
- **Runner**: `python -m dgm_agent.dacode_runner`
- **Eval**: `python eval_official.py` (uses `da-code-repo/da_agent/evaluators/metrics/`)
- **Docs**: `DA-Code-README.md` — chi tiết đầy đủ

### Kết quả Triadic DGM trên DA-Code (v2, paper-faithful)

Setup: **91 retained tasks · clean 147-file unified lake · real discovery · official DA-Code evaluator (`eval_official.py`) · Qwen3.5-35B-A3B · thinking-ON + 8192 tokens.**

**KMeans sweep (thinking-OFF — controlled K comparison), clean guarded lake:**

| K (KMeans) | clusters kept | μ_generation | perfect | μ_retrieval F1 | recall | halluc |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 0.1626 | 12/91 | 0.483 | 0.455 | 80% |
| 4 | 4 | 0.1459 | 12/91 | 0.482 | 0.458 | 76% |
| **8** | **5** | **0.2008** | **18/91** | **0.490** | **0.473** | 74% |
| 16 | 5 | 0.2060 | 17/91 | 0.416 | 0.403 | 64% |

→ **K=8 wins both metrics** (best generation AND best discovery F1). Generation saturates at K≥8; discovery F1 peaks at K=8 then drops at K=16. The relevance filter caps effective clusters at ~5 for K≥8. **v2 default = K=8** (env `DACODE_KMEANS_K`).

**K=8, thinking-ON (paper's reasoning-on protocol — v2 default):**

| Metric | v2 result | Blackboard (paper, discovery/91 tasks) |
|---|---:|---:|
| μ_generation | **0.1912** (15/91 perfect, 90/91 finished) | 0.0934 (Gemini-2.5-Pro) / 0.0714 (Opus-4) / 0.0111 (Qwen3-Coder) |
| μ_retrieval F1 (macro) | **0.599** (recall 0.566, precision 0.753) | 0.64 (recall 0.60, precision 0.84) |

→ **2.05× Blackboard+Gemini-2.5-Pro and 17.2× Blackboard+Qwen3-Coder** on generation; **94% of Blackboard's discovery F1**. Turning thinking ON (4096→8192) lifted discovery F1 0.49→0.60 with generation flat within noise (0.20→0.19).

> ⚠️ **Comparison caveats:** (a) **backbone** — Triadic DGM's Qwen3.5-35B-A3B is not one of the paper's backbones, so only the **Qwen3-Coder** comparison (17.2×) is strictly backbone-controlled; the Gemini multiple is not. (b) **single-sample** — all runs single-sample (LLM noise ~±0.02); K=8's 18-perfect vs K2/K4's 12 under thinking-OFF is the strongest signal. (c) **discovery F1 is a reproduction** (`eval_discovery_f1.py`) — the paper published no official retrieval script.
>
> ❌ **The old "0.2529 / 2.7× Gemini" headline (v1) is INVALIDATED.** It came from the **buggy 172-file legacy lake** — a Windows case-clobber bug (`tables.csv` vs `Tables.csv` merged into one path → 9 tasks read wrong data) plus 7 leaked answer/hint files sitting in the lake. It is not comparable to the paper protocol and is **superseded** by the controlled 147-file results above.

### v2 changelog (bug fixes vs v1's 0.2529)
1. **Clean 147-file lake** replacing the buggy 172-file lake: 0 case-collisions (task-ID-prefix disambiguation), 0 leaked answers (hint/gold-template exclusion), 6-dimension integrity-verified (completeness / no-leak / no-collision / hash / gold-map / run-stability).
2. **Discovery→solver interface fix**: solver now gets the real lake file paths + "read them yourself with `pd.read_csv`" instruction (v1 told it "data already loaded" but no real names existed in global mode → hallucinated filenames + printed dummy data); compact context now lists ALL lake files (was capped at 15/cluster → gold file truncated out of view).
3. **Lake-write-pollution fix**: agent no longer writes outputs into the lake dir (147→152 mid-run in v1); write-path instruction + runtime `scrub_lake()` guard → lake stays 147 end-to-end (guard never fired on the clean run).
4. **Anti-fabrication guard**: verifier rejects dummy/placeholder output as a failure so the repair loop fixes the path instead of silently scoring 0.
5. **Thinking-ON + 8192 tokens** (paper's reasoning-on protocol); v1's 4096 cap truncated the hybrid-reasoning think block → 0% finished.

## Lỗi đã biết
1. Group 3 Genomic: Context window 64K tokens quá nhỏ (cần 161K-1.7M)
2. Group 2 Regression: MSE cao gấp 100-200x paper (chưa chuẩn hóa data)
3. MNIST Transformer: Code sinh ra sai (11.28% ≈ random)
4. SMS Spam BERT: Dataset không download được
5. Polyglot C++: Thiếu boost libraries trong Docker
6. DA-Code (v2): 76/91 tasks chưa perfect (score<1) — nguyên nhân chính: sai lựa chọn file giữa các file cùng tên đã được disambiguate bằng task-prefix, sai cột/format ở CSV-hash eval, và một số task khó cần multi-step reasoning vượt khả năng backbone 35B. (v1 ghi "72/91 score=0 do wrong file loading" — đã được fix ở v2: discovery interface + clean lake.)
