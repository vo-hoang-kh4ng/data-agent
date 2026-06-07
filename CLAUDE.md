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
- **Setting**: Data Discovery (question + 179-file data lake, no pre-known correct files)
- **LLM**: Qwen 3.5 35B (proxy.onebot.meobeo.ai)
- **Pipeline**: Triadic DGM (Blackboard → Planner → Solver → Verifier + RIMRULE + Goldilocks)
- **Runner**: `python -m dgm_agent.dacode_runner`
- **Eval**: `python eval_official.py` (uses `da-code-repo/da_agent/evaluators/metrics/`)
- **Docs**: `DA-Code-README.md` — chi tiết đầy đủ

### Kết quả Triadic DGM trên DA-Code

| Version | Score | Perfect | Finished | Notes |
|---------|-------|---------|----------|-------|
| **Triadic DGM (KMeans=2)** | **0.2529** | **20/91** | **91/91 (100%)** | Best config, surpasses DA-Code baseline |
| Triadic DGM (KMeans=16) | 0.2146 | 17/91 | 91/91 (100%) | +15.8% vs KMeans=26 |
| Triadic DGM (KMeans=26) | 0.1853 | 15/91 | 91/91 (100%) | E5 + KMeans baseline |
| v4 Multi-Step | 0.1670 | 12/91 | 91/91 (100%) | Plan→Explore→Code pipeline |
| Blackboard reproduction | 0.2246 | 18/91 | 88/91 | Paper reproduction |
| DA-Code baseline (paper) | 0.244 | 21/91 | 87/91 | GPT-4, original paper |

### Theo loại task (KMeans=2)
| Type | Score | Perfect |
|------|-------|---------|
| Data Insight | 0.2489 | 13 |
| Data Manipulation | 0.1852 | 3 |
| Statistical Analysis | 0.3750 | 4 |

## Lỗi đã biết
1. Group 3 Genomic: Context window 64K tokens quá nhỏ (cần 161K-1.7M)
2. Group 2 Regression: MSE cao gấp 100-200x paper (chưa chuẩn hóa data)
3. MNIST Transformer: Code sinh ra sai (11.28% ≈ random)
4. SMS Spam BERT: Dataset không download được
5. Polyglot C++: Thiếu boost libraries trong Docker
6. DA-Code: 72/91 tasks score=0 do wrong file loading / wrong column selection
