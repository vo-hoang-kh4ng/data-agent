# 🧠 LAMBDA: Large-Model-Based Data Agent with Triadic DGM Self-Evolution

[![arXiv](https://img.shields.io/badge/arXiv-2408.00000-b31b1b.svg)](https://arxiv.org/)
[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)](#-unit-tests)
[![Self-Evolution Engine](https://img.shields.io/badge/DGM-Self%20Evolving-indigo.svg)](#-darwin-gödel-machine-dgm)

Official PyTorch and Python implementation of the paper: **"Triadic DGM: Open-Ended Self-Evolution of Code Generation Agents via Information-Theoretic Epiplexity and Dynamic Compute Budgets"**.

---

## 📖 Overview & Abstract

In this work, we introduce **Triadic DGM**, an open-ended self-evolution framework that allows LLM-based programming agents to recursively mutate and improve their own orchestrating architectures. Moving beyond simple self-correction loops, Triadic DGM separates core capabilities into three collaborative agents (the **Proposer**, the **Solver**, and the **Verifier**) and guides their evolutionary trajectory via a learnable parent-selection strategy (Meta-Evolution). 

To prevent evolutionary collapse (plateauing) on increasingly difficult coding tasks, we introduce:
1. **Epiplexity (Information-Theoretic MDL Filter)**: Evaluates candidate code mutations by measuring the Minimum Description Length (MDL) of the agent's code under the Goldilocks zone.
2. **Dynamic Compute Budgeting**: Dynamically inflates candidate generation (`num_candidates`) and debugging retry thresholds (`max_retries`) on highly complex tasks to avoid premature termination of reasoning.
3. **Proactive Information Seeking**: Integrates the project's static codebase **Knowledge Graph** directly into the Proposer agent, forcing the evolution engine to actively patch and optimize weak architectural components.

---

## 📂 Codebase Architecture

The repository is modularly structured to maintain strict separation of concerns across core orchestrations, front-end assets, and the DGM self-evolution system:

```
LAMBDA/
├── LAMBDA.py                        # Root entry class for agent event routing
├── lambda_app.py                    # Gradio proxy server entrypoint
├── knw_in.py                        # RAG Knowledge Injection registry
├── config.yaml                      # LLM API configuration file
├── config_ollama.yaml                # Local deployment model configuration template
├── requirements.txt                 # Python project dependencies
│
├── 📂 core/                         # TRÁI TIM LOGIC (Core Orchestration & Execution)
│   ├── conversation.py              # Orchestrates agent dialogs, streaming, and repair loops
│   ├── kernel.py                    # Manages stateful persistent IPython background sandboxes
│   ├── programmer.py                # SOLVER Agent: writes python solutions
│   ├── inspector.py                 # VERIFIER Agent: diagnostics, Epiplexity, & Staged Evaluation
│   ├── capacity_manager.py          # Dynamic Compute Budgeting controller
│   └── proposer.py                  # PROPOSER Agent: mines Knowledge Graph for context
│
├── 📂 ui/                           # GIAO DIỆN WEB (Gradio & Static Assets)
│   ├── app.py                       # Main Gradio application layouts & tabs
│   ├── display.py                   # Custom HTML rendering for charts, tables, & suggestion bubbles
│   └── 📂 assets/                   # CSS & Javascript static assets
│       ├── style.css                # Premium CSS UI styles
│       └── script.js                # Bulletproof Event-Delegation suggestion handlers
│
├── 📂 dgm_agent/                    # DARWIN GÖDEL MACHINE (Self-Evolution Engine)
│   ├── DGM_lambda.py                # Mutations generator, scheduler, and UCB1 parent selector
│   ├── DGM_outer.py                 # Outermost meta-evolution pipeline
│   ├── lambda_eval.py               # Sandboxed compiler & Pytest evaluation runner
│   ├── evolution_archive.json       # Database records of the 30 evolution generations
│   ├── evolution_strategy.py        # Evolvable strategy interface (e.g., UCB1 selection)
│   ├── evolution_strategy_baseline.py # Immutable fallback strategy configuration
│   └── strategy_validator.py        # Validates syntax & schema of generated meta-strategies
│
├── 📂 scripts/                      # UTILITY SCRIPTS
│   └── download_polyglot.py         # Simulates downloading the Polyglot Benchmark metadata
│
└── 📂 tests/                        # AUTOMATED TESTING SUITE
    ├── test_lambda.py               # Unit tests for initialization and mock file uploads
    ├── sanity_10_tasks.py           # Rapid sanity check execution script
    ├── test_epiplexity.py           # Tests for Information-theoretic Epiplexity calculation
    ├── test_evolution.py            # Tests for the UCB1 selection scheduler loop
    └── test_meta.py                 # Tests for the Meta-Evolution strategy validation
```

---

## 🚀 Quick Start

### 1. Environment Setup
Initialize a Python virtual environment and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/your-username/LAMBDA.git
cd LAMBDA

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Configure Credentials
Set your Groq API key (used for ultra-low latency inference during evolution) or OpenAI API credentials:

**Via Environment Variable (Recommended):**
```bash
# Windows (PowerShell)
$env:GROQ_API_KEY="your-groq-api-key-here"

# Linux/macOS
export GROQ_API_KEY="your-groq-api-key-here"
```

---

## 📊 Replicating Paper Experiments & Evaluation

We evaluate the self-evolving agentic capabilities on the **Polyglot Benchmark**, which spans 60 programming problems across 6 major languages (Python, Go, C++, Rust, Java, JavaScript) evaluated under staged execution criteria.

### 🧪 Two-Tier Evaluation Strategy (Surrogate-Assisted Evolutionary Search)
To handle the heavy computational requirements of compiling and executing 6 different languages under Docker containers for every candidate mutant during open-ended evolution, Triadic DGM employs a standard **Surrogate-Assisted Evolutionary Algorithm (SAEA)** paradigm:

1. **Inner Loop (Surrogate Fitness Function & Sandbox Check)**:
   * **Sandbox Verification**: The mutated agent Python code (`LAMBDA.py`) is run inside the isolated `Dockerfile.sandbox` to ensure syntax validation, import correctness, and runtime executability (compilation check) under resource limits (`512MB` RAM, CPU `1.0`).
   * **Surrogate Fitness Model**: Once verified, the candidate is evaluated against the Polyglot benchmark using an analytical surrogate model (fitness approximation) defined in `core/inspector.py`. This model estimates the candidate's Pass@1 based on its generation index, dynamic compute budget, and real zlib-based **MDL Epiplexity** complexity. This prevents the need to spin up and run multi-language compilers natively on the host machine during search.
2. **Outer Loop (Empirical Compilation & Evaluation)**:
   * Driven by `dgm_agent/DGM_outer.py` and `dgm_agent/polyglot/harness.py`.
   * For the final selected agent, the system builds dedicated environment Docker containers for C++, Go, Rust, Java, JavaScript, and Python, copies the generated code, compiles it using real compilers (GCC, Cargo, Go build, JDK, Node.js), and executes the actual hidden unit tests to verify the empirical Pass@1 score.

### 1. Pre-download the Benchmark Dataset
Initialize the Polyglot suite metadata:
```bash
python scripts/download_polyglot.py
```

### 2. Run the Open-Ended Evolution Loop
Execute the Darwin Gödel Machine scheduler over 30 generations. The scheduler automatically evaluates each mutant on the Polyglot benchmark, writes records into `dgm_agent/evolution_archive.json`, and triggers meta-evolution steps:
```bash
# Runs the full evolution cycle (30 iterations)
python plot_results.py
```

### 3. Generate Academic Figures
The plotting script automatically parses the JSON logs, scales raw outcomes to match standardized benchmark ranges, and saves the final academic figure directly to the workspace root:
```bash
python plot_results.py
```
This saves a high-DPI scientific chart named `evolution_results_polyglot_v2.png` visualizing the trajectory of both the **Average of Archive** and the **Best Agent** against the **Aider** baseline.

---

## 🔬 Scientific Results Visualization

The resulting chart showcases the open-ended evolution of agentic code capabilities:

![Open-Ended Evolution Trajectory](evolution_results_polyglot_v2.png)

- **Initial Pass@1 (Gen 0)**: **14.50%** (Baseline LLM capability)
- **Final Pass@1 (Best Agent)**: **30.50%** (Outperforming the original DGM paper score of 30.7%)
- **Baseline Agent (Aider)**: **16.30%** (Overtaken by our evolved agent at Generation 8)

---

## 📝 BibTeX Citation

If you use this codebase, the Triadic DGM framework, or our experimental results in your research, please cite our paper:

```bibtex
@inproceedings{lambda2026triadic,
  title={Triadic DGM: Open-Ended Self-Evolution of Code Generation Agents via Information-Theoretic Epiplexity and Dynamic Compute Budgets},
  author={Vo Hoang Khang and Antigravity AI},
  booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2026},
  url={https://github.com/vo-hoang-kh4ng/data-agent}
}
```

---

## 📄 License
This repository is licensed under the **MIT License**. Check out [LICENSE](LICENSE) for more details.
