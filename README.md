# 🧠 LAMBDA: Large-Model-Based Data Agent

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)](#-unit-tests)
[![Self-Evolution Engine](https://img.shields.io/badge/DGM-Self%20Evolving-indigo.svg)](#-darwin-gödel-machine-dgm)

**LAMBDA** (Large-Model-Based Data Agent) is a state-of-the-art agentic data analysis and self-evolving AI platform. Powered by an advanced dual-agent orchestration layer (`Programmer` + `Inspector`), a stateful Jupyter execution sandbox, and the revolutionary **Darwin Gödel Machine (DGM)**, LAMBDA makes complex data exploration, visualization, and modeling as simple as having a conversation.

---

## ✨ Key Features

* **💬 Chat-Driven Data Science:** Simply ask in natural language to load datasets, generate plots, perform correlations, or build models.
* **🖥️ Stateful Jupyter Kernel Sandbox:** Executes python cells in a persistent, secure background IPython kernel. Session state, variables, and imported libraries persist across the entire chat conversation.
* **🔄 Autonomous Code Correction:** If code execution fails due to runtime errors, the system automatically triggers a self-correction loop where the `Inspector` agent diagnoses the traceback and the `Programmer` agent rewrite and repairs the code on-the-fly.
* **🧬 Darwin Gödel Machine (DGM) Self-Evolution:** A premium self-evolution engine. The DGM reads the codebase's static dependency **Knowledge Graph** (scanned via `Understand-Anything`), performs LLM-driven architectural mutations, runs sandboxed unit tests, and commits successfully evolved code, allowing the agent to recursively self-improve.
* **🎨 Premium Aesthetic UI:** Crafted with Outfit & Plus Jakarta Sans Google Fonts, modern radial gradients, drop shadows, responsive card-lifting animations, and bulletproof event-delegated suggestion buttons.

---

## 📂 Codebase Architecture

LAMBDA has been reorganized into a professional, modular, and maintainable architecture separating core engine logic from front-end elements:

```
DGM_Dagent/
├── LAMBDA.py                    # Root entry class for file & event dispatching
├── lambda_app.py                # Proxy wrapper redirecting to ui.app
├── knw_in.py                    # RAG Knowledge Injection registry
├── config.yaml                  # Main YAML configuration file
├── config_ollama.yaml            # Config template for local Ollama deployments
├── requirements.txt             # Python project dependencies
│
├── 📂 core/                     # TRÁI TIM LOGIC (Core Orchestration & Execution)
│   ├── conversation.py          # Orchestrates agent dialogs, streaming, and repair loops
│   ├── kernel.py                # Manages the stateful IPython kernel wrapper
│   ├── programmer.py            # Agent responsible for writing python code
│   └── inspector.py             # Agent responsible for error trace diagnostics
│
├── 📂 ui/                       # GIAO DIỆN WEB (Gradio & Static Assets)
│   ├── app.py                   # Main Gradio application layouts & tabs
│   ├── display.py               # Custom HTML wrappers for charts, tables, and suggestions
│   └── 📂 assets/               # CSS & Javascript Static Files
│       ├── style.css            # Premium CSS UI customizations
│       └── script.js            # Bulletproof Javascript Event Delegation clicks
│
├── 📂 dgm_agent/                # DARWIN GÖDEL MACHINE (Self-Evolution Suite)
│   ├── DGM_lambda.py            # Mutations generator and evolution scheduler
│   └── lambda_eval.py           # Simulated sandbox evaluator running pytest
│
└── 📂 tests/                    # UNIT TESTS
    └── test_lambda.py           # Unit tests for initialization and mock file uploads
```

---

## 🚀 Quick Start

### 1. Prerequisites & Virtual Env Setup
Ensure you have Python 3.10+ installed. Clone the repository and initialize a virtual environment:

```bash
# Clone the repository
git clone https://github.com/your-username/LAMBDA.git
cd LAMBDA

# Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Your LLM Credentials
To run using Groq API (extremely fast Llama-3.1-8b) or OpenAI, set your API key as an environment variable or configure it inside `config.yaml`:

**Via Environment Variable (Recommended):**
```bash
# Windows (PowerShell)
$env:GROQ_API_KEY="your-groq-api-key-here"

# Linux/macOS
export GROQ_API_KEY="your-groq-api-key-here"
```

**Via Config File (`config.yaml`):**
Open `config.yaml` and set the `api_key` field:
```yaml
api_key : "your-groq-api-key-here"
```

### 3. Run the App
Launch the Gradio server:
```bash
python lambda_app.py
```
Open your browser and navigate to `http://localhost:8000`.

---

## 🧬 Darwin Gödel Machine (DGM)

The **Darwin Gödel Machine** tab inside the web UI allows the agent to trigger its own self-evolution loop:

1. **KG Scanning:** The system reads `knowledge-graph.json` to understand its own components.
2. **Mutation:** The Groq LLM generates an architectural improvement mutation for `LAMBDA.py`.
3. **Sandbox Testing:** The mutation is written into a sandbox environment (`dgm_agent/sandbox_lambda/`).
4. **Evaluation:** `pytest` is executed against the sandboxed code.
5. **Selection:** If all unit tests pass, the mutation is declared a **SUCCESS** and archived; otherwise, it is rolled back automatically.

---

## 🧪 Unit Tests

We maintain strict test integrity across all core components. To run the automated unit tests, use:

```bash
python -m pytest tests/test_lambda.py -v
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
