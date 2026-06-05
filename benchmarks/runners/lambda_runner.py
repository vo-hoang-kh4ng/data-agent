"""
BenchmarkRunner: Core engine that programmatically evaluates LAMBDA
on data analysis benchmarks from the LAMBDA paper (arxiv 2407.17535).

Uses Programmer + Inspector + CodeKernel directly (bypasses Gradio streaming).
"""
import os
import sys
import re
import json
import time
import yaml
import shutil
import traceback
from typing import Optional, Dict, Any
from datetime import datetime

# Add repo root to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.programmer import Programmer
from core.inspector import Inspector
from core.kernel import CodeKernel, execute
from cache.cache import data_cache
from prompt_engineering.prompts import (
    PROGRAMMER_PROMPT, RESULT_PROMPT, CODE_INSPECT, CODE_FIX, IMPORT
)
from benchmarks.parsers.result_parser import parse_accuracy, parse_mse, parse_knowledge_score
from benchmarks.datasets.registry import DatasetInfo


class BenchmarkRunner:
    """Programmatic LAMBDA runner for benchmark evaluation."""

    def __init__(self, config_path: str = None, max_attempts: int = 5, verbose: bool = True):
        if config_path is None:
            config_path = os.path.join(REPO_ROOT, "config.yaml")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.max_attempts = max_attempts
        self.verbose = verbose
        self.session_cache_path = self._create_session_dir()
        self.kernel = None
        self.programmer = None
        self.inspector = None
        self._init_agents()

    def _create_session_dir(self) -> str:
        project_cache = os.path.join(REPO_ROOT, self.config.get("project_cache_path", "cache/conv_cache/"))
        ts = time.strftime('%Y-%m-%d', time.localtime())
        hsid = str(hash(id(self)))[:8]
        path = os.path.join(project_cache, f"bench-{ts}-{hsid}")
        os.makedirs(path, exist_ok=True)
        return path

    def _init_agents(self):
        """Initialize Programmer, Inspector, and Jupyter kernel."""
        api_key = self.config["api_key"]
        base_url_prog = self.config.get("base_url_programmer", self.config.get("base_url_conv_model"))
        base_url_insp = self.config.get("base_url_inspector", self.config.get("base_url_conv_model"))
        prog_model = self.config.get("programmer_model", self.config.get("conv_model"))
        insp_model = self.config.get("inspector_model", self.config.get("conv_model"))

        self.programmer = Programmer(api_key=api_key, model=prog_model, base_url=base_url_prog)
        self.inspector = Inspector(api_key=api_key, model=insp_model, base_url=base_url_insp)

        self.kernel = CodeKernel(
            session_cache_path=self.session_cache_path,
            max_exe_time=self.config.get("max_exe_time", 18000),
            verbose=0,
        )

        # Run imports
        execute(IMPORT, self.kernel)

        if self.verbose:
            print(f"[Runner] Initialized | Session: {self.session_cache_path}")
            print(f"[Runner] Programmer: {prog_model} | Inspector: {insp_model}")

    def _reset_session(self):
        """Reset programmer/inspector messages and kernel state."""
        self.programmer.clear()
        self.inspector.clear()
        # Reset kernel variables
        execute("%reset -f", self.kernel)
        execute(IMPORT, self.kernel)

    def upload_dataset(self, csv_path: str) -> dict:
        """Upload dataset and inject metadata into programmer system prompt."""
        import shutil as sh
        filename = os.path.basename(csv_path)
        local_path = os.path.join(self.session_cache_path, filename)
        if not os.path.exists(local_path):
            sh.copy2(csv_path, local_path)

        # Get dataset description
        dc = data_cache(local_path)
        desc = dc.get_description()

        # Build system prompt
        system_prompt = PROGRAMMER_PROMPT.format(working_path=self.session_cache_path)
        system_prompt += (
            f"\n\n[SYSTEM ALERT: USER UPLOADED A NEW DATASET]\n"
            f"You MUST use this EXACT file path to read the data in your Python code: '{local_path}'.\n"
            f"Do NOT use any other path. The file is a dataset with the following general information:\n{desc}.\n"
            f"You should care about the missing values and type of each column in your later processing."
        )

        self.programmer.messages = [{"role": "system", "content": system_prompt}]

        if self.verbose:
            print(f"[Runner] Dataset uploaded: {filename} ({desc['num_rows']} rows, {desc['num_features']} cols)")

        return desc

    def run_task(self, instruction: str, metric_type: str = "accuracy",
                 max_attempts: int = None) -> Dict[str, Any]:
        """
        Execute a single evaluation task through the Programmer→Execute→Inspector loop.

        Returns dict with keys:
            success, metric_value, raw_output, code, attempts, error
        """
        if max_attempts is None:
            max_attempts = self.max_attempts

        self.programmer.messages.append({"role": "user", "content": instruction})

        # Step 1: Programmer generates code
        try:
            response = self.programmer._call_chat_model()
            if response is None:
                return {"success": False, "metric_value": None, "raw_output": "",
                        "code": "", "attempts": 0, "error": "API call failed"}
            prog_response = response.choices[0].message.content
        except Exception as e:
            return {"success": False, "metric_value": None, "raw_output": "",
                    "code": "", "attempts": 0, "error": f"API error: {e}"}

        self.programmer.messages.append({"role": "assistant", "content": prog_response})

        # Step 2: Extract and execute code
        is_python, code = self._extract_code(prog_response)
        if not is_python:
            return {"success": False, "metric_value": None, "raw_output": prog_response,
                    "code": "", "attempts": 0, "error": "No Python code found in response"}

        sign, msg_llm, exe_res = self._run_code(code)
        attempts = 1

        # Step 3: Self-correction loop if error
        if 'error' in sign:
            while 'error' in sign and attempts < max_attempts:
                if self.verbose:
                    print(f"    [repair] Attempt {attempts + 1}/{max_attempts}...")

                # Inspector diagnoses
                self.inspector.messages = []
                self.inspector.messages.append({
                    "role": "user",
                    "content": CODE_INSPECT.format(bug_code=code, error_message=msg_llm)
                })
                try:
                    insp_resp = self.inspector._call_chat_model()
                    insp_response = insp_resp.choices[0].message.content if insp_resp else "Try other packages or methods."
                except Exception:
                    insp_response = "Try other packages or methods."

                # Programmer fixes
                self.programmer.messages.append({
                    "role": "user",
                    "content": CODE_FIX.format(bug_code=code, error_message=msg_llm, fix_method=insp_response)
                })
                try:
                    fix_resp = self.programmer._call_chat_model()
                    prog_response = fix_resp.choices[0].message.content if fix_resp else ""
                except Exception:
                    break

                self.programmer.messages.append({"role": "assistant", "content": prog_response})

                is_python, code = self._extract_code(prog_response)
                if not is_python:
                    break

                sign, msg_llm, exe_res = self._run_code(code)
                attempts += 1

        # Step 4: Parse metric
        error_occurred = 'error' in sign
        metric_value = None

        if metric_type == "accuracy":
            metric_value = parse_accuracy(msg_llm)
            # Also try parsing from the full output
            if metric_value is None:
                metric_value = parse_accuracy(exe_res)
        elif metric_type == "mse":
            metric_value = parse_mse(msg_llm)
            if metric_value is None:
                metric_value = parse_mse(exe_res)
        elif metric_type == "score":
            metric_value = parse_knowledge_score(msg_llm, error_occurred)

        return {
            "success": not error_occurred,
            "metric_value": metric_value,
            "raw_output": msg_llm[:3000] if msg_llm else "",
            "code": code[:2000] if code else "",
            "attempts": attempts,
            "error": None if not error_occurred else msg_llm[:500]
        }

    def _extract_code(self, text: str) -> tuple:
        """Extract Python code from LLM response."""
        pattern = r'```python([^\n]*)(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        if len(matches) > 1:
            return True, ''.join(m[1] for m in matches)
        elif len(matches) == 1:
            return True, matches[0][1]
        return False, ''

    def _run_code(self, code: str) -> tuple:
        """Execute code in kernel, returns (sign, msg_llm, exe_res)."""
        try:
            return execute(code, self.kernel)
        except Exception as e:
            err_msg = str(e)
            return ('error', err_msg, err_msg)

    def shutdown(self):
        """Clean up kernel and session."""
        if self.kernel:
            try:
                self.kernel.shutdown()
            except Exception:
                pass
        if self.verbose:
            print("[Runner] Shutdown complete.")


def build_instruction(ds: DatasetInfo, model_name: str) -> str:
    """Build the instruction prompt for a given dataset × model combination."""
    group = ds.group
    task = ds.task_type
    metric = ds.metric

    # Special instructions for image/text/knowledge groups
    if group == "group5_image":
        if "CNN" in model_name:
            return (
                "Load the MNIST dataset using torchvision.datasets.MNIST.\n"
                "Train a Convolutional Neural Network (CNN) classifier.\n"
                "Use a standard train/test split (60000 train, 10000 test).\n"
                "Print ONLY the test accuracy in this exact format: FINAL_ACCURACY: <value>\n"
                "Keep the CNN simple (2 conv layers + FC) to run within reasonable time. Train for 5 epochs max."
            )
        else:
            return (
                "Load the MNIST dataset using torchvision.datasets.MNIST.\n"
                "Train a Transformer-based classifier.\n"
                "Use a standard train/test split (60000 train, 10000 test).\n"
                "Print ONLY the test accuracy in this exact format: FINAL_ACCURACY: <value>\n"
                "Keep the model simple to run within reasonable time. Train for 5 epochs max."
            )

    if group == "group6_text":
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", "data", "sms_spam.csv")
        if "BERT" in model_name:
            return (
                f"Load the SMS Spam Collection dataset from '{data_path}'.\n"
                "The CSV has columns 'label' (ham/spam) and 'text'.\n"
                "Use DistilBERT-base-uncased for transfer learning to classify spam vs ham.\n"
                "Use a standard train/test split (80/20).\n"
                "Print ONLY the test accuracy in this exact format: FINAL_ACCURACY: <value>\n"
                "Fine-tune for 3 epochs max with a small batch size to stay within time limits."
            )
        else:
            return (
                f"Load the SMS Spam Collection dataset from '{data_path}'.\n"
                "The CSV has columns 'label' (ham/spam) and 'text'.\n"
                "Use TF-IDF vectorization and train a Multinomial Naive Bayes classifier.\n"
                "Use 5-fold cross-validation.\n"
                "Print ONLY the final mean cross-validation accuracy in this exact format: FINAL_ACCURACY: <value>"
            )

    if group == "group7_knowledge":
        if "PAMI" in model_name:
            return (
                "Install the PAMI library (!pip install PAMI). Use FPGrowth algorithm to mine frequent patterns.\n"
                "Use this URL as input: 'https://u-aizu.ac.jp/~udayrage/datasets/transactionalDatabases/Transactional_T10I4D100K.csv'\n"
                "Set minSup=500. Save the mined patterns to a file.\n"
                "Print the total number of patterns, runtime, and memory usage.\n"
                "If the code runs successfully and produces output, that is a PASS."
            )
        elif "Correlation" in ds.name:
            return (
                "Compute the nearest correlation matrix using a Newton-type method.\n"
                "Create a 500x500 random symmetric positive semi-definite matrix.\n"
                "Set b vector to all 1s, tau=0.1, tolerance=1e-6.\n"
                "Print convergence information and the result.\n"
                "If the code runs successfully and produces output, that is a PASS."
            )
        else:
            return (
                "Train a simple neural network with non-negative constraints on weights.\n"
                "Use a synthetic dataset (e.g., random non-negative data).\n"
                "Ensure the network maps nonnegative vectors to nonnegative vectors.\n"
                "Print the training loss and a verification that outputs are non-negative.\n"
                "If the code runs successfully and produces output, that is a PASS."
            )

    # Standard tabular data instructions
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", "data", ds.filename)

    # Build classification/regression instruction
    if task == "classification":
        instruction = (
            f"Train a {model_name} classifier on the dataset at '{data_path}'.\n"
            f"The target column is '{ds.target_col}'.\n"
        )
        if ds.metric == "accuracy":
            instruction += (
                "Use 5-fold cross-validation (cross_val_score) to evaluate the model.\n"
                "Print ONLY the final mean cross-validation accuracy in this exact format: FINAL_ACCURACY: <value>\n"
            )
        else:
            instruction += (
                "Use 5-fold cross-validation to evaluate the model.\n"
                "Print ONLY the final mean cross-validation score.\n"
            )
    else:
        instruction = (
            f"Train a {model_name} regressor on the dataset at '{data_path}'.\n"
            f"The target column is '{ds.target_col}'.\n"
            "Use 5-fold cross-validation with scoring='neg_mean_squared_error'.\n"
            "Convert the negative MSE to positive and print ONLY in this format: FINAL_MSE: <value>\n"
        )

    # Add extra instructions
    if ds.extra_instruction:
        instruction += f"\nImportant: {ds.extra_instruction}"

    # Add preprocessing hints
    instruction += (
        "\nPreprocessing steps to include:\n"
        "1. Handle categorical columns with encoding if needed.\n"
        "2. Scale features if the model benefits from it.\n"
        "3. Handle missing values if any exist.\n"
    )

    return instruction
