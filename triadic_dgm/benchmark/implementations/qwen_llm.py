import json
import os
import threading
from typing import Optional

try:
    import openai
except ImportError:
    openai = None

from triadic_dgm.benchmark.interfaces.llm_client import ILLMClient


def _resolve_repo_root() -> str:
    """Walk up from this file to find the directory containing config.yaml / .env
    (the repo root). Robust to the package being run from any depth; the previous
    hardcoded depth pointed at triadic_dgm/ (one level too deep) and missed the
    repo-root .env. Falls back to the current working directory."""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        if os.path.exists(os.path.join(d, "config.yaml")) or os.path.exists(os.path.join(d, ".env")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.getcwd()


_REPO_ROOT = _resolve_repo_root()

# Qwen3.5 is a hybrid reasoning model: with thinking ON it emits reasoning_content
# (chain-of-thought) separately and puts the final answer in `content`. Thinking is
# more accurate but ~10x slower and burns more tokens; for a many-call benchmark run,
# QWEN_ENABLE_THINKING=false turns it off for speed (content is then returned directly).
_ENABLE_THINKING = os.environ.get("QWEN_ENABLE_THINKING", "true").lower() in ("1", "true", "yes", "on")
# Verbose per-call banners are useful in the interactive app but produce megabytes of
# noise during a benchmark run (thousands of LLM calls). Gate them behind this flag.
_DEBUG = os.environ.get("QWEN_DEBUG", "0") in ("1", "true", "yes", "on")


class OpenAICompatibleClient(ILLMClient):
    """
    Implementation of ILLMClient for OpenAI-compatible APIs (including Qwen via Proxy, GPT-4o, DeepSeek).
    """

    def __init__(self, api_key: str = "", base_url: str = "", model_name: str = "hosted_vllm/Qwen/Qwen3.5-35B-A3B-FP8"):
        if "hosted_vllm" in model_name or not base_url or base_url == "EMPTY":
            try:
                from dotenv import load_dotenv
                # Try to load .env from the workspace root (parent of triadic_dgm.benchmark)
                env_path = os.path.join(_REPO_ROOT, ".env")
                load_dotenv(env_path)
            except ImportError:
                pass
                
            default_url = "https://proxy.onebot.meobeo.ai/v1"
            try:
                import yaml
                config_path = os.path.join(_REPO_ROOT, "config.yaml")
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                base_url = config.get("base_url_conv_model", default_url)
                api_key_env = config.get("api_key_env_var", "QWEN_API_KEY")
            except Exception:
                base_url = default_url
                api_key_env = "QWEN_API_KEY"
            api_key = os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY", "EMPTY")
        
        self.api_key = api_key or "EMPTY"
        self.base_url = base_url
        self.model_name = model_name
        
        if openai is None:
            raise ImportError("The 'openai' package is required to use OpenAICompatibleClient.")
            
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=45.0)

    def _create_with_hard_timeout(self, payload: dict, hard_timeout: float):
        """Run the blocking chat.completions.create in a daemon thread with a HARD wall-clock
        cap. The LiteLLM proxy sometimes holds a connection open trickling keepalive bytes,
        which defeats httpx's read timeout (45s) and lets a single call hang indefinitely.
        If the call hasn't returned by `hard_timeout`, raise TimeoutError so generate()'s
        retry loop can back off and retry (or give up after max_retries). The abandoned
        thread, if any, is left as a daemon and dies with the process."""
        box: dict = {}
        target = payload["messages"]

        def _worker():
            try:
                box["resp"] = self.client.chat.completions.create(**payload)
            except BaseException as e:  # noqa: BLE001 - surface to caller
                box["err"] = e

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(hard_timeout)
        if t.is_alive():
            raise TimeoutError(f"LLM call exceeded hard timeout {hard_timeout}s")
        if "err" in box:
            raise box["err"]
        return box["resp"]

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.2,
        max_tokens: int = 8192,
        stop_sequences: Optional[list[str]] = None
    ) -> str:
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        if _DEBUG:
            print("\n" + "="*30 + " 🧠 LLM INPUT " + "="*30)
            if system_prompt:
                print(f"[SYSTEM]:\n{system_prompt}\n")
            print(f"[USER]:\n{prompt}")
            print("="*74 + "\n")

        import time
        # Pacing: pause before each request to avoid the proxy server's RPM rate limit.
        # Configurable via QWEN_CALL_DELAY (seconds; default 3). Lower it for faster eval runs
        # once you've confirmed the proxy tolerates the higher request rate.
        time.sleep(float(os.environ.get("QWEN_CALL_DELAY", "3")))
        
        # Retry budget for transient proxy drops / rate limits / empty responses. 10 (the
        # original value) × escalating backoff = ~275s per persistently-failing call, which
        # cripples a benchmark run; QWEN_MAX_RETRIES lets you fail faster (e.g. 4 for evals).
        max_retries = int(os.environ.get("QWEN_MAX_RETRIES", "10"))
        hard_timeout = float(os.environ.get("QWEN_CALL_TIMEOUT", "90"))
        for attempt in range(max_retries):
            try:
                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stop": stop_sequences,
                    "extra_body": {"cache": {"no-cache": True},
                                   "chat_template_kwargs": {"enable_thinking": _ENABLE_THINKING},
                                   "timeout": 120},
                }
                response = self._create_with_hard_timeout(payload, hard_timeout)

                choice = response.choices[0]
                content = choice.message.content or ""
                finish_reason = getattr(choice, "finish_reason", "stop")
                
                if not content.strip():
                    raise ValueError("Empty response from LLM")
                if finish_reason == "length":
                    raise ValueError("Response truncated due to length")
                    
                if _DEBUG:
                    print("\n" + "="*30 + " 🤖 LLM OUTPUT " + "="*30)
                    print(content)
                    print("="*75 + "\n")
                return content
            except Exception as e:
                err_msg = str(e).lower()
                if "429" in err_msg or "rate limit" in err_msg or "too many requests" in err_msg or "empty response" in err_msg or "timeout" in err_msg:
                    wait_time = 5 * (attempt + 1)
                    print(f"LLM API Error / Drop: Waiting {wait_time}s before retry {attempt+1}/{max_retries}... ({e})")
                    time.sleep(wait_time)
                else:
                    print(f"LLM API Error (Attempt {attempt+1}/{max_retries}): {e}")
                    if attempt == max_retries - 1:
                        return ""
                    time.sleep(2)
        return ""

    def generate_with_tools(
        self,
        messages: list,
        system_prompt: str,
        tools: list,
        temperature: float = 0.2,
        max_tokens: int = 8192
    ) -> tuple[str, Optional[dict]]:
        
        req_messages = []
        if system_prompt:
            req_messages.append({"role": "system", "content": system_prompt})
        req_messages.extend(messages)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=req_messages,
                tools=tools,
                tool_choice="auto",
                temperature=temperature,
                max_tokens=max_tokens
            )
            msg = response.choices[0].message
            content = msg.content or ""
            
            tool_call_dict = None
            if msg.tool_calls and len(msg.tool_calls) > 0:
                tcall = msg.tool_calls[0]
                tool_call_dict = {
                    "name": tcall.function.name,
                    "arguments": json.loads(tcall.function.arguments)
                }
            return content, tool_call_dict
        except Exception as e:
            print(f"LLM API Tool Error: {e}")
            return str(e), None
