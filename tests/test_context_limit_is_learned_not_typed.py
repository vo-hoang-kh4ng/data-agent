"""The model's context window is not something this repository can know by typing a number.

The serving side was resized to 32.000 tokens and the run died:

    This model's maximum context length is 32000 tokens. However, you requested 20000
    output tokens and your prompt contains at least 12001 input tokens, for a total of
    at least 32001 tokens.

Over by one. `MODEL_CONTEXT_LIMIT` said 62000, so `_compute_max_tokens` believed it had
49.000 tokens of room and asked for the full 20.000-token ceiling.

The number has now been wrong twice in opposite directions. It was 30000, which starved
generation and truncated every script mid-line; it was corrected to 62000 by asking a human;
the server then moved underneath it. A constant describing a machine this code cannot see
will keep going stale, and each time it does the failure lands on a user mid-run.

The server states its real limit in the rejection. So the limit is LEARNED: a request that
is refused for length teaches the client the true window, which it then respects for the
rest of the session. Retrying the identical request — which is what the streaming path did,
three times with backoff — cannot succeed, because nothing about a context overflow is
transient.

An explicit `LLM_CONTEXT_LIMIT` still wins when set: an operator who knows the deployment
should not have to provoke a failure to configure it.
"""
import os
from unittest.mock import patch

import pytest

from triadic_dgm.agent.programmer import Programmer, parse_context_limit


_REAL_ERROR = (
    "litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: "
    'Hosted_vllmException - {"error":{"message":"This model\'s maximum context length is 32000 '
    'tokens. However, you requested 20000 output tokens and your prompt contains at least 12001 '
    'input tokens, for a total of at least 32001 tokens. Please reduce the length of the input '
    'prompt or the number of requested output tokens. (parameter=input_tokens, value=12001)"}}'
)


# --- reading the limit out of the refusal ------------------------------------------------


def test_the_real_error_yields_the_real_limit():
    assert parse_context_limit(_REAL_ERROR) == 32000


@pytest.mark.parametrize("text", [
    "This model's maximum context length is 8192 tokens",
    "maximum context length is 131072 tokens, however you requested",
    "MAXIMUM CONTEXT LENGTH IS 4096 TOKENS",
])
def test_other_phrasings_of_the_same_refusal(text):
    assert parse_context_limit(text) in (8192, 131072, 4096)


@pytest.mark.parametrize("text", [
    "Connection reset by peer",
    "504 Gateway Timeout",
    "rate limit exceeded, please retry",
    "",
    None,
])
def test_an_unrelated_failure_teaches_nothing(text):
    """Learning a limit from a timeout would shrink the budget for no reason."""
    assert parse_context_limit(text) is None


def test_a_refusal_without_a_number_teaches_nothing():
    assert parse_context_limit("maximum context length exceeded") is None


# --- what the client asks for ---------------------------------------------------------------


def _programmer(**env):
    with patch.dict(os.environ, env, clear=False):
        return Programmer(api_key="x", model="m", base_url="http://localhost:1")


def _with_prompt(p, chars):
    p.messages = [{"role": "system", "content": "x" * chars}]
    return p


def test_an_operator_can_state_the_limit_without_provoking_a_failure():
    p = _programmer(LLM_CONTEXT_LIMIT="32000")
    assert p.context_limit == 32000


def test_a_bad_env_value_falls_back_to_the_default_rather_than_crashing():
    p = _programmer(LLM_CONTEXT_LIMIT="not a number")
    assert p.context_limit == Programmer.MODEL_CONTEXT_LIMIT


def test_the_request_never_exceeds_the_known_window():
    """The invariant. Everything else here exists to keep this true."""
    p = _programmer(LLM_CONTEXT_LIMIT="32000")
    for chars in (3_000, 36_003, 60_000, 90_000):
        _with_prompt(p, chars)
        estimated_prompt = chars // Programmer.CHARS_PER_TOKEN_ESTIMATE
        assert estimated_prompt + p._compute_max_tokens() <= 32000


def test_the_exact_failing_run_now_fits():
    """~12.001 prompt tokens against a 32.000 window: the request that was over by one."""
    p = _with_prompt(_programmer(LLM_CONTEXT_LIMIT="32000"),
                     12_001 * Programmer.CHARS_PER_TOKEN_ESTIMATE)
    assert 12_001 + p._compute_max_tokens() <= 32000


def test_a_prompt_that_fills_the_window_still_asks_for_something():
    """Better a truncated answer than a request the server cannot accept at all."""
    p = _with_prompt(_programmer(LLM_CONTEXT_LIMIT="32000"), 200_000)
    assert p._compute_max_tokens() >= 1


def test_a_roomy_window_still_reaches_the_ceiling():
    """The 20.000 ceiling exists because smaller values truncated generated scripts."""
    p = _with_prompt(_programmer(LLM_CONTEXT_LIMIT="200000"), 3_000)
    assert p._compute_max_tokens() == Programmer.OUTPUT_TOKENS_CEILING


# --- the history has to be trimmed against the same window ------------------------------------
#
# The trim loop dropped messages until the conversation fell under 120.000 characters — about
# 40.000 tokens, chosen when the window was believed to be 62.000. On a 32.000-token server
# that threshold never fires before the window is already full: the second repair round alone
# appends a whole generated script, and _compute_max_tokens then hands back its floor, which
# is how scripts come back truncated mid-line. Derived from the live limit, the same rule
# reproduces the old 120.000 at the old limit.


def test_the_trim_threshold_follows_the_window():
    small = _programmer(LLM_CONTEXT_LIMIT="32000")._history_char_budget()
    large = _programmer(LLM_CONTEXT_LIMIT="200000")._history_char_budget()
    assert small < large


def test_the_old_threshold_is_reproduced_at_the_old_limit():
    """62.000 tokens was the window when 120.000 characters was written down."""
    p = _programmer(LLM_CONTEXT_LIMIT="62000")
    assert 110_000 <= p._history_char_budget() <= 130_000


def test_history_is_trimmed_before_the_window_is_full():
    p = _programmer(LLM_CONTEXT_LIMIT="32000")
    p.messages = [{"role": "system", "content": "s" * 6_000},
                  {"role": "user", "content": "t" * 6_000}]
    p.messages += [{"role": "assistant", "content": "r" * 30_000} for _ in range(6)]
    p._trim_history()
    prompt_tokens = sum(len(m["content"]) for m in p.messages) // Programmer.CHARS_PER_TOKEN_ESTIMATE
    assert prompt_tokens + p._compute_max_tokens() <= 32000
    assert p._compute_max_tokens() > Programmer.OUTPUT_TOKENS_FLOOR


def test_the_system_prompt_and_the_task_are_never_dropped():
    """Trimming past these leaves the model without its instructions or its job."""
    p = _programmer(LLM_CONTEXT_LIMIT="32000")
    p.messages = [{"role": "system", "content": "SYSTEM" * 20_000},
                  {"role": "user", "content": "TASK" * 20_000}]
    p.messages += [{"role": "assistant", "content": "r" * 40_000} for _ in range(5)]
    p._trim_history()
    assert p.messages[0]["content"].startswith("SYSTEM")
    assert p.messages[1]["content"].startswith("TASK")
    assert len(p.messages) >= 2


# --- learning from the refusal --------------------------------------------------------------


class _Refuses:
    """Rejects anything over `limit` total, the way the real server does."""

    def __init__(self, limit, chars_per_token):
        self.limit = limit
        self.chars_per_token = chars_per_token
        self.requests = []

        outer = self

        class _Completions:
            def create(self, **params):
                prompt = sum(len(str(m.get("content", ""))) for m in params["messages"])
                prompt_tokens = prompt // outer.chars_per_token
                total = prompt_tokens + params["max_tokens"]
                outer.requests.append(total)
                if total > outer.limit:
                    raise RuntimeError(
                        f"This model's maximum context length is {outer.limit} tokens. "
                        f"However, you requested {params['max_tokens']} output tokens and your "
                        f"prompt contains at least {prompt_tokens} input tokens, for a total of "
                        f"at least {total} tokens."
                    )
                return _Response()

        class _Response:
            usage = type("U", (), {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})()

        self.chat = type("C", (), {"completions": _Completions()})()


def _wire(p, server):
    p.client = server
    return p


def test_a_refused_request_is_retried_within_the_limit_the_server_stated():
    p = _with_prompt(_programmer(), 12_001 * Programmer.CHARS_PER_TOKEN_ESTIMATE)
    server = _Refuses(32000, Programmer.CHARS_PER_TOKEN_ESTIMATE)
    assert _wire(p, server)._call_chat_model() is not None
    assert len(server.requests) == 2
    assert server.requests[0] > 32000
    assert server.requests[1] <= 32000


def test_the_limit_is_remembered_for_the_next_call():
    """Relearning it on every call means paying for a failed request every time."""
    p = _with_prompt(_programmer(), 12_001 * Programmer.CHARS_PER_TOKEN_ESTIMATE)
    server = _Refuses(32000, Programmer.CHARS_PER_TOKEN_ESTIMATE)
    _wire(p, server)._call_chat_model()
    assert p.context_limit == 32000
    server.requests.clear()
    p._call_chat_model()
    assert len(server.requests) == 1


def test_a_second_refusal_is_not_chased_forever():
    """One correction per call. A server refusing what it just asked for is a real fault."""
    p = _with_prompt(_programmer(), 12_001 * Programmer.CHARS_PER_TOKEN_ESTIMATE)

    class _AlwaysRefuses(_Refuses):
        def __init__(self):
            super().__init__(32000, Programmer.CHARS_PER_TOKEN_ESTIMATE)
            outer = self

            class _Completions:
                def create(self, **params):
                    outer.requests.append(0)
                    raise RuntimeError("This model's maximum context length is 32000 tokens.")

            self.chat = type("C", (), {"completions": _Completions()})()

    server = _AlwaysRefuses()
    assert _wire(p, server)._call_chat_model() is None
    assert len(server.requests) == 2


def test_a_transient_failure_is_not_treated_as_a_context_problem():
    p = _with_prompt(_programmer(), 3_000)
    calls = []

    class _Down:
        def __init__(self):
            outer = self

            class _Completions:
                def create(self, **params):
                    calls.append(params["max_tokens"])
                    raise RuntimeError("504 Gateway Timeout")

            self.chat = type("C", (), {"completions": _Completions()})()

    assert _wire(p, _Down())._call_chat_model() is None
    assert p.context_limit == Programmer.MODEL_CONTEXT_LIMIT
    assert len(calls) == 1


def test_the_streaming_path_learns_the_same_way():
    """The heaviest call in the pipeline runs here, and it retried the identical request
    three times with backoff — a context overflow is not transient, so that could only
    fail three times and give up."""
    p = _with_prompt(_programmer(), 12_001 * Programmer.CHARS_PER_TOKEN_ESTIMATE)
    seen = []

    class _StreamServer:
        def __init__(self):
            outer = self

            class _Completions:
                def create(self, **params):
                    prompt = sum(len(str(m.get("content", ""))) for m in params["messages"])
                    total = prompt // Programmer.CHARS_PER_TOKEN_ESTIMATE + params["max_tokens"]
                    seen.append(total)
                    if total > 32000:
                        raise RuntimeError(
                            "This model's maximum context length is 32000 tokens.")
                    return iter(())

            self.chat = type("C", (), {"completions": _Completions()})()

    text = "".join(_wire(p, _StreamServer())._call_chat_model_streaming())
    assert "LLM ERROR" not in text
    assert seen[-1] <= 32000
    assert p.context_limit == 32000
