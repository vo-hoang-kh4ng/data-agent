"""The persona core is reachable from production only as a string. This runs that string.

`grep -rn "run_persona_pipeline" --include=*.py .` outside `tests/` returns nothing. No
production module imports the persona package. It is reached exactly one way:

    prompts.py (a string)  ->  the sandbox LLM retypes it as code  ->  run_persona_pipeline
                           ->  prints [JSON_START_PERSONA]...[JSON_END_PERSONA]
                           ->  extract_persona_list()  ->  the report layer

Two joints, both made of string, with nothing checking them against each other:

1. The symbol. `tests/test_prompt_invariant.py` asserts `"run_persona_pipeline" in body` —
   a substring check on the prompt. It stays green after the function is deleted from the
   repo, because it never imports anything.
2. The markers. `extract_persona_list` returns [] when they are absent and never raises, so
   a one-sided rename produces an empty report in silence rather than an error.

So instead of describing the contract, this module executes it: the prompt's own code
blocks, verbatim, against a synthetic DataFrame, with the real production parser reading the
result. Renaming the function, changing its signature, moving the module, editing one marker,
or returning a type `json.dumps` chokes on all turn this red.

What it does NOT cover, stated so nobody mistakes it for full coverage: whatever the model
improvises when it departs from these blocks. Nothing can test that. Persona *names* are not
asserted here either — that is `tests/test_pipeline_naming.py`'s job.
"""
import contextlib
import io
import json
import os
import re

import numpy as np
import pandas as pd
import pytest

from triadic_dgm.prompts.prompts import PROGRAMMER_PROMPT_V2
from triadic_dgm.services.persona_json import extract_persona_list

# The prompt reaches the model in two different string forms, from two call sites:
#   LAMBDA.py:93                    -> PROGRAMMER_PROMPT.format(working_path=...)
#   triadic_dgm/agent/programmer.py -> passed raw, deliberately unformatted (see its
#                                      comment at line 145: formatting was tried, reverted)
# Only the formatted form can be damaged by the doubled-brace invariant, so both are run.
_PROMPT_FORMS = {
    "formatted": PROGRAMMER_PROMPT_V2.format(working_path="/tmp/does-not-matter"),
    "raw": PROGRAMMER_PROMPT_V2,
}

_REQUIRED_PERSONA_KEYS = {"persona_name", "support", "support_pct", "feature_means"}


def _pipeline_blocks(prompt_body: str) -> list[str]:
    """The ```python blocks the prompt tells the model to type, verbatim."""
    return [b for b in re.findall(r"```python\n(.*?)```", prompt_body, re.S)
            if "run_persona_pipeline" in b]


def _dataset(n=300):
    """Three separable groups. Deliberately not from `data/` — no real records in tests."""
    rng = np.random.default_rng(0)
    g = np.repeat([0, 1, 2], n // 3)
    return pd.DataFrame({
        "spend": rng.normal(10, 1, n) + g * 20,
        "visits": rng.normal(5, 1, n) + g * 9,
    })


def _run_block(block: str, tmp_path) -> str:
    """Execute one block the way the sandbox would, and return its stdout.

    The namespace holds only what the prompt assumes already exists — `data` and
    `behavioral_features`. Anything else the block needs it must import itself, which is
    precisely the joint under test. cwd moves to tmp_path because one block calls
    `save_cluster_chart`, which writes a file relative to the working directory.
    """
    namespace = {"data": _dataset(), "behavioral_features": ["spend", "visits"]}
    out = io.StringIO()
    tmp_path.mkdir(parents=True, exist_ok=True)
    previous = os.getcwd()
    os.chdir(tmp_path)  # contextlib.chdir is 3.11+; this project targets 3.10
    try:
        with contextlib.redirect_stdout(out):
            exec(compile(block, "<prompt code block>", "exec"), namespace)
    finally:
        os.chdir(previous)
    return out.getvalue()


@pytest.mark.parametrize("form", sorted(_PROMPT_FORMS))
def test_the_prompt_still_contains_a_pipeline_block(form):
    """Guards the blocks being deleted or the fence being renamed out from under the rest."""
    assert _pipeline_blocks(_PROMPT_FORMS[form]), "no ```python block calls run_persona_pipeline"


@pytest.mark.parametrize("form", sorted(_PROMPT_FORMS))
def test_every_block_the_prompt_dictates_actually_runs(form, tmp_path):
    """The symbol joint. A rename, a moved module or a changed signature lands here."""
    for i, block in enumerate(_pipeline_blocks(_PROMPT_FORMS[form])):
        _run_block(block, tmp_path / f"{form}-{i}")


@pytest.mark.parametrize("form", sorted(_PROMPT_FORMS))
def test_the_production_parser_reads_what_the_block_prints(form, tmp_path):
    """The marker joint, tested end to end rather than by comparing two string constants.

    `extract_persona_list` is imported from production, not reimplemented here: a copy would
    drift in exactly the way this test exists to catch.
    """
    for i, block in enumerate(_pipeline_blocks(_PROMPT_FORMS[form])):
        personas = extract_persona_list(_run_block(block, tmp_path / f"{form}-{i}"))
        assert len(personas) >= 2, f"block {i} produced no usable personas"
        for p in personas:
            assert _REQUIRED_PERSONA_KEYS <= set(p), f"missing keys: {_REQUIRED_PERSONA_KEYS - set(p)}"


def test_the_pipeline_result_is_json_serialisable(tmp_path):
    """`json.dumps(personas)` is the prompt's own next line, and numpy scalars break it.

    A static check of the symbol would pass while every real run died on
    `TypeError: Object of type int64 is not JSON serializable`.
    """
    from triadic_dgm.persona.pipeline import run_persona_pipeline

    json.dumps(run_persona_pipeline(_dataset()), ensure_ascii=False)


def test_both_prompt_forms_agree(tmp_path):
    """Formatting must not change the code the model is told to type."""
    assert _pipeline_blocks(_PROMPT_FORMS["formatted"]) == _pipeline_blocks(_PROMPT_FORMS["raw"])
