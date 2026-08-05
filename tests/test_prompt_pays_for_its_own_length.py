"""Every token the system prompt spends is a token the model cannot spend writing code.

The serving window was cut to 32.000 tokens and a real run died at 32.001. Reconstructing
what was in flight at that moment:

    system prompt                      3.933 tok
    task prompt                          666 tok
    generated script                   2.000 tok
    one repair round (CODE_FIX)        3.442 tok
    repaired script                    2.000 tok
    -------------------------------------------
                                      12.041 tok   (the error said 12.001)

The system prompt is a third of that, and it is the only part paid on EVERY call — the
others at least buy a script or a diagnosis. What it was spending it on:

  * "the pipeline is packaged, unit-tested, deterministic; do not retype the ~800 lines,
    that caused drift until the retries ran out and the user got no report" — stated three
    times, at lines 39, 51 and 56, in three different wordings.
  * "only write your own when the function cannot do it; filter the DataFrame first" —
    twice, at lines 52 and 76.
  * two ```python blocks calling `run_persona_pipeline`, differing only in whether they
    also call `save_cluster_chart`.
  * "sections 4, 4b, 5, 6, 6b, 11 below are kept for reference" — of those six, only 4
    exists. The other five were deleted when the pipeline moved into pipeline.py, and the
    references outlived them, pointing the model at instructions that are not there.

Repetition is not emphasis here. Section 3b opens by declaring itself "ƯU TIÊN CAO NHẤT,
GHI ĐÈ MỌI HƯỚNG DẪN Ở CÁC MỤC 4..." — it overrides a section that says the same thing.

These tests pin what must survive: the size, and the absence of the specific duplication.
They deliberately do NOT pin the wording, which the other two prompt test files already
guard in the ways that matter.
"""
import re

import pytest

from triadic_dgm.prompts.prompts import PROGRAMMER_PROMPT_V2 as PROMPT


#: Ceiling in characters. At CHARS_PER_TOKEN_ESTIMATE=3 this is ~3.100 tokens, under 10% of
#: a 32.000-token window. The prompt was 11.799 characters when this was written; the number
#: is a budget, not a description, and it is meant to be argued with rather than raised
#: quietly — anything added has to displace something.
_MAX_PROMPT_CHARS = 9_500


def test_the_prompt_fits_its_budget():
    assert len(PROMPT) <= _MAX_PROMPT_CHARS, (
        f"{len(PROMPT):,} characters (~{len(PROMPT) // 3:,} tokens) exceeds the "
        f"{_MAX_PROMPT_CHARS:,}-character budget by {len(PROMPT) - _MAX_PROMPT_CHARS:,}"
    )


def test_the_prompt_is_a_small_share_of_the_smallest_window():
    """The window this project actually runs against is 32.000 tokens."""
    assert (len(PROMPT) // 3) / 32_000 < 0.11


# --- the duplication that made it long ------------------------------------------------------


def _sections(body: str) -> dict[str, str]:
    """Numbered sections, keyed by their number."""
    marks = [(m.group(1), m.start()) for m in re.finditer(r"^(\d+b?)\.\s+\S", body, re.M)]
    out = {}
    for i, (number, start) in enumerate(marks):
        end = marks[i + 1][1] if i + 1 < len(marks) else len(body)
        out[number] = body[start:end]
    return out


def test_only_one_section_instructs_the_model_to_call_the_pipeline():
    """Two sections giving the same instruction is not emphasis, it is rent."""
    naming = [n for n, text in _sections(PROMPT).items() if "run_persona_pipeline" in text]
    assert len(naming) == 1, f"sections {naming} all instruct the pipeline call"


def test_the_pipeline_call_is_shown_once():
    blocks = [b for b in re.findall(r"```python\n(.*?)```", PROMPT, re.S)
              if "run_persona_pipeline" in b]
    assert len(blocks) == 1, f"{len(blocks)} blocks show the same call"


def test_the_chart_call_survived_the_merge():
    """It lived only in the section that was folded away."""
    assert "save_cluster_chart" in PROMPT


@pytest.mark.parametrize("kept", [
    # From the section that was merged away — each of these is the only place it is stated.
    "dataset_mode",
    "zero-inflation",
    "failure_reason",
    "features_used",
    "KIỂM TRA SCHEMA",
    # From the surviving section.
    "MỘT KHỐI CODE DUY NHẤT",
    "intermediate_features.csv",
    "[JSON_START_PERSONA]",
])
def test_the_merge_dropped_no_unique_instruction(kept):
    assert kept in PROMPT


def test_the_reason_for_calling_the_pipeline_is_given_once():
    """Three wordings of "retyping the script caused drift until the retries ran out"."""
    assert PROMPT.count("~800 dòng") <= 1


# --- sections that no longer exist ------------------------------------------------------------


def test_the_prompt_points_at_no_section_it_does_not_contain():
    """It named 4, 4b, 5, 6, 6b and 11. Only 4 was ever there; the rest left with the code.

    A dangling pointer in a prompt is worse than a dangling pointer in code, because nothing
    raises — the model is told authoritative detail lives somewhere, finds nothing, and
    fills the gap from whatever it remembers about datasets it has seen before.
    """
    present = set(_sections(PROMPT))
    referenced = {n for n in re.findall(r"mục ([\d]+b?)", PROMPT, re.I)}
    assert referenced <= present, f"points at missing sections: {sorted(referenced - present)}"


# --- the worked example, which taught the opposite of the rules --------------------------------
#
# The prompt closed with a demonstration turn:
#
#     Assistant: "```python
#                 data = load_dataset()
#                 data.head()
#                 ```"
#     User: 'This is the executing result by computer: | Sepal.Length | ... | setosa |'
#     Assistant: "The dataset appears to be the famous Iris dataset..."
#
# Three rules contradicted by the thing the model actually copies:
#
#   * "MỘT KHỐI CODE DUY NHẤT, KHÔNG CHIA LƯỢT" — the example ends its turn after a partial
#     block and waits for a next turn. There is no next turn; that is the failure the rule
#     was written for, quoted in the prompt itself as having happened on real data.
#   * "NO MATTER WHAT THE USER ASKS ... ALWAYS WRITE THE FULL CLUSTERING PIPELINE AND OUTPUT
#     THE JSON PERSONA" — the example writes an EDA turn with no pipeline and no JSON.
#   * the prompt must not hand the model a concrete schema it is not analysing, which is what
#     `test_prompt_names_no_customer_and_no_dataset_shape` in test_prompt_invariant.py guards
#     for the telecom columns. Sepal.Length is the same defect wearing a different dataset.
#
# The rules lost to the example once before, in the report narrative prompt. Same lesson.


def test_the_prompt_demonstrates_no_split_turn():
    """A worked example is what the model copies; it outranks the rule it contradicts."""
    assert "This is the executing result by computer" not in PROMPT


@pytest.mark.parametrize("column", ["Sepal.Length", "Petal.Width", "setosa"])
def test_the_prompt_hands_over_no_foreign_schema(column):
    assert column not in PROMPT


def test_the_single_block_rule_is_still_stated():
    """Removing the counter-example must not remove the rule it was undermining."""
    assert "MỘT KHỐI CODE DUY NHẤT" in PROMPT
    assert "KHÔNG CHIA LƯỢT" in PROMPT


def test_section_numbers_are_not_reused():
    """`_sections` keys by number, so a duplicate would silently hide one from every test
    above. The prompt had two sections numbered 2 before this."""
    numbers = re.findall(r"^(\d+b?)\.\s+\S", PROMPT, re.M)
    assert len(numbers) == len(set(numbers)), f"duplicate section numbers: {numbers}"
