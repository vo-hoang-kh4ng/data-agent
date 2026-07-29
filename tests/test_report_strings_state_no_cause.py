"""No user-facing string in the report layer may assert why a customer left.

The data owner's constraint on the 62,467-row churned export: describe the characteristics,
do not state a reason for leaving. The file holds interaction counts, fees and usage trends;
it holds no exit survey, no cancellation reason, no competitor offer.

Fixing the driver ladder and the three narrative maps was not enough — a real report still
carried the claim in its furniture:

    section heading   "🔎 Nguyên nhân rời mạng (suy luận từ hành vi)"
    persona tag       "(Root Cause Identified)" / "(Unclear Cause)"
    risk tier label   "... – ưu tiên điều tra nguyên nhân"
    executive summary "→ Có ít nhất 1 nguyên nhân doanh nghiệp có thể chủ động can thiệp."

Chasing those one at a time is how the next one gets missed, so this walks the module's AST
and checks every string literal it would ever render. Comments are not in the AST, and
docstrings are skipped deliberately: prose explaining WHY a causal claim was removed has to
be free to name the thing it removed — this file is itself an example.
"""
import ast
import pathlib

import pytest

_MODULE = pathlib.Path(__file__).resolve().parents[1] / "triadic_dgm" / "services" / "report_generator.py"

#: Wording that names a specific cause this data cannot evidence. Banned in prose.
_CAUSAL_PHRASES = (
    "root cause",
    "chủ động rời",
    "đối thủ",
    "cạnh tranh",
    "giá cước",
    "dẫn đến",
)

#: The roadmap catalogue names initiatives, KPIs and owners. "Benchmark Competitor Pricing"
#: is a task somebody performs, not a claim about why anyone left, and "Root Cause Coverage"
#: measures how many churns have a RECORDED reason — the metric exists precisely because the
#: data does not contain one. Banning these words outright would delete the report's honest
#: response to not knowing. Entries are exact strings, so each exemption stays visible.
_ACTION_AND_KPI_CATALOGUE = frozenset({
    "Kiểm tra lịch sử tương tác trước khi rời mạng (Root Cause Investigation)",
    "Check Ticket Categories, Trace Root Cause",
    "Exit Survey Response Rate, Root Cause Coverage",
    "Root Cause Coverage, Repeat Churn Pattern Rate",
    "Exit Survey Response Rate (High-Value), Root Cause Coverage",
    "Review Repeat Ticket Pattern, Trace Unresolved Technical Root Cause",
    "Phân tích đối thủ cạnh tranh và chính sách giá",
    "Đánh giá rủi ro cạnh tranh về giá cho nhóm khách hàng giá trị cao",
    "Đánh giá rủi ro cạnh tranh về giá",
    "đối thủ",
    "cạnh tranh",
    "chính sách giá",
})

#: "nguyên nhân" is banned as an ASSERTION but allowed in an ACTION that proposes going and
#: finding out. Recommending an exit survey is the honest response to not knowing why someone
#: left; refusing to say the word would make the report worse, not more careful. Each entry
#: here is a deliberate exception, so adding one is a visible decision rather than a slip.
_ACTIONS_THAT_GO_AND_ASK = frozenset({
    "Thực hiện khảo sát nguyên nhân rời mạng (Exit Survey)",
    "Khảo sát nguyên nhân rời mạng (Exit Survey) cho nhóm giá trị cao",
    "khảo sát nguyên nhân rời mạng",
    "Ghi nhận nguyên nhân rời mạng trực tiếp từ khách hàng",
})


def _rendered_strings() -> list[tuple[int, str]]:
    """Every string constant in the module except docstrings, with its line number."""
    tree = ast.parse(_MODULE.read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                docstrings.add(id(body[0].value))
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings
    ]


#: The narrative prompt is an instruction to the LLM, never shown to a reader, and it has to
#: name the forbidden phrases in order to forbid them. Excluded from the rendered scan and
#: covered by its own tests below — the risk it carries is a causal WORKED EXAMPLE, which
#: teaches the model the very habit the rules prohibit.
_PROMPT_MARKER = "Bạn là Consultant tại Deloitte"


def _narrative_prompt() -> str:
    matches = [text for _, text in _rendered_strings() if _PROMPT_MARKER in text]
    assert len(matches) == 1, f"expected exactly one narrative prompt, found {len(matches)}"
    return matches[0]


def test_the_narrative_prompt_forbids_explaining_why_anyone_left():
    prompt = _narrative_prompt()
    assert "MÔ TẢ, KHÔNG GIẢI THÍCH" in prompt
    for banned in ("nguyên nhân", "dẫn đến", "yếu tố góp phần", "chủ động rời mạng"):
        assert banned in prompt, f"the prohibition no longer names {banned!r}"


def test_the_prompts_worked_example_does_not_demonstrate_a_causal_conclusion():
    """The example is what the model actually copies.

    It used to end with "...cho thấy trải nghiệm dịch vụ tiêu cực nhiều khả năng là yếu tố
    góp phần vào quyết định chấm dứt dịch vụ" — a causal conclusion, modelled for the LLM in
    the same prompt that told it not to speculate. The rules lost; the report was full of
    exactly that sentence shape.
    """
    # Read from the raw source, not the AST: the prompt is an f-string, so ast.Constant
    # yields only the literal chunks between placeholders and the example sits past the
    # first one. Discovered when this test reported "moved" against a prompt still in place.
    source = _MODULE.read_text(encoding="utf-8")
    marker = 'churn_story_facts = {{'
    assert marker in source, "the worked example moved; this test no longer checks anything"
    example = source[source.index(marker):source.index(marker) + 2000]
    for phrase in ("yếu tố góp phần", "dẫn đến quyết định", "nguyên nhân nhiều khả năng"):
        assert phrase not in example, f"the worked example still demonstrates {phrase!r}"


def test_the_scan_finds_strings_at_all():
    """Without this, a broken walker would make every other test here pass vacuously."""
    assert len(_rendered_strings()) > 200


def _report(offenders):
    return "\n".join(f"  {_MODULE.name}:{line}  {text[:130]!r}" for line, text in offenders)


@pytest.mark.parametrize("phrase", _CAUSAL_PHRASES)
def test_no_rendered_string_names_a_cause_the_data_cannot_evidence(phrase):
    offenders = [
        (line, text) for line, text in _rendered_strings()
        if phrase in text.lower() and text.strip() not in _ACTION_AND_KPI_CATALOGUE
        and _PROMPT_MARKER not in text
    ]
    assert not offenders, _report(offenders)


def test_nguyen_nhan_appears_only_in_an_action_that_goes_and_asks():
    offenders = [
        (line, text) for line, text in _rendered_strings()
        if "nguyên nhân" in text.lower() and text.strip() not in _ACTIONS_THAT_GO_AND_ASK
        and _PROMPT_MARKER not in text
    ]
    assert not offenders, _report(offenders)
