"""The prohibition on stating a cause has to survive contact with the LLM's own prose.

`test_report_strings_state_no_cause.py` walks the module's AST and proves no STATIC string
asserts why a customer left. That guard is real and it holds. It also cannot see the half of
the report a model writes at run time, and that half kept doing it — from the same report
whose prompt forbids it in capital letters:

    "một thay đổi cụ thể trong kỳ quan sát đã DẪN ĐẾN quyết định rời mạng"
    "phản ánh trải nghiệm dịch vụ chưa tốt trước thời điểm chấm dứt"
    "không được giải quyết triệt để ngay từ đầu có thể DẪN ĐẾN mất khách hàng"
    "trải nghiệm dịch vụ giai đoạn đầu là ĐIỂM NÓNG CHÍNH"

The lesson this branch keeps relearning: a rule in a prompt is a request, and requests are
declined. The count was moved into Python for the same reason, and so is this.

The distinction that matters is between ASSERTING a cause and ADMITTING there is none.
"Nguyên nhân nằm ngoài phạm vi dữ liệu hiện có" contains the forbidden word and is the most
honest sentence in the report — the data really does not record why anyone left. Deleting
that would make the report worse, not more careful, so a sentence that says the cause is
unknown survives whatever words it uses.
"""
import pytest

from triadic_dgm.services.report_generator import strip_causal_sentences


# --- sentences the real report printed, which must go --------------------------------------


@pytest.mark.parametrize("sentence", [
    "Việc tín hiệu tiêu cực tập trung vào giai đoạn chót cho thấy một thay đổi cụ thể "
    "trong kỳ quan sát đã dẫn đến quyết định rời mạng.",
    "Các sự cố kỹ thuật không được giải quyết triệt để ngay từ đầu có thể dẫn đến mất khách hàng.",
    "Lịch sử phàn nàn cũ cho thấy trải nghiệm dịch vụ giai đoạn đầu là điểm nóng chính.",
    "Chất lượng dịch vụ kém là nguyên nhân chính khiến nhóm này rời mạng.",
    "Trải nghiệm tiêu cực là yếu tố góp phần vào quyết định chấm dứt dịch vụ.",
    "Sự cố kỹ thuật kéo dài đã gây ra việc khách hàng chấm dứt hợp đồng.",
])
def test_a_sentence_asserting_a_cause_is_dropped(sentence):
    assert strip_causal_sentences(sentence) == ""


def test_only_the_offending_sentence_goes():
    text = ("Khoảng 11.566 khách hàng rời mạng với khiếu nại tăng mạnh ở giai đoạn cuối. "
            "Việc này cho thấy một thay đổi đã dẫn đến quyết định rời mạng. "
            "Tỷ lệ khách hàng giá trị thấp chiếm khoảng 68%.")
    kept = strip_causal_sentences(text)
    assert "11.566 khách hàng" in kept
    assert "68%" in kept
    assert "dẫn đến" not in kept


# --- sentences that admit ignorance, which must stay ----------------------------------------


@pytest.mark.parametrize("sentence", [
    "Việc khách hàng rời đi mà không có biểu hiện bất thường gợi ý nguyên nhân nằm ngoài "
    "phạm vi dữ liệu hiện có.",
    "Dữ liệu hiện có không cho biết nguyên nhân của mức giảm này.",
    "Doanh nghiệp cần bổ sung khảo sát để ghi nhận nguyên nhân rời mạng trực tiếp từ khách hàng.",
    "Đây là nhóm khách hàng rời đi do các yếu tố không được phản ánh trong dữ liệu hành vi hiện tại.",
])
def test_a_sentence_admitting_the_cause_is_unknown_survives(sentence):
    """Refusing to say the word would make the report less honest, not more careful."""
    assert strip_causal_sentences(sentence) == sentence


def test_a_purely_descriptive_sentence_survives():
    sentence = ("Khoảng 22.358 khách hàng rời mạng, với 74,7% là khách hàng giá trị thấp "
                "và mức cước khoảng 194 nghìn đồng/tháng.")
    assert strip_causal_sentences(sentence) == sentence


def test_stating_what_the_data_records_survives():
    sentence = "Dữ liệu ghi nhận số lần phát sinh sự cố, không ghi nhận kết quả xử lý từng lần."
    assert strip_causal_sentences(sentence) == sentence


# --- behaviour at the edges ------------------------------------------------------------------


def test_empty_prose_is_safe():
    assert strip_causal_sentences("") == ""
    assert strip_causal_sentences(None) == ""


def test_prose_that_is_entirely_causal_becomes_empty():
    """Callers treat an empty narrative as "no LLM prose" and fall back to the
    deterministic sections, which is the right outcome — better a shorter report than a
    confident wrong one."""
    assert strip_causal_sentences("Chất lượng kém đã dẫn đến việc rời mạng.") == ""


def test_sentence_spacing_is_not_mangled():
    text = "Nhóm A chiếm 10%. Nhóm B chiếm 20%."
    assert strip_causal_sentences(text) == text


# --- the wiring, which is the half that reaches a reader ---------------------------------------


class _PersonaNarrative:
    def __init__(self, interpretation, impact):
        self.business_interpretation = interpretation
        self.operational_impact = impact


class _Summary:
    def __init__(self, overview):
        self.executive_overview = overview


class _Narrative:
    def __init__(self, overview, conclusion, personas):
        self.executive_summary = _Summary(overview)
        self.conclusion = conclusion
        self.personas_analysis = personas


def test_every_field_the_model_writes_is_cleaned():
    from triadic_dgm.services.report_generator import _strip_causal_narrative

    causal = "Sự cố kỹ thuật đã dẫn đến quyết định rời mạng."
    keep = "Nhóm này chiếm 18,5% tổng đàn."
    narrative = _Narrative(
        f"{keep} {causal}", f"{causal} {keep}",
        [_PersonaNarrative(f"{keep} {causal}", f"{causal} {keep}")],
    )
    _strip_causal_narrative(narrative)

    assert narrative.executive_summary.executive_overview == keep
    assert narrative.conclusion == keep
    assert narrative.personas_analysis[0].business_interpretation == keep
    assert narrative.personas_analysis[0].operational_impact == keep


def test_the_hook_never_raises_on_an_unexpected_narrative_shape():
    from triadic_dgm.services.report_generator import _strip_causal_narrative

    _strip_causal_narrative(object())


def test_the_hook_runs_on_every_generated_narrative():
    """A cleaner nobody calls cleans nothing — the defect this whole file exists for."""
    import inspect

    from triadic_dgm.services.report_generator import ReportGenerator

    source = inspect.getsource(ReportGenerator)
    assert "_strip_causal_narrative(" in source


# --- claims about how the customer felt ------------------------------------------------------
#
# The first version of the filter caught three of the four causal sentences in the real
# report and let this one through:
#
#     "Dữ liệu cho thấy sự cố kỹ thuật là chỉ số duy nhất vượt trội, phản ánh trải nghiệm
#      dịch vụ chưa tốt trước thời điểm chấm dứt."
#
# It states no mechanism, so none of the "dẫn đến" family matches — but it asserts an
# unmeasured internal state. The file records how many faults occurred; no column records
# whether anyone found the experience poor, was dissatisfied, or was unhappy. Naming a
# feeling is the same overreach as naming a cause.
#
# Deliberately narrow: bare "phản ánh" is NOT banned, because "khách hàng phản ánh sự cố" is
# an ordinary description of somebody filing a complaint, and "không được phản ánh trong dữ
# liệu" is the honest admission this filter exists to protect.


@pytest.mark.parametrize("sentence", [
    "Dữ liệu cho thấy sự cố kỹ thuật là chỉ số duy nhất vượt trội, phản ánh trải nghiệm "
    "dịch vụ chưa tốt trước thời điểm chấm dứt.",
    "Nhóm này có trải nghiệm dịch vụ kém trong suốt kỳ quan sát.",
    "Khách hàng không hài lòng với chất lượng đường truyền.",
    "Mức độ bất mãn của nhóm này cao hơn mặt bằng chung.",
])
def test_a_claim_about_how_the_customer_felt_is_dropped(sentence):
    assert strip_causal_sentences(sentence) == ""


def test_a_customer_filing_a_complaint_is_still_describable():
    """"phản ánh" also means "reported it", and that is a recorded event."""
    sentence = "Có 1.504 lượt khách hàng phản ánh sự cố trong giai đoạn cuối kỳ."
    assert strip_causal_sentences(sentence) == sentence


def test_the_admission_about_missing_data_still_survives():
    sentence = "Đây là nhóm rời đi do các yếu tố không được phản ánh trong dữ liệu hiện tại."
    assert strip_causal_sentences(sentence) == sentence


def test_every_causal_sentence_the_real_report_printed_is_now_dropped():
    """The four, together, as they appeared."""
    printed = [
        "Việc tín hiệu tiêu cực tập trung vào giai đoạn chót cho thấy một thay đổi cụ thể "
        "trong kỳ quan sát đã dẫn đến quyết định rời mạng.",
        "Dữ liệu cho thấy sự cố kỹ thuật là chỉ số duy nhất vượt trội, phản ánh trải nghiệm "
        "dịch vụ chưa tốt trước thời điểm chấm dứt.",
        "Các sự cố kỹ thuật không được giải quyết triệt để ngay từ đầu có thể dẫn đến mất "
        "khách hàng.",
        "Lịch sử phàn nàn cũ cho thấy trải nghiệm dịch vụ giai đoạn đầu là điểm nóng chính.",
    ]
    assert [strip_causal_sentences(s) for s in printed] == ["", "", "", ""]
