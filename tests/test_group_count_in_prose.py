"""The number of groups the prose states must be the number of groups there are.

The real report on 62,467 subscribers rendered five personas, and said this twice:

    Executive Summary: "...xác định ba chân dung chính với đặc điểm hành vi..."
    Conclusion:        "Ba nhóm khách hàng đã rời mạng được phân tích đều có..."

Nobody counted wrong; the count is written by the narrative LLM, which had no reason to
produce the right one and every opportunity to produce a plausible one. It is the first
sentence a reader checks against the table right underneath it, and the cheapest possible
thing to get caught on.

Asking the prompt more firmly is not a fix — the prompt already forbids inventing figures.
A count is arithmetic, so it belongs in Python: the prose is repaired after generation
against the number of personas actually rendered.
"""
import pytest

from triadic_dgm.services.report_generator import correct_group_count


@pytest.mark.parametrize("word,count", [
    ("hai", 2), ("ba", 3), ("bốn", 4), ("năm", 5), ("sáu", 6), ("bảy", 7),
])
def test_a_wrong_spelled_out_count_is_corrected(word, count):
    """Whatever the model wrote, the rendered total wins."""
    fixed = correct_group_count(f"Báo cáo xác định {word} nhóm khách hàng.", 5)
    assert "năm nhóm" in fixed.lower()


def test_a_correct_count_is_left_alone():
    text = "Báo cáo xác định năm nhóm khách hàng."
    assert correct_group_count(text, 5) == text


def test_capitalisation_at_the_start_of_a_sentence_survives():
    fixed = correct_group_count("Ba nhóm khách hàng đã rời mạng được phân tích.", 5)
    assert fixed.startswith("Năm nhóm")


@pytest.mark.parametrize("noun", ["nhóm", "chân dung", "phân khúc", "persona"])
def test_every_word_the_report_uses_for_a_group_is_covered(noun):
    fixed = correct_group_count(f"Báo cáo xác định ba {noun} chính.", 5)
    assert f"năm {noun}" in fixed.lower()


def test_a_digit_count_is_corrected_too():
    assert "5 nhóm" in correct_group_count("Báo cáo xác định 3 nhóm khách hàng.", 5)


def test_numbers_that_do_not_count_groups_are_untouched():
    """The guard that keeps this from mangling the rest of the prose."""
    text = "Trong ba tháng gần đây, năm chỉ số đã thay đổi và bốn khiếu nại được ghi nhận."
    assert correct_group_count(text, 5) == text


def test_both_occurrences_are_repaired():
    """The real report was wrong in the summary AND the conclusion."""
    text = "Xác định ba chân dung chính. Ba nhóm khách hàng này đều có giá trị thấp."
    fixed = correct_group_count(text, 5).lower()
    assert "ba " not in fixed
    assert fixed.count("năm ") == 2


def test_an_absent_or_zero_count_changes_nothing():
    """Never rewrite prose against a total we do not have."""
    text = "Báo cáo xác định ba nhóm khách hàng."
    assert correct_group_count(text, 0) == text
    assert correct_group_count(text, None) == text


def test_empty_prose_is_safe():
    assert correct_group_count("", 5) == ""
    assert correct_group_count(None, 5) is None


# --- the wiring, which is the half that actually reaches a reader --------------------


class _Summary:
    def __init__(self, overview):
        self.executive_overview = overview


class _Narrative:
    def __init__(self, overview, conclusion):
        self.executive_summary = _Summary(overview)
        self.conclusion = conclusion


def test_the_narrative_hook_repairs_both_fields():
    """A correct helper nobody calls fixes nothing."""
    from triadic_dgm.services.report_generator import _correct_narrative_group_counts

    n = _Narrative("Báo cáo xác định ba chân dung chính.", "Ba nhóm khách hàng đã rời mạng.")
    _correct_narrative_group_counts(n, 5)
    assert "năm chân dung" in n.executive_summary.executive_overview
    assert n.conclusion.startswith("Năm nhóm")


def test_the_hook_never_raises_on_an_unexpected_narrative_shape():
    """A wrong count is a blemish; losing the whole narrative to an AttributeError is worse."""
    from triadic_dgm.services.report_generator import _correct_narrative_group_counts

    _correct_narrative_group_counts(object(), 5)  # must not raise


# --- the regression this correction introduced --------------------------------------------
#
# Shipping the fix above produced a NEW contradiction in the next real report:
#
#     "Một nhóm nhỏ tập trung vào sự cố kỹ thuật... Sáu nhóm còn lại chiếm tỷ trọng lớn."
#
# One plus six is seven, and the report rendered six personas. The model had written "Năm
# nhóm còn lại", which was correct — 1 + 5 = 6 — and this function rewrote a right sentence
# into a wrong one, because it matched every "<number> <noun>" without asking what the
# number was counting.
#
# A count qualified by "còn lại", "khác", "trong đó" or a restrictive clause is about a
# SUBSET. The total is not the right answer there, and the original defect never involved
# one. Correcting fewer sentences leaves the model's number, which may be right or wrong;
# correcting a qualified one guarantees a wrong number. So the qualified ones are skipped.


@pytest.mark.parametrize("qualifier", ["còn lại", "khác", "đầu tiên", "cuối cùng"])
def test_a_count_of_a_subset_is_left_alone(qualifier):
    text = f"Một nhóm nhỏ tập trung vào sự cố. Năm nhóm {qualifier} chiếm tỷ trọng lớn."
    assert correct_group_count(text, 6) == text


def test_the_exact_sentence_the_report_printed_is_left_alone():
    text = "Một nhóm nhỏ tập trung vào sự cố kỹ thuật. Năm nhóm còn lại chiếm tỷ trọng lớn."
    assert correct_group_count(text, 6) == text


def test_a_count_restricted_by_what_the_groups_carry_is_left_alone():
    """"Có 3 nhóm mang tín hiệu hành vi rõ ràng" is a subset of six, not a miscount."""
    text = "Có 3 nhóm mang tín hiệu hành vi rõ ràng, đủ cụ thể để rà soát tiếp."
    assert correct_group_count(text, 6) == text


def test_a_count_qualified_by_trong_do_is_left_alone():
    text = "Trong đó ba nhóm có khiếu nại tăng mạnh."
    assert correct_group_count(text, 6) == text


def test_the_original_defect_is_still_corrected():
    """The guard must not disable the fix it is guarding."""
    assert correct_group_count("Báo cáo xác định ba chân dung chính.", 5).lower().startswith(
        "báo cáo xác định năm chân dung")
    assert correct_group_count("Ba nhóm khách hàng đã rời mạng được phân tích.", 5).startswith(
        "Năm nhóm")


def test_a_qualifier_far_away_does_not_protect_a_wrong_total():
    """Only a qualifier attached to the phrase counts, not one elsewhere in the paragraph."""
    text = "Báo cáo xác định ba chân dung chính. Một yếu tố khác nằm ngoài dữ liệu."
    assert "năm chân dung" in correct_group_count(text, 5).lower()
