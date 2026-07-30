"""You cannot retain a customer who has already left.

The real report on 62,467 subscribers stated, in its own Raw Facts panel:

    Churn Status  100%  Toàn bộ mẫu đã rời mạng (không áp dụng dự báo rủi ro)

and then offered, for four of five groups:

    Retention Scripts:
      Trải nghiệm kém / CSAT thấp: Xin lỗi về trải nghiệm liên hệ nhiều lần, tổng hợp
      lịch sử tương tác, xử lý dứt điểm trong 1 lần gọi (FCR), khảo sát lại sau xử lý.

A call script apologising to someone and promising first-call resolution is addressed to a
customer the company still has. Rendering it against a churned cohort tells the reader to
phone people who are gone.

The scripts themselves are not wrong — they are the right thing to say to the ACTIVE base
that still shows the same behaviour. They just do not belong in a post-mortem.
"""
import pytest

from triadic_dgm.services.report_generator import should_offer_retention


def _persona(**overrides):
    base = {
        "persona_name": "Nhóm ví dụ",
        "risk_tier": "Nhóm rủi ro cao – cần hành động ưu tiên",
        "severity": "HIGH",
        "risk": "HIGH",
    }
    base.update(overrides)
    return base


def test_a_churned_persona_is_offered_no_retention_script():
    """`churn_driver` is only ever emitted on the POST_CHURN path."""
    assert not should_offer_retention(_persona(churn_driver="Liên hệ CSKH/cuộc gọi nhỡ ở mức cao"))


def test_the_departed_are_spared_even_at_the_highest_severity():
    assert not should_offer_retention(
        _persona(churn_driver="Sự cố kỹ thuật ở mức cao, các kênh tương tác khác không nổi bật",
                 severity="EXTREME", risk="EXTREME",
                 risk_tier="Nhóm cần giữ chân ngay – ưu tiên giữ chân"))


def test_an_active_high_risk_persona_still_gets_one():
    """The guard against fixing this by removing the feature."""
    assert should_offer_retention(_persona())


@pytest.mark.parametrize("field,value", [("severity", "HIGH"), ("risk", "EXTREME")])
def test_an_active_persona_qualifies_on_severity_or_risk_alone(field, value):
    assert should_offer_retention(_persona(risk_tier="", **{field: value}))


def test_an_active_low_risk_persona_is_not_offered_one():
    assert not should_offer_retention(
        _persona(risk_tier="Nhóm bị động – theo dõi & cảnh báo", severity="LOW", risk="LOW"))


def test_the_retain_now_tier_still_qualifies_when_active():
    assert should_offer_retention(
        _persona(risk_tier="Nhóm cần giữ chân ngay – ưu tiên giữ chân", severity="LOW", risk="LOW"))
