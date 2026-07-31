"""Three places where the report stated the opposite of something it also stated.

Each is a different mechanism, and none of them is the LLM's fault — all three are computed
in Python and rendered verbatim.
"""
import pytest

from triadic_dgm.services.report_generator import (
    _POST_CHURN_TIER_DISPLAY_LABELS,
    ReportGenerator,
)


@pytest.fixture
def generator():
    return ReportGenerator(api_key="x", base_url="http://localhost:1", model_name="m")


# --- A3: a risk tier is not a statement about evidence -----------------------------------
#
# The executive summary counted the personas carrying a real driver and said "Có 3 nhóm mang
# tín hiệu hành vi rõ ràng" — which was right. Section 3b then listed FIVE personas under the
# heading "Nhóm có tín hiệu hành vi rõ ràng trước khi rời mạng", including two whose driver
# was NO_STANDOUT_SIGNAL.
#
# 3b groups by risk_tier and relabels it. "Nhóm rủi ro cao – cần hành động ưu tiên" is a
# statement about PRIORITY; renaming it to "có tín hiệu hành vi rõ ràng" turns it into a
# statement about EVIDENCE, which is a different property the tier never measured.


@pytest.mark.parametrize("label", _POST_CHURN_TIER_DISPLAY_LABELS.values())
def test_a_tier_label_never_claims_a_behavioural_signal(label):
    """The tier knows the priority. It does not know whether a signal was found."""
    lowered = label.lower()
    assert "tín hiệu" not in lowered, label
    assert "dấu hiệu" not in lowered, label


@pytest.mark.parametrize("tier", [
    "Nhóm rủi ro cao – cần hành động ưu tiên",
    "Nhóm bị động – theo dõi & cảnh báo",
    "Nhóm cần giữ chân ngay – ưu tiên giữ chân",
])
def test_every_tier_still_has_a_post_churn_label(tier):
    """Dropping the relabelling entirely would put "cần giữ chân ngay" back on the departed."""
    assert _POST_CHURN_TIER_DISPLAY_LABELS.get(tier)


def test_no_tier_label_asks_for_retention_on_a_churned_cohort():
    for label in _POST_CHURN_TIER_DISPLAY_LABELS.values():
        assert "giữ chân" not in label.lower(), label


# --- A5: a metric that fell to zero did not "appear recently" -----------------------------
#
# The 6,222-customer persona printed:
#
#     Trình tự tín hiệu: Phàn nàn/khiếu nại xuất hiện sớm nhất,
#                        Sự cố kỹ thuật chỉ mới xuất hiện gần đây trước khi rời mạng
#
# while its own trajectory table on the same page showed technical faults going 0.068 -> 0.0,
# labelled "giảm mạnh". `onset_sequence` sorts by the EARLY value descending and calls the
# last entry the most recent one — which is true of onset order, but says nothing about
# whether the metric is present at the end.


def _persona(trajectory):
    return {"onset_sequence": trajectory, "profile_attributes": {}, "feature_means": {}}


def test_a_metric_that_vanished_is_not_called_recent(generator):
    """The exact numbers the 6,222-customer persona rendered."""
    bullets = generator._build_customer_profile_bullets(_persona([
        {"metric": "Phàn nàn/khiếu nại", "old": 0.171, "recent": 0.540},
        {"metric": "Sự cố kỹ thuật", "old": 0.068, "recent": 0.0},
    ]), {})
    onset = " ".join(b for b in bullets if "Trình tự tín hiệu" in b)
    assert "Sự cố kỹ thuật chỉ mới xuất hiện gần đây" not in onset, onset


def test_a_metric_that_really_did_appear_late_is_still_reported(generator):
    """The guard must not delete the sequence whenever it is true."""
    bullets = generator._build_customer_profile_bullets(_persona([
        {"metric": "Sự cố kỹ thuật", "old": 0.227, "recent": 0.0},
        {"metric": "Phàn nàn/khiếu nại", "old": 0.087, "recent": 1.504},
    ]), {})
    onset = " ".join(b for b in bullets if "Trình tự tín hiệu" in b)
    assert "Phàn nàn/khiếu nại chỉ mới xuất hiện gần đây" in onset, onset


def test_the_earliest_metric_must_actually_have_been_present_early(generator):
    """Naming a metric "xuất hiện sớm nhất" when its early value is 0 is the same defect."""
    bullets = generator._build_customer_profile_bullets(_persona([
        {"metric": "Sự cố kỹ thuật", "old": 0.0, "recent": 1.2},
        {"metric": "Phàn nàn/khiếu nại", "old": 0.0, "recent": 0.4},
    ]), {})
    assert not [b for b in bullets if "Trình tự tín hiệu" in b]


def test_a_single_signal_states_no_sequence(generator):
    bullets = generator._build_customer_profile_bullets(_persona([
        {"metric": "Sự cố kỹ thuật", "old": 0.3, "recent": 1.2},
    ]), {})
    assert not [b for b in bullets if "Trình tự tín hiệu" in b]


def test_no_trajectory_states_no_sequence(generator):
    assert not [b for b in generator._build_customer_profile_bullets(_persona([]), {})
                if "Trình tự tín hiệu" in b]


def test_a_malformed_trajectory_entry_does_not_raise(generator):
    generator._build_customer_profile_bullets(_persona([{"metric": "x"}, "not a dict", None]), {})
