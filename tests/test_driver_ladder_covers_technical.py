"""A domain that dominates by 890% must not be reported as "no standout signal".

From a real report on 62,467 churned subscribers, one persona (10.0% of the base) printed
these two things about itself, three paragraphs apart:

    "rời mạng sau khi KHÔNG ghi nhận khiếu nại, sự cố kỹ thuật hay mức liên hệ CSKH
     nào vượt trội so với mặt bằng chung"

    old_cl            1.20   benchmark 0.12   +890.4%
    cl_total_4m       1.34   benchmark 0.14   +890.2%
    active_cl_months  1.03   benchmark 0.10   +888.2%

The star thresholds are not the problem: +890% is a deviation of 8.9, comfortably past the
5.0 cut, so the technical domain scored the full 5 stars. The problem is that no rung of the
ladder keys on technical faults alone —

    rung 2:  s_call >= 4 AND s_complaint >= 3 AND s_technical >= 3

— and this export has no `call_*` column at all, so s_call is stuck at 1 and rung 2 can
never fire on it. A cluster defined by technical faults and nothing else matched no rung and
fell to the final one, which says there is no signal.
"""
import pandas as pd
import pytest

from triadic_dgm.persona.profiling import NO_STANDOUT_SIGNAL, classify_churn_driver


def _signature(**stars):
    """Domain stars, defaulting every unnamed domain to 1 (nothing remarkable)."""
    base = {d: {"stars": 1} for d in ("complaint", "call", "missed", "technical", "usage", "value")}
    for domain, value in stars.items():
        base[domain] = {"stars": value}
    return base


def _group():
    """Technical faults concentrated early, which is what the real cluster looked like."""
    return pd.DataFrame({"old_cl": [1.2] * 50, "recent_cl": [0.14] * 50})


def test_dominant_technical_faults_are_recognised():
    result = classify_churn_driver(_group(), _signature(technical=5))
    assert result["churn_driver"] != NO_STANDOUT_SIGNAL


def test_the_recognised_name_says_it_is_technical():
    result = classify_churn_driver(_group(), _signature(technical=5))
    assert "kỹ thuật" in result["churn_driver"].lower()


def test_the_name_states_no_reason_for_leaving():
    """The standing constraint: describe, never explain."""
    result = classify_churn_driver(_group(), _signature(technical=5))
    blob = (result["churn_driver"] + " " + result["churn_driver_evidence"]).lower()
    for phrase in ("nguyên nhân", "dẫn đến", "đối thủ", "cạnh tranh", "giá cước"):
        assert phrase not in blob, f"causal claim: {blob}"


def test_a_merely_slightly_raised_technical_domain_is_not_promoted():
    """The guard against turning the new rung into a catch-all."""
    result = classify_churn_driver(_group(), _signature(technical=2))
    assert result["churn_driver"] == NO_STANDOUT_SIGNAL


@pytest.mark.parametrize("competing", ["complaint", "call"])
def test_a_louder_domain_still_wins(competing):
    """Technical must not out-rank a domain the earlier rungs were written to catch."""
    result = classify_churn_driver(_group(), _signature(technical=4, **{competing: 5}))
    assert "kỹ thuật" not in result["churn_driver"].lower()
