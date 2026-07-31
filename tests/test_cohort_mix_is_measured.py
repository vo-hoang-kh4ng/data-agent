"""How much of the cohort actually left has to be counted, not assumed.

The dashboard printed this whenever any persona carried a `churn_driver`:

    Churn Status
    100%
    Toàn bộ mẫu đã rời mạng (không áp dụng dự báo rủi ro)

Nothing measured that 100%. `isPostChurnDataset` is `data.some(item => item.churn_driver)`
— one persona having a driver made the whole sample "100% churned".

On the Churn_VT export it is wrong. The data owner confirmed HSSD and CTBDV are mutually
exclusive cancellation codes and that rows carrying neither are subscribers who RESTORED
service: 30,263 + 27,671 + 4,533, so 92.7% left and 7.3% came back. The claim was off by
4,533 people, and it was off in the direction that suppressed every retention script for
the only group retention still applies to.

The pipeline stays dataset-agnostic: it does not know what HSSD means, and must not. The
caller names a status column and says which of its values mean "still a customer"; the
pipeline counts. Deriving that column from HSSD/CTBDV belongs to a Churn_VT-specific
script, not here.
"""
import pandas as pd
import pytest

from triadic_dgm.persona.pipeline import cohort_mix
from triadic_dgm.services.report_generator import should_offer_retention


# --- counting ------------------------------------------------------------------------


def test_a_uniform_cohort_reports_one_status_at_everything():
    mix = cohort_mix(pd.Series(["đã rời"] * 10))
    assert mix == {"đã rời": 1.0}


def test_a_mixed_cohort_reports_the_shares_it_measured():
    mix = cohort_mix(pd.Series(["HSSD"] * 5 + ["CTBDV"] * 3 + ["khôi phục"] * 2))
    assert mix == {"HSSD": 0.5, "CTBDV": 0.3, "khôi phục": 0.2}


def test_the_shares_sum_to_one():
    mix = cohort_mix(pd.Series(["a"] * 7 + ["b"] * 6 + ["c"] * 5))
    assert sum(mix.values()) == pytest.approx(1.0)


def test_absent_statuses_do_not_count_towards_the_total():
    """An unlabelled row is not evidence of any outcome."""
    mix = cohort_mix(pd.Series(["đã rời", "đã rời", None, None]))
    assert mix == {"đã rời": 1.0}


def test_an_entirely_unlabelled_column_measures_nothing():
    assert cohort_mix(pd.Series([None, None])) == {}


def test_an_empty_column_measures_nothing():
    assert cohort_mix(pd.Series([], dtype=object)) == {}


def test_the_real_churn_vt_proportions_come_out():
    """30,263 HSSD + 27,671 CTBDV + 4,533 restored — the numbers that disprove the 100%."""
    series = pd.Series(["HSSD"] * 30_263 + ["CTBDV"] * 27_671 + ["khôi phục"] * 4_533)
    mix = cohort_mix(series)
    assert mix["khôi phục"] == pytest.approx(0.0726, abs=5e-5)
    assert mix["HSSD"] + mix["CTBDV"] == pytest.approx(0.9274, abs=5e-5)


# --- what the pipeline hands the report ----------------------------------------------


def _dataset(n: int = 240) -> pd.DataFrame:
    """Three separable blobs, a quarter of them restored."""
    import numpy as np

    rng = np.random.default_rng(7)
    third = n // 3
    frame = pd.DataFrame({
        "old_fee": np.concatenate([rng.normal(m, 0.4, third) for m in (0.0, 6.0, 12.0)]),
        "recent_fee": np.concatenate([rng.normal(m, 0.4, third) for m in (0.0, 6.0, 12.0)]),
        "old_cl": np.concatenate([rng.normal(m, 0.4, third) for m in (12.0, 6.0, 0.0)]),
        "recent_cl": np.concatenate([rng.normal(m, 0.4, third) for m in (12.0, 6.0, 0.0)]),
    })
    frame["trang_thai"] = ["khôi phục" if i % 4 == 0 else "đã rời" for i in range(len(frame))]
    return frame


def test_every_persona_carries_the_mix_it_was_built_from():
    from triadic_dgm.persona.pipeline import run_persona_pipeline

    personas = run_persona_pipeline(_dataset(), status_col="trang_thai")
    assert personas, "the fixture must produce personas for this test to mean anything"
    for persona in personas:
        mix = persona["cohort_mix"]
        assert mix, f"{persona['persona_name']} carries no measured mix"
        assert sum(mix.values()) == pytest.approx(1.0)


def test_the_status_column_is_never_clustered_on():
    """Segmenting by the outcome and then describing the segments is circular."""
    from triadic_dgm.persona.pipeline import run_persona_pipeline

    frame = _dataset()
    frame["trang_thai_so"] = (frame["trang_thai"] == "khôi phục").astype(int)
    personas = run_persona_pipeline(frame, status_col="trang_thai_so")
    for persona in personas:
        assert "trang_thai_so" not in persona.get("feature_means", {})


def test_without_a_status_column_nothing_is_claimed():
    """No column, no number. Silence beats a fabricated 100%."""
    from triadic_dgm.persona.pipeline import run_persona_pipeline

    personas = run_persona_pipeline(_dataset().drop(columns=["trang_thai"]))
    for persona in personas:
        assert not persona.get("cohort_mix")


def test_the_still_active_share_is_reported_when_the_caller_names_it():
    from triadic_dgm.persona.pipeline import run_persona_pipeline

    personas = run_persona_pipeline(
        _dataset(), status_col="trang_thai", active_status_values={"khôi phục"})
    assert personas
    for persona in personas:
        assert persona["active_pct"] == pytest.approx(persona["cohort_mix"].get("khôi phục", 0.0))


def test_naming_no_active_values_leaves_the_share_unstated():
    """`active_pct` of 0.0 would read as "nobody is active", which we did not measure."""
    from triadic_dgm.persona.pipeline import run_persona_pipeline

    personas = run_persona_pipeline(_dataset(), status_col="trang_thai")
    for persona in personas:
        assert persona.get("active_pct") is None


# --- the consequence that reaches a reader -------------------------------------------


def _persona(**overrides):
    base = {
        "persona_name": "Nhóm ví dụ",
        "risk_tier": "Nhóm rủi ro cao – cần hành động ưu tiên",
        "severity": "HIGH",
        "risk": "HIGH",
        "churn_driver": "Sự cố kỹ thuật ở mức cao, các kênh tương tác khác không nổi bật",
    }
    base.update(overrides)
    return base


def test_a_persona_still_holding_active_customers_gets_its_script_back():
    """The 4,533 restored subscribers are exactly who a retention script is for."""
    assert should_offer_retention(_persona(active_pct=0.34))


def test_a_persona_with_nobody_left_to_retain_still_gets_none():
    assert not should_offer_retention(_persona(active_pct=0.0))


def test_an_unmeasured_cohort_keeps_the_cautious_default():
    """No measurement is not permission to sell retention to the departed."""
    assert not should_offer_retention(_persona())
    assert not should_offer_retention(_persona(active_pct=None))


def test_an_active_share_does_not_override_a_low_risk_tier():
    """Being retainable is necessary, not sufficient — the tier test still decides."""
    assert not should_offer_retention(
        _persona(active_pct=0.34, risk_tier="Nhóm bị động – theo dõi & cảnh báo",
                 severity="LOW", risk="LOW"))
