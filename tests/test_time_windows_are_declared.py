"""Columns measured over different periods sit in one distance computation. Say so.

The Churn_VT export puts all of these into the same feature vector:

    fee_total, fee_avg, cl_total_4m, cl_avg_4m       4 months
    complaint_total_6m, complaint_avg_6m             6 months
    total_call_30d, total_missed_30d                 30 days
    total_call_60d                                   60 days
    total_call_90d                                   90 days
    LLSD_202603 … LLSD_202606                        specific months

Euclidean distance treats every axis as commensurable, so a 6-month complaint count and a
30-day call count are compared as though they covered the same span. Nothing in the data
can fix that — the underlying periods are what they are.

What CAN be fixed is the report presenting the result as though the question never arose.
So the windows are detected from the naming conventions the export already uses and stated
as a limitation. This is a "say what you don't know" guard, not an arithmetic one: it must
never silently rescale anything.
"""
import pytest

from triadic_dgm.persona.pipeline import declared_time_windows, time_window_caveat


# --- reading the conventions the export already uses ----------------------------------


@pytest.mark.parametrize("column,window", [
    ("cl_total_4m", "4 tháng"),
    ("cl_avg_4m", "4 tháng"),
    ("complaint_total_6m", "6 tháng"),
    ("complaint_avg_6m", "6 tháng"),
    ("total_call_30d", "30 ngày"),
    ("total_missed_60d", "60 ngày"),
    ("ratio_missed_90d", "90 ngày"),
])
def test_a_declared_span_is_read_off_the_suffix(column, window):
    assert declared_time_windows([column]) == {window: [column]}


@pytest.mark.parametrize("column", ["LLSD_202606", "HTKT_Net_202603", "total_negative_202601"])
def test_a_year_month_suffix_is_a_single_month(column):
    assert list(declared_time_windows([column])) == ["tháng cụ thể"]


@pytest.mark.parametrize("column", ["fee_total", "high_spender", "segment_avg", "OBJID"])
def test_a_column_declaring_nothing_is_not_guessed_at(column):
    """fee_total covers 4 months, but the NAME does not say so. Inferring it is how the
    report ends up asserting a period nobody wrote down."""
    assert declared_time_windows([column]) == {}


def test_columns_sharing_a_span_are_listed_together():
    windows = declared_time_windows(["cl_total_4m", "cl_avg_4m", "complaint_total_6m"])
    assert windows["4 tháng"] == ["cl_total_4m", "cl_avg_4m"]
    assert windows["6 tháng"] == ["complaint_total_6m"]


def test_no_columns_declare_nothing():
    assert declared_time_windows([]) == {}


def test_a_yyyymm_suffix_is_not_confused_with_a_day_count():
    """260329 is not 'day 260329'; it is the anomalous column the data owner could not name."""
    windows = declared_time_windows(["HTKT_CHECKLIST_260329"])
    assert "ngày" not in " ".join(windows)


# --- what a reader is told -------------------------------------------------------------


def test_one_window_needs_no_caveat():
    """Nothing is mixed, so there is nothing to warn about."""
    assert time_window_caveat(["cl_total_4m", "cl_avg_4m"]) == ""


def test_no_declared_windows_need_no_caveat():
    assert time_window_caveat(["fee_total", "high_spender"]) == ""


def test_mixed_windows_produce_a_caveat_naming_every_span():
    caveat = time_window_caveat(["cl_total_4m", "complaint_total_6m", "total_call_30d"])
    assert caveat
    for window in ("4 tháng", "6 tháng", "30 ngày"):
        assert window in caveat


def test_the_caveat_states_the_consequence_not_just_the_fact():
    """A list of suffixes tells a reader nothing about why it matters."""
    caveat = time_window_caveat(["cl_total_4m", "complaint_total_6m"])
    assert "khoảng cách" in caveat.lower() or "so sánh" in caveat.lower()


def test_the_caveat_never_claims_a_correction_was_applied():
    """Nothing rescales anything; the caveat must not imply otherwise."""
    caveat = time_window_caveat(["cl_total_4m", "complaint_total_6m", "total_call_30d"])
    for claim in ("đã chuẩn hoá", "đã quy đổi", "đã hiệu chỉnh"):
        assert claim not in caveat.lower()


def test_the_real_churn_vt_feature_set_is_flagged():
    columns = ["fee_total", "cl_total_4m", "cl_avg_4m", "complaint_total_6m",
               "complaint_avg_6m", "total_call_30d", "total_call_60d", "total_call_90d",
               "LLSD_202606"]
    caveat = time_window_caveat(columns)
    for window in ("4 tháng", "6 tháng", "30 ngày", "60 ngày", "90 ngày", "tháng cụ thể"):
        assert window in caveat


# --- the wiring ------------------------------------------------------------------------


def test_every_persona_carries_the_caveat_for_the_features_it_was_built_from():
    import numpy as np
    import pandas as pd

    from triadic_dgm.persona.pipeline import run_persona_pipeline

    rng = np.random.default_rng(13)
    third = 100
    blob = lambda means: np.concatenate([rng.normal(m, 0.4, third) for m in means])
    frame = pd.DataFrame({
        "old_cl_total_4m": blob((0.0, 6.0, 12.0)),
        "recent_cl_total_4m": blob((0.0, 6.0, 12.0)),
        "old_complaint_total_6m": blob((12.0, 6.0, 0.0)),
        "recent_complaint_total_6m": blob((12.0, 6.0, 0.0)),
    })
    personas = run_persona_pipeline(frame)
    assert personas
    for persona in personas:
        assert "4 tháng" in persona["time_window_caveat"]
        assert "6 tháng" in persona["time_window_caveat"]


def test_a_single_window_dataset_carries_no_caveat():
    import numpy as np
    import pandas as pd

    from triadic_dgm.persona.pipeline import run_persona_pipeline

    rng = np.random.default_rng(14)
    third = 100
    blob = lambda means: np.concatenate([rng.normal(m, 0.4, third) for m in means])
    frame = pd.DataFrame({
        "old_fee": blob((0.0, 6.0, 12.0)),
        "recent_fee": blob((0.0, 6.0, 12.0)),
        "old_cl": blob((12.0, 6.0, 0.0)),
        "recent_cl": blob((12.0, 6.0, 0.0)),
    })
    for persona in run_persona_pipeline(frame):
        assert persona["time_window_caveat"] == ""


# --- reaching a reader ------------------------------------------------------------------


def _report_markdown(personas):
    import json

    from triadic_dgm.services.report_generator import ReportGenerator

    payload = "[JSON_START_PERSONA]" + json.dumps(personas, ensure_ascii=False) + "[JSON_END_PERSONA]"
    rg = ReportGenerator(api_key="x", base_url="http://localhost:1", model_name="m")
    return rg.render_markdown(payload)


def _persona(**overrides):
    base = {
        "cluster_id": 0,
        "persona_name": "Nhóm ví dụ",
        "support": 100,
        "support_pct": 1.0,
        "feature_means": {"cl_total_4m": 1.0},
        "evidence": {"cl_total_4m": 1.0},
        "persona_type": "MAINSTREAM",
        "severity": "LOW",
        "risk": "LOW",
        "risk_tier": "Nhóm bị động – theo dõi & cảnh báo",
        "priority_score": 10,
        "confidence": "HIGH",
        "recommended_actions": [],
        "time_window_caveat": "",
    }
    base.update(overrides)
    return base


def _prose(markdown: str) -> str:
    """The report minus the raw JSON appendix.

    The appendix dumps the persona objects verbatim, so ANY field would be findable in the
    full markdown. Asserting against that is how a test passes while the reader still sees
    nothing — checked here after this very test went green before the renderer had been
    touched at all.
    """
    import re as _re

    return _re.sub(r"\[JSON_START_PERSONA\].*?\[JSON_END_PERSONA\]", "", markdown, flags=_re.DOTALL)


def test_the_report_states_the_mixed_windows_in_its_own_prose():
    """A caveat only the pipeline log carries never reaches the person reading the report."""
    caveat = time_window_caveat(["cl_total_4m", "complaint_total_6m", "total_call_30d"])
    prose = _prose(_report_markdown([_persona(time_window_caveat=caveat)]))
    for window in ("4 tháng", "6 tháng", "30 ngày"):
        assert window in prose, f"{window} appears only inside the raw JSON appendix"


def test_a_report_with_one_window_says_nothing_about_windows():
    prose = _prose(_report_markdown([_persona(time_window_caveat="")]))
    assert "kỳ quan sát" not in prose


def test_a_persona_from_before_this_field_existed_still_renders():
    """Old persona JSON in the database has no such key; the renderer must not require it."""
    persona = _persona()
    del persona["time_window_caveat"]
    assert _report_markdown([persona])
