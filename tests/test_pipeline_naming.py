"""Persona names must come out of the pipeline, not out of the report renderer.

`rules.py` names personas from a ladder of telco predicates (`status_worsening_pct`,
`high_spender_pct`, `loyalty_rank_avg`, …). On any other dataset none of them match, and
every cluster falls to the same fallback string. Measured on a real 50k-row retail upload:

    'Khách hàng ổn định - Nhóm 1'   support_pct=0.056
    'Khách hàng ổn định - Nhóm 2'   support_pct=0.079
    'Khách hàng ổn định - Nhóm 3'   support_pct=0.580
    'Khách hàng ổn định - Nhóm 4'   support_pct=0.285

The report looked fine because `render_markdown` re-names personas from the DatasetProfile
before rendering. Every OTHER consumer of the same JSON — the dashboard panel, the feed, the
database — kept the four identical names, so the user saw "Khách hàng ổn định" ×4 next to a
report calling them four different things.

Naming from the data belongs where the data is. The report may still upgrade a raw column
name to a human label; it must not be the only thing standing between the user and four
identical names.
"""
import numpy as np
import pandas as pd

from triadic_dgm.persona.pipeline import run_persona_pipeline


def _retail(n=600):
    """Three separable groups on three different axes, no telco columns anywhere."""
    rng = np.random.default_rng(7)
    big_spend = pd.DataFrame({
        "total_spend": rng.normal(900, 40, n // 3),
        "late_delivery_rate": rng.normal(0.05, 0.01, n // 3),
        "avg_installments": rng.normal(2.0, 0.2, n // 3),
    })
    late = pd.DataFrame({
        "total_spend": rng.normal(100, 20, n // 3),
        "late_delivery_rate": rng.normal(0.90, 0.03, n // 3),
        "avg_installments": rng.normal(2.1, 0.2, n // 3),
    })
    instal = pd.DataFrame({
        "total_spend": rng.normal(120, 20, n // 3),
        "late_delivery_rate": rng.normal(0.04, 0.01, n // 3),
        "avg_installments": rng.normal(9.0, 0.5, n // 3),
    })
    return pd.concat([big_spend, late, instal], ignore_index=True)


def _names(df=None):
    return [p["persona_name"] for p in run_persona_pipeline(df if df is not None else _retail())]


def test_generic_personas_do_not_all_share_one_name():
    """The defect, stated directly."""
    names = _names()
    assert len(set(names)) == len(names), f"duplicate persona names: {names}"


def test_the_telco_fallback_string_is_gone_on_a_generic_dataset():
    for name in _names():
        assert "Khách hàng ổn định" not in name


def _with_an_average_cluster(n=300):
    """Three groups spaced evenly, so the middle one sits ON the overall mean.

    `_retail` above separates every group on some axis, which is why it never exercised
    this path. Here the middle cluster deviates on nothing, so `name_by_top_feature`
    returns None for it.
    """
    rng = np.random.default_rng(0)
    g = np.repeat([0, 1, 2], n // 3)
    return pd.DataFrame({
        "spend": rng.normal(10, 1, n) + g * 20,
        "visits": rng.normal(5, 1, n) + g * 9,
    })


def test_a_cluster_near_the_overall_mean_still_gets_a_dataset_neutral_name():
    """The hole in the 2026-07-27 naming fix.

    `name_by_top_feature` returns None when nothing deviates enough, documented as "the
    caller keeps whatever it already had". On the GENERIC path what it already had is the
    rule engine's telco fallback, so the average cluster — the one present in almost every
    dataset — kept saying "Khách hàng ổn định" on data with no customers in it.
    """
    names = _names(_with_an_average_cluster())
    for name in names:
        assert "khách hàng" not in name.lower(), f"telco vocabulary on generic data: {names}"


def test_names_reference_the_features_that_separate_the_groups():
    """A name must be traceable to a measured deviation, not decorative."""
    joined = " ".join(_names()).lower()
    assert "total_spend" in joined or "late_delivery_rate" in joined or "avg_installments" in joined


def test_naming_is_deterministic():
    a, b = _names(_retail()), _names(_retail())
    assert a == b


def test_a_failed_run_keeps_its_stated_failure_name():
    """`_failed_persona` says what went wrong; naming must not overwrite that."""
    personas = run_persona_pipeline(
        pd.DataFrame({"a": [1.0] * 50, "b": [2.0] * 50}),
        behavioral_features=["nope_1", "nope_2"],
    )
    assert personas[0]["persona_name"] == "Không phân hoá được nhóm"


def test_the_telco_path_keeps_its_domain_names():
    """Dual-path guarantee: on data with telco columns the rule-engine names still win."""
    rng = np.random.default_rng(3)
    n = 300
    df = pd.DataFrame({
        "cl_total_6m": np.concatenate([rng.normal(12, 2, n // 2), rng.normal(1, 0.4, n // 2)]),
        "complaint_total_6m": np.concatenate([rng.normal(6, 1, n // 2), rng.normal(0.2, 0.1, n // 2)]),
        "fee_total": rng.normal(500, 50, n),
        "old_usage": rng.normal(100, 10, n),
        "recent_usage": rng.normal(80, 10, n),
    })
    personas = run_persona_pipeline(df)
    # Names come from the rule engine / churn-driver ladder, not from raw column names.
    joined = " ".join(p["persona_name"] for p in personas)
    for column in ("cl_total_6m", "complaint_total_6m", "fee_total"):
        assert column not in joined, f"generic column naming leaked into the telco path: {joined}"
