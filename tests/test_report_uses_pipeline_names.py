"""The report must show the persona names the pipeline produced, not invent its own.

From one real run on 62,467 churned subscribers, the dashboard and the report — reading the
SAME persona JSON — disagreed on every group but one:

    dashboard (reads persona_name)      report (reads churn_driver)
    Nhóm declining_complaint cao        Không có tín hiệu nổi bật ... — tỷ lệ chi tiêu cao hơn
    Nhóm no_fee_all_period cao          Không có tín hiệu nổi bật ... — tỷ lệ chi tiêu thấp hơn
    Nhóm old_cl cao                     Không có tín hiệu nổi bật ... (3)
    Nhóm no_complaint_all_period cao    Không có tín hiệu nổi bật ... (4)

`_disambiguate_display_names` preferred `churn_driver` and treated `persona_name` as a
fallback, so four groups collapsed onto one label and were told apart by "(3)" and "(4)".

That preference made sense when persona_name was the rule engine's telco string, identical
across every cluster. It is no longer: the pipeline names a cluster the ladder recognised by
its driver, and one the ladder did not by its own measured deviations — already distinct,
already deduplicated. Whatever the report renders, the user compares it against the
dashboard, so the two must not disagree.
"""
import re

import numpy as np
import pandas as pd
import pytest

from triadic_dgm.persona.pipeline import run_persona_pipeline
from triadic_dgm.services.report_generator import ReportGenerator


def _churned_base(n=900):
    """A churned cohort whose clusters differ, none of them loud enough for the ladder."""
    rng = np.random.default_rng(4)
    third = n // 3

    def block(fee, complaints, faults):
        return pd.DataFrame({
            "fee_total": rng.normal(fee, fee * 0.04, third),
            "complaint_total_6m": rng.normal(complaints, 0.2, third).clip(0),
            "cl_total_4m": rng.normal(faults, 0.2, third).clip(0),
            "old_usage": rng.normal(100, 4, third),
            "recent_usage": rng.normal(95, 4, third),
        })

    return pd.concat([block(200, 0.1, 0.1), block(800, 0.4, 0.1), block(300, 0.1, 0.9)],
                     ignore_index=True)


@pytest.fixture(scope="module")
def display_names():
    personas = run_persona_pipeline(_churned_base())
    generator = ReportGenerator.__new__(ReportGenerator)  # no LLM client needed for naming
    return personas, generator._disambiguate_display_names(personas)


def test_the_report_shows_the_name_the_pipeline_chose(display_names):
    personas, display = display_names
    for p in personas:
        shown = display[p["cluster_id"]]
        assert p["persona_name"] in shown, (
            f"report renamed {p['persona_name']!r} to {shown!r}")


def test_no_two_groups_are_told_apart_only_by_a_number(display_names):
    _, display = display_names
    stems = {re.sub(r"\s*\(\d+\)$", "", name) for name in display.values()}
    assert len(stems) == len(display), f"names differ only by a suffix number: {display}"
