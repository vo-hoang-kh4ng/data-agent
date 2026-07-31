"""A characteristic almost nobody in the group has is not a characteristic of the group.

The 3,816-customer persona in the real report was described entirely by:

    ratio_missed_30d  +1522.1%      value 0.01   benchmark 0.00
    total_missed_30d  +1518.2%      value 0.05   benchmark 0.00
    total_call_30d    +1514.6%      value 0.08   benchmark 0.00

Those columns are absent from 98.19% of the file. The pipeline's missing-value policy drops
them — but the policy sees a DataFrame, and the script that builds it is written by an LLM
each run. If that script fills the gaps before calling in, every blank has already become a
real zero and there is nothing left to detect: the column is 98% zero, which is under the
near-constant threshold, so it passes every guard and arrives with a population mean of
roughly 0.0006 that makes any present value look enormous.

Chasing the fabrication is a losing game. The harm is measurable without it: a cluster mean
of 0.08 on a count column means almost nobody in that cluster has the thing. Whether the
zeros were real or invented, "this group is characterised by X" is false when X is absent
from nearly all of it.

So personas carry the share of their own members the feature actually applies to, and a
signal below the floor is not offered as a description. It stays in the appendix, where the
raw numbers belong.
"""
import numpy as np
import pandas as pd
import pytest

from triadic_dgm.persona.pipeline import feature_coverage
from triadic_dgm.services.report_generator import ReportGenerator


@pytest.fixture
def generator():
    return ReportGenerator(api_key="x", base_url="http://localhost:1", model_name="m")


# --- measuring it -------------------------------------------------------------------------


def test_a_feature_nearly_everyone_has_is_fully_covered():
    """A measured quantity where every row differs. Coverage is (n-1)/n — one row always
    sits at the modal value — so it approaches 1 rather than reaching it."""
    frame = pd.DataFrame({"x": np.linspace(0.0, 1.0, 100)})
    assert feature_coverage(frame, ["x"])["x"] > 0.95


def test_a_feature_almost_nobody_has_is_barely_covered():
    """total_call_30d after an upstream fillna: 1.81% of rows carry anything."""
    frame = pd.DataFrame({"x": [0.0] * 982 + [3.0] * 18})
    assert feature_coverage(frame, ["x"])["x"] == pytest.approx(0.018)


def test_coverage_counts_rows_away_from_the_modal_value_not_away_from_zero():
    """A column whose common value is 5 is not "uncovered" for being non-zero."""
    frame = pd.DataFrame({"x": [5.0] * 90 + [1.0] * 10})
    assert feature_coverage(frame, ["x"])["x"] == pytest.approx(0.10)


def test_absent_values_do_not_count_as_covered():
    frame = pd.DataFrame({"x": [np.nan] * 90 + [1.0] * 10})
    assert feature_coverage(frame, ["x"])["x"] == pytest.approx(0.10)


def test_a_column_that_is_not_there_is_absent_from_the_map():
    assert feature_coverage(pd.DataFrame({"x": [1.0, 2.0]}), ["y"]) == {}


def test_an_empty_frame_measures_nothing():
    assert feature_coverage(pd.DataFrame({"x": []}), ["x"]) == {}


def test_every_persona_carries_the_coverage_of_its_own_cluster():
    """Measured per cluster: a column may describe one group and not another."""
    from triadic_dgm.persona.pipeline import run_persona_pipeline

    rng = np.random.default_rng(17)
    third = 100
    blob = lambda means: np.concatenate([rng.normal(m, 0.4, third) for m in means])
    frame = pd.DataFrame({
        "old_fee": blob((0.0, 6.0, 12.0)),
        "recent_fee": blob((0.0, 6.0, 12.0)),
        "old_cl": blob((12.0, 6.0, 0.0)),
        "recent_cl": blob((12.0, 6.0, 0.0)),
    })
    personas = run_persona_pipeline(frame)
    assert personas
    for persona in personas:
        coverage = persona["feature_coverage"]
        assert coverage
        assert set(coverage) <= set(persona["features_used"])
        assert all(0.0 <= v <= 1.0 for v in coverage.values())


# --- refusing to describe a group by something it does not have -----------------------------


def _persona(**overrides):
    base = {
        "feature_means": {"ratio_missed_30d": 0.01, "old_complaint": 1.18},
        "feature_coverage": {"ratio_missed_30d": 0.018, "old_complaint": 0.86},
    }
    base.update(overrides)
    return base


_GLOBALS = {"ratio_missed_30d": 0.0006, "old_complaint": 0.59}


def test_a_signal_almost_nobody_carries_is_not_offered_as_a_description(generator):
    """The +1522% headline that described 3,816 people by a column covering 1.8% of them."""
    signals = generator._top_signals_covered(_persona(), _GLOBALS)
    assert [f for f, *_ in signals] == ["old_complaint"]


def test_a_signal_the_group_really_carries_is_kept(generator):
    signals = generator._top_signals_covered(_persona(), _GLOBALS)
    assert signals
    assert signals[0][0] == "old_complaint"


def test_without_coverage_nothing_changes(generator):
    """Persona JSON written before this field existed must still render."""
    persona = _persona()
    del persona["feature_coverage"]
    signals = generator._top_signals_covered(persona, _GLOBALS)
    assert [f for f, *_ in signals] == ["ratio_missed_30d", "old_complaint"]


def test_a_group_whose_every_signal_is_uncovered_offers_none(generator):
    """Better to say nothing stood out than to name something nobody has."""
    persona = {
        "feature_means": {"ratio_missed_30d": 0.01, "total_call_30d": 0.08},
        "feature_coverage": {"ratio_missed_30d": 0.018, "total_call_30d": 0.018},
    }
    assert generator._top_signals_covered(persona, _GLOBALS) == []


def test_the_floor_is_the_boundary_it_claims_to_be(generator):
    at = {"feature_means": {"x": 1.0}, "feature_coverage": {"x": 0.20}}
    under = {"feature_means": {"x": 1.0}, "feature_coverage": {"x": 0.19}}
    assert generator._top_signals_covered(at, {"x": 0.5})
    assert generator._top_signals_covered(under, {"x": 0.5}) == []


def test_the_appendix_still_shows_every_number(generator):
    """Suppressing a signal is a statement about DESCRIPTION, not about the record.

    The raw deviation stays in Cluster Feature Statistics, so a reader who wants to check
    what was excluded can. Hiding it there would be the opposite mistake.
    """
    ranked = generator._ranked_deviations(_persona()["feature_means"], _GLOBALS)
    assert "ratio_missed_30d" in [f for f, *_ in ranked]


# --- the baseline has to be the dataset, not the group -------------------------------------
#
# The first version of this measure compared each cluster against its OWN modal value. Run on
# the real file it suppressed exactly the signals worth keeping: `no_complaint_all_period`
# sits at 1.00 across the entire 22,358-customer cluster, so nothing in that cluster departs
# from the cluster's mode, and coverage came out ~0 — scoring a flag true of every single
# member as describing none of them.
#
# A group is characterised by departing from what is normal in the DATASET.


def test_a_flag_true_of_the_whole_group_is_fully_covered():
    from triadic_dgm.persona.pipeline import modal_values

    dataset = pd.DataFrame({"no_complaint_all_period": [0.0] * 59 + [1.0] * 41})
    cluster = pd.DataFrame({"no_complaint_all_period": [1.0] * 22})
    modes = modal_values(dataset, ["no_complaint_all_period"])
    coverage = feature_coverage(cluster, ["no_complaint_all_period"], modes=modes)
    assert coverage["no_complaint_all_period"] == pytest.approx(1.0)


def test_a_flag_matching_the_dataset_norm_covers_nothing():
    """Being like everyone else is not a characteristic."""
    from triadic_dgm.persona.pipeline import modal_values

    dataset = pd.DataFrame({"flag": [0.0] * 59 + [1.0] * 41})
    cluster = pd.DataFrame({"flag": [0.0] * 22})
    modes = modal_values(dataset, ["flag"])
    assert feature_coverage(cluster, ["flag"], modes=modes)["flag"] == pytest.approx(0.0)


def test_the_sparse_column_is_still_caught_against_the_dataset_baseline():
    """ratio_missed_30d: the artefact this whole guard exists for."""
    from triadic_dgm.persona.pipeline import modal_values

    dataset = pd.DataFrame({"ratio_missed_30d": [0.0] * 982 + [0.7] * 18})
    cluster = pd.DataFrame({"ratio_missed_30d": [0.0] * 98 + [0.7] * 2})
    modes = modal_values(dataset, ["ratio_missed_30d"])
    coverage = feature_coverage(cluster, ["ratio_missed_30d"], modes=modes)
    assert coverage["ratio_missed_30d"] == pytest.approx(0.02)


def test_a_real_run_keeps_the_flag_that_defines_its_biggest_group():
    """End to end, on the shape the real file has."""
    from triadic_dgm.persona.pipeline import run_persona_pipeline

    rng = np.random.default_rng(23)
    third = 100
    blob = lambda means: np.concatenate([rng.normal(m, 0.4, third) for m in means])
    frame = pd.DataFrame({
        "old_fee": blob((0.0, 6.0, 12.0)),
        "recent_fee": blob((0.0, 6.0, 12.0)),
        "old_cl": blob((12.0, 6.0, 0.0)),
        "recent_cl": blob((12.0, 6.0, 0.0)),
    })
    # A flag true of one third of the file, i.e. of exactly one cluster.
    frame["no_complaint_all_period"] = [0.0] * (2 * third) + [1.0] * third

    personas = run_persona_pipeline(frame)
    covered = [p["feature_coverage"].get("no_complaint_all_period", 0.0) for p in personas]
    assert max(covered) > 0.9, covered
