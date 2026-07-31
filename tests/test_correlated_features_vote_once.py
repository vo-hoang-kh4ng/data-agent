"""Four columns measuring one thing must not outvote one column measuring another.

KMeans works in Euclidean distance over standardised columns, so every column contributes
one unit of variance and the count of columns IS the weighting. Nothing in the algorithm
notices that some of them are the same measurement written down repeatedly.

Measured on the 62,467-row Churn_VT export, after the absent-value and near-constant
guards had run:

    fee_total, fee_avg, fee_old, fee_recent          pairwise |r| > 0.95
    cl_total_4m, cl_avg_4m, cl_std, old_cl           pairwise |r| > 0.95
    complaint_total_6m, complaint_avg_6m             |r| > 0.95
    active_cl_months, no_cl_all_period               |r| > 0.95
    cl_recent_only, escalating_cl                    |r| > 0.95
    HTKT_Net_202606, HTKT_CHECKLIST_202606           |r| > 0.95

Spending and technical faults therefore got four votes each in every distance computation
while a domain represented by one column got one. The segmentation was reporting the
export's column layout as much as the subscribers' behaviour.

Dropping all but one member was rejected: fee_old and fee_recent are 0.95-correlated but
their DIFFERENCE is the temporal signal the whole POST_CHURN path reads. So the group is
kept and down-weighted instead — m mutually-correlated columns are each scaled by
1/sqrt(m), which makes the group contribute the same squared distance as one column while
every column keeps its own values.
"""
import numpy as np
import pandas as pd
import pytest

from triadic_dgm.persona.pipeline import correlation_groups, redundancy_weights


def _correlated(n: int, base: np.ndarray, noise: float, rng) -> np.ndarray:
    return base + rng.normal(0, noise, n)


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    n = 500
    fee = rng.normal(0, 1, n)
    other = rng.normal(0, 1, n)
    return pd.DataFrame({
        "fee_total": fee,
        "fee_avg": _correlated(n, fee, 0.05, rng),
        "fee_old": _correlated(n, fee, 0.05, rng),
        "fee_recent": _correlated(n, fee, 0.05, rng),
        "complaint_total": other,
    })


# --- finding the groups ----------------------------------------------------------------


def test_the_repeated_measurement_is_found_as_one_group(frame):
    groups = correlation_groups(frame)
    assert sorted(max(groups, key=len)) == ["fee_avg", "fee_old", "fee_recent", "fee_total"]


def test_an_independent_column_is_its_own_group(frame):
    groups = correlation_groups(frame)
    assert ["complaint_total"] in groups


def test_every_column_lands_in_exactly_one_group(frame):
    groups = correlation_groups(frame)
    flat = [c for g in groups for c in g]
    assert sorted(flat) == sorted(frame.columns)
    assert len(flat) == len(set(flat))


def test_anti_correlated_columns_are_the_same_measurement_too():
    """no_cl_all_period is active_cl_months negated; r = -1 is still one measurement."""
    rng = np.random.default_rng(6)
    values = rng.normal(0, 1, 300)
    frame = pd.DataFrame({"active_cl_months": values, "no_cl_all_period": -values,
                          "unrelated": rng.normal(0, 1, 300)})
    assert sorted(max(correlation_groups(frame), key=len)) == ["active_cl_months", "no_cl_all_period"]


def test_a_column_with_no_variance_does_not_join_a_group():
    """Correlation with a constant is undefined, not perfect."""
    rng = np.random.default_rng(8)
    frame = pd.DataFrame({"a": rng.normal(0, 1, 200), "flat": [1.0] * 200})
    assert ["flat"] in correlation_groups(frame)


def test_a_single_column_frame_is_one_group():
    assert correlation_groups(pd.DataFrame({"a": [1.0, 2.0, 3.0]})) == [["a"]]


def test_an_empty_frame_has_no_groups():
    assert correlation_groups(pd.DataFrame()) == []


# --- the weights -----------------------------------------------------------------------


def test_a_group_of_four_gets_each_column_halved(frame):
    """1/sqrt(4) = 0.5, so the four together contribute what one column would."""
    weights = redundancy_weights(frame)
    for column in ("fee_total", "fee_avg", "fee_old", "fee_recent"):
        assert weights[column] == pytest.approx(0.5)


def test_an_independent_column_keeps_its_full_vote(frame):
    assert redundancy_weights(frame)["complaint_total"] == pytest.approx(1.0)


def test_a_group_contributes_the_squared_distance_of_one_column(frame):
    """The property the weighting exists to produce."""
    weights = redundancy_weights(frame)
    fee_group = ["fee_total", "fee_avg", "fee_old", "fee_recent"]
    assert sum(weights[c] ** 2 for c in fee_group) == pytest.approx(1.0)
    assert weights["complaint_total"] ** 2 == pytest.approx(1.0)


def test_every_column_keeps_a_positive_weight(frame):
    """Down-weighted is not deleted — a zero weight would silently drop the column."""
    assert all(w > 0 for w in redundancy_weights(frame).values())


# --- the wiring, which is where the four votes were actually cast ----------------------


def test_the_scaled_matrix_carries_the_weights():
    from triadic_dgm.persona.pipeline import _prepare_matrix

    rng = np.random.default_rng(9)
    n = 400
    fee = rng.normal(0, 1, n)
    data = pd.DataFrame({
        "fee_total": fee,
        "fee_avg": fee + rng.normal(0, 0.05, n),
        "fee_old": fee + rng.normal(0, 0.05, n),
        "fee_recent": fee + rng.normal(0, 0.05, n),
        "complaint_total": rng.normal(0, 1, n),
    })
    raw, X = _prepare_matrix(data, list(data.columns))

    variances = X.var(axis=0)
    fee_indices = [list(raw.columns).index(c) for c in ("fee_total", "fee_avg", "fee_old", "fee_recent")]
    other_index = list(raw.columns).index("complaint_total")

    assert sum(variances[i] for i in fee_indices) == pytest.approx(variances[other_index], rel=0.05)


def test_the_unscaled_frame_is_left_alone():
    """Reported means and deviations must stay in the units a reader recognises."""
    from triadic_dgm.persona.pipeline import _prepare_matrix

    rng = np.random.default_rng(10)
    n = 300
    fee = rng.normal(100, 10, n)
    data = pd.DataFrame({
        "fee_total": fee,
        "fee_avg": fee + rng.normal(0, 0.5, n),
        "other": rng.normal(0, 1, n),
    })
    raw, _ = _prepare_matrix(data, list(data.columns))
    assert raw["fee_total"].mean() == pytest.approx(fee.mean())
