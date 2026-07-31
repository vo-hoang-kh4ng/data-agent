"""A value nobody measured must not enter the feature matrix as a measurement.

`_prepare_matrix` filled every gap with 0.0 and scaled the result. For a column that is
mostly present that is a small distortion; for a column that is mostly absent it is the
whole column.

Measured on the 62,467-row Churn_VT export:

    ratio_missed_30d      98.19% absent    true mean of the 1,133 present values: 0.674

Filling that with 0.0 does not say "we don't know" — 0.0 is the FLOOR of a ratio, the
farthest possible point from 0.674. 61,334 subscribers were handed the assertion "0% of
your outgoing calls failed", and KMeans then read it as a fact about them.

Zero is not a neutral filler. It is only correct when absence genuinely means "no event
occurred", and that is a property of the column that the data cannot reveal — the
business team has to say so. So the policy is declared, never assumed:

    absent above the threshold, undeclared  ->  drop the column, say so
    absent below the threshold, undeclared  ->  median, which moves the distribution least
    declared as "absence means none"        ->  0.0, the original behaviour, now on purpose
"""
import numpy as np
import pandas as pd
import pytest

from triadic_dgm.persona.pipeline import resolve_missing


def _frame(**cols) -> pd.DataFrame:
    return pd.DataFrame(cols)


def test_a_column_with_no_gaps_is_returned_untouched():
    frame = _frame(fee=[1.0, 2.0, 3.0, 4.0])
    filled, report = resolve_missing(frame)
    pd.testing.assert_series_equal(filled["fee"], frame["fee"])
    assert report["dropped"] == []
    assert report["imputed"] == {}


def test_a_mostly_absent_column_is_dropped_not_invented():
    """ratio_missed_30d, the column that started this."""
    frame = _frame(ratio_missed_30d=[np.nan] * 9 + [0.674], kept=list(range(10)))
    filled, report = resolve_missing(frame)
    assert "ratio_missed_30d" not in filled.columns
    assert "kept" in filled.columns
    assert report["dropped"] == ["ratio_missed_30d"]


def test_the_dropped_column_never_becomes_a_row_of_zeros():
    """The specific harm: 0.0 is the floor of a ratio, not the middle of one."""
    frame = _frame(ratio=[np.nan] * 9 + [0.674], other=list(range(10)))
    filled, _ = resolve_missing(frame)
    assert not (filled.to_numpy() == 0.0).any() or "ratio" not in filled.columns


def test_a_lightly_absent_column_is_filled_with_its_median():
    """Median moves the distribution least, and stays inside the observed range."""
    frame = _frame(fee=[10.0, 20.0, 30.0, np.nan])
    filled, report = resolve_missing(frame)
    assert filled["fee"].tolist() == [10.0, 20.0, 30.0, 20.0]
    assert report["imputed"] == {"fee": "median"}


def test_a_declared_column_is_filled_with_zero_on_purpose():
    """total_negative_* is 67-90% absent, and a month with no negative touchpoint really
    does produce no row. Zero is right there — but only because somebody said so."""
    frame = _frame(total_negative_202601=[np.nan] * 9 + [3.0], other=list(range(10)))
    filled, report = resolve_missing(frame, absent_means_zero={"total_negative_202601"})
    assert "total_negative_202601" in filled.columns
    assert filled["total_negative_202601"].tolist() == [0.0] * 9 + [3.0]
    assert report["dropped"] == []
    assert report["zero_filled"] == ["total_negative_202601"]


def test_a_declaration_beats_the_drop_threshold():
    """Declaring a column keeps it however absent it is — that is the point of declaring."""
    frame = _frame(counts=[np.nan] * 99 + [1.0], other=list(range(100)))
    filled, _ = resolve_missing(frame, absent_means_zero={"counts"})
    assert "counts" in filled.columns


def test_the_threshold_is_the_boundary_it_claims_to_be():
    """Exactly at the threshold is still imputable; past it is not."""
    at = _frame(x=[1.0, 2.0, np.nan, np.nan], y=[1.0, 2.0, 3.0, 4.0])
    past = _frame(x=[1.0, np.nan, np.nan, np.nan], y=[1.0, 2.0, 3.0, 4.0])
    assert "x" in resolve_missing(at, max_absent=0.5)[0].columns
    assert "x" not in resolve_missing(past, max_absent=0.5)[0].columns


def test_an_entirely_absent_column_is_dropped_and_has_no_median_to_use():
    frame = _frame(empty=[np.nan] * 4, other=[1.0, 2.0, 3.0, 4.0])
    filled, report = resolve_missing(frame)
    assert "empty" not in filled.columns
    assert report["dropped"] == ["empty"]


def test_an_entirely_absent_declared_column_is_still_all_zeros_not_nan():
    """A NaN reaching StandardScaler raises; the declaration must not open that door."""
    frame = _frame(empty=[np.nan] * 4, other=[1.0, 2.0, 3.0, 4.0])
    filled, _ = resolve_missing(frame, absent_means_zero={"empty"})
    assert filled["empty"].tolist() == [0.0] * 4


def test_nothing_survives_as_nan():
    """The property the caller depends on, whatever route each column took."""
    frame = _frame(
        a=[1.0, np.nan, 3.0, 4.0],
        b=[np.nan] * 3 + [1.0],
        c=[np.nan] * 3 + [2.0],
        d=[1.0, 2.0, 3.0, 4.0],
    )
    filled, _ = resolve_missing(frame, absent_means_zero={"c"})
    assert not filled.isna().to_numpy().any()


def test_the_report_names_every_column_it_touched():
    """Silence is how the old behaviour survived so long — it never said what it filled."""
    frame = _frame(
        light=[1.0, 2.0, 3.0, np.nan],
        heavy=[np.nan] * 3 + [1.0],
        declared=[np.nan] * 3 + [2.0],
        clean=[1.0, 2.0, 3.0, 4.0],
    )
    _, report = resolve_missing(frame, absent_means_zero={"declared"})
    assert report["dropped"] == ["heavy"]
    assert report["imputed"] == {"light": "median"}
    assert report["zero_filled"] == ["declared"]
    assert "clean" not in report["imputed"]


def test_a_frame_with_no_columns_left_is_allowed_to_be_empty():
    """The caller decides that an empty matrix is unusable; this function just reports."""
    frame = _frame(a=[np.nan] * 4, b=[np.nan] * 4)
    filled, report = resolve_missing(frame)
    assert list(filled.columns) == []
    assert sorted(report["dropped"]) == ["a", "b"]


# --- the wiring, which is where the fabricated zeros actually reached KMeans -----------


def test_the_feature_matrix_no_longer_invents_the_absent_values():
    """_prepare_matrix is the only caller that matters."""
    from triadic_dgm.persona.pipeline import _prepare_matrix

    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "solid_a": rng.normal(size=200),
        "solid_b": rng.normal(size=200),
        "mostly_absent": [np.nan] * 198 + [0.674, 0.7],
    })
    prepared = _prepare_matrix(frame, ["solid_a", "solid_b", "mostly_absent"])
    assert prepared is not None
    raw, _ = prepared
    assert "mostly_absent" not in raw.columns, "the 99%-absent column still reached KMeans"
    assert not raw.isna().to_numpy().any()


def test_a_matrix_left_with_too_few_columns_is_rejected():
    """Dropping absent columns can empty the set; that must fail the candidate, not crash."""
    from triadic_dgm.persona.pipeline import _prepare_matrix

    frame = pd.DataFrame({
        "a": [np.nan] * 99 + [1.0],
        "b": [np.nan] * 99 + [2.0],
        "c": list(range(100)),
    })
    assert _prepare_matrix(frame, ["a", "b"]) is None


@pytest.mark.parametrize("column,fraction", [
    ("ratio_missed_30d", 0.9819),
    ("total_call_90d", 0.9819),
])
def test_the_real_churn_vt_columns_would_be_dropped(column, fraction):
    """Documents the measurement that motivated the threshold."""
    n = 10_000
    absent = int(round(n * fraction))
    frame = _frame(**{column: [np.nan] * absent + [0.674] * (n - absent)})
    assert column in resolve_missing(frame)[1]["dropped"]


def test_the_pipeline_reports_only_the_features_that_survived():
    """The columns dropped for being unmeasured must leave the feature list too.

    Found by running the real 62,467-row export rather than a fixture: 16 columns were
    dropped from the matrix while `feats` still named them, and the first use of that list
    — the per-feature global mean — died on KeyError: 'LLSD_202606'. Every downstream
    consumer indexes X_raw by that list, so the two have to agree.
    """
    from triadic_dgm.persona.pipeline import run_persona_pipeline

    rng = np.random.default_rng(11)
    third = 80
    frame = pd.DataFrame({
        "old_fee": np.concatenate([rng.normal(m, 0.4, third) for m in (0.0, 6.0, 12.0)]),
        "recent_fee": np.concatenate([rng.normal(m, 0.4, third) for m in (0.0, 6.0, 12.0)]),
        "old_cl": np.concatenate([rng.normal(m, 0.4, third) for m in (12.0, 6.0, 0.0)]),
        "recent_cl": np.concatenate([rng.normal(m, 0.4, third) for m in (12.0, 6.0, 0.0)]),
    })
    frame["mostly_absent"] = [np.nan] * (len(frame) - 2) + [0.674, 0.7]

    personas = run_persona_pipeline(frame)

    assert personas, "the fixture must produce personas for this test to mean anything"
    for persona in personas:
        assert "mostly_absent" not in persona["features_used"]
        assert "mostly_absent" not in persona["feature_means"]
