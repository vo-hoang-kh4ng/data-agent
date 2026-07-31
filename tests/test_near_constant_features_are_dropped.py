"""A column that can't split the rows can still move the centroids.

`_auto_features` rejected columns with a single distinct value. One step past that the
guard stopped, and columns that are constant in every practical sense went straight into
the matrix. Measured on the 62,467-row Churn_VT export, after the absent-value policy had
already removed the unmeasured ones:

    persistent_cl            99.997% zero        2 rows differ
    persistent_negative      99.749% zero      157 rows differ
    HTKT_CHECKLIST_260329    99.523% zero      297 rows differ   (data owner: "Không rõ")

Two rows cannot form a cluster among 62,467. What those columns do instead is arrive at
StandardScaler with a standard deviation near zero, which turns each of the two differing
rows into a z-score around +170 on that axis — an outlier the nearest centroid then chases.
The column contributes no partition and a large distortion.

The line is drawn on the MINORITY share, not on the count: what matters is whether enough
rows differ to describe a group, and that is relative to the dataset. Columns with a real
if rare signal stay — HTKT_CHECKLIST_202604 differs on 567 rows (0.91%) and is kept.
"""
import numpy as np
import pandas as pd
import pytest

from triadic_dgm.persona.pipeline import is_near_constant


def _series(minority: int, total: int = 10_000) -> pd.Series:
    return pd.Series([0.0] * (total - minority) + [1.0] * minority)


def test_a_column_where_almost_nothing_differs_is_near_constant():
    assert is_near_constant(_series(2))


def test_a_column_with_a_rare_but_real_signal_is_kept():
    """0.91% of rows differ — rare, but enough rows to describe a group."""
    assert not is_near_constant(_series(91))


def test_a_balanced_column_is_never_near_constant():
    assert not is_near_constant(_series(5_000))


def test_a_genuinely_constant_column_is_near_constant():
    assert is_near_constant(pd.Series([1.0] * 100))


def test_the_threshold_is_the_boundary_it_claims_to_be():
    """Exactly at the modal limit is still usable; past it is not."""
    assert not is_near_constant(_series(50, total=10_000), max_modal=0.995)
    assert is_near_constant(_series(49, total=10_000), max_modal=0.995)


def test_absent_values_do_not_count_as_a_value():
    """resolve_missing runs later; a column must not look constant merely for being empty."""
    series = pd.Series([np.nan] * 90 + [1.0] * 5 + [2.0] * 5)
    assert not is_near_constant(series)


def test_an_entirely_absent_column_is_near_constant():
    assert is_near_constant(pd.Series([np.nan] * 100))


def test_an_empty_column_is_near_constant():
    assert is_near_constant(pd.Series([], dtype=float))


@pytest.mark.parametrize("column,minority,expected", [
    ("persistent_cl", 2, True),
    ("persistent_negative", 157, True),
    ("HTKT_CHECKLIST_260329", 297, True),
    ("HTKT_CHECKLIST_202605", 421, False),
    ("HTKT_CHECKLIST_202604", 567, False),
])
def test_the_real_churn_vt_columns_land_where_intended(column, minority, expected):
    """Documents the measurements the threshold was chosen against."""
    assert is_near_constant(_series(minority, total=62_467)) is expected


# --- the wiring ------------------------------------------------------------------------


def test_feature_selection_leaves_the_near_constant_columns_out():
    from triadic_dgm.persona.pipeline import _auto_features

    frame = pd.DataFrame({
        "real_a": np.linspace(0, 1, 10_000),
        "real_b": np.linspace(1, 0, 10_000),
        "persistent_cl": _series(2),
    })
    feats = _auto_features(frame, "cluster")
    assert "persistent_cl" not in feats
    assert {"real_a", "real_b"} <= set(feats)


def test_the_pipeline_never_reports_a_near_constant_feature():
    from triadic_dgm.persona.pipeline import run_persona_pipeline

    rng = np.random.default_rng(3)
    third = 100
    frame = pd.DataFrame({
        "old_fee": np.concatenate([rng.normal(m, 0.4, third) for m in (0.0, 6.0, 12.0)]),
        "recent_fee": np.concatenate([rng.normal(m, 0.4, third) for m in (0.0, 6.0, 12.0)]),
        "old_cl": np.concatenate([rng.normal(m, 0.4, third) for m in (12.0, 6.0, 0.0)]),
        "recent_cl": np.concatenate([rng.normal(m, 0.4, third) for m in (12.0, 6.0, 0.0)]),
    })
    frame["persistent_cl"] = [0.0] * (len(frame) - 1) + [1.0]

    personas = run_persona_pipeline(frame)
    assert personas
    for persona in personas:
        assert "persistent_cl" not in persona["features_used"]
