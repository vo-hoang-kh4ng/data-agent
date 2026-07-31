"""The guards have to protect the path that actually runs, not only the tidy one.

`_auto_features` refuses row identifiers and near-constant columns. It is consulted only
when the caller names no features — and in production the caller ALWAYS names features,
because it is an LLM improvising a column list per run. Every guard therefore protected the
fallback and left the live path open.

Seen in a real report on the 62,467-row export, after all of those guards were written:

    persistent_cl                     in Business Signals   (differs on 2 rows of 62,467)
    HTKT_CHECKLIST_260329             in Business Signals   (data owner: "Không rõ")
    ratio_missed_30d  +1522.1%        the top signal of a 3,816-customer persona
    total_call_30d    +1514.6%        measured on 1.8% of the file

The last two are the fabricated-zero artefact returning by a side door: absent in 98.19% of
rows, so the benchmark is ~0 and any present value looks enormous. A persona covering 3,816
subscribers was characterised entirely by columns that describe 1,133 of them.

So the filter moves to where every feature set passes through. A caller naming a column the
guards reject loses that column and is told; naming nothing usable still fails the run.
"""
import numpy as np
import pandas as pd
import pytest

from triadic_dgm.persona.pipeline import run_persona_pipeline, usable_features


def _clusterable(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    third = n // 3
    blob = lambda means: np.concatenate([rng.normal(m, 0.4, third) for m in means])
    return pd.DataFrame({
        "old_fee": blob((0.0, 6.0, 12.0)),
        "recent_fee": blob((0.0, 6.0, 12.0)),
        "old_cl": blob((12.0, 6.0, 0.0)),
        "recent_cl": blob((12.0, 6.0, 0.0)),
    })


_REAL = ["old_fee", "recent_fee", "old_cl", "recent_cl"]


# --- the filter itself ------------------------------------------------------------------


def test_a_row_identifier_is_refused_however_it_was_offered():
    frame = _clusterable()
    frame["ROW_ID"] = range(len(frame))
    kept, dropped = usable_features(frame, _REAL + ["ROW_ID"])
    assert "ROW_ID" not in kept
    assert dropped["ROW_ID"] == "định danh"


def test_a_near_constant_column_is_refused_however_it_was_offered():
    frame = _clusterable()
    frame["persistent_cl"] = [0.0] * (len(frame) - 1) + [1.0]
    kept, dropped = usable_features(frame, _REAL + ["persistent_cl"])
    assert "persistent_cl" not in kept
    assert dropped["persistent_cl"] == "gần như hằng số"


def test_a_non_numeric_column_is_refused():
    frame = _clusterable()
    frame["trang_thai"] = ["a"] * len(frame)
    kept, dropped = usable_features(frame, _REAL + ["trang_thai"])
    assert "trang_thai" not in kept


def test_a_column_that_is_not_there_is_refused_rather_than_crashing():
    kept, dropped = usable_features(_clusterable(), _REAL + ["khong_ton_tai"])
    assert "khong_ton_tai" not in kept
    assert kept == _REAL


def test_real_features_survive_untouched():
    kept, dropped = usable_features(_clusterable(), _REAL)
    assert kept == _REAL
    assert dropped == {}


def test_order_is_preserved():
    """Feature order decides column order in every downstream table."""
    frame = _clusterable()
    frame["ROW_ID"] = range(len(frame))
    kept, _ = usable_features(frame, ["recent_cl", "ROW_ID", "old_fee"])
    assert kept == ["recent_cl", "old_fee"]


def test_a_declared_event_flag_survives_the_near_constant_check():
    """Blanks are zeros once declared, so the flag varies. Same rule as feature selection."""
    frame = _clusterable()
    frame["total_negative_202601"] = [np.nan] * (len(frame) - 30) + [3.0] * 30
    kept, _ = usable_features(frame, _REAL + ["total_negative_202601"],
                              absent_means_zero={"total_negative_202601"})
    assert "total_negative_202601" in kept


# --- through the pipeline, which is the point --------------------------------------------


def test_a_caller_named_identifier_never_reaches_the_clustering():
    frame = _clusterable()
    frame["ROW_ID"] = range(len(frame))
    personas = run_persona_pipeline(frame, behavioral_features=_REAL + ["ROW_ID"])
    assert personas
    for persona in personas:
        assert "ROW_ID" not in persona["features_used"]
        assert "ROW_ID" not in persona["feature_means"]


def test_a_caller_named_near_constant_column_never_reaches_the_clustering():
    frame = _clusterable()
    frame["persistent_cl"] = [0.0] * (len(frame) - 1) + [1.0]
    personas = run_persona_pipeline(frame, behavioral_features=_REAL + ["persistent_cl"])
    assert personas
    for persona in personas:
        assert "persistent_cl" not in persona["features_used"]


def test_the_two_paths_now_agree():
    """The defect in one sentence: naming the columns changed which guards ran."""
    frame = _clusterable()
    frame["ROW_ID"] = range(len(frame))
    frame["persistent_cl"] = [0.0] * (len(frame) - 1) + [1.0]

    named = run_persona_pipeline(frame.copy(), behavioral_features=list(frame.columns))
    auto = run_persona_pipeline(frame.copy())
    assert sorted(named[0]["features_used"]) == sorted(auto[0]["features_used"])


def test_a_caller_naming_only_unusable_columns_fails_the_run():
    """Silently falling back to auto-selection would hide the caller's mistake."""
    frame = _clusterable()
    frame["ROW_ID"] = range(len(frame))
    frame["flat"] = [1.0] * len(frame)
    personas = run_persona_pipeline(frame, behavioral_features=["ROW_ID", "flat"])
    assert len(personas) == 1
    assert personas[0]["persona_name"] == "Không phân hoá được nhóm"


def test_a_caller_naming_a_column_that_does_not_exist_still_fails_loudly():
    """The existing check must survive: a feature list copied from another dataset is a
    different mistake from naming an unusable column, and still worth failing on."""
    personas = run_persona_pipeline(_clusterable(), behavioral_features=["sepal_length"])
    assert len(personas) == 1
    assert "unknown_columns" in (personas[0].get("failure_reason") or "")
