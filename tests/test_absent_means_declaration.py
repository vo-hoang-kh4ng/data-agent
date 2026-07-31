"""Whether a blank cell is a zero is a business fact, so the business gets to declare it.

`resolve_missing` has always accepted `absent_means_zero`, and nothing ever supplied it,
so every mostly-absent column was dropped — including six `total_negative_*` columns the
data owner had described as counts of negative touchpoints.

The question cannot be answered from the data. On the Churn_VT export the evidence points
BOTH ways: `total_negative_202601` records an explicit 0 among its 7,751 present values,
which suggests a blank means something other than "none occurred" — but that is an
inference, and inferring it is exactly how a report ends up asserting what nobody measured.

So it becomes a column in the review file the data owner already fills in, travels in the
metadata JSON, and reaches the pipeline through the same schema gate as the labels. A
column nobody has declared keeps the cautious default: dropped, not filled.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_column_metadata import (  # noqa: E402
    ABSENT_COLUMN,
    build,
    normalize_absent_means,
)


# --- what the reviewer may write --------------------------------------------------------


@pytest.mark.parametrize("written", [
    "khong_phat_sinh", "không phát sinh", "Không phát sinh", "  KHÔNG PHÁT SINH  ", "0",
])
def test_a_declaration_that_nothing_happened_reads_as_zero(written):
    assert normalize_absent_means(written) == "zero"


@pytest.mark.parametrize("written", [
    "khong_do", "không đo", "Không đo", "không thuộc phạm vi", "unmeasured",
])
def test_a_declaration_that_nothing_was_measured_reads_as_unmeasured(written):
    assert normalize_absent_means(written) == "unmeasured"


@pytest.mark.parametrize("written", ["", None, "   ", "chưa biết", "?"])
def test_anything_undeclared_stays_undeclared(written):
    """A reviewer's shrug must not be read as an answer either way."""
    assert normalize_absent_means(written) == ""


# --- what the reviewer is asked ---------------------------------------------------------


def _csv(tmp_path: Path) -> Path:
    csv = tmp_path / "demo.csv"
    pd.DataFrame({
        "fee_total": [1.0, 2.0, 3.0, 4.0],
        "total_negative_202601": [np.nan, np.nan, np.nan, 3.0],
    }).to_csv(csv, index=False)
    return csv


def test_only_columns_with_blanks_are_asked_about(tmp_path):
    """Asking about a column that has no blanks is noise in a 97-row review file."""
    _, review_path, _, _ = build(_csv(tmp_path))
    review = pd.read_csv(review_path).set_index("cot")
    assert review.loc["total_negative_202601", "can_khai_bao_o_trong"] == "CÓ"
    assert not str(review.loc["fee_total", "can_khai_bao_o_trong"]).strip() or \
        pd.isna(review.loc["fee_total", "can_khai_bao_o_trong"])


def test_the_review_file_carries_a_column_to_fill_in(tmp_path):
    _, review_path, _, _ = build(_csv(tmp_path))
    assert ABSENT_COLUMN in pd.read_csv(review_path).columns


def test_an_undeclared_column_reaches_the_json_undeclared(tmp_path):
    json_path, _, _, _ = build(_csv(tmp_path))
    columns = {c["column"]: c for c in json.loads(json_path.read_text(encoding="utf-8"))["columns"]}
    assert columns["total_negative_202601"]["absent_means"] == ""


def _declare(review_path: Path, column: str, written: str) -> None:
    """Fill in the reviewer's answer the way a spreadsheet would leave it."""
    review = pd.read_csv(review_path)
    review[ABSENT_COLUMN] = review[ABSENT_COLUMN].astype("object")
    review.loc[review["cot"] == column, ABSENT_COLUMN] = written
    review.to_csv(review_path, index=False, encoding="utf-8-sig")


def test_a_declaration_survives_the_round_trip(tmp_path):
    """The reviewer fills the column in; re-running must carry it into the JSON."""
    csv = _csv(tmp_path)
    _, review_path, _, _ = build(csv)

    _declare(review_path, "total_negative_202601", "không phát sinh")

    json_path, _, _, _ = build(csv)
    columns = {c["column"]: c for c in json.loads(json_path.read_text(encoding="utf-8"))["columns"]}
    assert columns["total_negative_202601"]["absent_means"] == "zero"


def test_a_declaration_is_not_lost_on_a_second_regeneration(tmp_path):
    csv = _csv(tmp_path)
    _, review_path, _, _ = build(csv)
    _declare(review_path, "total_negative_202601", "không phát sinh")

    build(csv)
    json_path, _, _, _ = build(csv)
    columns = {c["column"]: c for c in json.loads(json_path.read_text(encoding="utf-8"))["columns"]}
    assert columns["total_negative_202601"]["absent_means"] == "zero"


# --- reaching the pipeline ----------------------------------------------------------------


def _metadata(tmp_path: Path, entries: list[dict]) -> None:
    (tmp_path / "demo_metadata.json").write_text(
        json.dumps({"dataset_name": "demo", "columns": entries}, ensure_ascii=False),
        encoding="utf-8")


def test_only_the_zero_declarations_are_served(tmp_path):
    from api.services.metadata_gate import absent_zero_columns

    _metadata(tmp_path, [
        {"column": "total_negative_202601", "absent_means": "zero"},
        {"column": "ratio_missed_30d", "absent_means": "unmeasured"},
        {"column": "LLSD_202606", "absent_means": ""},
        {"column": "fee_total"},
    ])
    columns = ["total_negative_202601", "ratio_missed_30d", "LLSD_202606", "fee_total"]
    assert absent_zero_columns(str(tmp_path), columns) == {"total_negative_202601"}


def test_declarations_for_another_export_are_not_served(tmp_path):
    """Same schema gate as the labels: a declaration about a different file is worse than
    none, because it fills a column here on another dataset's authority."""
    from api.services.metadata_gate import absent_zero_columns

    _metadata(tmp_path, [
        {"column": "total_negative_202601", "absent_means": "zero"},
        {"column": "ratio_missed_30d", "absent_means": "zero"},
        {"column": "LLSD_202606", "absent_means": "zero"},
        {"column": "fee_total", "absent_means": "zero"},
    ])
    assert absent_zero_columns(str(tmp_path), ["a", "b", "c", "price", "qty"]) == set()


def test_an_unreadable_file_serves_no_declarations(tmp_path):
    from api.services.metadata_gate import absent_zero_columns

    (tmp_path / "broken_metadata.json").write_text("{not json", encoding="utf-8")
    assert absent_zero_columns(str(tmp_path), ["a", "b"]) == set()


def _clusterable(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(31)
    third = n // 3
    blob = lambda means: np.concatenate([rng.normal(m, 0.4, third) for m in means])
    return pd.DataFrame({
        "old_fee": blob((0.0, 6.0, 12.0)),
        "recent_fee": blob((0.0, 6.0, 12.0)),
        "old_cl": blob((12.0, 6.0, 0.0)),
        "recent_cl": blob((12.0, 6.0, 0.0)),
    })


def test_a_declared_column_is_kept_instead_of_dropped(tmp_path):
    """The whole point: the six negative-touchpoint columns come back."""
    from triadic_dgm.persona.pipeline import run_persona_pipeline

    frame = _clusterable()
    frame["total_negative_202601"] = [np.nan] * (len(frame) - 30) + [3.0] * 30
    _metadata(tmp_path, [
        {"column": "old_fee"}, {"column": "recent_fee"}, {"column": "old_cl"},
        {"column": "recent_cl"},
        {"column": "total_negative_202601", "absent_means": "zero"},
    ])

    personas = run_persona_pipeline(frame, label_dir=str(tmp_path))
    assert personas
    assert "total_negative_202601" in personas[0]["features_used"]


def test_an_undeclared_column_is_still_dropped(tmp_path):
    """The cautious default, unchanged. This is the guard against the feature being a
    silent way to reinstate fillna(0.0) for everything."""
    from triadic_dgm.persona.pipeline import run_persona_pipeline

    frame = _clusterable()
    frame["total_negative_202601"] = [np.nan] * (len(frame) - 30) + [3.0] * 30
    _metadata(tmp_path, [
        {"column": "old_fee"}, {"column": "recent_fee"}, {"column": "old_cl"},
        {"column": "recent_cl"},
        {"column": "total_negative_202601"},
    ])

    personas = run_persona_pipeline(frame, label_dir=str(tmp_path))
    assert personas
    assert "total_negative_202601" not in personas[0]["features_used"]


def test_a_column_declared_unmeasured_is_dropped(tmp_path):
    from triadic_dgm.persona.pipeline import run_persona_pipeline

    frame = _clusterable()
    frame["ratio_missed_30d"] = [np.nan] * (len(frame) - 30) + [0.674] * 30
    _metadata(tmp_path, [
        {"column": "old_fee"}, {"column": "recent_fee"}, {"column": "old_cl"},
        {"column": "recent_cl"},
        {"column": "ratio_missed_30d", "absent_means": "unmeasured"},
    ])

    personas = run_persona_pipeline(frame, label_dir=str(tmp_path))
    assert personas
    assert "ratio_missed_30d" not in personas[0]["features_used"]


def test_with_no_metadata_at_all_nothing_changes(tmp_path):
    from triadic_dgm.persona.pipeline import run_persona_pipeline

    frame = _clusterable()
    frame["total_negative_202601"] = [np.nan] * (len(frame) - 30) + [3.0] * 30
    personas = run_persona_pipeline(frame, label_dir=str(tmp_path))
    assert personas
    assert "total_negative_202601" not in personas[0]["features_used"]
