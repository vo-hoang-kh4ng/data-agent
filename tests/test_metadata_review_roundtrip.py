"""The review file comes back from a spreadsheet, not from this script.

The confirmation loop only pays off if the file the business team actually returns can be
read. The Churn_VT round trip is the worked example of what "actually returns" means: 96
rows of a 97-row file, headers translated into accented Vietnamese, one column deleted
outright, and every one of the 52 answers typed OVER the machine's guess instead of into
the blank column set aside for them.

The first loader looked for exactly `cot` and `mo_ta_nghiep_vu_xac_nhan`, found neither,
and reported zero confirmations against a file holding fifty-two. It did not fail — it
regenerated the machine's own guesses and looked like it had worked, which is the part
worth a test.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_column_metadata import (  # noqa: E402
    REVIEW_COLUMN,
    infer,
    load_confirmations,
    load_group_overrides,
)


def _write(tmp_path: Path, rows: list[dict], name: str = "review.csv") -> Path:
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


# --- the shape this script sends out -------------------------------------------------


def test_an_answer_in_the_reserved_column_is_taken(tmp_path):
    """The intended path, which must keep working."""
    path = _write(tmp_path, [{"cot": "HSSD", "mo_ta_suy_doan_cua_may": infer("HSSD")[1],
                              REVIEW_COLUMN: "Hủy sau sử dụng"}])
    assert load_confirmations(path) == {"HSSD": "Hủy sau sử dụng"}


# --- the shape that actually came back ------------------------------------------------


def test_accented_headers_are_understood(tmp_path):
    """`cot` → `Cột`, `mo_ta_suy_doan_cua_may` → `Mô tả`, and no reserved column at all."""
    path = _write(tmp_path, [{"Cột": "CTBDV", "Mô tả": "Chủ thuê bao đi vắng"}])
    assert load_confirmations(path) == {"CTBDV": "Chủ thuê bao đi vắng"}


def test_an_answer_typed_over_the_guess_counts_as_confirmed(tmp_path):
    path = _write(tmp_path, [{"cot": "LLSD_202606", "mo_ta_suy_doan_cua_may": "Lưu lượng sử dụng tháng 6 năm 2026"}])
    assert load_confirmations(path) == {"LLSD_202606": "Lưu lượng sử dụng tháng 6 năm 2026"}


def test_an_untouched_guess_is_not_mistaken_for_an_answer(tmp_path):
    """Otherwise every unanswered column would be reported as confirmed by the business."""
    guess = infer("downtime_trend")[1]
    path = _write(tmp_path, [{"cot": "downtime_trend", "mo_ta_suy_doan_cua_may": guess}])
    assert load_confirmations(path) == {}


def test_the_reserved_column_wins_over_an_edited_guess(tmp_path):
    """Filling the blank is the least ambiguous thing a reviewer can do."""
    path = _write(tmp_path, [{"cot": "HSSD", "mo_ta_suy_doan_cua_may": "sửa tạm",
                              REVIEW_COLUMN: "Hủy sau sử dụng"}])
    assert load_confirmations(path)["HSSD"] == "Hủy sau sử dụng"


def test_a_deleted_row_simply_has_no_confirmation(tmp_path):
    """They deleted OBJID, as instructed. That must not drop the other 96 answers."""
    path = _write(tmp_path, [{"Cột": "HSSD", "Mô tả": "Hủy sau sử dụng"}])
    confirmed = load_confirmations(path)
    assert "OBJID" not in confirmed
    assert confirmed["HSSD"] == "Hủy sau sử dụng"


def test_blank_cells_are_not_confirmations(tmp_path):
    path = _write(tmp_path, [{"Cột": "segment_avg", "Mô tả": ""},
                             {"Cột": "high_spender", "Mô tả": None}])
    assert load_confirmations(path) == {}


def test_a_file_with_no_recognisable_name_column_yields_nothing(tmp_path):
    """Better to report no confirmations than to key them off the wrong column."""
    path = _write(tmp_path, [{"field": "HSSD", "Mô tả": "Hủy sau sử dụng"}])
    assert load_confirmations(path) == {}


def test_a_missing_file_is_the_first_run(tmp_path):
    assert load_confirmations(tmp_path / "nope.csv") == {}


# --- regrouping, which the reviewers also did ----------------------------------------


def test_a_reassigned_group_is_kept(tmp_path):
    """`branch_*` is not a branch metric; it is LLSD traffic. The machine had it wrong."""
    path = _write(tmp_path, [{"Cột": "branch_avg", "Nhóm": "Mức sử dụng"}])
    assert load_group_overrides(path) == {"branch_avg": "Mức sử dụng"}


def test_an_unchanged_group_is_not_an_override(tmp_path):
    path = _write(tmp_path, [{"cot": "fee_total", "nhom": infer("fee_total")[0]}])
    assert load_group_overrides(path) == {}


def test_group_overrides_survive_a_file_with_no_group_column(tmp_path):
    path = _write(tmp_path, [{"Cột": "branch_avg", "Mô tả": "x"}])
    assert load_group_overrides(path) == {}


# --- the property that makes re-running safe ------------------------------------------


def test_confirmations_survive_a_regeneration(tmp_path):
    """Round two must not quietly revert to the machine's guesses.

    build() writes the confirmed text back into the reserved column and restores the
    machine guess in the description column. Reading that regenerated file has to return
    the same answers, or every re-run would silently lose the business team's work.
    """
    regenerated = _write(tmp_path, [{
        "cot": "HSSD",
        "nhom": "Chưa rõ",
        "mo_ta_suy_doan_cua_may": infer("HSSD")[1],
        REVIEW_COLUMN: "Hủy sau sử dụng",
    }])
    assert load_confirmations(regenerated) == {"HSSD": "Hủy sau sử dụng"}


@pytest.mark.parametrize("column,expected", [
    ("HSSD", "Hủy sau sử dụng"),
    ("CTBDV", "Chủ thuê bao đi vắng"),
    ("total_call_30d", "Tổng số cuộc gọi đi trong 30 ngày"),
    ("branch_avg", "avg phân khúc LLSD 4 tháng gần nhất"),
])
def test_the_real_returned_file_still_parses(column, expected):
    """Guards against a header tweak that silently empties the confirmations again.

    Skips rather than fails when the file is absent — it holds real subscriber column
    names and is not committed.
    """
    returned = Path(__file__).resolve().parents[1] / "Churn_VT_metadata_review - Churn_VT_metadata_review.csv"
    if not returned.exists():
        pytest.skip("bản review nghiệp vụ gửi về không có trong repo")
    assert load_confirmations(returned).get(column) == expected
