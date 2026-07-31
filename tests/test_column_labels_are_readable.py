"""A persona name is read by a business owner, so it must not contain a column name.

The real report on the 62,467-row export named its groups:

    Nhóm no_fee_all_period cao
    Nhóm declining_complaint cao
    Nhóm no_complaint_all_period cao
    Nhóm high_spender thấp
    Nhóm fee_std cao

`name_by_top_feature` has always accepted a column -> label map and falls back to the raw
column name; nothing ever passed one, so the fallback WAS the behaviour.

The labels now exist: the business team confirmed 52 of 97 column descriptions. Those are
sentences, not names — "Cờ: không phát sinh cước trong suốt kỳ." — so a short label is
derived from each, and the caveats the reviewers wrote in capitals ("NGƯỠNG PHÂN LOẠI CHƯA
RÕ") are dropped from the label while staying in the description.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_column_metadata import short_label  # noqa: E402


@pytest.mark.parametrize("description,expected", [
    ("Cờ: không phát sinh cước trong suốt kỳ.", "không phát sinh cước trong suốt kỳ"),
    ("Cờ: khiếu nại giảm dần theo thời gian.", "khiếu nại giảm dần theo thời gian"),
    ("Cờ: không có khiếu nại nào trong suốt kỳ.", "không có khiếu nại nào trong suốt kỳ"),
    ("Độ lệch chuẩn cước phí trong 4 tháng", "độ lệch chuẩn cước phí trong 4 tháng"),
    ("Tổng phí khách hàng chi trả trong 4 tháng", "tổng phí khách hàng chi trả trong 4 tháng"),
])
def test_a_description_becomes_a_short_label(description, expected):
    assert short_label(description) == expected


def test_the_flag_prefix_is_dropped():
    """"Cờ" says how the column is encoded, not what it measures."""
    assert not short_label("Cờ khách hàng chi tiêu cao.").startswith("cờ")


def test_a_capitalised_caveat_is_left_out_of_the_label():
    """The reviewers' open questions belong in the description, not in a persona name."""
    label = short_label("Cờ khách hàng chi tiêu cao. NGƯỠNG PHÂN LOẠI CHƯA RÕ.")
    assert label == "khách hàng chi tiêu cao"


def test_only_the_first_clause_survives():
    label = short_label("Giá trị phân khúc trung bình, tính qua các tháng trong kỳ quan sát.")
    assert label == "giá trị phân khúc trung bình"


def test_an_acronym_keeps_its_capitals():
    """"LLSD" is a name, not a capitalised sentence start."""
    assert short_label("LLSD trung bình 4 tháng gần nhất").startswith("LLSD")


def test_a_long_label_is_cut_at_a_word_boundary():
    label = short_label("Chỉ số theo chi nhánh quản lý thuê bao được gộp theo từng khu vực "
                        "hành chính và tính trung bình trong kỳ quan sát")
    assert len(label) <= 60
    assert not label.endswith(" ")
    assert "…" in label or len(label.split()) < 20


def test_an_empty_description_has_no_label():
    assert short_label("") == ""
    assert short_label(None) == ""


def test_a_description_that_is_only_a_caveat_has_no_label():
    """Better no label — the caller falls back to the column name — than a shouted one."""
    assert short_label("VIẾT TẮT CHƯA GIẢI MÃ ĐƯỢC.") == ""


# --- the map the pipeline consumes ------------------------------------------------------


def test_the_generated_metadata_carries_a_label_per_column(tmp_path):
    import json

    import pandas as pd

    from build_column_metadata import build

    csv = tmp_path / "demo.csv"
    pd.DataFrame({"fee_total": [1.0, 2.0], "no_fee_all_period": [0, 1]}).to_csv(csv, index=False)
    json_path, _, _, _ = build(csv)

    columns = {c["column"]: c for c in json.loads(json_path.read_text(encoding="utf-8"))["columns"]}
    assert columns["fee_total"]["label"] == "tổng cước phí trong kỳ quan sát"
    assert columns["no_fee_all_period"]["label"] == "không phát sinh cước trong suốt kỳ"


def test_labels_are_offered_only_for_the_dataset_they_describe(tmp_path):
    """Same schema gate as the metadata injection: a label map for another export is worse
    than none, because it renames columns that happen to share a name."""
    import json

    from api.services.metadata_gate import labels_for_columns

    (tmp_path / "demo_metadata.json").write_text(json.dumps({
        "dataset_name": "demo",
        "columns": [
            {"column": "fee_total", "label": "tổng cước phí", "description": "x"},
            {"column": "no_fee_all_period", "label": "không phát sinh cước", "description": "y"},
            {"column": "cl_total_4m", "label": "số sự cố kỹ thuật", "description": "z"},
            {"column": "complaint_total_6m", "label": "số khiếu nại", "description": "w"},
        ],
    }, ensure_ascii=False), encoding="utf-8")

    matching = ["fee_total", "no_fee_all_period", "cl_total_4m", "complaint_total_6m"]
    assert labels_for_columns(str(tmp_path), matching)["fee_total"] == "tổng cước phí"
    assert labels_for_columns(str(tmp_path), ["a", "b", "c", "price", "qty"]) == {}


def test_a_column_with_no_label_is_simply_absent_from_the_map(tmp_path):
    """Absent means the caller keeps the column name; an empty string would blank the name."""
    import json

    from api.services.metadata_gate import labels_for_columns

    (tmp_path / "demo_metadata.json").write_text(json.dumps({
        "dataset_name": "demo",
        "columns": [
            {"column": "fee_total", "label": "tổng cước phí"},
            {"column": "HSSD", "label": ""},
            {"column": "CTBDV"},
            {"column": "cl_total_4m", "label": "số sự cố"},
        ],
    }, ensure_ascii=False), encoding="utf-8")

    labels = labels_for_columns(str(tmp_path), ["fee_total", "HSSD", "CTBDV", "cl_total_4m"])
    assert labels == {"fee_total": "tổng cước phí", "cl_total_4m": "số sự cố"}


def test_an_unreadable_metadata_file_yields_no_labels(tmp_path):
    """A broken file in the working directory must not take the run down."""
    from api.services.metadata_gate import labels_for_columns

    (tmp_path / "broken_metadata.json").write_text("{not json", encoding="utf-8")
    assert labels_for_columns(str(tmp_path), ["a", "b"]) == {}
