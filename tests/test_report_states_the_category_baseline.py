"""Printing a group's region share without the file's own share invents a finding.

"18,6% nhóm này ở Hà Nội" reads as concentration. Ha Noi is 16,6% of the whole export, so
the true statement is that this group is distributed almost exactly like everyone else.
Across the six national personas the strongest concentration in the large groups runs at
1,1–1,6× the national share; region explains 2,51% of behavioural variance against a 0,10%
noise floor, and Cramér's V between region and cluster is 0,10.

The renderer therefore never prints a share alone, and when nothing departs from the file
it says so in words — a reader who skims the numbers and stops must still come away with
the right conclusion.
"""
import pytest

from triadic_dgm.services.report_generator import ReportGenerator, format_category_mix


@pytest.fixture
def generator():
    return ReportGenerator(api_key="x", base_url="http://localhost:1", model_name="m")


def _entry(value, share, dataset_share, rows=500):
    return {
        "value": value, "rows": rows, "share": share, "dataset_share": dataset_share,
        "lift": (share / dataset_share) if dataset_share else None,
    }


# --- the baseline is never omitted --------------------------------------------------------


def test_every_share_is_printed_with_the_share_it_should_be_read_against():
    line = format_category_mix("Khu vực", [
        _entry("Ha Noi", 0.186, 0.1655),
        _entry("Ho Chi Minh", 0.168, 0.1733),
    ])
    assert "18,6%" in line
    assert "16,6%" in line
    assert "16,8%" in line
    assert "17,3%" in line


def test_a_group_spread_like_the_file_says_so_in_words():
    """The number a skimming reader takes away has to carry its own interpretation."""
    line = format_category_mix("Khu vực", [
        _entry("Ha Noi", 0.186, 0.1655),
        _entry("Ho Chi Minh", 0.168, 0.1733),
    ])
    assert "gần như mặt bằng chung" in line


def test_a_real_concentration_is_not_dismissed():
    line = format_category_mix("Khu vực", [_entry("Da Nang", 0.44, 0.0219)])
    assert "gần như mặt bằng chung" not in line
    assert "20,1 lần" in line


def test_the_column_label_names_the_line():
    assert format_category_mix("Khu vực", [_entry("HUE", 0.5, 0.5)]).startswith("Khu vực:")


# --- what the floor hid -------------------------------------------------------------------


def test_the_suppressed_remainder_is_reported_as_a_count_not_a_name():
    line = format_category_mix("Khu vực", [
        _entry("Ha Noi", 0.9, 0.9, rows=900),
        {"value": None, "rows": 100, "share": 0.1, "dataset_share": None, "lift": None},
    ])
    assert "còn lại" in line
    assert "10,0%" in line
    assert "None" not in line


def test_a_value_missing_from_the_file_prints_no_multiple():
    line = format_category_mix("Khu vực", [_entry("X", 1.0, 0.0)])
    assert "lần" not in line


def test_nothing_to_say_produces_no_line():
    assert format_category_mix("Khu vực", []) == ""
    assert format_category_mix("Khu vực", None) == ""


# --- reaching the document ------------------------------------------------------------------


_PERSONAS = [{
    "cluster_id": 0,
    "persona_name": "Nhóm không có khiếu nại nào trong suốt kỳ",
    "support": 22666,
    "support_pct": 0.363,
    "feature_means": {"old_cl": 1.2},
    "feature_coverage": {"old_cl": 0.8},
    "category_mix": {"LOCATIONNAME": [
        _entry("Ha Noi", 0.186, 0.1655, rows=4216),
        _entry("Ho Chi Minh", 0.168, 0.1733, rows=3808),
    ]},
    "evidence": {"old_cl": 1.2},
    "persona_type": "MAINSTREAM",
    "segmentation_quality": "WEAK",
}]


def test_the_persona_section_carries_the_breakdown(generator):
    md = generator._render_persona_sections(_PERSONAS, {"old_cl": 0.6}, {}, {})
    assert "18,6%" in md
    assert "16,6%" in md


def test_the_business_label_is_used_rather_than_the_raw_column(generator):
    md = generator._render_persona_sections(_PERSONAS, {"old_cl": 0.6}, {},
                                            {"LOCATIONNAME": "Khu vực"})
    assert "Khu vực:" in md
    assert "LOCATIONNAME" not in md


def test_the_raw_column_name_is_the_fallback(generator):
    """Better an unlovely header than a silently dropped section."""
    md = generator._render_persona_sections(_PERSONAS, {"old_cl": 0.6}, {}, {})
    assert "LOCATIONNAME:" in md


def test_a_persona_without_a_breakdown_still_renders(generator):
    persona = dict(_PERSONAS[0])
    del persona["category_mix"]
    assert generator._render_persona_sections([persona], {"old_cl": 0.6}, {}, {})
