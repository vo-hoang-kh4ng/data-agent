"""A region code is a name, not a quantity, and KMeans cannot tell the difference.

`RM_add_location.csv` added `LOCATIONNAME`, the customer's area: 61 values, no blanks,
"Ha Noi", "Ho Chi Minh - Binh Duong", "Tuyen Quang - Ha Giang".

As text it is harmless — `usable_features` already refuses a non-numeric column and
`_auto_features` only looks at numeric ones. The hole is one step upstream. The script that
prepares the frame is written by an LLM on every run, and encoding categorical columns is
the single most standard thing anyone does before clustering. Measured:

    LOCATIONNAME as text     -> {'LOCATIONNAME': 'không phải kiểu số'}
    after pd.factorize()     -> kept as a feature

Encoded, it is 61 dense integer codes: it varies, it is nowhere near constant, and it is
not a row identifier, so every guard on this branch passes it through. What arrives at
StandardScaler is an axis on which "Ha Noi"(17) sits 5 units from "Ho Chi Minh"(22) and
39 units from "Cao Bang"(56) — distances that mean nothing, weighted equally with fee and
complaint counts.

Two layers, because the declaration is authoritative and the fingerprint is a net:

  1. The metadata records which source columns are text. That is measured off the RAW file
     before any preprocessing touches it, so it survives the encoding.
  2. Where no metadata applies, a dense 0..n-1 integer column with many codes is refused on
     sight. The real 62,467-row export's widest genuine count column has 11 such codes and
     the encoded region has 61, so the threshold sits in open space between them.
"""
import json

import numpy as np
import pandas as pd
import pytest

from api.services.metadata_gate import nominal_columns
from triadic_dgm.persona.pipeline import (
    looks_like_encoded_category,
    run_persona_pipeline,
    usable_features,
)


# --- the fingerprint --------------------------------------------------------------------


def test_an_encoded_region_column_is_recognised():
    """61 regions, factorised. The shape this test file exists for."""
    regions = pd.Series([f"region {i % 61}" for i in range(62_467)])
    assert looks_like_encoded_category(pd.Series(pd.factorize(regions)[0]))


def test_a_month_count_is_not_mistaken_for_a_category():
    """`positive_months` is 0..6 and genuinely ordinal — 5 months IS more than 2."""
    assert not looks_like_encoded_category(pd.Series([0, 1, 2, 3, 4, 5, 6] * 100))


def test_the_widest_real_count_column_survives():
    """`recent_complaint` holds 11 dense codes, the most of any real column in the export."""
    assert not looks_like_encoded_category(pd.Series(list(range(11)) * 500))


def test_a_flag_is_not_a_category():
    assert not looks_like_encoded_category(pd.Series([0, 1] * 500))


def test_a_continuous_measurement_is_not_a_category():
    assert not looks_like_encoded_category(pd.Series(np.linspace(0.0, 900.0, 5000)))


def test_codes_that_do_not_start_at_zero_are_left_alone():
    """A year, an age, a month number — integers with a floor, not an encoding."""
    assert not looks_like_encoded_category(pd.Series(list(range(1990, 2026)) * 50))


def test_a_sparse_integer_column_is_left_alone():
    """Label encoding is dense by construction; a count with gaps is not one."""
    assert not looks_like_encoded_category(pd.Series([0, 5, 40, 900, 3000] * 200))


def test_an_empty_column_is_not_a_category():
    assert not looks_like_encoded_category(pd.Series([], dtype=float))


def test_the_threshold_is_the_boundary_it_claims_to_be():
    under = pd.Series(list(range(19)) * 100)
    at = pd.Series(list(range(20)) * 100)
    assert not looks_like_encoded_category(under)
    assert looks_like_encoded_category(at)


# --- refusing it as a feature ------------------------------------------------------------


def _frame_with_encoded_region(n: int = 2_000) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    return pd.DataFrame({
        "fee_avg": rng.normal(200.0, 40.0, n),
        "old_cl": rng.integers(0, 8, n).astype(float),
        "LOCATIONNAME": rng.integers(0, 61, n),
    })


def test_an_encoded_category_is_kept_out_of_the_feature_list():
    kept, dropped = usable_features(_frame_with_encoded_region(), ["fee_avg", "old_cl", "LOCATIONNAME"])
    assert "LOCATIONNAME" not in kept
    assert dropped["LOCATIONNAME"] == "danh mục đã mã hoá"
    assert kept == ["fee_avg", "old_cl"]


def test_the_pipeline_does_not_choose_one_for_itself():
    personas = run_persona_pipeline(_frame_with_encoded_region())
    assert personas
    for persona in personas:
        assert "LOCATIONNAME" not in persona["features_used"]


def test_a_row_identifier_keeps_its_own_reason():
    """`range(n)` carries BOTH fingerprints — dense from zero, and distinct on every row.

    Found by the existing suite when the category check ran first and `ROW_ID` started
    coming back as "danh mục đã mã hoá". The column was still refused either way, but
    "định danh" is the more precise statement: a label encoding repeats its codes.
    """
    frame = _frame_with_encoded_region(500)
    frame["ROW_ID"] = range(len(frame))
    _, dropped = usable_features(frame, ["fee_avg", "ROW_ID", "LOCATIONNAME"])
    assert dropped["ROW_ID"] == "định danh"
    assert dropped["LOCATIONNAME"] == "danh mục đã mã hoá"


def test_text_is_still_refused_for_being_text():
    """The older reason is more precise; the new one must not swallow it."""
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "LOCATIONNAME": ["Ha Noi", "HUE", "Da Nang"]})
    _, dropped = usable_features(frame, ["x", "LOCATIONNAME"])
    assert dropped["LOCATIONNAME"] == "không phải kiểu số"


# --- the declaration, which outranks the fingerprint --------------------------------------


def _metadata(tmp_path, columns, **entry_overrides):
    entries = []
    for name in columns:
        entry = {"column": name, "type": "float64"}
        entry.update(entry_overrides.get(name, {}))
        entries.append(entry)
    (tmp_path / "x_metadata.json").write_text(
        json.dumps({"dataset_name": "x", "columns": entries}, ensure_ascii=False),
        encoding="utf-8",
    )
    return str(tmp_path)


def test_a_column_declared_text_in_the_source_is_served(tmp_path):
    columns = [f"c{i}" for i in range(9)] + ["LOCATIONNAME"]
    search_dir = _metadata(tmp_path, columns, LOCATIONNAME={"type": "object", "nominal": True})
    assert nominal_columns(search_dir, columns) == {"LOCATIONNAME"}


def test_a_declaration_about_another_export_does_not_apply_here(tmp_path):
    """Gated exactly like the label and blank-means-zero declarations."""
    search_dir = _metadata(
        tmp_path, [f"c{i}" for i in range(9)] + ["LOCATIONNAME"],
        LOCATIONNAME={"type": "object", "nominal": True},
    )
    assert nominal_columns(search_dir, [f"other_{i}" for i in range(10)]) == set()


def test_no_metadata_serves_nothing(tmp_path):
    assert nominal_columns(str(tmp_path), ["a", "b"]) == set()


def test_an_unreadable_metadata_file_never_raises(tmp_path):
    (tmp_path / "broken_metadata.json").write_text("{not json", encoding="utf-8")
    assert nominal_columns(str(tmp_path), ["a", "b"]) == set()


def test_a_declared_category_is_refused_even_with_few_codes():
    """The declaration is authoritative — it catches what the fingerprint cannot see.

    A 6-region encoding is indistinguishable from a month count by shape alone. The source
    dtype settles it, and nothing about the encoding can erase that record.
    """
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({
        "fee_avg": rng.normal(200.0, 40.0, 500),
        "REGION": rng.integers(0, 6, 500),
    })
    kept, dropped = usable_features(frame, ["fee_avg", "REGION"], nominal={"REGION"})
    assert kept == ["fee_avg"]
    assert dropped["REGION"] == "danh mục đã mã hoá"


# --- what the metadata build records ------------------------------------------------------


def test_the_build_marks_a_text_column_nominal(tmp_path):
    from scripts.build_column_metadata import build

    csv = tmp_path / "src.csv"
    pd.DataFrame({
        "fee_avg": [1.0, 2.0, 3.0],
        "LOCATIONNAME": ["Ha Noi", "HUE", "Da Nang"],
    }).to_csv(csv, index=False)
    json_path, _, _, _ = build(csv)
    entries = {c["column"]: c for c in json.loads(json_path.read_text(encoding="utf-8"))["columns"]}
    assert entries["LOCATIONNAME"]["nominal"] is True
    assert entries["fee_avg"]["nominal"] is False


def test_a_boolean_column_is_not_marked_nominal(tmp_path):
    """`CHECKLIST_DUPLICATED_202606` reads as object dtype and holds True/False.

    Marking it nominal would delete a real behavioural flag from the segmentation, so the
    test is whether the values are readable as numbers — not what dtype pandas guessed.
    """
    from scripts.build_column_metadata import build

    csv = tmp_path / "src.csv"
    pd.DataFrame({
        "fee_avg": [1.0, 2.0, 3.0],
        "CHECKLIST_DUPLICATED_202606": [True, False, True],
    }).to_csv(csv, index=False)
    json_path, _, _, _ = build(csv)
    entries = {c["column"]: c for c in json.loads(json_path.read_text(encoding="utf-8"))["columns"]}
    assert entries["CHECKLIST_DUPLICATED_202606"]["nominal"] is False


@pytest.mark.parametrize("values", [
    ["1", "2", "3"],          # numbers that arrived as strings
    ["1.5", "2.5", "3.5"],
])
def test_numbers_stored_as_text_are_not_nominal(tmp_path, values):
    from scripts.build_column_metadata import build

    csv = tmp_path / "src.csv"
    pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": values}).to_csv(csv, index=False)
    json_path, _, _, _ = build(csv)
    entries = {c["column"]: c for c in json.loads(json_path.read_text(encoding="utf-8"))["columns"]}
    assert entries["b"]["nominal"] is False
