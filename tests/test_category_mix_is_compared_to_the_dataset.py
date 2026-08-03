"""A group's region breakdown means nothing without the dataset's own breakdown beside it.

`LOCATIONNAME` is now part of each persona's description. The trap is that the plain share
reads as a finding: "18,6% of this group is in Ha Noi" sounds like Ha Noi is over-represented
until you notice Ha Noi is 16,6% of the whole file. Measured across the six national
personas, the strongest region concentration in the largest groups runs at 1,1–1,6× the
national share, and region explains 2,51% of behavioural variance against a 0,10% noise
floor from random labels of the same shape. Cramér's V between region and cluster is 0,10.

So every share is carried with the dataset share it should be read against. This is the
same rule feature_coverage follows: a group is characterised by DEPARTING from what is
normal across the file, not by containing a lot of whatever is common everywhere.

Two further constraints come from the file itself:

  * The smallest region has 27 rows in all of 62.467. Split across six personas that is
    four or five people, and a report line naming a region on four churned customers is
    both noise and close to identifying. Cells below a floor are counted, not named.

  * 28 of the 61 values are compound ("Ho Chi Minh - Binh Duong") and every one of them
    has a standalone twin ("Ho Chi Minh"). Whether the compound rolls up into the
    standalone is question C5 to the data owner and is not yet answered, so values are
    reported exactly as written. Guessing would move TP.HCM between 17,3% and 27,3%.
"""
import numpy as np
import pandas as pd
import pytest

from triadic_dgm.persona.pipeline import category_mix, run_persona_pipeline


# --- the measure ----------------------------------------------------------------------


@pytest.fixture
def national():
    """Ha Noi is 40% of the file; a group at 40% Ha Noi is unremarkable."""
    return pd.Series(["Ha Noi"] * 40 + ["Da Nang"] * 35 + ["HUE"] * 25)


def test_a_share_is_carried_with_the_share_it_should_be_read_against(national):
    group = pd.Series(["Ha Noi"] * 50 + ["Da Nang"] * 50)
    entries = category_mix(group, national, min_rows=1)
    ha_noi = next(e for e in entries if e["value"] == "Ha Noi")
    assert ha_noi["share"] == pytest.approx(0.5)
    assert ha_noi["dataset_share"] == pytest.approx(0.4)
    assert ha_noi["lift"] == pytest.approx(1.25)


def test_a_group_that_matches_the_dataset_lifts_nothing(national):
    group = pd.Series(["Ha Noi"] * 40 + ["Da Nang"] * 35 + ["HUE"] * 25)
    for entry in category_mix(group, national, min_rows=1):
        assert entry["lift"] == pytest.approx(1.0)


def test_entries_come_back_biggest_share_first(national):
    group = pd.Series(["HUE"] * 60 + ["Ha Noi"] * 30 + ["Da Nang"] * 10)
    assert [e["value"] for e in category_mix(group, national, min_rows=1)] == \
        ["HUE", "Ha Noi", "Da Nang"]


def test_the_row_count_is_reported_not_only_the_share(national):
    """A reader has to be able to see how many people a percentage stands on."""
    group = pd.Series(["Ha Noi"] * 7 + ["HUE"] * 3)
    ha_noi = next(e for e in category_mix(group, national, min_rows=1) if e["value"] == "Ha Noi")
    assert ha_noi["rows"] == 7


# --- small cells ------------------------------------------------------------------------


def test_a_cell_below_the_floor_is_not_named(national):
    """`Tuyen Quang - Ha Giang` has 27 rows nationally, four or five inside a persona."""
    group = pd.Series(["Ha Noi"] * 200 + ["HUE"] * 4)
    entries = category_mix(group, national, min_rows=30)
    assert [e["value"] for e in entries if e["value"] is not None] == ["Ha Noi"]
    # The four are still in the group; they arrive as the unnamed remainder below.
    assert sum(e["rows"] for e in entries) == 204


def test_what_the_floor_removed_is_still_counted(national):
    """Suppressing a NAME is not the same as dropping the people from the group."""
    group = pd.Series(["Ha Noi"] * 200 + ["HUE"] * 4 + ["Da Nang"] * 6)
    entries = category_mix(group, national, min_rows=30)
    other = next(e for e in entries if e["value"] is None)
    assert other["rows"] == 10
    assert other["share"] == pytest.approx(10 / 210)


def test_nothing_above_the_floor_leaves_only_the_remainder(national):
    group = pd.Series(["Ha Noi"] * 5 + ["HUE"] * 5)
    entries = category_mix(group, national, min_rows=30)
    assert [e["value"] for e in entries] == [None]
    assert entries[0]["rows"] == 10


def test_no_remainder_entry_when_nothing_was_suppressed(national):
    group = pd.Series(["Ha Noi"] * 50 + ["HUE"] * 50)
    assert all(e["value"] is not None for e in category_mix(group, national, min_rows=30))


def test_only_the_top_values_are_named(national):
    group = pd.Series(sum(([f"R{i}"] * 40 for i in range(9)), []))
    entries = category_mix(group, pd.Series(group), min_rows=1, top_n=3)
    assert len([e for e in entries if e["value"] is not None]) == 3
    assert entries[-1]["value"] is None
    assert entries[-1]["rows"] == 6 * 40


# --- values are reported as written -------------------------------------------------------


def test_a_compound_name_is_not_folded_into_its_standalone_twin():
    """Question C5 is open; folding would move TP.HCM between 17,3% and 27,3% of the file."""
    national = pd.Series(["Ho Chi Minh"] * 60 + ["Ho Chi Minh - Binh Duong"] * 40)
    entries = category_mix(national, national, min_rows=1)
    assert {e["value"] for e in entries} == {"Ho Chi Minh", "Ho Chi Minh - Binh Duong"}


def test_case_is_left_alone():
    """`HUE` is upper case where the other 60 values are not — also a question, not a fix."""
    national = pd.Series(["HUE"] * 50 + ["Hue"] * 50)
    assert {e["value"] for e in category_mix(national, national, min_rows=1)} == {"HUE", "Hue"}


# --- edges -------------------------------------------------------------------------------


def test_an_empty_group_measures_nothing(national):
    assert category_mix(pd.Series([], dtype=object), national) == []


def test_a_value_absent_from_the_dataset_series_reports_no_lift():
    entries = category_mix(pd.Series(["X"] * 50), pd.Series(["Y"] * 50), min_rows=1)
    assert entries[0]["dataset_share"] == 0.0
    assert entries[0]["lift"] is None


def test_blanks_are_not_a_region(national):
    group = pd.Series(["Ha Noi"] * 50 + [None] * 50)
    entries = category_mix(group, national, min_rows=1)
    assert [e["value"] for e in entries] == ["Ha Noi"]
    assert entries[0]["share"] == pytest.approx(1.0)


# --- reaching the persona -----------------------------------------------------------------


def _frame_with_region(n_per_blob: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    blob = lambda means: np.concatenate([rng.normal(m, 0.4, n_per_blob) for m in means])
    frame = pd.DataFrame({
        "old_fee": blob((0.0, 6.0, 12.0)),
        "recent_fee": blob((0.0, 6.0, 12.0)),
        "old_cl": blob((12.0, 6.0, 0.0)),
        "recent_cl": blob((12.0, 6.0, 0.0)),
    })
    regions = ["Ha Noi", "Ho Chi Minh", "Da Nang", "HUE"]
    frame["LOCATIONNAME"] = [regions[i % 4] for i in range(len(frame))]
    return frame


def test_every_persona_carries_its_own_breakdown(tmp_path):
    frame = _frame_with_region()
    personas = run_persona_pipeline(frame, label_dir=str(tmp_path))
    assert personas
    for persona in personas:
        mix = persona["category_mix"]
        assert "LOCATIONNAME" in mix
        entries = mix["LOCATIONNAME"]
        assert entries
        assert abs(sum(e["share"] for e in entries) - 1.0) < 1e-6


def test_the_region_column_is_still_kept_out_of_the_clustering(tmp_path):
    """Describing a group by region and segmenting on it are different things."""
    personas = run_persona_pipeline(_frame_with_region(), label_dir=str(tmp_path))
    for persona in personas:
        assert "LOCATIONNAME" not in persona["features_used"]


def test_a_dataset_with_no_category_column_carries_an_empty_map(tmp_path):
    frame = _frame_with_region()
    del frame["LOCATIONNAME"]
    for persona in run_persona_pipeline(frame, label_dir=str(tmp_path)):
        assert persona["category_mix"] == {}


def test_a_boolean_flag_is_not_described_as_a_category(tmp_path):
    """Found by running the real file, not by any unit test here.

    `CHECKLIST_DUPLICATED_202606` and three siblings arrive as object dtype holding
    True/False. Nothing about them is a category: a two-value breakdown restates the mean
    the feature statistics already print, and four such lines landed on every persona:

        số checklist lặp ... tháng 6: False 98,2% (toàn tập 96,8%) · True 1,8% (3,2%)

    The test is the same one is_nominal() applies — whether the values READ as numbers —
    so the two ends of this mechanism cannot drift apart.
    """
    frame = _frame_with_region()
    # Object dtype, not bool: this is what `pd.read_csv` hands back for the real column,
    # which is exactly why is_numeric_dtype() alone let it through.
    frame["CHECKLIST_DUPLICATED_202606"] = pd.Series(
        [True if i % 2 == 0 else False for i in range(len(frame))], dtype=object)
    assert not pd.api.types.is_numeric_dtype(frame["CHECKLIST_DUPLICATED_202606"])
    for persona in run_persona_pipeline(frame, label_dir=str(tmp_path)):
        assert "CHECKLIST_DUPLICATED_202606" not in persona["category_mix"]


def test_numbers_stored_as_text_are_not_a_category(tmp_path):
    frame = _frame_with_region()
    frame["code"] = [str(i % 7) for i in range(len(frame))]
    for persona in run_persona_pipeline(frame, label_dir=str(tmp_path)):
        assert "code" not in persona["category_mix"]


def test_a_declared_category_is_described_even_when_it_reads_as_numbers():
    """A region encoded to integers is still a region — the declaration outranks the shape."""
    from triadic_dgm.persona.pipeline import describable_categories

    frame = pd.DataFrame({"REGION": [0, 1, 2, 0, 1, 2], "fee": [1.0, 2, 3, 4, 5, 6]})
    assert describable_categories(frame, declared={"REGION"}) == ["REGION"]
    assert describable_categories(frame) == []


def test_an_identifier_is_not_described_as_a_category(tmp_path):
    """A column with one value per row would produce a breakdown of one row per entry."""
    frame = _frame_with_region()
    frame["OBJID"] = [f"id-{i}" for i in range(len(frame))]
    for persona in run_persona_pipeline(frame, label_dir=str(tmp_path)):
        assert "OBJID" not in persona["category_mix"]
