"""HSSD and CTBDV mean something only for Churn_VT, so they are decoded only here.

The pipeline counts whatever status column it is handed and has no idea what the values
mean. That is deliberate — this branch exists to get telco column names out of the generic
path. Deriving a status from two Viettel cancellation codes is dataset knowledge, so it
lives in a preparation step the caller runs first.

What the data owner confirmed about the 62,467-row export:

    HSSD  = 1   Hủy sau sử dụng          30,263
    CTBDV = 1   Chủ thuê bao đi vắng     27,671
    both  = 0   đã khôi phục dịch vụ      4,533

and the crosstab shows the two codes are mutually exclusive — no row carries both. That
exclusivity is the assumption the whole derivation rests on, so it is checked rather than
trusted: if a future export breaks it, the run must stop instead of silently classifying
those rows as restored.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_churn_vt import RESTORED, derive_status  # noqa: E402


def _frame(hssd, ctbdv):
    return pd.DataFrame({"HSSD": hssd, "CTBDV": ctbdv})


def test_a_cancel_after_use_row_is_labelled_by_its_code():
    assert derive_status(_frame([1], [0])).tolist() == ["Hủy sau sử dụng"]


def test_a_subscriber_away_row_is_labelled_by_its_code():
    assert derive_status(_frame([0], [1])).tolist() == ["Chủ thuê bao đi vắng"]


def test_a_row_with_neither_code_is_a_restored_subscriber():
    """The 4,533 the report was counting as departed."""
    assert derive_status(_frame([0], [0])).tolist() == [RESTORED]


def test_the_real_proportions_come_out():
    frame = _frame([1] * 30_263 + [0] * 32_204, [0] * 30_263 + [1] * 27_671 + [0] * 4_533)
    counts = derive_status(frame).value_counts()
    assert counts["Hủy sau sử dụng"] == 30_263
    assert counts["Chủ thuê bao đi vắng"] == 27_671
    assert counts[RESTORED] == 4_533


def test_a_row_carrying_both_codes_stops_the_run():
    """Exclusivity is the assumption everything else rests on."""
    with pytest.raises(ValueError, match="loại trừ"):
        derive_status(_frame([1], [1]))


def test_a_missing_code_column_stops_the_run():
    """Better than labelling the entire export as restored."""
    with pytest.raises(ValueError, match="HSSD"):
        derive_status(pd.DataFrame({"CTBDV": [0, 1]}))


def test_absent_codes_are_not_read_as_zero():
    """NaN is not evidence that a subscriber restored service."""
    frame = pd.DataFrame({"HSSD": [1.0, None], "CTBDV": [0.0, None]})
    assert derive_status(frame).tolist()[1] is None or pd.isna(derive_status(frame).iloc[1])


def test_the_restored_label_is_what_the_pipeline_is_told_is_active():
    """The two constants must not drift apart — the caller passes one into the other."""
    from prepare_churn_vt import ACTIVE_STATUS_VALUES

    assert ACTIVE_STATUS_VALUES == {RESTORED}


def test_the_derived_column_measures_the_cohort_the_dashboard_reports():
    """End to end: derivation -> cohort_mix -> the number that replaced the hardcoded 100%."""
    from triadic_dgm.persona.pipeline import cohort_mix

    frame = _frame([1] * 30_263 + [0] * 32_204, [0] * 30_263 + [1] * 27_671 + [0] * 4_533)
    mix = cohort_mix(derive_status(frame))
    assert mix[RESTORED] == pytest.approx(0.0726, abs=5e-5)
    assert sum(v for k, v in mix.items() if k != RESTORED) == pytest.approx(0.9274, abs=5e-5)
