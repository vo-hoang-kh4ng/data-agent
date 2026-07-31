"""A persona name has to survive being read aloud in a meeting.

With human labels wired in, the raw column names are gone but two phrasing defects remain,
and both invert the meaning of the group they name.

A flag is a share, not a level. `no_fee_all_period` is 0 or 1 per subscriber, so the
cluster mean is "what fraction of this group had no billing". Appending "cao" gives

    Nhóm không phát sinh cước trong suốt kỳ cao

which reads as a high quantity of something that is defined by its absence. The group is
simply the one characterised by that flag, so it takes the flag's own words and no
direction at all: "Nhóm không phát sinh cước trong suốt kỳ".

The other direction is worse. `high_spender` deviating LOW produced

    Nhóm high_spender thấp

— naming a group after a property it does NOT have. Being less likely than average to
carry a flag describes almost every group and distinguishes none of them, so a flag
deviating downward is not a naming candidate; the next-strongest feature is used instead.

Continuous features are unaffected: a standard deviation genuinely can be high or low.
"""
import pytest

from triadic_dgm.persona.characterization import name_by_top_feature


def _persona(**means):
    return {"feature_means": dict(means)}


# --- flags -----------------------------------------------------------------------------


def test_a_flag_names_the_group_without_a_direction_word():
    names = name_by_top_feature(
        [_persona(no_fee_all_period=0.52, fee_std=1.0)],
        {"no_fee_all_period": 0.04, "fee_std": 1.0},
        labels={"no_fee_all_period": "không phát sinh cước trong suốt kỳ"},
        binary_features={"no_fee_all_period"},
    )
    assert names[0] == "Nhóm không phát sinh cước trong suốt kỳ"


def test_a_flag_name_never_ends_in_a_direction_word():
    names = name_by_top_feature(
        [_persona(declining_complaint=0.8)],
        {"declining_complaint": 0.1},
        labels={"declining_complaint": "khiếu nại giảm dần theo thời gian"},
        binary_features={"declining_complaint"},
    )
    assert not names[0].endswith(" cao")
    assert not names[0].endswith(" thấp")


def test_a_flag_deviating_downward_is_not_worth_naming_a_group_after():
    """"Nhóm high_spender thấp" named a group for what it lacks."""
    names = name_by_top_feature(
        [_persona(high_spender=0.01, fee_std=5.0)],
        {"high_spender": 0.5, "fee_std": 1.0},
        labels={"high_spender": "khách hàng chi tiêu cao",
                "fee_std": "độ lệch chuẩn cước phí trong 4 tháng"},
        binary_features={"high_spender"},
    )
    assert names[0] == "Nhóm độ lệch chuẩn cước phí trong 4 tháng cao"


def test_a_group_whose_only_signal_is_a_missing_flag_stays_unnamed():
    """None hands the decision back to the caller rather than inventing a name."""
    names = name_by_top_feature(
        [_persona(high_spender=0.01)],
        {"high_spender": 0.5},
        labels={"high_spender": "khách hàng chi tiêu cao"},
        binary_features={"high_spender"},
    )
    assert names[0] is None


# --- continuous features keep their direction -------------------------------------------


def test_a_continuous_feature_keeps_cao():
    names = name_by_top_feature(
        [_persona(fee_std=5.0)], {"fee_std": 1.0},
        labels={"fee_std": "độ lệch chuẩn cước phí trong 4 tháng"},
    )
    assert names[0] == "Nhóm độ lệch chuẩn cước phí trong 4 tháng cao"


def test_a_continuous_feature_keeps_thap():
    names = name_by_top_feature(
        [_persona(fee_avg=0.2)], {"fee_avg": 1.0},
        labels={"fee_avg": "cước phí trung bình trong 4 tháng"},
    )
    assert names[0] == "Nhóm cước phí trung bình trong 4 tháng thấp"


def test_declaring_no_binary_features_keeps_the_old_phrasing():
    """Callers that cannot tell which columns are flags must not silently change output."""
    names = name_by_top_feature(
        [_persona(no_fee_all_period=0.52)], {"no_fee_all_period": 0.04},
        labels={"no_fee_all_period": "không phát sinh cước trong suốt kỳ"},
    )
    assert names[0] == "Nhóm không phát sinh cước trong suốt kỳ cao"


# --- labels ------------------------------------------------------------------------------


def test_a_column_name_is_used_only_when_no_label_exists():
    names = name_by_top_feature([_persona(fee_std=5.0)], {"fee_std": 1.0})
    assert names[0] == "Nhóm fee_std cao"


def test_no_persona_name_contains_a_raw_column_name_when_labels_exist():
    """The defect as a reader would state it."""
    personas = [
        _persona(no_fee_all_period=0.52, fee_std=1.0),
        _persona(declining_complaint=0.80, fee_std=1.0),
        _persona(fee_std=6.0, no_fee_all_period=0.04),
    ]
    labels = {
        "no_fee_all_period": "không phát sinh cước trong suốt kỳ",
        "declining_complaint": "khiếu nại giảm dần theo thời gian",
        "fee_std": "độ lệch chuẩn cước phí trong 4 tháng",
    }
    names = name_by_top_feature(
        personas, {"no_fee_all_period": 0.04, "declining_complaint": 0.1, "fee_std": 1.0},
        labels=labels, binary_features={"no_fee_all_period", "declining_complaint"},
    )
    for name in names:
        assert name
        for column in labels:
            assert column not in name


# --- the wiring --------------------------------------------------------------------------


def test_the_pipeline_detects_which_columns_are_flags():
    from triadic_dgm.persona.pipeline import binary_features

    import pandas as pd

    frame = pd.DataFrame({
        "no_fee_all_period": [0, 1, 0, 1],
        "high_spender": [1.0, 0.0, 1.0, 0.0],
        "fee_std": [1.0, 2.0, 3.0, 4.0],
        "active_fee_months": [0, 1, 2, 3],
    })
    found = binary_features(frame, list(frame.columns))
    assert found == {"no_fee_all_period", "high_spender"}


def test_a_constant_column_is_not_reported_as_a_flag():
    from triadic_dgm.persona.pipeline import binary_features

    import pandas as pd

    frame = pd.DataFrame({"always_one": [1, 1, 1], "real": [0.0, 1.0, 2.0]})
    assert binary_features(frame, list(frame.columns)) == set()


def test_personas_from_a_real_run_carry_no_column_names(tmp_path):
    """End to end: labels from a metadata file the gate accepts reach the persona names."""
    import json

    import numpy as np
    import pandas as pd

    from triadic_dgm.persona.pipeline import run_persona_pipeline

    rng = np.random.default_rng(21)
    third = 100
    blob = lambda means: np.concatenate([rng.normal(m, 0.4, third) for m in means])
    frame = pd.DataFrame({
        "fee_std": blob((0.0, 6.0, 12.0)),
        "fee_avg": blob((0.0, 6.0, 12.0)),
        "cl_total_4m": blob((12.0, 6.0, 0.0)),
        "complaint_total_6m": blob((12.0, 6.0, 0.0)),
    })
    (tmp_path / "demo_metadata.json").write_text(json.dumps({
        "dataset_name": "demo",
        "columns": [
            {"column": "fee_std", "label": "độ lệch chuẩn cước phí"},
            {"column": "fee_avg", "label": "cước phí trung bình"},
            {"column": "cl_total_4m", "label": "số sự cố kỹ thuật"},
            {"column": "complaint_total_6m", "label": "số khiếu nại"},
        ],
    }, ensure_ascii=False), encoding="utf-8")

    personas = run_persona_pipeline(frame, label_dir=str(tmp_path))
    assert personas
    for persona in personas:
        for column in frame.columns:
            assert column not in persona["persona_name"], persona["persona_name"]
