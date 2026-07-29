"""On a churned cohort the pipeline may describe what it observes, never why they left.

Stated by the data owner about the real 62,467-row export: *"File này là tất cả KH đã rời
mạng rồi, chỉ nói đặc điểm, không được nói vì sao là nguyên nhân rời."*

That is not a stylistic preference, it is a claim about what the data can support. The file
carries interaction counts, fees and usage trends. It carries nothing about why anyone left —
no exit survey, no cancellation reason, no competitor offer. Every causal sentence the
POST_CHURN ladder produced was therefore an assertion the data could not back:

    'Khách hàng giá trị cao, chủ động rời mạng'
        -> "chủ động" claims intent that was never recorded
    'Không rõ nguyên nhân hành vi (có thể do giá cước/cạnh tranh/khác)'
        -> speculates three causes, and named 6 of 8 clusters on the real file

The second one is also why the segmentation looked meaningless to a reader: six groups with
different statistics all carried one identical label.
"""
import numpy as np
import pandas as pd
import pytest

from triadic_dgm.persona.pipeline import detect_dataset_mode, run_persona_pipeline

#: Wording that asserts a reason for leaving rather than reporting a measurement.
_CAUSAL_PHRASES = (
    "nguyên nhân",
    "chủ động rời",
    "cạnh tranh",
    "đối thủ",
    "giá cước",
    "vì sao",
    "dẫn đến",
)


def _churned_cohort(n=600):
    """A churned base with three genuinely different behavioural shapes.

    Column names are the ones detect_dataset_mode keys on, so this exercises the real
    POST_CHURN path rather than a generic one.
    """
    rng = np.random.default_rng(11)
    third = n // 3
    def block(fee, cl, comp, old_u, rec_u):
        return pd.DataFrame({
            "fee_total": rng.normal(fee, fee * 0.05, third),
            "cl_total_4m": rng.normal(cl, 0.5, third).clip(0),
            "complaint_total_6m": rng.normal(comp, 0.3, third).clip(0),
            "old_usage": rng.normal(old_u, 5, third),
            "recent_usage": rng.normal(rec_u, 5, third),
        })
    return pd.concat([
        block(1500, 0.2, 0.1, 100, 98),   # high value, quiet
        block(300, 9.0, 6.0, 90, 40),     # heavy support contact, usage collapsed
        block(320, 0.3, 0.2, 95, 92),     # unremarkable
    ], ignore_index=True)


@pytest.fixture(scope="module")
def personas():
    df = _churned_cohort()
    assert detect_dataset_mode(list(df.columns)) == "POST_CHURN", "fixture must exercise POST_CHURN"
    return run_persona_pipeline(df)


def test_no_persona_name_claims_a_reason_for_leaving(personas):
    for p in personas:
        low = p["persona_name"].lower()
        for phrase in _CAUSAL_PHRASES:
            assert phrase not in low, f"causal claim in persona_name: {p['persona_name']!r}"


def test_no_evidence_text_claims_a_reason_for_leaving(personas):
    for p in personas:
        blob = " ".join(str(v) for v in (p.get("evidence"), p.get("sample_persona_text"))).lower()
        for phrase in _CAUSAL_PHRASES:
            assert phrase not in blob, f"causal claim in evidence of {p['persona_name']!r}: {blob[:200]}"


def test_clusters_with_different_statistics_get_different_names(personas):
    """Six of eight clusters shared one label on the real file; the reader saw one segment."""
    names = [p["persona_name"] for p in personas if not p["is_anomaly"]]
    assert len(set(names)) == len(names), f"duplicate persona names: {names}"


def _quiet_cohort(n=800):
    """A churned base where NO cluster has a standout interaction signal.

    Every group is quiet on complaints, support calls and technical faults; they differ only
    in fee level and usage. This is the shape of the real 62,467-row file, where 5 of 6
    clusters fell through the whole driver ladder to its final rule.
    """
    rng = np.random.default_rng(5)
    quarter = n // 4
    def block(fee, old_u, rec_u):
        return pd.DataFrame({
            "fee_total": rng.normal(fee, fee * 0.04, quarter),
            "cl_total_4m": rng.normal(0.1, 0.05, quarter).clip(0),
            "complaint_total_6m": rng.normal(0.1, 0.05, quarter).clip(0),
            "old_usage": rng.normal(old_u, 3, quarter),
            "recent_usage": rng.normal(rec_u, 3, quarter),
        })
    return pd.concat([block(200, 60, 58), block(600, 95, 93),
                      block(1100, 130, 128), block(1700, 165, 163)], ignore_index=True)


def test_clusters_with_no_standout_signal_are_told_apart_by_their_numbers():
    """A trailing "- Nhóm 3" is not a distinguishing name.

    When the driver ladder finds no interaction signal it says so — and that is honest — but
    then every such cluster carries one identical label. On the real file the reader saw five
    groups with 10.0%, 9.9%, 50.2%, 3.2% and 3.6% support, all named the same thing. The
    clusters differ measurably (that is why KMeans separated them), so the name must say how.
    """
    personas = run_persona_pipeline(_quiet_cohort())
    names = [p["persona_name"] for p in personas if not p["is_anomaly"]]
    stems = {n.split(" - Nhóm ")[0] for n in names}
    assert len(stems) == len(names), f"names differ only by a trailing number: {names}"
