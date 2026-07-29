"""A flag meaning "this never happened" must not be scored as if it happened a lot.

From a real report on 62,467 churned subscribers. The largest group, 33.7% of the base:

    no_complaint_all_period   1.00   benchmark 0.41   +144.7%
    complaint_total_6m        0.00   benchmark 1.12   -100.0%
    active_complaint_months   0.00   benchmark 0.87   -100.0%

Every direct measure says this group never filed a complaint. The report said:

    Customer Profile: "Có lịch sử khiếu nại/phàn nàn đáng kể"
    Narrative:        "Mặc dù có lịch sử phàn nàn cao hơn..."

Cause: DOMAIN_KEYWORD_GROUPS['complaint'] lists the fragment 'no_complaint', so
`no_complaint_all_period` is a member of the complaint domain — and that domain is scored
"higher = worse", deliberately keeping the sign so that a cluster BELOW average is not
called alarming. The polarity of that one column is inverted relative to its domain, so its
+144.7% read as a strong complaint signal.

The same trap exists for 'technical' via the 'no_cl' fragment.
"""
import numpy as np
import pandas as pd

from triadic_dgm.persona.profiling import compute_domain_signature


def _two_clusters() -> pd.DataFrame:
    """Cluster 0 complains constantly; cluster 1 never complains, and says so via the flag."""
    n = 200
    complainers = pd.DataFrame({
        "complaint_total_6m": np.full(n, 6.0),
        "active_complaint_months": np.full(n, 5.0),
        "no_complaint_all_period": np.zeros(n),
        "cl_total_4m": np.full(n, 4.0),
        "no_cl_all_period": np.zeros(n),
        "cluster": 0,
    })
    quiet = pd.DataFrame({
        "complaint_total_6m": np.zeros(n),
        "active_complaint_months": np.zeros(n),
        "no_complaint_all_period": np.ones(n),
        "cl_total_4m": np.zeros(n),
        "no_cl_all_period": np.ones(n),
        "cluster": 1,
    })
    return pd.concat([complainers, quiet], ignore_index=True)


def _stars(signature, cluster, domain):
    return signature[cluster][domain]["stars"]


def test_a_group_that_never_complained_does_not_score_as_a_complaint_signal():
    sig = compute_domain_signature(_two_clusters())
    assert _stars(sig, 1, "complaint") <= 2, (
        "the 'never complained' flag was read as a complaint signal: "
        f"{sig[1]['complaint']}"
    )


def test_the_group_that_actually_complained_still_scores_highest():
    """The guard against fixing the sign by silencing the domain altogether."""
    sig = compute_domain_signature(_two_clusters())
    assert _stars(sig, 0, "complaint") > _stars(sig, 1, "complaint")


def test_the_same_inversion_is_handled_for_technical_faults():
    sig = compute_domain_signature(_two_clusters())
    assert _stars(sig, 1, "technical") <= 2, (
        f"'no_cl_all_period' was read as a technical-fault signal: {sig[1]['technical']}")
    assert _stars(sig, 0, "technical") > _stars(sig, 1, "technical")
