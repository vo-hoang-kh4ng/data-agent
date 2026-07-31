"""One fixed entry point for the persona pipeline's core.

Everything from "pick features" to "emit persona dicts" used to live as prose + code inside
``PROGRAMMER_PROMPT_V2``, retyped by the sandbox LLM on every run. That made the core
non-deterministic (same data, different code), untestable (it was a string), and fragile
under repair: any error sent the loop rewriting the script from memory, drifting further
each retry until the retries ran out and the user got no report at all.

This module is the "fixed core" half of that split. The LLM no longer writes clustering or
rule-engine code; it calls :func:`run_persona_pipeline` and keeps doing what actually needs
language — interpreting and narrating the result. It may still write extra code around this
call for a request the default path does not cover.

The rule-engine, profiling and Stage-2 functions this orchestrates were ported verbatim, so
the telco path is unchanged. What is new is that the ORDER of steps is now code instead of
instructions the model may reorder, skip or half-apply.
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from triadic_dgm.persona.characterization import name_by_top_feature
from triadic_dgm.persona.clustering import try_substage_cluster
from triadic_dgm.persona.profiling import (
    NO_STANDOUT_SIGNAL,
    compute_churn_drivers,
    compute_domain_signature,
    compute_profile_attributes,
    compute_profile_global_means,
)
from triadic_dgm.persona.rules import apply_business_rules, classify_risk_tier, generate_actions

SEED = 42
_K_RANGE = (3, 7)
_SILHOUETTE_SAMPLE = 5000

#: Above this share in one cluster, the split carries no information and the run is reported
#: as failed rather than dressed up as three personas that are really one.
_HARD_STOP_DOMINANT = 0.8
#: Above this share, Stage-2 is attempted on the dominant cluster first.
_STAGE2_TRIGGER = 0.5



#: Above this share of zeros across the WHOLE selected matrix, clustering has nothing to
#: separate. Checked on the aggregate only: behavioural data legitimately has individual
#: columns that are 90-99% zero, and treating one sparse column as grounds to abort was a
#: real failure mode.
_MAX_ZERO_FRACTION = 0.99

#: Share of absent values above which an UNDECLARED column is dropped rather than imputed.
#: Past half a column, no statistic is describing the data any more — it is describing the
#: filler. See resolve_missing(); a column the data owner has confirmed is event-recorded
#: is exempt, because there a gap is a real zero rather than a missing measurement.
_MAX_ABSENT_FRACTION = 0.5

#: Share the commonest value may occupy before a column is dropped as near-constant. See
#: is_near_constant() — the line sits at a 0.5% minority, which on the real export removes
#: the columns differing on 2, 157 and 297 rows while keeping the one differing on 567.
_MAX_MODAL_FRACTION = 0.995

#: Absolute correlation at which two columns are treated as the same measurement, and the
#: group is down-weighted so it votes once. See redundancy_weights().
_REDUNDANCY_THRESHOLD = 0.95

#: Name for a GENERIC cluster that deviates on nothing. States what was measured and
#: assumes nothing about what the dataset describes.
_NEAR_MEAN_NAME = "Nhóm gần trung bình toàn tập"


def _sample_persona_text(name: str, means: dict, global_mean: dict, top_n: int = 3) -> str:
    """One Vietnamese sentence describing a cluster's standout features.

    Consumed by SemanticVerifier (triadic_dgm/agent/verifier.py) and the dashboard, so it
    must always be a non-empty string and must never contain "nan" — a NaN leaking in here
    used to surface verbatim in the UI.
    """
    devs = []
    for f, v in means.items():
        g = global_mean.get(f, 0)
        if not isinstance(v, (int, float)) or v != v:  # NaN-safe
            continue
        dev = (v - g) / abs(g) if g else 0.0
        devs.append((f, dev))
    devs.sort(key=lambda x: -abs(x[1]))
    bits = [
        f"{f} {'cao hơn' if d >= 0 else 'thấp hơn'} trung bình {abs(d) * 100:.0f}%"
        for f, d in devs[:top_n] if abs(d) >= 0.1
    ]
    text = f"{name}: " + ("; ".join(bits) if bits else "không lệch rõ rệt so với mặt bằng chung")
    return text.replace("nan", "0")


def hidden_drivers(X_raw: pd.DataFrame, labels, features: list[str]) -> dict[str, float]:
    """Which raw features actually separate the clusters, via a shallow decision tree.

    Depth and leaf size are capped so a single outlier cannot become a "driver", and
    classes are balanced so small clusters still register. Only features above 5%
    importance are returned — below that the tree is describing noise. Best-effort.

    Args:
        X_raw: Unscaled feature matrix.
        labels: Cluster assignment per row.
        features: Column names matching X_raw.

    Returns:
        Feature -> importance, descending; empty when nothing clears the threshold.
    """
    try:
        from sklearn.tree import DecisionTreeClassifier

        dt = DecisionTreeClassifier(
            max_depth=3, min_samples_leaf=500, class_weight="balanced", random_state=SEED
        )
        dt.fit(X_raw, labels)
        imp = pd.Series(dt.feature_importances_, index=features)
        imp = imp[imp > 0.05].sort_values(ascending=False)
        return {k: round(float(v), 4) for k, v in imp.items()}
    except Exception as e:
        print(f"[PIPELINE] hidden_drivers skipped: {e}")
        return {}


def save_cluster_chart(personas: list[dict], out_dir: str = "workspace/generated/reports") -> str:
    """Write the cluster-size bar chart and return the markdown line that displays it.

    Kept out of run_persona_pipeline because it performs file I/O against a caller-chosen
    path. Best-effort: returns "" if plotting is unavailable, so a missing chart never
    costs the caller its personas.
    """
    try:
        import os

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "cluster_distribution.png")
        plt.figure(figsize=(10, 6))
        sns.barplot(x=[p["persona_name"] for p in personas], y=[p["support"] for p in personas])
        plt.xticks(rotation=45, ha="right")
        plt.title("Cluster Distribution")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return f"![Cluster Distribution](/file?path={path})"
    except Exception as e:
        print(f"[PIPELINE] cluster chart skipped: {e}")
        return ""


def detect_dataset_mode(columns) -> str:
    """Classify the dataset without guessing beyond what the columns prove.

    GENERIC is the safe default. POST_CHURN needs paired ``old_*``/``recent_*`` columns (the
    before/after churn signature); PRE_CHURN needs an explicit churn target. Historic fee or
    ARPU columns are NOT evidence of an active customer base — they exist perfectly well in a
    set of customers who have already left, and treating them as evidence produced the
    nonsense of scoring "future churn risk" for people who had already churned.

    Args:
        columns: The dataset's column names.

    Returns:
        One of "PRE_CHURN", "POST_CHURN", "GENERIC".
    """
    lower = [str(c).lower() for c in columns]
    if "rmdt" in lower:
        return "PRE_CHURN"
    has_old = any(c.startswith("old_") for c in lower)
    has_recent = any(c.startswith("recent_") for c in lower)
    if has_old and has_recent:
        return "POST_CHURN"
    return "GENERIC"


def choose_k(X: np.ndarray, k_range: tuple[int, int] = _K_RANGE) -> tuple[int, float, np.ndarray]:
    """Pick the k with the best silhouette over ``k_range``.

    Args:
        X: Scaled feature matrix.
        k_range: Half-open (min_k, max_k) range to search.

    Returns:
        (best_k, best_silhouette, labels).
    """
    best = (k_range[0], -1.0, None)
    for k in range(*k_range):
        if k >= len(X):
            break
        labels = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score(X, labels, sample_size=min(_SILHOUETTE_SAMPLE, len(X)), random_state=SEED))
        if score > best[1]:
            best = (k, score, labels)
    if best[2] is None:  # degenerate data: one cluster for everything
        best = (1, -1.0, np.zeros(len(X), dtype=int))
    return best


def segmentation_quality(silhouette: float, dominant_pct: float) -> str:
    """Label the split's usefulness: OUTLIER_DRIVEN, WEAK or NORMAL."""
    if silhouette > 0.7 and dominant_pct > 0.8:
        return "OUTLIER_DRIVEN"
    if silhouette < 0.15:
        return "WEAK"
    return "NORMAL"


def _dedupe_names(base_names: dict) -> dict:
    """Suffix repeated persona names so two clusters never render identically."""
    counts = Counter(base_names.values())
    seen: dict[str, int] = {}
    final = {}
    for cid, name in base_names.items():
        if counts[name] > 1:
            seen[name] = seen.get(name, 0) + 1
            final[cid] = f"{name} - Nhóm {seen[name]}"
        else:
            final[cid] = name
    return final


def _is_row_identifier(series: pd.Series) -> bool:
    """True for a whole-numbered column holding a distinct value on every row.

    Such a column cannot group rows — it is in bijection with them — so it contributes a
    full-variance axis of pure noise to KMeans. Measured on a real 62,467-row export: OBJID
    sat at a uniqueness ratio of 1.0000 while the highest real feature reached 0.1047.

    The whole-number condition is what keeps this from eating real measurements: 450 draws
    from a continuous distribution are all distinct too, and those must be kept.
    """
    values = series.dropna()
    if len(values) < 2 or values.nunique() != len(values):
        return False
    return bool((values % 1 == 0).all())


def is_near_constant(series: pd.Series, max_modal: float = _MAX_MODAL_FRACTION) -> bool:
    """True when too few rows differ from the commonest value to describe a group.

    Rejecting only single-valued columns left the ones that are constant in every practical
    sense. On the 62,467-row Churn_VT export ``persistent_cl`` differs on TWO rows. Two rows
    cannot form a cluster; what the column does instead is reach StandardScaler with a
    standard deviation near zero, which lifts each differing row to a z-score around +170 —
    an outlier the nearest centroid chases across an axis carrying no partition at all.

    The test is on the MINORITY share rather than the count, because whether enough rows
    differ to describe a group is relative to the dataset. Columns with a rare but real
    signal survive: ``HTKT_CHECKLIST_202604`` differs on 567 rows (0.91%) and is kept.

    Args:
        series: One candidate feature.
        max_modal: Share the commonest value may occupy before the column is rejected.

    Returns:
        Whether the column should be kept out of the feature matrix.
    """
    values = series.dropna()
    if len(values) == 0:
        return True
    return bool(float(values.value_counts(normalize=True).iloc[0]) > max_modal)


def usable_features(data: pd.DataFrame, feats, absent_means_zero: set[str] | None = None
                    ) -> tuple[list[str], dict[str, str]]:
    """Filter ``feats`` down to the columns that can carry a segmentation, and say what went.

    The same three refusals :func:`_auto_features` makes, applied to ANY feature list. That
    is the whole point: the guards used to live only in the auto-selection branch, which
    runs only when the caller names nothing — and in production the caller is an LLM that
    always names something. A real report consequently characterised a 3,816-customer
    persona entirely by ``ratio_missed_30d`` (+1522%), a column absent from 98.19% of rows,
    and listed ``persistent_cl`` (which differs on two rows of 62,467) as a business signal.

    Args:
        data: The dataset.
        feats: Candidate feature names, in the caller's order.
        absent_means_zero: Columns whose blanks are declared zeros; judged on the values as
            they will be used, so a sparse event flag is not mistaken for a constant.

    Returns:
        (kept in the given order, {dropped column: reason}).
    """
    declared = set(absent_means_zero or ())
    kept: list[str] = []
    dropped: dict[str, str] = {}
    for column in feats:
        if column not in data.columns:
            dropped[str(column)] = "không tồn tại"
            continue
        series = data[column]
        if not pd.api.types.is_numeric_dtype(series):
            dropped[str(column)] = "không phải kiểu số"
            continue
        if is_near_constant(series.fillna(0.0) if column in declared else series):
            dropped[str(column)] = "gần như hằng số"
            continue
        if _is_row_identifier(series):
            dropped[str(column)] = "định danh"
            continue
        kept.append(column)
    return kept, dropped


def _auto_features(data: pd.DataFrame, cluster_col: str, exclude: set[str] | None = None,
                   absent_means_zero: set[str] | None = None) -> list[str]:
    """Every numeric column that actually varies — the pipeline's own deterministic choice.

    ``exclude`` holds columns that describe the OUTCOME rather than the behaviour. Cluster
    on those and the segmentation is partly a restatement of the answer, which the report
    then presents as a finding about the customers.

    ``absent_means_zero`` names the columns the data owner has declared record events, and
    it is needed HERE, not only later: a sparse event flag stores 1 where the event happened
    and nothing where it did not, so its present values are all identical and the
    near-constant guard drops it before the declaration is ever consulted — which would make
    declaring it pointless. Those columns are judged on the values as they will be used.
    """
    numeric = data.select_dtypes(include="number")
    excluded = set(exclude or ())
    declared = set(absent_means_zero or ())
    feats, identifiers, flat = [], [], []
    for c in numeric.columns:
        if c == cluster_col or c in excluded:
            continue
        if is_near_constant(numeric[c].fillna(0.0) if c in declared else numeric[c]):
            flat.append(str(c))
        elif _is_row_identifier(numeric[c]):
            identifiers.append(str(c))
        else:
            feats.append(c)
    if identifiers:
        print(f"[PIPELINE] bỏ {len(identifiers)} cột định danh khỏi feature: "
              + ", ".join(identifiers))
    if flat:
        print(f"[PIPELINE] bỏ {len(flat)} cột gần như hằng số khỏi feature: "
              + ", ".join(flat))
    return feats


def resolve_missing(
    raw: pd.DataFrame,
    absent_means_zero: set[str] | None = None,
    max_absent: float = _MAX_ABSENT_FRACTION,
) -> tuple[pd.DataFrame, dict]:
    """Close the gaps in ``raw`` without inventing measurements, and say what was done.

    The previous behaviour was ``fillna(0.0)`` on everything. Zero is not a neutral filler:
    for a ratio it is the floor, for a fee it is "spent nothing". On the 62,467-row
    Churn_VT export ``ratio_missed_30d`` is 98.19% absent and the 1,133 present values
    average 0.674 — filling with zero handed 61,334 subscribers the assertion "none of your
    outgoing calls failed", at the far end of the column from every value actually seen.

    Whether an absent cell means "no event occurred" or "nobody measured this" is a fact
    about the column that the column cannot reveal. So it is declared, not guessed:

    * listed in ``absent_means_zero`` — filled with 0.0, kept however absent it is
    * absent in more than ``max_absent`` of rows — dropped, because imputing the majority
      of a column is inventing it whatever statistic is used
    * otherwise — filled with the median, which shifts the distribution least and stays
      inside the observed range

    Args:
        raw: Numeric frame, one column per candidate feature.
        absent_means_zero: Columns the data owner has confirmed record events, where a gap
            genuinely means none occurred.
        max_absent: Share of absent values above which an undeclared column is dropped.

    Returns:
        (frame with no NaN left, report naming every column dropped, imputed or zeroed).
    """
    declared = set(absent_means_zero or ())
    report: dict = {"dropped": [], "imputed": {}, "zero_filled": []}
    out = raw.copy()

    for column in raw.columns:
        absent = float(raw[column].isna().mean())
        if column in declared:
            if absent:
                out[column] = raw[column].fillna(0.0)
                report["zero_filled"].append(column)
            continue
        if not absent:
            continue
        if absent > max_absent:
            out = out.drop(columns=[column])
            report["dropped"].append(column)
            continue
        out[column] = raw[column].fillna(raw[column].median())
        report["imputed"][column] = "median"

    return out, report


def binary_features(data: pd.DataFrame, feats: list[str]) -> set[str]:
    """Which of ``feats`` are 0/1 flags rather than measured quantities.

    A flag's cluster mean is a SHARE — the fraction of the group carrying it — so it must
    not be described as "cao" or "thấp" like a level. Detected from the values rather than
    the name: ``no_``-prefixed columns are the obvious flags on this export, but
    ``high_spender`` and ``ever_downtrend`` are flags too and carry no such prefix.

    Returns:
        Columns holding only 0 and 1, both present. A constant column is not a flag; it
        distinguishes nothing and is already dropped as near-constant.
    """
    flags = set()
    for column in feats:
        if column not in data.columns:
            continue
        values = pd.to_numeric(data[column], errors="coerce").dropna().unique()
        if len(values) == 2 and set(values) <= {0, 1}:
            flags.add(str(column))
    return flags


def cohort_mix(statuses: pd.Series) -> dict[str, float]:
    """Measure the share of each outcome present in ``statuses``.

    The dashboard used to print "100% — Toàn bộ mẫu đã rời mạng" whenever ANY persona
    carried a churn_driver. Nothing counted that 100%, and on the Churn_VT export it was
    wrong by 4,533 people: the data owner confirmed that rows carrying neither cancellation
    code are subscribers who restored service, so the cohort is 92.7% departed, not 100%.

    This function only counts. It has no idea which value means "left" and must not
    acquire one — the caller names the column, and a dataset-specific script is what
    decides that HSSD means anything at all.

    Args:
        statuses: One outcome label per row. Unlabelled rows are excluded from the
            denominator: an absent label is not evidence of any outcome.

    Returns:
        {label: share}, summing to 1.0, or {} when nothing is labelled.
    """
    labelled = statuses.dropna()
    if labelled.empty:
        return {}
    return {str(k): float(v) for k, v in labelled.value_counts(normalize=True).items()}


#: Suffixes the exports already use to declare an observation period. Matched at the end of
#: a column name only, so `total_negative_202601` is read as a month rather than as "1 day".
_WINDOW_PATTERNS = (
    (re.compile(r"_(\d{6})$"), None),          # YYYYMM — one specific month
    (re.compile(r"_(\d+)m$", re.I), "{} tháng"),
    (re.compile(r"_(\d+)d$", re.I), "{} ngày"),
)
_SPECIFIC_MONTH = "tháng cụ thể"


def declared_time_windows(columns) -> dict[str, list[str]]:
    """Group column names by the observation period their own naming declares.

    Only what the name SAYS counts. ``fee_total`` really does cover four months on the
    Churn_VT export, but nothing in the name says so, and inferring it is how a report ends
    up asserting a period nobody wrote down.

    Args:
        columns: Column names.

    Returns:
        {window label: columns}, in first-seen order; {} when nothing declares a period.
    """
    windows: dict[str, list[str]] = {}
    for column in columns:
        name = str(column)
        for pattern, template in _WINDOW_PATTERNS:
            match = pattern.search(name)
            if not match:
                continue
            label = _SPECIFIC_MONTH if template is None else template.format(match.group(1))
            windows.setdefault(label, []).append(name)
            break
    return windows


def time_window_caveat(columns) -> str:
    """State that features covering different periods share one distance computation.

    Euclidean distance treats every axis as commensurable, so a 6-month complaint count and
    a 30-day call count are compared as though they spanned the same time. Nothing in the
    data can fix that. What can be fixed is a report presenting the result as though the
    question never came up, so this says it — and says only that. Nothing is rescaled.

    Returns:
        A sentence for the reader, or "" when at most one period is declared.
    """
    windows = declared_time_windows(columns)
    if len(windows) < 2:
        return ""
    spans = ", ".join(f"{label} ({len(cols)} cột)" for label, cols in windows.items())
    return (
        f"Các feature khai báo nhiều kỳ quan sát khác nhau: {spans}. Phép phân cụm đo "
        f"khoảng cách trên mọi cột như nhau, nên các chỉ số này được so sánh như thể cùng "
        f"một kỳ. Số liệu giữ nguyên theo kỳ gốc của từng cột."
    )


def correlation_groups(frame: pd.DataFrame, threshold: float = _REDUNDANCY_THRESHOLD) -> list[list[str]]:
    """Partition ``frame``'s columns into groups that measure the same thing.

    Two columns join the same group when ``|r| >= threshold``, transitively — the relation
    is treated as connected components, so a chain a~b~c lands together even if a and c fall
    just short of each other. On the real export this recovers exactly the families a reader
    would name by eye: the four fee columns, the four technical-fault columns, and so on.

    Anti-correlation counts: ``no_cl_all_period`` is ``active_cl_months`` negated, which is
    one measurement written twice, not two findings.

    Args:
        frame: Numeric frame, one column per feature.
        threshold: Absolute correlation at which two columns are the same measurement.

    Returns:
        Groups in column order; every column appears in exactly one, singletons included.
    """
    columns = list(frame.columns)
    if not columns:
        return []

    # A constant column correlates with nothing — corr() yields NaN, which must not be read
    # as agreement. Filling with 0 keeps it in a group of its own.
    corr = frame.corr().abs().fillna(0.0)

    parent = {c: c for c in columns}

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    for i, a in enumerate(columns):
        for b in columns[i + 1:]:
            if corr.loc[a, b] >= threshold:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

    grouped: dict[str, list[str]] = {}
    for c in columns:
        grouped.setdefault(find(c), []).append(c)
    return list(grouped.values())


def redundancy_weights(frame: pd.DataFrame, threshold: float = _REDUNDANCY_THRESHOLD) -> dict[str, float]:
    """How much each column may count for, given how often its measurement is repeated.

    KMeans weights by column count: m standardised copies of one measurement contribute m
    times the squared distance of a domain represented once. Scaling each member of a group
    of m by ``1/sqrt(m)`` makes the group contribute exactly one column's worth, while every
    column keeps its own values and its own direction.

    Dropping all but one member would do the same to the distance and lose the temporal
    signal: fee_old and fee_recent correlate at 0.95, and their DIFFERENCE is what the
    POST_CHURN path reads.

    Returns:
        {column: weight}, every weight strictly positive.
    """
    return {
        column: 1.0 / math.sqrt(len(group))
        for group in correlation_groups(frame, threshold)
        for column in group
    }


def _prepare_matrix(data: pd.DataFrame, feats: list[str], absent_means_zero: set[str] | None = None):
    """Coerce ``feats`` to a scaled matrix, or None if the set cannot be clustered on.

    Returns None rather than raising so a candidate feature set that turns out unusable
    (too few columns, almost entirely zeros) simply loses the comparison. Dropping the
    columns nobody measured is one way to end up under that bar, so the length check runs
    again afterwards.
    """
    if len(feats) < 2:
        return None
    numeric = data[feats].apply(pd.to_numeric, errors="coerce")
    raw, report = resolve_missing(numeric, absent_means_zero)
    if report["dropped"] or report["imputed"] or report["zero_filled"]:
        print(f"[PIPELINE] giá trị thiếu: bỏ {len(report['dropped'])} cột "
              f"({', '.join(report['dropped']) or '—'}), điền trung vị "
              f"{len(report['imputed'])} cột, điền 0 theo khai báo {len(report['zero_filled'])} cột")
    if len(raw.columns) < 2:
        return None
    if float((raw == 0).to_numpy().mean()) > _MAX_ZERO_FRACTION:
        return None

    X = StandardScaler().fit_transform(raw.to_numpy(dtype=float))

    # Stop one measurement written down four times from outvoting one written once. Applied
    # to the SCALED matrix only: `raw` feeds every reported mean and deviation, and those
    # have to stay in the units a reader recognises.
    weights = redundancy_weights(raw)
    groups = [g for g in correlation_groups(raw) if len(g) > 1]
    if groups:
        print(f"[PIPELINE] {len(groups)} nhóm feature trùng đo, hạ trọng số: "
              + "; ".join("+".join(g) for g in groups))
    return raw, X * np.array([weights[c] for c in raw.columns])


def _failed_persona(data: pd.DataFrame, reason: str) -> list[dict]:
    """The single persona emitted when the data genuinely does not split.

    Worth emitting rather than raising: the report layer needs valid JSON, and a stated
    failure is more useful than an exception the repair loop will chase. The wording stays
    dataset-neutral — the previous version blamed "khách hàng" and asserted the dataset
    lacked variance, which was wrong whenever the real cause was a missing column set.
    """
    n = len(data)
    return [{
        "cluster_id": 0,
        "persona_name": "Không phân hoá được nhóm",
        "support": int(n),
        "support_pct": 1.0,
        "confidence": "LOW",
        "persona_type": "MAINSTREAM",
        "severity": None,
        "risk": None,
        "risk_tier": None,
        "priority_score": 10,
        "feature_means": {},
        "evidence": {},
        "profile_attributes": {},
        "domain_signature": {},
        "temporal_trajectory": [],
        "segmentation_quality": "WEAK",
        "recommended_actions": [
            "Thu thập thêm biến mô tả hành vi để phân nhóm hiệu quả hơn",
        ],
        "is_anomaly": False,
        "sample_persona_text": f"Không phân hoá được nhóm ({reason}).",
        "failure_reason": reason,
    }]


def run_persona_pipeline(
    data: pd.DataFrame,
    behavioral_features: list[str] | None = None,
    dataset_mode: str | None = None,
    cluster_col: str = "cluster",
    status_col: str | None = None,
    active_status_values: set[str] | None = None,
    label_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Cluster ``data`` and return the persona dicts the report layer consumes.

    Deterministic for a given (data, features): fixed seed, fixed k search, fixed rule
    engine. Mutates ``data`` only by adding ``cluster_col``.

    Args:
        data: The dataset, one row per entity.
        behavioral_features: Columns to cluster on. Defaults to every numeric column with
            more than one distinct value.
        dataset_mode: Override for :func:`detect_dataset_mode`.
        cluster_col: Name of the cluster-label column to add.
        status_col: Column holding each row's outcome, if the dataset records one. Each
            persona then carries the mix actually measured in its cluster, instead of the
            report asserting a proportion nobody counted. Excluded from clustering:
            segmenting by the outcome and then describing the segments is circular.
        active_status_values: Which values of ``status_col`` mean the row is still a
            customer. Supplied by the caller because the pipeline has no way to know —
            without it the still-active share stays unstated rather than reported as zero.
        label_dir: Directory to search for a data dictionary supplying human column
            labels for the persona names. Defaults to the working directory. The
            dictionary must pass the same schema gate as metadata injection.

    Returns:
        A list of persona dicts. On a genuinely unsplittable dataset, a single persona
        describing that outcome — never an exception, because the caller needs valid JSON.
    """
    if data is None or len(data) == 0:
        return _failed_persona(pd.DataFrame(), "empty_dataset")

    # A caller that names features is making a claim about this dataset's schema. Check it
    # instead of quietly repairing it: the old code filtered the list down to whatever
    # existed and auto-selected when nothing did, so a feature list copied from a DIFFERENT
    # dataset still produced a confident, plausible report. Observed live — a model emitted
    # Iris column names for a retail upload, all four were dropped, and the run "succeeded".
    if behavioral_features:
        missing = [f for f in behavioral_features if f not in data.columns]
        if missing:
            return _failed_persona(
                data,
                f"unknown_columns: {', '.join(missing)} — không tồn tại trong dataset này "
                f"(các cột thực có: {', '.join(map(str, data.columns[:20]))}"
                f"{'…' if len(data.columns) > 20 else ''})",
            )
        non_numeric = [f for f in behavioral_features if not pd.api.types.is_numeric_dtype(data[f])]
        if non_numeric:
            return _failed_persona(
                data,
                f"non_numeric_columns: {', '.join(non_numeric)} — tồn tại nhưng không phải "
                f"kiểu số, không dùng để phân cụm được",
            )

    # Which blanks are real zeros — a business fact, so only an explicit declaration counts.
    # Loaded first: it decides both which columns survive feature selection and which
    # survive the matrix.
    absent_zero: set[str] = set()
    try:
        from api.services.metadata_gate import absent_zero_columns

        absent_zero = absent_zero_columns(label_dir or os.getcwd(), list(data.columns))
        if absent_zero:
            print(f"[PIPELINE] {len(absent_zero)} cột được nghiệp vụ khai báo 'ô trống = "
                  f"không phát sinh', điền 0: " + ", ".join(sorted(absent_zero)))
    except Exception as e:  # a dictionary problem must not take the run down
        print(f"[PIPELINE] không nạp được khai báo ô trống (bỏ qua): {e}")

    mode = dataset_mode or detect_dataset_mode(data.columns)
    auto_feats = _auto_features(data, cluster_col, exclude={status_col} if status_col else None,
                                absent_means_zero=absent_zero)
    caller_feats = [f for f in (behavioral_features or []) if f != status_col]

    # On GENERIC data the pipeline picks the features, not the caller.
    #
    # The caller is an LLM improvising a list per run. Two runs over the same 50k file gave
    # silhouette 0.426 (12 features) and 0.286 (9 features) — one dataset, two segmentations,
    # the second a third worse purely because the model named fewer columns that time. The
    # pipeline's own selection scored 0.426, matching the model's best attempt.
    #
    # Scoring both and keeping the higher silhouette was tried and rejected: silhouette
    # measures how compact the partition it FOUND is, not whether that structure is real, so
    # two pure-noise columns scored 0.351 against 0.308 for a set containing the actual
    # signal. A measure that prefers noise cannot arbitrate.
    #
    # So the deterministic rule wins: use every numeric column that varies. Callers wanting a
    # subset filter the DataFrame before calling, which is what the prompt already instructs.
    # The caller's list is still validated above — catching a list copied from another
    # dataset is its real value — and what was actually used is recorded on every persona.
    #
    # Unchanged on the telco path: there the column list carries domain meaning, and that
    # path is deliberately left exactly as it was.
    if mode == "GENERIC" and auto_feats:
        feats, feature_selection = auto_feats, "auto"
        if caller_feats and caller_feats != auto_feats:
            print(f"[PIPELINE] feature set: bỏ qua {len(caller_feats)} feature do caller đề xuất, "
                  f"dùng {len(auto_feats)} cột số biến thiên của dataset (tất định)")
    else:
        feats = caller_feats or auto_feats
        feature_selection = "caller" if caller_feats else "auto"

    # Applied to WHATEVER feature set won above, not only the auto-selected one. Living
    # inside _auto_features meant the guards ran only when the caller named nothing, and in
    # production the caller always names something — so the path that actually runs was the
    # unprotected one. Auto-selection has already filtered; this is a no-op for it.
    feats, refused = usable_features(data, feats, absent_means_zero=absent_zero)
    if refused:
        print("[PIPELINE] bỏ khỏi feature do caller đề xuất: "
              + ", ".join(f"{c} ({why})" for c, why in refused.items()))

    if len(feats) < 2:
        return _failed_persona(data, "insufficient_numeric_features")

    prepared = _prepare_matrix(data, feats, absent_means_zero=absent_zero)
    if prepared is None:
        zero_fraction = float(
            (data[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0) == 0).to_numpy().mean()
        )
        return _failed_persona(data, f"zero_inflated_{zero_fraction:.3f}")
    X_raw, X = prepared

    # resolve_missing() drops the columns nobody measured, so the surviving matrix is the
    # authority on which features exist — not the list we asked for. Everything downstream
    # (global means, hidden drivers, features_used on every persona) indexes X_raw by this
    # list. Observed on the real 62,467-row export: 16 columns were dropped and the very
    # next line died with KeyError: 'LLSD_202606'.
    feats = list(X_raw.columns)

    window_caveat = time_window_caveat(feats)
    if window_caveat:
        print(f"[PIPELINE] {window_caveat}")

    # Human labels for the persona names, from a data dictionary that provably describes
    # THIS dataset. Without them the naming code falls back to the column name, which is
    # how a business owner was shown "Nhóm no_fee_all_period cao".
    flag_feats = binary_features(data, feats)
    column_labels: dict[str, str] = {}
    try:
        from api.services.metadata_gate import labels_for_columns

        column_labels = labels_for_columns(label_dir or os.getcwd(), list(data.columns))
        if column_labels:
            print(f"[PIPELINE] nhãn nghiệp vụ cho {len(column_labels)} cột, dùng để đặt tên nhóm")
    except Exception as e:  # a dictionary problem must not take the run down
        print(f"[PIPELINE] không nạp được nhãn cột (bỏ qua): {e}")

    best_k, best_sil, labels = choose_k(X)
    data[cluster_col] = labels

    sizes = data[cluster_col].value_counts()
    dominant_pct = float(sizes.max()) / len(data)

    stage2_triggered = False
    if dominant_pct > _STAGE2_TRIGGER:
        dominant_cid = int(sizes.idxmax())
        data, stage2_triggered, stage2_info = try_substage_cluster(data, dominant_cid, cluster_col=cluster_col)
        print(f"[STAGE-2] cluster {dominant_cid} ({dominant_pct * 100:.1f}%): {stage2_info}")
        sizes = data[cluster_col].value_counts()
        dominant_pct = float(sizes.max()) / len(data)

    if dominant_pct > _HARD_STOP_DOMINANT and not stage2_triggered:
        return _failed_persona(data, f"dominant_cluster_{dominant_pct:.2f}")

    cluster_sizes = sizes.sort_index().to_dict()
    quality = segmentation_quality(best_sil, dominant_pct)

    profile_attributes = compute_profile_attributes(data, cluster_col=cluster_col)
    profile_global = compute_profile_global_means(profile_attributes, cluster_sizes)
    domain_sig = compute_domain_signature(data, cluster_col=cluster_col)
    churn_drivers = (
        compute_churn_drivers(data, domain_sig, cluster_col=cluster_col)
        if mode == "POST_CHURN" else {}
    )

    cluster_stats = data.groupby(cluster_col)[feats].mean()
    global_mean = {f: float(X_raw[f].mean()) for f in feats}

    metadata, base_names = {}, {}
    for cid, row in cluster_stats.iterrows():
        support_pct = cluster_sizes[cid] / len(data)
        meta = apply_business_rules(
            row.to_dict(), support_pct, profile_attributes.get(cid, {}), profile_global,
            mode, churn_drivers.get(cid, {}), domain_sig.get(cid, {}),
        )
        metadata[cid] = meta
        base_names[cid] = meta["persona_name"]
    final_names = _dedupe_names(base_names)

    # Measured per cluster, never inferred. An unnamed status column leaves both fields
    # empty so the report has nothing to state — which is the honest outcome, and the one
    # the hardcoded "100% đã rời mạng" skipped straight past.
    mix_by_cluster: dict[Any, dict[str, float]] = {}
    active_by_cluster: dict[Any, float | None] = {}
    if status_col and status_col in data.columns:
        for cid in cluster_sizes:
            mix = cohort_mix(data.loc[data[cluster_col] == cid, status_col])
            mix_by_cluster[cid] = mix
            active_by_cluster[cid] = (
                float(sum(mix.get(str(v), 0.0) for v in active_status_values))
                if active_status_values else None
            )

    personas: list[dict[str, Any]] = []
    for cid in sorted(cluster_sizes):
        meta = metadata[cid]
        means = {k: float(v) for k, v in cluster_stats.loc[cid].to_dict().items()}
        profile = profile_attributes.get(cid, {})
        evidence = {
            f: round(v, 4) for f, v in means.items()
            if (global_mean.get(f, 0) and abs(v - global_mean[f]) / abs(global_mean[f]) >= 0.2)
            or (not global_mean.get(f, 0) and v > 0)
        }
        is_anomaly = meta["persona_type"] == "ANOMALY"
        name = "Hành vi bất thường" if is_anomaly else final_names[cid]
        personas.append({
            "cluster_id": int(cid),
            "support": int(cluster_sizes[cid]),
            "support_pct": cluster_sizes[cid] / len(data),
            "feature_means": means,
            "evidence": evidence or means,
            "persona_type": meta["persona_type"],
            "severity": meta["severity"],
            "risk": meta["risk"],
            "persona_name": name,
            "priority_score": meta["priority_score"],
            "confidence": "LOW" if is_anomaly else "HIGH",
            "churn_driver": meta.get("churn_driver"),
            "churn_driver_evidence": meta.get("churn_driver_evidence"),
            "churn_driver_confidence": meta.get("churn_driver_confidence"),
            "temporal_trajectory": meta.get("temporal_trajectory", []),
            "onset_sequence": churn_drivers.get(cid, {}).get("onset_sequence", []),
            "domain_signature": domain_sig.get(cid, {}),
            "profile_attributes": profile,
            "risk_tier": classify_risk_tier(meta, profile),
            "cohort_mix": mix_by_cluster.get(cid, {}),
            "active_pct": active_by_cluster.get(cid),
            "time_window_caveat": window_caveat,
            "is_anomaly": is_anomaly,
            "segmentation_quality": quality,
            # "caller" = the feature list supplied to this call was used; "auto" = the
            # pipeline's own selection scored better and replaced it. Surfaced so a reader
            # can see WHICH set produced the segmentation in front of them.
            "feature_selection": feature_selection,
            "features_used": list(feats),
            "recommended_actions": generate_actions(mode, name, meta["severity"], meta["risk"], profile),
            "sample_persona_text": _sample_persona_text(name, means, global_mean),
        })

    # Name generic personas from their own measured deviations, HERE rather than in the
    # report renderer. The rule engine's ladder is entirely telco predicates, so on any other
    # dataset every cluster fell to one fallback string: a real 50k-row retail upload came
    # out as "Khách hàng ổn định - Nhóm 1..4". The report looked right only because it
    # re-named personas itself; the dashboard, feed and database all showed the four
    # identical names. Anomalies keep their own label, and the report may still upgrade a raw
    # column name to a human one.
    if mode == "GENERIC":
        generic_names = name_by_top_feature(personas, global_mean, labels=column_labels,
                                            binary_features=flag_feats)
        for p, new_name in zip(personas, generic_names):
            if p["is_anomaly"]:
                continue
            # `name_by_top_feature` returns None when nothing deviates enough to name the
            # cluster after, documented as "the caller keeps whatever it already had". Here
            # that would be the rule engine's telco fallback, so the average cluster — the
            # one present in nearly every dataset — came out as "Khách hàng ổn định" on data
            # with no customers in it. Name it after what was actually measured instead.
            chosen = new_name or _NEAR_MEAN_NAME
            p["persona_name"] = chosen
            p["sample_persona_text"] = _sample_persona_text(
                chosen, p["feature_means"], global_mean
            )
        deduped = _dedupe_names({p["cluster_id"]: p["persona_name"] for p in personas})
        for p in personas:
            p["persona_name"] = deduped[p["cluster_id"]]

    # Same defect, other path. The driver ladder's last rule fires when no interaction domain
    # stands out, which is honest but says nothing that tells one such cluster from another.
    # On a real 62,467-row churned base, 5 of 6 clusters landed there and the reader saw five
    # groups — 10.0%, 9.9%, 50.2%, 3.2%, 3.6% — under one identical label. They are separate
    # clusters precisely because they differ measurably, so name them by that difference.
    # Clusters that DID match a driver rule keep its domain wording.
    if mode != "GENERIC":
        unsignalled = [p for p in personas
                       if not p["is_anomaly"] and p.get("churn_driver") == NO_STANDOUT_SIGNAL]
        if len(unsignalled) > 1:
            renamed = name_by_top_feature(unsignalled, global_mean, labels=column_labels,
                                          binary_features=flag_feats)
            for p, new_name in zip(unsignalled, renamed):
                if new_name:
                    p["persona_name"] = new_name
                    p["sample_persona_text"] = _sample_persona_text(
                        new_name, p["feature_means"], global_mean
                    )
            deduped = _dedupe_names({p["cluster_id"]: p["persona_name"] for p in personas})
            for p in personas:
                p["persona_name"] = deduped[p["cluster_id"]]

    drivers = hidden_drivers(X_raw, data[cluster_col], feats)
    if drivers:
        print("[PIPELINE] hidden drivers (>5% importance): " + ", ".join(
            f"{k}={v}" for k, v in drivers.items()))
    else:
        print("[PIPELINE] hidden drivers: không feature nào vượt 5% importance")

    print(
        f"[PIPELINE] mode={mode} k={len(cluster_sizes)} silhouette={best_sil:.3f} "
        f"dominant={dominant_pct * 100:.1f}% quality={quality} features={len(feats)}"
    )
    return personas
