"""Decide whether a data dictionary on disk describes the dataset actually loaded.

The chat route appends any ``*metadata*.json`` it finds to the user's message, telling the
model to obey it strictly. That is a genuinely useful feature — a column dictionary is
exactly what an analyst wants the model to have — but it was ungated, so a dictionary left
in the repository from an earlier project was handed to every later analysis.

The gate here is a *match* test, not a vocabulary test. A dictionary is wrong for this run
when it describes columns the loaded dataframe does not have, whatever domain it belongs
to; that also protects a third dataset from a second one's dictionary, which a hardcoded
vocabulary list never would.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Any, Iterable, Sequence

#: Share of a dictionary's described columns that must exist in the active dataset before
#: it is considered to describe that dataset. Half is deliberately lenient — dictionaries
#: routinely document only the interesting columns, and may be written before a
#: preprocessing step drops some — while still rejecting a wholly foreign schema, whose
#: overlap is normally zero and at most a stray ``customer_id``.
MIN_COLUMN_OVERLAP: float = 0.5


def described_columns(metadata: Any) -> set[str]:
    """Extract the set of column names a data dictionary describes, lowercased.

    Handles the three shapes this repository produces and consumes:
    a list of ``{"column": ...}`` / ``{"name": ...}`` entries (what ``generate_metadata.py``
    writes), a ``{"columns": [...]}`` wrapper whose entries are strings or dicts, and a
    plain ``{column: description}`` mapping.

    Args:
        metadata: Parsed JSON of a metadata file.

    Returns:
        Lowercased column names; empty if the shape is unrecognised.
    """
    def _name(entry: Any) -> str | None:
        if isinstance(entry, str):
            return entry
        if isinstance(entry, dict):
            for key in ("column", "name", "field", "col"):
                value = entry.get(key)
                if isinstance(value, str):
                    return value
        return None

    entries: Iterable[Any]
    if isinstance(metadata, list):
        entries = metadata
    elif isinstance(metadata, dict):
        inner = metadata.get("columns") or metadata.get("fields")
        if isinstance(inner, list):
            entries = inner
        else:
            # A plain {column: description} mapping. Values are descriptions, not columns.
            entries = list(metadata)
    else:
        return set()

    return {n.strip().lower() for n in map(_name, entries) if n and n.strip()}


def describes_active_dataset(metadata: Any, active_columns: Sequence[str] | None,
                             min_overlap: float = MIN_COLUMN_OVERLAP) -> bool:
    """Report whether ``metadata`` documents the dataset currently loaded.

    Args:
        metadata: Parsed JSON of a metadata file.
        active_columns: Columns of the active dataset. ``None`` (the column peek failed)
            and ``[]`` (nothing readable is registered) both answer False: an unverifiable
            dictionary is precisely the case that caused the bug, and staying silent costs
            the model only a description — it still has the real dataframe in hand.
        min_overlap: Required share of described columns present in the dataset.

    Returns:
        True only when the dictionary is provably about this data.
    """
    if not active_columns:
        return False
    described = described_columns(metadata)
    if not described:
        return False
    present = {str(c).strip().lower() for c in active_columns}
    return len(described & present) / len(described) >= min_overlap


def collect_matching_metadata(search_dir: str, active_columns: Sequence[str] | None) -> str:
    """Gather the metadata files in ``search_dir`` that describe the active dataset.

    Args:
        search_dir: Directory to scan for ``*metadata*.json`` (non-recursive).
        active_columns: Columns of the active dataset, or None if unknown.

    Returns:
        A block of text ready to append to the prompt, or "" when nothing matches. A
        malformed or unreadable file is skipped rather than raised on — a broken file left
        in the working directory must not take the chat endpoint down.
    """
    if not active_columns:
        return ""

    blocks: list[str] = []
    for path in sorted(glob.glob(os.path.join(search_dir, "*metadata*.json"))):
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
            parsed = json.loads(raw)
        except (OSError, ValueError):
            continue
        if not describes_active_dataset(parsed, active_columns):
            continue
        blocks.append(f"\n--- Metadata từ file: {os.path.basename(path)} ---\n{raw}\n")

    return "".join(blocks)


def labels_for_columns(search_dir: str, active_columns: Sequence[str] | None) -> dict[str, str]:
    """Human labels for the active dataset's columns, from a dictionary that describes it.

    Persona names used to read "Nhóm no_fee_all_period cao" — a column name shown to a
    business owner. The naming code has always accepted a column -> label map; nothing ever
    supplied one, so the raw-name fallback WAS the behaviour.

    Gated exactly like :func:`collect_matching_metadata`, and for the same reason: a label
    map from another export is worse than none, because it silently renames whichever
    columns happen to share a name. Columns with no usable label are absent from the map
    rather than mapped to "" — the caller falls back to the column name, and a blank label
    would produce a blank persona name.

    Args:
        search_dir: Directory to scan for ``*metadata*.json`` (non-recursive).
        active_columns: Columns of the active dataset, or None if unknown.

    Returns:
        {column: label}; empty when no dictionary describes this dataset. Never raises.
    """
    if not active_columns:
        return {}

    labels: dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(search_dir, "*metadata*.json"))):
        try:
            with open(path, "r", encoding="utf-8") as f:
                parsed = json.load(f)
        except (OSError, ValueError):
            continue
        if not describes_active_dataset(parsed, active_columns):
            continue
        entries = parsed.get("columns") if isinstance(parsed, dict) else parsed
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            column = entry.get("column") or entry.get("name")
            label = entry.get("label")
            if isinstance(column, str) and isinstance(label, str) and label.strip():
                labels.setdefault(column, label.strip())
    return labels


def absent_zero_columns(search_dir: str, active_columns: Sequence[str] | None) -> set[str]:
    """Columns the data owner has declared record events, where a blank really is a zero.

    Blank-versus-zero cannot be settled from the data. On the Churn_VT export the evidence
    points both ways at once: ``total_negative_202601`` writes an explicit 0 among its 7,751
    present values, which suggests a blank means something else — but that is an inference,
    and acting on inferences about what was measured is the failure this pipeline keeps
    finding. So only an explicit declaration counts.

    Gated exactly like :func:`labels_for_columns`: a declaration made about another export
    is worse than none, because it fills a column here on a different file's authority.
    Only ``"zero"`` is served. ``"unmeasured"`` and an absent declaration are the same
    instruction to the caller — leave the cautious default alone — so they are not
    distinguished here.

    Args:
        search_dir: Directory to scan for ``*metadata*.json`` (non-recursive).
        active_columns: Columns of the active dataset, or None if unknown.

    Returns:
        Column names whose blanks may be filled with 0. Never raises.
    """
    if not active_columns:
        return set()

    declared: set[str] = set()
    for path in sorted(glob.glob(os.path.join(search_dir, "*metadata*.json"))):
        try:
            with open(path, "r", encoding="utf-8") as f:
                parsed = json.load(f)
        except (OSError, ValueError):
            continue
        if not describes_active_dataset(parsed, active_columns):
            continue
        entries = parsed.get("columns") if isinstance(parsed, dict) else parsed
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            column = entry.get("column") or entry.get("name")
            if isinstance(column, str) and entry.get("absent_means") == "zero":
                declared.add(column)
    return declared
