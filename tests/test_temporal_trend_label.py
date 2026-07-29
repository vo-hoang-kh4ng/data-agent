"""The trend label must not contradict the two numbers printed next to it.

From a real report on 62,467 churned subscribers, the "Diễn biến theo thời gian" table:

    Chỉ số               Đầu kỳ   Gần đây   Xu hướng
    Phàn nàn/khiếu nại    1.581     0.369   "ổn định"     <- fell 77%
    Phàn nàn/khiếu nại    0.170     1.548   "ổn định"     <- rose 9x

Every row in every persona said "ổn định". `get_temporal_trajectory` reads the direction
from a dedicated `{root}_trend` column — `complaint_trend`, `cl_trend` — and that export has
neither, so `get_column` returned None, `trend_v` defaulted to 0.0, and 0.0 falls in the
"ổn định" band. A missing column was silently reported as a measured finding of stability.

The old/recent values it prints in the same row ARE present. When the trend column is
missing, the direction must come from them rather than from a default.
"""
import numpy as np
import pandas as pd

from triadic_dgm.persona.profiling import get_temporal_trajectory


def _group(old, recent, with_trend_column=None):
    n = 100
    data = {"old_complaint": np.full(n, old), "recent_complaint": np.full(n, recent)}
    if with_trend_column is not None:
        data["complaint_trend"] = np.full(n, with_trend_column)
    return pd.DataFrame(data)


def _direction(old, recent, with_trend_column=None):
    rows = get_temporal_trajectory(_group(old, recent, with_trend_column))
    assert rows, "trajectory produced no row for a metric that has both old and recent values"
    return rows[0]["trend"]


def test_a_steep_fall_is_not_reported_as_stable():
    """The exact numbers from the real report."""
    assert _direction(1.581, 0.369) != "ổn định"


def test_a_ninefold_rise_is_not_reported_as_stable():
    assert _direction(0.170, 1.548) != "ổn định"


def test_the_direction_matches_the_numbers():
    assert "giảm" in _direction(1.581, 0.369)
    assert "tăng" in _direction(0.170, 1.548)


def test_genuinely_flat_values_are_still_called_stable():
    """The guard against calling everything a trend."""
    assert _direction(1.00, 1.02) == "ổn định"


def test_an_explicit_trend_column_still_wins_when_present():
    """Datasets that ship a computed trend keep using it; nothing changes for them."""
    assert _direction(1.0, 1.0, with_trend_column=-0.9) == "giảm mạnh"
    assert _direction(1.0, 1.0, with_trend_column=0.9) == "tăng mạnh"
