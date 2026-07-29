"""Every rung of the driver ladder needs an entry in all three narrative maps.

The two live in different files and nothing linked them. When a rung is added and the maps
are not, `report_generator` finds no entry, decides the driver is unknown, and quietly
renames the persona through `_compose_fallback_driver` — so the work of adding the rung
disappears and the report goes back to saying nothing useful about that group.

Caught by hand while adding the technical rung on 2026-07-29; this makes the check run.
"""
import inspect
import re

import pytest

from triadic_dgm.persona import profiling
from triadic_dgm.services import report_generator

_MAPS = (
    "_CHURN_DRIVER_NARRATIVE_CLAUSE",
    "_CHURN_DRIVER_NARRATIVE_NOUN",
    "_CHURN_DRIVER_BUSINESS_INSIGHT",
)


def _ladder_names() -> set[str]:
    """Every driver name classify_churn_driver can return."""
    source = inspect.getsource(profiling.classify_churn_driver)
    names = set(re.findall(r"result\(\s*\n?\s*'([^']+)'", source))
    names.add(profiling.NO_STANDOUT_SIGNAL)
    return names


def test_the_ladder_is_not_empty():
    """If the scrape breaks, every other test here would pass vacuously."""
    assert len(_ladder_names()) >= 5


@pytest.mark.parametrize("map_name", _MAPS)
def test_every_rung_has_an_entry(map_name):
    missing = _ladder_names() - set(getattr(report_generator, map_name))
    assert not missing, f"{map_name} is missing: {sorted(missing)}"


@pytest.mark.parametrize("map_name", _MAPS)
def test_no_entry_describes_a_rung_that_no_longer_exists(map_name):
    """A stale key is dead weight that reads as though it still fires."""
    extra = set(getattr(report_generator, map_name)) - _ladder_names()
    assert not extra, f"{map_name} has stale keys: {sorted(extra)}"


@pytest.mark.parametrize("map_name", _MAPS)
def test_no_entry_claims_a_reason_for_leaving(map_name):
    """The standing constraint from the data owner, enforced where the text is written."""
    for key, text in getattr(report_generator, map_name).items():
        low = str(text).lower()
        for phrase in ("nguyên nhân", "chủ động rời", "đối thủ", "cạnh tranh", "giá cước", "dẫn đến"):
            assert phrase not in low, f"{map_name}[{key!r}] claims a cause: {text}"
