"""Text zprávy: jedna definice značky ___ pro počet i věkovou značku."""

from __future__ import annotations

from operator import itemgetter
from typing import Any, Final

import pytest

from mediparse.domain.note_text import (
    AGE_MARKER,
    deid_markers,
)
from tests.support import AGE_MARKER_CASES

TWO: Final = 2


def test_marker_is_exactly_three_underscores() -> None:
    """Značka je právě ___, delší i kratší řady podtržítek se nepočítají."""
    assert deid_markers("x ___ y") == 1
    assert deid_markers("____") == 0
    assert deid_markers("__") == 0
    assert deid_markers("______") == 0
    assert deid_markers("___ ___") == TWO


@pytest.mark.parametrize("case", AGE_MARKER_CASES, ids=itemgetter("name"))
def test_shared_age_marker_cases(case: dict[str, Any]) -> None:
    """Sdílené případy definují masku věku v Pythonu i R."""
    assert len(AGE_MARKER.findall(case["text"])) == case["markers"]
