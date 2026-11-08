"""Značka syntetických tabulek: formát souboru a zpětné čtení."""

from __future__ import annotations

import json
from operator import itemgetter
from typing import TYPE_CHECKING

import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.synthetic_marker import (
    MARKER_NAME,
    SyntheticMarker,
    read_marker,
    write_marker,
)
from tests.support import REPOSITORY

if TYPE_CHECKING:
    from pathlib import Path


def test_marker_is_indented_json_with_trailing_newline(tmp_path: Path) -> None:
    """Značka má stejný formát jako ostatní JSON repa a ``read_marker`` ji přečte zpět."""
    marker = SyntheticMarker(prefix="canary", rows=12)

    write_marker(tmp_path / "tables", marker)

    text = (tmp_path / "tables" / MARKER_NAME).read_text(encoding="utf-8")
    assert text == '{\n  "prefix": "canary",\n  "rows": 12\n}\n'
    assert read_marker(tmp_path / "tables") == marker


@pytest.mark.parametrize(
    "case",
    json.loads(
        (REPOSITORY / "tests/fixtures/synthetic_marker_cases.json").read_text(
            encoding="utf-8"
        )
    ),
    ids=itemgetter("name"),
)
def test_shared_marker_cases(tmp_path: Path, case: dict[str, str | bool]) -> None:
    """Stejné případy značky ověřují Python i nezávislý konzument R."""
    (tmp_path / MARKER_NAME).write_text(str(case["text"]), encoding="utf-8")
    if case["valid"]:
        assert read_marker(tmp_path) == SyntheticMarker(
            prefix=json.loads(str(case["text"]))["prefix"], rows=1
        )
    else:
        with pytest.raises(InvalidInputError, match=r"synthetic\.json"):
            read_marker(tmp_path)
