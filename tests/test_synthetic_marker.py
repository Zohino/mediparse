"""Značka syntetických tabulek: formát souboru a zpětné čtení."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mediparse.infrastructure.synthetic_marker import (
    MARKER_NAME,
    SyntheticMarker,
    read_marker,
    write_marker,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_marker_is_indented_json_with_trailing_newline(tmp_path: Path) -> None:
    """Značka má stejný formát jako ostatní JSON repa a ``read_marker`` ji přečte zpět."""
    marker = SyntheticMarker(prefix="canary", rows=12)

    write_marker(tmp_path / "tables", marker)

    text = (tmp_path / "tables" / MARKER_NAME).read_text(encoding="utf-8")
    assert text == '{\n  "prefix": "canary",\n  "rows": 12\n}\n'
    assert read_marker(tmp_path / "tables") == marker
