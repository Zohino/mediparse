"""Report shod: JSON s odsazením 2 a koncovým řádkem, pozice v pořadí vstupu, kolize seřazené."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from mediparse.domain.corpus_audit import Position
from mediparse.infrastructure.overlap_report import OverlapReportFile

if TYPE_CHECKING:
    from pathlib import Path


def test_empty_report_has_empty_lists(tmp_path: Path) -> None:
    """Bez pozic a kolizí vzniknou prázdná pole."""
    path = tmp_path / "out" / "report.json"

    OverlapReportFile(path).write([], set())

    expected = {"positions": [], "colliding_subjects": []}
    assert path.read_text(encoding="utf-8") == f"{json.dumps(expected, indent=2)}\n"


def test_positions_keep_order_and_subjects_are_sorted(tmp_path: Path) -> None:
    """Pozice zůstanou v pořadí vstupu, kolidující subject_id se seřadí."""
    path = tmp_path / "report.json"

    OverlapReportFile(path).write([Position("n2", 5), Position("n1", 0)], {30, 10, 20})

    expected = {
        "positions": [{"note": "n2", "token": 5}, {"note": "n1", "token": 0}],
        "colliding_subjects": [10, 20, 30],
    }
    assert path.read_text(encoding="utf-8") == f"{json.dumps(expected, indent=2)}\n"
