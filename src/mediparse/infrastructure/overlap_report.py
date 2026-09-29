"""Report shod auditu: pozice sdílených n-gramů a kolize subject_id, nikdy text zpráv."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence
    from pathlib import Path

    from mediparse.domain.corpus_audit import Position


def write_overlap_report(
    path: Path, positions: Sequence[Position], colliding_subjects: Collection[str]
) -> None:
    """Zapíše pozice shod a kolidující subject_id do JSON reportu."""
    content = {
        "positions": [
            {"note": position.note, "token": position.token} for position in positions
        ],
        "colliding_subjects": sorted(colliding_subjects),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(content, indent=2)}\n", encoding="utf-8")
