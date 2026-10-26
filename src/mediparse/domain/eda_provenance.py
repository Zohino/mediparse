"""České popisky a textové hodnoty přílohy provenance renderu EDA."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping

PROVENANCE_LABELS: Final[Mapping[str, str]] = {
    "commit": "Revize kódu (commit)",
    "dirty": "Necommitnuté změny",
    "pixi_lock_sha256": "Otisk pixi.lock (SHA-256)",
    "data_source": "Zdroj dat",
    "python": "Python",
    "duckdb": "DuckDB",
    "r": "R",
}


def _value(key: str, value: str) -> str:
    if key == "dirty":
        return "ano" if value == "true" else "ne"
    if re.fullmatch(r"[0-9a-f]{64}", value):
        return " ".join(value[start : start + 16] for start in range(0, 64, 16))
    return value


def provenance_rows(details: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    """Převede provenance na textové řádky ve vstupním pořadí klíčů.

    Args:
        details: Klíče a textové hodnoty provenance renderu.

    Returns:
        České popisky a hodnoty; SHA-256 je členěn po šestnácti znacích.
    """
    return tuple(
        (
            PROVENANCE_LABELS.get(
                key, re.sub(r"^csv_sha256 (.*)$", r"Otisk \1 (SHA-256)", key)
            ),
            _value(key, value),
        )
        for key, value in details.items()
    )
