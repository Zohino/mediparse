"""Společné vstupy a referenční výstupy tabulek pro práci."""

import json
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures" / "thesis_table"
CASES = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))


def golden_case(name: str) -> dict:
    """Najde referenční vstup podle jména, nezávisle na pořadí případů.

    Args:
        name: Jméno případu v cases.json.

    Returns:
        Hlavička, řádky a rozvržení pojmenovaného případu.
    """
    return next(case for case in CASES if case["name"] == name)
