"""Načtení lexikonu sekcí z configu."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.section_lexicon_config import load_section_lexicon
from tests.support import REPOSITORY_SECTION_HEADERS

if TYPE_CHECKING:
    from pathlib import Path

SECTION_COUNT = 17


def test_repository_lexicon_loads() -> None:
    """Lexikon z repa projde schématem."""
    lexicon = load_section_lexicon(REPOSITORY_SECTION_HEADERS)

    assert len(lexicon.sections) == SECTION_COUNT
    assert {"discharge_diagnosis", "discharge_condition"} <= set(lexicon.sections)


def test_missing_file_is_invalid_input(tmp_path: Path) -> None:
    """Chybějící soubor je InvalidInputError se jménem souboru."""
    with pytest.raises(InvalidInputError, match=r"missing\.json"):
        load_section_lexicon(tmp_path / "missing.json")


def test_wrong_schema_is_invalid_input(tmp_path: Path) -> None:
    """Lexikon se špatným schématem je InvalidInputError se jménem souboru."""
    path = tmp_path / "headers.json"
    path.write_text(json.dumps({"sections": {}}), encoding="utf-8")

    with pytest.raises(InvalidInputError, match=r"headers\.json"):
        load_section_lexicon(path)
