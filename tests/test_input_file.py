"""Čtení vstupního souboru: chybějící soubor i chyba schématu jsou doménová chyba vstupu."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.synthetic_plan import SamplerConfig
from mediparse.infrastructure.input_file import parse_file

if TYPE_CHECKING:
    from pathlib import Path


def test_missing_file_is_invalid_input(tmp_path: Path) -> None:
    """Chybějící soubor se ohlásí jménem, pydantic ani OSError za adaptér neprosáknou."""
    with pytest.raises(InvalidInputError, match="neexistuje"):
        parse_file(tmp_path / "missing.json", SamplerConfig.model_validate_json)


def test_invalid_content_is_invalid_input(tmp_path: Path) -> None:
    """Obsah, který neodpovídá schématu, se ohlásí jménem souboru."""
    path = tmp_path / "config.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(InvalidInputError, match="neodpovídá schématu"):
        parse_file(path, SamplerConfig.model_validate_json)
