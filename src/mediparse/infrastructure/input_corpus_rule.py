"""Načtení sdílené konfigurace vstupního korpusu z JSON."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mediparse.domain.cohort import InputCorpusRule
from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.input_file import parse_file

if TYPE_CHECKING:
    from pathlib import Path


def load_input_corpus_rule(path: Path) -> InputCorpusRule:
    """Přečte a validuje pravidlo vstupního korpusu.

    Returns:
        Pravidlo se striktně celočíselným prahem délky.

    Raises:
        InvalidInputError: Soubor nelze přečíst nebo neodpovídá schématu.
    """
    try:
        return parse_file(path, InputCorpusRule.model_validate_json)
    except OSError as error:
        msg = f"Soubor {path} nelze přečíst."
        raise InvalidInputError(msg) from error
