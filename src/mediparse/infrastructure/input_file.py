"""Čtení vstupního souboru kroku: chybějící soubor i chybu schématu překládá na doménovou výjimku."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import ValidationError

from mediparse.domain.inputs import InvalidInputError

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def parse_file[T](path: Path, parse: Callable[[str], T]) -> T:
    """Přečte soubor v UTF-8 a předá jeho text parseru.

    Returns:
        Výsledek parseru.

    Raises:
        InvalidInputError: Soubor neexistuje nebo jeho obsah neodpovídá schématu.
    """
    try:
        return parse(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        msg = f"Soubor {path} neexistuje."
        raise InvalidInputError(msg) from error
    except ValidationError as error:
        msg = f"Soubor {path} neodpovídá schématu:\n{error}"
        raise InvalidInputError(msg) from error
