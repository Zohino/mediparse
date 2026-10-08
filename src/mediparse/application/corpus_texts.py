"""Texty zpráv jednoho jazyka z korpusu ve tvaru ``<jazyk>/<note_id>.txt``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping

LANGUAGE: Final = "en"
TRANSLATION_LANGUAGE: Final = "cs"
_SUFFIX: Final = ".txt"


def texts_in_language(notes: Mapping[str, str], language: str) -> dict[str, str]:
    """Texty zpráv jednoho jazyka klíčované note_id.

    Args:
        notes: Texty klíčované relativní cestou ``<jazyk>/<note_id>.txt``.
        language: Jazyk, jehož zprávy se vyberou.

    Returns:
        Slovník note_id → text; zprávy ostatních jazyků vynechá.
    """
    prefix = f"{language}/"
    return {
        path.removeprefix(prefix).removesuffix(_SUFFIX): text
        for path, text in notes.items()
        if path.startswith(prefix)
    }
