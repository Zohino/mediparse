"""Export tabulek ze sešitu podle explicitně předaného prostředí Quarta."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.thesis_export import ThesisExport

if TYPE_CHECKING:
    from collections.abc import Mapping

DOCUMENT_FILE_VARIABLE: Final = "QUARTO_DOCUMENT_FILE"
DOCUMENT_PATH_VARIABLE: Final = "QUARTO_DOCUMENT_PATH"


def export(environ: Mapping[str, str]) -> ThesisExport:
    """Odvodí adresář exportů ze jména a adresáře dokumentu Quarta.

    Args:
        environ: Prostředí renderu; obě proměnné Quarta musí být neprázdné.

    Returns:
        Exportér do results/eda pod rodičem adresáře dokumentu.

    Raises:
        InvalidInputError: Chybí některá proměnná Quarta.
    """
    for variable in (DOCUMENT_PATH_VARIABLE, DOCUMENT_FILE_VARIABLE):
        if not environ.get(variable):
            msg = f"Pro export tabulek nastavte {variable}."
            raise InvalidInputError(msg)
    parent = Path(environ[DOCUMENT_PATH_VARIABLE]).parent
    stem = Path(environ[DOCUMENT_FILE_VARIABLE]).stem
    return ThesisExport(parent / "results" / "eda" / stem)
