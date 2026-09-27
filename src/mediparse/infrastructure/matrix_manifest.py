"""Načtení manifestu experimentální matice z JSON souboru."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mediparse.domain.matrix import ExperimentMatrix

if TYPE_CHECKING:
    from pathlib import Path


def load_matrix(path: Path) -> ExperimentMatrix:
    """Přečte JSON manifest a předá ho validaci; neplatný obsah vyhodí ValidationError.

    Returns:
        Matice, která prošla schématem i invarianty.
    """
    return ExperimentMatrix.model_validate_json(path.read_text(encoding="utf-8"))
