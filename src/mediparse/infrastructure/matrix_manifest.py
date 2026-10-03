"""Načtení manifestu experimentální matice z JSON souboru."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mediparse.domain.matrix import ExperimentMatrix
from mediparse.infrastructure.input_file import parse_file

if TYPE_CHECKING:
    from pathlib import Path


def load_matrix(path: Path) -> ExperimentMatrix:
    """Přečte JSON manifest a předá ho validaci.

    Returns:
        Matice, která prošla schématem i invarianty.
    """
    return parse_file(path, ExperimentMatrix.model_validate_json)
