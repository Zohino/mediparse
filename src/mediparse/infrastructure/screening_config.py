"""Načtení configu klasického screeningu z JSON souboru."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mediparse.domain.classical_screening import ClassicalScreeningConfig
from mediparse.infrastructure.input_file import parse_file

if TYPE_CHECKING:
    from pathlib import Path


def load_classical_screening_config(path: Path) -> ClassicalScreeningConfig:
    """Přečte JSON a ověří mřížku screeningu.

    Returns:
        Striktně validovaný config screeningu.
    """
    return parse_file(path, ClassicalScreeningConfig.model_validate_json)
