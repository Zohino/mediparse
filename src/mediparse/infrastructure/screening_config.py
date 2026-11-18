"""Načtení configu screeningu stupně z JSON souboru."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mediparse.domain.screening import ScreeningConfig
from mediparse.infrastructure.input_file import parse_file

if TYPE_CHECKING:
    from pathlib import Path


def load_screening_config(path: Path) -> ScreeningConfig:
    """Přečte JSON a ověří mřížku screeningu.

    Returns:
        Striktně validovaný config screeningu.
    """
    return parse_file(path, ScreeningConfig.model_validate_json)
