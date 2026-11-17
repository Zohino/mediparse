"""Načtení obecného prostoru a protokolu ladění z JSON souboru."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mediparse.domain.tuning import TuningConfig
from mediparse.infrastructure.input_file import parse_file

if TYPE_CHECKING:
    from pathlib import Path


def load_tuning_config(path: Path) -> TuningConfig:
    """Přečte a striktně validuje konfiguraci ladění.

    Returns:
        Config ověřený doménovým schématem.
    """
    return parse_file(path, TuningConfig.model_validate_json)
