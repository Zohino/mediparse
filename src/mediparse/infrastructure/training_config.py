"""Načtení configu tréninku smoketestu z JSON souboru."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mediparse.domain.smoketest_training import TrainingConfig
from mediparse.infrastructure.input_file import parse_file

if TYPE_CHECKING:
    from pathlib import Path


def load_training_config(path: Path) -> TrainingConfig:
    """Přečte JSON config a předá ho validaci.

    Returns:
        Config, který prošel schématem.
    """
    return parse_file(path, TrainingConfig.model_validate_json)
