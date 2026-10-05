"""Načtení configu tréninku smoketestu z JSON souboru."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from mediparse.domain.smoketest_training import TrainingConfig
from mediparse.infrastructure.input_file import parse_file

TRAINING_CONFIG_PATH: Final = Path("config/smoketest_training.json")


def load_training_config(path: Path) -> TrainingConfig:
    """Přečte JSON config a předá ho validaci.

    Returns:
        Config, který prošel schématem.
    """
    return parse_file(path, TrainingConfig.model_validate_json)
