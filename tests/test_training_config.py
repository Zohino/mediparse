"""Načtení configu tréninku smoketestu."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.labels import Diagnosis
from mediparse.infrastructure.training_config import (
    TRAINING_CONFIG_PATH,
    load_training_config,
)
from tests.support import REPOSITORY

if TYPE_CHECKING:
    from pathlib import Path


def test_repository_config_loads() -> None:
    """Config z repa projde schématem."""
    config = load_training_config(REPOSITORY / TRAINING_CONFIG_PATH)

    assert config.diagnosis is Diagnosis.DIABETES
    assert config.regularization > 0


def test_missing_file_is_invalid_input(tmp_path: Path) -> None:
    """Chybějící soubor je InvalidInputError."""
    with pytest.raises(InvalidInputError, match="neexistuje"):
        load_training_config(tmp_path / "missing.json")


def test_wrong_schema_is_invalid_input(tmp_path: Path) -> None:
    """Config se špatným schématem je InvalidInputError."""
    path = tmp_path / "config.json"
    path.write_text('{"diagnosis": "diabetes"}', encoding="utf-8")

    with pytest.raises(InvalidInputError, match="neodpovídá schématu"):
        load_training_config(path)
