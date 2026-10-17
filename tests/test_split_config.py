"""Načtení configu splitu."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.split_config import load_split_config
from tests.model_support import split_config_json
from tests.support import REPOSITORY_SPLIT_CONFIG

if TYPE_CHECKING:
    from pathlib import Path


def test_repository_config_loads() -> None:
    """Config z repa projde schématem."""
    config = load_split_config(REPOSITORY_SPLIT_CONFIG)

    assert (config.seed, config.folds) == (0, 5)


def test_missing_file_is_invalid_input(tmp_path: Path) -> None:
    """Chybějící soubor je InvalidInputError se jménem souboru."""
    with pytest.raises(InvalidInputError, match=r"missing\.json"):
        load_split_config(tmp_path / "missing.json")


def test_wrong_schema_is_invalid_input(tmp_path: Path) -> None:
    """Config se špatným schématem je InvalidInputError se jménem souboru."""
    path = tmp_path / "split.json"
    path.write_text(json.dumps(split_config_json(folds=1)), encoding="utf-8")

    with pytest.raises(InvalidInputError, match=r"split\.json"):
        load_split_config(path)
