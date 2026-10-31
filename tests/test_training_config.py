"""Načtení configu tréninku smoketestu."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.training import TrainingConfig
from mediparse.infrastructure.training_config import load_training_config
from tests.model_support import training_config
from tests.support import REPOSITORY_TRAINING_CONFIG

if TYPE_CHECKING:
    from pathlib import Path


def test_repository_config_loads() -> None:
    """Config z repa projde schématem."""
    config = load_training_config(REPOSITORY_TRAINING_CONFIG)

    assert config.row_id == "smoketest-en-svm-strip"
    assert config.strip_diagnostic_sections is True
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


def test_config_without_strip_switch_is_rejected() -> None:
    """Přepínač stripu je povinný, bez výchozí hodnoty."""
    fields = training_config()
    del fields["strip_diagnostic_sections"]

    with pytest.raises(ValidationError, match="strip_diagnostic_sections"):
        TrainingConfig.model_validate_json(json.dumps(fields))


def test_strip_switch_as_string_is_rejected() -> None:
    """Přepínač je striktní bool, řetězec ``"true"`` neprojde."""
    with pytest.raises(ValidationError, match="strip_diagnostic_sections"):
        TrainingConfig.model_validate_json(
            json.dumps(training_config(strip_diagnostic_sections="true"))
        )


def test_repository_config_normalizes() -> None:
    """Smoketest běží s normalizací."""
    assert load_training_config(REPOSITORY_TRAINING_CONFIG).normalize_clinical_text


def test_config_without_normalization_switch_is_rejected() -> None:
    """Přepínač normalizace je povinný, bez výchozí hodnoty."""
    fields = training_config()
    del fields["normalize_clinical_text"]

    with pytest.raises(ValidationError, match="normalize_clinical_text"):
        TrainingConfig.model_validate_json(json.dumps(fields))


def test_normalization_switch_as_string_is_rejected() -> None:
    """Přepínač je striktní bool, řetězec ``"true"`` neprojde."""
    with pytest.raises(ValidationError, match="normalize_clinical_text"):
        TrainingConfig.model_validate_json(
            json.dumps(training_config(normalize_clinical_text="true"))
        )
