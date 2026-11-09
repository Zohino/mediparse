"""Načtení configu tréninku modelu; repo config je config smoketestu."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.model_input import TextPreprocessing
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
    assert config.input.strip_diagnostic_sections is True
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


@pytest.mark.parametrize(
    "field", ["input", "strip_diagnostic_sections", "text_preprocessing"]
)
def test_input_fields_are_required(field: str) -> None:
    """Vstup i obě jeho volby jsou povinné, bez výchozích hodnot."""
    fields = training_config()
    if field == "input":
        del fields[field]
    else:
        input_fields: dict[str, object] = {
            "strip_diagnostic_sections": False,
            "text_preprocessing": "none",
        }
        del input_fields[field]
        fields["input"] = input_fields

    with pytest.raises(ValidationError, match=field):
        TrainingConfig.model_validate_json(json.dumps(fields))


@pytest.mark.parametrize(
    "input_fields",
    [
        {"strip_diagnostic_sections": "true", "text_preprocessing": "none"},
        {"strip_diagnostic_sections": 1, "text_preprocessing": "none"},
        {"strip_diagnostic_sections": None, "text_preprocessing": "none"},
        {"strip_diagnostic_sections": False, "text_preprocessing": "full"},
        {"strip_diagnostic_sections": False, "text_preprocessing": True},
        {"strip_diagnostic_sections": False, "text_preprocessing": None},
        {"strip_diagnostic_sections": False, "text_preprocessing": "none", "extra": 0},
    ],
)
def test_input_rejects_invalid_values(input_fields: dict[str, object]) -> None:
    """Vstup odmítne neznámé pole či úroveň a záměnu typu."""
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate_json(
            json.dumps(training_config(input=input_fields))
        )


@pytest.mark.parametrize(
    "field",
    [
        "normalize_clinical_text",
        "strip_diagnostic_sections",
        "normalize_diagnosis_variants",
    ],
)
def test_legacy_top_level_switches_are_rejected(field: str) -> None:
    """Staré přepínače na nejvyšší úrovni jsou pole navíc."""
    with pytest.raises(ValidationError, match=field):
        TrainingConfig.model_validate_json(
            json.dumps(training_config(**{field: False}))
        )


def test_smoketest_enables_dictionary() -> None:
    """Smoketest zapíná celou cestu včetně přepisu variant."""
    assert (
        load_training_config(REPOSITORY_TRAINING_CONFIG).input.text_preprocessing
        is TextPreprocessing.DICTIONARY
    )
