"""Sdílené volby vstupu modelu a význam úrovní předzpracování."""

import json

import pytest
from pydantic import ValidationError

from mediparse.domain.model_input import ModelInput, TextPreprocessing


@pytest.mark.parametrize(
    ("level", "normalizes", "rewrites_variants"),
    [
        (TextPreprocessing.NONE, False, False),
        (TextPreprocessing.NORMALIZED, True, False),
        (TextPreprocessing.DICTIONARY, True, True),
    ],
)
def test_preprocessing_level_properties(
    level: TextPreprocessing, *, normalizes: bool, rewrites_variants: bool
) -> None:
    """Žebříček rozlišuje původní text, normalizaci a přepis variant slovníkem."""
    assert level.normalizes is normalizes
    assert level.rewrites_variants is rewrites_variants


@pytest.mark.parametrize(
    ("fields", "field"),
    [
        ({"text_preprocessing": "none"}, "strip_diagnostic_sections"),
        ({"strip_diagnostic_sections": False}, "text_preprocessing"),
        (
            {"strip_diagnostic_sections": "true", "text_preprocessing": "none"},
            "strip_diagnostic_sections",
        ),
        (
            {"strip_diagnostic_sections": 1, "text_preprocessing": "none"},
            "strip_diagnostic_sections",
        ),
        (
            {"strip_diagnostic_sections": None, "text_preprocessing": "none"},
            "strip_diagnostic_sections",
        ),
        (
            {"strip_diagnostic_sections": False, "text_preprocessing": "full"},
            "text_preprocessing",
        ),
        (
            {"strip_diagnostic_sections": False, "text_preprocessing": True},
            "text_preprocessing",
        ),
        (
            {"strip_diagnostic_sections": False, "text_preprocessing": None},
            "text_preprocessing",
        ),
        (
            {
                "strip_diagnostic_sections": False,
                "text_preprocessing": "none",
                "extra": 0,
            },
            "extra",
        ),
    ],
)
def test_model_input_rejects_invalid_fields(
    fields: dict[str, object], field: str
) -> None:
    """Podmodel odmítne chybějící pole, neznámé hodnoty, typové záměny i pole navíc."""
    with pytest.raises(ValidationError, match=field):
        ModelInput.model_validate_json(json.dumps(fields))
