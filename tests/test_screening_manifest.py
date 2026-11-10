"""Striktní provenance screeningu a soulad úrovně předzpracování se slovníky."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mediparse.domain.clinical_text import TextNormalization
from mediparse.domain.environment import Environment
from mediparse.domain.model_input import TextPreprocessing
from mediparse.domain.run_manifest import ScreeningManifest
from tests.support import REVISION


def _fields(**changes: object) -> dict[str, object]:
    return {
        "dataset_sha256": "a" * 64,
        "split_seed": 23,
        "split_folds": 2,
        "split_sha256": "b" * 64,
        "config_sha256": "c" * 64,
        "seed": 7,
        "results_sha256": "d" * 64,
        "predictions_sha256": "e" * 64,
        "development_notes": 40,
        "development_patients": 20,
        "diagnostic_strip": None,
        "text_preprocessing": TextPreprocessing.NONE,
        "text_normalization": None,
        "environment": Environment(packages={"python": "3.13.0"}, source=REVISION),
    } | changes


@pytest.mark.parametrize("field", list(_fields()))
def test_screening_manifest_requires_every_field(field: str) -> None:
    """Každý článek provenance je povinný, i když smí nést null."""
    fields = _fields()
    del fields[field]
    with pytest.raises(ValidationError, match=field):
        ScreeningManifest.model_validate(fields)


@pytest.mark.parametrize(
    ("level", "normalization"),
    [
        (TextPreprocessing.NONE, TextNormalization(acronyms={}, variants={})),
        (TextPreprocessing.NORMALIZED, None),
        (TextPreprocessing.DICTIONARY, None),
        (
            TextPreprocessing.NORMALIZED,
            TextNormalization(acronyms={}, variants={"kindey": "kidney"}),
        ),
    ],
)
def test_screening_rejects_inconsistent_preprocessing(
    level: TextPreprocessing, normalization: TextNormalization | None
) -> None:
    """None se slovníkem a normalized s variantami odporují úrovni předzpracování."""
    with pytest.raises(ValidationError, match="předzpracování"):
        ScreeningManifest.model_validate(
            _fields(text_preprocessing=level, text_normalization=normalization)
        )


@pytest.mark.parametrize(
    ("level", "normalization"),
    [
        (TextPreprocessing.NONE, None),
        (
            TextPreprocessing.NORMALIZED,
            TextNormalization(
                acronyms={"COPD": "chronic obstructive pulmonary disease"}, variants={}
            ),
        ),
        (
            TextPreprocessing.DICTIONARY,
            TextNormalization(acronyms={}, variants={"kindey": "kidney"}),
        ),
    ],
)
def test_screening_accepts_consistent_preprocessing(
    level: TextPreprocessing, normalization: TextNormalization | None
) -> None:
    """Platná úroveň a použitý slovník přežijí serializaci i opětovné načtení."""
    manifest = ScreeningManifest.model_validate(
        _fields(text_preprocessing=level, text_normalization=normalization)
    )
    assert ScreeningManifest.model_validate_json(manifest.model_dump_json()) == manifest


@pytest.mark.parametrize(
    "field",
    [
        "dataset_sha256",
        "split_sha256",
        "config_sha256",
        "results_sha256",
        "predictions_sha256",
    ],
)
@pytest.mark.parametrize("value", ["a" * 63, "g" * 64, 1])
def test_screening_rejects_invalid_hash(field: str, value: object) -> None:
    """Všechny otisky jsou přísné řetězce SHA-256."""
    with pytest.raises(ValidationError, match=field):
        ScreeningManifest.model_validate(_fields(**{field: value}))


@pytest.mark.parametrize("field", ["development_notes", "development_patients"])
@pytest.mark.parametrize("value", [0, -1, "20", True])
def test_screening_requires_positive_strict_counts(field: str, value: object) -> None:
    """Velikost vývoje musí být kladné celé číslo, řetězec ani bool neprojdou."""
    with pytest.raises(ValidationError, match=field):
        ScreeningManifest.model_validate(_fields(**{field: value}))


def test_screening_forbids_extra_fields() -> None:
    """Screening nepřijme neznámé pole."""
    with pytest.raises(ValidationError, match="extra"):
        ScreeningManifest.model_validate(_fields(extra=1))


def test_screening_is_frozen() -> None:
    """Hotová provenance je neměnná."""
    manifest = ScreeningManifest.model_validate(_fields())
    with pytest.raises(ValidationError, match="frozen"):
        manifest.seed = 8  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize("folds", [0, 1])
def test_screening_requires_at_least_two_split_folds(folds: int) -> None:
    """Manifest screeningu odmítne split s méně než dvěma foldy."""
    with pytest.raises(ValidationError, match="split_folds"):
        ScreeningManifest.model_validate(_fields(split_folds=folds))
