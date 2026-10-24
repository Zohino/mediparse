"""Schéma run manifestu: přísná validace otisků a vnoření prostředí."""

from __future__ import annotations

from typing import Final

import pytest
from pydantic import ValidationError

from mediparse.domain.environment import Environment, SourceRevision
from mediparse.domain.run_manifest import DiagnosticStrip, RunManifest

COMMIT: Final = "a" * 40
DIGEST: Final = "b" * 64


def _fields(**changes: object) -> dict[str, object]:
    fields: dict[str, object] = {
        "row_id": "row",
        "dataset_sha256": DIGEST,
        "config_sha256": DIGEST,
        "seed": 0,
        "split_seed": 1,
        "split_folds": 5,
        "split_sha256": DIGEST,
        "diagnostic_strip": None,
        "model": "sklearn TfidfVectorizer+LinearSVC",
        "model_revision": None,
        "tokenizer_revision": None,
        "environment": Environment(
            packages={"python": "3.13.0"},
            source=SourceRevision(commit=COMMIT, dirty=False),
        ),
    }
    return fields | changes


def test_valid_manifest_with_null_revisions() -> None:
    """Platný manifest projde a revize modelu i tokenizeru smí být null."""
    manifest = RunManifest.model_validate(_fields())

    assert manifest.model_revision is None
    assert manifest.tokenizer_revision is None
    assert manifest.environment.source.commit == COMMIT


@pytest.mark.parametrize("field", ["dataset_sha256", "config_sha256", "split_sha256"])
@pytest.mark.parametrize("digest", ["b" * 63, "b" * 65])
def test_digest_of_other_length_is_rejected(field: str, digest: str) -> None:
    """Hash jiné délky než 64 znaků je odmítnut."""
    with pytest.raises(ValidationError):
        RunManifest.model_validate(_fields(**{field: digest}))


@pytest.mark.parametrize("field", ["seed", "split_seed"])
@pytest.mark.parametrize("seed", [-1, 2**32])
def test_seed_outside_generator_range_is_rejected(field: str, seed: int) -> None:
    """Seed mimo rozsah generátoru manifest odmítne."""
    with pytest.raises(ValidationError):
        RunManifest.model_validate(_fields(**{field: seed}))


@pytest.mark.parametrize("folds", [0, 1])
def test_fewer_than_two_split_folds_are_rejected(folds: int) -> None:
    """Split má aspoň dva foldy, manifest s méně je odmítnut."""
    with pytest.raises(ValidationError):
        RunManifest.model_validate(_fields(split_folds=folds))


def test_extra_field_is_rejected() -> None:
    """Pole navíc je odmítnuto."""
    with pytest.raises(ValidationError):
        RunManifest.model_validate(_fields(started="now"))


@pytest.mark.parametrize("row_id", ["", "Row 1"])
def test_invalid_row_id_is_rejected(row_id: str) -> None:
    """Prázdné id i id mimo skladbu řádku matice manifest odmítne."""
    with pytest.raises(ValidationError):
        RunManifest.model_validate(_fields(row_id=row_id))


def test_manifest_nests_environment_instead_of_flat_fields() -> None:
    """Serializovaný manifest nese ``environment`` a nemá ``packages`` ani ``source`` nahoře."""
    dumped = RunManifest.model_validate(_fields()).model_dump(mode="json")

    assert dumped["environment"] == {
        "packages": {"python": "3.13.0"},
        "source": {"commit": COMMIT, "dirty": False},
    }
    assert "packages" not in dumped
    assert "source" not in dumped


def test_manifest_without_strip_serializes_null() -> None:
    """Bez stripu je ``diagnostic_strip`` ``null``."""
    dumped = RunManifest.model_validate(_fields()).model_dump(mode="json")

    assert dumped["diagnostic_strip"] is None


def test_strip_without_sections_is_rejected() -> None:
    """Záznam stripu bez sekcí nelze zapsat."""
    with pytest.raises(ValidationError):
        DiagnosticStrip.model_validate({"sections": (), "lexicon_sha256": DIGEST})


def test_strip_serializes_sections_as_sorted_list() -> None:
    """Sekce zadané v jiném pořadí jdou do JSON jako seřazený seznam."""
    strip = DiagnosticStrip(sections=("b_section", "a_section"), lexicon_sha256=DIGEST)
    manifest = RunManifest.model_validate(_fields(diagnostic_strip=strip))

    dumped = manifest.model_dump(mode="json")

    assert dumped["diagnostic_strip"] == {
        "sections": ["a_section", "b_section"],
        "lexicon_sha256": DIGEST,
    }


def test_manifest_without_strip_key_is_rejected() -> None:
    """Klíč ``diagnostic_strip`` je povinný, chybějící klíč není totéž co ``null``."""
    fields = _fields()
    del fields["diagnostic_strip"]

    with pytest.raises(ValidationError, match="diagnostic_strip"):
        RunManifest.model_validate(fields)


def test_strip_with_extra_field_is_rejected() -> None:
    """Záznam stripu s polem navíc je odmítnut."""
    with pytest.raises(ValidationError, match="extra"):
        DiagnosticStrip.model_validate({
            "sections": ("a",),
            "lexicon_sha256": DIGEST,
            "extra": 1,
        })


def test_strip_duplicate_sections_are_merged() -> None:
    """Opakovaný klíč sekce se sloučí, záznam říká, které sekce se odstranily."""
    strip = DiagnosticStrip(sections=("b", "a", "b"), lexicon_sha256=DIGEST)

    assert strip.sections == ("a", "b")
