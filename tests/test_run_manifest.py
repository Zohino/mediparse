"""Schéma run manifestu: přísná validace revize zdroje a otisků."""

from __future__ import annotations

from typing import Final

import pytest
from pydantic import ValidationError

from mediparse.domain.run_manifest import Environment, RunManifest, SourceRevision

COMMIT: Final = "a" * 40
DIGEST: Final = "b" * 64


def _fields(**changes: object) -> dict[str, object]:
    fields: dict[str, object] = {
        "row_id": "row",
        "dataset_sha256": DIGEST,
        "config_sha256": DIGEST,
        "seed": 0,
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


@pytest.mark.parametrize("commit", ["a" * 7, "A" * 40, "a" * 41])
def test_invalid_commit_is_rejected(commit: str) -> None:
    """Zkrácený, velkými písmeny psaný i příliš dlouhý commit je odmítnut."""
    with pytest.raises(ValidationError):
        SourceRevision(commit=commit, dirty=False)


@pytest.mark.parametrize("field", ["dataset_sha256", "config_sha256"])
@pytest.mark.parametrize("digest", ["b" * 63, "b" * 65])
def test_digest_of_other_length_is_rejected(field: str, digest: str) -> None:
    """Hash jiné délky než 64 znaků je odmítnut."""
    with pytest.raises(ValidationError):
        RunManifest.model_validate(_fields(**{field: digest}))


def test_extra_field_is_rejected() -> None:
    """Pole navíc je odmítnuto."""
    with pytest.raises(ValidationError):
        RunManifest.model_validate(_fields(started="now"))


@pytest.mark.parametrize("row_id", ["", "Row 1"])
def test_invalid_row_id_is_rejected(row_id: str) -> None:
    """Prázdné id i id mimo skladbu řádku matice manifest odmítne."""
    with pytest.raises(ValidationError):
        RunManifest.model_validate(_fields(row_id=row_id))


def test_environment_rejects_unknown_field_and_non_strict_types() -> None:
    """Prostředí je přísné: cizí pole a balíček s jinou verzí než text neprojdou."""
    source = SourceRevision(commit=COMMIT, dirty=False)

    with pytest.raises(ValidationError):
        Environment.model_validate({"packages": {}, "source": source, "extra": 1})
    with pytest.raises(ValidationError):
        Environment.model_validate({"packages": {"python": 3}, "source": source})


def test_manifest_nests_environment_instead_of_flat_fields() -> None:
    """Serializovaný manifest nese ``environment`` a nemá ``packages`` ani ``source`` nahoře."""
    dumped = RunManifest.model_validate(_fields()).model_dump(mode="json")

    assert dumped["environment"] == {
        "packages": {"python": "3.13.0"},
        "source": {"commit": COMMIT, "dirty": False},
    }
    assert "packages" not in dumped
    assert "source" not in dumped
