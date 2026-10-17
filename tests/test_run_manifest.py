"""Schéma run manifestu: přísná validace otisků a vnoření prostředí."""

from __future__ import annotations

from typing import Final

import pytest
from pydantic import ValidationError

from mediparse.domain.environment import Environment, SourceRevision
from mediparse.domain.run_manifest import RunManifest

COMMIT: Final = "a" * 40
DIGEST: Final = "b" * 64


def _fields(**changes: object) -> dict[str, object]:
    fields: dict[str, object] = {
        "row_id": "row",
        "dataset_sha256": DIGEST,
        "config_sha256": DIGEST,
        "seed": 0,
        "split_seed": 1,
        "split_sha256": DIGEST,
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
