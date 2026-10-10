"""Zápis run manifestu do JSON a zpětné načtení."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mediparse.domain.run_manifest import RunManifest, SourceRevision
from mediparse.infrastructure.run_manifest_file import JsonRunManifestFile

if TYPE_CHECKING:
    from pathlib import Path

MANIFEST = RunManifest(
    row_id="row",
    dataset_sha256="b" * 64,
    config_sha256="c" * 64,
    seed=0,
    model="sklearn TfidfVectorizer+LinearSVC",
    model_revision=None,
    tokenizer_revision=None,
    packages={"python": "3.13.0"},
    source=SourceRevision(commit="a" * 40, dirty=True),
)


def test_round_trip(tmp_path: Path) -> None:
    """Zapsaný JSON se zpět validuje na stejný objekt a končí novým řádkem."""
    path = tmp_path / "manifest.json"

    JsonRunManifestFile(path).write(MANIFEST)

    text = path.read_text(encoding="utf-8")
    assert RunManifest.model_validate_json(text) == MANIFEST
    assert text.endswith("}\n")
    assert '\n  "row_id"' in text


def test_missing_directory_is_created(tmp_path: Path) -> None:
    """Chybějící nadřazený adresář se založí."""
    path = tmp_path / "build" / "smoketest" / "manifest.json"

    JsonRunManifestFile(path).write(MANIFEST)

    assert path.is_file()
