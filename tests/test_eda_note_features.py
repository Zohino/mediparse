"""Cache měření zpráv, stráž vstupu a kontrakt zapisovače pro R."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mediparse.domain.data_access import DataAccessError
from mediparse.entrypoints.eda_note_features import run
from mediparse.infrastructure import note_features_cache
from mediparse.infrastructure.note_features_cache import cache_key, is_current
from mediparse.infrastructure.note_features_table import ARTIFACTS, ParquetNoteFeatures
from tests.note_features_support import CONTRACT_NOTES, compute
from tests.support import (
    REPOSITORY,
    REPOSITORY_DIAGNOSIS_CANDIDATES,
    REPOSITORY_SECTION_HEADERS,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Připraví tabulku syntetiky a soukromé kopie konfigurace.

    Returns:
        Argumenty nástroje kromě prostředí.
    """
    parquet = tmp_path / "tables"
    parquet.mkdir()
    (parquet / "synthetic.json").write_text('{"prefix":"t","rows":3}', encoding="utf-8")
    pq.write_table(
        pa.table({
            "note_id": [row[0] for row in CONTRACT_NOTES],
            "text": [row[1] for row in CONTRACT_NOTES],
        }),
        parquet / "discharge.parquet",
    )
    lexicon, candidates = tmp_path / "lexicon.json", tmp_path / "candidates.json"
    lexicon.write_bytes(REPOSITORY_SECTION_HEADERS.read_bytes())
    candidates.write_bytes(REPOSITORY_DIAGNOSIS_CANDIDATES.read_bytes())
    return parquet, lexicon, candidates, tmp_path / "cache"


def test_second_run_preserves_bytes_mtimes_and_matches_fresh_computation(
    inputs: tuple[Path, Path, Path, Path],
) -> None:
    """Cache nemění soubory a její obsah odpovídá nezávislému běhu bez cache."""
    parquet, lexicon, candidates, out = inputs
    assert run(*inputs, {"CLAUDECODE": "1"}) is True
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in out.iterdir()
    }
    assert run(*inputs, {"CLAUDECODE": "1"}) is False
    assert before == {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in out.iterdir()
    }
    fresh = out.parent / "fresh"
    assert run(parquet, lexicon, candidates, fresh, {"CLAUDECODE": "1"}) is True
    for name, _ in ARTIFACTS:
        assert pq.read_table(fresh / name).equals(pq.read_table(out / name))


@pytest.mark.parametrize("changed", ["lexicon", "candidates", "discharge"])
def test_changed_input_recomputes(
    inputs: tuple[Path, Path, Path, Path], changed: str
) -> None:
    """I nevýznamný bílý znak configu či jiné bajty parquetu invalidují cache."""
    parquet, lexicon, candidates, out = inputs
    assert run(*inputs, {}) is True
    old = json.loads((out / "cache.json").read_text(encoding="utf-8"))
    path = {
        "lexicon": lexicon,
        "candidates": candidates,
        "discharge": parquet / "discharge.parquet",
    }[changed]
    if changed == "discharge":
        table = pq.read_table(path)
        pq.write_table(table.replace_schema_metadata({b"changed": b"1"}), path)
    else:
        path.write_bytes(path.read_bytes() + b" ")
    assert run(*inputs, {}) is True
    assert json.loads((out / "cache.json").read_text(encoding="utf-8")) != old


@pytest.mark.parametrize("changed", ["code", "version", "format"])
def test_implementation_changes_invalidate_key(
    inputs: tuple[Path, Path, Path, Path],
    changed: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Klíč pokrývá obsah i relativní cesty Pythonu, verzi sklearn a formát."""
    parquet, lexicon, candidates, _ = inputs
    package = tmp_path / "package"
    package.mkdir()
    module = package / "one.py"
    module.write_text("one", encoding="utf-8")
    monkeypatch.setattr(note_features_cache, "PACKAGE", package)
    before = cache_key(parquet / "discharge.parquet", lexicon, candidates)
    if changed == "code":
        module.rename(package / "two.py")
    elif changed == "version":
        monkeypatch.setattr(note_features_cache.sklearn, "__version__", "changed")
    else:
        monkeypatch.setattr(note_features_cache, "FORMAT_VERSION", "changed")
    assert cache_key(parquet / "discharge.parquet", lexicon, candidates) != before


def test_missing_artifact_or_invalid_manifest_is_not_current(
    inputs: tuple[Path, Path, Path, Path],
) -> None:
    """Klíč bez všech tří artefaktů ani poškozený manifest nestačí."""
    parquet, lexicon, candidates, out = inputs
    run(*inputs, {})
    key = cache_key(parquet / "discharge.parquet", lexicon, candidates)
    (out / ARTIFACTS[0][0]).unlink()
    assert not is_current(out, key)
    run(*inputs, {})
    (out / "cache.json").write_text("[]", encoding="utf-8")
    assert not is_current(out, key)


def test_interrupted_sink_does_not_publish_cache(
    inputs: tuple[Path, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Výjimka po zápisu dávky nenechá platný manifest ani dočasné soubory."""
    original = pq.ParquetWriter.write_table

    def interrupted(self: pq.ParquetWriter, table: pa.Table) -> None:
        original(self, table)
        msg = "přerušený sink"
        raise RuntimeError(msg)

    monkeypatch.setattr(pq.ParquetWriter, "write_table", interrupted)
    with pytest.raises(RuntimeError, match="přerušený sink"):
        run(*inputs, {})
    assert not (inputs[3] / "cache.json").exists()
    assert list(inputs[3].iterdir()) == []


def test_guard_precedes_any_parquet_read(inputs: tuple[Path, Path, Path, Path]) -> None:
    """Bez značky syntetiky stráž odmítne i nečitelný parquet před hashováním."""
    parquet, _, _, out = inputs
    (parquet / "synthetic.json").unlink()
    (parquet / "discharge.parquet").write_bytes(b"not parquet")
    with pytest.raises(DataAccessError):
        run(*inputs, {"CLAUDECODE": "1"})
    assert not out.exists()


def test_contract_fixtures_equal_writer(tmp_path: Path) -> None:
    """Fixtury konzumenta jsou přesně výstup zapisovače nad zkonstruovanými zprávami."""
    with ParquetNoteFeatures(tmp_path).writer() as sink:
        assert compute(CONTRACT_NOTES, sink) == len(CONTRACT_NOTES)
    for name, schema in ARTIFACTS:
        actual = pq.read_table(tmp_path / name)
        assert actual.schema.equals(schema)
        assert all(not field.nullable for field in actual.schema)
        assert actual.equals(
            pq.read_table(REPOSITORY / "tests/fixtures/contracts" / name)
        )


def test_empty_input_writes_three_typed_files(tmp_path: Path) -> None:
    """Prázdný zdroj má tři čitelné prázdné artefakty s pevným schématem."""
    with ParquetNoteFeatures(tmp_path).writer() as sink:
        assert compute([], sink) == 0
    for name, schema in ARTIFACTS:
        table = pq.read_table(tmp_path / name)
        assert table.schema.equals(schema)
        assert table.num_rows == 0
