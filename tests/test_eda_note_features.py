"""Cache měření zpráv, stráž vstupu a kontrakt zapisovače pro R."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mediparse.domain.data_access import DataAccessError
from mediparse.domain.inputs import InvalidInputError
from mediparse.entrypoints.eda_note_features import main, run
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure import eda_cache, note_features_table
from mediparse.infrastructure.eda_cache import is_current
from mediparse.infrastructure.note_features_table import ARTIFACTS, ParquetNoteFeatures
from mediparse.infrastructure.note_features_table import note_features_key as cache_key
from tests.note_features_support import CONTRACT_NOTES, compute
from tests.support import (
    REPOSITORY,
    REPOSITORY_DIAGNOSIS_CANDIDATES,
    REPOSITORY_SECTION_HEADERS,
)


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
        actual = pq.read_table(out / name)
        assert pq.read_table(fresh / name).equals(actual)
        assert actual.equals(
            pq.read_table(REPOSITORY / "tests/fixtures/contracts" / name)
        )
    first = pq.read_table(out / "note_preprocessing.parquet").to_pylist()[0]
    # Ručně: raw má deset tokenů; normalizace rozdělí 5mg a rozvine AF, obojí přidá jeden.
    assert (first["tokens_raw"], first["tokens_preprocessed"]) == (10, 12)


@pytest.mark.parametrize(
    "changed", ["lexicon", "candidates", "variant-terms", "discharge"]
)
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
        "variant-terms": candidates,
        "discharge": parquet / "discharge.parquet",
    }[changed]
    if changed == "discharge":
        table = pq.read_table(path)
        pq.write_table(table.replace_schema_metadata({b"changed": b"1"}), path)
    elif changed == "variant-terms":
        tree = json.loads(path.read_text(encoding="utf-8"))
        tree["candidates"]["diabetes"]["variants_en"].append("syntheticvariant")
        path.write_text(json.dumps(tree), encoding="utf-8")
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
    monkeypatch.setattr(eda_cache, "PACKAGE", package)
    before = cache_key(parquet / "discharge.parquet", lexicon, candidates)
    if changed == "code":
        module.rename(package / "two.py")
    elif changed == "version":
        monkeypatch.setattr(note_features_table.sklearn, "__version__", "changed")
    else:
        monkeypatch.setattr(eda_cache, "FORMAT_VERSION", "changed")
    assert cache_key(parquet / "discharge.parquet", lexicon, candidates) != before


def test_missing_artifact_or_invalid_manifest_is_not_current(
    inputs: tuple[Path, Path, Path, Path],
) -> None:
    """Klíč bez všech čtyř artefaktů ani poškozený manifest nestačí."""
    parquet, lexicon, candidates, out = inputs
    run(*inputs, {})
    key = cache_key(parquet / "discharge.parquet", lexicon, candidates)
    (out / ARTIFACTS[0][0]).unlink()
    assert not is_current(out, key, [name for name, _ in ARTIFACTS])
    run(*inputs, {})
    (out / "cache.json").write_text("[]", encoding="utf-8")
    assert not is_current(out, key, [name for name, _ in ARTIFACTS])


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


def test_guard_precedes_any_parquet_read(
    inputs: tuple[Path, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bez značky syntetiky stráž odmítne i nečitelný parquet před hashováním."""
    parquet, _, _, out = inputs
    (parquet / "synthetic.json").unlink()
    (parquet / "discharge.parquet").write_bytes(b"not parquet")
    hash_file = Mock(return_value="0" * 64)
    open_parquet = Mock(wraps=pq.ParquetFile)
    monkeypatch.setattr(note_features_table, "file_sha256", hash_file)
    monkeypatch.setattr(pq, "ParquetFile", open_parquet)
    with pytest.raises(DataAccessError):
        run(*inputs, {"CLAUDECODE": "1"})
    hash_file.assert_not_called()
    open_parquet.assert_not_called()
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


def test_empty_input_writes_four_typed_files(tmp_path: Path) -> None:
    """Prázdný zdroj má čtyři čitelné prázdné artefakty s pevným schématem."""
    with ParquetNoteFeatures(tmp_path).writer() as sink:
        assert compute([], sink) == 0
    for name, schema in ARTIFACTS:
        table = pq.read_table(tmp_path / name)
        assert table.schema.equals(schema)
        assert table.num_rows == 0


def test_r_aggregation_fixtures_equal_writer(tmp_path: Path) -> None:
    """Pevné vstupy agregací R mají stejný kontrakt a výstup zapisovače."""
    directory = REPOSITORY / "tests/fixtures/note_features"
    notes = pq.read_table(directory / "notes.parquet").to_pylist()
    with ParquetNoteFeatures(tmp_path).writer() as sink:
        compute([(row["note_id"], row["text"]) for row in notes], sink)
    for name, _ in ARTIFACTS:
        assert pq.read_table(tmp_path / name).equals(pq.read_table(directory / name))


def test_interrupted_publication_invalidates_old_and_new_key(
    inputs: tuple[Path, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Směs artefaktů po přerušení publikace nesmí zplatnit starý ani nový klíč."""
    parquet, lexicon, candidates, out = inputs
    assert run(*inputs, {}) is True
    discharge = parquet / "discharge.parquet"
    key_a = cache_key(discharge, lexicon, candidates)
    lexicon.write_bytes(lexicon.read_bytes() + b" ")
    key_b = cache_key(discharge, lexicon, candidates)
    replace = Path.replace

    def interrupted(path: Path, target: Path) -> Path:
        if target.name == "note_preprocessing.parquet":
            msg = "přerušená publikace"
            raise OSError(msg)
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", interrupted)
    with pytest.raises(
        InvalidInputError, match="Cache měření zpráv nelze publikovat: OSError\\."
    ):
        run(*inputs, {})
    assert is_current(out, key_a, [name for name, _ in ARTIFACTS]) is False
    assert is_current(out, key_b, [name for name, _ in ARTIFACTS]) is False


@pytest.fixture
def cli_args(
    inputs: tuple[Path, Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Předá konzolovému vstupu explicitní syntetické cesty z fixtury."""
    command_argv = ["mediparse-eda-note-features"]
    for name, path in zip(
        ("parquet", "lexicon", "candidates", "out"), inputs, strict=True
    ):
        command_argv.extend((f"--{name}", str(path)))
    monkeypatch.setattr(sys, "argv", command_argv)


@pytest.mark.parametrize(
    "invalid", ["missing-discharge", "missing-lexicon", "corrupt-parquet", "null-text"]
)
@pytest.mark.usefixtures("cli_args")
def test_invalid_input_has_domain_error_and_cli_refuses(
    inputs: tuple[Path, Path, Path, Path],
    invalid: str,
) -> None:
    """Vadné syntetické vstupy mají přesnou doménovou hlášku a CLI je odmítne."""
    parquet, lexicon, _, _ = inputs
    discharge = parquet / "discharge.parquet"
    if invalid == "missing-discharge":
        discharge.rename(parquet / "other.parquet")
        message = "V adresáři tabulek chybí discharge.parquet."
    elif invalid == "missing-lexicon":
        lexicon.unlink()
        message = "Vstup klíče cache nelze přečíst: FileNotFoundError."
    elif invalid == "corrupt-parquet":
        discharge.write_bytes(b"not parquet")
        message = "Tabulku discharge nelze číst: ArrowInvalid."
    else:
        pq.write_table(
            pa.table({"note_id": ["n"], "text": pa.array([None], type=pa.string())}),
            discharge,
        )
        message = (
            "Tabulka discharge musí obsahovat note_id a text bez NULL jako řetězce."
        )
    with pytest.raises(InvalidInputError, match=f"^{re.escape(message)}$"):
        run(*inputs, {"CLAUDECODE": "1"})
    assert main() is ExitCode.REFUSED


@pytest.mark.usefixtures("cli_args")
def test_cli_refuses_data_access_error(
    inputs: tuple[Path, Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Odmítnutí stráží jde stejnou cestou jako neplatný vstup."""
    (inputs[0] / "synthetic.json").unlink()
    monkeypatch.setenv("CLAUDECODE", "1")
    assert main() is ExitCode.REFUSED


@pytest.mark.usefixtures("cli_args")
def test_cli_accepts_computation_and_current_cache() -> None:
    """Úspěšný výpočet i použití cache mají návratový kód OK."""
    assert main() is ExitCode.OK
    assert main() is ExitCode.OK
