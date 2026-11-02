"""Ruční orákula tokenizace, stráž, invalidace, publikace a kontrakt pro R."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Literal
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mediparse.domain.data_access import DataAccessError
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.note_tokens import TokenizerInfo
from mediparse.entrypoints import eda_note_tokens
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure import (
    hf_tokenizers,
    note_tokens_sample,
    note_tokens_table,
)
from mediparse.infrastructure.eda_cache import is_current
from mediparse.infrastructure.eda_duckdb import EdaDatabase
from mediparse.infrastructure.hf_tokenizers import (
    HfTokenLengths,
    prompt_overhead,
    special_tokens,
)
from mediparse.infrastructure.note_texts_parquet import ParquetNoteTexts
from mediparse.infrastructure.note_tokens_sample import SampleNoteTexts
from mediparse.infrastructure.note_tokens_table import ARTIFACTS, ParquetNoteTokens
from mediparse.infrastructure.parallel_texts import read_parallel
from mediparse.infrastructure.translation_script import ScriptSplitter, load_translation
from tests.note_tokens_support import (
    NOTES,
    FakeSplitter,
    whitespace_tokenizer,
    word_tokenizer,
)
from tests.support import REPOSITORY

type Inputs = tuple[Path, Path, Path, Path, Path]


@pytest.fixture
def inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Inputs:
    """Připraví syntetické tabulky a nahradí stahování lokálním tokenizérem.

    Returns:
        Pět cest pro run nástroje.
    """
    parquet = tmp_path / "tables"
    parquet.mkdir()
    (parquet / "synthetic.json").write_text('{"prefix":"t","rows":4}', encoding="utf-8")
    for name in ("discharge", "radiology"):
        pq.write_table(
            pa.table({
                "note_id": [row[0] for row in NOTES],
                "text": [row[1] for row in NOTES],
            }),
            parquet / f"{name}.parquet",
        )
    translation = tmp_path / "translation"
    shutil.copytree(
        REPOSITORY / "translation",
        translation,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    english, czech = tmp_path / "en", tmp_path / "cs"
    english.mkdir()
    czech.mkdir()
    (english / "a.txt").write_bytes(b"a b\r\n")
    (czech / "a.txt").write_bytes("a b č\r\n".encode())
    monkeypatch.setattr(eda_note_tokens, "load_tokenizer", lambda *_: word_tokenizer())
    monkeypatch.setattr(eda_note_tokens, "ScriptSplitter", lambda *_: FakeSplitter())
    return parquet, translation, english, czech, tmp_path / "cache"


def test_manual_wordlevel_lengths_and_prompt() -> None:
    """Tři slova plus BOS/EOS mají pět tokenů; prompt přidá BOS, x a y."""
    tokenizer = word_tokenizer()
    texts = ["a b c", "", "  a b \n"]
    assert HfTokenLengths(tokenizer, special=True).lengths(texts) == [5, 2, 4]
    assert HfTokenLengths(tokenizer, special=False).lengths(texts) == [3, 0, 2]
    assert HfTokenLengths(tokenizer, special=False, strip=True).lengths(texts) == [
        3,
        0,
        2,
    ]
    assert HfTokenLengths(tokenizer, special=False).lengths([]) == []
    assert (
        special_tokens(tokenizer),
        prompt_overhead(
            load_translation(REPOSITORY / "translation").prompt_ids(
                tokenizer, "The weather is nice today."
            ),
            5,
        ),
    ) == (2, 3)


def test_cache_and_fresh_result_match_manual_oracle(inputs: Inputs) -> None:
    """Cache zachová bajty a mtime; čerstvý běh má stejné řádky a ruční délky."""
    out = inputs[-1]
    assert eda_note_tokens.run(*inputs, {"CLAUDECODE": "1"}) is True
    directory = out / "tokens"
    before = {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()
    }
    assert eda_note_tokens.run(*inputs, {"CLAUDECODE": "1"}) is False
    assert before == {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()
    }
    fresh = out.parent / "fresh"
    assert (
        eda_note_tokens.run(inputs[0], inputs[1], inputs[2], inputs[3], fresh, {})
        is True
    )
    for name, schema in ARTIFACTS:
        actual = pq.read_table(directory / name)
        assert actual.schema.equals(schema)
        assert all(not field.nullable for field in actual.schema)
        assert actual.equals(pq.read_table(fresh / "tokens" / name))
        assert actual.equals(
            pq.read_table(REPOSITORY / "tests/fixtures/contracts" / name)
        )
    for name in ("discharge", "radiology"):
        actual = pq.read_table(directory / f"note_tokens_{name}.parquet").to_pydict()
        assert actual == {
            "note_id": ["n1", "n2", "n3", "n4"],
            "xlmr": [5, 2, 4, 5],
            "mdeberta": [5, 2, 4, 5],
            "translategemma": [3, 0, 2, 3],
        }
    assert pq.read_table(directory / "translation_prompt.parquet").to_pylist() == [
        {"prompt_overhead": 3, "split_ratio": 2.2, "sample_limit": 1000}
    ]
    sample = pq.read_table(directory / "translation_sample.parquet").to_pylist()
    assert {
        (r["note_id"], r["fits"], r["parts"], r["unsplittable"]) for r in sample
    } == {("n1", False, 0, True), ("n3", True, 1, False), ("n4", False, 3, False)}
    assert pq.read_table(directory / "parallel_tokens.parquet").to_pydict() == {
        "pair_id": ["a"] * 3,
        "tokenizer": ["xlmr", "mdeberta", "translategemma"],
        "english": [2, 2, 2],
        "czech": [3, 3, 3],
    }


@pytest.mark.parametrize(
    "changed",
    [
        "radiology",
        "script",
        "pair",
        "czech_pair",
        "revision",
        "transformers",
        "tokenizers",
    ],
)
def test_each_scientific_input_invalidates_cache(
    inputs: Inputs, changed: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Klíč zachytí tabulky, Python překladu, pár, model i obě knihovny."""
    parquet, translation, english, czech, out = inputs
    eda_note_tokens.run(*inputs, {})
    old = (out / "tokens/cache.json").read_bytes()
    if changed == "radiology":
        path = parquet / "radiology.parquet"
        pq.write_table(
            pq.read_table(path).replace_schema_metadata({b"changed": b"1"}), path
        )
    elif changed == "script":
        path = translation / "note_parts.py"
        path.write_bytes(path.read_bytes() + b"\n")
    elif changed == "pair":
        (english / "a.txt").write_bytes(b"a\n")
    elif changed == "czech_pair":
        (czech / "a.txt").write_bytes("č b\n".encode())
    elif changed == "revision":
        key, model, _, window = hf_tokenizers.ENCODERS[0]
        monkeypatch.setattr(
            hf_tokenizers,
            "ENCODERS",
            ((key, model, "f" * 40, window), hf_tokenizers.ENCODERS[1]),
        )
    else:
        original_version = hf_tokenizers.version
        monkeypatch.setattr(
            hf_tokenizers,
            "version",
            lambda package: (
                "changed" if package == changed else original_version(package)
            ),
        )
    assert eda_note_tokens.run(*inputs, {}) is True
    assert old != (out / "tokens/cache.json").read_bytes()


def test_guard_precedes_hashing_and_parquet_open(
    inputs: Inputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bez značky syntetiky se neotevře ani nehashuje žádný parquet."""
    (inputs[0] / "synthetic.json").unlink()
    digest, reader = Mock(), Mock()
    monkeypatch.setattr(note_tokens_table, "file_sha256", digest)
    monkeypatch.setattr(pq, "ParquetFile", reader)
    with pytest.raises(DataAccessError):
        eda_note_tokens.run(*inputs, {"CLAUDECODE": "1"})
    digest.assert_not_called()
    reader.assert_not_called()
    assert not inputs[-1].exists()


def test_sample_excludes_empty_and_keeps_duckdb_hash_order(
    inputs: Inputs,
) -> None:
    """Vzorek zachová původní hash pořadí a menší tabulka vrátí všechny neprázdné."""
    database = EdaDatabase(inputs[0], {})
    first = next(SampleNoteTexts(database, "discharge").batches())
    assert first == next(SampleNoteTexts(database, "discharge").batches())
    assert {note_id for note_id, _ in first} == {"n1", "n3", "n4"}
    with database.connect() as con:
        assert (
            first
            == con.execute(
                "SELECT note_id, text FROM discharge WHERE coalesce(length(text), 0) > 0 ORDER BY hash(note_id), note_id LIMIT 1000"
            ).fetchall()
        )
        assert con.execute("SELECT hash('n1')").fetchone() == (5454746921843688977,)


def test_sample_has_limit_1000(inputs: Inputs) -> None:
    """Tisícovkový limit se uplatní před dočtením textů."""
    path = inputs[0] / "discharge.parquet"
    pq.write_table(
        pa.table({"note_id": [f"n{i}" for i in range(1005)], "text": ["a"] * 1005}),
        path,
    )
    rows = next(SampleNoteTexts(EdaDatabase(inputs[0], {}), "discharge").batches())
    expected_size = 1000
    assert len(rows) == expected_size


def test_translation_import_restores_modules_and_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Jiná kopie skriptů nesmí sdílet moduly; úklid funguje i po selhání."""
    original = {name: ModuleType(name) for name in ("prepare", "markers", "note_parts")}
    for name, module in original.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = sys.path[:]
    script = load_translation(REPOSITORY / "translation")
    assert script.model == "google/translategemma-27b-it"
    assert (
        script.window,
        script.split_ratio,
        len(script.revision),
    ) == (2048, 2.2, 40)
    assert sys.path == path
    assert all(sys.modules[name] is module for name, module in original.items())
    with pytest.raises(InvalidInputError):
        load_translation(tmp_path)
    assert sys.path == path
    assert all(sys.modules[name] is module for name, module in original.items())


def test_real_script_splitter_handles_fitting_divided_and_unsplittable(
    inputs: Inputs,
) -> None:
    """Skutečné PEP 723 dělení pracuje nad tokenizérem bez Hubu."""
    prepare = inputs[1] / "prepare.py"
    prepare.write_bytes(
        prepare
        .read_bytes()
        .replace(b"WINDOW: Final = 2048", b"WINDOW: Final = 12")
        .replace(b"SPLIT_RATIO: Final = 2.2", b"SPLIT_RATIO: Final = 1.0")
    )
    script = load_translation(inputs[1])
    splitter = ScriptSplitter(script, word_tokenizer())
    assert splitter.fits("a b c")
    assert splitter.parts("a b c") == 1
    assert not splitter.fits("a b c\n\na b c")
    expected_parts = 2
    assert splitter.parts("a b c\n\na b c") == expected_parts
    assert splitter.parts("a b c a b c") is None


@pytest.mark.parametrize("missing", ["english", "czech", "empty"])
def test_parallel_missing_pair_or_empty_is_refused(
    inputs: Inputs, missing: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chybějící protějšek odmítne run i CLI a uvede jméno souboru."""
    _, _, english, czech, _ = inputs
    if missing in {"english", "empty"}:
        (english / "a.txt").unlink()
    if missing in {"czech", "empty"}:
        (czech / "a.txt").unlink()
    with pytest.raises(InvalidInputError, match=r"žádné.*txt|a\.txt"):
        eda_note_tokens.run(*inputs, {})
    argv = ["mediparse-eda-note-tokens"]
    for name, path in zip(
        ("parquet", "translation", "english", "czech", "out"), inputs, strict=True
    ):
        argv.extend((f"--{name}", str(path)))
    monkeypatch.setattr(sys, "argv", argv)
    assert eda_note_tokens.main() is ExitCode.REFUSED


def test_parallel_preserves_utf8_and_crlf(inputs: Inputs) -> None:
    """Čtení bajtů nezmění CRLF ani české znaky."""
    assert read_parallel(inputs[2], inputs[3]) == [("a", "a b\r\n", "a b č\r\n")]


def test_interrupted_publication_invalidates_both_keys_and_preserves_features(
    inputs: Inputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Přerušená skupina nezplatní směs artefaktů a nemění kořenovou cache."""
    out = inputs[-1]
    out.mkdir()
    sentinel = out / "cache.json"
    sentinel.write_bytes(b'{"key":"features"}')
    before = (sentinel.read_bytes(), sentinel.stat().st_mtime_ns)
    eda_note_tokens.run(*inputs, {})
    directory = out / "tokens"
    key = json.loads((directory / "cache.json").read_bytes())["key"]
    original = Path.replace

    def interrupted(path: Path, target: Path) -> Path:
        if target.name == "tokenizers.parquet":
            msg = "přerušeno"
            raise OSError(msg)
        return original(path, target)

    (inputs[2] / "a.txt").write_bytes(b"a\n")
    monkeypatch.setattr(Path, "replace", interrupted)
    with pytest.raises(InvalidInputError, match="publikovat"):
        eda_note_tokens.run(*inputs, {})
    assert not is_current(directory, key, [name for name, _ in ARTIFACTS])
    assert not (directory / "cache.json").exists()
    assert (sentinel.read_bytes(), sentinel.stat().st_mtime_ns) == before


def test_empty_writer_preserves_all_schemas(tmp_path: Path) -> None:
    """I prázdná tabulka má čitelné artefakty s kontraktem."""
    with ParquetNoteTokens(tmp_path).writer():
        pass
    for name, schema in ARTIFACTS:
        table = pq.read_table(tmp_path / name)
        assert table.schema.equals(schema)
        assert table.num_rows == 0


@pytest.mark.parametrize("table", ["discharge", "radiology"])
def test_null_text_is_refused_with_table_name(inputs: Inputs, table: str) -> None:
    """NULL není prázdná zpráva; čtečka odmítá obě tabulky stejným kontraktem."""
    path = inputs[0] / f"{table}.parquet"
    pq.write_table(
        pa.table({"note_id": ["n"], "text": pa.array([None], type=pa.string())}), path
    )
    with pytest.raises(InvalidInputError, match=f"Tabulka {table}.*NULL"):
        list(ParquetNoteTexts(path).batches())


def test_streaming_keeps_order_across_batches(inputs: Inputs) -> None:
    """Poslední neúplná dávka zachová prázdnou zprávu i pořadí souboru."""
    path = inputs[0] / "discharge.parquet"
    ids = [f"n{i}" for i in reversed(range(2001))]
    pq.write_table(pa.table({"note_id": ids, "text": ["a"] * 2000 + [""]}), path)
    eda_note_tokens.run(*inputs, {})
    rows = pq.read_table(
        inputs[-1] / "tokens/note_tokens_discharge.parquet"
    ).to_pydict()
    assert rows == {
        "note_id": ids,
        "xlmr": [3] * 2000 + [2],
        "mdeberta": [3] * 2000 + [2],
        "translategemma": [1] * 2000 + [0],
    }


def test_strip_only_when_requested() -> None:
    """Adaptér předá ořezaný text včetně nezlomitelných mezer jen překladu."""
    tokenizer = Mock(wraps=word_tokenizer())
    texts = ["  a b \n", "\u00a0a b\u00a0"]
    assert HfTokenLengths(tokenizer, special=False, strip=True).lengths(texts) == [2, 2]
    tokenizer.assert_called_once_with(["a b", "a b"], add_special_tokens=False)
    tokenizer.reset_mock()
    HfTokenLengths(tokenizer, special=True).lengths(texts)
    tokenizer.assert_called_once_with(texts, add_special_tokens=True)


def test_parallel_pairs_by_sorted_filename(inputs: Inputs) -> None:
    """Párování nezávisí na pořadí vytvoření anglických a českých souborů."""
    (inputs[3] / "c.txt").write_bytes(b"c")
    (inputs[2] / "b.txt").write_bytes(b"b")
    (inputs[2] / "c.txt").write_bytes(b"c")
    (inputs[3] / "b.txt").write_bytes(b"b")
    assert read_parallel(inputs[2], inputs[3]) == [
        ("a", "a b\r\n", "a b č\r\n"),
        ("b", "b", "b"),
        ("c", "c", "c"),
    ]


def test_sample_uses_declared_python_limit(
    inputs: Inputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Limit SQL pochází z jediné deklarace velikosti vzorku v Pythonu."""
    monkeypatch.setattr(note_tokens_sample, "SAMPLE_LIMIT", 2)
    rows = next(SampleNoteTexts(EdaDatabase(inputs[0], {}), "discharge").batches())
    expected_count = 2
    assert len(rows) == expected_count


@pytest.mark.parametrize(
    ("role", "parallel", "expected"),
    [
        ("encoder", False, [5, 2, 5]),
        ("encoder", True, [3, 0, 3]),
        ("translation", False, [3, 0, 2]),
        ("translation", True, [3, 0, 2]),
    ],
)
def test_counter_policy_counts_whitespace_sensitive_tokens(
    role: Literal["encoder", "translation"],
    *,
    parallel: bool,
    expected: list[int],
) -> None:
    """Dvě slova, nový řádek a BOS/EOS dávají ručně pět tokenů encoderu."""
    info = TokenizerInfo(
        "test", "model", "revision", 512, role, 2 if role == "encoder" else 0, "test"
    )
    counter = hf_tokenizers.counter_for(info, whitespace_tokenizer(), parallel=parallel)
    assert counter.lengths(["a b c", "", "  a b \n"]) == expected


@pytest.mark.parametrize("invalid", [False, True])
def test_translation_import_cleans_new_modules_and_precedes_decoy(
    inputs: Inputs,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    invalid: bool,
) -> None:
    """Import používá sourozenecké markers a uklidí i původně nepřítomné moduly."""
    names = ("prepare", "note_parts", "markers")
    for name in names:
        monkeypatch.delitem(sys.modules, name, raising=False)
    decoy = tmp_path / "decoy"
    decoy.mkdir()
    (decoy / "markers.py").write_text(
        'PLACEHOLDER = "návnada"\ndef mask(text):\n    return text\n', encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(decoy))
    original_path = sys.path[:]
    prepare = inputs[1] / "prepare.py"
    if invalid:
        prepare.unlink()
        with pytest.raises(
            InvalidInputError,
            match=r"^Skripty překladu nelze načíst: FileNotFoundError\.$",
        ):
            load_translation(inputs[1])
    else:
        prepare.write_bytes(prepare.read_bytes() + b"\nMODEL = PLACEHOLDER\n")
        script = load_translation(inputs[1])
        assert script.model == "[[n]]"
    assert not any(name in sys.modules for name in names)
    assert sys.path == original_path
