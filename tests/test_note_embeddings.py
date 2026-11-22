"""Cache podle přesného textu, pořadí a obnova po atomických dílech bez torch."""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.model_input import TextPreprocessing
from mediparse.infrastructure import note_embeddings
from mediparse.infrastructure.note_embeddings import SCHEMA, NoteEmbeddings, cache_key
from mediparse.infrastructure.parquet_file import write_table
from tests.hf_encoder_support import SPEC, VERSIONS

if TYPE_CHECKING:
    from pathlib import Path


def _vector(text: str) -> np.ndarray:
    return np.array([len(text), ord(text[0])], dtype=np.float32)


def test_cache_order_uniqueness_and_lazy_factory(tmp_path: Path) -> None:
    """Pořadí i duplicity přežijí cache a plná cache encoder vůbec nenačte."""
    fake = Mock(side_effect=_vector)
    factory = Mock(return_value=fake)
    result = NoteEmbeddings(tmp_path, VERSIONS, factory).embed(SPEC, ["a", "b", "a"])
    np.testing.assert_array_equal(result, [[1, 97], [1, 98], [1, 97]])
    assert result.dtype == np.float32
    assert fake.call_count == 2
    factory.assert_called_once_with(SPEC)
    factory = Mock(side_effect=AssertionError("Plná cache nesmí načítat encoder"))
    np.testing.assert_array_equal(
        NoteEmbeddings(tmp_path, VERSIONS, factory).embed(SPEC, ["a", "b", "a"]), result
    )
    factory.assert_not_called()


def test_cache_logging_omits_text_and_hash(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log postupu neobsahuje vstupní text ani jeho identifikační otisk."""
    sentinel = "SENTINEL-TEXT-private"
    with caplog.at_level(logging.INFO):
        NoteEmbeddings(tmp_path, VERSIONS, lambda _: _vector).embed(SPEC, [sentinel])
    assert sentinel not in caplog.text
    assert hashlib.sha256(sentinel.encode()).hexdigest() not in caplog.text


def test_restart_reuses_completed_parts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Pád na třetím textu ponechá první díl a restart dopočte jen tři texty."""
    monkeypatch.setattr(note_embeddings, "PART_NOTES", 2)
    broken = Mock(side_effect=[_vector("a"), _vector("b"), RuntimeError("pád")])
    texts = ["a", "b", "c", "d", "e"]
    with pytest.raises(RuntimeError, match="pád"):
        NoteEmbeddings(tmp_path / "restart", VERSIONS, lambda _: broken).embed(
            SPEC, texts
        )
    healthy = Mock(side_effect=_vector)
    with caplog.at_level(logging.INFO):
        actual = NoteEmbeddings(
            tmp_path / "restart", VERSIONS, lambda _: healthy
        ).embed(SPEC, texts)
    assert "Embeddingy: 2 hotových, 1 chybějících zpráv" in caplog.text
    expected = NoteEmbeddings(tmp_path / "fresh", VERSIONS, lambda _: _vector).embed(
        SPEC, texts
    )
    assert np.array_equal(actual, expected)
    assert healthy.call_count == 3
    assert sorted(
        path.name
        for path in (tmp_path / "restart" / cache_key(SPEC, VERSIONS)).iterdir()
    ) == ["0000.parquet", "0001.parquet", "0002.parquet"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", "other"),
        ("revision", "b" * 40),
        ("window", 9),
        ("device", "cuda"),
        ("dtype", "bfloat16"),
    ],
)
def test_cache_key_tracks_numerical_inputs(field: str, value: object) -> None:
    """Každý parametr ovlivňující embedding invaliduje cache."""
    assert cache_key(SPEC.model_copy(update={field: value}), VERSIONS) != cache_key(
        SPEC, VERSIONS
    )


@pytest.mark.parametrize("package", list(VERSIONS))
def test_cache_key_tracks_versions(package: str) -> None:
    """Cache odděluje verze všech encoderových knihoven, jméno nerozhoduje."""
    assert cache_key(SPEC, VERSIONS | {package: "other"}) != cache_key(SPEC, VERSIONS)
    assert cache_key(SPEC.model_copy(update={"name": "other"}), VERSIONS) == cache_key(
        SPEC, VERSIONS
    )


def test_inconsistent_cached_width_is_rejected(tmp_path: Path) -> None:
    """Díl s rozdílnými délkami vektorů není použitelnou reprezentací."""
    write_table(
        tmp_path / cache_key(SPEC, VERSIONS) / "0000.parquet",
        {"text_sha256": ["a" * 64, "b" * 64], "embedding": [[1.0], [1.0, 2.0]]},
        SCHEMA,
    )
    with pytest.raises(InvalidInputError, match="délk"):
        NoteEmbeddings(tmp_path, VERSIONS, Mock()).embed(SPEC, ["a"])


@pytest.mark.parametrize("module", ["hf_encoder.py", "hf_tokenizers.py"])
def test_cache_key_tracks_input_modules(
    monkeypatch: pytest.MonkeyPatch, module: str
) -> None:
    """Změna AST encoderu i tokenizace musí změnit obsahový klíč."""
    digests = {"hf_encoder.py": "x", "hf_tokenizers.py": "x"}
    monkeypatch.setattr(
        note_embeddings, "module_sha256", lambda path: digests[path.name]
    )
    before = cache_key(SPEC, VERSIONS)
    digests[module] = "y"
    assert cache_key(SPEC, VERSIONS) != before


def test_cache_key_io_error_is_invalid_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chyba čtení modulu při výpočtu klíče patří pod I/O hranici cache."""
    monkeypatch.setattr(
        note_embeddings, "module_sha256", Mock(side_effect=OSError("syntetická chyba"))
    )
    with pytest.raises(InvalidInputError, match="Cache embeddingů"):
        NoteEmbeddings(tmp_path, VERSIONS, Mock()).embed(SPEC, ["a"])


def test_cache_rejects_gap_in_parts(tmp_path: Path) -> None:
    """Druhý díl bez prvního nesmí být použit jako úplná obnova."""
    write_table(
        tmp_path / cache_key(SPEC, VERSIONS) / "0001.parquet",
        {"text_sha256": ["a" * 64], "embedding": [[1.0]]},
        SCHEMA,
    )
    with pytest.raises(InvalidInputError, match="netvoří souvislou řadu"):
        NoteEmbeddings(tmp_path, VERSIONS, Mock()).embed(SPEC, ["a"])


def test_cache_key_ignores_preprocessing_level() -> None:
    """Úroveň vstupu nemění cache adresovanou otiskem přesného textu."""
    dictionary = SPEC.model_copy(
        update={"text_preprocessing": TextPreprocessing.DICTIONARY}
    )
    assert cache_key(dictionary, VERSIONS) == cache_key(SPEC, VERSIONS)
