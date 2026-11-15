"""Parquet adaptér vstupní tabulky: pevné schéma a text beze změny."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.input_note import InputNote
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.language import Language
from mediparse.infrastructure.note_table import SCHEMA, ParquetNoteTable

if TYPE_CHECKING:
    from pathlib import Path

TEXT = "Pacientka přijata.\r\nDiagnóza: ___\n"


def test_writes_rows_in_fixed_schema(tmp_path: Path) -> None:
    """Labely jsou bool sloupce v pořadí výčtu, text projde beze změny."""
    path = tmp_path / "missing" / "notes.parquet"
    notes = [
        InputNote(
            "90000001-DS-1", 90000001, Language.EN, TEXT, frozenset({Diagnosis.CKD})
        ),
        InputNote("90000002-DS-1", 90000002, Language.EN, "b", frozenset()),
    ]

    ParquetNoteTable(path).write(notes)

    table = pq.read_table(path)
    assert table.schema.equals(SCHEMA)
    assert table.column_names == [
        "note_id",
        "subject_id",
        "language",
        "text",
        *(diagnosis.value for diagnosis in Diagnosis),
    ]
    assert table.to_pylist()[0] == {
        "note_id": "90000001-DS-1",
        "subject_id": 90000001,
        "language": "en",
        "text": TEXT,
        "diabetes": False,
        "ckd": True,
        "copd": False,
        "atrial_fibrillation": False,
        "pneumonia": False,
    }
    assert table.column("note_id").to_pylist() == ["90000001-DS-1", "90000002-DS-1"]


def test_rewrites_previous_table(tmp_path: Path) -> None:
    """Druhý zápis nahradí obsah, nepřipojí řádky."""
    path = tmp_path / "notes.parquet"
    note = InputNote("90000001-DS-1", 90000001, Language.EN, "a", frozenset())
    table = ParquetNoteTable(path)

    table.write([note, note])
    table.write([note])

    assert pq.read_table(path).num_rows == 1


def test_read_returns_written_rows(tmp_path: Path) -> None:
    """Zápis a čtení vrátí stejné řádky včetně labelů a textu."""
    table = ParquetNoteTable(tmp_path / "notes.parquet")
    notes = (
        InputNote(
            "90000001-DS-1",
            90000001,
            Language.EN,
            TEXT,
            frozenset({Diagnosis.CKD, Diagnosis.PNEUMONIA}),
        ),
        InputNote("90000002-DS-1", 90000002, Language.EN, "b", frozenset()),
    )

    table.write(notes)

    assert table.read() == notes


def test_read_missing_file_is_invalid_input(tmp_path: Path) -> None:
    """Chybějící soubor je InvalidInputError."""
    with pytest.raises(InvalidInputError, match="neexistuje"):
        ParquetNoteTable(tmp_path / "missing.parquet").read()


def test_read_other_schema_is_invalid_input(tmp_path: Path) -> None:
    """Soubor s jiným schématem je InvalidInputError."""
    path = tmp_path / "other.parquet"
    pq.write_table(pa.table({"note_id": ["a"]}), path)

    with pytest.raises(InvalidInputError, match="neodpovídá schématu"):
        ParquetNoteTable(path).read()


def test_read_returns_language_as_enum(tmp_path: Path) -> None:
    """Jazyk z parquetu se čte jako Language, ne jako řetězec."""
    table = ParquetNoteTable(tmp_path / "notes.parquet")
    table.write([InputNote("1-DS-1", 1, Language.CS, "text", frozenset())])

    assert table.read()[0].language is Language.CS


def _write_row(path: Path, **overrides: object) -> None:
    """Zapíše jeden řádek ve schématu tabulky mimo adaptér, s hodnotami z ``overrides``."""
    row = {
        "note_id": "1-DS-1",
        "subject_id": 1,
        "language": "en",
        "text": "text",
        **dict.fromkeys((diagnosis.value for diagnosis in Diagnosis), False),
    } | overrides
    columns = {name: [row[name]] for name in SCHEMA.names}
    pq.write_table(pa.Table.from_pydict(columns, schema=SCHEMA), path)


def test_read_unknown_language_is_invalid_input(tmp_path: Path) -> None:
    """Neznámý jazyk ohlásí cestu a sloupec bez hodnoty buňky a kontextu."""
    path = tmp_path / "notes.parquet"
    value = "tajny-text-456"
    _write_row(path, language=value)

    with pytest.raises(InvalidInputError) as raised:
        ParquetNoteTable(path).read()

    assert value not in str(raised.value)
    assert str(raised.value) == (
        f"Soubor {path} obsahuje neznámý jazyk ve sloupci language."
    )
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ is True


@pytest.mark.parametrize(
    ("note_id", "subject_id"),
    [("90000001-DS-1", 90000002), ("x90000001-DS-1", 90000001)],
    ids=["jiny-pacient", "mimo-skladbu"],
)
def test_read_note_id_not_of_its_subject_is_invalid_input_without_ids(
    tmp_path: Path, note_id: str, subject_id: int
) -> None:
    """Note_id jiného pacienta nebo mimo skladbu je InvalidInputError bez identifikátorů."""
    path = tmp_path / "notes.parquet"
    _write_row(path, note_id=note_id, subject_id=subject_id)

    with pytest.raises(InvalidInputError, match=re.escape(str(path))) as raised:
        ParquetNoteTable(path).read()

    assert "90000001" not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__


def test_written_file_is_zstd_compressed(tmp_path: Path) -> None:
    """Tabulka se zapisuje se stejnou kompresí jako ostatní parquet repa."""
    path = tmp_path / "notes.parquet"

    ParquetNoteTable(path).write([
        InputNote("90000002-DS-1", 90000002, Language.EN, "b", frozenset())
    ])

    assert pq.ParquetFile(path).metadata.row_group(0).column(0).compression == "ZSTD"


@pytest.mark.parametrize("batch_size", [2, 100])
def test_patient_reads_preserve_manual_selection(
    tmp_path: Path, batch_size: int
) -> None:
    """Ruční výběr pacientů 1 a 2 zachová zprávy 1, 2, 3 a 5 přes hranice dávek."""
    notes = tuple(
        InputNote(
            f"{subject}-DS-{index}",
            subject,
            Language.EN,
            f"text {index}",
            frozenset({Diagnosis.CKD}) if index % 2 else frozenset(),
        )
        for index, subject in enumerate((1, 1, 2, 3, 2), start=1)
    )
    table = ParquetNoteTable(tmp_path / "notes.parquet", batch_size=batch_size)
    table.write(notes)

    assert table.subjects() == frozenset({1, 2, 3})
    assert table.read_patients(frozenset({2, 1})) == (
        notes[0],
        notes[1],
        notes[2],
        notes[4],
    )
    assert table.read() == notes
    assert table.read_patients(frozenset()) == ()


@pytest.mark.parametrize("method", ["subjects", "read_patients"])
@pytest.mark.parametrize("invalid", ["missing", "schema", "corrupt"])
def test_patient_reads_reject_invalid_file(
    tmp_path: Path, method: str, invalid: str
) -> None:
    """Obě pacientské metody překládají vadný vstup na doménovou chybu."""
    path = tmp_path / "notes.parquet"
    if invalid == "schema":
        pq.write_table(pa.table({"subject_id": [1]}), path)
    elif invalid == "corrupt":
        path.write_text("není parquet", encoding="utf-8")
    messages = {
        "missing": f"Soubor {path} neexistuje.",
        "schema": (
            f"Soubor {path} neodpovídá schématu: čekané sloupce "
            "note_id, subject_id, language, text, diabetes, copd, ckd, "
            "pneumonia, atrial_fibrillation."
        ),
        "corrupt": (
            f"Soubor {path} neodpovídá schématu: Parquet magic bytes not found in footer. "
            "Either the file is corrupted or this is not a parquet file."
        ),
    }
    table = ParquetNoteTable(path)

    read = (
        table.subjects
        if method == "subjects"
        else lambda: table.read_patients(frozenset({1}))
    )
    with pytest.raises(InvalidInputError, match=f"^{re.escape(messages[invalid])}$"):
        read()


def test_patient_read_validates_only_selected_identities(tmp_path: Path) -> None:
    """Nesoulad identity se odmítne až při čtení zpráv vybraného pacienta."""
    path = tmp_path / "notes.parquet"
    _write_row(path, note_id="2-DS-1")
    table = ParquetNoteTable(path)

    assert table.subjects() == frozenset({1})
    assert table.read_patients(frozenset({2})) == ()
    with pytest.raises(InvalidInputError, match="note_id nepatří"):
        table.read_patients(frozenset({1}))
