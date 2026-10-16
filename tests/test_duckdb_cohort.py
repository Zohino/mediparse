"""Adaptér kohorty nad zkonstruovanými tabulkami MIMIC: filtry, štítky, zápis a spill."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mediparse.domain.cohort import CohortCounts, FunnelStepName, label_codes
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.duckdb_cohort import DuckDbCohort
from mediparse.infrastructure.note_table import ParquetNoteTable
from tests.cohort_support import (
    CODES,
    icd,
    note,
    text,
    write_tables,
)

if TYPE_CHECKING:
    from pathlib import Path

    from mediparse.domain.input_note import InputNote

FUNNEL: Final = {
    FunnelStepName.ALL_NOTES: (62, 60),
    FunnelStepName.WITH_ICD: (59, 57),
    FunnelStepName.LONG_ENOUGH: (57, 55),
    FunnelStepName.UNIQUE: (56, 55),
}


def _build(root: Path, name: str = "cohort") -> tuple[CohortCounts, Path]:
    discharge, diagnoses = write_tables(root)
    out = root / f"{name}.parquet"
    source = DuckDbCohort(discharge, diagnoses, out, root / f"spill-{name}")
    labels = label_codes(source.codes(), CODES).diagnoses
    with source.select(labels) as selected:
        selected.write()
        return selected.counts, out


def _notes(out: Path) -> dict[str, InputNote]:
    return {item.note_id: item for item in ParquetNoteTable(out).read()}


def test_funnel_counts_notes_and_patients_of_every_filter(tmp_path: Path) -> None:
    """Zprávy bez hadm_id, bez ICD, krátké, prázdné a duplicitní trychtýř vyřadí po krocích."""
    counts, _ = _build(tmp_path)

    assert {step.step: (step.notes, step.patients) for step in counts.funnel} == FUNNEL


def test_kept_notes_and_their_labels(tmp_path: Path) -> None:
    """Zůstanou zprávy, které prošly filtry; štítek dá kód na libovolné pozici seq_num."""
    _, out = _build(tmp_path)

    notes = _notes(out)
    expected = {
        "n01": {Diagnosis.DIABETES},
        "n02": {Diagnosis.ATRIAL_FIBRILLATION},
        "n07": set(),
        "n10": {Diagnosis.PNEUMONIA},
        "n11": {Diagnosis.COPD},
        "n12": {Diagnosis.COPD},
    }
    for note_id, labels in expected.items():
        assert notes[note_id].labels == labels
    dropped = {"n03", "n04", "n05", "n06", "n08", "n09"}
    assert dropped.isdisjoint(notes)


def test_positives_count_notes_per_diagnosis(tmp_path: Path) -> None:
    """Pozitivní zprávy diagnózy: padding po deseti plus zprávy ze základu."""
    counts, _ = _build(tmp_path)

    assert counts.positives == {
        Diagnosis.DIABETES: 11,
        Diagnosis.COPD: 12,
        Diagnosis.CKD: 10,
        Diagnosis.PNEUMONIA: 11,
        Diagnosis.ATRIAL_FIBRILLATION: 11,
    }


def test_duplicate_keeps_copy_with_smallest_note_id(tmp_path: Path) -> None:
    """Z textu A zůstane n01 a n08 s kódem ledvin vypadne."""
    _, out = _build(tmp_path)

    assert Diagnosis.CKD not in _notes(out)["n01"].labels
    assert "n08" not in _notes(out)


def test_first_copy_without_icd_does_not_drop_later_copy(tmp_path: Path) -> None:
    """Filtry platí před deduplikací: kopie n10 s ICD přežije kopii n09 bez ICD."""
    _, out = _build(tmp_path)

    assert "n10" in _notes(out)
    assert "n09" not in _notes(out)


def test_null_code_admission_counts_as_with_icd_and_has_no_labels(
    tmp_path: Path,
) -> None:
    """Přijetí s jediným řádkem ICD s NULL kódem je ``S ICD`` a všechny štítky jsou false."""
    _, out = _build(tmp_path)

    assert _notes(out)["n07"].labels == frozenset()


def test_notes_of_one_admission_share_labels(tmp_path: Path) -> None:
    """Dvě zprávy téhož přijetí dostanou tytéž štítky."""
    _, out = _build(tmp_path)

    assert _notes(out)["n11"].labels == _notes(out)["n12"].labels


def test_output_is_sorted_english_and_matches_schema(tmp_path: Path) -> None:
    """Výstup čte ParquetNoteTable, je seřazený podle note_id a jazyk je en."""
    _, out = _build(tmp_path)

    items = ParquetNoteTable(out).read()

    assert [item.note_id for item in items] == sorted(item.note_id for item in items)
    assert {item.language for item in items} == {"en"}
    assert len(items) == FUNNEL[FunnelStepName.UNIQUE][0]


def test_two_writes_are_byte_identical(tmp_path: Path) -> None:
    """Zápis je deterministický."""
    _, first = _build(tmp_path, "first")
    _, second = _build(tmp_path, "second")

    assert first.read_bytes() == second.read_bytes()


def test_connection_uses_given_spill_directory(tmp_path: Path) -> None:
    """Spill DuckDB jde do předaného adresáře, ne do ``.tmp`` v pracovním adresáři."""
    discharge, diagnoses = write_tables(tmp_path)
    spill = tmp_path / "duckdb-tmp"
    source = DuckDbCohort(discharge, diagnoses, tmp_path / "c.parquet", spill)

    with source.connect() as con:
        row = con.execute("SELECT current_setting('temp_directory')").fetchone()

    assert row == (str(spill),)


def test_missing_input_is_invalid_input(tmp_path: Path) -> None:
    """Chybějící vstupní soubor je InvalidInputError."""
    discharge, diagnoses = write_tables(tmp_path)
    discharge.unlink()
    source = DuckDbCohort(discharge, diagnoses, tmp_path / "c.parquet", tmp_path / "s")

    with pytest.raises(InvalidInputError, match="discharge"):
        source.codes()


def _select_and_write(source: DuckDbCohort) -> None:
    with source.select(label_codes(source.codes(), CODES).diagnoses) as selected:
        selected.write()


def _small_source(root: Path) -> DuckDbCohort:
    discharge, diagnoses = write_tables(
        root,
        [note("n01", 1, 10, text("a"))],
        [icd(10, 1, "E119", "10")],
        padded=False,
    )
    return DuckDbCohort(discharge, diagnoses, root / "c.parquet", root / "s")


def test_small_cohort_is_selected_and_written_by_adapter(tmp_path: Path) -> None:
    """Pravidlo DUA už adaptér nedrží: malou kohortu vybere i zapíše."""
    source = _small_source(tmp_path)

    with source.select(label_codes(source.codes(), CODES).diagnoses) as selected:
        selected.write()

    assert selected.counts.funnel[-1].notes == 1
    assert [
        item.note_id for item in ParquetNoteTable(tmp_path / "c.parquet").read()
    ] == ["n01"]


def test_select_without_write_creates_no_file(tmp_path: Path) -> None:
    """Výběr sám nic nezapisuje; soubor vzniká až ve ``write``."""
    source = _small_source(tmp_path)

    with source.select(label_codes(source.codes(), CODES).diagnoses):
        pass

    assert not (tmp_path / "c.parquet").exists()


def test_output_parent_that_is_a_file_is_invalid_input(tmp_path: Path) -> None:
    """Chyba disku při zápisu je InvalidInputError."""
    discharge, diagnoses = write_tables(tmp_path)
    blocker = tmp_path / "blocker"
    blocker.write_text("soubor místo adresáře", encoding="utf-8")
    source = DuckDbCohort(discharge, diagnoses, blocker / "c.parquet", tmp_path / "s")

    with pytest.raises(InvalidInputError, match="zapsat"):
        _select_and_write(source)


def test_duckdb_error_in_write_is_invalid_input(tmp_path: Path) -> None:
    """Chyba DuckDB ve ``write`` po úspěšném výběru je InvalidInputError."""
    source = _small_source(tmp_path)

    with source.select(label_codes(source.codes(), CODES).diagnoses) as selected:
        selected.connection.execute("DROP TABLE base")
        with pytest.raises(InvalidInputError, match="zapsat"):
            selected.write()


def test_null_subject_id_in_write_is_invalid_input_and_leaves_no_file(
    tmp_path: Path,
) -> None:
    """Přetypování na schéma selže na NULL ve ``subject_id``; chyba je InvalidInputError a soubor nezůstane."""
    discharge, diagnoses = write_tables(tmp_path)
    pq.write_table(
        pa.table({
            "note_id": ["n01"],
            "subject_id": pa.array([None], pa.string()),
            "hadm_id": ["10"],
            "text": [text("a")],
        }),
        discharge,
    )
    out = tmp_path / "c.parquet"
    source = DuckDbCohort(discharge, diagnoses, out, tmp_path / "s")

    with pytest.raises(InvalidInputError, match="zapsat"):
        _select_and_write(source)

    assert not out.exists()


def test_positive_counts_match_label_columns_of_the_file(tmp_path: Path) -> None:
    """Počet true v každém sloupci souboru se rovná počtu pozitivních v agregátech."""
    counts, out = _build(tmp_path)

    items = ParquetNoteTable(out).read()

    for diagnosis in Diagnosis:
        in_file = sum(diagnosis in item.labels for item in items)
        assert in_file == counts.positives[diagnosis]
