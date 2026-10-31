"""Syntetické vstupy kohorty zachovávají labely a vazby mezi tabulkami MIMIC."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mediparse.domain.cohort import label_codes
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.icd import DiagnosisCodes, IcdVersion
from mediparse.domain.input_note import InputNote
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.language import Language
from mediparse.domain.mimic_table import TableShape
from mediparse.domain.note import subject_of
from mediparse.domain.synthetic_mimic import (
    UnpairedNotesError,
    input_notes,
    synthetic_mimic_rows,
)
from mediparse.infrastructure.mimic_inventory import read_manifest
from tests.cohort_support import CODES, PREFIXES
from tests.support import REPOSITORY_MIMIC_MANIFEST

if TYPE_CHECKING:
    from mediparse.domain.note_plan import NotePlan
    from tests.conftest import PlannedNote

SHAPES: Final = read_manifest(REPOSITORY_MIMIC_MANIFEST)


def _notes(labels: list[set[Diagnosis]]) -> tuple[InputNote, ...]:
    return tuple(
        InputNote(
            f"{90000000 + index}-DS-{index + 10}",
            90000000 + index,
            Language.EN,
            f"Syntetická zpráva {index}.",
            frozenset(assigned),
        )
        for index, assigned in enumerate(labels)
    )


@given(st.lists(st.sets(st.sampled_from(Diagnosis)), min_size=2, max_size=20))
def test_codes_preserve_labels_and_admission_links(
    labels: list[set[Diagnosis]],
) -> None:
    """Libovolné labely přežijí mapování ICD a každé přijetí má vlastní vazbu i výplň."""
    notes = _notes(labels)
    tables = synthetic_mimic_rows(tuple(reversed(notes)), CODES, SHAPES)
    discharge = tables["discharge"]
    diagnoses = tables["diagnoses_icd"]
    hadms = {row["hadm_id"] for row in discharge}

    assert len(hadms) == len(notes)
    assert hadms == {row["hadm_id"] for row in diagnoses}
    assert {row["icd_version"] for row in diagnoses} == {"9", "10"}
    assert [row["note_id"] for row in discharge] == [note.note_id for note in notes]
    for note, row in zip(notes, discharge, strict=True):
        admission = [r for r in diagnoses if r["hadm_id"] == row["hadm_id"]]
        pairs = [(r["icd_version"], r["icd_code"]) for r in admission]
        mapped = label_codes(pairs, CODES)
        assert set(mapped.diagnoses.values()) == note.labels
        assert mapped.invalid == 0
        assert admission[0]["seq_num"] == "1"
        filler = ("10", "Z0000") if pairs[0][0] == "10" else ("9", "V700")
        assert pairs[0] == filler
        assert len(admission) == 1 + len(note.labels)


def test_rows_have_fixed_identity_values_and_diagnosis_order() -> None:
    """Datum a identita jsou pevné, diagnózy řadí řetězcové hodnoty enumu abecedně."""
    notes = _notes([
        set(),
        {Diagnosis.CKD},
        {Diagnosis.PNEUMONIA, Diagnosis.ATRIAL_FIBRILLATION, Diagnosis.DIABETES},
    ])
    tables = synthetic_mimic_rows(notes, CODES, SHAPES)

    assert set(tables) == {"discharge", "diagnoses_icd"}
    assert tables["discharge"][0] == {
        "note_id": "90000000-DS-10",
        "subject_id": "90000000",
        "hadm_id": "20000000",
        "note_type": "DS",
        "note_seq": "10",
        "charttime": "2150-01-01 00:00:00",
        "storetime": "2150-01-01 00:00:00",
        "text": "Syntetická zpráva 0.",
    }
    assert tables["diagnoses_icd"] == [
        {
            "subject_id": "90000000",
            "hadm_id": "20000000",
            "seq_num": "1",
            "icd_code": "Z0000",
            "icd_version": "10",
        },
        {
            "subject_id": "90000001",
            "hadm_id": "20000001",
            "seq_num": "1",
            "icd_code": "V700",
            "icd_version": "9",
        },
        {
            "subject_id": "90000001",
            "hadm_id": "20000001",
            "seq_num": "2",
            "icd_code": "585",
            "icd_version": "9",
        },
        {
            "subject_id": "90000002",
            "hadm_id": "20000002",
            "seq_num": "1",
            "icd_code": "Z0000",
            "icd_version": "10",
        },
        {
            "subject_id": "90000002",
            "hadm_id": "20000002",
            "seq_num": "2",
            "icd_code": "I48",
            "icd_version": "10",
        },
        {
            "subject_id": "90000002",
            "hadm_id": "20000002",
            "seq_num": "3",
            "icd_code": "E11",
            "icd_version": "10",
        },
        {
            "subject_id": "90000002",
            "hadm_id": "20000002",
            "seq_num": "4",
            "icd_code": "J18",
            "icd_version": "10",
        },
    ]
    assert synthetic_mimic_rows(tuple(reversed(notes)), CODES, SHAPES) == tables


@pytest.mark.parametrize("table", ["discharge", "diagnoses_icd"])
def test_missing_table_is_refused(table: str) -> None:
    """Chybějící tabulka manifestu se nesmí nahradit vlastním schématem."""
    shapes = {name: shape for name, shape in SHAPES.items() if name != table}
    with pytest.raises(InvalidInputError, match=table):
        synthetic_mimic_rows(_notes([set()]), CODES, shapes)


@pytest.mark.parametrize("table", ["discharge", "diagnoses_icd"])
def test_unknown_column_is_refused(table: str) -> None:
    """Nový sloupec nesmí tiše dostat NULL."""
    shapes = {
        **SHAPES,
        table: TableShape(records=0, columns=(*SHAPES[table].columns, "unknown")),
    }
    with pytest.raises(InvalidInputError, match="unknown"):
        synthetic_mimic_rows(_notes([set()]), CODES, shapes)


@pytest.mark.parametrize(
    ("version", "filler"), [(IcdVersion.NINE, "V700"), (IcdVersion.TEN, "Z0000")]
)
def test_filler_in_diagnosis_is_refused(version: IcdVersion, filler: str) -> None:
    """Změna předpon nesmí výplní přidat neplánovaný label."""
    prefixes = {**PREFIXES, Diagnosis.DIABETES: {version: (filler,)}}
    with pytest.raises(InvalidInputError, match=r"výplň|Výplň"):
        synthetic_mimic_rows(_notes([set()]), DiagnosisCodes(prefixes=prefixes), SHAPES)


@pytest.mark.parametrize(
    ("version", "labels"),
    [(IcdVersion.NINE, [set(), {Diagnosis.CKD}]), (IcdVersion.TEN, [{Diagnosis.CKD}])],
)
def test_diagnosis_without_version_prefix_is_refused(
    version: IcdVersion, labels: list[set[Diagnosis]]
) -> None:
    """Label bez předpony ve verzi přijetí nelze převést na kód."""
    other = IcdVersion.TEN if version is IcdVersion.NINE else IcdVersion.NINE
    prefixes = {**PREFIXES, Diagnosis.CKD: {other: ("585",)}}
    with pytest.raises(InvalidInputError, match=rf"ckd.*{version.value}"):
        synthetic_mimic_rows(_notes(labels), DiagnosisCodes(prefixes=prefixes), SHAPES)


def _plan(planned_note: PlannedNote, note_id: str, *labels: Diagnosis) -> NotePlan:
    return planned_note.plan.model_copy(
        update={"note_id": note_id, "subject_id": subject_of(note_id), "labels": labels}
    )


def test_pairs_plans_with_texts_sorted_by_note_id(planned_note: PlannedNote) -> None:
    """Každá zpráva dostane subject_id a labely svého plánu; pořadí nezávisí na zdroji."""
    plans = (
        _plan(planned_note, "90000002-DS-1", Diagnosis.PNEUMONIA),
        _plan(planned_note, "90000001-DS-1"),
    )
    texts = {"90000002-DS-1": "druhá", "90000001-DS-1": "první"}

    notes = input_notes(plans, texts, Language.EN)

    assert notes == (
        InputNote("90000001-DS-1", 90000001, Language.EN, "první", frozenset()),
        InputNote(
            "90000002-DS-1",
            90000002,
            Language.EN,
            "druhá",
            frozenset({Diagnosis.PNEUMONIA}),
        ),
    )


def test_plan_without_text_is_refused(planned_note: PlannedNote) -> None:
    """Smoketest nad neúplným korpusem by nic nedokazoval."""
    plans = (_plan(planned_note, "90000001-DS-1"), _plan(planned_note, "90000002-DS-1"))

    with pytest.raises(
        UnpairedNotesError, match="Plány bez zprávy v jazyce en: 90000002-DS-1"
    ):
        input_notes(plans, {"90000001-DS-1": "první"}, Language.EN)


def test_text_without_plan_is_refused(planned_note: PlannedNote) -> None:
    """Zpráva bez plánu nemá labely."""
    texts = {"90000001-DS-1": "první", "90000009-DS-1": "cizí"}

    with pytest.raises(
        UnpairedNotesError, match="Zprávy bez plánu v jazyce en: 90000009-DS-1"
    ):
        input_notes((_plan(planned_note, "90000001-DS-1"),), texts, Language.EN)


def test_duplicate_plans_are_refused(planned_note: PlannedNote) -> None:
    """Dva plány jedné zprávy by labely určilo pořadí ve zdroji."""
    plans = (
        _plan(planned_note, "90000001-DS-1"),
        _plan(planned_note, "90000001-DS-1", Diagnosis.PNEUMONIA),
    )

    with pytest.raises(UnpairedNotesError, match="Duplicitní plány: 90000001-DS-1"):
        input_notes(plans, {"90000001-DS-1": "první"}, Language.EN)


def test_empty_corpus_is_refused() -> None:
    """Tabulka bez řádků by smoketest prošla bez dat."""
    with pytest.raises(
        UnpairedNotesError, match="Korpus nemá v jazyce en žádnou zprávu"
    ):
        input_notes((), {}, Language.EN)
