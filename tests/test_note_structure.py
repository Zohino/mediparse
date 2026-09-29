"""Struktura plánu zprávy: pohlaví pacienta, věková značka, sekce a vylučující se podnadpisy."""

from __future__ import annotations

from random import Random

import pytest
from pydantic import ValidationError

from mediparse.domain.note_structure import (
    SectionModel,
    Sex,
    StructureModel,
    Subheading,
    sample_structure,
)
from mediparse.domain.synthetic_patients import SyntheticNote

SEED = 20260929
LARGE = 40000
SLACK = 0.015
FEMALE = 0.513
AGE_MARKER = 0.5908
RARE = 0.14
PAIR = (
    Subheading(header="Lungs", probability=0.38),
    Subheading(header="PULM", probability=0.16),
)
SINGLE = (Subheading(header="Vitals", probability=0.29),)


def _model() -> StructureModel:
    return StructureModel(
        sections=(
            SectionModel(key="allergies", header="Allergies", probability=1.0),
            SectionModel(
                key="past_surgical_history",
                header="Past Surgical History",
                probability=RARE,
            ),
            SectionModel(
                key="physical_exam",
                header="Physical Exam",
                probability=1.0,
                subheadings=(PAIR, SINGLE),
            ),
        ),
        female_probability=FEMALE,
        age_marker_probability=AGE_MARKER,
    )


def _notes(count: int, per_patient: int = 3) -> tuple[SyntheticNote, ...]:
    return tuple(
        SyntheticNote(
            f"{90000001 + i // per_patient}-DS-{i % per_patient + 1}",
            90000001 + i // per_patient,
            frozenset(),
        )
        for i in range(count)
    )


def _share(values: list[bool]) -> float:
    return sum(values) / len(values)


def test_same_seed_gives_same_structure() -> None:
    """Stejný seed dává stejnou strukturu."""
    notes = _notes(300)

    assert sample_structure(notes, _model(), Random(SEED)) == sample_structure(
        notes, _model(), Random(SEED)
    )


def test_sex_is_shared_by_patient_notes() -> None:
    """Pohlaví je atribut pacienta, všechny jeho zprávy ho nesou stejné."""
    notes = _notes(3000)
    structures = sample_structure(notes, _model(), Random(SEED))

    sexes: dict[int, set[Sex]] = {}
    for note, structure in zip(notes, structures, strict=True):
        sexes.setdefault(note.subject_id, set()).add(structure.sex)
    assert all(len(values) == 1 for values in sexes.values())


def test_sections_keep_model_order() -> None:
    """Přítomné sekce stojí v pořadí modelu a jistá sekce nechybí nikdy."""
    order = [section.key for section in _model().sections]

    for structure in sample_structure(_notes(500), _model(), Random(SEED)):
        assert list(structure.sections) == sorted(structure.sections, key=order.index)
        assert "allergies" in structure.sections


def test_exclusive_variants_never_appear_together() -> None:
    """Ze skupiny vylučujících se variant se losuje nejvýš jedna."""
    for structure in sample_structure(_notes(LARGE), _model(), Random(SEED)):
        assert not {"Lungs", "PULM"} <= set(structure.subheadings)


def test_frequencies_follow_the_model() -> None:
    """Na velkém vzorku sedí četnost sekce, varianty podnadpisu, pohlaví i věkové značky."""
    structures = sample_structure(_notes(LARGE, per_patient=1), _model(), Random(SEED))

    assert (
        abs(_share(["past_surgical_history" in s.sections for s in structures]) - RARE)
        <= SLACK
    )
    assert (
        abs(_share(["PULM" in s.subheadings for s in structures]) - PAIR[1].probability)
        <= SLACK
    )
    assert abs(_share([s.sex is Sex.FEMALE for s in structures]) - FEMALE) <= SLACK
    assert abs(_share([s.age_marker for s in structures]) - AGE_MARKER) <= SLACK


def test_variants_over_one_are_rejected() -> None:
    """Varianty, jejichž pravděpodobnosti přesáhnou jedničku, schéma odmítne."""
    group = (
        Subheading(header="A", probability=0.7),
        Subheading(header="B", probability=0.4),
    )

    with pytest.raises(ValidationError, match="nad 1"):
        SectionModel(key="x", header="X", probability=1.0, subheadings=(group,))


def test_duplicate_section_keys_are_rejected() -> None:
    """Dvě sekce se stejným klíčem schéma odmítne."""
    section = SectionModel(key="x", header="X", probability=1.0)

    with pytest.raises(ValidationError, match="opakují"):
        StructureModel(
            sections=(section, section),
            female_probability=0.5,
            age_marker_probability=0.5,
        )
