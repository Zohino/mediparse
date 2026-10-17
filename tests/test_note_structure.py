"""Struktura plánu zprávy: pohlaví, věková značka, sekce, vylučující se podnadpisy a délka narativu."""

from __future__ import annotations

from math import log
from random import Random
from statistics import median, stdev

import pytest
from pydantic import ValidationError

from mediparse.domain.note_structure import (
    NarrativeModel,
    NoteStructure,
    PreambleField,
    PreambleValue,
    SectionModel,
    Sex,
    StructureModel,
    Subheading,
    sample_structure,
)
from mediparse.domain.synthetic_note import SyntheticNote

SEED = 20260929
LARGE = 40000
SLACK = 0.015
FEMALE = 0.513
AGE_MARKER = 0.5908
RARE = 0.14
HOSPITAL_COURSE = 0.88
HOSPITAL_COURSE_SHARE = 0.4
MEDIAN_WORDS = 197
SIGMA = 0.387
MIN_WORDS = 75
MAX_WORDS = 510
DEID_PER_WORD = 0.0347
PAIR = (
    Subheading(header="Lungs", probability=0.38),
    Subheading(header="PULM", probability=0.16),
)
SINGLE = (Subheading(header="Vitals", probability=0.29),)
NAME = PreambleField(name="Name", value=PreambleValue.DEID)
SEX = PreambleField(name="Sex", value=PreambleValue.SEX)
SERVICE = PreambleField(name="Service", value=PreambleValue.TEXT)
PREAMBLE = ((NAME, SEX), (SERVICE,))
NARRATIVE = NarrativeModel(
    median_words=MEDIAN_WORDS,
    sigma=SIGMA,
    min_words=MIN_WORDS,
    max_words=MAX_WORDS,
    deid_per_word=DEID_PER_WORD,
    median_tolerance=0.1,
    sigma_tolerance=0.1,
    deid_tolerance=0.15,
    note_tolerance=0.2,
)


def _model(
    preamble: tuple[tuple[PreambleField, ...], ...] = PREAMBLE,
) -> StructureModel:
    return StructureModel(
        preamble=preamble,
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
                narrative_share=1 - HOSPITAL_COURSE_SHARE,
                subheadings=(PAIR, SINGLE),
            ),
            SectionModel(
                key="hospital_course",
                header="Brief Hospital Course",
                probability=HOSPITAL_COURSE,
                narrative_share=HOSPITAL_COURSE_SHARE,
            ),
        ),
        narrative=NARRATIVE,
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


def _allocated_words(structure: NoteStructure) -> int:
    return sum(structure.section_words.values())


def test_frequencies_follow_the_model() -> None:
    """Na velkém vzorku sedí četnost sekce, varianty podnadpisu, pohlaví i věkové značky."""
    structures = sample_structure(_notes(LARGE, per_patient=1), _model(), Random(SEED))
    rare = _share(["past_surgical_history" in s.sections for s in structures])
    variant = _share(["PULM" in s.subheadings for s in structures])

    assert abs(rare - RARE) <= SLACK
    assert abs(variant - PAIR[1].probability) <= SLACK
    assert abs(_share([s.sex is Sex.FEMALE for s in structures]) - FEMALE) <= SLACK
    assert abs(_share([s.age_marker for s in structures]) - AGE_MARKER) <= SLACK


def test_narrative_length_stays_within_bounds() -> None:
    """Délka narativu leží v mezích oříznutí a počet značek odpovídá hustotě."""
    for structure in sample_structure(_notes(LARGE), _model(), Random(SEED)):
        words = _allocated_words(structure)
        assert MIN_WORDS <= words <= MAX_WORDS
        assert structure.narrative_deid == round(words * DEID_PER_WORD)


def test_allocation_covers_only_present_narrative_sections() -> None:
    """Slova dostanou jen narativní sekce, které zpráva obsahuje."""
    narrative_keys = {s.key for s in _model().sections if s.narrative_share > 0}
    for structure in sample_structure(_notes(LARGE), _model(), Random(SEED)):
        assert structure.section_words.keys() <= narrative_keys & set(
            structure.sections
        )


def test_note_without_narrative_section_has_no_narrative() -> None:
    """Zpráva bez narativní sekce nemá narativní slova ani značky, aby plán neodporoval sám sobě."""
    model = StructureModel(
        preamble=PREAMBLE,
        sections=(
            SectionModel(key="allergies", header="Allergies", probability=1.0),
            SectionModel(
                key="hospital_course",
                header="Brief Hospital Course",
                probability=0.0,
                narrative_share=1.0,
            ),
        ),
        narrative=NARRATIVE,
        female_probability=FEMALE,
        age_marker_probability=AGE_MARKER,
    )

    for structure in sample_structure(_notes(30), model, Random(SEED)):
        assert structure.section_words == {}
        assert structure.narrative_deid == 0


def test_allocation_follows_shares() -> None:
    """Při obou narativních sekcích dostane každá podíl zaokrouhlený nejvýš o slovo."""
    structures = sample_structure(_notes(2000), _model(), Random(SEED))
    cases = [s for s in structures if "hospital_course" in s.section_words]

    assert cases
    for structure in cases:
        quota = HOSPITAL_COURSE_SHARE * _allocated_words(structure)
        assert abs(structure.section_words["hospital_course"] - quota) < 1


def test_length_distribution_matches_notes_synthesis() -> None:
    """Medián a rozptyl logaritmu délky leží v tolerancích kontrol z notes-synthesis."""
    structures = sample_structure(_notes(LARGE), _model(), Random(SEED))
    lengths = [_allocated_words(structure) for structure in structures]

    assert abs(median(lengths) / MEDIAN_WORDS - 1) <= NARRATIVE.median_tolerance
    assert (
        abs(stdev(log(words) for words in lengths) - SIGMA) <= NARRATIVE.sigma_tolerance
    )


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
    section = SectionModel(key="x", header="X", probability=1.0, narrative_share=0.5)

    with pytest.raises(ValidationError, match="opakují"):
        StructureModel(
            preamble=PREAMBLE,
            sections=(section, section),
            narrative=NARRATIVE,
            female_probability=0.5,
            age_marker_probability=0.5,
        )


def test_shares_not_summing_to_one_are_rejected() -> None:
    """Podíly narativních sekcí, které nedávají jedničku, schéma odmítne."""
    section = SectionModel(key="x", header="X", probability=1.0, narrative_share=0.5)

    with pytest.raises(ValidationError, match="dávat 1"):
        StructureModel(
            preamble=PREAMBLE,
            sections=(section,),
            narrative=NARRATIVE,
            female_probability=0.5,
            age_marker_probability=0.5,
        )


def test_median_outside_bounds_is_rejected() -> None:
    """Medián mimo meze oříznutí schéma odmítne."""
    with pytest.raises(ValidationError, match="uvnitř mezí"):
        NarrativeModel(
            median_words=MEDIAN_WORDS,
            sigma=SIGMA,
            min_words=MEDIAN_WORDS,
            max_words=MAX_WORDS,
            deid_per_word=DEID_PER_WORD,
            median_tolerance=0.1,
            sigma_tolerance=0.1,
            deid_tolerance=0.15,
            note_tolerance=0.2,
        )


def test_zero_deid_density_is_rejected() -> None:
    """Nulová hustota narativních značek neprojde; kontrola korpusu se k ní vztahuje relativně."""
    with pytest.raises(ValidationError, match="deid_per_word"):
        NARRATIVE.model_validate(NARRATIVE.model_dump() | {"deid_per_word": 0.0})


@pytest.mark.parametrize(
    "preamble",
    [
        pytest.param(((NAME,), (SERVICE,)), id="bez-sex"),
        pytest.param(((NAME, SEX), (SEX, SERVICE)), id="sex-dvakrat"),
        pytest.param(((NAME, SEX), ()), id="prazdny-radek"),
    ],
)
def test_preamble_has_exactly_one_sex_field(
    preamble: tuple[tuple[PreambleField, ...], ...],
) -> None:
    """Pohlaví z plánu nese v preambuli právě jedno pole a žádný řádek není prázdný."""
    with pytest.raises(ValidationError):
        _model(preamble)


def test_labels_cover_preamble_headers_and_subheadings() -> None:
    """Štítky struktury jsou pole preambule, hlavičky sekcí a podnadpisy z configu."""
    assert _model().structure_labels == {
        "Name",
        "Sex",
        "Service",
        "Allergies",
        "Past Surgical History",
        "Physical Exam",
        "Brief Hospital Course",
        "Lungs",
        "PULM",
        "Vitals",
    }
