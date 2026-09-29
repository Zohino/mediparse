"""Plán zmínek: sekce jen ze struktury, Y=0 mimo diagnostickou sekci, podíly podle modelu."""

from __future__ import annotations

from random import Random

import pytest
from pydantic import ValidationError

from mediparse.domain.labels import Diagnosis
from mediparse.domain.mentions import (
    DiagnosisMentions,
    Mention,
    MentionModel,
    MentionStatus,
    Negation,
    sample_mentions,
)
from mediparse.domain.note_structure import NoteStructure, Sex
from mediparse.domain.synthetic_patients import SyntheticNote

SEED = 20260929
LARGE = 40000
SLACK = 0.015
GIVEN_POSITIVE = 0.9
GIVEN_NEGATIVE = 0.2
IN_DIAGNOSIS = 0.85
IN_NARRATIVE = 0.75
NEGATED = 0.4
DD = "discharge_diagnosis"
FAMILY = "family_history"
PMH = "past_medical_history"
HPI = "history_of_present_illness"
BHC = "hospital_course"
ALL_SECTIONS = (PMH, FAMILY, HPI, BHC, DD)
STATUSES = {
    MentionStatus.NEGATED: NEGATED,
    MentionStatus.FAMILY_HISTORY: 0.2,
    MentionStatus.UNCERTAIN: 0.2,
    MentionStatus.AFFIRMED_UNCODED: 0.2,
}


def _model(statuses: dict[MentionStatus, float] | None = None) -> MentionModel:
    return MentionModel(
        diagnoses={
            diagnosis: DiagnosisMentions(
                given_positive=GIVEN_POSITIVE,
                given_negative=GIVEN_NEGATIVE,
                keywords=(diagnosis.value,),
            )
            for diagnosis in Diagnosis
        },
        diagnosis_section=DD,
        family_section=FAMILY,
        diagnosis_section_probability=IN_DIAGNOSIS,
        narrative_probability=IN_NARRATIVE,
        chronic_sections={PMH: 0.6, HPI: 0.2, BHC: 0.2},
        acute_sections={HPI: 0.5, BHC: 0.5},
        negative_statuses=STATUSES if statuses is None else statuses,
        negation=Negation(cues=("no", "denies"), window_chars=40),
    )


def _structure(sections: tuple[str, ...]) -> NoteStructure:
    return NoteStructure(
        note_id="90000001-DS-1",
        sex=Sex.FEMALE,
        age_marker=False,
        sections=sections,
        subheadings=(),
        narrative_words=100,
        section_words=(),
        narrative_deid=3,
    )


def _sample(
    count: int, labels: frozenset[Diagnosis], sections: tuple[str, ...] = ALL_SECTIONS
) -> tuple[tuple[Mention, ...], ...]:
    notes = [
        SyntheticNote(f"{90000001 + i}-DS-1", 90000001 + i, labels)
        for i in range(count)
    ]
    structures = [_structure(sections)] * count
    return sample_mentions(notes, structures, _model(), Random(SEED))


def _flat(mentions: tuple[tuple[Mention, ...], ...]) -> list[Mention]:
    return [mention for note in mentions for mention in note]


def test_same_seed_gives_same_mentions() -> None:
    """Stejný seed dává stejné zmínky."""
    assert _sample(200, frozenset({Diagnosis.CKD})) == _sample(
        200, frozenset({Diagnosis.CKD})
    )


def test_mentions_stay_in_present_sections() -> None:
    """Zmínka stojí jen v sekci, kterou struktura zprávy obsahuje, v pořadí struktury."""
    sections = (FAMILY, BHC, DD)
    for mention in _flat(_sample(LARGE, frozenset({Diagnosis.DIABETES}), sections)):
        assert mention.sections
        assert set(mention.sections) <= set(sections)
        assert list(mention.sections) == sorted(mention.sections, key=sections.index)


def test_negative_mentions_never_reach_diagnosis_section() -> None:
    """Zmínka u negativního labelu nikdy nestojí v Discharge Diagnosis a není potvrzená s kódem."""
    for mention in _flat(_sample(LARGE, frozenset())):
        assert DD not in mention.sections
        assert mention.status is not MentionStatus.AFFIRMED


def test_positive_mentions_are_affirmed_and_follow_placement() -> None:
    """Pozitivní zmínky jsou potvrzené a podíl zmínek jen v DD odpovídá umístění."""
    labels = frozenset({Diagnosis.AKI})
    mentions = [
        m for m in _flat(_sample(LARGE, labels)) if m.diagnosis is Diagnosis.AKI
    ]
    only_diagnosis = sum(m.sections == (DD,) for m in mentions) / len(mentions)

    assert all(m.status is MentionStatus.AFFIRMED for m in mentions)
    assert abs(len(mentions) / LARGE - GIVEN_POSITIVE) <= SLACK
    assert abs(only_diagnosis - IN_DIAGNOSIS * (1 - IN_NARRATIVE)) <= SLACK


def test_missing_diagnosis_section_moves_mention_to_narrative() -> None:
    """Bez Discharge Diagnosis ve struktuře jde pozitivní zmínka do narativu."""
    mentions = _flat(_sample(2000, frozenset({Diagnosis.CKD}), (PMH, HPI, BHC)))

    assert mentions
    assert all(DD not in m.sections for m in mentions)


def test_no_allowed_section_means_no_mention() -> None:
    """Když struktura nemá žádnou povolenou sekci pro AKI, zmínka AKI se neplánuje."""
    mentions = _flat(_sample(2000, frozenset({Diagnosis.AKI}), (PMH, FAMILY)))

    assert all(m.diagnosis is not Diagnosis.AKI for m in mentions)


def test_negative_statuses_follow_the_model() -> None:
    """U negativního labelu sedí podíl zmínek i rozdělení statusů."""
    mentions = _flat(_sample(LARGE, frozenset()))
    per_diagnosis = len(mentions) / len(Diagnosis) / LARGE
    negated = sum(m.status is MentionStatus.NEGATED for m in mentions) / len(mentions)

    assert abs(per_diagnosis - GIVEN_NEGATIVE) <= SLACK
    assert abs(negated - NEGATED) <= SLACK


def test_family_history_status_prefers_family_section() -> None:
    """Zmínka ve statusu rodinné anamnézy stojí v sekci Family History, když ji zpráva má."""
    for mention in _flat(_sample(LARGE, frozenset())):
        if mention.status is MentionStatus.FAMILY_HISTORY:
            assert mention.sections == (FAMILY,)


def test_affirmed_status_for_negative_label_is_rejected() -> None:
    """Status potvrzené zmínky s kódem u negativního labelu schéma odmítne."""
    with pytest.raises(ValidationError, match="potvrzená s kódem"):
        _model({MentionStatus.AFFIRMED: 1.0})


def test_statuses_not_summing_to_one_are_rejected() -> None:
    """Statusy negativního labelu, které nedávají 1, schéma odmítne."""
    with pytest.raises(ValidationError, match="dávat 1"):
        _model({MentionStatus.NEGATED: 0.5})
