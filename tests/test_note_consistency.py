"""Soulad textu syntetické zprávy s plánem: každé porušení pravidel specifikace se ozve."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from mediparse.domain.diagnosis import Diagnosis
from mediparse.synthetic.domain.mentions import Mention, MentionStatus
from mediparse.synthetic.domain.note_consistency import (
    measure_narrative,
    note_violations,
)

if TYPE_CHECKING:
    from mediparse.synthetic.domain.note_consistency import NarrativeMeasure
    from mediparse.synthetic.domain.note_plan import NotePlan
    from mediparse.synthetic.domain.synthetic_plan import SamplerConfig
    from tests.conftest import PlannedNote


def _violations(text: str, plan: NotePlan, config: SamplerConfig) -> tuple[str, ...]:
    return note_violations(text, plan, config.structure, config.mentions)


def test_matching_note_passes(
    planned_note: PlannedNote, sampler_config: SamplerConfig
) -> None:
    """Zpráva, která dodržuje plán, nemá žádné porušení."""
    assert _violations(planned_note.text, planned_note.plan, sampler_config) == ()


@pytest.mark.parametrize(
    ("status", "sentence", "expected"),
    [
        pytest.param(
            MentionStatus.NEGATED, "She has no diabetes.", False, id="negace-ok"
        ),
        pytest.param(MentionStatus.NEGATED, "NO DIABETES.", False, id="velka-pismena"),
        pytest.param(
            MentionStatus.NEGATED,
            "Negative  for diabetes.",
            False,
            id="vic-slov-mezery",
        ),
        pytest.param(
            MentionStatus.FAMILY_HISTORY,
            "Family history of diabetes in her mother.",
            False,
            id="rodina-ok",
        ),
        pytest.param(
            MentionStatus.NEGATED, "She has diabetes.", True, id="odstraneny-vyraz"
        ),
        pytest.param(
            MentionStatus.NEGATED,
            "She has no fever. She has diabetes.",
            True,
            id="jina-veta",
        ),
        pytest.param(MentionStatus.NEGATED, "No fever; diabetes.", True, id="strednik"),
        pytest.param(
            MentionStatus.NEGATED, "No fever\ndiabetes.", True, id="konec-radku"
        ),
        pytest.param(
            MentionStatus.FAMILY_HISTORY,
            "Family History:\nMother with diabetes.",
            True,
            id="hlavicka-rodiny",
        ),
        pytest.param(
            MentionStatus.NEGATED,
            "No diabetes. She has diabetes.",
            True,
            id="jedna-negovana-druha-ne",
        ),
        pytest.param(
            MentionStatus.UNCERTAIN, "Possible diabetes.", False, id="nejista-ok"
        ),
        pytest.param(
            MentionStatus.UNCERTAIN, "R/o diabetes.", True, id="nejista-negovana"
        ),
        pytest.param(
            MentionStatus.UNCERTAIN,
            "R/o diabetes. Possible diabetes.",
            False,
            id="nejista-jinde-volna",
        ),
        pytest.param(
            MentionStatus.NEGATED,
            "Nonsmoker with diabetes.",
            True,
            id="slovo-konci-vyrazem",
        ),
        pytest.param(
            MentionStatus.NEGATED,
            "She lives in Reno with diabetes.",
            True,
            id="slovo-zacina-vyrazem",
        ),
        pytest.param(MentionStatus.AFFIRMED, "No diabetes.", False, id="potvrzena"),
        pytest.param(
            MentionStatus.NEGATED,
            "Negative\u00a0for diabetes.",
            True,
            id="pevna-mezera",
        ),
        pytest.param(
            MentionStatus.NEGATED,
            "No evidence of diabetes or of any xxxxxxxxxxxx diabetes.",
            True,
            id="poradi-vyrazu",
        ),
    ],
)
def test_negation_window_is_checked(
    planned_note: PlannedNote,
    sampler_config: SamplerConfig,
    status: MentionStatus,
    sentence: str,
    *,
    expected: bool,
) -> None:
    """Status zmínky odpovídá tomu, zda klíčové slovo přežije odstranění negací."""
    assert (
        _negation_reported(planned_note, sampler_config, status, sentence) is expected
    )


def _negation_reported(
    note: PlannedNote, config: SamplerConfig, status: MentionStatus, sentence: str
) -> bool:
    plan = note.plan.model_copy(
        update={
            "mentions": (
                *note.plan.mentions,
                Mention(
                    diagnosis=Diagnosis.DIABETES,
                    status=status,
                    sections=("history_of_present_illness",),
                ),
            )
        }
    )
    text = note.text.replace("oral intake.", "oral intake. " + sentence)
    assert text != note.text
    return any("negační okno" in reason for reason in _violations(text, plan, config))


@given(n=st.integers(min_value=2, max_value=80))
@example(n=40)
@example(n=41)
def test_negation_window_boundary(
    planned_note: PlannedNote, sampler_config: SamplerConfig, n: int
) -> None:
    """Okno zahrnuje právě nejvýš 40 znaků mezi výrazem a klíčovým slovem."""
    sentence = "no" + " " + "x" * (n - 2) + " " + "diabetes."
    assert _negation_reported(
        planned_note, sampler_config, MentionStatus.NEGATED, sentence
    ) is (n > sampler_config.mentions.negation.window_chars)


@pytest.mark.parametrize(
    ("old", "new", "rule"),
    [
        pytest.param("Chief Complaint:\nFatigue\n\n", "", "hlavičky", id="chybi-sekce"),
        pytest.param(
            "Allergies:\nNo Known Allergies / Adverse Drug Reactions\n\nChief Complaint:\nFatigue",
            "Chief Complaint:\nFatigue\n\nAllergies:\nNo Known Allergies / Adverse Drug Reactions",
            "hlavičky",
            id="prehozene-poradi",
        ),
        pytest.param(
            "Discharge Disposition:",
            "Family History:\nNon-contributory\n\nDischarge Disposition:",
            "hlavičky",
            id="sekce-navic",
        ),
        pytest.param(
            "Unit No: ___", "Unit No: 4711", "Unit No", id="hodnota-preambule"
        ),
        pytest.param("Sex: F", "Sex: M", "Sex", id="pohlavi"),
        pytest.param(
            "Service: MEDICINE  ", "Service:  ", "Service", id="prazdny-service"
        ),
        pytest.param(
            "Social History:\n___",
            "Social History:\nLives alone",
            "Social History",
            id="telo-social-history",
        ),
        pytest.param("Facility:\n___", "Facility:\nHome", "Facility", id="facility"),
        pytest.param(
            "a ___ year old woman", "a woman", "věková značka", id="chybi-vek"
        ),
        pytest.param(
            "a ___ year old woman",
            "a 67 year old woman",
            "číselný věk",
            id="ciselny-vek",
        ),
        pytest.param(
            "Ms. ___ is", "Ms. [**Name**] is", "jinou formu", id="jina-znacka"
        ),
        pytest.param(
            "poor\noral intake.",
            "poor\noral intake and diabetes.",
            "diabetes",
            id="zminka-mimo-plan",
        ),
        pytest.param(
            "Chronic kidney disease", "Stable renal function", "ckd", id="chybi-zminka"
        ),
    ],
)
def test_violation_is_reported(
    planned_note: PlannedNote,
    sampler_config: SamplerConfig,
    old: str,
    new: str,
    rule: str,
) -> None:
    """Každé porušení pravidel jedné zprávy se projeví důvodem, který ho jmenuje."""
    text = planned_note.text.replace(old, new)
    assert text != planned_note.text

    violations = _violations(text, planned_note.plan, sampler_config)

    assert any(rule.lower() in reason.lower() for reason in violations), violations


def test_note_without_headers_is_reported_not_raised(
    planned_note: PlannedNote, sampler_config: SamplerConfig
) -> None:
    """Zpráva bez jediné hlavičky je porušení plánu, ne pád kontroly."""
    violations = _violations(
        "Patient admitted with fatigue.", planned_note.plan, sampler_config
    )

    assert any(
        reason.startswith("Hlavičky sekcí") and reason.endswith("text žádné.")
        for reason in violations
    ), violations


def test_diagnosis_section_keyword_outside_plan_is_reported(
    planned_note: PlannedNote, sampler_config: SamplerConfig
) -> None:
    """Klíčové slovo v Discharge Diagnosis, kam ho plán nedává, je porušení."""
    plan = planned_note.plan
    mention = plan.mentions[0].model_copy(
        update={"sections": ("past_medical_history",)}
    )
    moved = plan.model_copy(update={"mentions": (mention,)})

    violations = _violations(planned_note.text, moved, sampler_config)

    assert any("Discharge Diagnosis" in reason for reason in violations)


def test_age_without_planned_marker_is_reported(
    planned_note: PlannedNote, sampler_config: SamplerConfig
) -> None:
    """Věk v textu, když ho plán nemá, je porušení."""
    plan = planned_note.plan.model_copy(update={"age_marker": False})

    violations = _violations(planned_note.text, plan, sampler_config)

    assert any("věk" in reason.lower() for reason in violations)


def test_long_underscore_run_is_neither_age_marker_nor_known_form(
    planned_note: PlannedNote, sampler_config: SamplerConfig
) -> None:
    """Věk za ____ není věková značka, takže se hlásí cizí forma i chybějící značka."""
    text = planned_note.text.replace("___ year old", "____ year old", 1)

    violations = _violations(text, planned_note.plan, sampler_config)

    assert any("jinou formu" in reason for reason in violations)
    assert any("Věková značka chybí" in reason for reason in violations)


def test_long_underscore_run_is_not_subheading_value(
    planned_note: PlannedNote, sampler_config: SamplerConfig
) -> None:
    """Podnadpis s hodnotou ____ nemá hodnotu ___."""
    text = planned_note.text.replace("Facility:\n___", "Facility:\n____", 1)

    violations = _violations(text, planned_note.plan, sampler_config)

    assert any("Podnadpis Facility" in reason for reason in violations)


def _length_reasons(
    note: PlannedNote, config: SamplerConfig, planned_words: int
) -> list[str]:
    plan = note.plan.model_copy(
        update={"section_words": {"history_of_present_illness": planned_words}}
    )
    return [
        reason
        for reason in _violations(note.text, plan, config)
        if reason.startswith("Délka narativu")
    ]


@pytest.mark.parametrize("planned_words", [28, 35], ids=["presne", "hranice-0.8"])
def test_narrative_within_tolerance_passes(
    planned_note: PlannedNote, sampler_config: SamplerConfig, planned_words: int
) -> None:
    """Narativ o 28 slovech projde proti plánu, který se liší nejvýš o 20 %."""
    assert not _length_reasons(planned_note, sampler_config, planned_words)


@pytest.mark.parametrize("planned_words", [37, 22], ids=["kratka", "dlouha"])
def test_narrative_outside_tolerance_is_reported(
    planned_note: PlannedNote, sampler_config: SamplerConfig, planned_words: int
) -> None:
    """Narativ o 28 slovech kratší i delší než plán o víc než 20 % je porušení."""
    assert _length_reasons(planned_note, sampler_config, planned_words)


def test_note_without_planned_narrative_passes(
    planned_note: PlannedNote, sampler_config: SamplerConfig
) -> None:
    """Plán bez narativních sekcí neměří žádný narativ, takže délka nemá s čím nesouhlasit."""
    plan = planned_note.plan.model_copy(update={"section_words": {}})

    violations = _violations(planned_note.text, plan, sampler_config)

    assert not [reason for reason in violations if reason.startswith("Délka narativu")]


def test_narrative_against_zero_plan_is_reported(
    planned_note: PlannedNote, sampler_config: SamplerConfig
) -> None:
    """Narativ v sekci, které plán přidělil nula slov, je porušení, ne dělení nulou."""
    assert _length_reasons(planned_note, sampler_config, 0)


def _measure(
    body: str, planned_note: PlannedNote, config: SamplerConfig
) -> NarrativeMeasure:
    key = next(iter(planned_note.plan.section_words))
    return measure_narrative(((key, body),), planned_note.plan, config.structure)


def test_narrative_counts_only_standalone_markers(
    planned_note: PlannedNote, sampler_config: SamplerConfig
) -> None:
    """Řada ______ není značka narativu, stejně jako v kontrole překladu."""
    measure = _measure("a ______ b ___ c", planned_note, sampler_config)

    assert measure.marks == 1


def test_narrative_marks_are_never_negative(
    planned_note: PlannedNote, sampler_config: SamplerConfig
) -> None:
    """Věk za ____ není věková značka, takže se od značek neodečítá."""
    measure = _measure("a ____ year old woman", planned_note, sampler_config)

    assert measure.marks == 0


def test_narrative_measure_has_named_fields(
    planned_note: PlannedNote, sampler_config: SamplerConfig
) -> None:
    """Výsledek nese slova a značky jako pojmenovaná pole."""
    body = "a ___ year old woman with ___"
    measure = _measure(body, planned_note, sampler_config)

    assert (measure.words, measure.marks) == (len(body.split()), 1)
