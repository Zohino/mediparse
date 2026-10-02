"""Shoda textu syntetické zprávy s plánem: každé porušení pravidel specifikace se ozve."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from mediparse.domain.note_consistency import note_violations
from mediparse.infrastructure.sampler_config import load_sampler_config

if TYPE_CHECKING:
    from mediparse.domain.note_plan import NotePlan
    from tests.conftest import PlannedNote

CONFIG = load_sampler_config(
    Path(__file__).parents[1] / "config" / "synthetic_plan.json"
)


def _violations(text: str, plan: NotePlan) -> tuple[str, ...]:
    return note_violations(text, plan, CONFIG.structure, CONFIG.mentions)


def test_matching_note_passes(planned_note: PlannedNote) -> None:
    """Zpráva, která dodržuje plán, nemá žádné porušení."""
    assert _violations(planned_note.text, planned_note.plan) == ()


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
    planned_note: PlannedNote, old: str, new: str, rule: str
) -> None:
    """Každé porušení pravidel jedné zprávy se projeví důvodem, který ho jmenuje."""
    text = planned_note.text.replace(old, new)
    assert text != planned_note.text

    violations = _violations(text, planned_note.plan)

    assert any(rule.lower() in reason.lower() for reason in violations), violations


def test_diagnosis_section_keyword_outside_plan_is_reported(
    planned_note: PlannedNote,
) -> None:
    """Klíčové slovo v Discharge Diagnosis, kam ho plán nedává, je porušení."""
    plan = planned_note.plan
    mention = plan.mentions[0].model_copy(
        update={"sections": ("past_medical_history",)}
    )
    moved = plan.model_copy(update={"mentions": (mention,)})

    violations = _violations(planned_note.text, moved)

    assert any("Discharge Diagnosis" in reason for reason in violations)


def test_age_without_planned_marker_is_reported(planned_note: PlannedNote) -> None:
    """Věk v textu, když ho plán nemá, je porušení."""
    plan = planned_note.plan.model_copy(update={"age_marker": False})

    violations = _violations(planned_note.text, plan)

    assert any("věk" in reason.lower() for reason in violations)
