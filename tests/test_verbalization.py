"""Zadání verbalizace: deterministický text se vším, co pro zprávu určuje plán a config."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mediparse.domain.verbalization import render_prompt

if TYPE_CHECKING:
    from mediparse.domain.synthetic_plan import SamplerConfig
    from tests.conftest import PlannedNote


@pytest.fixture
def prompt(
    planned_note: PlannedNote,
    sampler_config: SamplerConfig,
    verbalization_template: str,
) -> str:
    """Zadání pro fixture plán.

    Returns:
        Vyrenderovaný text zadání.
    """
    return render_prompt(verbalization_template, planned_note.plan, sampler_config)


def test_prompt_is_deterministic(
    prompt: str,
    planned_note: PlannedNote,
    sampler_config: SamplerConfig,
    verbalization_template: str,
) -> None:
    """Stejná šablona, plán a config dávají stejný text."""
    assert (
        render_prompt(verbalization_template, planned_note.plan, sampler_config)
        == prompt
    )


def test_prompt_has_no_unfilled_placeholders(prompt: str) -> None:
    """Šablona je vyplněná celá."""
    assert "$" not in prompt


@pytest.mark.parametrize(
    "expected",
    [
        pytest.param("Name: ___  Date of Birth: ___", id="preambule-par"),
        pytest.param("Sex: F  Service: <", id="pohlavi-a-service"),
        pytest.param("- History of Present Illness: about 30 words", id="delka-sekce"),
        pytest.param("- Social History: the body is exactly ___", id="telo-deid"),
        pytest.param("  - Facility: ___", id="podnadpis-deid"),
        pytest.param("use exactly 1 more `___`", id="narativni-znacky"),
        pytest.param("exactly once as `___ year old`", id="vekova-znacka"),
        pytest.param("chronic kidney disease", id="planovana-zminka"),
        pytest.param("Past Medical History, Discharge Diagnosis", id="sekce-zminky"),
        pytest.param("diabetes mellitus", id="zakazane-klicove-slovo"),
    ],
)
def test_prompt_carries_plan_details(prompt: str, expected: str) -> None:
    """Zadání nese preambuli, sekce, značky, věk, zmínky i zakázaná klíčová slova."""
    assert expected in prompt


def test_sections_follow_plan_order(prompt: str, planned_note: PlannedNote) -> None:
    """Hlavičky sekcí stojí v zadání v pořadí plánu."""
    headers = ("Allergies", "Chief Complaint", "History of Present Illness")
    positions = [prompt.index(f"- {header}:") for header in headers]

    assert positions == sorted(positions)
    assert planned_note.plan.sections[:3] == (
        "allergies",
        "chief_complaint",
        "history_of_present_illness",
    )


def test_age_is_forbidden_without_planned_marker(
    planned_note: PlannedNote,
    sampler_config: SamplerConfig,
    verbalization_template: str,
) -> None:
    """Bez věkové značky v plánu zadání věk zakáže."""
    plan = planned_note.plan.model_copy(update={"age_marker": False})

    prompt = render_prompt(verbalization_template, plan, sampler_config)

    assert "Do not state the patient's age" in prompt
