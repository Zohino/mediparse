"""Zadání verbalizace přes use case a příkaz: k note_id najde plán, neznámé odmítne."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mediparse.application.verbalization_prompt import VerbalizationPrompt
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.entrypoints.verbalization_prompt import run
from tests.support import (
    REPOSITORY_CONFIG,
    REPOSITORY_PLANS,
    REPOSITORY_TEMPLATE,
    PlansFake,
)

if TYPE_CHECKING:
    import pytest

    from mediparse.domain.note_plan import NotePlan
    from mediparse.domain.synthetic_plan import SamplerConfig
    from tests.conftest import PlannedNote


def test_known_note_gets_its_prompt(
    planned_note: PlannedNote,
    sampler_config: SamplerConfig,
    verbalization_template: str,
) -> None:
    """Zadání vzniká z plánu zprávy s daným note_id."""
    plan = planned_note.plan
    use_case = VerbalizationPrompt(PlansFake((plan,)))

    prompt = use_case.run(plan.note_id, verbalization_template, sampler_config)

    assert prompt is not None
    assert plan.note_id in prompt


def test_unknown_note_has_no_prompt(
    planned_note: PlannedNote, sampler_config: SamplerConfig
) -> None:
    """Pro note_id bez plánu zadání nevznikne."""
    use_case = VerbalizationPrompt(PlansFake((planned_note.plan,)))

    assert use_case.run("90000999-DS-1", "", sampler_config) is None


def test_command_prints_prompt_for_repository_plan(
    capsys: pytest.CaptureFixture[str], repository_plans: tuple[NotePlan, ...]
) -> None:
    """Příkaz vypíše zadání pro plán z repa."""
    plan = repository_plans[0]

    code = run(_argv(plan.note_id))

    assert code == ExitCode.OK
    assert f"for note\n{plan.note_id}" in capsys.readouterr().out


def test_command_refuses_unknown_note(caplog: pytest.LogCaptureFixture) -> None:
    """Neznámé note_id příkaz odmítne srozumitelnou hláškou."""
    assert run(_argv("90000999-DS-1")) == ExitCode.REFUSED
    assert "neexistuje" in caplog.text


def test_every_repository_plan_renders(
    sampler_config: SamplerConfig,
    verbalization_template: str,
    repository_plans: tuple[NotePlan, ...],
) -> None:
    """Šablona se vyplní pro všechny plány v repu; žádný plán ji nerozbije."""
    use_case = VerbalizationPrompt(PlansFake(repository_plans))
    assert all(
        use_case.run(plan.note_id, verbalization_template, sampler_config)
        for plan in repository_plans
    )


def _argv(note_id: str) -> list[str]:
    return [
        note_id,
        "--plans",
        str(REPOSITORY_PLANS),
        "--config",
        str(REPOSITORY_CONFIG),
        "--template",
        str(REPOSITORY_TEMPLATE),
    ]
