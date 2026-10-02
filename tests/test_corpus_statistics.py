"""Korpus jako celek: úplnost, délka narativu, hustota značek a prevalence labelů proti modelu."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from mediparse.domain.corpus_statistics import (
    CorpusStatistics,
    corpus_statistics,
    corpus_violations,
    statistics_violations,
)
from mediparse.domain.labels import Diagnosis

if TYPE_CHECKING:
    from mediparse.domain.synthetic_plan import SamplerConfig
    from tests.conftest import PlannedNote


def _on_target(config: SamplerConfig) -> CorpusStatistics:
    narrative = config.structure.narrative
    return CorpusStatistics(
        median_words=narrative.median_words,
        log_sigma=narrative.sigma,
        deid_density=narrative.deid_per_word,
        prevalence=dict(config.labels.prevalence),
    )


def _violations(statistics: CorpusStatistics, config: SamplerConfig) -> tuple[str, ...]:
    return statistics_violations(statistics, config.structure.narrative, config.labels)


def test_statistics_measure_narrative_of_the_text(
    planned_note: PlannedNote, sampler_config: SamplerConfig
) -> None:
    """Délka i značky se počítají z narativních sekcí textu; věková značka do hustoty nepatří."""
    plan = planned_note.plan

    statistics = corpus_statistics(
        [(plan, planned_note.text)], sampler_config.structure
    )

    assert statistics.median_words == plan.narrative_words
    assert statistics.log_sigma == pytest.approx(0.0)
    assert statistics.deid_density == plan.narrative_deid / plan.narrative_words
    assert statistics.prevalence[Diagnosis.CKD] == pytest.approx(1.0)
    assert statistics.prevalence[Diagnosis.DIABETES] == pytest.approx(0.0)


def test_subheadings_are_not_narrative(
    planned_note: PlannedNote, sampler_config: SamplerConfig
) -> None:
    """Název podnadpisu je struktura zprávy, do délky narativu se nepočítá."""
    text = planned_note.text.replace(
        "History of Present Illness:\n",
        "History of Present Illness:\nREVIEW OF SYSTEMS:\n",
    )

    statistics = corpus_statistics(
        [(planned_note.plan, text)], sampler_config.structure
    )

    assert statistics.median_words == planned_note.plan.narrative_words


def test_statistics_on_target_pass(sampler_config: SamplerConfig) -> None:
    """Korpus přesně podle modelu nemá žádné porušení."""
    assert _violations(_on_target(sampler_config), sampler_config) == ()


@pytest.mark.parametrize(
    ("change", "named"),
    [
        pytest.param({"median_words": 230.0}, "Medián", id="median"),
        pytest.param({"log_sigma": 0.6}, "Směrodatná odchylka", id="sigma"),
        pytest.param({"deid_density": 0.05}, "Hustota", id="hustota"),
    ],
)
def test_narrative_off_target_is_reported(
    sampler_config: SamplerConfig, change: dict[str, float], named: str
) -> None:
    """Délka narativu, její rozptyl a hustota značek mimo toleranci configu jsou porušení."""
    shifted = replace(_on_target(sampler_config), **change)

    violations = _violations(shifted, sampler_config)

    assert any(named in reason for reason in violations), violations


def test_prevalence_off_target_is_reported(sampler_config: SamplerConfig) -> None:
    """Prevalence labelu mimo toleranci configu je porušení, které jmenuje diagnózu."""
    statistics = _on_target(sampler_config)
    prevalence = dict(statistics.prevalence) | {Diagnosis.CKD: 0.3}

    violations = _violations(replace(statistics, prevalence=prevalence), sampler_config)

    assert any("Prevalence ckd" in reason for reason in violations), violations


def test_corpus_without_notes_violates_nothing(
    planned_note: PlannedNote, sampler_config: SamplerConfig
) -> None:
    """Dokud korpus nemá žádnou zprávu, nemá co splňovat."""
    plans = {planned_note.plan.note_id: planned_note.plan}

    assert corpus_violations({}, plans, sampler_config) == ()


def test_partial_corpus_is_incomplete(
    planned_note: PlannedNote, sampler_config: SamplerConfig
) -> None:
    """Korpus, který už nějaké zprávy má, musí mít všechny."""
    plan = planned_note.plan
    other = plan.model_copy(update={"note_id": "90000099-DS-1"})
    plans = {plan.note_id: plan, other.note_id: other}

    violations = corpus_violations(
        {plan.note_id: planned_note.text}, plans, sampler_config
    )

    assert violations == ("Korpus není úplný: chybí 1 z 2 zpráv.",)


def test_complete_corpus_is_measured(
    planned_note: PlannedNote, sampler_config: SamplerConfig
) -> None:
    """Úplný korpus se změří; jediná krátká zpráva modelu délky ani prevalencí neodpovídá."""
    plan = planned_note.plan

    violations = corpus_violations(
        {plan.note_id: planned_note.text}, {plan.note_id: plan}, sampler_config
    )

    assert any("Medián" in reason for reason in violations)
    assert any("Prevalence ckd" in reason for reason in violations)
