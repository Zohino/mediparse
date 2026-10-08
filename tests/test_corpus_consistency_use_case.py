"""Use case kontrol shody nad fakes: párování zpráv s plány a zprávy k přegenerování."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mediparse.application.corpus_consistency import (
    ConsistencyReport,
    CorpusConsistency,
)
from tests.support import PlansFake

if TYPE_CHECKING:
    from mediparse.domain.note_plan import NotePlan
    from mediparse.domain.synthetic_plan import SamplerConfig
    from tests.conftest import PlannedNote


@dataclass(frozen=True)
class _Corpus:
    texts: dict[str, str]

    def notes(self) -> dict[str, str]:
        return self.texts


def _check(plans: tuple[NotePlan, ...], texts: dict[str, str]) -> CorpusConsistency:
    return CorpusConsistency(plans=PlansFake(plans), corpus=_Corpus(texts))


def test_matching_corpus_has_nothing_to_regenerate(
    planned_note: PlannedNote, sampler_config: SamplerConfig
) -> None:
    """Zpráva, která odpovídá plánu, se k přegenerování nehlásí."""
    plan = planned_note.plan
    texts = {f"en/{plan.note_id}.txt": planned_note.text}

    report = _check((plan,), texts).run(sampler_config)

    assert report.notes == ()
    assert report.corpus


def test_plan_without_note_is_skipped(
    planned_note: PlannedNote, sampler_config: SamplerConfig
) -> None:
    """Plán bez zprávy není porušení jedné zprávy; korpus bez zpráv nemá co splňovat."""
    report = _check((planned_note.plan,), {}).run(sampler_config)

    assert report == ConsistencyReport((), (), ())


def test_note_without_plan_is_reported(
    planned_note: PlannedNote, sampler_config: SamplerConfig
) -> None:
    """Zpráva bez plánu nemá podle čeho vzniknout ani se zkontrolovat."""
    results = _check((), {"en/90000099-DS-1.txt": planned_note.text}).run(
        sampler_config
    )

    assert [result.note_id for result in results.notes] == ["90000099-DS-1"]


def test_violating_note_is_reported_once_with_reasons(
    planned_note: PlannedNote, sampler_config: SamplerConfig
) -> None:
    """Zpráva s porušením se hlásí jednou, se všemi důvody."""
    plan = planned_note.plan
    text = planned_note.text.replace("Sex: F", "Sex: M")

    (result,) = (
        _check((plan,), {f"en/{plan.note_id}.txt": text}).run(sampler_config).notes
    )

    assert result.note_id == plan.note_id
    assert any("Sex" in reason for reason in result.reasons)


def test_other_languages_are_not_checked_against_plans(
    planned_note: PlannedNote, sampler_config: SamplerConfig
) -> None:
    """České zprávy nejdou přes kontroly plánu, ty hlídají jen anglické."""
    note_id = planned_note.plan.note_id
    texts = {
        f"en/{note_id}.txt": planned_note.text,
        f"cs/{note_id}.txt": planned_note.text,
    }

    report = _check((planned_note.plan,), texts).run(sampler_config)

    assert report.notes == ()
    assert report.notices == ()


def test_corpus_without_czech_has_no_translation_findings(
    planned_note: PlannedNote, sampler_config: SamplerConfig
) -> None:
    """Korpus bez cs/ se chová jako dřív: žádné upozornění ani důvod k cs/."""
    plan = planned_note.plan
    texts = {f"en/{plan.note_id}.txt": planned_note.text}

    report = _check((plan,), texts).run(sampler_config)

    assert report.notices == ()
    assert not any("cs/" in reason for reason in report.corpus)


def test_incomplete_czech_is_a_corpus_violation(
    planned_note: PlannedNote, sampler_config: SamplerConfig
) -> None:
    """Chybějící překlad zprávy je porušení korpusu, ne zpráva k přegenerování."""
    plan = planned_note.plan
    texts = {
        f"en/{plan.note_id}.txt": planned_note.text,
        "en/90000099-DS-1.txt": planned_note.text,
        f"cs/{plan.note_id}.txt": planned_note.text,
    }

    report = _check((plan,), texts).run(sampler_config)

    assert any("cs/90000099-DS-1" in reason for reason in report.corpus)
    assert [result.note_id for result in report.notes] == ["90000099-DS-1"]


def test_marker_mismatch_is_only_a_notice(
    planned_note: PlannedNote, sampler_config: SamplerConfig
) -> None:
    """Neshoda počtu ___ se eviduje jako upozornění, korpus kvůli ní neporušuje."""
    plan = planned_note.plan
    assert "___" in planned_note.text
    texts = {
        f"en/{plan.note_id}.txt": planned_note.text,
        f"cs/{plan.note_id}.txt": planned_note.text.replace("___", "xxx", 1),
    }

    report = _check((plan,), texts).run(sampler_config)

    assert len(report.notices) == 1
    assert plan.note_id in report.notices[0]
    assert not any("cs/" in reason for reason in report.corpus)
