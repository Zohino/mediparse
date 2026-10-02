"""Use case kontrol shody nad fakes: párování zpráv s plány a zprávy k přegenerování."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mediparse.application.corpus_consistency import CorpusConsistency

if TYPE_CHECKING:
    from mediparse.domain.note_plan import NotePlan
    from mediparse.domain.synthetic_plan import SamplerConfig
    from tests.conftest import PlannedNote


@dataclass(frozen=True)
class _Plans:
    plans: tuple[NotePlan, ...]

    def load(self) -> tuple[NotePlan, ...]:
        return self.plans


@dataclass(frozen=True)
class _Corpus:
    texts: dict[str, str]

    def notes(self) -> dict[str, str]:
        return self.texts


def _check(plans: tuple[NotePlan, ...], texts: dict[str, str]) -> CorpusConsistency:
    return CorpusConsistency(plans=_Plans(plans), corpus=_Corpus(texts))


def test_matching_corpus_has_nothing_to_regenerate(
    planned_note: PlannedNote, sampler_config: SamplerConfig
) -> None:
    """Zpráva, která odpovídá plánu, se k přegenerování nehlásí."""
    plan = planned_note.plan
    texts = {f"en/{plan.note_id}.txt": planned_note.text}

    assert _check((plan,), texts).run(sampler_config) == ()


def test_plan_without_note_is_skipped(
    planned_note: PlannedNote, sampler_config: SamplerConfig
) -> None:
    """Korpus vzniká po polovinách, chybějící zpráva proto porušením není."""
    assert _check((planned_note.plan,), {}).run(sampler_config) == ()


def test_note_without_plan_is_reported(
    planned_note: PlannedNote, sampler_config: SamplerConfig
) -> None:
    """Zpráva bez plánu nemá podle čeho vzniknout ani se zkontrolovat."""
    results = _check((), {"en/90000099-DS-1.txt": planned_note.text}).run(
        sampler_config
    )

    assert [result.note_id for result in results] == ["90000099-DS-1"]


def test_violating_note_is_reported_once_with_reasons(
    planned_note: PlannedNote, sampler_config: SamplerConfig
) -> None:
    """Zpráva s porušením se hlásí jednou, se všemi důvody."""
    plan = planned_note.plan
    text = planned_note.text.replace("Sex: F", "Sex: M")

    (result,) = _check((plan,), {f"en/{plan.note_id}.txt": text}).run(sampler_config)

    assert result.note_id == plan.note_id
    assert any("Sex" in reason for reason in result.reasons)


def test_other_languages_are_not_checked(
    planned_note: PlannedNote, sampler_config: SamplerConfig
) -> None:
    """Český korpus má vlastní slovník (S11e), anglické kontroly se ho netýkají."""
    texts = {f"cs/{planned_note.plan.note_id}.txt": "Jiný text"}

    assert _check((planned_note.plan,), texts).run(sampler_config) == ()
