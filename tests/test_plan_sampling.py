"""Use case vzorkování plánů: determinismus, počet plánů a nezávislost proudů jednotlivých fází."""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import TYPE_CHECKING

from mediparse.application.plan_sampling import PlanSampling

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.domain.note_plan import NotePlan
    from mediparse.domain.synthetic_plan import SamplerConfig


@dataclass
class _Store:
    plans: list[NotePlan] = field(default_factory=list)

    def save(self, plans: Sequence[NotePlan]) -> None:
        self.plans = list(plans)


def _run(config: SamplerConfig) -> list[NotePlan]:
    store = _Store()
    PlanSampling(store=store, random_source=Random).run(config)
    return store.plans


def test_same_seed_gives_same_plans(sampler_config: SamplerConfig) -> None:
    """Stejný seed a config dávají stejné plány."""
    assert _run(sampler_config) == _run(sampler_config)


def test_every_note_gets_a_plan(sampler_config: SamplerConfig) -> None:
    """Use case uloží plán pro každou zprávu korpusu a vrátí jejich počet."""
    store = _Store()

    count = PlanSampling(store=store, random_source=Random).run(sampler_config)

    assert count == sampler_config.patients.notes == len(store.plans)


def test_other_seed_gives_other_plans(sampler_config: SamplerConfig) -> None:
    """Jiný seed dává jiné plány."""
    assert _run(sampler_config) != _run(
        sampler_config.model_copy(update={"seed": sampler_config.seed + 1})
    )


def test_changing_mentions_keeps_labels_and_structure(
    sampler_config: SamplerConfig,
) -> None:
    """Každá fáze má vlastní proud: změna modelu zmínek nepřelosuje labely ani strukturu."""
    mentions = sampler_config.mentions.model_copy(update={"narrative_probability": 0.1})
    changed = _run(sampler_config.model_copy(update={"mentions": mentions}))
    original = _run(sampler_config)

    assert [p.model_dump(exclude={"mentions"}) for p in changed] == [
        p.model_dump(exclude={"mentions"}) for p in original
    ]
    assert [p.mentions for p in changed] != [p.mentions for p in original]
