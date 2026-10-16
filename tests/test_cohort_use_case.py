"""Use case materializace kohorty: kódy ze zdroje, štítky z domény, pravidlo DUA před zápisem."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import pytest

from mediparse.application.cohort import CohortMaterialization
from mediparse.domain.cohort import CohortCounts, FunnelStep, FunnelStepName
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.labels import Diagnosis
from tests.cohort_support import CODES

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Mapping

    from mediparse.domain.cohort import RawCode

UNIQUE: Final = 50


def _counts(
    positives: dict[Diagnosis, int] | None = None, unique: int = UNIQUE
) -> CohortCounts:
    return CohortCounts(
        funnel=tuple(
            FunnelStep(step=step, notes=unique, patients=unique)
            for step in FunnelStepName
        ),
        positives=positives or dict.fromkeys(Diagnosis, UNIQUE // 2),
    )


@dataclass
class _Selected:
    counts: CohortCounts
    events: list[str]

    def write(self) -> None:
        self.events.append("write")


@dataclass
class _Source:
    pairs: list[RawCode]
    counts: CohortCounts = field(default_factory=_counts)
    selected: list[Mapping[RawCode, Diagnosis]] = field(default_factory=list)
    events: list[str] = field(default_factory=list)

    def codes(self) -> Iterable[RawCode]:
        return self.pairs

    @contextmanager
    def select(self, labels: Mapping[RawCode, Diagnosis]) -> Generator[_Selected]:
        self.selected.append(labels)
        self.events.append("open")
        try:
            yield _Selected(self.counts, self.events)
        finally:
            self.events.append("close")


def test_publishable_cohort_is_written_once_and_result_carries_counts() -> None:
    """Zdroj dostane mapování z domény, zapíše se jednou; výsledek nese agregáty i neplatné."""
    source = _Source([("10", "E119"), ("10", "E11.9"), ("10", "S72")])

    result = CohortMaterialization(source, CODES).run()

    assert source.selected == [{("10", "E119"): Diagnosis.DIABETES}]
    assert source.events == ["open", "write", "close"]
    assert result.counts == source.counts
    assert result.invalid_codes == 1


@pytest.mark.parametrize(
    "counts",
    [
        _counts({**dict.fromkeys(Diagnosis, UNIQUE // 2), Diagnosis.COPD: 7}),
        _counts({**dict.fromkeys(Diagnosis, UNIQUE // 2), Diagnosis.COPD: 43}),
        _counts(unique=0),
    ],
    ids=["few-positives", "few-negatives", "empty"],
)
def test_unpublishable_cohort_is_refused_before_write_and_selection_closes(
    counts: CohortCounts,
) -> None:
    """Pravidlo DUA drží use case: odmítnutí zápis nezavolá a výběr se zavře."""
    source = _Source([("10", "E119")], counts)

    with pytest.raises(InvalidInputError):
        CohortMaterialization(source, CODES).run()

    assert source.events == ["open", "close"]
