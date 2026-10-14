"""Use case materializace kohorty: kódy ze zdroje, štítky z domény, zápis zdrojem."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mediparse.application.cohort import CohortMaterialization
from mediparse.domain.cohort import CohortCounts, FunnelStep, FunnelStepName
from mediparse.domain.labels import Diagnosis
from tests.cohort_support import CODES

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from mediparse.domain.cohort import RawCode

COUNTS = CohortCounts(
    funnel=tuple(
        FunnelStep(step=step, notes=40, patients=40) for step in FunnelStepName
    ),
    positives=dict.fromkeys(Diagnosis, 20),
)


@dataclass
class _Source:
    pairs: list[RawCode]
    written: list[Mapping[RawCode, Diagnosis]] = field(default_factory=list)

    def codes(self) -> Iterable[RawCode]:
        return self.pairs

    def write(self, labels: Mapping[RawCode, Diagnosis]) -> CohortCounts:
        self.written.append(labels)
        return COUNTS


def test_labels_from_domain_go_to_source_and_result_carries_invalid_count() -> None:
    """Zdroj dostane mapování z domény; neplatné dvojice se vrátí v počtu."""
    source = _Source([("10", "E119"), ("10", "E11.9"), ("10", "S72")])

    result = CohortMaterialization(source, CODES).run()

    assert source.written == [{("10", "E119"): Diagnosis.DIABETES}]
    assert result.counts == COUNTS
    assert result.invalid_codes == 1
