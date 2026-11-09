"""Use case evaluace: metriky predikcí proti labelům kohorty po diagnóze a řádku matice."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from mediparse.domain.evaluation import (
    DiagnosisMetrics,
    ensure_both_classes,
    ensure_complete_folds,
    ensure_unique_keys,
    truth_of,
)
from mediparse.domain.inputs import InvalidInputError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.application.ports.experiment import (
        InputTable,
        PredictionSource,
        Scorer,
        SplitSource,
    )
    from mediparse.domain.diagnosis import Diagnosis
    from mediparse.domain.evaluation import Prediction


class MetricsSink(Protocol):
    """Výstup metrik evaluace."""

    def write(self, metrics: Sequence[DiagnosisMetrics]) -> None:
        """Zapíše metriky."""


@dataclass(frozen=True)
class ModelEvaluation:
    """Metriky na diagnózu a řádek matice z predikcí a labelů zpráv kohorty."""

    predictions: PredictionSource
    table: InputTable
    split: SplitSource
    scorer: Scorer
    metrics: MetricsSink

    def run(self) -> tuple[DiagnosisMetrics, ...]:
        """Seskupí predikce podle řádku matice a diagnózy, spočte a zapíše metriky.

        Predikce všech foldů skupiny se slévají do jedné metriky přes všechny testovací
        zprávy: metriky mají jeden řádek na řádek matice a diagnózu a srovnání jazyků
        je párované přes všechny pacienty kohorty, ne po foldech.

        Predikce každého foldu musí pokrývat přesně testovací zprávy foldu podle
        splitu, jinak by metriky vznikly nad jinou množinou, než jakou split určil.
        Pravda se odvozuje z labelů těchto zpráv v kohortě.

        Returns:
            Metriky seřazené podle řádku matice a diagnózy.

        Raises:
            InvalidInputError: Predikce jsou prázdné, opakuje se klíč predikce,
                nepokrývají přesně test foldu nebo některá skupina nemá obě třídy
                pravdy podle labelů kohorty.
        """
        items = self.predictions.read()
        if not items:
            msg = "Soubor predikcí je prázdný."
            raise InvalidInputError(msg)
        # Před úplností: porovnání množin duplicitu sloučí a neohlásila by se.
        ensure_unique_keys(items)
        notes = self.table.read()
        ensure_complete_folds(items, notes, self.split.read().split)
        truth = truth_of(items, notes)
        groups: defaultdict[tuple[str, Diagnosis], list[tuple[Prediction, bool]]] = (
            defaultdict(list)
        )
        for item, is_positive in zip(items, truth, strict=True):
            groups[item.row_id, item.diagnosis].append((item, is_positive))
        for (_, diagnosis), group in groups.items():
            ensure_both_classes(diagnosis, [is_positive for _, is_positive in group])
        result = tuple(
            DiagnosisMetrics(
                row_id,
                diagnosis,
                len(group),
                sum(is_positive for _, is_positive in group),
                self.scorer.score(
                    [is_positive for _, is_positive in group],
                    [item.predicted for item, _ in group],
                    [item.score for item, _ in group],
                ),
            )
            for (row_id, diagnosis), group in sorted(groups.items())
        )
        self.metrics.write(result)
        return result
