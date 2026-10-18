"""Use case evaluace: metriky z predikcí po diagnóze a řádku matice."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from mediparse.domain.evaluation import (
    DiagnosisMetrics,
    ensure_both_classes,
    ensure_complete_folds,
    ensure_unique_keys,
)
from mediparse.domain.inputs import InvalidInputError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.application.ports.experiment import (
        InputTable,
        PredictionSource,
        SplitSource,
    )
    from mediparse.domain.diagnosis import Diagnosis
    from mediparse.domain.evaluation import BinaryMetrics, Prediction


class Scorer(Protocol):
    """Výpočet metrik binární klasifikace."""

    def score(
        self, truth: Sequence[bool], predicted: Sequence[bool], scores: Sequence[float]
    ) -> BinaryMetrics:
        """Spočte metriky predikcí a skóre proti pravdě."""


class MetricsSink(Protocol):
    """Výstup metrik evaluace."""

    def write(self, metrics: Sequence[DiagnosisMetrics]) -> None:
        """Zapíše metriky."""


@dataclass(frozen=True)
class ModelEvaluation:
    """Metriky na diagnózu a řádek matice spočtené z predikcí."""

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

        Returns:
            Metriky seřazené podle řádku matice a diagnózy.

        Raises:
            InvalidInputError: Predikce jsou prázdné, opakuje se klíč predikce,
                nepokrývají přesně test foldu nebo některá skupina nemá obě třídy
                pravdy.
        """
        items = self.predictions.read()
        if not items:
            msg = "Soubor predikcí je prázdný."
            raise InvalidInputError(msg)
        # Před úplností: porovnání množin duplicitu sloučí a neohlásila by se.
        ensure_unique_keys(items)
        ensure_complete_folds(items, self.table.read(), self.split.read().split)
        groups: defaultdict[tuple[str, Diagnosis], list[Prediction]] = defaultdict(list)
        for item in items:
            groups[item.row_id, item.diagnosis].append(item)
        for (_, diagnosis), group in groups.items():
            ensure_both_classes(diagnosis, [item.truth for item in group])
        result = tuple(
            DiagnosisMetrics(
                row_id,
                diagnosis,
                len(group),
                sum(item.truth for item in group),
                self.scorer.score(
                    [item.truth for item in group],
                    [item.predicted for item in group],
                    [item.score for item in group],
                ),
            )
            for (row_id, diagnosis), group in sorted(groups.items())
        )
        self.metrics.write(result)
        return result
