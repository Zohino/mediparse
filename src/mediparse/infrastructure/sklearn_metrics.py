"""Metriky binární klasifikace z knihovního výpočtu scikit-learn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from mediparse.domain.evaluation import BinaryMetrics

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class SklearnScorer:
    """Metriky pozitivní třídy včetně ROC-AUC a PR-AUC."""

    @staticmethod
    def score(
        truth: Sequence[bool], predicted: Sequence[bool], scores: Sequence[float]
    ) -> BinaryMetrics:
        """Spočte precision, recall, F1, accuracy, ROC-AUC a PR-AUC.

        Nedefinované precision, recall a F1 jsou 0.0. AUC potřebuje obě třídy
        pravdy, což zajišťuje volající.

        Returns:
            Metriky pozitivní třídy.
        """
        precision, recall, f1, _ = precision_recall_fscore_support(
            list(truth), list(predicted), average="binary", zero_division=0.0
        )
        return BinaryMetrics(
            float(precision),
            float(recall),
            float(f1),
            float(accuracy_score(list(truth), list(predicted))),
            float(roc_auc_score(list(truth), list(scores))),
            float(average_precision_score(list(truth), list(scores))),
        )
