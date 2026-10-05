"""Klasický model v scikit-learn: odložení po pacientech, TF-IDF s lineárním SVM, skops a metriky."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import skops.io as sio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from mediparse.domain.smoketest_training import BinaryMetrics, Holdout

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


@dataclass(frozen=True)
class SklearnClassifier:
    """Natrénovaná pipeline TF-IDF a lineárního SVM."""

    pipeline: Pipeline

    def predict(self, texts: Sequence[str]) -> tuple[bool, ...]:
        """Predikuje přítomnost diagnózy v textech.

        Returns:
            Predikce v pořadí textů.
        """
        return tuple(bool(label) for label in self.pipeline.predict(list(texts)))


@dataclass(frozen=True)
class GroupedHoldout:
    """Odložení první části ze stratifikovaného rozdělení po pacientech."""

    folds: int
    seed: int

    def split(self, labels: Sequence[bool], groups: Sequence[int]) -> Holdout:
        """Rozdělí zprávy tak, že pacient je celý v jedné části.

        Returns:
            Indexy tréninkové části a testovací části, tedy prvního foldu.
        """
        splitter = StratifiedGroupKFold(
            n_splits=self.folds, shuffle=True, random_state=self.seed
        )
        train, test = next(splitter.split(list(labels), list(labels), list(groups)))
        return Holdout(tuple(map(int, train)), tuple(map(int, test)))


@dataclass(frozen=True)
class LinearSvmTrainer:
    """Trénink TF-IDF s lineárním SVM."""

    regularization: float
    seed: int

    def fit(self, texts: Sequence[str], labels: Sequence[bool]) -> SklearnClassifier:
        """Natrénuje pipeline na textech a labelech.

        Returns:
            Natrénovaný klasifikátor.
        """
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(sublinear_tf=True)),
            ("svm", LinearSVC(C=self.regularization, random_state=self.seed)),
        ])
        pipeline.fit(list(texts), list(labels))
        return SklearnClassifier(pipeline)


@dataclass(frozen=True)
class SklearnScorer:
    """Metriky pozitivní třídy z knihovního výpočtu."""

    @staticmethod
    def score(truth: Sequence[bool], predicted: Sequence[bool]) -> BinaryMetrics:
        """Spočte precision, recall, F1 a accuracy; nedefinované hodnoty jsou 0.0.

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
        )


@dataclass(frozen=True)
class SkopsModelFile:
    """Model uložený ve formátu skops, který se načítá bez důvěry v cizí typy."""

    path: Path

    def save(self, classifier: object) -> None:
        """Uloží pipeline klasifikátoru a nadřazený adresář založí.

        Raises:
            TypeError: Klasifikátor nevznikl v tomto adaptéru.
        """
        if not isinstance(classifier, SklearnClassifier):
            msg = f"Uložit lze jen SklearnClassifier, ne {type(classifier).__name__}."
            raise TypeError(msg)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        sio.dump(classifier.pipeline, self.path)

    def load(self) -> SklearnClassifier:
        """Načte model s prázdným seznamem důvěryhodných typů.

        Returns:
            Klasifikátor s načtenou pipeline.
        """
        return SklearnClassifier(sio.load(self.path, trusted=[]))
