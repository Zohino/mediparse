"""Klasický model v scikit-learn: TF-IDF s lineárním SVM a skops."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import skops.io as sio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from skops.io.exceptions import UntrustedTypesFoundException

from mediparse.domain.evaluation import Classification
from mediparse.domain.inputs import InvalidInputError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


@dataclass(frozen=True)
class SklearnClassifier:
    """Natrénovaná pipeline TF-IDF a lineárního SVM."""

    pipeline: Pipeline

    def classify(self, texts: Sequence[str]) -> tuple[Classification, ...]:
        """Určí label a rozhodovací skóre lineárního SVM v jednom průchodu.

        Returns:
            Klasifikace v pořadí textů; label je ``predict``, skóre ``decision_function``.
        """
        listed = list(texts)
        labels = self.pipeline.predict(listed)
        scores = self.pipeline.decision_function(listed)
        return tuple(
            Classification(bool(label), float(score))
            for label, score in zip(labels, scores, strict=True)
        )


ESTIMATOR: Final = "sklearn TfidfVectorizer+LinearSVC"


@dataclass(frozen=True)
class LinearSvmTrainer:
    """Trénink TF-IDF s lineárním SVM."""

    regularization: float
    seed: int

    def fit(self, texts: Sequence[str], labels: Sequence[bool]) -> SklearnClassifier:
        """Natrénuje pipeline na textech a labelech.

        Returns:
            Natrénovaný klasifikátor.

        Raises:
            InvalidInputError: Z žádného textu nevznikne token, třeba po stripu
                diagnostických sekcí ze zpráv, které z nich sestávají.
        """
        vectorizer = TfidfVectorizer(sublinear_tf=True)
        analyzer = vectorizer.build_analyzer()
        if not any(analyzer(text) for text in texts):
            msg = (
                "Trénovací texty neobsahují žádný token, slovník TF-IDF by byl prázdný."
            )
            raise InvalidInputError(msg)
        pipeline = Pipeline([
            ("tfidf", vectorizer),
            ("svm", LinearSVC(C=self.regularization, random_state=self.seed)),
        ])
        pipeline.fit(list(texts), list(labels))
        return SklearnClassifier(pipeline)


@dataclass(frozen=True)
class SkopsModelFile:
    """Model uložený ve formátu skops, který se načítá bez důvěry v cizí typy."""

    path: Path

    def save(self, classifier: SklearnClassifier) -> None:
        """Uloží pipeline klasifikátoru a nadřazený adresář založí."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        sio.dump(classifier.pipeline, self.path)

    def load(self) -> SklearnClassifier:
        """Načte model s prázdným seznamem důvěryhodných typů.

        Returns:
            Klasifikátor s načtenou pipeline.

        Raises:
            InvalidInputError: Soubor modelu neexistuje, je poškozený nebo
                obsahuje typ mimo důvěryhodné.
        """
        try:
            pipeline = sio.load(self.path, trusted=[])
        except FileNotFoundError as error:
            msg = f"Soubor modelu {self.path} neexistuje."
            raise InvalidInputError(msg) from error
        except (zipfile.BadZipFile, UntrustedTypesFoundException) as error:
            msg = f"Soubor modelu {self.path} nelze bezpečně načíst: {error}"
            raise InvalidInputError(msg) from error
        return SklearnClassifier(pipeline)
