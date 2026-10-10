"""Klasický model v scikit-learn: odložení po pacientech, TF-IDF s lineárním SVM a skops."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import skops.io as sio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from skops.io.exceptions import UntrustedTypesFoundException

from mediparse.domain.evaluation import Classification
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.smoketest_training import Holdout

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
        """
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(sublinear_tf=True)),
            ("svm", LinearSVC(C=self.regularization, random_state=self.seed)),
        ])
        pipeline.fit(list(texts), list(labels))
        return SklearnClassifier(pipeline)


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
