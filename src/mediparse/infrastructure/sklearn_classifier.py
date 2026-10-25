"""Klasický model v scikit-learn: TF-IDF s lineárním SVM a skops."""

from __future__ import annotations

import functools
import zipfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import skops.io as sio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from skops.io.exceptions import UntrustedTypesFoundException

from mediparse.domain.clinical_text import (
    TextNormalization,
    expand_diagnosis_acronyms,
    normalize_clinical_text,
)
from mediparse.domain.evaluation import Classification
from mediparse.domain.inputs import InvalidInputError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
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


def tfidf_preprocessor(text: str, acronyms: Mapping[str, str]) -> str:
    """Normalizace, rozvinutí zkratek diagnóz a snížení písmen před tokenizací TF-IDF.

    Returns:
        Normalizovaný text malými písmeny, zkratky ze slovníku rozvinuté.
    """
    # Rozvinutí rozlišuje velikost písmen, proto jde před .lower(); jde po
    # normalizaci, aby ___COPD nebylo zablokované podtržítky.
    expanded = expand_diagnosis_acronyms(normalize_clinical_text(text), acronyms)
    # Vlastní preprocessor vypíná lowercase vectorizeru, snížení písmen proto nese
    # tato funkce. Bez něj by CAST a Cast byly dva příznaky.
    return expanded.lower()


# Seznam platí při načtení, ne při uložení: skops model uloží vždy a při načtení
# pustí funkci jen z tohoto seznamu. Přejmenování funkce proto rozbije načítání
# dříve uložených modelů. Slovník zkratek nese functools.partial, který musí být
# v seznamu také; funkci uvnitř partialu skops kontroluje zvlášť.
TRUSTED_FUNCTIONS: Final = (
    "functools.partial",
    "mediparse.infrastructure.sklearn_classifier.tfidf_preprocessor",
)


@dataclass(frozen=True)
class LinearSvmTrainer:
    """Trénink TF-IDF s lineárním SVM."""

    regularization: float
    seed: int
    normalization: TextNormalization | None

    def fit(self, texts: Sequence[str], labels: Sequence[bool]) -> SklearnClassifier:
        """Natrénuje pipeline na textech a labelech.

        Returns:
            Natrénovaný klasifikátor.

        Raises:
            InvalidInputError: Z žádného textu nevznikne token, třeba po stripu
                diagnostických sekcí ze zpráv, které z nich sestávají.
        """
        vectorizer = TfidfVectorizer(
            sublinear_tf=True, preprocessor=self._preprocessor()
        )
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

    def _preprocessor(self) -> functools.partial[str] | None:
        if self.normalization is None:
            return None
        # Holý dict, ne Mapping ani pydantic objekt: skops uloží dict, jiný typ
        # by vyžadoval další důvěryhodný typ.
        return functools.partial(
            tfidf_preprocessor, acronyms=dict(self.normalization.acronyms)
        )


@dataclass(frozen=True)
class SkopsModelFile:
    """Model uložený ve formátu skops, který se načítá bez důvěry v cizí typy a funkce."""

    path: Path

    def save(self, classifier: SklearnClassifier) -> None:
        """Uloží pipeline klasifikátoru a nadřazený adresář založí."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        sio.dump(classifier.pipeline, self.path)

    def load(self) -> SklearnClassifier:
        """Načte model; z cizích typů důvěřuje jen funkcím ze ``TRUSTED_FUNCTIONS``.

        Returns:
            Klasifikátor s načtenou pipeline.

        Raises:
            InvalidInputError: Soubor modelu neexistuje, je poškozený nebo
                obsahuje typ mimo důvěryhodné.
        """
        try:
            pipeline = sio.load(self.path, trusted=list(TRUSTED_FUNCTIONS))
        except FileNotFoundError as error:
            msg = f"Soubor modelu {self.path} neexistuje."
            raise InvalidInputError(msg) from error
        except (zipfile.BadZipFile, UntrustedTypesFoundException) as error:
            msg = f"Soubor modelu {self.path} nelze bezpečně načíst: {error}"
            raise InvalidInputError(msg) from error
        return SklearnClassifier(pipeline)
