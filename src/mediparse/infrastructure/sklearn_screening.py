"""TF-IDF a volba prahu klasických rodin při CV po pacientech."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, assert_never

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from mediparse.domain.classical_screening import ClassifierFamily
from mediparse.domain.evaluation import Classification
from mediparse.infrastructure.complement_nb import ComplementNbLogOdds
from mediparse.infrastructure.nb_svm import NbSvmClassifier
from mediparse.infrastructure.sklearn_classifier import ensure_tokens, tfidf_vectorizer
from mediparse.infrastructure.sklearn_split import StratifiedGroupFolds

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy import ndarray
    from scipy.sparse import csr_matrix
    from sklearn.base import BaseEstimator

    from mediparse.domain.classical_screening import VectorizerSpec
    from mediparse.domain.clinical_text import TextNormalization
    from mediparse.domain.identity import SubjectId
    from mediparse.domain.seed import Seed

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TfidfFeaturizer:
    """Příznaky se společným předzpracováním fitované jen na tréninkových textech."""

    normalization: TextNormalization | None

    def fit_transform(
        self, spec: VectorizerSpec, train: Sequence[str], test: Sequence[str]
    ) -> tuple[csr_matrix, csr_matrix]:
        """Fitne TF-IDF na tréninku a transformuje odložený test.

        Returns:
            Matice příznaků tréninku a testu v pořadí vstupů.

        Raises:
            InvalidInputError: Z tréninkových textů nevznikne žádný příznak.
        """
        vectorizer = tfidf_vectorizer(
            self.normalization, analyzer=spec.analyzer, ngram_range=spec.ngram_range
        )
        ensure_tokens(vectorizer, train)
        features = vectorizer.fit_transform(list(train))
        logger.info(
            "Vektorizér %s: %d trénovacích a %d testovacích zpráv, %d příznaků.",
            spec.name,
            len(train),
            len(test),
            len(vectorizer.vocabulary_),
        )
        return features, vectorizer.transform(list(test))


@dataclass(frozen=True)
class ClassifierFactory:
    """Továrna rodin s explicitním počtem komponent LSA z configu screeningu."""

    lsa_components: int

    def __call__(self, family: ClassifierFamily, seed: Seed) -> BaseEstimator:
        """Vytvoří estimator rodiny, LSA se učí uvnitř každého fitu.

        Returns:
            Nenatrénovaný estimator s výchozími hyperparametry rodiny.
        """
        match family:
            case ClassifierFamily.COMPLEMENT_NB:
                return ComplementNbLogOdds()
            case ClassifierFamily.LOGREG:
                return LogisticRegression()
            case ClassifierFamily.LINEAR_SVM:
                return LinearSVC(random_state=seed)
            case ClassifierFamily.NB_SVM:
                return NbSvmClassifier(random_state=seed)
            case ClassifierFamily.LIGHTGBM_LSA:
                return make_pipeline(
                    TruncatedSVD(n_components=self.lsa_components, random_state=seed),
                    # LightGBM doporučuje force_row_wise pro stabilitu mezi počty vláken.
                    LGBMClassifier(
                        random_state=seed,
                        deterministic=True,
                        force_row_wise=True,
                        verbose=-1,
                    ),
                )
            case _:
                assert_never(family)


def threshold_splits(
    labels: Sequence[bool], subjects: Sequence[SubjectId], folds: int, seed: Seed
) -> list[tuple[ndarray, ndarray]]:
    """Převede stratifikované části pacientů na indexy zpráv pro vnitřní CV.

    Returns:
        Páry indexů tréninku a testu; každý pacient náleží jedinému testu.
    """
    # Předpočítané indexy nevyžadují globální enable_metadata_routing pro groups.
    partitions = StratifiedGroupFolds.partition(
        subjects, [int(label) for label in labels], folds, seed
    )
    return [
        (
            np.array(
                [i for i, subject in enumerate(subjects) if subject not in part],
                dtype=int,
            ),
            np.array(
                [i for i, subject in enumerate(subjects) if subject in part], dtype=int
            ),
        )
        for part in partitions
    ]


@dataclass(frozen=True)
class SklearnTunedClassifier:
    """Klasifikátor s prahem maximalizujícím F1 vnitřní CV."""

    model: TunedThresholdClassifierCV
    threshold: float
    converged: bool

    def classify(self, features: csr_matrix) -> tuple[Classification, ...]:
        """Predikuje při zvoleném prahu a vrací odezvu na stejné škále.

        Returns:
            Labely tuneru a pravděpodobnosti, případně rozhodovací skóre.
        """
        labels = self.model.predict(features)
        if hasattr(self.model.estimator_, "predict_proba"):
            scores = self.model.predict_proba(features)[:, 1]
        else:
            scores = self.model.decision_function(features)
        return tuple(
            Classification(bool(label), float(score))
            for label, score in zip(labels, scores, strict=True)
        )


@dataclass(frozen=True)
class SklearnThresholdTuner:
    """Volba prahu a refit s evidencí nekonvergence všech vnitřních fitů."""

    estimator: Callable[[ClassifierFamily, Seed], BaseEstimator]

    def fit(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
        self,
        family: ClassifierFamily,
        features: csr_matrix,
        labels: Sequence[bool],
        subjects: Sequence[SubjectId],
        folds: int,
        seed: Seed,
    ) -> SklearnTunedClassifier:
        """Zvolí práh podle F1 a refitne estimator na všech dodaných příznacích.

        Returns:
            Model s prahem a příznakem konvergence včetně vnitřních fitů.
        """
        # Výchozí n_jobs drží fity ve stejném procesu, kde lze zachytit jejich varování.
        model = TunedThresholdClassifierCV(
            self.estimator(family, seed),
            scoring="f1",
            cv=threshold_splits(labels, subjects, folds, seed),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(features, list(labels))
        # record=True zachytí i jiné kategorie; jejich diagnostika nesmí v reálném běhu zmizet.
        for item in caught:
            if not issubclass(item.category, ConvergenceWarning):
                warnings.warn_explicit(
                    item.message, item.category, item.filename, item.lineno
                )
        converged = not any(
            issubclass(item.category, ConvergenceWarning) for item in caught
        )
        return SklearnTunedClassifier(model, float(model.best_threshold_), converged)
