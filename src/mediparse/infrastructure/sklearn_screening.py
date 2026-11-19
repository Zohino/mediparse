"""Příznaky a volba prahu rodin screeningu při CV po pacientech."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, assert_never

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from mediparse.domain.evaluation import Classification
from mediparse.domain.screening import (
    ClassifierFamily,
    EncoderSpec,
    TokensSpec,
    VectorizerSpec,
)
from mediparse.infrastructure.complement_nb import ComplementNbLogOdds
from mediparse.infrastructure.fasttext_classifier import FastTextClassifier
from mediparse.infrastructure.machine_hardware import peak_rss_gb
from mediparse.infrastructure.nb_svm import NbSvmClassifier
from mediparse.infrastructure.sklearn_classifier import (
    ensure_tokens,
    fit_converged,
    tfidf_vectorizer,
)
from mediparse.infrastructure.sklearn_split import threshold_splits

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from numpy import ndarray
    from scipy.sparse import csr_matrix
    from sklearn.base import BaseEstimator

    from mediparse.domain.clinical_text import TextNormalization
    from mediparse.domain.identity import SubjectId
    from mediparse.domain.screening import Representation
    from mediparse.domain.seed import Seed
    from mediparse.infrastructure.note_embeddings import NoteEmbeddings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Lsa:
    """Počet komponent a seed redukce sdílené vnějším foldem."""

    components: int
    seed: Seed


def screening_lsa(
    families: Sequence[ClassifierFamily], components: int, seed: Seed
) -> Lsa | None:
    """Připraví redukci pouze pro screening obsahující rodinu nad LSA.

    Returns:
        Komponenty a seed LSA pro LightGBM, jinak žádná redukce.
    """
    if ClassifierFamily.LIGHTGBM_LSA in families:
        return Lsa(components, seed)
    return None


@dataclass(frozen=True)
class ScreeningFeatures:
    """Původní a případně redukovaná reprezentace stejných zpráv foldu."""

    matrix: csr_matrix | ndarray
    lsa: ndarray | None

    def for_family(self, family: ClassifierFamily) -> csr_matrix | ndarray:
        """Vybere tutéž reprezentaci pro fit i klasifikaci rodiny.

        Returns:
            LSA pro LightGBM, jinak původní reprezentace.

        Raises:
            ValueError: LightGBM nedostal LSA (chyba složení screeningu).
        """
        if family is ClassifierFamily.LIGHTGBM_LSA:
            if self.lsa is None:
                msg = "LightGBM vyžaduje příznaky LSA."
                raise ValueError(msg)
            return self.lsa
        return self.matrix


@dataclass(frozen=True)
class TfidfFeaturizer:
    """Příznaky se společným předzpracováním fitované jen na tréninkových textech."""

    normalization: TextNormalization | None
    lsa: Lsa | None = None

    def fit_transform(
        self, spec: VectorizerSpec, train: Sequence[str], test: Sequence[str]
    ) -> tuple[ScreeningFeatures, ScreeningFeatures]:
        """Fitne TF-IDF a volitelnou LSA na vnějším tréninku a transformuje test.

        Returns:
            Obě reprezentace tréninku a testu v pořadí vstupů.

        Raises:
            InvalidInputError: Z tréninkových textů nevznikne žádný příznak.
        """
        vectorizer = tfidf_vectorizer(
            self.normalization, analyzer=spec.analyzer, ngram_range=spec.ngram_range
        )
        ensure_tokens(vectorizer, train)
        features = vectorizer.fit_transform(list(train))
        test_features = vectorizer.transform(list(test))
        train_lsa, test_lsa = None, None
        if self.lsa is not None:
            svd = TruncatedSVD(
                n_components=self.lsa.components, random_state=self.lsa.seed
            )
            # Jednou na vnější fold jako TF-IDF: vnitřní CV prahu vidí SVD ze svých
            # validací, test vnějšího foldu do učení redukce nevstoupí.
            train_lsa = svd.fit_transform(features)
            test_lsa = svd.transform(test_features)
        logger.info(
            "Vektorizér %s: %d trénovacích a %d testovacích zpráv, %d příznaků, "
            "%d nenulových hodnot, %s.",
            spec.name,
            len(train),
            len(test),
            len(vectorizer.vocabulary_),
            features.nnz,
            peak_rss_gb(),
        )
        return ScreeningFeatures(features, train_lsa), ScreeningFeatures(
            test_features, test_lsa
        )


@dataclass(frozen=True)
class TokenFeaturizer:
    """Slovní unigramy se společným předzpracováním bez matice TF-IDF."""

    normalization: TextNormalization | None

    def fit_transform(
        self, spec: TokensSpec, train: Sequence[str], test: Sequence[str]
    ) -> tuple[ScreeningFeatures, ScreeningFeatures]:
        """Spojí tokeny analyzátoru do dokumentů pro trénink i test.

        Returns:
            Pole řetězců ve vstupním pořadí bez LSA.
        """
        del spec
        vectorizer = tfidf_vectorizer(self.normalization)
        ensure_tokens(vectorizer, train)
        analyzer = vectorizer.build_analyzer()
        return (
            ScreeningFeatures(
                np.array([" ".join(analyzer(text)) for text in train], dtype=object),
                None,
            ),
            ScreeningFeatures(
                np.array([" ".join(analyzer(text)) for text in test], dtype=object),
                None,
            ),
        )


@dataclass(frozen=True)
class EncoderFeaturizer:
    """Embeddingy standardizované podle vnějšího tréninku."""

    embeddings: NoteEmbeddings

    def fit_transform(
        self, spec: EncoderSpec, train: Sequence[str], test: Sequence[str]
    ) -> tuple[ScreeningFeatures, ScreeningFeatures]:
        """Fitne škálování na tréninku a transformuje test.

        Returns:
            Husté příznaky v pořadí zpráv bez LSA.
        """
        scaler = StandardScaler()
        # Stejně jako TF-IDF se fituje jednou na vnější fold; vnitřní CV prahu
        # vidí scaler ze svých validací, vnější test do fitu nevstoupí.
        # Společná matice a řez umožní načíst model v prvním foldu jen jednou.
        matrix = self.embeddings.embed(spec, [*train, *test])
        train_features = scaler.fit_transform(matrix[: len(train)])
        test_features = scaler.transform(matrix[len(train) :])
        return ScreeningFeatures(train_features, None), ScreeningFeatures(
            test_features, None
        )


@dataclass(frozen=True)
class ScreeningFeaturizer:
    """Vybere adaptér příznaků podle specifikace reprezentace."""

    tfidf: TfidfFeaturizer
    encoder: EncoderFeaturizer
    tokens: TokenFeaturizer

    def fit_transform(
        self, spec: Representation, train: Sequence[str], test: Sequence[str]
    ) -> tuple[ScreeningFeatures, ScreeningFeatures]:
        """Deleguje příznaky podle typu reprezentace.

        Returns:
            Reprezentace tréninku a testu.
        """
        match spec:
            case VectorizerSpec():
                return self.tfidf.fit_transform(spec, train, test)
            case EncoderSpec():
                return self.encoder.fit_transform(spec, train, test)
            case TokensSpec():
                return self.tokens.fit_transform(spec, train, test)
            case _:
                assert_never(spec)


def classifier_estimator(
    family: ClassifierFamily, seed: Seed, scratch: Path
) -> BaseEstimator:
    """Vytvoří estimator rodiny; případnou redukci už provedl featurizér.

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
            # LightGBM doporučuje force_row_wise pro stabilitu mezi počty vláken.
            return LGBMClassifier(
                random_state=seed,
                deterministic=True,
                force_row_wise=True,
                verbose=-1,
            )
        case ClassifierFamily.FASTTEXT:
            return FastTextClassifier(scratch, random_state=seed)
        case _:
            assert_never(family)


@dataclass(frozen=True)
class SklearnTunedClassifier:
    """Klasifikátor s prahem maximalizujícím F1 vnitřní CV."""

    family: ClassifierFamily
    model: TunedThresholdClassifierCV
    threshold: float
    converged: bool

    def classify(self, features: ScreeningFeatures) -> tuple[Classification, ...]:
        """Predikuje při zvoleném prahu a vrací odezvu na stejné škále.

        Returns:
            Labely tuneru a pravděpodobnosti, případně rozhodovací skóre.
        """
        selected = features.for_family(self.family)
        labels = self.model.predict(selected)
        if hasattr(self.model.estimator_, "predict_proba"):
            scores = self.model.predict_proba(selected)[:, 1]
        else:
            scores = self.model.decision_function(selected)
        return tuple(
            Classification(bool(label), float(score))
            for label, score in zip(labels, scores, strict=True)
        )


@dataclass(frozen=True)
class SklearnThresholdTuner:
    """Volba prahu a refit s evidencí nekonvergence všech vnitřních fitů."""

    scratch: Path
    estimator: Callable[[ClassifierFamily, Seed, Path], BaseEstimator] = (
        classifier_estimator
    )

    def fit(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
        self,
        family: ClassifierFamily,
        features: ScreeningFeatures,
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
            self.estimator(family, seed, self.scratch),
            scoring="f1",
            cv=threshold_splits(labels, subjects, folds, seed),
        )
        converged = fit_converged(
            lambda: model.fit(features.for_family(family), list(labels))
        )
        logger.info("Rodina %s: %s.", family, peak_rss_gb())
        return SklearnTunedClassifier(
            family, model, float(model.best_threshold_), converged
        )
