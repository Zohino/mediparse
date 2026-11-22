"""Skórování větví stupně s fitováním reprezentace v každém pokusu."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from itertools import batched
from typing import TYPE_CHECKING, assert_never, cast

import numpy as np
from gensim.models import Word2Vec
from gensim.models.word2vec import MAX_WORDS_IN_BATCH
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.representation import (
    ClassifierFamily,
    EncoderSpec,
    TfidfRepresentation,
    TokensSpec,
    Word2VecRepresentation,
)
from mediparse.domain.tuning import ValidationScores
from mediparse.infrastructure.sklearn_classifier import (
    TfidfParameters,
    ensure_tokens,
    fit_converged,
    tfidf_from,
    tfidf_vectorizer,
)
from mediparse.infrastructure.sklearn_families import (
    classifier_estimator,
    positive_scores,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from gensim.models import KeyedVectors
    from numpy import ndarray
    from scipy.sparse import csr_matrix

    from mediparse.domain.clinical_text import TextNormalization
    from mediparse.domain.seed import Seed
    from mediparse.domain.split import Holdout
    from mediparse.domain.stage_tuning import Branch, StageTuningConfig
    from mediparse.domain.tuning import ParamValue
    from mediparse.infrastructure.note_embeddings import NoteEmbeddings


@dataclass(frozen=True)
class SklearnStageScorer:
    """Fitne reprezentaci jednou pro pět diagnóz pokusu a foldu."""

    config: StageTuningConfig
    normalization: TextNormalization | None
    seed: Seed
    scratch: Path
    embeddings: NoteEmbeddings

    def score(
        self, params: Mapping[str, ParamValue], fold: Holdout
    ) -> ValidationScores:
        """Vrátí skóre validačních zpráv bez volby prahu v adaptéru.

        Returns:
            Pět vektorů skóre a konvergence všech fitů.

        Raises:
            InvalidInputError: Parametry nebo trénovací slovník nejsou platné.
        """
        try:
            return self._score(params, fold)
        except ValueError as error:
            msg = f"Parametry nebo trénovací slovník stupně nejsou platné: {error}"
            raise InvalidInputError(msg) from error

    def _score(
        self, params: Mapping[str, ParamValue], fold: Holdout
    ) -> ValidationScores:
        branch = self.config.branch(params)
        rep, family = branch.params_of(params)
        train, validation, n_features = self._features(branch, rep, fold)
        scores = {}
        converged = True
        for diagnosis in Diagnosis:
            model = classifier_estimator(
                branch.family.family, self.seed, self.scratch
            ).set_params(**family)
            if branch.family.family is ClassifierFamily.LOGREG:
                # Rozpočet řešiče není hyperparametr a nesmí znevýhodnit rodinu.
                model.set_params(max_iter=1000)
            converged &= fit_converged(
                partial(
                    model.fit, train, [note.has_label(diagnosis) for note in fold.train]
                )
            )
            scores[diagnosis] = tuple(
                float(v) for v in positive_scores(model, validation)
            )
        return ValidationScores(scores, converged, n_features)

    def _features(
        self, branch: Branch, rep_params: Mapping[str, ParamValue], fold: Holdout
    ) -> tuple[ndarray | csr_matrix, ndarray | csr_matrix, int]:
        train_texts = [note.text for note in fold.train]
        validation_texts = [note.text for note in fold.test]
        match branch.representation:
            case EncoderSpec() as spec:
                train, validation = encoder_features(
                    self.embeddings, spec, train_texts, validation_texts
                )
                return train, validation, train.shape[1]
            case Word2VecRepresentation() as spec:
                # Více vláken z configu zrychlí trénink za cenu nedeterminismu.
                train, validation = word2vec_features(
                    self.normalization,
                    {**rep_params, "workers": spec.workers, "seed": self.seed},
                    train_texts,
                    validation_texts,
                )
                return train, validation, train.shape[1]
            case TokensSpec():
                train, validation = token_documents(
                    self.normalization, train_texts, validation_texts
                )
                n_features = len({
                    token for document in train for token in document.split()
                })
                return train, validation, n_features
            case TfidfRepresentation():
                return self._tfidf(branch, rep_params, train_texts, validation_texts)
            case _:
                assert_never(branch.representation)

    def _tfidf(
        self,
        branch: Branch,
        rep_params: Mapping[str, ParamValue],
        train_texts: Sequence[str],
        validation_texts: Sequence[str],
    ) -> tuple[ndarray | csr_matrix, ndarray | csr_matrix, int]:
        tfidf = TfidfParameters.model_validate(rep_params)
        lsa = (
            self.config.lsa
            if branch.family.family is ClassifierFamily.LIGHTGBM_LSA
            else None
        )
        # SVD alokuje husté matice úměrné šířce; strop drží paměť, jiné rodiny mají celý slovník.
        vectorizer = tfidf_from(
            tfidf,
            cast("TfidfRepresentation", branch.representation).analyzer,
            self.normalization,
            max_features=lsa.max_features if lsa else None,
        )
        ensure_tokens(vectorizer, train_texts)
        train = vectorizer.fit_transform(train_texts)
        validation = vectorizer.transform(validation_texts)
        n_features = train.shape[1]
        if lsa is not None:
            reducer = TruncatedSVD(n_components=lsa.components, random_state=self.seed)
            train = reducer.fit_transform(train)
            validation = reducer.transform(validation)
        return train, validation, n_features


def encoder_features(
    embeddings: NoteEmbeddings,
    spec: EncoderSpec,
    train: Sequence[str],
    validation: Sequence[str],
) -> tuple[ndarray, ndarray]:
    """Standardizuje embeddingy podle tréninku a transformuje validaci.

    Returns:
        Husté matice v pořadí vstupních zpráv.
    """
    # Validace nesmí ovlivnit škálování; společná matice načte encoder jen jednou.
    matrix = embeddings.embed(spec, [*train, *validation])
    return _standardized(matrix[: len(train)], matrix[len(train) :])


def token_documents(
    normalization: TextNormalization | None,
    train: Sequence[str],
    validation: Sequence[str],
) -> tuple[ndarray, ndarray]:
    """Spojí tokeny společného analyzátoru do dokumentů.

    Returns:
        Pole řetězců pro trénink a validaci ve vstupním pořadí.

    Raises:
        InvalidInputError: Trénovací texty neobsahují token.
    """
    analyzer = _train_analyzer(normalization, train)
    # Pevná šířka <U zarovná zprávy na nejdelší: 44 tis. krát 60 tis. znaků je ~9,8 GiB.
    return (
        np.array([" ".join(analyzer(text)) for text in train], dtype=object),
        np.array([" ".join(analyzer(text)) for text in validation], dtype=object),
    )


def _train_analyzer(
    normalization: TextNormalization | None, train: Sequence[str]
) -> Callable[[str], list[str]]:
    vectorizer = tfidf_vectorizer(normalization)
    ensure_tokens(vectorizer, train)
    return vectorizer.build_analyzer()


def _standardized(train: ndarray, validation: ndarray) -> tuple[ndarray, ndarray]:
    scaler = StandardScaler()
    return scaler.fit_transform(train), scaler.transform(validation)


def word2vec_features(
    normalization: TextNormalization | None,
    options: Mapping[str, ParamValue],
    train: Sequence[str],
    validation: Sequence[str],
) -> tuple[ndarray, ndarray]:
    """Naučí word2vec jen na tréninku a standardizuje průměry slov.

    Returns:
        Husté matice zpráv v pořadí tréninku a validace.

    Raises:
        InvalidInputError: Trénink neobsahuje tokeny nebo je slovník prázdný.
    """
    analyzer = _train_analyzer(normalization, train)
    documents = [analyzer(text) for text in train]
    vectors = _trained_vectors(options, documents)
    return _standardized(
        _mean_vectors(vectors, documents),
        _mean_vectors(vectors, [analyzer(text) for text in validation]),
    )


def _trained_vectors(
    options: Mapping[str, ParamValue], documents: Sequence[list[str]]
) -> KeyedVectors:
    # Gensim slova za 10 000. pozicí věty tiše zahodí; kusy zachovají všechny tokeny.
    sentences = [
        list(chunk)
        for tokens in documents
        for chunk in batched(tokens, MAX_WORDS_IN_BATCH, strict=False)
    ]
    model = Word2Vec(**options)
    model.build_vocab(sentences)
    # Gensim prázdný slovník neodmítne, train by pak skončil nesrozumitelným RuntimeError.
    if not model.wv.index_to_key:
        msg = "Žádné slovo nedosáhlo min_count, slovník word2vec by byl prázdný."
        raise InvalidInputError(msg)
    model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
    return model.wv


def _mean_vectors(vectors: KeyedVectors, documents: Sequence[list[str]]) -> ndarray:
    # Výchozí pre_normalize=True by průměrovalo jednotkové vektory místo vektorů slov.
    return np.array([
        vectors.get_mean_vector(tokens, pre_normalize=False)
        if tokens
        else np.zeros(vectors.vector_size, dtype=np.float32)
        for tokens in documents
    ])
