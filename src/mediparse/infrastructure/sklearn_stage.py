"""Skórování větví stupně se sdílenými četnostmi n-gramů mezi pokusy."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Final, Literal, Protocol, assert_never, cast

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

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
    EMPTY_VOCABULARY,
    TfidfParameters,
    fit_converged,
    require_tokens,
    tfidf_vectorizer,
)
from mediparse.infrastructure.sklearn_families import (
    classifier_estimator,
    positive_scores,
)
from mediparse.infrastructure.sklearn_split import holdout_mask
from mediparse.infrastructure.word2vec import trained_vectors

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence
    from pathlib import Path

    from gensim.models import KeyedVectors
    from numpy import ndarray
    from scipy.sparse import csr_matrix
    from sklearn.base import BaseEstimator

    from mediparse.domain.clinical_text import TextNormalization
    from mediparse.domain.input_note import InputNote
    from mediparse.domain.seed import Seed
    from mediparse.domain.split import Holdout
    from mediparse.domain.stage_tuning import Branch, StageNetwork, StageTuningConfig
    from mediparse.domain.tuning import ParamValue
    from mediparse.infrastructure.note_embeddings import NoteEmbeddings


MIN_PADDED_WORD_LENGTH: Final = 3


@dataclass(frozen=True)
class SklearnStageScorer:
    """Sdílí četnosti mezi pokusy a fitne reprezentaci pro pět diagnóz foldu."""

    config: StageTuningConfig
    normalizations: Mapping[str, TextNormalization | None]
    seed: Seed
    scratch: Path
    embeddings: NoteEmbeddings
    jobs: int = 1
    _prepared: dict[tuple[str, str], str] = field(
        default_factory=dict, init=False, repr=False
    )
    _counts: dict[str, NgramCounts] = field(
        default_factory=dict, init=False, repr=False
    )

    def _texts(
        self, name: str, notes: Sequence[InputNote], prepare: Callable[[str], str]
    ) -> list[str]:
        texts = []
        for note in notes:
            # Text v klíči nevyžaduje, aby note_id vždy určovalo tentýž obsah.
            key = (name, note.text)
            if key not in self._prepared:
                self._prepared[key] = prepare(note.text)
            texts.append(self._prepared[key])
        return texts

    def score_folds(
        self, params: Mapping[str, ParamValue], folds: Sequence[Holdout]
    ) -> tuple[ValidationScores, ...]:
        """Vyhodnotí foldy v pořadí vstupu.

        Returns:
            Skóre každého foldu bez volby prahu.
        """
        return tuple(self.score(params, fold) for fold in folds)

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
        if branch.family.family is ClassifierFamily.MULTIRESCNN:
            return self._multilabel_scores(family, train, validation, fold, n_features)
        models = []
        for _ in Diagnosis:
            model = classifier_estimator(
                branch.family.family, self.seed, self.scratch
            ).set_params(**family)
            if branch.family.family is ClassifierFamily.LOGREG:
                # Rozpočet řešiče není hyperparametr a nesmí znevýhodnit rodinu.
                model.set_params(max_iter=1000)
            models.append(model)
        # DUA: výchozí memmap v /dev/shm by ležel mimo chráněný scratch.
        results = Parallel(n_jobs=self.jobs, backend="loky", temp_folder=self.scratch)(
            delayed(_diagnosis_scores)(
                model,
                train,
                [note.has_label(diagnosis) for note in fold.train],
                validation,
            )
            for diagnosis, model in zip(Diagnosis, models, strict=True)
        )
        scores, converged = zip(*results, strict=True)
        return ValidationScores(
            dict(zip(Diagnosis, scores, strict=True)), all(converged), n_features
        )

    def _multilabel_scores(
        self,
        family: Mapping[str, ParamValue],
        train: ndarray | csr_matrix,
        validation: ndarray | csr_matrix,
        fold: Holdout,
        n_features: int,
    ) -> ValidationScores:
        # Validátor zaručuje síť právě s rodinou multirescnn.
        network = cast("StageNetwork", self.config.network)
        stop = holdout_mask(fold.train, network.stopping_parts, self.seed)
        labels = np.array([
            [note.has_label(d) for d in Diagnosis] for note in fold.train
        ])
        model = classifier_estimator(
            ClassifierFamily.MULTIRESCNN, self.seed, self.scratch
        ).set_params(**family, **network.model_dump(exclude={"stopping_parts"}))
        model.fit(train[~stop], labels[~stop], train[stop], labels[stop])
        scores = model.decision_function(validation)
        return ValidationScores(
            {d: tuple(float(v) for v in scores[:, j]) for j, d in enumerate(Diagnosis)},
            model.converged_,
            n_features,
        )

    def _features(
        self, branch: Branch, rep_params: Mapping[str, ParamValue], fold: Holdout
    ) -> tuple[ndarray | csr_matrix, ndarray | csr_matrix, int]:
        name = branch.representation.name
        normalization = self.normalizations[name]
        match branch.representation:
            case EncoderSpec() as spec:
                prepare = (
                    tfidf_vectorizer(normalization).build_preprocessor()
                    if normalization is not None
                    else str
                )
                train_texts = self._texts(name, fold.train, prepare)
                validation_texts = self._texts(name, fold.test, prepare)
                train, validation = encoder_features(
                    self.embeddings, spec, train_texts, validation_texts
                )
                return train, validation, train.shape[1]
            case Word2VecRepresentation() as spec:
                # Více vláken z configu zrychlí trénink za cenu nedeterminismu.
                prepare = joined_tokens(normalization)
                train, validation = word2vec_features(
                    {**rep_params, "workers": spec.workers, "seed": self.seed},
                    self._texts(name, fold.train, prepare),
                    self._texts(name, fold.test, prepare),
                )
                return train, validation, train.shape[1]
            case TokensSpec():
                prepare = joined_tokens(normalization)
                train, validation = token_documents(
                    self._texts(name, fold.train, prepare),
                    self._texts(name, fold.test, prepare),
                )
                n_features = len({
                    token for document in train for token in document.split()
                })
                return train, validation, n_features
            case TfidfRepresentation() as spec:
                prepare = tfidf_vectorizer(normalization).build_preprocessor()
                return self._tfidf(
                    branch,
                    spec,
                    rep_params,
                    self._texts(name, fold.train, prepare),
                    self._texts(name, fold.test, prepare),
                )
            case _:
                assert_never(branch.representation)

    def _tfidf(
        self,
        branch: Branch,
        spec: TfidfRepresentation,
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
        counts = self._ngram_counts(
            spec.name,
            spec.analyzer,
            (tfidf.ngram_min, tfidf.ngram_max),
            [*train_texts, *validation_texts],
        )
        # SVD alokuje husté matice úměrné šířce; strop platí jen pro LSA.
        train, validation = tfidf_features(
            counts,
            train_texts,
            validation_texts,
            tfidf,
            max_features=lsa.max_features if lsa else None,
        )
        n_features = train.shape[1]
        if lsa is not None:
            reducer = TruncatedSVD(n_components=lsa.components, random_state=self.seed)
            train = reducer.fit_transform(train)
            validation = reducer.transform(validation)
        return train, validation, n_features

    def _ngram_counts(
        self,
        name: str,
        analyzer: Literal["word", "char_wb"],
        ngram_range: tuple[int, int],
        texts: Sequence[str],
    ) -> NgramCounts:
        cached = self._counts.get(name)
        if cached is not None:
            low, high = cached.ngram_range
            if (
                low <= ngram_range[0]
                and high >= ngram_range[1]
                and all(text in cached.rows for text in texts)
            ):
                return cached
            ngram_range = (min(low, ngram_range[0]), max(high, ngram_range[1]))
            texts = [*cached.rows, *texts]
        # Počty zprávy na foldu nezávisí; slovník foldu vyberou trénovací řádky.
        # Uvolnění obou odkazů zabrání držení staré matice během přestavby.
        self._counts.pop(name, None)
        del cached
        self._counts[name] = ngram_counts(analyzer, ngram_range, texts)
        return self._counts[name]


@dataclass(frozen=True)
class NgramCounts:
    """Četnosti všech textů reprezentace a metadata pro výběr n-gramů."""

    matrix: csr_matrix
    lengths: ndarray
    rows: Mapping[str, int]
    whole_words: ndarray
    ngram_range: tuple[int, int]

    def of(self, texts: Sequence[str]) -> csr_matrix:
        """Vybere řádky matice v pořadí předaných textů.

        Returns:
            Četnosti zpráv včetně opakovaných textů ve vstupním pořadí.
        """
        return self.matrix[[self.rows[text] for text in texts]]


def ngram_counts(
    analyzer: Literal["word", "char_wb"],
    ngram_range: tuple[int, int],
    texts: Iterable[str],
) -> NgramCounts:
    """Spočítá celočíselné četnosti různých textů v pořadí prvního výskytu.

    Returns:
        Matice s abecedně seřazenými sloupci a metadata rozsahu.
    """
    rows = {text: i for i, text in enumerate(dict.fromkeys(texts))}
    vectorizer = CountVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        lowercase=False,
        dtype=np.int32,
    )
    matrix = vectorizer.fit_transform(list(rows))
    lengths = np.empty(len(vectorizer.vocabulary_), dtype=np.int16)
    whole_words = np.zeros(len(lengths), dtype=bool)
    for term, column in vectorizer.vocabulary_.items():
        # Slovní token neobsahuje mezeru; u znaků se počítají i okrajové výplně.
        lengths[column] = term.count(" ") + 1 if analyzer == "word" else len(term)
        # Sklearn přidá celé krátké slovo jednou i při délce menší než ngram_min.
        whole_words[column] = (
            analyzer == "char_wb"
            and len(term) >= MIN_PADDED_WORD_LENGTH
            and term[0] == term[-1] == " "
        )
    return NgramCounts(matrix, lengths, rows, whole_words, ngram_range)


def tfidf_features(
    counts: NgramCounts,
    train: Sequence[str],
    validation: Sequence[str],
    params: TfidfParameters,
    max_features: int | None,
) -> tuple[csr_matrix, csr_matrix]:
    """Vybere slovník jen z tréninku a zváží četnosti pokusu.

    Returns:
        TF-IDF tréninku a validace se společným trénovacím slovníkem.

    Raises:
        InvalidInputError: Trénink neobsahuje token.
        ValueError: Meze četností se vylučují nebo odstraní celý slovník.
    """
    in_range = (counts.lengths <= params.ngram_max) & (
        (counts.lengths >= params.ngram_min) | counts.whole_words
    )
    columns = np.flatnonzero(in_range)
    train_counts = counts.of(train)[:, columns]
    if train_counts.nnz == 0:
        raise InvalidInputError(EMPTY_VOCABULARY)
    max_doc_count = params.max_df * len(train)
    if max_doc_count < params.min_df:
        msg = "max_df corresponds to < documents than min_df"
        raise ValueError(msg)
    # Kopie CountVectorizer._limit_features (sklearn 1.9.1) zachová i remízy stropu.
    df = np.bincount(train_counts.indices, minlength=train_counts.shape[1])
    mask = (df > 0) & (df >= params.min_df) & (df <= max_doc_count)
    if max_features is not None and mask.sum() > max_features:
        tfs = np.asarray(train_counts.astype(np.float64).sum(axis=0)).ravel()
        selected = np.flatnonzero(mask)[(-tfs[mask]).argsort()[:max_features]]
        mask = np.zeros(len(df), dtype=bool)
        mask[selected] = True
    if not mask.any():
        msg = "After pruning, no terms remain. Try a lower min_df or a higher max_df."
        raise ValueError(msg)
    train_counts = train_counts[:, mask].astype(np.float64)
    validation_counts = counts.of(validation)[:, columns[mask]].astype(np.float64)
    transformer = TfidfTransformer(sublinear_tf=params.sublinear_tf)
    return transformer.fit_transform(train_counts), transformer.transform(
        validation_counts
    )


def _diagnosis_scores(
    model: BaseEstimator,
    train: ndarray | csr_matrix,
    labels: list[bool],
    validation: ndarray | csr_matrix,
) -> tuple[tuple[float, ...], bool]:
    # Jedno vlákno BLAS zrychlí malé fity a odstraní vliv počtu procesů na skóre.
    with threadpool_limits(limits=1, user_api="blas"):
        converged = fit_converged(
            partial(cast("_TrainableModel", model).fit, train, labels)
        )
        scores = tuple(float(value) for value in positive_scores(model, validation))
        return scores, converged


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
    train: Sequence[str],
    validation: Sequence[str],
) -> tuple[ndarray, ndarray]:
    """Převede hotové tokenové dokumenty na pole řetězců.

    Returns:
        Pole řetězců pro trénink a validaci ve vstupním pořadí.

    Raises:
        InvalidInputError: Trénovací texty neobsahují token.
    """
    require_tokens(document.split() for document in train)
    # Pevná šířka <U zarovná zprávy na nejdelší: 44 tis. krát 60 tis. znaků je ~9,8 GiB.
    return (
        np.array(train, dtype=object),
        np.array(validation, dtype=object),
    )


def joined_tokens(normalization: TextNormalization | None) -> Callable[[str], str]:
    """Sestaví funkci spojující tokeny společného analyzátoru.

    Returns:
        Předzpracování textu na tokeny oddělené mezerou.
    """
    analyzer = tfidf_vectorizer(normalization).build_analyzer()
    return lambda text: " ".join(analyzer(text))


def _standardized(train: ndarray, validation: ndarray) -> tuple[ndarray, ndarray]:
    scaler = StandardScaler()
    return scaler.fit_transform(train), scaler.transform(validation)


def word2vec_features(
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
    documents = [text.split() for text in train]
    require_tokens(documents)
    vectors = trained_vectors(options, documents)
    return _standardized(
        _mean_vectors(vectors, documents),
        _mean_vectors(vectors, [text.split() for text in validation]),
    )


def _mean_vectors(vectors: KeyedVectors, documents: Sequence[list[str]]) -> ndarray:
    # Výchozí pre_normalize=True by průměrovalo jednotkové vektory místo vektorů slov.
    return np.array([
        vectors.get_mean_vector(tokens, pre_normalize=False)
        if tokens
        else np.zeros(vectors.vector_size, dtype=np.float32)
        for tokens in documents
    ])


class _TrainableModel(Protocol):
    def fit(self, features: ndarray | csr_matrix, labels: list[bool]) -> object: ...
