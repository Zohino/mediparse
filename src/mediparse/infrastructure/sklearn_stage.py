"""Skórování větví stupně s fitováním reprezentace v každém pokusu."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, assert_never

from sklearn.decomposition import TruncatedSVD

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.representation import (
    ClassifierFamily,
    EncoderSpec,
    TfidfRepresentation,
    TokensSpec,
)
from mediparse.domain.tuning import ValidationScores
from mediparse.infrastructure.sklearn_classifier import (
    TfidfParameters,
    ensure_tokens,
    fit_converged,
    tfidf_from,
)
from mediparse.infrastructure.sklearn_screening import (
    EncoderFeaturizer,
    TokenFeaturizer,
    classifier_estimator,
    positive_scores,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

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
        train, validation, features = self._features(branch, rep, fold)
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
        return ValidationScores(scores, converged, features)

    def _features(
        self, branch: Branch, rep_params: Mapping[str, ParamValue], fold: Holdout
    ) -> tuple[ndarray | csr_matrix, ndarray | csr_matrix, int]:
        texts = [note.text for note in fold.train]
        test_texts = [note.text for note in fold.test]
        match branch.representation:
            case EncoderSpec() as spec:
                train, validation = EncoderFeaturizer(self.embeddings).fit_transform(
                    spec, texts, test_texts
                )
                return train.matrix, validation.matrix, train.matrix.shape[1]
            case TokensSpec() as spec:
                train, validation = TokenFeaturizer(self.normalization).fit_transform(
                    spec, texts, test_texts
                )
                features = len({
                    token for document in train.matrix for token in document.split()
                })
                return train.matrix, validation.matrix, features
            case TfidfRepresentation():
                tfidf = TfidfParameters.model_validate(rep_params)
                lsa = (
                    self.config.lsa
                    if branch.family.family is ClassifierFamily.LIGHTGBM_LSA
                    else None
                )
                # SVD alokuje husté matice úměrné šířce; strop drží paměť, jiné rodiny mají celý slovník.
                vectorizer = tfidf_from(
                    tfidf,
                    branch.representation.analyzer,
                    self.normalization,
                    max_features=lsa.max_features if lsa else None,
                )
                ensure_tokens(vectorizer, texts)
                train = vectorizer.fit_transform(texts)
                validation = vectorizer.transform(test_texts)
                features = train.shape[1]
                if lsa is not None:
                    reducer = TruncatedSVD(
                        n_components=lsa.components, random_state=self.seed
                    )
                    train = reducer.fit_transform(train)
                    validation = reducer.transform(validation)
                return train, validation, features
            case _:
                assert_never(branch.representation)
