"""Skórování větví klasického stupně s fitováním reprezentace v každém pokusu."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

from sklearn.decomposition import TruncatedSVD

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.screening import ClassifierFamily
from mediparse.domain.tuning import ValidationScores
from mediparse.infrastructure.sklearn_classifier import (
    TfidfParameters,
    ensure_tokens,
    fit_converged,
    tfidf_from,
)
from mediparse.infrastructure.sklearn_screening import (
    classifier_estimator,
    positive_scores,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from mediparse.domain.clinical_text import TextNormalization
    from mediparse.domain.seed import Seed
    from mediparse.domain.split import Holdout
    from mediparse.domain.stage_tuning import StageTuningConfig
    from mediparse.domain.tuning import ParamValue


@dataclass(frozen=True)
class SklearnStageScorer:
    """Fitne reprezentaci jednou pro pět diagnóz pokusu a foldu."""

    config: StageTuningConfig
    normalization: TextNormalization | None
    seed: Seed
    scratch: Path

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
        tfidf = TfidfParameters.model_validate(rep)
        lsa = branch.family.family is ClassifierFamily.LIGHTGBM_LSA
        # SVD alokuje husté matice úměrné šířce; strop drží paměť, jiné rodiny mají celý slovník.
        vectorizer = tfidf_from(
            tfidf,
            branch.representation.analyzer,
            self.normalization,
            max_features=self.config.lsa_max_features if lsa else None,
        )
        texts = [note.text for note in fold.train]
        ensure_tokens(vectorizer, texts)
        train = vectorizer.fit_transform(texts)
        validation = vectorizer.transform([note.text for note in fold.test])
        features = train.shape[1]
        if lsa:
            reducer = TruncatedSVD(
                n_components=self.config.lsa_components, random_state=self.seed
            )
            train = reducer.fit_transform(train)
            validation = reducer.transform(validation)
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
