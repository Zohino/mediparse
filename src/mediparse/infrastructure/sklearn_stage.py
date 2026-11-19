"""Skórování větví klasického stupně s fitováním reprezentace v každém pokusu."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, ConfigDict, Field, PositiveInt
from sklearn.decomposition import TruncatedSVD

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.screening import ClassifierFamily
from mediparse.domain.tuning import ValidationScores
from mediparse.infrastructure.sklearn_classifier import (
    ensure_tokens,
    fit_converged,
    tfidf_vectorizer,
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
    from mediparse.domain.stage_tuning import Branch
    from mediparse.domain.tuning import ParamValue


class TfidfParameters(BaseModel):
    """Typované parametry reprezentace na hranici sklearn."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    ngram_min: PositiveInt
    ngram_max: PositiveInt
    min_df: PositiveInt
    max_df: Annotated[float, Field(gt=0, le=1)]
    sublinear_tf: bool


@dataclass(frozen=True)
class SklearnStageScorer:
    """Fitne reprezentaci jednou pro pět diagnóz pokusu a foldu."""

    branches: Mapping[str, Branch]
    normalization: TextNormalization | None
    lsa_components: int
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
        branch = self.branches[str(params["branch"])]
        rep, family = branch.params_of(params)
        tfidf = TfidfParameters.model_validate(rep)
        vectorizer = tfidf_vectorizer(
            self.normalization,
            analyzer=branch.representation.analyzer,
            ngram_range=(tfidf.ngram_min, tfidf.ngram_max),
            min_df=tfidf.min_df,
            max_df=tfidf.max_df,
            sublinear_tf=tfidf.sublinear_tf,
        )
        texts = [note.text for note in fold.train]
        ensure_tokens(vectorizer, texts)
        train = vectorizer.fit_transform(texts)
        validation = vectorizer.transform([note.text for note in fold.test])
        if branch.family.family is ClassifierFamily.LIGHTGBM_LSA:
            lsa = TruncatedSVD(n_components=self.lsa_components, random_state=self.seed)
            train = lsa.fit_transform(train)
            validation = lsa.transform(validation)
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
        return ValidationScores(scores, converged)
