"""Validační skóre pěti lineárních SVM nad společnými příznaky vnitřního foldu."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.tuning import ValidationScores
from mediparse.infrastructure.sklearn_classifier import (
    SvmParameters,
    ensure_tokens,
    fit_converged,
    svm_estimator,
    svm_vectorizer,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mediparse.domain.clinical_text import TextNormalization
    from mediparse.domain.seed import Seed
    from mediparse.domain.split import Holdout
    from mediparse.domain.tuning import ParamValue


@dataclass(frozen=True)
class SklearnLinearSvmScorer:
    """Fitne vektorizér jednou pro pět diagnóz každého vnitřního foldu pokusu."""

    normalization: TextNormalization | None
    seed: Seed

    @property
    def parameters(self) -> frozenset[str]:
        """Jména podporovaných parametrů včetně pevných voleb vektorizéru."""
        return frozenset(SvmParameters.model_fields)

    def score(
        self, params: Mapping[str, ParamValue], fold: Holdout
    ) -> ValidationScores:
        """Fitne pouze trénink a vrátí decision_function validace po diagnózách.

        Returns:
            Skóre validačních zpráv a konvergence všech pěti fitů.

        Raises:
            InvalidInputError: Parametry neodpovídají SVM nebo nevznikne slovník.
        """
        try:
            return self._score(SvmParameters.model_validate(params), fold)
        except ValueError as error:
            msg = f"Parametry nebo trénovací slovník ladění SVM nejsou platné: {error}"
            raise InvalidInputError(msg) from error

    def _score(self, params: SvmParameters, fold: Holdout) -> ValidationScores:
        # Parametry TF-IDF se ladí, proto nový fit patří do každého pokusu a foldu.
        vectorizer = svm_vectorizer(params, self.normalization)
        texts = [note.text for note in fold.train]
        ensure_tokens(vectorizer, texts)
        train = vectorizer.fit_transform(texts)
        validation = vectorizer.transform([note.text for note in fold.test])
        scores = {}
        converged = True
        for diagnosis in Diagnosis:
            model = svm_estimator(params, self.seed)
            labels = [note.has_label(diagnosis) for note in fold.train]
            converged &= fit_converged(partial(model.fit, train, labels))
            scores[diagnosis] = tuple(
                float(value) for value in model.decision_function(validation)
            )
        return ValidationScores(scores, converged)
