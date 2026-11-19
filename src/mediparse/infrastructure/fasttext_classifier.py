"""Sklearn estimátor fastTextu nad slovními tokeny s deterministickým tréninkem."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Final, Self

import fasttext
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy import ndarray

# Dvojtečka nemůže být součástí tokenu analyzátoru, tedy ani falešného labelu.
LABEL_PREFIX: Final = "label:"


class FastTextClassifier(ClassifierMixin, BaseEstimator):
    """Binární fastText s výchozími hyperparametry a explicitním scratch adresářem."""

    def __init__(self, scratch: Path, random_state: int = 0) -> None:
        """Uloží parametry bez naučeného stavu pro sklearn clone."""
        self.scratch = scratch
        self.random_state = random_state

    def fit(self, documents: ndarray, labels: Sequence[bool]) -> Self:
        """Natrénuje fastText a odstraní soubor s tokeny i při chybě.

        Returns:
            Tentýž estimátor s classes_ a model_.

        Raises:
            ValueError: Trénink neobsahuje právě dvě třídy.
        """
        self.classes_ = np.unique(labels)
        if len(self.classes_) != 2:  # ruff: ignore[magic-value-comparison]
            msg = "fastText vyžaduje právě dvě třídy."
            raise ValueError(msg)
        self.scratch.mkdir(parents=True, exist_ok=True)
        # Tokeny zpráv MIMIC patří pod chráněný scratch; systémové /tmp je mimo ochranu DUA.
        with TemporaryDirectory(dir=self.scratch) as directory:
            path = Path(directory) / "train.txt"
            with path.open("w", encoding="utf-8") as output:
                for document, label in zip(documents, labels, strict=True):
                    output.write(f"{LABEL_PREFIX}{label} {document}\n")
            self.model_ = fasttext.train_supervised(
                input=str(path),
                # Více vláken mění pravděpodobnosti i při stejném seedu.
                thread=1,
                seed=self.random_state,
                verbose=0,
                label=LABEL_PREFIX,
            )
        return self

    # Log-odds brání odmítnutí téměř konstantních P ≈ 0,5 v TunedThresholdClassifierCV.
    def decision_function(self, documents: ndarray) -> ndarray:
        """Vrátí log-odds pozitivní vůči negativní třídě.

        Returns:
            Jedno skóre log p(True) - log p(False) na dokument.
        """
        check_is_fitted(self, "model_")
        # Predikce jediného řetězce v 0.9.2 selže s numpy 2 na copy=False.
        labels, probabilities = self.model_.predict(list(documents), k=-1)
        scores = []
        for names, values in zip(labels, probabilities, strict=True):
            by_label = dict(zip(names, values, strict=True))
            scores.append(
                np.log(by_label[f"{LABEL_PREFIX}{self.classes_[1]}"])
                - np.log(by_label[f"{LABEL_PREFIX}{self.classes_[0]}"])
            )
        return np.array(scores)
