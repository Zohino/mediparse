"""Sdílené porty experimentu ML: vstup, dělení pacientů, klasifikace, predikce a metriky."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mediparse.domain.evaluation import BinaryMetrics, Classification, Prediction
    from mediparse.domain.identity import SubjectId
    from mediparse.domain.input_note import InputNote
    from mediparse.domain.seed import Seed
    from mediparse.domain.split import Holdout, SharedSplit
    from mediparse.domain.stage_tuning import (
        StageFoldResult,
        StageTrialRecord,
        StageTuningConfig,
    )
    from mediparse.domain.tuning import ParamValue, Trial, ValidationScores


class InputTable(Protocol):
    """Vstupní tabulka zpráv s labely."""

    def read(self) -> tuple[InputNote, ...]:
        """Přečte řádky v pořadí tabulky."""


class PatientNoteTable(InputTable, Protocol):
    """Tabulka s levným přehledem pacientů a výběrovým čtením zpráv."""

    def subjects(self) -> frozenset[SubjectId]:
        """Vrátí všechny pacienty tabulky bez načítání textů."""

    def read_patients(self, subjects: frozenset[SubjectId]) -> tuple[InputNote, ...]:
        """Přečte zprávy daných pacientů v pořadí tabulky."""


class SplitSource(Protocol):
    """Zdroj hotového splitu, který spočetl krok před tréninkem a evaluací."""

    def read(self) -> SharedSplit:
        """Vrátí split s jeho parametry."""


class TextClassifier(Protocol):
    """Natrénovaný binární klasifikátor textu."""

    def classify(self, texts: Sequence[str]) -> tuple[Classification, ...]:
        """Určí label a rozhodovací skóre pro každý text."""


class PredictionSource(Protocol):
    """Zdroj predikcí testovacích zpráv."""

    def read(self) -> tuple[Prediction, ...]:
        """Přečte predikce v pořadí zdroje."""


class PredictionSink(Protocol):
    """Výstup predikcí testovacích zpráv."""

    def write(self, predictions: Sequence[Prediction]) -> None:
        """Zapíše predikce."""


class PatientPartitioner(Protocol):
    """Stratifikované rozdělení pacientů do disjunktních částí."""

    def partition(
        self,
        subjects: Sequence[SubjectId],
        strata: Sequence[int],
        parts: int,
        seed: Seed,
    ) -> tuple[frozenset[SubjectId], ...]:
        """Rozdělí pacienty do daného počtu částí podle stratifikace zpráv."""


class Scorer(Protocol):
    """Výpočet metrik binární klasifikace."""

    def score(
        self, truth: Sequence[bool], predicted: Sequence[bool], scores: Sequence[float]
    ) -> BinaryMetrics:
        """Spočte metriky predikcí a skóre proti pravdě."""


class Clock(Protocol):
    """Monotonní hodiny pro měření rozdílu časů v sekundách."""

    def now(self) -> float:
        """Vrátí čas v sekundách s libovolným pevným počátkem."""


class StageSearch(Protocol):
    """Návrhy větví s výchozími body a obnovou historie."""

    def ask(self) -> Trial:
        """Navrhne parametry nového pokusu."""

    def complete(self, trial: Trial, value: float) -> None:
        """Uzavře pokus s cílovou hodnotou."""

    def enqueue(self, params: Mapping[str, ParamValue]) -> None:
        """Vloží parametry následujícího pokusu."""

    def restore(self, trials: Sequence[tuple[Trial, float]]) -> None:
        """Obnoví dokončené pokusy v pořadí čísel."""


class StageProgress(Protocol):
    """Trvalé pokusy a jejich validační predikce."""

    def completed(self) -> tuple[StageTrialRecord, ...]:
        """Ověří klíč a vrátí dokončené pokusy."""

    def write(
        self, record: StageTrialRecord, results: Sequence[StageFoldResult]
    ) -> None:
        """Zapíše jeden pokus se všemi foldy a diagnózami."""

    def finish(self) -> None:
        """Složí výstupy a uklidí checkpoint."""


class TrialScorer(Protocol):
    """Skóre validačních zpráv jednoho pokusu."""

    def score_folds(
        self, params: Mapping[str, ParamValue], folds: Sequence[Holdout]
    ) -> tuple[ValidationScores, ...]:
        """Fitne modely a vrátí skóre všech foldů v pořadí vstupu."""


class StageSearches(Protocol):
    """Továrna studií vznikajících po kontrole kandidátů."""

    def create(self, config: StageTuningConfig) -> StageSearch:
        """Vytvoří prázdnou studii zadané větve."""
