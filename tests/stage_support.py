"""Malé prostory stupně pro syntetická ověření."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.stage_tuning import StageTrialRecord, StageTuningConfig
from mediparse.domain.tuning import Trial, ValidationScores

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mediparse.domain.split import Holdout
    from mediparse.domain.stage_tuning import StageFoldResult
    from mediparse.domain.tuning import ParamValue


def stage_config(**changes: object) -> StageTuningConfig:
    """Sestaví dvě reprezentace a dvě rodiny.

    Returns:
        Šest pokusů s jedním náhodným startem.
    """
    representations = []
    for name in ("r1", "r2"):
        default = {
            "ngram_min": 1,
            "ngram_max": 2,
            "min_df": 1,
            "max_df": 1.0,
            "sublinear_tf": True,
        }
        params: list[dict[str, object]] = [
            {"name": key, "kind": "categorical", "choices": [value]}
            for key, value in default.items()
        ]
        params[3] = {
            "name": "max_df",
            "kind": "float",
            "low": 0.5,
            "high": 1.0,
            "log": False,
        }
        if name == "r1":
            params[2] = {
                "name": "min_df",
                "kind": "int",
                "low": 1,
                "high": 2,
                "log": True,
            }
        representations.append({
            "name": name,
            "text_preprocessing": "dictionary" if name == "r1" else "none",
            "analyzer": "word",
            "space": {"params": params},
            "default": default,
        })
    return StageTuningConfig.model_validate_json(
        json.dumps(
            {
                "seed": 7,
                "subsample_parts": 1,
                "folds": 2,
                "trials": 6,
                "random_startup_trials": 1,
                "strip_diagnostic_sections": False,
                "representations": representations,
                "families": [
                    {
                        "family": family,
                        "space": {
                            "params": [
                                {
                                    "kind": "float",
                                    "name": param,
                                    "low": 0.001,
                                    "high": 1000.0,
                                    "log": True,
                                }
                            ]
                        },
                        "default": {param: 1.0},
                    }
                    for family, param in (("logreg", "C"), ("complement_nb", "alpha"))
                ],
            }
            | changes
        )
    )


@dataclass
class StepClock:
    """Hodiny s konstantním krokem pro bajtově shodné výstupy."""

    value: float = 0.0

    def now(self) -> float:
        """Posune čas o sekundu.

        Returns:
            Nový stav hodin.
        """
        self.value += 1.0
        return self.value


def token_stage_config() -> StageTuningConfig:
    """Sestaví malý prostor fastTextu bez parametrů reprezentace.

    Returns:
        Tokenová větev s pevným levným fitem.
    """
    default = {"lr": 0.5, "epoch": 1, "word_ngrams": 2, "dim": 2}
    return stage_config(
        representations=[{"name": "tok", "text_preprocessing": "dictionary"}],
        families=[
            {
                "family": "fasttext",
                "space": {
                    "params": [
                        {"kind": "categorical", "name": name, "choices": [value]}
                        for name, value in default.items()
                    ]
                },
                "default": default,
            }
        ],
    )


def word2vec_stage_config() -> StageTuningConfig:
    """Sestaví levnou větev word2vec pro syntetická ověření.

    Returns:
        Word2vec s jednovláknovým tréninkem a logistickou regresí.
    """
    default = {
        "vector_size": 7,
        "window": 2,
        "min_count": 2,
        "sg": 0,
        "epochs": 5,
        "negative": 5,
        "sample": 0.001,
    }
    params: list[dict[str, object]] = [
        {"kind": "categorical", "name": name, "choices": [value]}
        for name, value in default.items()
        if name != "sample"
    ]
    params.append({
        "kind": "float",
        "name": "sample",
        "low": 0.001,
        "high": 0.002,
        "log": False,
    })
    return stage_config(
        representations=[
            {
                "name": "w2v",
                "text_preprocessing": "dictionary",
                "workers": 1,
                "space": {"params": params},
                "default": default,
            }
        ],
        families=stage_config().model_dump(mode="json")["families"][:1],
    )


def network_stage_config() -> StageTuningConfig:
    """Sestaví levnou CPU síť s pěti výstupy.

    Returns:
        Tokenová větev s odloženou třetinou pacientů.
    """
    default = {"lr": 0.01, "weight_decay": 0.0001, "dropout": 0.0, "filter_maps": 4}
    return stage_config(
        representations=[{"name": "tok", "text_preprocessing": "dictionary"}],
        families=[
            {
                "family": "multirescnn",
                "space": {
                    "params": [
                        {"kind": "categorical", "name": name, "choices": [value]}
                        for name, value in default.items()
                    ]
                },
                "default": default,
            }
        ],
        network={
            "device": "cpu",
            "epochs": 2,
            "batch_size": 4,
            "max_words": 50,
            "warmup_steps": 0,
            "embedding_dim": 8,
            "min_count": 1,
            "stopping_parts": 3,
        },
    )


@dataclass
class MemoryProgress:
    """Paměťový checkpoint pro ověření protokolu."""

    records: list[StageTrialRecord] = field(default_factory=list)
    results: list[StageFoldResult] = field(default_factory=list)
    finished: bool = False

    def completed(self) -> tuple[StageTrialRecord, ...]:
        """Vrátí dosud uložené pokusy.

        Returns:
            Záznamy v pořadí zápisu.
        """
        return tuple(self.records)

    def write(
        self, record: StageTrialRecord, results: Sequence[StageFoldResult]
    ) -> None:
        """Uloží pokus a jeho predikce."""
        self.results.extend(results)
        self.records.append(record)

    def finish(self) -> None:
        """Označí úspěšné dokončení."""
        self.finished = True


@dataclass
class FixedSearch:
    """Paměťová studie s předem určenými návrhy."""

    asked: int = 0
    enqueued: list[tuple[int, Mapping[str, ParamValue]]] = field(default_factory=list)
    restored: list[tuple[Trial, float]] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def restore(self, trials: Sequence[tuple[Trial, float]]) -> None:
        """Obnoví historii pro ověření protokolu."""
        self.restored.extend(trials)
        self.asked = len(trials)

    def enqueue(self, params: Mapping[str, ParamValue]) -> None:
        """Připraví bod pro další návrh."""
        self.enqueued.append((self.asked, params))

    def ask(self) -> Trial:
        """Navrhne připravený bod.

        Returns:
            Nový pokus.
        """
        trial = Trial(self.asked, self.enqueued[-1][1])
        self.asked += 1
        return trial

    def complete(self, trial: Trial, value: float) -> None:
        """Uloží cílovou hodnotu dokončeného pokusu."""
        del trial
        self.values.append(value)


@dataclass
class CapturingScorer:
    """Deterministická skóre a volitelné přerušení."""

    fail_at: int | None = None
    observed: list[Holdout] = field(default_factory=list)

    def score_folds(
        self, params: Mapping[str, ParamValue], folds: Sequence[Holdout]
    ) -> tuple[ValidationScores, ...]:
        """Zachytí každý fold.

        Returns:
            Skóre ve vstupním pořadí foldů.
        """
        return tuple(self.score(params, fold) for fold in folds)

    def score(
        self, params: Mapping[str, ParamValue], fold: Holdout
    ) -> ValidationScores:
        """Vydá deterministická skóre nebo přeruší zvolený pokus.

        Returns:
            Skóre podle labelů první validace, jinak nulová.

        Raises:
            RuntimeError: Nastal požadovaný bod syntetického přerušení.
        """
        del params
        if len(self.observed) == self.fail_at:
            msg = "syntetické přerušení"
            raise RuntimeError(msg)
        self.observed.append(fold)
        return ValidationScores(
            {
                d: tuple(
                    float(n.has_label(d)) if fold.fold == 0 else 0.0 for n in fold.test
                )
                for d in Diagnosis
            },
            converged=True,
            n_features=10 + fold.fold,
        )
