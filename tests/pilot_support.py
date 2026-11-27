"""Syntetické zprávy, porty a ruční skóre pro pilot transformeru."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.input_note import InputNote
from mediparse.domain.language import Language
from mediparse.domain.split import SharedSplit, Split, SplitConfig
from mediparse.domain.transformer_pilot import (
    PilotEpoch,
    PilotRun,
    PilotTraining,
    TransformerPilotConfig,
)
from mediparse.domain.tuning import ValidationScores
from tests.hf_encoder_support import SPEC

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


def corpus() -> tuple[InputNote, ...]:
    """Sestaví 1200 vyvážených pacientů s jednou zprávou.

    Returns:
        Syntetická čísla pacientů jako texty, všech pět labelů nebo žádný.
    """
    return tuple(
        InputNote(
            f"{i}-DS-0",
            i,
            Language.EN,
            str(i),
            frozenset(Diagnosis) if i % 2 else frozenset(),
        )
        for i in range(90000001, 90001201)
    )


def pilot_config(*, epochs: int = 3, parts: int = 1) -> TransformerPilotConfig:
    """Sestaví dvě levné varianty s celým a polovičním vzorkem.

    Returns:
        Validní config s CPU encoderem.
    """
    return TransformerPilotConfig(
        seed=0,
        subsample_parts=parts,
        folds=2,
        stopping_parts=2,
        strip_diagnostic_sections=False,
        training=PilotTraining(
            epochs=epochs,
            batch_tokens=72,
            lr=0.001,
            weight_decay=0.01,
            warmup_ratio=0.1,
        ),
        runs=tuple(
            PilotRun(
                encoder=SPEC.model_copy(update={"name": f"run{i}"}),
                aggregation="max",
                freeze_embeddings=True,
                fraction=fraction,
            )
            for i, fraction in enumerate((1.0, 0.5))
        ),
    )


@dataclass(frozen=True)
class Table:
    """Paměťová tabulka syntetických zpráv."""

    notes: tuple[InputNote, ...] = field(default_factory=corpus)

    def read(self) -> tuple[InputNote, ...]:
        """Vrátí všechny zprávy.

        Returns:
            Původní pořadí tabulky.
        """
        return self.notes

    def subjects(self) -> frozenset[int]:
        """Vrátí pacienty tabulky.

        Returns:
            Jedinečná syntetická čísla pacientů.
        """
        return frozenset(n.subject_id for n in self.notes)

    def read_patients(self, subjects: frozenset[int]) -> tuple[InputNote, ...]:
        """Vybere celé pacienty.

        Returns:
            Zprávy ve vstupním pořadí.
        """
        return tuple(n for n in self.notes if n.subject_id in subjects)


@dataclass(frozen=True)
class Source:
    """Pevný split se všemi syntetickými pacienty ve vývoji."""

    @staticmethod
    def read() -> SharedSplit:
        """Vrátí split a jeho parametry.

        Returns:
            Žádná zpráva korpusu neleží v held-out foldech.
        """
        return SharedSplit(
            SplitConfig(seed=23, folds=2),
            Split((frozenset({1}), frozenset({2})), development=Table().subjects()),
        )


@dataclass
class Trainer:
    """Zachytí rozdělení a vydá skóre s ručním optimem F1 0,8."""

    observed: list[tuple[Sequence[InputNote], Sequence[InputNote], Sequence[str]]] = (
        field(default_factory=list)
    )

    def train(
        self,
        run: PilotRun,
        training: PilotTraining,
        fit: Sequence[InputNote],
        stop: Sequence[InputNote],
        validation: Sequence[str],
    ) -> Iterator[PilotEpoch]:
        """Nepřebírá labely validace; zná labely svého syntetického korpusu.

        Yields:
            Předepsaný počet deterministických epoch.
        """
        del run
        self.observed.append((fit, stop, validation))
        labels = {n.text: bool(n.labels) for n in corpus()}
        counts: Counter[bool] = Counter()
        values = []
        for text in validation:
            positive = labels[text]
            values.append(
                ((0.9, 0.3) if positive else (0.8, 0.1))[counts[positive] % 2]
            )
            counts[positive] += 1
        for epoch in range(1, training.epochs + 1):
            yield PilotEpoch(
                epoch,
                2.0,
                0.5,
                0,
                1.0 / epoch,
                ValidationScores(
                    {d: tuple(values) for d in Diagnosis}, converged=True, n_features=16
                ),
            )
