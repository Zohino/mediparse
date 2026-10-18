"""Use case evaluace: skupiny podle řádku matice a diagnózy, kontrola vstupu před výpočtem."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from mediparse.application.model_evaluation import ModelEvaluation
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.evaluation import BinaryMetrics, DiagnosisMetrics, Prediction
from mediparse.domain.input_note import InputNote
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.language import Language
from mediparse.domain.note import note_id
from mediparse.domain.split import SharedSplit, Split, SplitConfig
from tests.model_support import prediction

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class _FakePredictions:
    items: Sequence[Prediction]

    def read(self) -> tuple[Prediction, ...]:
        return tuple(self.items)


@dataclass
class _FakeScorer:
    calls: list[tuple[tuple[bool, ...], tuple[bool, ...], tuple[float, ...]]] = field(
        default_factory=list
    )

    def score(
        self, truth: Sequence[bool], predicted: Sequence[bool], scores: Sequence[float]
    ) -> BinaryMetrics:
        self.calls.append((tuple(truth), tuple(predicted), tuple(scores)))
        return BinaryMetrics(*(float(len(self.calls)),) * 6)


@dataclass
class _FakeSink:
    written: list[Sequence[DiagnosisMetrics]] = field(default_factory=list)

    def write(self, metrics: Sequence[DiagnosisMetrics]) -> None:
        self.written.append(tuple(metrics))


def _notes(*subjects: int) -> tuple[InputNote, ...]:
    return tuple(
        InputNote(note_id(subject, 1), subject, Language.EN, "text", frozenset())
        for subject in subjects
    )


NOTES = _notes(1, 2, 3, 4)
SPLIT = Split((frozenset({1, 2, 3}), frozenset({4})))
HALVES = Split((frozenset({1, 2}), frozenset({3, 4})))


@dataclass
class _FakeTable:
    notes: Sequence[InputNote]

    def read(self) -> tuple[InputNote, ...]:
        return tuple(self.notes)


@dataclass
class _FakeSplitSource:
    split: Split

    def read(self) -> SharedSplit:
        return SharedSplit(SplitConfig(seed=0, folds=len(self.split.folds)), self.split)


def _use_case(
    items: Sequence[Prediction], split: Split = SPLIT
) -> tuple[ModelEvaluation, _FakeScorer, _FakeSink]:
    scorer, sink = _FakeScorer(), _FakeSink()
    use_case = ModelEvaluation(
        _FakePredictions(items),
        _FakeTable(NOTES),
        _FakeSplitSource(split),
        scorer,
        sink,
    )
    return use_case, scorer, sink


def _group(
    row_id: str,
    diagnosis: Diagnosis,
    outcomes: Sequence[tuple[bool, float]],
    fold: int = 0,
    subjects: Sequence[int] = (1, 2, 3),
) -> tuple[Prediction, ...]:
    return tuple(
        prediction(
            note_id(subject, 1),
            truth=truth,
            score=score,
            diagnosis=diagnosis,
            row_id=row_id,
            fold=fold,
        )
        for subject, (truth, score) in zip(subjects, outcomes, strict=True)
    )


MIXED = (
    *_group("b", Diagnosis.DIABETES, ((True, 1.0), (False, 0.5), (False, -3.0))),
    *_group("a", Diagnosis.CKD, ((False, -1.0), (True, 3.0), (True, -0.5))),
    *_group("a", Diagnosis.DIABETES, ((False, -2.0), (True, 2.0), (True, 1.0))),
)


def test_groups_are_sorted_with_sizes() -> None:
    """Dvě diagnózy a dva row_id dají seřazené řádky se správnými velikostmi."""
    use_case, _, sink = _use_case(MIXED)

    result = use_case.run()

    assert [(item.row_id, item.diagnosis) for item in result] == [
        ("a", Diagnosis.CKD),
        ("a", Diagnosis.DIABETES),
        ("b", Diagnosis.DIABETES),
    ]
    assert [(item.test_notes, item.test_positives) for item in result] == [
        (3, 2),
        (3, 2),
        (3, 1),
    ]
    assert sink.written == [result]


def test_scorer_gets_only_its_group_in_input_order() -> None:
    """Scorer dostane pravdu, predikce a skóre jen své skupiny ve vstupním pořadí."""
    use_case, scorer, _ = _use_case(MIXED)

    use_case.run()

    expected = [
        ((False, True, True), (False, True, False), (-1.0, 3.0, -0.5)),
        ((False, True, True), (False, True, True), (-2.0, 2.0, 1.0)),
        ((True, False, False), (True, True, False), (1.0, 0.5, -3.0)),
    ]
    assert Counter(scorer.calls) == Counter(expected)


def test_empty_input_is_rejected_without_writing() -> None:
    """Prázdné predikce skončí InvalidInputError a nic se nezapíše."""
    use_case, scorer, sink = _use_case(())

    with pytest.raises(InvalidInputError, match="prázdn"):
        use_case.run()

    assert not scorer.calls
    assert not sink.written


def test_single_class_is_rejected_before_scoring() -> None:
    """Skupina s jedinou třídou pravdy skončí InvalidInputError před výpočtem."""
    items = (
        *MIXED,
        *_group("c", Diagnosis.PNEUMONIA, ((True, 1.0), (True, 2.0), (True, 3.0))),
    )
    use_case, scorer, sink = _use_case(items)

    with pytest.raises(InvalidInputError, match="pneumonia"):
        use_case.run()

    assert not scorer.calls
    assert not sink.written


def test_duplicate_key_is_rejected_before_scoring() -> None:
    """Opakovaný klíč predikce skončí InvalidInputError před výpočtem."""
    use_case, scorer, sink = _use_case((*MIXED, MIXED[0]))

    with pytest.raises(InvalidInputError, match="Opakovaná predikce"):
        use_case.run()

    assert not scorer.calls
    assert not sink.written


def test_folds_of_one_row_and_diagnosis_are_pooled() -> None:
    """Predikce dvou foldů dají jednu metriku přes všechny testovací zprávy.

    Fold 0 má jen pozitivní a fold 1 jen negativní pravdu, takže seskupení po foldech
    by skončilo na kontrole obou tříd.
    """
    items = (
        *_group("row", Diagnosis.DIABETES, ((True, 1.0), (True, -1.0)), 0, (1, 2)),
        *_group("row", Diagnosis.DIABETES, ((False, -2.0), (False, 0.5)), 1, (3, 4)),
    )
    use_case, scorer, _ = _use_case(items, HALVES)

    result = use_case.run()

    assert [(item.test_notes, item.test_positives) for item in result] == [(4, 2)]
    assert scorer.calls == [
        ((True, True, False, False), (True, False, False, True), (1.0, -1.0, -2.0, 0.5))
    ]


def test_incomplete_fold_is_rejected_before_scoring() -> None:
    """Predikce bez jedné testovací zprávy foldu skončí chybou a nic se nezapíše."""
    use_case, scorer, sink = _use_case(MIXED[1:])

    with pytest.raises(InvalidInputError, match="chybí 1, navíc 0"):
        use_case.run()

    assert not scorer.calls
    assert not sink.written


def test_duplicate_is_reported_as_duplicate_not_as_incompleteness() -> None:
    """Skupina s duplicitou a chybějící zprávou se ohlásí jako duplicita.

    Množina duplicitu sloučí, takže kdyby kontrola úplnosti běžela první, hlásila by
    jen chybějící zprávu.
    """
    use_case, scorer, sink = _use_case(MIXED[1:] + MIXED[1:2])

    with pytest.raises(InvalidInputError, match="Opakovaná predikce"):
        use_case.run()

    assert not scorer.calls
    assert not sink.written
