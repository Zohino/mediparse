"""Use case evaluace: skupiny podle řádku matice a diagnózy, kontrola vstupu před výpočtem."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from mediparse.application.model_evaluation import ModelEvaluation
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.evaluation import BinaryMetrics, DiagnosisMetrics, Prediction
from mediparse.domain.inputs import InvalidInputError
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


def _use_case(
    items: Sequence[Prediction],
) -> tuple[ModelEvaluation, _FakeScorer, _FakeSink]:
    scorer, sink = _FakeScorer(), _FakeSink()
    return ModelEvaluation(_FakePredictions(items), scorer, sink), scorer, sink


MIXED = (
    prediction(
        "1-DS-1", truth=True, score=1.0, diagnosis=Diagnosis.DIABETES, row_id="b"
    ),
    prediction("2-DS-1", truth=False, score=-1.0, diagnosis=Diagnosis.CKD, row_id="a"),
    prediction(
        "3-DS-1", truth=False, score=-2.0, diagnosis=Diagnosis.DIABETES, row_id="a"
    ),
    prediction(
        "4-DS-1", truth=False, score=0.5, diagnosis=Diagnosis.DIABETES, row_id="b"
    ),
    prediction("5-DS-1", truth=True, score=3.0, diagnosis=Diagnosis.CKD, row_id="a"),
    prediction(
        "6-DS-1", truth=True, score=2.0, diagnosis=Diagnosis.DIABETES, row_id="a"
    ),
    prediction("7-DS-1", truth=True, score=-0.5, diagnosis=Diagnosis.CKD, row_id="a"),
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
        (2, 1),
        (2, 1),
    ]
    assert sink.written == [result]


def test_scorer_gets_only_its_group_in_input_order() -> None:
    """Scorer dostane pravdu, predikce a skóre jen své skupiny ve vstupním pořadí."""
    use_case, scorer, _ = _use_case(MIXED)

    use_case.run()

    expected = [
        ((False, True, True), (False, True, False), (-1.0, 3.0, -0.5)),
        ((False, True), (False, True), (-2.0, 2.0)),
        ((True, False), (True, True), (1.0, 0.5)),
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
        prediction(
            "8-DS-1", truth=True, score=1.0, diagnosis=Diagnosis.PNEUMONIA, row_id="c"
        ),
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
        prediction("1-DS-1", truth=True, score=1.0, fold=0),
        prediction("2-DS-1", truth=True, score=-1.0, fold=0),
        prediction("3-DS-1", truth=False, score=-2.0, fold=1),
        prediction("4-DS-1", truth=False, score=0.5, fold=1),
    )
    use_case, scorer, _ = _use_case(items)

    result = use_case.run()

    assert [(item.test_notes, item.test_positives) for item in result] == [(4, 2)]
    assert scorer.calls == [
        ((True, True, False, False), (True, False, False, True), (1.0, -1.0, -2.0, 0.5))
    ]
