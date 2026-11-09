"""Use case materializace splitu: zprávy → strata → dělení → zápis, s kontrolou počtu pacientů."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mediparse.application.split_materialization import SplitMaterialization
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.input_note import InputNote
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.language import Language
from mediparse.domain.split import SharedSplit, Split
from mediparse.infrastructure.sklearn_split import StratifiedGroupFolds
from tests.model_support import split_config

if TYPE_CHECKING:
    from collections.abc import Sequence

CONFIG = split_config(folds=2, seed=4)
SPLIT = Split((frozenset({1}), frozenset({2})), development=frozenset({3}))
NOTES = (
    InputNote("1-DS-1", 1, Language.EN, "a", frozenset({Diagnosis.CKD})),
    InputNote("2-DS-1", 2, Language.EN, "b", frozenset()),
    InputNote("2-DS-2", 2, Language.EN, "c", frozenset()),
    InputNote("3-DS-1", 3, Language.EN, "d", frozenset({Diagnosis.CKD})),
)


@dataclass
class _FakeTable:
    notes: Sequence[InputNote]

    def read(self) -> tuple[InputNote, ...]:
        return tuple(self.notes)


@dataclass
class _FakePartitioner:
    calls: list[tuple[tuple[int, ...], tuple[int, ...], int, int]] = field(
        default_factory=list
    )

    def partition(
        self, subjects: Sequence[int], strata: Sequence[int], parts: int, seed: int
    ) -> tuple[frozenset[int], ...]:
        self.calls.append((tuple(subjects), tuple(strata), parts, seed))
        return (*SPLIT.folds, SPLIT.development)


@dataclass
class _FakeSink:
    written: list[SharedSplit] = field(default_factory=list)

    def write(self, shared: SharedSplit) -> None:
        self.written.append(shared)


def _use_case(
    notes: Sequence[InputNote], partitioner: _FakePartitioner, sink: _FakeSink
) -> SplitMaterialization:
    return SplitMaterialization(
        table=_FakeTable(notes), partitioner=partitioner, sink=sink
    )


def test_sink_receives_split_with_config_of_run() -> None:
    """Sink dostane SharedSplit s configem z run a týž výsledek se vrátí."""
    partitioner, sink = _FakePartitioner(), _FakeSink()

    shared = _use_case(NOTES, partitioner, sink).run(CONFIG)

    assert shared == SharedSplit(CONFIG, SPLIT)
    assert sink.written == [shared]


def test_partitioner_receives_strata_parts_and_seed() -> None:
    """Partitioner dostane pacienty, strata pro K + 1 částí, počet částí a seed."""
    partitioner, sink = _FakePartitioner(), _FakeSink()

    _use_case(NOTES, partitioner, sink).run(CONFIG)

    assert partitioner.calls == [((1, 2, 2, 3), (-1, -1, -1, -1), 3, 4)]


def test_too_few_patients_stop_before_splitting_and_writing() -> None:
    """Tři pacienti nestačí na tři foldy a vývoj; partitioner ani sink se nevolá."""
    partitioner, sink = _FakePartitioner(), _FakeSink()

    with pytest.raises(InvalidInputError, match="Pacientů je 3"):
        _use_case(NOTES, partitioner, sink).run(split_config(folds=3))

    assert partitioner.calls == []
    assert sink.written == []


@given(
    patients=st.lists(
        st.lists(st.sets(st.sampled_from(Diagnosis)), min_size=1, max_size=3),
        min_size=3,
        max_size=30,
    ),
    requested_folds=st.integers(min_value=2, max_value=5),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_materialization_partitions_all_patient_notes(
    patients: list[list[set[Diagnosis]]], requested_folds: int, seed: int
) -> None:
    """Každý pacient i všechny jeho zprávy jsou právě v jedné neprázdné části."""
    notes = tuple(
        InputNote(
            f"{subject}-DS-{visit}",
            subject,
            Language.EN,
            "syntetický text",
            frozenset(labels),
        )
        for subject, visits in enumerate(patients, start=1)
        for visit, labels in enumerate(visits, start=1)
    )
    config = split_config(folds=min(requested_folds, len(patients) - 1), seed=seed)
    shared = SplitMaterialization(
        _FakeTable(notes), StratifiedGroupFolds(), _FakeSink()
    ).run(config)

    parts = (*shared.split.folds, shared.split.development)
    assigned = [subject for part in parts for subject in part]
    assert len(shared.split.folds) == config.folds
    assert all(parts)
    assert len(assigned) == len(set(assigned))
    assert set(assigned) == {note.subject_id for note in notes}
    note_parts = [
        shared.split.holdout(notes, fold).test for fold in range(config.folds)
    ]
    note_parts.append(
        tuple(note for note in notes if note.subject_id in shared.split.development)
    )
    assert sorted(note.note_id for part in note_parts for note in part) == sorted(
        note.note_id for note in notes
    )
    for subject in assigned:
        assert (
            sum(any(note.subject_id == subject for note in part) for part in note_parts)
            == 1
        )
