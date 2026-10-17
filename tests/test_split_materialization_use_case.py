"""Use case materializace splitu: zprávy → strata → dělení → zápis, s kontrolou počtu pacientů."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from mediparse.application.split_materialization import SplitMaterialization
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.input_note import InputNote
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.language import Language
from mediparse.domain.split import SharedSplit, Split, SplitConfig, strata
from tests.model_support import split_config

if TYPE_CHECKING:
    from collections.abc import Sequence

CONFIG = split_config(folds=2, seed=4)
SPLIT = Split((frozenset({1, 2}), frozenset({3})))
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
class _FakeSplitter:
    calls: list[tuple[tuple[int, ...], tuple[int, ...], SplitConfig]] = field(
        default_factory=list
    )

    def split(
        self, subjects: Sequence[int], strata: Sequence[int], config: SplitConfig
    ) -> Split:
        self.calls.append((tuple(subjects), tuple(strata), config))
        return SPLIT


@dataclass
class _FakeSink:
    written: list[SharedSplit] = field(default_factory=list)

    def write(self, shared: SharedSplit) -> None:
        self.written.append(shared)


def _use_case(
    notes: Sequence[InputNote], splitter: _FakeSplitter, sink: _FakeSink
) -> SplitMaterialization:
    return SplitMaterialization(table=_FakeTable(notes), splitter=splitter, sink=sink)


def test_sink_receives_split_with_config_of_run() -> None:
    """Sink dostane SharedSplit s configem z run a týž výsledek se vrátí."""
    splitter, sink = _FakeSplitter(), _FakeSink()

    shared = _use_case(NOTES, splitter, sink).run(CONFIG)

    assert shared == SharedSplit(CONFIG, SPLIT)
    assert sink.written == [shared]


def test_splitter_receives_strata_and_config() -> None:
    """Splitter dostane pacienty zpráv, jejich strata z domény a týž config."""
    splitter, sink = _FakeSplitter(), _FakeSink()

    _use_case(NOTES, splitter, sink).run(CONFIG)

    assert splitter.calls == [((1, 2, 2, 3), strata(NOTES, CONFIG.folds), CONFIG)]


def test_too_few_patients_stop_before_splitting_and_writing() -> None:
    """Méně pacientů než foldů je InvalidInputError a splitter ani sink se nevolá."""
    splitter, sink = _FakeSplitter(), _FakeSink()

    with pytest.raises(InvalidInputError, match="Pacientů je 3"):
        _use_case(NOTES, splitter, sink).run(split_config(folds=4))

    assert splitter.calls == []
    assert sink.written == []
