"""Report shod auditu: pozice sdílených n-gramů a kolize subject_id, nikdy text zpráv."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from mediparse.domain.corpus_audit import Position
from mediparse.domain.identity import SubjectId
from mediparse.infrastructure.json_file import write_model

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence
    from pathlib import Path


class _OverlapReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    positions: tuple[Position, ...]
    colliding_subjects: tuple[SubjectId, ...]


@dataclass(frozen=True)
class OverlapReportFile:
    """JSON report shod na cestě, kterou určil autor auditu."""

    path: Path

    def write(
        self, positions: Sequence[Position], colliding_subjects: Collection[SubjectId]
    ) -> None:
        """Zapíše pozice shod a kolidující subject_id."""
        write_model(
            self.path,
            _OverlapReport(
                positions=tuple(positions),
                colliding_subjects=tuple(sorted(colliding_subjects)),
            ),
        )
