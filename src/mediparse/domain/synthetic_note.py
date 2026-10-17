"""Vylosovaná zpráva syntetického korpusu před verbalizací: kdo ji má a jaké labely nese."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mediparse.domain.identity import SubjectId
    from mediparse.domain.labels import LabelSet


@dataclass(frozen=True)
class SyntheticNote:
    """Zpráva syntetického pacienta s labely, které nese."""

    note_id: str
    subject_id: SubjectId
    labels: LabelSet
