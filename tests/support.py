"""Cesty k souborům repa na jednom místě, sdílená data testů a fakes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from mediparse.domain.corpus_audit import NGRAM_SIZE, NORMALIZATION, AuditRecord
from mediparse.infrastructure.mimic_tables import TABLES_PATH
from mediparse.infrastructure.plans_file import LABELS_PATH, PLANS_PATH
from mediparse.infrastructure.sampler_config import SAMPLER_CONFIG_PATH
from mediparse.infrastructure.synthetic_corpus import CORPUS_ROOT
from mediparse.infrastructure.verbalization_template import (
    VERBALIZATION_TEMPLATE_PATH,
)

if TYPE_CHECKING:
    from mediparse.domain.note_plan import NotePlan

REPOSITORY: Final = Path(__file__).parents[1]
REPOSITORY_CONFIG: Final = REPOSITORY / SAMPLER_CONFIG_PATH
REPOSITORY_TEMPLATE: Final = REPOSITORY / VERBALIZATION_TEMPLATE_PATH
REPOSITORY_PLANS: Final = REPOSITORY / PLANS_PATH
REPOSITORY_LABELS: Final = REPOSITORY / LABELS_PATH
REPOSITORY_CORPUS: Final = REPOSITORY / CORPUS_ROOT
REPOSITORY_TABLES: Final = REPOSITORY / TABLES_PATH
REPOSITORY_MATRIX: Final = REPOSITORY / "config" / "matrix.json"

COMMIT: Final = "c" * 40
NOTE: Final = "en/90000001-DS-1.txt"
SHORT_NOTE: Final = {NOTE: "a short synthetic note"}
SENTENCE: Final = (
    "the old lighthouse keeper counted seven gulls before the storm "
    "reached the northern harbor wall"
)
CORPUS_SHA: Final = "a" * 64
PINNED: Final = {"discharge.csv.gz": "b" * 64, "radiology.csv.gz": "e" * 64}
DISCHARGE: Final = {"name": "discharge.csv.gz", "sha256": "b" * 64, "rows": 10}
RADIOLOGY: Final = {"name": "radiology.csv.gz", "sha256": "e" * 64, "rows": 20}


def audit_record(**overrides: object) -> AuditRecord:
    """Záznam čistého auditu korpusu ``CORPUS_SHA`` proti referenci ``PINNED``.

    Returns:
        Záznam, v němž ``overrides`` nahradí jednotlivá pole JSON.
    """
    fields = {
        "corpus_sha256": CORPUS_SHA,
        "corpus_files": 1,
        "ngram_size": NGRAM_SIZE,
        "normalization": NORMALIZATION,
        "synthetic_ngrams": 3,
        "reference": [DISCHARGE, RADIOLOGY],
        "shared_ngrams": 0,
        "colliding_subjects": 0,
        "tool_commit": COMMIT,
        "created_at": "2026-09-28T00:00:00+00:00",
    } | overrides
    return AuditRecord.model_validate_json(json.dumps(fields))


@dataclass(frozen=True)
class PlansFake:
    """Zdroj plánů v paměti."""

    plans: tuple[NotePlan, ...]

    def load(self) -> tuple[NotePlan, ...]:
        """Plány v pořadí zadání.

        Returns:
            Uložené plány.
        """
        return self.plans
