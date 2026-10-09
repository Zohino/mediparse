"""Cesty k souborům repa na jednom místě, sdílená data testů a fakes."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final

from mediparse.domain.corpus_audit import NGRAM_SIZE, NORMALIZATION, AuditRecord
from mediparse.domain.corpus_provenance import Decoding, Translation, TranslationRun
from mediparse.domain.labels import Diagnosis
from mediparse.domain.smoketest_input import InputNote
from mediparse.infrastructure.mimic_tables import TABLES_PATH
from mediparse.infrastructure.plans_file import LABELS_PATH, PLANS_PATH
from mediparse.infrastructure.sampler_config import (
    SAMPLER_CONFIG_PATH,
    load_sampler_config,
)
from mediparse.infrastructure.synthetic_corpus import CORPUS_ROOT
from mediparse.infrastructure.verbalization_template import (
    VERBALIZATION_TEMPLATE_PATH,
)

if TYPE_CHECKING:
    from types import ModuleType

    import pytest

    from mediparse.domain.note_plan import NotePlan

REPOSITORY: Final = Path(__file__).parents[1]
REPOSITORY_CONFIG: Final = REPOSITORY / SAMPLER_CONFIG_PATH
REPOSITORY_TEMPLATE: Final = REPOSITORY / VERBALIZATION_TEMPLATE_PATH
REPOSITORY_PLANS: Final = REPOSITORY / PLANS_PATH
REPOSITORY_LABELS: Final = REPOSITORY / LABELS_PATH
REPOSITORY_CORPUS: Final = REPOSITORY / CORPUS_ROOT
REPOSITORY_TABLES: Final = REPOSITORY / TABLES_PATH
REPOSITORY_MATRIX: Final = REPOSITORY / "config" / "matrix.json"
STRUCTURE_LABELS: Final = load_sampler_config(
    REPOSITORY_CONFIG
).structure.structure_labels

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
        "structure_labels": sorted(STRUCTURE_LABELS),
        "synthetic_ngrams": 3,
        "reference": [DISCHARGE, RADIOLOGY],
        "shared_ngrams": 0,
        "colliding_subjects": 0,
        "tool_commit": COMMIT,
        "created_at": "2026-09-28T00:00:00+00:00",
    } | overrides
    return AuditRecord.model_validate_json(json.dumps(fields))


SEPARABLE_VOCABULARY: Final = (
    "diabetes insulin glucose metformin hyperglycemia",
    "fracture cast orthopedic surgery splint",
)


def separable_notes(
    patients: int, vocabulary: tuple[str, str] = SEPARABLE_VOCABULARY
) -> list[InputNote]:
    """Zprávy pacientů střídavě s diabetem a bez něj, pacient má 1 až 7 zpráv nepravidelně.

    Args:
        patients: Počet pacientů; sudí mají diabetes.
        vocabulary: Slova zpráv pozitivních a negativních pacientů; společná
            slova z nich dělají neseparovatelný korpus.

    Returns:
        Zprávy po pacientech v pořadí pacientů.
    """
    notes = []
    for patient in range(patients):
        positive = patient % 2 == 0
        labels = frozenset({Diagnosis.DIABETES} if positive else ())
        notes.extend(
            InputNote(
                f"{patient}-DS-{visit}",
                patient,
                "en",
                f"{vocabulary[0] if positive else vocabulary[1]} visit {visit}",
                labels,
            )
            for visit in range(1 + (patient * patient * 3 + patient) % 7)
        )
    return notes


def translation_record(**overrides: object) -> Translation:
    """Provenance překladu s jedním během a bez neshod značek.

    Returns:
        Záznam, v němž ``overrides`` nahradí jednotlivá pole.
    """
    run = TranslationRun(
        started=datetime(2026, 10, 7, 17, 21, tzinfo=UTC),
        requests_sha256="4" * 64,
        script_sha256="5" * 64,
        gpu="NVIDIA A40",
        cuda="13.0",
        packages={"vllm": "0.30.0"},
    )
    fields = {
        "model": "google/translategemma-12b-it",
        "revision": "d" * 40,
        "dtype": "bfloat16",
        "window": 2048,
        "decoding": Decoding(temperature=0.0, stop_token_ids=(1, 106)),
        "requests_sha256": "4" * 64,
        "runs": (run,),
        "marker_mismatches": (),
    } | overrides
    return Translation.model_validate(fields)


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


def load_script(path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Načte skript z adresáře translation jako modul a po testu po sobě uklidí.

    Args:
        path: Cesta ke skriptu.
        monkeypatch: Fixture, přes kterou se vrátí ``sys.path`` a ``sys.modules``.

    Returns:
        Načtený modul zaregistrovaný v ``sys.modules``.
    """
    monkeypatch.syspath_prepend(str(path.parent))
    for name in (path.stem, "note_parts", "markers"):
        monkeypatch.setitem(sys.modules, name, None)
        monkeypatch.delitem(sys.modules, name)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, path.stem, module)
    spec.loader.exec_module(module)
    return module
