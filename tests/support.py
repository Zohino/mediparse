"""Cesty k souborům repa na jednom místě, sdílená data testů a fakes."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from mediparse.domain.corpus_audit import (
    NGRAM_SIZE,
    NORMALIZATION,
    AuditPin,
    AuditRecord,
)
from mediparse.domain.corpus_provenance import (
    Decoding,
    Generation,
    Translation,
    TranslationRun,
)
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.environment import SourceRevision
from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.mimic_tables import TABLES_PATH
from mediparse.infrastructure.plans_file import LABELS_PATH, PLANS_PATH
from mediparse.infrastructure.sampler_config import SAMPLER_CONFIG_PATH
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
REPOSITORY_TRAINING_CONFIG: Final = REPOSITORY / "config" / "smoketest_training.json"
REPOSITORY_SPLIT_CONFIG: Final = REPOSITORY / "config" / "split.json"
REPOSITORY_DIAGNOSIS_CANDIDATES: Final = (
    REPOSITORY / "config" / "diagnosis_candidates.json"
)
REPOSITORY_MATRIX: Final = REPOSITORY / "config" / "matrix.json"
STRUCTURE_LABELS: Final = frozenset({"Name", "Unit No"})

COMMIT: Final = "c" * 40
NOTE_ID: Final = "90000001-DS-1"
SHORT_TEXT: Final = "a short synthetic note"
NOTE: Final = f"en/{NOTE_ID}.txt"
SHORT_NOTE: Final = {NOTE: SHORT_TEXT}
SENTENCE: Final = (
    "the old lighthouse keeper counted seven gulls before the storm "
    "reached the northern harbor wall"
)
CORPUS_SHA: Final = "a" * 64
PINNED: Final = {"discharge.csv.gz": "b" * 64, "radiology.csv.gz": "e" * 64}
DISCHARGE: Final = {"name": "discharge.csv.gz", "sha256": "b" * 64, "rows": 10}
RADIOLOGY: Final = {"name": "radiology.csv.gz", "sha256": "e" * 64, "rows": 20}
PIN: Final = AuditPin(PINNED, STRUCTURE_LABELS)
REVISION: Final = SourceRevision(commit=COMMIT, dirty=False)


def fixed_revision() -> SourceRevision:
    """Zdroj revize, který vždy vrátí ``REVISION``.

    Returns:
        Pevná revize bez dotazu na git.
    """
    return REVISION


def missing_revision() -> SourceRevision:
    """Zdroj revize, který selže jako git mimo repozitář.

    Raises:
        InvalidInputError: Vždy.
    """
    msg = "git chybí"
    raise InvalidInputError(msg)


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


GENERATION_FIELDS: Final = {
    "seed": 0,
    "sampler_config_sha256": "1" * 64,
    "prompts_sha256": "3" * 64,
    "model": "claude-sonnet-5-5",
    "claude_code_version": "2.1.5",
    "generated_on": "2026-10-10",
    "specification_commit": COMMIT,
}
GENERATION_ARGS: Final = {
    "model": GENERATION_FIELDS["model"],
    "claude-code-version": GENERATION_FIELDS["claude_code_version"],
    "date": GENERATION_FIELDS["generated_on"],
    "specification-commit": GENERATION_FIELDS["specification_commit"],
}


def generation_record(**overrides: object) -> Generation:
    """Záznam generování korpusu ve tvaru ``GENERATION_FIELDS`` na disku.

    Returns:
        Záznam, v němž ``overrides`` (hodnoty ve tvaru JSON) nahradí jednotlivá pole.
    """
    return Generation.model_validate_json(json.dumps(GENERATION_FIELDS | overrides))


def translation_record(**overrides: object) -> Translation:
    """Provenance překladu s jedním během a bez nesouladů značek.

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


def export_payload(
    selection: tuple[str, ...] = tuple(Diagnosis), prevalence: float = 0.2
) -> dict[str, Any]:
    """Export parametrů pětice s nezávislými labely a jednotnou prevalencí.

    Returns:
        Strom JSON ve tvaru exportu sešitu 8.
    """
    return {
        "selection": list(selection),
        "prevalence": dict.fromkeys(selection, prevalence),
        "conditional": {
            row: {column: 1.0 if row == column else prevalence for column in selection}
            for row in selection
        },
        "mentions": {
            key: {
                "given_positive": 0.9,
                "given_negative": 0.05,
                "keywords": [f"{key} phrase", key.upper()],
            }
            for key in selection
        },
    }
