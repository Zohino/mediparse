"""Sdílená data testů modelů: predikce, config tréninku a builder souboru splitu."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import pyarrow as pa
import pyarrow.parquet as pq

from mediparse.application.split_materialization import SplitMaterialization
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.evaluation import Prediction
from mediparse.domain.input_note import InputNote
from mediparse.domain.language import Language
from mediparse.domain.note import note_id, subject_of
from mediparse.domain.smoketest_training import TrainingConfig
from mediparse.domain.split import SharedSplit, SplitConfig
from mediparse.infrastructure.sklearn_split import StratifiedGroupFolds
from mediparse.infrastructure.split_table import METADATA_KEY, SCHEMA, ParquetSplitTable

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

SEPARABLE_VOCABULARY: Final = (
    "diabetes insulin glucose metformin hyperglycemia",
    "fracture cast orthopedic surgery splint",
)

OVERLAPPING: Final = (
    "diabetes insulin glucose clinic review",
    "fracture cast splint clinic review",
)


def separable_notes(
    patients: int, vocabulary: tuple[str, str] = SEPARABLE_VOCABULARY
) -> list[InputNote]:
    """Zprávy pacientů střídavě s diabetem a bez něj, pacient má 1 až 7 zpráv nepravidelně.

    Args:
        patients: Počet pacientů s ``subject_id`` od 1; diabetes má každý druhý
            počínaje prvním.
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
                note_id(patient + 1, visit),
                patient + 1,
                Language.EN,
                f"{vocabulary[0] if positive else vocabulary[1]} visit {visit}",
                labels,
            )
            for visit in range(1 + (patient * patient * 3 + patient) % 7)
        )
    return notes


def prediction(
    note_id: str,
    *,
    truth: bool,
    score: float,
    diagnosis: Diagnosis = Diagnosis.DIABETES,
    fold: int = 0,
    row_id: str = "row",
) -> Prediction:
    """Predikce, jejíž příznak ``predicted`` plyne ze znaménka skóre.

    Returns:
        Predikce pacienta, kterého určuje ``note_id``, kladná při ``score > 0``.
    """
    return Prediction(
        row_id,
        fold,
        note_id,
        subject_of(note_id),
        diagnosis,
        truth=truth,
        predicted=score > 0,
        score=score,
    )


def training_config(**overrides: object) -> dict[str, object]:
    """Platný config tréninku ve tvaru JSON; test změní jen pole, o kterém je.

    Returns:
        Slovník s klíči ``row_id``, ``diagnosis``, ``seed``, ``fold`` a
        ``regularization`` a ``strip_diagnostic_sections``, přepsanými hodnotami z ``overrides``.
    """
    return {
        "row_id": "row",
        "diagnosis": "diabetes",
        "seed": 0,
        "fold": 0,
        "regularization": 1.0,
        "strip_diagnostic_sections": False,
    } | overrides


def parsed_training_config(**overrides: object) -> TrainingConfig:
    """Config tréninku z ``training_config`` rozparsovaný jako v produkci.

    Returns:
        ``TrainingConfig`` s přepsanými hodnotami z ``overrides``.
    """
    return TrainingConfig.model_validate_json(json.dumps(training_config(**overrides)))


def split_config_json(**overrides: object) -> dict[str, object]:
    """Platný config splitu ve tvaru JSON; test změní jen pole, o kterém je.

    Returns:
        Slovník s klíči ``seed`` a ``folds``, přepsanými hodnotami z ``overrides``.
    """
    return {"seed": 0, "folds": 2} | overrides


def split_config(**overrides: object) -> SplitConfig:
    """Config splitu z ``split_config_json`` rozparsovaný jako v produkci.

    Returns:
        ``SplitConfig`` s přepsanými hodnotami z ``overrides``.
    """
    return SplitConfig.model_validate_json(json.dumps(split_config_json(**overrides)))


@dataclass(frozen=True)
class _MemoryNotes:
    notes: Sequence[InputNote]

    def read(self) -> tuple[InputNote, ...]:
        return tuple(self.notes)


def write_split(
    path: Path, notes: Sequence[InputNote], **overrides: object
) -> SharedSplit:
    """Zapíše soubor splitu zpráv přes use case kroku ``make_smoketest_splits``.

    Args:
        path: Cíl ``splits.parquet``.
        notes: Zprávy, jejichž pacienti se rozdělí.
        **overrides: Přepsaná pole ``split_config_json``.

    Returns:
        Zapsaný split s configem.
    """
    step = SplitMaterialization(
        table=_MemoryNotes(notes),
        splitter=StratifiedGroupFolds(),
        sink=ParquetSplitTable(path),
    )
    return step.run(split_config(**overrides))


def write_raw_split(
    path: Path,
    rows: Sequence[tuple[int, int]],
    metadata: dict[str, object] | None,
) -> None:
    """Zapíše soubor splitu mimo adaptér, aby šlo vyrobit stav, který adaptér nezapíše.

    Args:
        path: Cíl souboru.
        rows: Dvojice ``subject_id`` a ``fold``.
        metadata: Obsah metadat splitu; ``None`` soubor zapíše bez nich.
    """
    schema = SCHEMA
    if metadata is not None:
        schema = SCHEMA.with_metadata({METADATA_KEY: json.dumps(metadata).encode()})
    columns = {
        "subject_id": [subject for subject, _ in rows],
        "fold": [fold for _, fold in rows],
    }
    pq.write_table(pa.Table.from_pydict(columns, schema=schema), path)
