"""Tabulky pokusů stupně a řádkových predikcí pod DUA."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import pyarrow as pa

from mediparse.domain.stage_tuning import BRANCH_KEY, StageTrialRecord, branch_parts
from mediparse.infrastructure.parquet_checkpoint import ParquetParts
from mediparse.infrastructure.parquet_file import read_table

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from mediparse.domain.stage_tuning import StageFoldResult

# default je bod mimo návrhy sampleru: výchozí bod předkola nebo kontrola kandidáta.
TRIALS_SCHEMA: Final = pa.schema([
    pa.field(name, kind, nullable=False)
    for name, kind in (
        ("trial", pa.int64()),
        ("fold", pa.int64()),
        ("diagnosis", pa.string()),
        ("branch", pa.string()),
        ("representation", pa.string()),
        ("family", pa.string()),
        ("default", pa.bool_()),
        ("params", pa.string()),
        ("value", pa.float64()),
        ("seconds", pa.float64()),
        ("max_f1", pa.float64()),
        ("threshold", pa.float64()),
        ("converged", pa.bool_()),
        ("n_features", pa.int64()),
    )
])
TRIAL_PREDICTIONS_SCHEMA: Final = pa.schema([
    pa.field(name, kind, nullable=False)
    for name, kind in (
        ("trial", pa.int64()),
        ("fold", pa.int64()),
        ("diagnosis", pa.string()),
        ("note_id", pa.string()),
        ("label", pa.bool_()),
        ("predicted", pa.bool_()),
        ("score", pa.float64()),
    )
])


@dataclass(frozen=True)
class ParquetStageCheckpoint:
    """Trvalé díly po pokusech se souhrnem odděleným od predikcí."""

    directory: Path
    key: str
    results: Path
    predictions: Path

    @property
    def parts(self) -> ParquetParts:
        """Mechanika dílů s výsledky jako značkou dokončení."""
        return ParquetParts(
            self.directory,
            self.key,
            (("predictions", TRIAL_PREDICTIONS_SCHEMA), ("results", TRIALS_SCHEMA)),
            "ladění stupně",
        )

    def completed(self) -> tuple[StageTrialRecord, ...]:
        """Přečte jen souhrny dokončených pokusů.

        Returns:
            Záznamy v pořadí čísel pokusů.
        """
        records = []
        for path in self.parts.completed_parts():
            rows = read_table(
                path,
                TRIALS_SCHEMA,
            ).to_pylist()
            first = rows[0]
            records.append(
                StageTrialRecord(
                    number=first["trial"],
                    params=json.loads(first["params"]),
                    default=first["default"],
                    value=first["value"],
                    converged=all(row["converged"] for row in rows),
                    seconds=first["seconds"],
                )
            )
        return tuple(records)

    def write(
        self, record: StageTrialRecord, results: Sequence[StageFoldResult]
    ) -> None:
        """Serializuje uloženou metriku a skóre bez přepočítávání cíle."""
        branch = str(record.params[BRANCH_KEY])
        representation, family = branch_parts(branch)
        shared = {
            "trial": record.number,
            "branch": branch,
            "representation": representation,
            "family": family.value,
            "default": record.default,
            "params": json.dumps(dict(record.params), sort_keys=True),
            "value": record.value,
            "seconds": record.seconds,
        }
        trials = []
        predictions = []
        for result in results:
            cell = {
                "trial": record.number,
                "fold": result.fold,
                "diagnosis": result.diagnosis.value,
            }
            trials.append(
                shared
                | cell
                | {
                    "max_f1": result.max_f1.value,
                    "threshold": result.max_f1.threshold,
                    "converged": result.converged,
                    "n_features": result.n_features,
                }
            )
            predictions.extend(
                cell
                | {
                    "note_id": note_id,
                    "label": label,
                    "predicted": result.max_f1.predicts(score),
                    "score": score,
                }
                for note_id, label, score in zip(
                    result.note_ids, result.labels, result.scores, strict=True
                )
            )
        self.parts.write({
            "predictions": pa.Table.from_pylist(
                predictions, schema=TRIAL_PREDICTIONS_SCHEMA
            ),
            "results": pa.Table.from_pylist(trials, schema=TRIALS_SCHEMA),
        })

        logging.getLogger(__name__).info(
            "Pokus %d COMPLETE: macro-F1 %.6f, čas %.3f s.",
            record.number,
            record.value,
            record.seconds,
        )

    def finish(self) -> None:
        """Složí oba výstupy a uklidí checkpoint."""
        self.parts.finish({"results": self.results, "predictions": self.predictions})
