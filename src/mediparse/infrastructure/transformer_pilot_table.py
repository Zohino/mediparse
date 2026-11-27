"""Veřejná tabulka agregátů pilotu po bězích a epochách."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Final

import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from mediparse.domain.transformer_pilot import PilotRecord

PILOT_SCHEMA: Final = pa.schema([
    pa.field(name, kind, nullable=False)
    for name, kind in (
        ("run", pa.string()),
        ("model", pa.string()),
        ("window", pa.int64()),
        ("aggregation", pa.string()),
        ("freeze_embeddings", pa.bool_()),
        ("fraction", pa.float64()),
        ("epoch", pa.int64()),
        ("fit_notes", pa.int64()),
        ("fit_patients", pa.int64()),
        ("stop_notes", pa.int64()),
        ("validation_notes", pa.int64()),
        ("seconds", pa.float64()),
        ("validation_seconds", pa.float64()),
        ("peak_memory_bytes", pa.int64()),
        ("stop_loss", pa.float64()),
        ("macro_f1", pa.float64()),
    )
])


@dataclass(frozen=True)
class ParquetPilotSink:
    """Přepíše malou tabulku po každém dokončeném běhu."""

    path: Path

    def write(self, records: Sequence[PilotRecord]) -> None:
        """Zapíše jen agregáty s explicitním veřejným schématem."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(
                [asdict(record) for record in records], schema=PILOT_SCHEMA
            ),
            self.path,
        )
