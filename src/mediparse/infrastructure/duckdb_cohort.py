"""Kohorta z převedených tabulek MIMIC v DuckDB, zapsaná proudem dávek do parquetu s pevným schématem."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from mediparse.domain.cohort import (
    CohortCounts,
    FunnelStep,
    FunnelStepName,
    ensure_publishable,
)
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.labels import Diagnosis
from mediparse.infrastructure.note_table import SCHEMA

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from mediparse.domain.cohort import RawCode

MIN_CHARS: Final = 100
BATCH_ROWS: Final = 4096
COMPRESSION: Final = "zstd"
_STEPS: Final = tuple(FunnelStepName)
_BASE: Final = """
CREATE TEMPORARY TABLE base AS
WITH admissions AS (
    SELECT
        i.hadm_id,
        list(DISTINCT l.diagnosis) FILTER (WHERE l.diagnosis IS NOT NULL) AS labels
    FROM diagnoses i
    LEFT JOIN code_labels l
        ON i.icd_version = l.icd_version AND i.icd_code = l.icd_code
    GROUP BY i.hadm_id
),
metrics AS (
    SELECT
        note_id,
        CAST(subject_id AS BIGINT) AS subject_id,
        hadm_id,
        hash(text) AS fingerprint,
        coalesce(length(text), 0) AS chars
    FROM discharge
),
joined AS (
    SELECT
        m.note_id,
        m.subject_id,
        m.fingerprint,
        a.hadm_id IS NOT NULL AS has_icd,
        m.chars >= $min_chars AS long_enough,
        coalesce(a.labels, CAST([] AS VARCHAR[])) AS labels
    FROM metrics m
    LEFT JOIN admissions a ON a.hadm_id = m.hadm_id
),
eligible AS (
    SELECT
        note_id,
        row_number() OVER (PARTITION BY fingerprint ORDER BY note_id) AS copy
    FROM joined
    WHERE has_icd AND long_enough
)
SELECT j.*, e.copy
FROM joined j
LEFT JOIN eligible e ON e.note_id = j.note_id
"""
_FUNNEL: Final = """
SELECT 1, count(*), count(DISTINCT subject_id) FROM base
UNION ALL SELECT 2, count(*), count(DISTINCT subject_id) FROM base WHERE has_icd
UNION ALL SELECT 3, count(*), count(DISTINCT subject_id) FROM base WHERE copy IS NOT NULL
UNION ALL SELECT 4, count(*), count(DISTINCT subject_id) FROM base WHERE copy = 1
ORDER BY 1
"""
_POSITIVES: Final = """
SELECT diagnosis, count(*)
FROM (SELECT unnest(labels) AS diagnosis FROM base WHERE copy = 1)
GROUP BY diagnosis
"""
_COHORT: Final = """
SELECT
    b.note_id,
    b.subject_id,
    'en' AS language,
    d.text,
    list_contains(b.labels, 'diabetes') AS diabetes,
    list_contains(b.labels, 'copd') AS copd,
    list_contains(b.labels, 'ckd') AS ckd,
    list_contains(b.labels, 'pneumonia') AS pneumonia,
    list_contains(b.labels, 'atrial_fibrillation') AS atrial_fibrillation
FROM base b
JOIN discharge d ON d.note_id = b.note_id
WHERE b.copy = 1
ORDER BY b.note_id
"""


@dataclass(frozen=True)
class DuckDbCohort:
    """Kohorta z ``discharge`` a ``diagnoses_icd``; spill DuckDB jde do ``spill``."""

    discharge: Path
    diagnoses: Path
    out: Path
    spill: Path

    def connect(self) -> duckdb.DuckDBPyConnection:
        """Spojení v paměti se spillem v předaném adresáři a s oběma tabulkami.

        Returns:
            Spojení s pohledy ``discharge`` a ``diagnoses``.

        Raises:
            InvalidInputError: Vstupní tabulka chybí nebo se nedá číst.
        """
        self.spill.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(config={"temp_directory": str(self.spill)})
        try:
            for name, path in (
                ("discharge", self.discharge),
                ("diagnoses", self.diagnoses),
            ):
                _view(con, name, path)
        except InvalidInputError:
            con.close()
            raise
        return con

    def codes(self) -> list[RawCode]:
        """Odlišné dvojice verze a kódu z ``diagnoses_icd``.

        Returns:
            Dvojice tak, jak leží v tabulce.

        Raises:
            InvalidInputError: Tabulka chybí nebo nemá očekávané sloupce.
        """
        with closing(self.connect()) as con:
            try:
                rows = con.execute(
                    "SELECT DISTINCT icd_version, icd_code FROM diagnoses"
                ).fetchall()
            except duckdb.Error as error:
                msg = f"Tabulka {self.diagnoses} nejde číst: {error}"
                raise InvalidInputError(msg) from error
        return rows

    def write(self, labels: Mapping[RawCode, Diagnosis]) -> CohortCounts:
        """Vybere kohortu, ověří agregáty a teprve poté zapíše parquet.

        Returns:
            Trychtýř a počty pozitivních zpráv zapsané kohorty.

        Raises:
            InvalidInputError: Vstup nejde číst nebo kohorta porušuje pravidla domény.
        """
        with closing(self.connect()) as con:
            try:
                _select(con, labels)
                counts = _counts(con)
                ensure_publishable(counts)
                _stream(con, self.out)
            except duckdb.Error as error:
                msg = f"Kohortu nelze sestavit z {self.discharge} a {self.diagnoses}: {error}"
                raise InvalidInputError(msg) from error
        return counts


def _view(con: duckdb.DuckDBPyConnection, name: str, path: Path) -> None:
    try:
        con.read_parquet(str(path)).create_view(name)
    except duckdb.Error as error:
        msg = f"Tabulka {path} nejde číst: {error}"
        raise InvalidInputError(msg) from error


def _select(
    con: duckdb.DuckDBPyConnection, labels: Mapping[RawCode, Diagnosis]
) -> None:
    pairs = list(labels)
    table = pa.table({
        "icd_version": pa.array([pair[0] for pair in pairs], pa.string()),
        "icd_code": pa.array([pair[1] for pair in pairs], pa.string()),
        "diagnosis": pa.array([labels[pair].value for pair in pairs], pa.string()),
    })
    con.register("code_labels", table)
    con.execute(_BASE, {"min_chars": MIN_CHARS})


def _counts(con: duckdb.DuckDBPyConnection) -> CohortCounts:
    funnel = tuple(
        FunnelStep(step=_STEPS[index - 1], notes=notes, patients=patients)
        for index, notes, patients in con.execute(_FUNNEL).fetchall()
    )
    found = dict(con.execute(_POSITIVES).fetchall())
    positives = {diagnosis: found.get(diagnosis.value, 0) for diagnosis in Diagnosis}
    return CohortCounts(funnel=funnel, positives=positives)


def _stream(con: duckdb.DuckDBPyConnection, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    reader = con.sql(_COHORT).to_arrow_reader(BATCH_ROWS)
    with pq.ParquetWriter(out, SCHEMA, compression=COMPRESSION) as writer:
        for batch in reader:
            writer.write_batch(batch.cast(SCHEMA), row_group_size=BATCH_ROWS)
