"""Kohorta z převedených tabulek MIMIC v DuckDB, zapsaná proudem dávek do parquetu s pevným schématem."""

from __future__ import annotations

from contextlib import closing, contextmanager, suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from duckdb import ColumnExpression, ConstantExpression, Expression, FunctionExpression

from mediparse.domain.cohort import (
    CohortCounts,
    FunnelStep,
    FunnelStepName,
)
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.language import Language
from mediparse.infrastructure.duckdb_parquet import create_parquet_view
from mediparse.infrastructure.note_table import SCHEMA
from mediparse.infrastructure.parquet_file import COMPRESSION

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping
    from pathlib import Path

    from mediparse.domain.cohort import RawCode

MIN_CHARS: Final = 100
BATCH_ROWS: Final = 4096
LANGUAGE: Final = Language.EN
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
-- Otisk a délka vznikají v samostatné CTE před oknem: okno přímo nad hash(text)
-- by materializovalo celý text. Text se připojí až v _COHORT. Celý text jako klíč
-- deduplikace (PARTITION BY text) spadl v sondě S23 na paměť.
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
-- Filtry platí před deduplikací, aby kopie bez ICD nebo příliš krátká nevyřadila
-- platnou kopii; počty tak odpovídají sešitům EDA 5, 7 a 8.
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
_FILTERS: Final[dict[FunnelStepName, Expression]] = {
    FunnelStepName.ALL_NOTES: ConstantExpression(value=True),
    FunnelStepName.WITH_ICD: ColumnExpression("has_icd"),
    FunnelStepName.LONG_ENOUGH: ColumnExpression("copy").isnotnull(),
    FunnelStepName.UNIQUE: ColumnExpression("copy") == ConstantExpression(1),
}
_POSITIVES: Final = """
SELECT diagnosis, count(*)
FROM (SELECT unnest(labels) AS diagnosis FROM base WHERE copy = 1)
GROUP BY diagnosis
"""
_COHORT: Final = """
SELECT b.*, d.text
FROM base b
JOIN discharge d USING (note_id)
WHERE b.copy = 1
"""
_COLUMNS: Final = (
    ColumnExpression("note_id"),
    ColumnExpression("subject_id"),
    ConstantExpression(LANGUAGE.value).alias("language"),
    ColumnExpression("text"),
    *(
        FunctionExpression(
            "list_contains",
            ColumnExpression("labels"),
            ConstantExpression(diagnosis.value),
        ).alias(diagnosis.value)
        for diagnosis in Diagnosis
    ),
)


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
                create_parquet_view(con, name, path)
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
                msg = f"Tabulka {self.diagnoses} nejde číst: {type(error).__name__}."
                # Text výjimky nese hodnoty buněk z MIMIC; nesmí do hlášky ani do logu, a proto
                # jen její typ a ``from None``.
                raise InvalidInputError(msg) from None
        return rows

    @contextmanager
    def select(self, labels: Mapping[RawCode, Diagnosis]) -> Generator[DuckDbSelection]:
        """Vybere kohortu do dočasné tabulky a spočítá agregáty; spojení žije do konce kontextu.

        Yields:
            Výběr s agregáty a zápisem téže dočasné tabulky.

        Raises:
            InvalidInputError: Vstup nejde číst nebo ho DuckDB nezpracuje.
        """
        with closing(self.connect()) as con:
            try:
                _select(con, labels)
                counts = _counts(con)
            except duckdb.Error as error:
                msg = (
                    f"Kohortu nelze sestavit z {self.discharge} a {self.diagnoses}: "
                    f"{type(error).__name__}."
                )
                # Text výjimky nese hodnoty buněk z MIMIC; nesmí do hlášky ani do logu, a proto
                # jen její typ a ``from None``.
                raise InvalidInputError(msg) from None
            yield DuckDbSelection(con, self.out, counts)


@dataclass(frozen=True)
class DuckDbSelection:
    """Vybraná kohorta v dočasné tabulce ``base`` otevřeného spojení."""

    connection: duckdb.DuckDBPyConnection
    out: Path
    counts: CohortCounts

    def write(self) -> None:
        """Zapíše vybrané zprávy do parquetu.

        Raises:
            InvalidInputError: Zápis proudu selhal v DuckDB nebo na disku.
        """
        try:
            _stream(self.connection, self.out)
        # pyarrow při přetypování na SCHEMA (NULL, jiná jména či pořadí sloupců) hází holý ValueError, ArrowInvalid ho nechytí.
        except (duckdb.Error, OSError, ValueError) as error:
            # Rodič výstupu může být soubor; úklid nesmí zakrýt původní chybu.
            with suppress(OSError):
                self.out.unlink(missing_ok=True)
            msg = f"Kohortu nelze zapsat do {self.out}: {type(error).__name__}."
            # Text výjimky nese hodnoty buněk z MIMIC; nesmí do hlášky ani do logu, a proto
            # jen její typ a ``from None``.
            raise InvalidInputError(msg) from None


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


def _step(base: duckdb.DuckDBPyRelation, step: FunnelStepName) -> FunnelStep:
    notes, patients = (
        base
        .filter(_FILTERS[step])
        .aggregate("count(*), count(DISTINCT subject_id)")
        .fetchall()[0]
    )
    return FunnelStep(step=step, notes=notes, patients=patients)


def _counts(con: duckdb.DuckDBPyConnection) -> CohortCounts:
    base = con.table("base")
    funnel = tuple(_step(base, step) for step in FunnelStepName)
    found = dict(con.execute(_POSITIVES).fetchall())
    positives = {diagnosis: found.get(diagnosis.value, 0) for diagnosis in Diagnosis}
    return CohortCounts(funnel=funnel, positives=positives)


def _stream(con: duckdb.DuckDBPyConnection, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    # Řazení je poslední operace: pořadí řádků vstupuje do cohort_sha256 v manifestu.
    cohort = con.sql(_COHORT).select(*_COLUMNS).order("note_id")
    with pq.ParquetWriter(out, SCHEMA, compression=COMPRESSION) as writer:
        for batch in cohort.to_arrow_reader(BATCH_ROWS):
            writer.write_batch(batch.cast(SCHEMA), row_group_size=BATCH_ROWS)
