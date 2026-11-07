"""Vstupní bod pravidla Snakemake ``parquet_mimic_table``: validovaná tabulka MIMIC z CSV na parquet.

Cesty jsou vstupy a výstupy pravidla; skript workflow jen rozbalí objekt
``snakemake`` a zavolá ``main``. Nesoulad počtu záznamů nebo sloupců s inventářem
validace ukončí krok kódem REFUSED a Snakemake rozpracovaný výstup smaže.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mediparse.application.mimic_parquet import MimicParquetConversion
from mediparse.domain.inputs import InvalidInputError
from mediparse.entrypoints.cli import configure_logging, refusing_invalid_input
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.csv_parquet import CsvParquetConversion
from mediparse.infrastructure.mimic_inventory import InventoryFile
from mediparse.infrastructure.parquet_source import source_path, write_parquet_source

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def main(
    *, table: Path, inventory: Path, parquet: Path, source: Path, log: Path
) -> ExitCode:
    """Krok pravidla: diagnostika do logu kroku, chyby i na stderr.

    Returns:
        Návratový kód kroku.
    """
    configure_logging(log)
    return run(table=table, inventory=inventory, parquet=parquet, source=source)


@refusing_invalid_input
def run(*, table: Path, inventory: Path, parquet: Path, source: Path) -> ExitCode:
    """Převede tabulku na parquet a ověří ji proti inventáři validace.

    Returns:
        OK po ověřeném převodu; REFUSED pro chybějící nebo rozbitý vstup
        a pro tvar, který se liší od inventáře.

    Raises:
        InvalidInputError: Nesprávnou cestu vazby dekorátor přeloží na REFUSED.
    """
    if source != source_path(parquet):
        msg = f"Vazba zdroje musí mít cestu {source_path(parquet)}."
        raise InvalidInputError(msg)
    inventory_file = InventoryFile(inventory)
    shape = MimicParquetConversion(
        inventory_file, CsvParquetConversion(table, parquet)
    ).run()
    write_parquet_source(
        parquet, source=table.name, source_sha256=inventory_file.sha256()
    )
    logger.info("%s: %d záznamů v %s", table.name, shape.records, parquet)
    return ExitCode.OK
