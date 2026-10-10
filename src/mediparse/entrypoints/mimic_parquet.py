"""Vstupní bod pravidla Snakemake ``parquet_mimic_table``: validovaná tabulka MIMIC z CSV na parquet.

Cesty jsou vstupy a výstupy pravidla; skript workflow jen rozbalí objekt
``snakemake`` a zavolá ``main``. Neshoda počtu záznamů nebo sloupců s inventářem
validace ukončí krok kódem REFUSED a Snakemake rozpracovaný výstup smaže.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mediparse.application.mimic_parquet import MimicParquetConversion
from mediparse.entrypoints.cli import configure_logging, refusing_invalid_input
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.csv_parquet import CsvParquetConversion
from mediparse.infrastructure.mimic_inventory import InventoryFile

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def main(*, table: Path, inventory: Path, parquet: Path, log: Path) -> ExitCode:
    """Krok pravidla: diagnostika do logu kroku, chyby i na stderr.

    Returns:
        Návratový kód kroku.
    """
    configure_logging(log)
    return run(table=table, inventory=inventory, parquet=parquet)


@refusing_invalid_input
def run(*, table: Path, inventory: Path, parquet: Path) -> ExitCode:
    """Převede tabulku na parquet a ověří ji proti inventáři validace.

    Returns:
        OK po ověřeném převodu; REFUSED pro chybějící nebo rozbitý vstup
        a pro tvar, který se liší od inventáře.
    """
    shape = MimicParquetConversion(
        InventoryFile(inventory), CsvParquetConversion(table, parquet)
    ).run()
    logger.info("%s: %d záznamů v %s", table.name, shape.records, parquet)
    return ExitCode.OK
