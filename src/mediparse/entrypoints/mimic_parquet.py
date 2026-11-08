"""Vstupní bod pravidla Snakemake ``parquet_mimic_table``: validovaná tabulka MIMIC z CSV na parquet.

Cesty jsou vstupy a výstupy pravidla; skript workflow jen rozbalí objekt
``snakemake`` a zavolá ``main``. Nesoulad otisku CSV, počtu záznamů nebo sloupců s inventářem
validace ukončí krok kódem REFUSED a Snakemake rozpracovaný výstup smaže.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from mediparse.application.mimic_parquet import MimicParquetConversion
from mediparse.domain.inputs import InvalidInputError
from mediparse.entrypoints.cli import configure_logging, refusing_invalid_input
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.csv_parquet import CsvParquetConversion
from mediparse.infrastructure.data_access_guard import check_inputs_access
from mediparse.infrastructure.file_digest import file_sha256
from mediparse.infrastructure.mimic_inventory import InventoryFile
from mediparse.infrastructure.table_provenance import write_table_provenance

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

logger = logging.getLogger(__name__)


def main(*, table: Path, inventory: Path, parquet: Path, log: Path) -> ExitCode:
    """Krok pravidla: diagnostika do logu kroku, chyby i na stderr.

    Returns:
        Návratový kód kroku.
    """
    configure_logging(log)
    return run(environ=os.environ, table=table, inventory=inventory, parquet=parquet)


@refusing_invalid_input
def run(
    *, environ: Mapping[str, str], table: Path, inventory: Path, parquet: Path
) -> ExitCode:
    """Převede tabulku na parquet a ověří ji proti inventáři validace.

    Returns:
        OK po ověřeném převodu; REFUSED pro chybějící nebo rozbitý vstup
        a pro otisk CSV nebo tvar, který se liší od inventáře,
        a pro neoznačená data v prostředí, které nesmí číst MIMIC.

    Raises:
        InvalidInputError: Rozdíl v otisku CSV dekorátor přeloží na REFUSED.
    """
    check_inputs_access(environ, (table,))
    inventory_file = InventoryFile(inventory)
    shape = MimicParquetConversion(
        inventory_file, CsvParquetConversion(table, parquet)
    ).run()
    source_sha256 = inventory_file.sha256()
    # CSV mohlo být vyměněno se starším mtime, kdy Snakemake inventář znovu nevytvoří.
    if file_sha256(table) != source_sha256:
        msg = f"Soubor {table.name} se od validace změnil; spusťte validaci znovu."
        raise InvalidInputError(msg)
    write_table_provenance(parquet, source=table.name, source_sha256=source_sha256)
    logger.info("%s: %d záznamů v %s", table.name, shape.records, parquet)
    return ExitCode.OK
