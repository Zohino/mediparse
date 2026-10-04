"""Vstupní bod kroku Snakemake ``prepare_smoketest_subset``: vstupní tabulka smoketestu.

Cesty určuje pravidlo; skript workflow jen rozbalí objekt ``snakemake`` a zavolá
``main``. Pracuje jen se syntetickým korpusem, proto smí běžet kdekoli.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mediparse.application.smoketest_input import SmoketestInput
from mediparse.entrypoints.cli import configure_logging, refusing_invalid_input
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.note_table import ParquetNoteTable
from mediparse.infrastructure.plans_file import PlansFile
from mediparse.infrastructure.synthetic_corpus import CorpusDirectory

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def main(*, plans: Path, corpus: Path, output: Path, log: Path) -> ExitCode:
    """Krok pravidla: diagnostika do souboru logu pravidla.

    Returns:
        Návratový kód kroku.
    """
    configure_logging(log)
    return run(plans=plans, corpus=corpus, output=output)


@refusing_invalid_input
def run(*, plans: Path, corpus: Path, output: Path) -> ExitCode:
    """Složí use case z plánů, korpusu a parquet tabulky a zapíše vstupní tabulku.

    Returns:
        OK po zápisu tabulky; REFUSED pro chybějící nebo neplatné plány
        a korpus, který s plány netvoří páry.
    """
    step = SmoketestInput(
        plans=PlansFile(plans),
        corpus=CorpusDirectory(corpus),
        table=ParquetNoteTable(output),
    )
    count = step.run()
    logger.info("Vstupní tabulka smoketestu: %d zpráv v %s.", count, output)
    return ExitCode.OK
