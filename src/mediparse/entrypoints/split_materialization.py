"""Vstupní bod kroku Snakemake ``make_splits``: sdílený split pacientů zpráv.

Cesty určuje pravidlo; skript workflow jen rozbalí objekt ``snakemake`` a zavolá
``main``. Krok nic netrénuje, proto nezapisuje run manifest: seed, počet foldů
a otisk jsou v souboru a odtud v manifestech kroků, které split čtou.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mediparse.application.split_materialization import SplitMaterialization
from mediparse.entrypoints.cli import configure_logging, refusing_invalid_input
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.note_table import ParquetNoteTable
from mediparse.infrastructure.sklearn_split import StratifiedGroupFolds
from mediparse.infrastructure.split_config import load_split_config
from mediparse.infrastructure.split_table import ParquetSplitTable

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def main(*, cohort: Path, config: Path, output: Path, log: Path) -> ExitCode:
    """Krok pravidla: diagnostika do souboru logu pravidla.

    Returns:
        Návratový kód kroku.
    """
    configure_logging(log)
    return run(cohort=cohort, config=config, output=output)


@refusing_invalid_input
def run(*, cohort: Path, config: Path, output: Path) -> ExitCode:
    """Složí use case z tabulky, configu a adaptérů a zapíše soubor splitu.

    Returns:
        OK po zápisu splitu; REFUSED pro chybějící nebo neplatnou tabulku
        a config a pro méně pacientů, než je foldů.
    """
    step = SplitMaterialization(
        table=ParquetNoteTable(cohort),
        splitter=StratifiedGroupFolds(),
        sink=ParquetSplitTable(output),
    )
    shared = step.run(load_split_config(config))
    logger.info(
        "Split do %d foldů, pacientů na fold: %s, uložen do %s.",
        shared.config.folds,
        [len(subjects) for subjects in shared.split.folds],
        output,
    )
    return ExitCode.OK
