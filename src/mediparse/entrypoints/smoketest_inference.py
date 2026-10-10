"""Vstupní bod kroku Snakemake ``infer_smoketest_model``: uložený model nad odloženými zprávami.

Cesty určuje pravidlo; skript workflow jen rozbalí objekt ``snakemake`` a zavolá
``main``. Pracuje jen s výstupy tréninku smoketestu, proto smí běžet kdekoli.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mediparse.application.smoketest_inference import SmoketestInference
from mediparse.entrypoints.cli import configure_logging, refusing_invalid_input
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.note_table import ParquetNoteTable
from mediparse.infrastructure.prediction_table import ParquetPredictionTable
from mediparse.infrastructure.sklearn_classifier import SkopsModelFile

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def main(
    *, notes: Path, model: Path, predictions: Path, demo: Path, log: Path
) -> ExitCode:
    """Krok pravidla: diagnostika do souboru logu pravidla.

    Returns:
        Návratový kód kroku.
    """
    configure_logging(log)
    return run(notes=notes, model=model, predictions=predictions, demo=demo)


@refusing_invalid_input
def run(*, notes: Path, model: Path, predictions: Path, demo: Path) -> ExitCode:
    """Složí use case z tabulek a uloženého modelu a zapíše ukázky predikcí.

    Returns:
        OK po zápisu ukázek; REFUSED pro chybějící nebo neplatný vstup a pro
        uložený model, který nedává stejné predikce jako trénink.
    """
    step = SmoketestInference(
        table=ParquetNoteTable(notes),
        predictions=ParquetPredictionTable(predictions),
        model=SkopsModelFile(model),
        demo=ParquetPredictionTable(demo),
    )
    examples = step.run()
    logger.info(
        "Uložený model %s dává stejné predikce jako trénink, ukázek: %d.",
        model,
        len(examples),
    )
    return ExitCode.OK
