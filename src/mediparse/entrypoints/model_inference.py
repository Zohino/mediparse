"""Vstupní bod kroku Snakemake ``infer_model``: uložený model nad odloženými zprávami.

Cesty určuje pravidlo; skript workflow jen rozbalí objekt ``snakemake`` a zavolá
``main``. Čte kohortu a uložený model, proto běží tam, kde jsou data.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mediparse.application.model_inference import ModelInference
from mediparse.entrypoints.cli import configure_logging, refusing_invalid_input
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.entrypoints.training_input import training_input
from mediparse.infrastructure.prediction_table import ParquetPredictionTable
from mediparse.infrastructure.sklearn_classifier import SkopsModelFile
from mediparse.infrastructure.training_config import load_training_config

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def main(
    *,
    cohort: Path,
    config: Path,
    lexicon: Path,
    model: Path,
    predictions: Path,
    inference: Path,
    log: Path,
) -> ExitCode:
    """Krok pravidla: diagnostika do souboru logu pravidla.

    Returns:
        Návratový kód kroku.
    """
    configure_logging(log)
    return run(
        cohort=cohort,
        config=config,
        lexicon=lexicon,
        model=model,
        predictions=predictions,
        inference=inference,
    )


@refusing_invalid_input
def run(
    *,
    cohort: Path,
    config: Path,
    lexicon: Path,
    model: Path,
    predictions: Path,
    inference: Path,
) -> ExitCode:
    """Složí use case z tabulky podle configu tréninku a uloženého modelu a zapíše predikce načteného modelu.

    Returns:
        OK po zápisu predikcí; REFUSED pro chybějící nebo neplatný vstup a pro
        uložený model, který nedává stejné predikce jako trénink.
    """
    table, _ = training_input(cohort, load_training_config(config), lexicon)
    step = ModelInference(
        table=table,
        predictions=ParquetPredictionTable(predictions),
        model=SkopsModelFile(model),
        inference=ParquetPredictionTable(inference),
    )
    inferred = step.run()
    logger.info(
        "Uložený model %s dává stejné predikce jako trénink, predikcí: %d.",
        model,
        len(inferred),
    )
    return ExitCode.OK
