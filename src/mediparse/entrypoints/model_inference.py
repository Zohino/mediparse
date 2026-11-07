"""Vstupní bod kroku Snakemake ``infer_model``: uložený model nad odloženými zprávami.

Cesty určuje pravidlo; skript workflow jen rozbalí objekt ``snakemake`` a zavolá
``main``. Čte kohortu a uložený model, proto běží tam, kde jsou data.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

from mediparse.application.model_inference import ModelInference
from mediparse.domain.costs import GpuMeasurement, InferenceCosts, LatencyPlan
from mediparse.entrypoints.cli import configure_logging, refusing_invalid_input
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.entrypoints.training_input import training_input
from mediparse.infrastructure.json_file import write_model
from mediparse.infrastructure.machine_hardware import machine_hardware
from mediparse.infrastructure.perf_clock import PerfCounterClock
from mediparse.infrastructure.prediction_table import ParquetPredictionTable
from mediparse.infrastructure.sklearn_classifier import SkopsModelFile
from mediparse.infrastructure.training_config import load_training_config

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

LATENCY_PLAN: Final = LatencyPlan(warmup=10, samples=100)


def main(
    *,
    cohort: Path,
    config: Path,
    lexicon: Path,
    model: Path,
    predictions: Path,
    inference: Path,
    costs: Path,
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
        costs=costs,
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
    costs: Path,
) -> ExitCode:
    """Složí inferenci z adaptérů, zapíše predikce a agregované náklady s HW stroje.

    Returns:
        OK po zápisu predikcí a nákladů; REFUSED pro chybějící nebo neplatný vstup a pro
        uložený model, který nedává stejné predikce jako trénink, nebo nezjistitelný HW stroje.
    """
    table, _ = training_input(
        cohort=cohort, training=load_training_config(config), lexicon=lexicon
    )
    step = ModelInference(
        table=table,
        predictions=ParquetPredictionTable(predictions),
        model=SkopsModelFile(model),
        inference=ParquetPredictionTable(inference),
        clock=PerfCounterClock(),
        plan=LATENCY_PLAN,
    )
    inferred = step.run()
    write_model(
        costs,
        InferenceCosts(
            plan=LATENCY_PLAN,
            latency=inferred.latency,
            batch=inferred.batch,
            hardware=machine_hardware(gpus=()),
            gpu=GpuMeasurement(seconds=0.0, devices=()),
        ),
    )
    logger.info(
        "Uložený model %s dává stejné predikce jako trénink, predikcí: %d.",
        model,
        len(inferred.predictions),
    )
    return ExitCode.OK
