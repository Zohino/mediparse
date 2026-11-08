"""Vstupní bod kroku Snakemake ``infer_model``: uložený model nad odloženými zprávami.

Cesty určuje pravidlo; skript workflow jen rozbalí objekt ``snakemake`` a zavolá
``main``. Čte kohortu a uložený model, proto běží tam, kde jsou data.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Final

from mediparse.application.model_inference import ModelInference
from mediparse.domain.costs import InferenceCosts, LatencyPlan
from mediparse.entrypoints.cli import configure_logging, refusing_invalid_input
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.entrypoints.training_input import training_input
from mediparse.infrastructure.data_access_guard import check_inputs_access
from mediparse.infrastructure.json_file import write_model
from mediparse.infrastructure.machine_hardware import machine_hardware
from mediparse.infrastructure.nvml_gpu import gpu_monitor
from mediparse.infrastructure.perf_clock import PerfCounterClock
from mediparse.infrastructure.prediction_table import ParquetPredictionTable
from mediparse.infrastructure.sklearn_classifier import SkopsModelFile
from mediparse.infrastructure.training_config import load_training_config

if TYPE_CHECKING:
    from collections.abc import Mapping
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
        environ=os.environ,
        cohort=cohort,
        config=config,
        lexicon=lexicon,
        model=model,
        predictions=predictions,
        inference=inference,
        costs=costs,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        cuda_device_order=os.environ.get("CUDA_DEVICE_ORDER"),
    )


@refusing_invalid_input
def run(
    *,
    environ: Mapping[str, str],
    cohort: Path,
    config: Path,
    lexicon: Path,
    model: Path,
    predictions: Path,
    inference: Path,
    costs: Path,
    cuda_visible_devices: str | None,
    cuda_device_order: str | None,
) -> ExitCode:
    """Složí inferenci z adaptérů, zapíše predikce a agregované náklady s HW stroje.

    Returns:
        OK po zápisu predikcí a nákladů; REFUSED pro neplatný vstup, nesoulad
        predikcí uloženého modelu s tréninkem, nezjistitelný HW, selhání měření GPU
        a pro neoznačená data v prostředí, které nesmí číst MIMIC.
    """
    check_inputs_access(environ, (cohort, model, predictions))
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
    with gpu_monitor(cuda_visible_devices, cuda_device_order) as monitor:
        hardware = machine_hardware(monitor.devices)
        inferred = step.run()
    write_model(
        costs,
        InferenceCosts(
            plan=LATENCY_PLAN,
            latency=inferred.latency,
            batch=inferred.batch,
            hardware=hardware,
            gpu=monitor.measurement(),
        ),
    )
    logger.info(
        "Uložený model %s dává stejné predikce jako trénink, predikcí: %d.",
        model,
        len(inferred.predictions),
    )
    return ExitCode.OK
