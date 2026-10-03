"""Vstupní bod vzorkovače plánů syntetického korpusu.

Konzolový skript ``mediparse-sample-plans`` vylosuje z configu plány zpráv a zapíše
je do ``resources/synthetic/plans.jsonl``. Není to krok pipeline: korpus se generuje
jednou a commituje se.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING

from mediparse.application.plan_sampling import PlanSampling
from mediparse.entrypoints.cli import refusing_invalid_input
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.plans_file import (
    LABELS_PATH,
    PLANS_PATH,
    SyntheticPlanFiles,
)
from mediparse.infrastructure.sampler_config import (
    SAMPLER_CONFIG_PATH,
    load_sampler_config,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def main() -> ExitCode:
    """Konzolový skript ``mediparse-sample-plans``.

    Returns:
        Návratový kód vzorkovače.
    """
    return run(sys.argv[1:])


@refusing_invalid_input
def run(argv: Sequence[str]) -> ExitCode:
    """Složí vzorkovač z configu a souboru plánů a spustí ho.

    Returns:
        OK po zapsání plánů, REFUSED pro chybějící nebo neplatný config.
    """
    args = _parser().parse_args(argv)
    store = SyntheticPlanFiles(args.output, args.labels)
    sampling = PlanSampling(store=store, random_source=Random)
    count = sampling.run(load_sampler_config(args.config))
    sys.stdout.write(
        f"Zapsáno {count} plánů do {args.output} a labelů do {args.labels}.\n"
    )
    return ExitCode.OK


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mediparse-sample-plans",
        description="Vzorkovač plánů syntetických propouštěcích zpráv.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=SAMPLER_CONFIG_PATH,
        help="konfigurace vzorkovače",
    )
    parser.add_argument(
        "--output", type=Path, default=PLANS_PATH, help="soubor plánů JSON Lines"
    )
    parser.add_argument(
        "--labels", type=Path, default=LABELS_PATH, help="soubor labelů CSV"
    )
    return parser
