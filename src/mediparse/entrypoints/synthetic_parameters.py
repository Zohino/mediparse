"""Vstupní bod převzetí parametrů pětice z exportu sešitu 8 do configu vzorkovače.

Konzolový skript ``mediparse-synthetic-parameters`` nahradí v configu prevalence,
matici společného výskytu, zmínky diagnóz a počet zpráv. Není to krok pipeline:
spouští ho autor korpusu po renderu sešitu 8.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from mediparse.application.synthetic_parameters import (
    ParametersRefused,
    ParameterTransfer,
)
from mediparse.entrypoints.cli import configure_logging
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.sampler_config import (
    SAMPLER_CONFIG_PATH,
    SamplerConfigFile,
)
from mediparse.infrastructure.synthetic_parameters_file import (
    PARAMETERS_PATH,
    SyntheticParametersFile,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


def main() -> ExitCode:
    """Konzolový skript ``mediparse-synthetic-parameters``.

    Returns:
        Návratový kód převzetí.
    """
    configure_logging()
    return run(sys.argv[1:])


def run(argv: Sequence[str]) -> ExitCode:
    """Složí převzetí z exportu a configu a spustí ho.

    Returns:
        OK po zápisu configu, REFUSED s důvodem v logu, když se config nezměnil.
    """
    args = _parser().parse_args(argv)
    transfer = ParameterTransfer(
        export=SyntheticParametersFile(args.export),
        config=SamplerConfigFile(args.config),
    )
    result = transfer.run()
    if isinstance(result, ParametersRefused):
        logger.error("Parametry se nepřevzaly: %s", result.reason)
        return ExitCode.REFUSED
    logger.info("Config %s přepsán, patients.notes = %d.", args.config, result.notes)
    return ExitCode.OK


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mediparse-synthetic-parameters",
        description="Převzetí parametrů pětice z exportu sešitu 8 do configu vzorkovače.",
    )
    parser.add_argument(
        "--export", type=Path, default=PARAMETERS_PATH, help="export sešitu 8 (JSON)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=SAMPLER_CONFIG_PATH,
        help="konfigurace vzorkovače",
    )
    return parser
