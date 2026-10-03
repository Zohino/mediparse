"""Vstupní bod zadání verbalizace: vypíše přesný text, který dostane model pro jednu zprávu.

Konzolový skript ``mediparse-verbalization-prompt`` pracuje jen s plány, configem
a šablonou v repu; data MIMIC nečte.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from mediparse.application.verbalization_prompt import VerbalizationPrompt
from mediparse.entrypoints.cli import refusing_invalid_input
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.plans_file import PLANS_PATH, PlansFile
from mediparse.infrastructure.sampler_config import (
    SAMPLER_CONFIG_PATH,
    load_sampler_config,
)
from mediparse.infrastructure.verbalization_template import (
    VERBALIZATION_TEMPLATE_PATH,
    load_verbalization_template,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def main() -> ExitCode:
    """Konzolový skript ``mediparse-verbalization-prompt``.

    Returns:
        Návratový kód.
    """
    return run(sys.argv[1:])


@refusing_invalid_input
def run(argv: Sequence[str]) -> ExitCode:
    """Složí zadání pro note_id z plánů, configu a šablony a vypíše ho.

    Returns:
        OK po vypsání zadání, REFUSED pro neznámé note_id nebo chybějící či neplatný
        vstupní soubor.
    """
    args = _parser().parse_args(argv)
    prompt = VerbalizationPrompt(PlansFile(args.plans)).run(
        args.note_id,
        load_verbalization_template(args.template),
        load_sampler_config(args.config),
    )
    if prompt is None:
        sys.stderr.write(f"Plán pro note_id {args.note_id} neexistuje.\n")
        return ExitCode.REFUSED
    sys.stdout.write(prompt)
    return ExitCode.OK


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mediparse-verbalization-prompt",
        description="Zadání verbalizace jedné syntetické zprávy.",
    )
    parser.add_argument("note_id", help="note_id zprávy, například 90000001-DS-1")
    parser.add_argument(
        "--plans", type=Path, default=PLANS_PATH, help="soubor plánů JSON Lines"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=SAMPLER_CONFIG_PATH,
        help="konfigurace vzorkovače",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=VERBALIZATION_TEMPLATE_PATH,
        help="šablona instrukcí verbalizace",
    )
    return parser
