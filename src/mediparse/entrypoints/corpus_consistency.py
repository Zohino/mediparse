"""Vstupní bod kontrol shody syntetického korpusu s plány.

Konzolový skript ``mediparse-corpus-check`` vypíše porušení pravidel jedné zprávy
a note_id zpráv k přegenerování. Pracuje jen se syntetickým korpusem, proto smí
běžet kdekoli, i v CI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from mediparse.application.corpus_consistency import CorpusConsistency
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.plans_file import PLANS_PATH, PlansFile
from mediparse.infrastructure.sampler_config import (
    SAMPLER_CONFIG_PATH,
    load_sampler_config,
)
from mediparse.infrastructure.synthetic_corpus import CORPUS_ROOT, CorpusDirectory

if TYPE_CHECKING:
    from collections.abc import Sequence


def main() -> ExitCode:
    """Konzolový skript ``mediparse-corpus-check``.

    Returns:
        Návratový kód kontrol.
    """
    return run(sys.argv[1:])


def run(argv: Sequence[str]) -> ExitCode:
    """Složí kontroly z plánů, korpusu a configu a vypíše zprávy k přegenerování.

    Returns:
        OK, když všechny zprávy odpovídají plánu, jinak BLOCKED.
    """
    args = _parser().parse_args(argv)
    check = CorpusConsistency(
        plans=PlansFile(args.plans), corpus=CorpusDirectory(args.corpus)
    )
    results = check.run(load_sampler_config(args.config))
    for result in results:
        sys.stdout.writelines(
            f"{result.note_id}: {reason}\n" for reason in result.reasons
        )
    if not results:
        sys.stdout.write("Kontroly shody: všechny zprávy odpovídají plánu.\n")
        return ExitCode.OK
    notes = ", ".join(result.note_id for result in results)
    sys.stdout.write(f"Zprávy k přegenerování: {notes}\n")
    return ExitCode.BLOCKED


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mediparse-corpus-check",
        description="Kontroly shody syntetických zpráv s jejich plány.",
    )
    parser.add_argument(
        "--corpus", type=Path, default=CORPUS_ROOT, help="kořen syntetického korpusu"
    )
    parser.add_argument(
        "--plans", type=Path, default=PLANS_PATH, help="soubor plánů JSON Lines"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=SAMPLER_CONFIG_PATH,
        help="konfigurace vzorkovače s hlavičkami a klíčovými slovy",
    )
    return parser
