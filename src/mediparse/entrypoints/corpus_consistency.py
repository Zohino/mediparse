"""Vstupní bod kontrol souladu syntetického korpusu s plány.

Konzolový skript ``mediparse-corpus-check`` vypíše porušení pravidel jedné zprávy
a note_id zpráv k přegenerování; nesoulad počtu značek ___ v překladu hlásí jako
upozornění, které neblokuje. Pracuje jen se syntetickým korpusem, proto smí
běžet kdekoli, i v CI.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from mediparse.application.corpus_consistency import CorpusConsistency
from mediparse.entrypoints.cli import configure_logging, refusing_invalid_input
from mediparse.entrypoints.corpus_cli import (
    add_corpus_option,
    add_plans_option,
    add_sampler_config_option,
)
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.plans_file import PlansFile
from mediparse.infrastructure.sampler_config import (
    load_sampler_config,
)
from mediparse.infrastructure.synthetic_corpus import CorpusDirectory

if TYPE_CHECKING:
    from collections.abc import Sequence


def main() -> ExitCode:
    """Konzolový skript ``mediparse-corpus-check``.

    Returns:
        Návratový kód kontrol.
    """
    configure_logging()
    return run(sys.argv[1:])


@refusing_invalid_input
def run(argv: Sequence[str]) -> ExitCode:
    """Složí kontroly z plánů, korpusu a configu a vypíše zprávy k přegenerování.

    Returns:
        OK, když zprávy odpovídají plánům, neprázdný korpus modelu a překlad originálu
        (upozornění neblokují), jinak BLOCKED;
        REFUSED pro chybějící nebo neplatný vstupní soubor.
    """
    args = _parser().parse_args(argv)
    check = CorpusConsistency(
        plans=PlansFile(args.plans), corpus=CorpusDirectory(args.corpus)
    )
    report = check.run(load_sampler_config(args.config))
    for result in report.notes:
        sys.stdout.writelines(
            f"{result.note_id}: {reason}\n" for reason in result.reasons
        )
    sys.stdout.writelines(f"Korpus: {reason}\n" for reason in report.corpus)
    sys.stdout.writelines(f"Upozornění: {notice}\n" for notice in report.notices)
    if not report.notes and not report.corpus:
        sys.stdout.write("Kontroly souladu: korpus odpovídá plánům i modelu.\n")
        return ExitCode.OK
    if report.notes:
        notes = ", ".join(result.note_id for result in report.notes)
        sys.stdout.write(f"Zprávy k přegenerování: {notes}\n")
    return ExitCode.BLOCKED


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mediparse-corpus-check",
        description="Kontroly souladu syntetických zpráv s jejich plány.",
    )
    add_corpus_option(parser)
    add_plans_option(parser)
    add_sampler_config_option(parser)
    return parser
