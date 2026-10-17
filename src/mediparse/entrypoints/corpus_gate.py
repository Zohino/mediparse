"""Vstupní bod brány syntetického korpusu: do repozitáře smí jen korpus, jehož otisk sedí se záznamem auditu proti připnuté referenci a jehož provenance na tento audit ukazuje.

Konzolový skript ``mediparse-corpus-gate`` běží jako hook prek při commitu i pushi
a jako test v CI. S daty MIMIC nepracuje, jen s jejich oficiálními otisky z configu,
proto smí běžet i ve veřejném CI.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from mediparse.application.corpus_gate import CorpusGate
from mediparse.entrypoints.cli import configure_logging, refusing_invalid_input
from mediparse.entrypoints.corpus_cli import (
    add_corpus_option,
    add_sampler_config_option,
    add_tables_option,
)
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.audit_pin import load_audit_pin
from mediparse.infrastructure.synthetic_corpus import CorpusDirectory

if TYPE_CHECKING:
    from collections.abc import Sequence


def main() -> ExitCode:
    """Konzolový skript ``mediparse-corpus-gate``.

    Returns:
        Návratový kód brány.
    """
    configure_logging()
    return run(sys.argv[1:])


@refusing_invalid_input
def run(argv: Sequence[str]) -> ExitCode:
    """Složí bránu nad korpusem z argumentů a přeloží porušení na návratový kód.

    Returns:
        OK, když korpus neexistuje nebo odpovídá auditu i provenance, jinak BLOCKED;
        REFUSED pro chybějící nebo neplatný config tabulek či vzorkovače.
    """
    args = _parser().parse_args(argv)
    gate = CorpusGate(CorpusDirectory(args.corpus))
    violations = gate.run(
        load_audit_pin(args.tables, args.config),
    )
    sys.stdout.writelines(f"{violation}\n" for violation in violations)
    return ExitCode.BLOCKED if violations else ExitCode.OK


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mediparse-corpus-gate",
        description="Brána: korpus v repozitáři odpovídá svému auditu.",
    )
    add_corpus_option(parser)
    add_tables_option(parser)
    add_sampler_config_option(parser)
    return parser
