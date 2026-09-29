"""Vstupní bod brány syntetického korpusu: do repozitáře smí jen korpus, jehož otisk sedí se záznamem auditu.

Konzolový skript ``mediparse-corpus-gate`` běží jako hook prek při commitu i pushi
a jako test v CI. S daty MIMIC nepracuje, proto smí běžet i ve veřejném CI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from mediparse.application.corpus_gate import CorpusGate
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.synthetic_corpus import CORPUS_ROOT, CorpusDirectory

if TYPE_CHECKING:
    from collections.abc import Sequence


def main() -> ExitCode:
    """Konzolový skript ``mediparse-corpus-gate``.

    Returns:
        Návratový kód brány.
    """
    return run(sys.argv[1:])


def run(argv: Sequence[str]) -> ExitCode:
    """Složí bránu nad korpusem z argumentů a přeloží porušení na návratový kód.

    Returns:
        OK, když korpus neexistuje nebo odpovídá auditu, jinak BLOCKED.
    """
    corpus = _parser().parse_args(argv).corpus
    violations = CorpusGate(CorpusDirectory(corpus)).run()
    for violation in violations:
        sys.stderr.write(f"{violation}\n")
    return ExitCode.BLOCKED if violations else ExitCode.OK


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mediparse-corpus-gate",
        description="Brána: korpus v repozitáři odpovídá svému auditu.",
    )
    parser.add_argument(
        "--corpus", type=Path, default=CORPUS_ROOT, help="kořen syntetického korpusu"
    )
    return parser
