"""Vstupní bod brány syntetického korpusu: do repozitáře smí jen korpus, jehož otisk sedí se záznamem auditu proti připnuté referenci a jehož provenance na tento audit ukazuje.

Konzolový skript ``mediparse-corpus-gate`` běží jako hook prek při commitu i pushi
a jako test v CI. S daty MIMIC nepracuje, jen s jejich oficiálními otisky z configu,
proto smí běžet i ve veřejném CI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from mediparse.application.corpus_gate import CorpusGate
from mediparse.entrypoints.cli import refusing_invalid_input
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.mimic_tables import TABLES_PATH, load_reference_sha256
from mediparse.infrastructure.sampler_config import (
    SAMPLER_CONFIG_PATH,
    load_sampler_config,
)
from mediparse.infrastructure.synthetic_corpus import CORPUS_ROOT, CorpusDirectory

if TYPE_CHECKING:
    from collections.abc import Sequence


def main() -> ExitCode:
    """Konzolový skript ``mediparse-corpus-gate``.

    Returns:
        Návratový kód brány.
    """
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
        load_reference_sha256(args.tables),
        load_sampler_config(args.config).structure.structure_labels,
    )
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
    parser.add_argument(
        "--tables",
        type=Path,
        default=TABLES_PATH,
        help="config tabulek MIMIC s oficiálními otisky",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=SAMPLER_CONFIG_PATH,
        help="konfigurace vzorkovače se štítky struktury zprávy",
    )
    return parser
