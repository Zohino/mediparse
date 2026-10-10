"""Společné chování konzolových skriptů: chybný vstupní soubor a argumenty validované doménou."""

from __future__ import annotations

import argparse
import functools
import logging
from typing import TYPE_CHECKING

from mediparse.domain.inputs import InvalidInputError
from mediparse.entrypoints.exit_code import ExitCode

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

logger = logging.getLogger(__name__)


def configure_logging(log: Path | None = None) -> None:
    """Diagnostika ve formátu čas, úroveň, zpráva; knihovny až od WARNING.

    Args:
        log: Soubor logu kroku Snakemake; chyby jdou navíc na stderr. Bez něj jde
            veškerá diagnostika na stderr.
    """
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s", filename=log, encoding="utf-8"
    )
    if log is not None:
        console = logging.StreamHandler()
        console.setLevel(logging.ERROR)
        console.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        logging.getLogger().addHandler(console)
    logging.getLogger("mediparse").setLevel(logging.INFO)


def refusing_invalid_input[**P](run: Callable[P, ExitCode]) -> Callable[P, ExitCode]:
    """Chybějící nebo neplatný vstupní soubor ukončí příkaz hláškou a kódem REFUSED.

    Returns:
        Vstupní bod se stejnými argumenty.
    """

    @functools.wraps(run)
    def guarded(*args: P.args, **kwargs: P.kwargs) -> ExitCode:
        try:
            return run(*args, **kwargs)
        except InvalidInputError as error:
            logger.error("%s", error)
            return ExitCode.REFUSED

    return guarded


def argument[T](parse: Callable[[str], T]) -> Callable[[str], T]:
    """Typ argumentu pro argparse z doménové validace; hláška domény jde do chyby argparse.

    Returns:
        Funkce pro ``type=`` u ``add_argument``.
    """

    def checked(value: str) -> T:
        try:
            return parse(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(str(error)) from error

    return checked
