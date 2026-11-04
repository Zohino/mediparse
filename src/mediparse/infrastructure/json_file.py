"""Sdílený formát JSON souborů: zápis modelu či slovníku a čtení JSON Lines."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from mediparse.domain.inputs import InvalidInputError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from pydantic import BaseModel


def write_model(path: Path, model: BaseModel) -> None:
    """Zapíše model jako JSON s odsazením 2 a koncovým řádkem v UTF-8 s LF.

    Chybějící nadřazený adresář založí a existující soubor přepíše.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"{model.model_dump_json(indent=2)}\n".encode())


def parse_lines[T: BaseModel](model: type[T]) -> Callable[[str], tuple[T, ...]]:
    """Vytvoří parser JSON Lines pro ``parse_file``.

    Prázdné řádky a řádky ze samých mezer nenesou záznam, proto se přeskočí.

    Returns:
        Funkce, která převede text na modely v pořadí řádků.
    """

    def parse(content: str) -> tuple[T, ...]:
        # splitlines() láme i na U+0085 a U+2028, které JSON v řetězci připouští.
        return tuple(
            model.model_validate_json(line)
            for line in content.split("\n")
            if line.strip()
        )

    return parse


def write_mapping(path: Path, values: Mapping[str, str]) -> None:
    """Zapíše položky JSON v pořadí slovníku v UTF-8 s koncovým řádkem.

    Raises:
        InvalidInputError: Výstup nelze zapsat.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(dict(values), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        msg = f"Soubor {path} nelze zapsat."
        raise InvalidInputError(msg) from error
