"""Tabulka MIMIC: jméno odvozené ze staženého souboru a tvar, podle kterého se porovnává inventář s převodem."""

from typing import Final

from pydantic import BaseModel, ConfigDict

from mediparse.domain.inputs import InvalidInputError

CSV_SUFFIXES: Final = (".csv.gz", ".csv")


class TableShape(BaseModel):
    """Počet záznamů a sloupce tabulky v pořadí souboru."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    records: int
    columns: tuple[str, ...]


def table_name(file_name: str) -> str:
    """Odvodí jméno tabulky ze jména souboru CSV, komprimovaného i prostého.

    Returns:
        Jméno souboru bez přípony CSV.

    Raises:
        InvalidInputError: Soubor nekončí žádnou z přípon `CSV_SUFFIXES`.
    """
    for suffix in CSV_SUFFIXES:
        if file_name.endswith(suffix):
            return file_name.removesuffix(suffix)
    msg = f"Soubor {file_name} není CSV ani CSV v gzipu ({', '.join(CSV_SUFFIXES)})."
    raise InvalidInputError(msg)
