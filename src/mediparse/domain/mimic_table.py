"""Tabulka MIMIC: jméno, tvar pro porovnání inventáře s převodem a původ převedené tabulky."""

from typing import Final

from pydantic import BaseModel, ConfigDict

from mediparse.domain.digest import Sha256
from mediparse.domain.inputs import InvalidInputError

CSV_SUFFIXES: Final = (".csv.gz", ".csv")


class TableShape(BaseModel):
    """Počet záznamů a sloupce tabulky v pořadí souboru."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    records: int
    columns: tuple[str, ...]


class TableProvenance(BaseModel):
    """Zdroj tabulky a vazba otisku ověřeného CSV na otisk převedeného parquetu.

    Attributes:
        source: Jméno zdrojového CSV, u syntetiky URL na doméně `.invalid`.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    source: str
    source_sha256: Sha256
    sha256: Sha256


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
