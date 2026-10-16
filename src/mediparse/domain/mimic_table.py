"""Tvar tabulky MIMIC: počet záznamů a jména sloupců, podle kterých se porovnává inventář s převodem."""

from pydantic import BaseModel, ConfigDict


class TableShape(BaseModel):
    """Počet záznamů a sloupce tabulky v pořadí souboru."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    records: int
    columns: tuple[str, ...]
