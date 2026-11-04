"""Pravidlo přístupu k datům MIMIC: v jakém prostředí je smí krok číst."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mediparse.domain.inputs import InvalidInputError

if TYPE_CHECKING:
    from collections.abc import Mapping


class BlockingVariable(BaseModel):
    """Proměnná prostředí, jejíž neprázdná hodnota zakazuje čtení MIMIC."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    name: Annotated[str, Field(min_length=1)]
    reason: Annotated[str, Field(min_length=1)]


class DataAccessRules(BaseModel):
    """Pravidla přístupu v pořadí přednosti důvodů odmítnutí."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    blocking: Annotated[tuple[BlockingVariable, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def _unique_names(self) -> Self:
        names = [variable.name for variable in self.blocking]
        if len(names) != len(set(names)):
            msg = "Pravidla přístupu obsahují duplicitní jména proměnných."
            raise ValueError(msg)
        return self


class DataAccessError(InvalidInputError):
    """Prostředí nesmí číst skutečná data MIMIC."""


def blocked_reason(environ: Mapping[str, str], rules: DataAccessRules) -> str | None:
    """Najde první neprázdnou blokující proměnnou.

    Args:
        environ: Proměnné prostředí.
        rules: Sdílená pravidla přístupu.

    Returns:
        Důvod zákazu čtení MIMIC, nebo ``None``, když žádná blokující proměnná neplatí.
    """
    return next(
        (variable.reason for variable in rules.blocking if environ.get(variable.name)),
        None,
    )


def check_data_access(
    environ: Mapping[str, str],
    rules: DataAccessRules,
    *,
    synthetic: bool,
    location: str,
    marker: str,
) -> None:
    """Odmítne čtení skutečných dat v prostředí, které je nesmí číst.

    Args:
        environ: Proměnné prostředí.
        rules: Sdílená pravidla přístupu.
        synthetic: Zda data nesou platnou značku syntetiky.
        location: Místo dat pro hlášku.
        marker: Jméno souboru se značkou syntetiky pro hlášku.

    Raises:
        DataAccessError: Data nejsou syntetická a prostředí je blokuje.
    """
    if synthetic:
        return
    reason = blocked_reason(environ, rules)
    if reason is not None:
        msg = (
            f"Adresář {location} neobsahuje {marker}, "
            f"takže jde o skutečná data. {reason}"
        )
        raise DataAccessError(msg)
