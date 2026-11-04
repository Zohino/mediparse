"""Načtení sdíleného pravidla přístupu z dat balíčku."""

from importlib.resources import files

from pydantic import ValidationError

from mediparse.domain.data_access import DataAccessRules
from mediparse.domain.inputs import InvalidInputError


def load_data_access_rules() -> DataAccessRules:
    """Přečte a ověří pravidla, která čtou Python i R.

    Returns:
        Validovaná pravidla v pořadí přednosti důvodů.

    Raises:
        InvalidInputError: Soubor nelze přečíst nebo neodpovídá schématu.
    """
    try:
        content = (
            files("mediparse.domain")
            .joinpath("data_access.json")
            .read_text(encoding="utf-8")
        )
        return DataAccessRules.model_validate_json(content)
    except (OSError, UnicodeError, ValidationError) as error:
        msg = "Soubor data_access.json nelze přečíst nebo neodpovídá schématu."
        raise InvalidInputError(msg) from error
