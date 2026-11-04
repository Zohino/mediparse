"""Čtení vstupního souboru: chybějící soubor i chyba schématu jsou doménová chyba vstupu."""

from __future__ import annotations

from pathlib import Path

import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.input_file import parse_file
from mediparse.synthetic.domain.synthetic_plan import SamplerConfig


def test_missing_file_is_invalid_input(tmp_path: Path) -> None:
    """Chybějící soubor se ohlásí jménem, pydantic ani OSError za adaptér neprosáknou."""
    with pytest.raises(InvalidInputError, match="neexistuje"):
        parse_file(tmp_path / "missing.json", SamplerConfig.model_validate_json)


def test_invalid_content_is_invalid_input(tmp_path: Path) -> None:
    """Obsah, který neodpovídá schématu, se ohlásí jménem souboru."""
    path = tmp_path / "config.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(InvalidInputError, match="neodpovídá schématu"):
        parse_file(path, SamplerConfig.model_validate_json)


@pytest.mark.parametrize("failure", ["utf8", "permission"])
def test_unreadable_file_is_invalid_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Neplatné UTF-8 ani chyba oprávnění neopustí helper jako syrová výjimka."""
    path = tmp_path / "config.json"
    path.write_bytes(b'{"min_chars": 100}\xff')
    if failure == "permission":

        def denied(*_args: object, **_kwargs: object) -> str:
            raise PermissionError

        monkeypatch.setattr(Path, "read_text", denied)

    with pytest.raises(InvalidInputError, match="nelze přečíst") as error:
        parse_file(path, SamplerConfig.model_validate_json)

    assert isinstance(error.value.__cause__, (OSError, UnicodeError))
