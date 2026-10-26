"""Načtení configu zmínek EDA (negace a příznaky) do domény."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.text_mentions_file import load_text_mentions
from tests.support import REPOSITORY_TEXT_MENTIONS

if TYPE_CHECKING:
    from pathlib import Path

CUE_COUNT = 11
WINDOW = 40
SYMPTOM_COUNT = 20


def _write(tmp_path: Path, tree: object) -> Path:
    path = tmp_path / "mentions.json"
    path.write_text(json.dumps(tree), encoding="utf-8")
    return path


def _tree(
    symptoms: list[str], cues: list[str] | None = None, **extra: object
) -> dict[str, object]:
    return {
        "negation": {"cues": cues or ["no"], "window_chars": WINDOW},
        "symptoms": {"keywords_en": symptoms},
        **extra,
    }


def test_repository_config_loads() -> None:
    """Config v repu: 11 negačních výrazů, okno 40 znaků a 20 příznaků."""
    mentions = load_text_mentions(REPOSITORY_TEXT_MENTIONS)

    assert len(mentions.negation.cues) == CUE_COUNT
    assert mentions.negation.cues[0] == "no"
    assert mentions.negation.window_chars == WINDOW
    assert len(mentions.symptoms) == SYMPTOM_COUNT


def test_missing_file_is_invalid_input(tmp_path: Path) -> None:
    """Chybějící soubor je chyba vstupu."""
    with pytest.raises(InvalidInputError):
        load_text_mentions(tmp_path / "none.json")


def test_invalid_json_is_invalid_input(tmp_path: Path) -> None:
    """Neplatný JSON je chyba vstupu."""
    path = tmp_path / "broken.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(InvalidInputError):
        load_text_mentions(path)


def test_empty_symptoms_are_invalid_input(tmp_path: Path) -> None:
    """Bez příznaku by se rozvržení zmínek nemělo z čeho skládat."""
    with pytest.raises(InvalidInputError):
        load_text_mentions(_write(tmp_path, _tree([])))


def test_unknown_key_is_invalid_input(tmp_path: Path) -> None:
    """Neznámý klíč nejvyšší úrovně je chyba, ne tiše ignorované pole."""
    with pytest.raises(InvalidInputError):
        load_text_mentions(_write(tmp_path, _tree(["fever"], other=1)))


def test_single_symptom_is_accepted(tmp_path: Path) -> None:
    """Dolní mez je jeden příznak a jeho hodnota se zachová."""
    mentions = load_text_mentions(_write(tmp_path, _tree(["fever"])))

    assert mentions.symptoms == ("fever",)


@pytest.mark.parametrize("blank", ["", "  ", "\t"])
def test_blank_symptom_is_invalid_input(tmp_path: Path, blank: str) -> None:
    """Prázdný nebo bílý příznak by našel každý text, proto je chyba vstupu."""
    with pytest.raises(InvalidInputError):
        load_text_mentions(_write(tmp_path, _tree(["fever", blank])))


@pytest.mark.parametrize("blank", ["", "  ", "\t"])
def test_blank_negation_cue_is_invalid_input(tmp_path: Path, blank: str) -> None:
    """Prázdný nebo bílý negační výraz by posunul stavy zmínek, proto je chyba vstupu."""
    with pytest.raises(InvalidInputError):
        load_text_mentions(_write(tmp_path, _tree(["fever"], cues=["no", blank])))
