"""Načtení samostatných PEP 723 skriptů překladu bez kopírování jejich pravidel."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mediparse.domain.inputs import InvalidInputError

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType

    from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class TranslationScript:
    """Příprava překladu a sourozenecký algoritmus dělení."""

    prepare: ModuleType
    note_parts: ModuleType


@dataclass(frozen=True)
class ScriptSplitter:
    """Adaptér přesného okna a dělení ze stejné přípravy jako překlad."""

    script: TranslationScript
    tokenizer: PreTrainedTokenizerBase

    def fits(self, text: str) -> bool:
        """Použije kontrolu okna v přípravě překladu.

        Returns:
            Zda se zpráva vejde včetně rezervy na překlad.
        """
        return self.script.prepare._fits(self.tokenizer, text)  # ruff: ignore[private-member-access] — stejná kontrola jako v PEP 723 přípravě

    def parts(self, text: str) -> int | None:
        """Použije dělení přípravy překladu.

        Returns:
            Počet částí; None pro nedělitelný řádek.
        """
        try:
            return len(self.script.note_parts.split(text, self.fits))
        except self.script.note_parts.UnsplittableError:
            return None


def load_translation(directory: Path) -> TranslationScript:
    """Načte skripty s jejich importy a obnoví původní stav interpretu.

    Returns:
        Moduly přípravy a dělení, platné i po úklidu sys.modules.

    Raises:
        InvalidInputError: Skripty nejde načíst.
    """
    # Skripty musí běžet i bez mediparse v prostředí translate. Dočasná cesta
    # zachová jejich sourozenecké importy, úklid brání použití jiné kopie skriptů.
    names = ("prepare", "note_parts", "markers")
    previous = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    original_path = sys.path[:]
    try:
        sys.path.insert(0, str(directory.resolve()))
        prepare = _load_prepare(directory)
        return TranslationScript(prepare, sys.modules["note_parts"])
    except (OSError, ImportError, SyntaxError, KeyError) as error:
        msg = f"Skripty překladu nelze načíst: {type(error).__name__}."
        raise InvalidInputError(msg) from None
    finally:
        sys.path[:] = original_path
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(previous)


def _load_prepare(directory: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("prepare", directory / "prepare.py")
    if spec is None or spec.loader is None:
        msg = "Chybí loader translation/prepare.py."
        raise InvalidInputError(msg)
    prepare = importlib.util.module_from_spec(spec)
    sys.modules["prepare"] = prepare
    spec.loader.exec_module(prepare)
    return prepare
