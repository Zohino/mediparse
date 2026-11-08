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
    """Typovaná hranice přípravy překladu a sourozeneckého dělení."""

    _prepare: ModuleType
    _note_parts: ModuleType

    @property
    def model(self) -> str:
        """Identifikátor překladového modelu."""
        return self._prepare.MODEL

    @property
    def revision(self) -> str:
        """Pevná revize překladového modelu."""
        return self._prepare.REVISION

    @property
    def window(self) -> int:
        """Počet tokenů okna překladu."""
        return self._prepare.WINDOW

    @property
    def split_ratio(self) -> float:
        """Rezerva tokenů pro překlad vůči zdroji."""
        return self._prepare.SPLIT_RATIO

    def prompt_ids(self, tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
        """Sestaví prompt stejnou chat šablonou jako příprava překladu.

        Returns:
            Tokeny promptu včetně právě jednoho BOS.
        """
        # PEP 723 příprava je jediným zdrojem chat promptu pro překlad.
        return self._prepare._token_ids(tokenizer, text)  # ruff: ignore[private-member-access]

    def fits(self, tokenizer: PreTrainedTokenizerBase, text: str) -> bool:
        """Ověří okno překladu po maskování celé zprávy.

        Returns:
            Zda se vejde celý text včetně rezervy na překlad.
        """
        # Soukromá funkce skriptu je skutečná kontrola používaná překladem.
        return self._prepare._note_fits(tokenizer, text)  # ruff: ignore[private-member-access]

    def parts(self, tokenizer: PreTrainedTokenizerBase, text: str) -> int | None:
        """Použije přípravu překladu včetně maskování před dělením.

        Returns:
            Počet částí nebo None pro nedělitelný řádek.
        """
        try:
            # PEP 723 příprava je jediným zdrojem pořadí maskování a dělení.
            return len(self._prepare._parts(tokenizer, text))  # ruff: ignore[private-member-access]
        except self._note_parts.UnsplittableError:
            return None


@dataclass(frozen=True)
class ScriptSplitter:
    """Adaptér přesného okna a dělení ze stejné přípravy jako překlad."""

    script: TranslationScript
    tokenizer: PreTrainedTokenizerBase

    def fits(self, text: str) -> bool:
        """Ověří okno původní přípravou překladu.

        Returns:
            Zda se zpráva vejde včetně rezervy.
        """
        return self.script.fits(self.tokenizer, text)

    def parts(self, text: str) -> int | None:
        """Rozdělí zprávu původní přípravou překladu.

        Returns:
            Počet částí nebo None pro nedělitelnou zprávu.
        """
        return self.script.parts(self.tokenizer, text)


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
