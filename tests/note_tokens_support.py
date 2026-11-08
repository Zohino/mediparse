"""Lokální tokenizér a zkonstruované vstupy pro kontrakt tokenových artefaktů."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split, Whitespace, WhitespaceSplit
from tokenizers.processors import TemplateProcessing

from tests.support import REPOSITORY

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedTokenizerFast

UNKNOWN = "<unk>"
BOS = "<bos>"
EOS = "<eos>"

NOTES = (("n1", "a b c"), ("n2", ""), ("n3", "  a b \n"), ("n4", "a\n\nb\n\nc"))


def copy_prepare(directory: Path, *, window: int, split_ratio: float) -> Path:
    """Zkopíruje přípravu s testovacím oknem a ověří změnu obou deklarací.

    Returns:
        Cesta upraveného prepare.py ve složce se sourozeneckými skripty.
    """
    source = (REPOSITORY / "translation/prepare.py").read_bytes()
    replacements = {
        b"WINDOW: Final = 2048": f"WINDOW: Final = {window}".encode(),
        b"SPLIT_RATIO: Final = 2.2": f"SPLIT_RATIO: Final = {split_ratio}".encode(),
    }
    for original, replacement in replacements.items():
        assert source.count(original) == 1, original
        source = source.replace(original, replacement)
        assert replacement in source
    path = directory / "prepare.py"
    path.write_bytes(source)
    return path


def word_tokenizer() -> PreTrainedTokenizerFast:
    """Vytvoří tokenizér se dvěma speciálními tokeny bez souborů a sítě.

    Returns:
        WordLevel s chat režií BOS, x a y, tedy tři tokeny.
    """
    from transformers import PreTrainedTokenizerFast  # ruff: ignore[import-outside-top-level]

    backend = Tokenizer(
        WordLevel(
            {
                "<unk>": 0,
                "<bos>": 1,
                "<eos>": 2,
                "a": 3,
                "b": 4,
                "c": 5,
                "x": 6,
                "y": 7,
            },
            unk_token=UNKNOWN,
        )
    )
    backend.pre_tokenizer = WhitespaceSplit()
    backend.post_processor = TemplateProcessing(
        single="<bos> $A <eos>", special_tokens=[("<bos>", 1), ("<eos>", 2)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token=BOS,
        eos_token=EOS,
        unk_token=UNKNOWN,
        chat_template="{{ bos_token }}x {{ messages[0]['content'][0]['text'] }} y",
    )


@dataclass(frozen=True)
class FakeSplitter:
    """Ruční orákulum: nejvýš dvě slova, jinak dělení po prázdných řádcích."""

    capacity: int = 2

    def fits(self, text: str) -> bool:
        """Vrátí pravdu nejvýš pro dvě slova.

        Returns:
            Zda se vejde celý text.
        """
        return len(text.split()) <= self.capacity

    def parts(self, text: str) -> int | None:
        """Vrátí počet bloků jen tehdy, pokud se každý vejde.

        Returns:
            Počet částí nebo None pro nedělitelný řádek.
        """
        if self.fits(text):
            return 1
        blocks = text.split("\n\n")
        return len(blocks) if all(self.fits(block) for block in blocks) else None


def marker_tokenizer() -> PreTrainedTokenizerFast:
    """Rozliší růst značky z jednoho tokenu ___ na tři tokeny [[n]].

    Returns:
        WordLevel s chat režií tří tokenů a dělením slov a interpunkce.
    """
    tokenizer = word_tokenizer()
    tokenizer.backend_tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


def whitespace_tokenizer() -> PreTrainedTokenizerFast:
    """Rozliší nový řádek jako token od mezer oddělujících slova.

    Returns:
        WordLevel, kde dvě slova a nový řádek mají bez ořezu tři tokeny.
    """
    tokenizer = word_tokenizer()
    tokenizer.backend_tokenizer.pre_tokenizer = Split(" ", behavior="removed")
    return tokenizer
