"""Pevné revize encoderů a adaptéry tokenizace Hugging Face."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from typing import TYPE_CHECKING, Final, cast

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.note_tokens import TokenizerInfo

if TYPE_CHECKING:
    from collections.abc import Sequence

    from transformers import PreTrainedTokenizerBase

    from mediparse.infrastructure.translation_script import TranslationScript

ENCODERS: Final = (
    (
        "xlmr",
        "FacebookAI/xlm-roberta-base",
        "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
        512,
    ),
    (
        "mdeberta",
        "microsoft/mdeberta-v3-base",
        "a0484667b22365f84929a935b5e50a51f71f159d",
        512,
    ),
)
OVERHEAD_SENTENCE: Final = "The weather is nice today."


@dataclass(frozen=True)
class HfTokenLengths:
    """Počty tokenů dávky, s explicitní volbou ořezu a speciálních tokenů."""

    tokenizer: PreTrainedTokenizerBase
    special: bool
    strip: bool = False

    def lengths(self, texts: Sequence[str]) -> list[int]:
        """Tokenizuje beze zkracování a výplně.

        Returns:
            Počty v pořadí vstupu, pro prázdný vstup prázdný seznam.

        Raises:
            InvalidInputError: Tokenizér nepřijal vstup.
        """
        if not texts:
            return []
        prepared = [text.strip() if self.strip else text for text in texts]
        try:
            encoded = self.tokenizer(prepared, add_special_tokens=self.special)
            return [len(ids) for ids in encoded["input_ids"]]
        except (ValueError, TypeError) as error:
            msg = f"Tokenizace selhala: {type(error).__name__}."
            raise InvalidInputError(msg) from None


def counter_for(
    info: TokenizerInfo,
    tokenizer: PreTrainedTokenizerBase,
    *,
    parallel: bool,
) -> HfTokenLengths:
    """Určí režim speciálních tokenů a ořezu pro měření zpráv nebo párů.

    Returns:
        Encoder se speciálními tokeny jen mimo páry; překlad vždy po ořezu.
    """
    return HfTokenLengths(
        tokenizer,
        special=info.role == "encoder" and not parallel,
        strip=info.role == "translation",
    )


def load_tokenizer(model: str, revision: str) -> PreTrainedTokenizerBase:
    """Načte tokenizér v pevné revizi bez modelových vah.

    Returns:
        Tokenizér modelu.

    Raises:
        InvalidInputError: Tokenizér nejde načíst.
    """
    # Cache potřebuje jen metadata verzí; odložený import navíc zabrání tomu,
    # aby skops při introspekci načítal nepoužívané obrazové backendy transformers.
    from transformers import AutoTokenizer, logging  # ruff: ignore[import-outside-top-level]

    # Diagnostika knihovny nesmí přidat řetězce do výstupů sešitu.
    logging.set_verbosity_error()
    try:
        return cast(
            "PreTrainedTokenizerBase",
            AutoTokenizer.from_pretrained(model, revision=revision),
        )
    except (OSError, ValueError) as error:
        msg = f"Tokenizér {model}@{revision} nelze načíst: {type(error).__name__}."
        raise InvalidInputError(msg) from None


def special_tokens(tokenizer: PreTrainedTokenizerBase) -> int:
    """Změří počet speciálních tokenů na prázdném textu.

    Returns:
        Délka prázdného vstupu encoderu.
    """
    return HfTokenLengths(tokenizer, special=True).lengths([""])[0]


def prompt_overhead(
    script: TranslationScript, tokenizer: PreTrainedTokenizerBase
) -> int:
    """Odečte délku pevné neklinické věty od jejího chat promptu.

    Returns:
        Režie promptu bez tokenů zdroje.
    """
    # PEP 723 příprava je jediným zdrojem chat promptu pro překlad.
    ids = script.prepare._token_ids(tokenizer, OVERHEAD_SENTENCE)  # ruff: ignore[private-member-access]
    source = HfTokenLengths(tokenizer, special=False, strip=True)
    return len(ids) - source.lengths([OVERHEAD_SENTENCE])[0]


def tokenizer_info(script: TranslationScript) -> list[TokenizerInfo]:
    """Složí revize a okna ještě před načtením tokenizérů.

    Returns:
        Dva encodery a překladový tokenizér, zatím s nulovou režií encoderů.
    """
    transformers_version = version("transformers")
    return [
        *(
            TokenizerInfo(
                key, model, revision, window, "encoder", 0, transformers_version
            )
            for key, model, revision, window in ENCODERS
        ),
        TokenizerInfo(
            "translategemma",
            script.prepare.MODEL,
            script.prepare.REVISION,
            script.prepare.WINDOW,
            "translation",
            0,
            transformers_version,
        ),
    ]


def library_versions() -> tuple[tuple[str, bytes], ...]:
    """Vrátí otisky knihoven, které mohou ovlivnit délky tokenů.

    Returns:
        Pojmenované verze pro klíč cache.
    """
    return (
        ("transformers", version("transformers").encode()),
        ("tokenizers", version("tokenizers").encode()),
    )
