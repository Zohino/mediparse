"""Měření předzpracování TF-IDF pro sešit EDA, který modul volá přes reticulate."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Final

from mediparse.domain.clinical_text import (
    TextNormalization,
    find_diagnosis_acronyms,
    normalize_clinical_text,
)
from mediparse.infrastructure.diagnosis_candidates_file import load_text_normalization
from mediparse.infrastructure.sklearn_classifier import tfidf_vectorizer

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

RAW: Final = "raw"
PREPROCESSED: Final = "preprocessed"


def _analyzers(
    normalization: TextNormalization,
) -> dict[str, Callable[[str], list[str]]]:
    return {
        RAW: tfidf_vectorizer(None).build_analyzer(),
        PREPROCESSED: tfidf_vectorizer(normalization).build_analyzer(),
    }


def diagnosis_acronyms(candidates: Path) -> list[str]:
    """Vrátí zkratky diagnóz pětice z configu kandidátů.

    Returns:
        Zkratky seřazené abecedně.

    Raises:
        InvalidInputError: Config kandidátů chybí nebo neodpovídá schématu.
    """
    return sorted(load_text_normalization(candidates).acronyms)


def note_preprocessing(
    candidates: Path, note_ids: Sequence[str], texts: Sequence[str]
) -> dict[str, list[str] | list[int]]:
    """Spočítá tokeny před a po předzpracování a výskyty zkratek na zprávu.

    Tokeny dává analyzér vectorizeru z téže továrny, kterou používá trénink, takže
    měření sleduje i změny ``token_pattern``. Zkratky se hledají v normalizovaném
    textu, protože ``_`` je znak slova a ``___COPD`` by se jinak nenašlo. Text se
    přes most nevrací.

    Args:
        candidates: Cesta k configu kandidátů se zkratkami.
        note_ids: Identifikátory zpráv.
        texts: Texty zpráv ve stejném pořadí jako ``note_ids``.

    Returns:
        Sloupce ``note_id``, ``tokens_raw``, ``tokens_preprocessed`` a jeden sloupec
        na zkratku (klíč je zkratka), vždy všechny, i pro prázdný vstup.

    Raises:
        ValueError: ``note_ids`` a ``texts`` mají různou délku.
        InvalidInputError: Config kandidátů chybí nebo neodpovídá schématu.
    """
    normalization = load_text_normalization(candidates)
    analyzers = _analyzers(normalization)
    raw = analyzers[RAW]
    preprocessed = analyzers[PREPROCESSED]
    ids: list[str] = []
    tokens_raw: list[int] = []
    tokens_preprocessed: list[int] = []
    found: list[Counter[str]] = []
    for note_id, text in zip(note_ids, texts, strict=True):
        ids.append(note_id)
        tokens_raw.append(len(raw(text)))
        tokens_preprocessed.append(len(preprocessed(text)))
        found.append(
            Counter(
                find_diagnosis_acronyms(
                    normalize_clinical_text(text), normalization.acronyms
                )
            )
        )
    counts = {
        acronym: [counter[acronym] for counter in found]
        for acronym in sorted(normalization.acronyms)
    }
    return {
        "note_id": ids,
        "tokens_raw": tokens_raw,
        "tokens_preprocessed": tokens_preprocessed,
        **counts,
    }


def note_vocabulary(candidates: Path, texts: Sequence[str]) -> dict[str, list[str]]:
    """Vrátí distinct tokeny dávky po variantách.

    Slovník korpusu vzniká sjednocením dávek v databázi: stavový akumulátor by při
    opakovaném volání z R zůstal prázdný, protože ``batched_table`` existující tabulku
    nepřepočítává.

    Args:
        candidates: Cesta k configu kandidátů se zkratkami.
        texts: Texty zpráv dávky.

    Returns:
        Sloupce ``variant`` (``RAW`` nebo ``PREPROCESSED``) a ``token``, nejdřív
        ``RAW``, pak ``PREPROCESSED``, uvnitř podle tokenu, vždy oba.

    Raises:
        InvalidInputError: Config kandidátů chybí nebo neodpovídá schématu.
    """
    variants: list[str] = []
    tokens: list[str] = []
    for variant, analyzer in _analyzers(load_text_normalization(candidates)).items():
        distinct = sorted({token for text in texts for token in analyzer(text)})
        variants.extend([variant] * len(distinct))
        tokens.extend(distinct)
    return {"variant": variants, "token": tokens}
