"""Měření předzpracování TF-IDF pro sešit EDA, který modul volá přes reticulate."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Final

from mediparse.domain.clinical_text import (
    find_diagnosis_acronyms,
    normalize_clinical_text,
)
from mediparse.infrastructure.diagnosis_candidates_file import load_text_normalization
from mediparse.infrastructure.sklearn_classifier import tfidf_vectorizer

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

RAW: Final = "raw"
PREPROCESSED: Final = "preprocessed"


def diagnosis_acronyms(candidates: Path) -> list[str]:
    """Vrátí zkratky diagnóz pětice z configu kandidátů.

    Returns:
        Zkratky seřazené abecedně.
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
    """
    normalization = load_text_normalization(candidates)
    raw = tfidf_vectorizer(None).build_analyzer()
    preprocessed = tfidf_vectorizer(normalization).build_analyzer()
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


def note_vocabulary(
    candidates: Path, texts: Sequence[str]
) -> dict[str, list[str] | list[int]]:
    """Vrátí distinct tokeny dávky po variantách s počtem zpráv, které je obsahují.

    Slovník korpusu vzniká sjednocením dávek v databázi: stavový akumulátor by při
    opakovaném volání z R zůstal prázdný, protože ``batched_table`` existující tabulku
    nepřepočítává.

    Args:
        candidates: Cesta k configu kandidátů se zkratkami.
        texts: Texty zpráv dávky.

    Returns:
        Sloupce ``variant`` (``RAW`` nebo ``PREPROCESSED``), ``token`` a ``notes``
        (počet zpráv dávky s tokenem) nejdřív ``RAW``, pak ``PREPROCESSED``, uvnitř podle tokenu, vždy všechny.
    """
    normalization = load_text_normalization(candidates)
    analyzers = {
        RAW: tfidf_vectorizer(None).build_analyzer(),
        PREPROCESSED: tfidf_vectorizer(normalization).build_analyzer(),
    }
    variants: list[str] = []
    tokens: list[str] = []
    notes: list[int] = []
    for variant, analyzer in analyzers.items():
        documents = Counter(token for text in texts for token in set(analyzer(text)))
        for token, count in sorted(documents.items()):
            variants.append(variant)
            tokens.append(token)
            notes.append(count)
    return {"variant": variants, "token": tokens, "notes": notes}
