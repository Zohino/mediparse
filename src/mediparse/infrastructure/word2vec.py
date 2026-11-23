"""Trénink slovních vektorů gensim bez ztráty dlouhých dokumentů."""

from __future__ import annotations

from itertools import batched
from typing import TYPE_CHECKING

from gensim.models import Word2Vec
from gensim.models.word2vec import MAX_WORDS_IN_BATCH

from mediparse.domain.inputs import InvalidInputError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from gensim.models import KeyedVectors

    from mediparse.domain.tuning import ParamValue


def trained_vectors(
    options: Mapping[str, ParamValue], documents: Sequence[list[str]]
) -> KeyedVectors:
    """Naučí slovní vektory se zachováním tokenů dlouhých dokumentů.

    Returns:
        Vektory slov trénovacího slovníku.

    Raises:
        InvalidInputError: Žádné slovo nedosáhne min_count.
    """
    # Gensim slova za 10 000. pozicí věty tiše zahodí; kusy zachovají všechny tokeny.
    sentences = [
        list(chunk)
        for tokens in documents
        for chunk in batched(tokens, MAX_WORDS_IN_BATCH, strict=False)
    ]
    model = Word2Vec(**options)
    model.build_vocab(sentences)
    # Gensim prázdný slovník neodmítne, train by pak skončil nesrozumitelným RuntimeError.
    if not model.wv.index_to_key:
        msg = "Žádné slovo nedosáhlo min_count, slovník word2vec by byl prázdný."
        raise InvalidInputError(msg)
    model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
    return model.wv
