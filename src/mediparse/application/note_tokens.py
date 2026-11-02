"""Měření celých tabulek, přesného dělení vzorku a paralelní syntetiky."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from mediparse.domain.note_tokens import NoteTokens, ParallelTokens, SplitSample

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mediparse.application.ports.corpus import NoteTexts


class TokenLengths(Protocol):
    """Počítadlo délek dávky textů v pořadí vstupu."""

    def lengths(self, texts: Sequence[str]) -> list[int]:
        """Vrátí jednu délku na vstupní text."""
        ...


class TranslationSplitter(Protocol):
    """Přesné rozhodnutí o okně a počet částí překladu."""

    def fits(self, text: str) -> bool:
        """Určí, zda se zpráva vejde bez dělení."""
        ...

    def parts(self, text: str) -> int | None:
        """Vrátí počet částí, nebo None pro nedělitelnou zprávu."""
        ...


class NoteTokenSink(Protocol):
    """Zápis faktů o tokenech, vzorku a syntetických párech."""

    def notes(self, table: str, rows: Sequence[NoteTokens]) -> None:
        """Zapíše dávku délek zpráv dané tabulky."""
        ...

    def sample(self, rows: Sequence[SplitSample]) -> None:
        """Zapíše fakta o přesném dělení vzorku."""
        ...

    def parallel(self, rows: Sequence[ParallelTokens]) -> None:
        """Zapíše délky veřejných paralelních textů."""
        ...


@dataclass(frozen=True)
class MeasureNoteTokens:
    """Změří tokeny přes porty bez znalosti souborů a modelových knihoven."""

    texts: Mapping[str, NoteTexts]
    samples: Mapping[str, NoteTexts]
    xlmr: TokenLengths
    mdeberta: TokenLengths
    translategemma: TokenLengths
    parallel_tokenizers: Sequence[tuple[str, TokenLengths]]
    splitter: TranslationSplitter
    sink: NoteTokenSink

    def run(self, pairs: Sequence[tuple[str, str, str]]) -> int:
        """Zapíše délky všech zpráv, dělení vzorku a délky syntetických párů.

        Returns:
            Počet zpráv celých tabulek.
        """
        total = sum(self._notes(table, source) for table, source in self.texts.items())
        for table, source in self.samples.items():
            self._sample(table, source)
        for key, counter in self.parallel_tokenizers:
            english = counter.lengths([en for _, en, _ in pairs])
            czech = counter.lengths([cs for _, _, cs in pairs])
            self.sink.parallel([
                ParallelTokens(pair[0], key, en, cs)
                for pair, en, cs in zip(pairs, english, czech, strict=True)
            ])
        return total

    def _notes(self, table: str, source: NoteTexts) -> int:
        total = 0
        for batch in source.batches():
            texts = [text for _, text in batch]
            self.sink.notes(
                table,
                [
                    NoteTokens(note[0], xlmr, mdeberta, gemma)
                    for note, xlmr, mdeberta, gemma in zip(
                        batch,
                        self.xlmr.lengths(texts),
                        self.mdeberta.lengths(texts),
                        self.translategemma.lengths(texts),
                        strict=True,
                    )
                ],
            )
            total += len(batch)
        return total

    def _sample(self, table: str, source: NoteTexts) -> None:
        for batch in source.batches():
            rows = []
            for note_id, text in batch:
                parts = self.splitter.parts(text)
                rows.append(
                    SplitSample(
                        table,
                        note_id,
                        self.splitter.fits(text),
                        0 if parts is None else parts,
                        parts is None,
                    )
                )
            self.sink.sample(rows)
