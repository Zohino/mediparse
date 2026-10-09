"""Use case provenance překladu: k provenance auditovaného korpusu doplní, čím a odkud vznikl český korpus.

Generování ponechá, překlad zapíše a přepíše otisk auditu. Překlad se váže na
korpus: originály, množina zpráv i české texty musí odpovídat pracovnímu adresáři.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from mediparse.application.corpus_gate import GatedCorpus
from mediparse.application.corpus_provenance import (
    ProvenanceCorpus,
    ProvenanceOutcome,
    ProvenanceRefused,
    ProvenanceWritten,
    audit_refusal,
)
from mediparse.application.corpus_texts import (
    LANGUAGE,
    TRANSLATION_LANGUAGE,
    texts_in_language,
)
from mediparse.domain.corpus_provenance import (
    InvalidProvenanceRecordError,
    ProvenanceRecord,
)
from mediparse.domain.translation_consistency import (
    binding_violations,
    marker_mismatches,
    translation_violations,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from mediparse.domain.corpus_provenance import Translation
    from mediparse.domain.translation_consistency import TranslatedNote


class TranslationSource(Protocol):
    """Pracovní adresář překladu: vstupy překladu a přeložené zprávy."""

    def translation(self) -> Translation:
        """Vstupy překladu bez neshod značek.

        Returns:
            Provenance překladu s prázdným ``marker_mismatches``.

        Raises:
            InvalidInputError: Pracovní adresář chybí nebo neodpovídá schématu.
        """

    def notes(self) -> tuple[TranslatedNote, ...]:
        """Zprávy z požadavků s přeloženými částmi.

        Returns:
            Zprávy v pořadí požadavků.

        Raises:
            InvalidInputError: Pracovní adresář chybí nebo neodpovídá schématu.
        """


class TranslatedCorpus(ProvenanceCorpus, GatedCorpus, Protocol):
    """Korpus, k němuž se provenance čte i zapisuje."""


@dataclass(frozen=True)
class TranslationProvenance:
    """Provenance překladu auditovaného korpusu, který má provenance generování."""

    corpus: TranslatedCorpus
    source: TranslationSource

    def run(
        self,
        reference_sha256: Mapping[str, str],
        structure_labels: Collection[str],
    ) -> ProvenanceOutcome:
        """Ověří audit a vazbu na pracovní adresář a zapíše provenance s překladem.

        Returns:
            Zápis záznamu, nebo odmítnutí s důvodem.
        """
        existing = self._existing(reference_sha256, structure_labels)
        if isinstance(existing, ProvenanceRefused):
            return existing
        texts = self.corpus.notes()
        english = texts_in_language(texts, LANGUAGE)
        czech = texts_in_language(texts, TRANSLATION_LANGUAGE)
        if not czech:
            return ProvenanceRefused("Korpus nemá český překlad.")
        incomplete = translation_violations(english, czech)
        if incomplete:
            return ProvenanceRefused(" ".join(incomplete))
        translation = self.source.translation()
        bound = binding_violations(
            self.source.notes(), english, czech, translation.masking
        )
        if bound:
            return ProvenanceRefused(" ".join(bound))
        translation = translation.model_copy(
            update={"marker_mismatches": marker_mismatches(english, czech)}
        )
        self.corpus.save_provenance(
            ProvenanceRecord(
                generation=existing.generation,
                audit_sha256=self.corpus.audit_sha256(),
                translation=translation,
            )
        )
        return ProvenanceWritten()

    def _existing(
        self, reference_sha256: Mapping[str, str], structure_labels: Collection[str]
    ) -> ProvenanceRecord | ProvenanceRefused:
        refusal = audit_refusal(self.corpus, reference_sha256, structure_labels)
        if refusal is not None:
            return refusal
        try:
            existing = self.corpus.provenance_record()
        except InvalidProvenanceRecordError:
            return ProvenanceRefused("Záznam provenance neodpovídá schématu.")
        if existing is None:
            return ProvenanceRefused(
                "Korpus nemá provenance generování, nejdřív ji zapiš."
            )
        return existing
