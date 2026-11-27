"""Sdílený korpus modelových kroků s tabulkou, splitem a otiskem."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Self

from mediparse.application.diagnostic_strip import DiagnosticStripTable
from mediparse.domain.run_manifest import DiagnosticStrip
from mediparse.infrastructure.code_fingerprint import code_key
from mediparse.infrastructure.file_digest import file_sha256
from mediparse.infrastructure.note_table import ParquetNoteTable
from mediparse.infrastructure.section_lexicon_config import load_section_lexicon
from mediparse.infrastructure.split_table import ParquetSplitTable

if TYPE_CHECKING:
    import argparse
    from collections.abc import Mapping, Sequence

    from mediparse.application.ports.experiment import PatientNoteTable, SplitSource


@dataclass(frozen=True, kw_only=True)
class ModelCorpusPaths:
    """Sdílené cesty ke korpusu modelových kroků."""

    cohort: Path
    candidates: Path
    lexicon: Path
    split: Path

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Přidá povinné argumenty cest korpusu do parseru.

        Args:
            parser: Parser příkazové řádky modelového kroku.
        """
        for field in fields(cls):
            parser.add_argument(f"--{field.name}", type=Path, required=True)

    @classmethod
    def from_arguments(cls, args: argparse.Namespace) -> Self:
        """Sestaví cesty korpusu z argumentů příkazové řádky.

        Args:
            args: Naparsované argumenty modelového kroku.

        Returns:
            Sdílené cesty ke korpusu.
        """
        return cls(**{field.name: getattr(args, field.name) for field in fields(cls)})

    @property
    def data(self) -> tuple[Path, Path]:
        """Vrátí vstupy pod DUA; stráž volá vstupní bod před načtením configu.

        Lexikon a kandidáti jsou konfigurace z repozitáře.
        """
        return self.cohort, self.split


@dataclass(frozen=True)
class ModelCorpus:
    """Složené adaptéry korpusu a jeho záznamy pro manifest."""

    table: PatientNoteTable
    strip: DiagnosticStrip | None
    split: SplitSource
    dataset_sha256: str


def model_corpus(
    paths: ModelCorpusPaths, *, strip_diagnostic_sections: bool
) -> ModelCorpus:
    """Složí tabulku vstupu, strip, split a otisk kohorty.

    Returns:
        Sdílený korpus modelového kroku.

    Raises:
        InvalidInputError: Kohorta chybí nebo lexikon není platný.
    """
    table, strip = model_input_table(
        cohort=paths.cohort,
        strip_diagnostic_sections=strip_diagnostic_sections,
        lexicon=paths.lexicon,
    )
    return ModelCorpus(
        table=table,
        strip=strip,
        split=ParquetSplitTable(paths.split),
        dataset_sha256=file_sha256(paths.cohort),
    )


def model_input_table(
    *, cohort: Path, strip_diagnostic_sections: bool, lexicon: Path
) -> tuple[PatientNoteTable, DiagnosticStrip | None]:
    """Složí tabulku zpráv pro trénink, inferenci i ladění a záznam stripu pro manifest.

    Trénink, ladění modelů i stupňů a inference používají stejné volby,
    aby klasifikovaly týž text.

    Args:
        cohort: Cesta k tabulce kohorty.
        strip_diagnostic_sections: Přepínač odstranění diagnostických sekcí.
        lexicon: Cesta k lexikonu diagnostických sekcí.

    Returns:
        Tabulka beze změny a ``None``, když přepínač stripu je vypnutý. Jinak tabulka bez
        diagnostických sekcí a záznam s jejich seřazenými klíči a otiskem lexikonu.
    """
    table = ParquetNoteTable(cohort)
    if not strip_diagnostic_sections:
        return table, None
    sections = load_section_lexicon(lexicon)
    record = DiagnosticStrip(
        sections=tuple(sections.diagnostic_sections),
        lexicon_sha256=file_sha256(lexicon),
    )
    return DiagnosticStripTable(table, sections), record


def checkpoint_key(
    config: Path,
    corpus: ModelCorpusPaths,
    dataset_sha256: str,
    packages: Mapping[str, str],
    extra: Sequence[Path] = (),
) -> str:
    """Otiskne vstupy, kód a verze knihoven pro obnovu modelového kroku.

    Returns:
        Klíč rozpracovaného běhu.
    """
    return code_key([
        ("config", file_sha256(config).encode()),
        ("cohort", dataset_sha256.encode()),
        ("split", file_sha256(corpus.split).encode()),
        ("candidates", file_sha256(corpus.candidates).encode()),
        ("lexicon", file_sha256(corpus.lexicon).encode()),
        *((f"extra/{i}", file_sha256(path).encode()) for i, path in enumerate(extra)),
        *((f"packages/{name}", value.encode()) for name, value in packages.items()),
    ])
