"""Sdílený split v jednom souboru parquet: řádek na pacienta, parametry a otisk v metadatech."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import pyarrow as pa
from pydantic import ValidationError

from mediparse.domain.digest import Sha256  # ruff: ignore[typing-only-first-party-import] - pydantic čte anotaci za běhu
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.split import SharedSplit, Split, SplitConfig, SubjectLeakError
from mediparse.infrastructure.parquet_file import read_table, write_table

if TYPE_CHECKING:
    from pathlib import Path

    from mediparse.domain.identity import SubjectId

SCHEMA: Final = pa.schema([
    pa.field("subject_id", pa.int64(), nullable=False),
    pa.field("fold", pa.int64(), nullable=False),
])
METADATA_KEY: Final = b"mediparse.split"


class _SplitMetadata(SplitConfig):
    """Parametry splitu a otisk přiřazení uložené v metadatech schématu souboru."""

    sha256: Sha256


@dataclass(frozen=True)
class ParquetSplitTable:
    """Split v souboru parquet; seed, počet foldů a otisk přiřazení nese metadata."""

    path: Path

    def write(self, shared: SharedSplit) -> None:
        """Zapíše pacienty seřazené podle ``subject_id``, nadřazený adresář založí."""
        assignment = sorted(
            (subject, fold)
            for fold, subjects in enumerate(shared.split.folds)
            for subject in subjects
        )
        metadata = _SplitMetadata(
            folds=shared.config.folds,
            seed=shared.config.seed,
            sha256=shared.split.sha256(),
        )
        # Kanonický JSON nezávislý na pořadí polí modelu, aby bajty souboru
        # nezávisely na refaktoru ``_SplitMetadata``.
        encoded = json.dumps(metadata.model_dump(), sort_keys=True).encode()
        write_table(
            self.path,
            {
                "subject_id": [subject for subject, _ in assignment],
                "fold": [fold for _, fold in assignment],
            },
            SCHEMA.with_metadata({METADATA_KEY: encoded}),
        )

    def read(self) -> SharedSplit:
        """Přečte split a ověří ho proti metadatům souboru.

        Returns:
            Split s parametry ze souboru.

        Raises:
            InvalidInputError: Soubor chybí, má jiné schéma nebo neplatná
                metadata, fold mimo rozsah, pacienta ve dvou foldech, opakovaný
                řádek pacienta nebo otisk, který nesedí s řádky.
        """
        table = read_table(self.path, SCHEMA)
        metadata = self._metadata(table.schema)
        rows = list(
            zip(
                table.column("subject_id").to_pylist(),
                table.column("fold").to_pylist(),
                strict=True,
            )
        )
        outside = sum(not 0 <= fold < metadata.folds for _, fold in rows)
        if outside:
            msg = (
                f"Soubor {self.path} obsahuje pacienty ve foldu mimo "
                f"0 až {metadata.folds - 1}: {outside}."
            )
            raise InvalidInputError(msg)
        split = self._split(rows, metadata.folds)
        patients = sum(len(subjects) for subjects in split.folds)
        if len(rows) != patients:
            msg = (
                f"Soubor {self.path} má {len(rows)} řádků, ale "
                f"{patients} pacientů; některý pacient je v něm víckrát."
            )
            raise InvalidInputError(msg)
        if split.sha256() != metadata.sha256:
            msg = f"Soubor {self.path}: otisk v metadatech nesedí s řádky."
            raise InvalidInputError(msg)
        return SharedSplit(SplitConfig(seed=metadata.seed, folds=metadata.folds), split)

    def _metadata(self, schema: pa.Schema) -> _SplitMetadata:
        raw = (schema.metadata or {}).get(METADATA_KEY)
        if raw is None:
            msg = f"Soubor {self.path} nemá metadata splitu."
            raise InvalidInputError(msg)
        try:
            return _SplitMetadata.model_validate_json(raw)
        except ValidationError as error:
            msg = f"Soubor {self.path} má neplatná metadata splitu:\n{error}"
            raise InvalidInputError(msg) from error

    def _split(self, rows: list[tuple[SubjectId, int]], folds: int) -> Split:
        # K z metadat, ne z řádků: prázdný fold se z řádků nepozná.
        members: list[set[SubjectId]] = [set() for _ in range(folds)]
        for subject, fold in rows:
            members[fold].add(subject)
        try:
            return Split(tuple(frozenset(subjects) for subjects in members))
        except SubjectLeakError as error:
            msg = f"Soubor {self.path} je neplatný. {error}"
            # Text výjimky nese jen počet, ale ``from None`` zahodí i řetěz příčin,
            # aby se do logu nedostal traceback domény; stejný vzor jako note_table.
            raise InvalidInputError(msg) from None
