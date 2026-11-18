"""Obnovitelná cache embeddingů klíčovaná přesným textem a numerickým režimem."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import pyarrow as pa

from mediparse.domain.digest import fingerprint
from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure import hf_encoder, hf_tokenizers
from mediparse.infrastructure.code_fingerprint import module_sha256
from mediparse.infrastructure.machine_hardware import peak_rss_gb
from mediparse.infrastructure.parquet_file import read_table, write_table

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Mapping, Sequence

    from numpy import ndarray

    from mediparse.domain.screening import EncoderSpec

logger = logging.getLogger(__name__)
PART_NOTES: Final = 256
SCHEMA: Final = pa.schema([
    pa.field("text_sha256", pa.string(), nullable=False),
    pa.field("embedding", pa.list_(pa.float32()), nullable=False),
])
type TextEncoder = Callable[[str], ndarray]


def cache_key(spec: EncoderSpec, versions: Mapping[str, str]) -> str:
    """Otiskne parametry výpočtu, knihovny a AST encoderu bez I/O cache.

    Returns:
        Klíč obsahové cache nezávislý na jménu reprezentace.
    """
    return fingerprint([
        *(
            (name, str(value).encode())
            for name, value in spec.model_dump(exclude={"name"}).items()
        ),
        *((f"packages/{name}", value.encode()) for name, value in versions.items()),
        ("encoder", module_sha256(Path(hf_encoder.__file__)).encode()),
        ("tokenizer", module_sha256(Path(hf_tokenizers.__file__)).encode()),
    ])


@dataclass(frozen=True)
class NoteEmbeddings:
    """Cache po atomických dílech s odloženým vytvořením encoderu."""

    directory: Path
    versions: Mapping[str, str]
    encoders: Callable[[EncoderSpec], TextEncoder]

    def embed(self, spec: EncoderSpec, texts: Sequence[str]) -> ndarray:
        """Vrátí embeddingy v pořadí vstupu a dopočte jedinečné chybějící texty.

        Returns:
            Matice float32, jeden řádek na vstupní text.

        Raises:
            InvalidInputError: Cache má nekonzistentní šířku, nesouvislé díly či chybu I/O.
        """
        with self._errors():
            directory = self.directory / cache_key(spec, self.versions)
            parts = sorted(directory.glob("*.parquet"))
            if [part.name for part in parts] != [
                f"{index:04d}.parquet" for index in range(len(parts))
            ]:
                msg = "Díly cache embeddingů netvoří souvislou řadu."
                raise InvalidInputError(msg)
            vectors = {}
            for part in parts:
                rows = read_table(part, SCHEMA).to_pydict()
                vectors.update(zip(rows["text_sha256"], rows["embedding"], strict=True))
            self._ensure_width(vectors.values())
            identities = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
            missing = dict(zip(identities, texts, strict=True))
            missing = [
                (key, text) for key, text in missing.items() if key not in vectors
            ]
            if missing:
                encoder = self.encoders(spec)
                for start in range(0, len(missing), PART_NOTES):
                    batch = missing[start : start + PART_NOTES]
                    computed = {key: encoder(text).tolist() for key, text in batch}
                    self._ensure_width([*vectors.values(), *computed.values()])
                    target = (
                        directory / f"{len(parts) + start // PART_NOTES:04d}.parquet"
                    )
                    temporary = target.with_suffix(".parquet.tmp")
                    write_table(
                        temporary,
                        {
                            "text_sha256": list(computed),
                            "embedding": list(computed.values()),
                        },
                        SCHEMA,
                    )
                    temporary.replace(target)
                    vectors.update(computed)
                    logger.info(
                        "Embeddingy: %d hotových, %d chybějících zpráv, %s.",
                        start + len(batch),
                        len(missing) - start - len(batch),
                        peak_rss_gb(),
                    )
            return np.asarray([vectors[key] for key in identities], dtype=np.float32)

    @contextmanager
    def _errors(self) -> Generator[None]:
        try:
            yield
        except OSError as error:
            msg = f"Cache embeddingů {self.directory} nelze přečíst nebo zapsat: {type(error).__name__}."
            raise InvalidInputError(msg) from None

    @staticmethod
    def _ensure_width(vectors: Iterable[Sequence[float]]) -> None:
        if len({len(vector) for vector in vectors}) > 1:
            msg = "Embeddingy v cache mají nejednotné délky."
            raise InvalidInputError(msg)
