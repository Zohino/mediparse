"""Invalidace a publikace cache měření zpráv mimo adresář tabulek MIMIC."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Final

import sklearn

from mediparse.domain.digest import fingerprint
from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.file_digest import file_sha256
from mediparse.infrastructure.note_features_table import ARTIFACTS

if TYPE_CHECKING:
    from collections.abc import Generator

FORMAT_VERSION: Final = "1"
PACKAGE: Final = Path(__file__).parents[1]
CACHE_FILE: Final = "cache.json"


def cache_key(discharge: Path, lexicon: Path, candidates: Path) -> str:
    """Otisk vstupu, konfigurace, formátu, Python kódu a verze scikit-learn.

    Returns:
        SHA-256 klíč cache.

    Raises:
        InvalidInputError: Vstup klíče nelze přečíst.
    """
    # Commit by zneplatnil cache i po kosmetické změně sešitu; celý balíček naopak
    # zachytí změnu parseru či analyzéru bez křehkého seznamu importovaných modulů.
    try:
        files = [
            ("format", FORMAT_VERSION.encode()),
            ("discharge", file_sha256(discharge).encode()),
            ("lexicon", lexicon.read_bytes()),
            ("candidates", candidates.read_bytes()),
            ("scikit-learn", sklearn.__version__.encode()),
        ]
        files.extend(
            (path.relative_to(PACKAGE).as_posix(), path.read_bytes())
            for path in sorted(PACKAGE.rglob("*.py"))
        )
        return fingerprint(files)
    except OSError as error:
        msg = f"Vstup klíče cache nelze přečíst: {type(error).__name__}."
        raise InvalidInputError(msg) from None


def is_current(directory: Path, key: str) -> bool:
    """Ověří klíč a existenci všech artefaktů bez čtení řádkových dat.

    Returns:
        Zda lze cache použít.
    """
    try:
        manifest = json.loads((directory / CACHE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        isinstance(manifest, dict)
        and manifest.get("key") == key
        and all((directory / name).is_file() for name, _ in ARTIFACTS)
    )


@contextmanager
def pending_cache(directory: Path, key: str) -> Generator[Path]:
    """Publikuje hotové soubory atomicky po souborech s klíčem jako posledním.

    Yields:
        Dočasný podadresář pro výpočet.

    Raises:
        InvalidInputError: Cache nelze publikovat.
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="pending-", dir=directory) as temporary:
            staging = Path(temporary)
            yield staging
            _publish(staging, directory, key)
    except OSError as error:
        msg = f"Cache měření zpráv nelze publikovat: {type(error).__name__}."
        raise InvalidInputError(msg) from None


def _publish(staging: Path, directory: Path, key: str) -> None:
    (staging / CACHE_FILE).write_text(json.dumps({"key": key}) + "\n", encoding="utf-8")
    # Starý klíč nesmí zplatnit směs souborů, pokud publikaci přeruší chyba.
    (directory / CACHE_FILE).unlink(missing_ok=True)
    for name in [*(name for name, _ in ARTIFACTS), CACHE_FILE]:
        (staging / name).replace(directory / name)
