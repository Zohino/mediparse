"""Databáze EDA, kterou volají dokumenty přes reticulate a sešity v Pythonu."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.eda_duckdb import EdaDatabase

if TYPE_CHECKING:
    from collections.abc import Mapping

PARQUET_VARIABLE: Final = "MEDIPARSE_PARQUET"
ROOT_VARIABLE: Final = "PIXI_PROJECT_ROOT"
DEFAULT_PARQUET: Final = Path("resources") / "mimic" / "parquet"


def project_root(environ: Mapping[str, str]) -> Path:
    """Přečte explicitní kořen projektu z prostředí pixi.

    Args:
        environ: Proměnné prostředí renderu.

    Returns:
        Kořen projektu jako cesta.

    Raises:
        InvalidInputError: Kořen chybí nebo je prázdný.
    """
    root = environ.get(ROOT_VARIABLE)
    if not root:
        msg = f"Nastavte {ROOT_VARIABLE} spuštěním přes pixi."
        raise InvalidInputError(msg)
    return Path(root)


def parquet_dir(environ: Mapping[str, str]) -> str:
    """Adresář tabulek parquet: ``MEDIPARSE_PARQUET``, jinak pod kořenem projektu.

    Args:
        environ: Proměnné prostředí; z R jako ``as.list(Sys.getenv())``, protože
            ``os.environ`` pod reticulate je snímek ze startu interpretu.

    Returns:
        Cesta jako text, protože reticulate předá ``Path`` jako objekt.

    Raises:
        InvalidInputError: Chybí adresář i kořen projektu.
    """
    explicit = environ.get(PARQUET_VARIABLE)
    if explicit:
        return explicit
    root = environ.get(ROOT_VARIABLE)
    if not root:
        msg = f"Nastavte {PARQUET_VARIABLE} nebo spusťte přes pixi ({ROOT_VARIABLE})."
        raise InvalidInputError(msg)
    return str(Path(root) / DEFAULT_PARQUET)


def database(environ: Mapping[str, str]) -> EdaDatabase:
    """Databáze nad adresářem z prostředí.

    Args:
        environ: Proměnné prostředí.

    Returns:
        Databáze; ke spojení vede ``connect()``, které volá stráž.
    """
    return EdaDatabase(Path(parquet_dir(environ)), environ)


def connection_settings(environ: Mapping[str, str]) -> dict[str, str | dict[str, str]]:
    """Pravidla spojení pro R: provede všechny kontroly a spojení nechá otevřít volajícímu.

    Args:
        environ: Proměnné prostředí.

    Returns:
        ``memory_limit``, ``temp_directory`` a ``views`` (jméno na cestu), vše text.
    """
    db = database(environ)
    views = db.guarded_views()
    settings = db.settings()
    return {
        "memory_limit": settings.memory_limit,
        "temp_directory": str(settings.temp_directory),
        "views": {name: str(path) for name, path in views.items()},
    }
