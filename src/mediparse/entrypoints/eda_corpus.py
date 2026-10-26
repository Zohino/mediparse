"""Přehled korpusu pro první sešit EDA přes strážené spojení a manifest."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

from mediparse.domain.corpus_overview import CorpusOverview, summarize_corpus
from mediparse.entrypoints.eda_database import database, project_root
from mediparse.infrastructure.mimic_inventory import read_manifest, read_manifest_bytes

if TYPE_CHECKING:
    from collections.abc import Mapping

MANIFEST_PATH: Final = Path("results") / "mimic" / "manifest.json"


def corpus_overview(environ: Mapping[str, str]) -> CorpusOverview:
    """Sestaví přehled databáze a ověří její počty proti manifestu validace.

    Args:
        environ: Prostředí renderu s kořenem projektu a adresářem tabulek.

    Returns:
        Tabulka a inline texty pro sešit.
    """
    root = project_root(environ)
    counts = database(environ).record_counts()
    manifest = root / MANIFEST_PATH
    return summarize_corpus(
        counts, read_manifest(manifest), read_manifest_bytes(manifest)
    )
