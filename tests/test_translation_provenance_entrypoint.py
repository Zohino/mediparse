"""Vstupní bod provenance překladu nad skutečnými soubory: audit, pracovní adresář a gate."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mediparse.entrypoints import corpus_gate, translation_provenance
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.synthetic_corpus import CorpusDirectory
from tests.support import REPOSITORY_CONFIG
from tests.workdir_support import write_single_part_workdir

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import AuditFiles

ENGLISH = "a short synthetic note ___ ___"
CZECH = "krátká syntetická zpráva ___\n"


def _argv(audit_files: AuditFiles, corpus: Path, workdir: Path) -> list[str]:
    return [
        f"--workdir={workdir}",
        f"--corpus={corpus}",
        f"--config={REPOSITORY_CONFIG}",
        f"--tables={audit_files.tables}",
    ]


def _released(audit_files: AuditFiles) -> tuple[Path, Path]:
    root = audit_files.released_corpus({
        "en/90000001-DS-1.txt": ENGLISH,
        "cs/90000001-DS-1.txt": CZECH,
    })
    workdir = audit_files.root / "work"
    write_single_part_workdir(workdir, {"90000001-DS-1": (ENGLISH, CZECH)})
    return root, workdir


def test_translation_provenance_is_written_and_gate_passes(
    audit_files: AuditFiles,
) -> None:
    """Zápis doplní translation k původnímu generování a brána pak korpus pustí."""
    root, workdir = _released(audit_files)
    generation = CorpusDirectory(root).provenance_record()
    assert generation is not None

    code = translation_provenance.run(_argv(audit_files, root, workdir))

    record = CorpusDirectory(root).provenance_record()
    assert code == ExitCode.OK
    assert record is not None
    assert record.generation == generation.generation
    assert record.translation is not None
    assert record.translation.marker_mismatches == ("90000001-DS-1",)
    gate = corpus_gate.run([f"--corpus={root}", f"--tables={audit_files.tables}"])
    assert gate == ExitCode.OK


def test_missing_workdir_is_refused(audit_files: AuditFiles) -> None:
    """Chybějící pracovní adresář je chybný vstup, nic se nezapíše."""
    root, workdir = _released(audit_files)
    (workdir / "requests.json").unlink()

    code = translation_provenance.run(_argv(audit_files, root, workdir))

    record = CorpusDirectory(root).provenance_record()
    assert code == ExitCode.REFUSED
    assert record is not None
    assert record.translation is None


def test_czech_not_matching_workdir_is_refused(audit_files: AuditFiles) -> None:
    """Překlad, který nevznikl z tohoto adresáře, provenance nedostane."""
    root, workdir = _released(audit_files)
    write_single_part_workdir(workdir, {"90000001-DS-1": (ENGLISH, "jiný text\n")})

    assert (
        translation_provenance.run(_argv(audit_files, root, workdir))
        == ExitCode.REFUSED
    )
