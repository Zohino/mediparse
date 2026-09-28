"""Brána syntetického korpusu: commitnout ani pushnout jde jen korpus, který prošel auditem."""

import csv
import gzip
from pathlib import Path

from mediparse.infrastructure.corpus_audit_cli import Exit, run_audit
from mediparse.infrastructure.corpus_gate_cli import main
from mediparse.infrastructure.synthetic_corpus import RECORD_NAME

REPOSITORY_CORPUS = Path(__file__).parents[1] / "resources" / "synthetic"
NOTE = "en/90000001-DS-1.txt"


def _audited_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "repo" / "resources" / "synthetic"
    (root / "en").mkdir(parents=True)
    (tmp_path / "repo" / ".git").mkdir()
    (root / NOTE).write_text("a short synthetic note", encoding="utf-8")
    reference = tmp_path / "discharge.csv.gz"
    with gzip.open(reference, mode="wt", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows([
            ["note_id", "subject_id", "text"],
            ["1-DS-1", "1", "reference text"],
        ])
    assert (
        run_audit(root, [reference], tmp_path / "report.json", "c" * 40, {})
        == Exit.CLEAN
    )
    return root


def _gate(root: Path) -> int:
    return main(["--corpus", str(root)])


def test_gate_passes_without_corpus(tmp_path: Path) -> None:
    """Dokud korpus neexistuje, brána commit ani push neblokuje."""
    assert _gate(tmp_path / "missing") == Exit.CLEAN


def test_gate_rejects_unaudited_corpus(tmp_path: Path) -> None:
    """Korpus bez záznamu auditu neprojde."""
    root = _audited_corpus(tmp_path)
    (root / RECORD_NAME).unlink()

    assert _gate(root) == Exit.FOUND


def test_gate_passes_audited_corpus(tmp_path: Path) -> None:
    """Korpus beze změny od auditu projde."""
    assert _gate(_audited_corpus(tmp_path)) == Exit.CLEAN


def test_gate_rejects_corpus_changed_after_audit(tmp_path: Path) -> None:
    """Přegenerovaná zpráva bez nového auditu neprojde."""
    root = _audited_corpus(tmp_path)
    (root / NOTE).write_text("a regenerated synthetic note", encoding="utf-8")

    assert _gate(root) == Exit.FOUND


def test_gate_rejects_added_note(tmp_path: Path) -> None:
    """Nová zpráva přidaná po auditu otisk také změní."""
    root = _audited_corpus(tmp_path)
    (root / "en" / "90000002-DS-1.txt").write_text("another note", encoding="utf-8")

    assert _gate(root) == Exit.FOUND


def test_gate_rejects_invalid_record(tmp_path: Path) -> None:
    """Záznam, který neodpovídá schématu, korpus neatestuje."""
    root = _audited_corpus(tmp_path)
    (root / RECORD_NAME).write_text("{}", encoding="utf-8")

    assert _gate(root) == Exit.FOUND


def test_repository_corpus_passes_gate() -> None:
    """Korpus v repu odpovídá svému auditu — neauditovaný korpus neprojde CI."""
    assert _gate(REPOSITORY_CORPUS) == Exit.CLEAN
