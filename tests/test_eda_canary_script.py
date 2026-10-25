"""Kanárek renderu nad syntetikou: funkce ``eda_canary`` ze scripts/eda-canary.sh.

Funkce běží nad dočasným pracovním adresářem ve tvaru renderu (``eda/``, ``results/``
a ``pdftext/`` s textem PDF) a hledá buňku s prefixem i token textu ze samých písmen.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Final

import pytest

SCRIPT: Final = Path(__file__).parents[1] / "scripts" / "eda-canary.sh"
PREFIX: Final = "eda-sentinel"
SOFT_HYPHEN: Final = "\N{SOFT HYPHEN}"
CLEAN, LEAKED, FAILED = 0, 1, 2


def _work(root: Path, files: dict[str, str]) -> Path:
    for directory in ("eda", "results/eda", "pdftext"):
        (root / directory).mkdir(parents=True)
    for name, text in files.items():
        (root / name).write_text(text, encoding="utf-8")
    return root


def _canary(work: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            "-c",
            'source "$0" && eda_canary "$1" "$2"',
            str(SCRIPT),
            str(work),
            PREFIX,
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )


def test_provenance_line_without_hyphen_after_prefix_is_clean(tmp_path: Path) -> None:
    """Provenance uvádí prefix bez pomlčky za ním, takže kanárek mlčí."""
    work = _work(
        tmp_path, {"pdftext/5.txt": f"Zdroj dat syntetické tabulky (prefix {PREFIX})\n"}
    )

    result = _canary(work)

    assert result.returncode == CLEAN, result.stderr


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("results/eda/tabulka.tex", f"{PREFIX}-patients-gender-3 & 1 \\\\\n"),
        ("pdftext/6.txt", f"Pohlaví {PREFIX}-patients-gender-3 1\n"),
        ("results/eda/ngramy.tex", "EdaSentinelD & 1 \\\\\n"),
        ("pdftext/9.txt", f"1 edasen{SOFT_HYPHEN}\ntinelb 1\n"),
    ],
    ids=[
        "cell-in-export",
        "cell-in-pdf",
        "token-any-case",
        "token-split-by-soft-hyphen",
    ],
)
def test_leaked_cell_or_token_fails_and_names_file(
    tmp_path: Path, name: str, text: str
) -> None:
    """Buňka s pomlčkou i token v jakékoli velikosti písmen a přes dělení slova je únik."""
    work = _work(tmp_path, {name: text})

    result = _canary(work)

    assert result.returncode == LEAKED
    assert Path(name).stem in result.stderr


def test_cell_leak_is_not_masked_by_clean_token_search(tmp_path: Path) -> None:
    """Únik buňky shodí kanárka, i když hledání tokenu nic nenajde."""
    work = _work(tmp_path, {"results/eda/tabulka.tex": f"{PREFIX}-admissions-race-1\n"})
    (work / "pdftext" / "1.txt").write_text("bez nálezu\n", encoding="utf-8")

    result = _canary(work)

    assert result.returncode == LEAKED
    assert "tabulka.tex" in result.stderr


def test_missing_directory_is_an_error_not_a_clean_result(tmp_path: Path) -> None:
    """Chybějící adresář výstupu je chyba hledání, ne čistý výsledek."""
    (tmp_path / "pdftext").mkdir()

    result = _canary(tmp_path)

    assert result.returncode == FAILED
    assert "selhal" in result.stderr
