"""Shellové pomocníky EDA: kanárek, staging a spuštění přípravy cache.

Kanárek běží nad dočasným pracovním adresářem ve tvaru renderu (``eda/``, ``results/``
a ``pdftext/`` s textem PDF) a hledá buňku s prefixem i token textu ze samých písmen.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest

from mediparse.infrastructure.data_access_rules import load_data_access_rules

if TYPE_CHECKING:
    from mediparse.domain.data_access import BlockingVariable

SCRIPT: Final = Path(__file__).parents[1] / "scripts" / "eda-canary.sh"
EDA_PRESENT: Final = pytest.mark.skipif(
    not (SCRIPT.parent.parent / "eda").is_dir(),
    reason="Skripty renderu EDA nejsou v obrazu kontejneru; hlídá job Test v CI.",
)
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


def test_neighbouring_label_lines_do_not_form_a_token(tmp_path: Path) -> None:
    """Sousední řádky popisků ``eda`` a ``sentinel`` nejsou únik tokenu ``edasentinel``."""
    work = _work(tmp_path, {"pdftext/9.txt": "eda\nsentinel\n"})

    result = _canary(work)

    assert result.returncode == CLEAN, result.stderr


def test_token_split_by_soft_hyphen_before_blank_line_is_leak(tmp_path: Path) -> None:
    """Měkký spojovník před prázdným řádkem slovo nerozděluje."""
    work = _work(tmp_path, {"pdftext/9.txt": f"1 edasen{SOFT_HYPHEN}\n\ntinelb 1\n"})

    result = _canary(work)

    assert result.returncode == LEAKED
    assert "9.txt" in result.stderr


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


def _render_environment(tmp_path: Path) -> dict[str, str]:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    render = scripts / "eda.sh"
    render.write_bytes((SCRIPT.parent / "eda.sh").read_bytes())
    (scripts / "eda-render.sh").write_text(
        "eda_stage() { :; }\neda_revision() { :; }\n"
        'eda_render() { printf "%s\\n" "$MEDIPARSE_EDA_PROVENANCE" > "$EDA_TEST_PROVENANCE_PATH"; }\n',
        encoding="utf-8",
    )
    commands = tmp_path / "bin"
    commands.mkdir()
    stubs = {
        "git": '#!/bin/sh\nprintf "git %s\\n" "$*" >> "$EDA_TEST_GIT_CALLS"\n',
        "mediparse-eda-provenance": '#!/bin/sh\nprintf "provenance %s\\n" "$*" > "$EDA_TEST_PROVENANCE"\n',
        "mediparse-eda-note-features": '#!/bin/sh\nprintf "features %s\\n" "$*" >> "$EDA_TEST_CALLS"\n',
        "mediparse-eda-note-tokens": '#!/bin/sh\nprintf "tokens %s\\n" "$*" >> "$EDA_TEST_CALLS"\n',
    }
    for name, text in stubs.items():
        command = commands / name
        command.write_text(text, encoding="utf-8")
        command.chmod(0o755)
    return {
        **os.environ,
        **{variable.name: "" for variable in load_data_access_rules().blocking},
        "PATH": f"{commands}{os.pathsep}{os.environ['PATH']}",
        "PIXI_PROJECT_ROOT": str(tmp_path),
        "MEDIPARSE_PARQUET": str(tmp_path / "tables"),
        "MEDIPARSE_EDA_CACHE": str(tmp_path / "cache"),
        "EDA_TEST_CALLS": str(tmp_path / "feature-calls.txt"),
        "EDA_TEST_GIT_CALLS": str(tmp_path / "git-calls.txt"),
        "EDA_TEST_PROVENANCE": str(tmp_path / "provenance-calls.txt"),
        "EDA_TEST_PROVENANCE_PATH": str(tmp_path / "provenance-path.txt"),
    }


@EDA_PRESENT
@pytest.mark.parametrize(
    "variable", load_data_access_rules().blocking, ids=lambda variable: variable.name
)
def test_render_script_blocks_before_any_preparation(
    tmp_path: Path, variable: BlockingVariable
) -> None:
    """Každé pravidlo z JSON zastaví render ještě před git a přípravou cache."""
    environ = _render_environment(tmp_path)
    environ[variable.name] = "1"

    result = subprocess.run(
        ["bash", str(tmp_path / "scripts/eda.sh"), "9"],
        env=environ,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )

    assert result.returncode != CLEAN
    assert variable.reason in result.stderr
    for key in ("EDA_TEST_GIT_CALLS", "EDA_TEST_CALLS", "EDA_TEST_PROVENANCE"):
        assert not Path(environ[key]).exists()


@EDA_PRESENT
@pytest.mark.parametrize("number", ["1", "4", "9"])
def test_render_script_prepares_cache_for_its_consumer(
    tmp_path: Path, number: str
) -> None:
    """Izolované stuby spouštějí jen skupinu cache požadovaného sešitu."""
    environ = _render_environment(tmp_path)
    result = subprocess.run(
        ["bash", str(tmp_path / "scripts/eda.sh"), number],
        env=environ,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert result.returncode == CLEAN, result.stderr
    out = Path(environ["EDA_TEST_PROVENANCE_PATH"]).read_text(encoding="utf-8").strip()
    assert out.endswith("/provenance.json")
    assert Path(environ["EDA_TEST_PROVENANCE"]).read_text(encoding="utf-8") == (
        f"provenance --root {tmp_path} --parquet {tmp_path / 'tables'} --out {out}\n"
    )
    calls = Path(environ["EDA_TEST_CALLS"])
    assert calls.exists() is (number in {"4", "9"})
    if number in {"4", "9"}:
        group = "tokens" if number == "4" else "features"
        line = calls.read_text(encoding="utf-8")
        assert line.startswith(f"{group} --parquet {tmp_path / 'tables'} ")
        assert line.count("\n") == 1
        assert f"--out {tmp_path / 'cache'}" in line
        if number == "4":
            assert f"--translation {tmp_path / 'translation'}" in line
            assert f"--english {tmp_path / 'resources/synthetic/en'}" in line
            assert f"--czech {tmp_path / 'resources/synthetic/cs'}" in line


@EDA_PRESENT
def test_staging_includes_shared_contract_and_excludes_render_cache(
    tmp_path: Path,
) -> None:
    """Pracovní kopie má sdílený kontrakt ve správné relativní cestě bez freeze."""
    root, work = tmp_path / "root", tmp_path / "work"
    work.mkdir()
    for name in ("eda/_freeze", "eda/.quarto", "analysis/R"):
        (root / name).mkdir(parents=True)
    for name in (
        "eda/_freeze/old",
        "eda/.quarto/old",
        "eda/9.qmd",
        "analysis/R/contract.R",
    ):
        (root / name).write_text("zkonstruovaná fixtura", encoding="utf-8")
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; eda_stage "$2" "$3"',
            "eda-stage-test",
            str(SCRIPT.parent / "eda-render.sh"),
            str(root),
            str(work),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert result.returncode == CLEAN, result.stderr
    assert (work / "analysis/R/contract.R").read_bytes() == (
        root / "analysis/R/contract.R"
    ).read_bytes()
    assert (work / "eda/9.qmd").exists()
    assert not (work / "eda/_freeze").exists()
    assert not (work / "eda/.quarto").exists()
