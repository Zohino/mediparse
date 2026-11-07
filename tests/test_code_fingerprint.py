"""Otisk uzávěru pkg.a → pkg.b, nezávislého pkg.c a dat balíčku."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.code_fingerprint import code_fingerprint

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def package(tmp_path: Path) -> Path:
    """Vytvoří balíček s ručně známým uzávěrem importů skriptu.

    Returns:
        Adresář pkg; skript run.py leží vedle něj.
    """
    root = tmp_path / "pkg"
    root.mkdir()
    for name, source in {
        "__init__.py": "",
        "a.py": "import json\nimport sys\nfrom pkg.b import f\nx = f()\n",
        "b.py": "def f():\n    return 1\n",
        "c.py": "x = 1\n",
        "data.sql": "SELECT 1;\n",
    }.items():
        (root / name).write_text(source, encoding="utf-8")
    (tmp_path / "run.py").write_text("from pkg.a import x\n", encoding="utf-8")
    return root


def test_transitive_function_change_changes_fingerprint(package: Path) -> None:
    """Změna těla funkce v transitivně dosažitelném b.py mění otisk."""
    script = package.parent / "run.py"
    before = code_fingerprint(script, package)
    (package / "b.py").write_text("def f():\n    return 2\n", encoding="utf-8")

    assert code_fingerprint(script, package) != before


def test_unreachable_module_does_not_change_fingerprint(package: Path) -> None:
    """Nezávislý c.py do uzávěru importů nepatří."""
    script = package.parent / "run.py"
    before = code_fingerprint(script, package)
    (package / "c.py").write_text("x = 2\n", encoding="utf-8")

    assert code_fingerprint(script, package) == before


def test_comments_and_formatting_do_not_change_fingerprint(package: Path) -> None:
    """Komentář, prázdný řádek a jiné formátování nechají AST stejné."""
    script = package.parent / "run.py"
    before = code_fingerprint(script, package)
    (package / "b.py").write_text(
        "# Komentář\n\ndef f( ):\n    return (1)\n", encoding="utf-8"
    )

    assert code_fingerprint(script, package) == before


def test_package_data_change_changes_fingerprint(package: Path) -> None:
    """Data balíčku se započítají i bez importu z Pythonu."""
    script = package.parent / "run.py"
    before = code_fingerprint(script, package)
    (package / "data.sql").write_text("SELECT 2;\n", encoding="utf-8")

    assert code_fingerprint(script, package) != before


def test_bytecode_cache_does_not_change_fingerprint(package: Path) -> None:
    """Změna bajtů v __pycache__ nepatří ke změnám zdrojů ani dat balíčku."""
    script = package.parent / "run.py"
    cache = package / "__pycache__"
    cache.mkdir()
    bytecode = cache / "b.cpython-313.pyc"
    bytecode.write_bytes(b"old bytecode")
    before = code_fingerprint(script, package)
    bytecode.write_bytes(b"new bytecode")

    assert code_fingerprint(script, package) == before


def test_missing_package_import_is_rejected(package: Path) -> None:
    """Chybějící modul balíčku nesmí vytvořit neúplný otisk."""
    script = package.parent / "run.py"
    script.write_text("from pkg.missing import y\n", encoding="utf-8")

    with pytest.raises(InvalidInputError, match=r"pkg\.missing"):
        code_fingerprint(script, package)


def test_shell_script_byte_change_changes_fingerprint(package: Path) -> None:
    """Ne-Python skript se otiskuje po bajtech."""
    script = package.parent / "inventory.sh"
    script.write_bytes(b"echo 1\n")
    before = code_fingerprint(script, package)
    script.write_bytes(b"echo 2\n")

    assert code_fingerprint(script, package) != before


def test_fingerprint_is_deterministic_sha256(package: Path) -> None:
    """Stejný strom dává při opakovaném volání stejný hexadecimální SHA-256."""
    script = package.parent / "run.py"
    before = code_fingerprint(script, package)

    assert code_fingerprint(script, package) == before
    assert re.fullmatch(r"[0-9a-f]{64}", before)


@pytest.mark.parametrize("name", ["missing.py", "missing.sh"])
def test_missing_script_is_rejected(package: Path, name: str) -> None:
    """Chybějící skript skončí doménovou chybou se jménem souboru."""
    with pytest.raises(InvalidInputError, match=name):
        code_fingerprint(package.parent / name, package)
