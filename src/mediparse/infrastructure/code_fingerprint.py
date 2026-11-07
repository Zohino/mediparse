"""Otisk kódu dosažitelného importy skriptu a dat balíčku pro Snakemake."""

from __future__ import annotations

import ast
import hashlib
from modulefinder import ModuleFinder
from pathlib import Path

from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.file_digest import file_sha256


def code_fingerprint(script: Path, package: Path) -> str:
    """Otiskne uzávěr importů v balíčku, jeho data nebo bajty shellového skriptu.

    Args:
        script: Skript pravidla, Python se rozpozná podle přípony .py.
        package: Adresář balíčku s __init__.py.

    Returns:
        Hexadecimální SHA-256 seřazených řádků druhu, jména a otisku.

    Raises:
        InvalidInputError: Skript nebo modul chybí, zdroj nelze přečíst či parsovat
            nebo skript neimportuje žádný modul požadovaného balíčku.
    """
    try:
        rows = _fingerprint_rows(script, package)
    except (OSError, SyntaxError, UnicodeError) as exc:
        msg = f"Nelze otisknout kód skriptu {script}: {exc}"
        raise InvalidInputError(msg) from exc
    return hashlib.sha256("\n".join(sorted(rows)).encode("utf-8")).hexdigest()


def _fingerprint_rows(script: Path, package: Path) -> list[str]:
    if script.suffix != ".py":
        return [f"script {script.name} {file_sha256(script)}"]

    finder = ModuleFinder(path=[str(package.parent)])
    finder.run_script(str(script))
    missing = sorted(
        name
        for name in finder.any_missing_maybe()[0]
        if _in_package(name, package.name)
    )
    if missing:
        msg = f"Nelze otisknout {script}: chybí moduly {', '.join(missing)}"
        raise InvalidInputError(msg)

    modules = {
        name: module
        for name, module in finder.modules.items()
        if _in_package(name, package.name)
    }
    if not modules:
        msg = f"Nelze otisknout {script}: žádný import z balíčku {package}"
        raise InvalidInputError(msg)

    rows = [f"python script {_python_sha256(script)}"]
    # Typeshed u Module neuvádí __file__, který ModuleFinder nastavuje při načtení.
    rows.extend(
        f"python {name} {_python_sha256(Path(vars(module)['__file__']))}"
        for name, module in modules.items()
    )
    rows.extend(
        f"data {path.relative_to(package).as_posix()} {file_sha256(path)}"
        for path in package.rglob("*")
        if path.is_file()
        and path.suffix != ".py"
        and "__pycache__" not in path.relative_to(package).parts
    )
    return rows


def _in_package(module: str, package: str) -> bool:
    return module == package or module.startswith(f"{package}.")


def _python_sha256(path: Path) -> str:
    # AST vynechá komentáře a polohu uzlů, aby formátování nespouštělo trénink.
    tree = ast.dump(ast.parse(path.read_text(encoding="utf-8")))
    return hashlib.sha256(tree.encode("utf-8")).hexdigest()
