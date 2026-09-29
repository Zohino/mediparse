"""Architektura mimo dosah import-linteru: vstupy do balíčku a skripty workflow."""

import re
import tomllib
from pathlib import Path
from typing import Any

import mediparse

REPOSITORY = Path(__file__).parents[1]
PACKAGE = Path(mediparse.__file__).parent
WORKFLOW = REPOSITORY / "workflow"
MAX_SCRIPT_LINES = 20
MAIN_GUARD = re.compile(r"__name__\s*==\s*[\"']__main__[\"']")
RUN_DIRECTIVE = re.compile(r"^\s*run\s*:", re.MULTILINE)


def _pyproject() -> dict[str, Any]:
    return tomllib.loads((REPOSITORY / "pyproject.toml").read_text(encoding="utf-8"))


def _workflow_scripts() -> list[Path]:
    return sorted((WORKFLOW / "scripts").glob("*.py"))


def _code_lines(path: Path) -> int:
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return sum(1 for line in lines if line and not line.startswith("#"))


def test_console_scripts_enter_through_entrypoints() -> None:
    """Konzolový skript vede jen do mediparse.entrypoints, jinudy se do balíčku nevstupuje."""
    targets = _pyproject()["project"]["scripts"].values()

    assert targets
    assert [
        target for target in targets if not target.startswith("mediparse.entrypoints.")
    ] == []


def test_package_has_no_main_guard() -> None:
    """Balíček nemá vstup přes `python -m`; vstupem jsou konzolové skripty a Snakemake."""
    modules = sorted(PACKAGE.rglob("*.py"))

    assert modules
    assert [
        module.name
        for module in modules
        if MAIN_GUARD.search(module.read_text(encoding="utf-8"))
    ] == []


def test_workflow_scripts_stay_thin() -> None:
    """Skript Snakemake jen rozbalí vstupy pravidla a zavolá vstupní bod."""
    assert [
        script.name
        for script in _workflow_scripts()
        if _code_lines(script) > MAX_SCRIPT_LINES
    ] == []


def test_workflow_rules_have_no_inline_python() -> None:
    """Blok `run:` by skryl Python před import-linterem, ruffem i ty."""
    rules = [WORKFLOW / "Snakefile", *sorted((WORKFLOW / "rules").glob("*.smk"))]

    assert [
        rule.name
        for rule in rules
        if rule.is_file() and RUN_DIRECTIVE.search(rule.read_text(encoding="utf-8"))
    ] == []


def test_import_linter_sees_workflow_scripts() -> None:
    """Jakmile vznikne skript workflow, import-linter ho musí vidět jako root package."""
    root_packages = _pyproject()["tool"]["importlinter"]["root_packages"]

    assert not _workflow_scripts() or "workflow" in root_packages
