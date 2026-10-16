"""Architektura mimo dosah import-linteru: vstupy do balíčku, skripty workflow, umístění sdílených portů a konfigurace modelů domény."""

import ast
import importlib
import pkgutil
import re
import tomllib
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

import mediparse
import mediparse.domain
from mediparse.domain.cohort import CohortCounts
from tests.support import REPOSITORY

PACKAGE = Path(mediparse.__file__).parent
WORKFLOW = REPOSITORY / "workflow"
MAX_SCRIPT_LINES = 20
MAIN_GUARD = re.compile(r"__name__\s*==\s*[\"']__main__[\"']")
RUN_DIRECTIVE = re.compile(r"^\s*run\s*:", re.MULTILINE)
APPLICATION = PACKAGE / "application"
APPLICATION_PREFIX = "mediparse.application."


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


def _domain_models() -> list[type[BaseModel]]:
    models: list[type[BaseModel]] = []
    for info in pkgutil.walk_packages(
        mediparse.domain.__path__, prefix="mediparse.domain."
    ):
        module = importlib.import_module(info.name)
        models.extend(
            value
            for value in vars(module).values()
            if isinstance(value, type)
            and issubclass(value, BaseModel)
            and value.__module__ == module.__name__
        )
    return models


def test_domain_models_are_discovered() -> None:
    """Prázdná parametrizace by test jen přeskočila; objevení modelů hlídá zvlášť."""
    assert CohortCounts in _domain_models()


@pytest.mark.parametrize(
    "model", _domain_models(), ids=lambda model: f"{model.__module__}.{model.__name__}"
)
def test_domain_models_are_frozen_forbid_and_strict(model: type[BaseModel]) -> None:
    """Každý model domény je neměnný, odmítá cizí pole a nepřetypovává vstup."""
    config = model.model_config

    assert config.get("frozen") is True
    assert config.get("extra") == "forbid"
    assert config.get("strict") is True


def _protocol_names(module: Path) -> set[str]:
    tree = ast.parse(module.read_text(encoding="utf-8"))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and any(
            (isinstance(base, ast.Name) and base.id == "Protocol")
            or (isinstance(base, ast.Attribute) and base.attr == "Protocol")
            for base in node.bases
        )
    }


def _imported_module(module: Path, application: Path, node: ast.ImportFrom) -> str:
    """Úplné jméno importovaného modulu; relativní import se přeloží podle ``node.level``.

    Returns:
        Tečkované jméno modulu.
    """
    if node.level == 0:
        return node.module or ""
    package = ["mediparse", "application", *module.relative_to(application).parts[:-1]]
    base = package[: len(package) - (node.level - 1)]
    return ".".join([*base, *(node.module.split(".") if node.module else [])])


def _foreign_protocol_imports(application: Path) -> list[str]:
    """Importy portu z modulu jiného use casu; ``ports.*`` je jediné místo sdílených portů.

    Returns:
        Popisy porušení; prázdný seznam znamená, že pravidlo platí.
    """
    violations: list[str] = []
    modules = [
        module
        for module in sorted(application.rglob("*.py"))
        if module.relative_to(application).parts[0] != "ports"
    ]
    for module in modules:
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            imported = _imported_module(module, application, node)
            if not imported.startswith(APPLICATION_PREFIX):
                continue
            source = imported.removeprefix(APPLICATION_PREFIX)
            if source == "ports" or source.startswith("ports."):
                continue
            source_file = application / Path(*source.split(".")).with_suffix(".py")
            if not source_file.is_file():
                continue
            protocols = _protocol_names(source_file)
            violations.extend(
                f"{module.name} importuje port {alias.name} z {source}"
                for alias in node.names
                if alias.name in protocols
            )
    return violations


def test_shared_port_lives_in_ports_package() -> None:
    """Port, který používá víc use casů, leží v ``application.ports``, ne v modulu use casu."""
    assert any(APPLICATION.glob("*.py"))
    assert _foreign_protocol_imports(APPLICATION) == []


def test_use_case_may_import_function_of_another_use_case(tmp_path: Path) -> None:
    """Pravidlo se týká portů; funkci jiného use casu smí use case volat (S11s)."""
    (tmp_path / "gate.py").write_text(
        "from typing import Protocol\n\n"
        "class Port(Protocol):\n    ...\n\n"
        "def audit_violations() -> None:\n    ...\n",
        encoding="utf-8",
    )
    (tmp_path / "user.py").write_text(
        "from mediparse.application.gate import audit_violations\n", encoding="utf-8"
    )
    (tmp_path / "bad.py").write_text(
        "if True:\n    from mediparse.application.gate import Port\n",
        encoding="utf-8",
    )

    assert _foreign_protocol_imports(tmp_path) == ["bad.py importuje port Port z gate"]
