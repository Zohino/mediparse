"""Architektura mimo dosah import-linteru: vstupy do balíčku, skripty workflow, umístění sdílených portů a konfigurace modelů domény a explicitní kódování textového I/O."""

from __future__ import annotations

import ast
import importlib
import pkgutil
import re
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel

import mediparse
import mediparse.domain
from mediparse.domain.cohort import CohortCounts
from mediparse.domain.dua import MIN_CELL
from tests.support import REPOSITORY

if TYPE_CHECKING:
    from collections.abc import Iterable

PACKAGE = Path(mediparse.__file__).parent
WORKFLOW = REPOSITORY / "workflow"
MAX_SCRIPT_LINES = 20
MAIN_GUARD = re.compile(r"__name__\s*==\s*[\"']__main__[\"']")
RULE_IMPORT = re.compile(r"^\s*(?:from|import)\s+(mediparse[\w.]*)", re.MULTILINE)
ENTRYPOINTS_MODULE = "mediparse.entrypoints"
RUN_DIRECTIVE = re.compile(r"^\s*run\s*:", re.MULTILINE)
APPLICATION = PACKAGE / "application"
APPLICATION_PREFIX = "mediparse.application."
ENCODING_ROOTS = tuple(
    REPOSITORY / name for name in ("src", "tests", "workflow", "translation")
)
TEXT_IO_METHODS = frozenset({"open", "read_text", "write_text"})
SUBPROCESS_CALLS = frozenset({"run", "check_output", "check_call", "call", "Popen"})
ENCODING_POSITION = {"read_text": 0, "write_text": 1}
FILE_MODE = re.compile(r"[rwxabt+]+")

EDA = REPOSITORY / "eda"
# Import-linter nečte Quarto ani R; stejně jako u Snakemake hlídáme text importů.
R_IMPORT = re.compile(r"\bimport\(\s*[\"']([\w.]+)[\"']\s*\)")
PYTHON_IMPORT = re.compile(r"^\s*(?:from|import)\s+([\w.]+)", re.MULTILINE)
EDA_PYTHON = re.compile(
    r"(?m)^\s*```\{python(?:[ ,}])|`\{python\}|^\s*engine:\s*jupyter\b"
)
EDA_PRESENT = pytest.mark.skipif(
    not EDA.is_dir(),
    reason="eda/ není v obrazu kontejneru; hlídá job Test v CI",
)


def _pyproject() -> dict[str, Any]:
    return tomllib.loads((REPOSITORY / "pyproject.toml").read_text(encoding="utf-8"))


def _workflow_scripts() -> list[Path]:
    return sorted((WORKFLOW / "scripts").glob("*.py"))


def _workflow_rules() -> list[Path]:
    snakefile = WORKFLOW / "Snakefile"
    rules = sorted((WORKFLOW / "rules").glob("*.smk"))
    assert snakefile.is_file()
    assert rules
    return [snakefile, *rules]


def _in_entrypoints(module: str) -> bool:
    return module == ENTRYPOINTS_MODULE or module.startswith(f"{ENTRYPOINTS_MODULE}.")


def _code_lines(path: Path) -> int:
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return sum(1 for line in lines if line and not line.startswith("#"))


def test_console_scripts_enter_through_entrypoints() -> None:
    """Konzolový skript vede jen do mediparse.entrypoints, jinudy se do balíčku nevstupuje."""
    targets = _pyproject()["project"]["scripts"].values()

    assert targets
    assert [
        target for target in targets if not _in_entrypoints(target.split(":")[0])
    ] == []


def test_console_scripts_resolve_to_callables() -> None:
    """Překlep v cíli skriptu by jinak spadl až při spuštění nástroje."""
    targets = _pyproject()["project"]["scripts"].values()

    assert targets
    assert [
        target for target in targets if not callable(pkgutil.resolve_name(target))
    ] == []


def _entry_violations(root: Path) -> list[str]:
    """Moduly, které dělají z balíčku vstup přes `python -m`: `__main__.py` nebo guard.

    Returns:
        Cesty porušujících souborů relativně ke kořeni; prázdný seznam znamená, že pravidlo platí.
    """
    return [
        module.relative_to(root).as_posix()
        for module in sorted(root.rglob("*.py"))
        if module.name == "__main__.py"
        or MAIN_GUARD.search(module.read_text(encoding="utf-8"))
    ]


def test_package_has_no_python_m_entry() -> None:
    """Balíček nemá vstup přes `python -m`; vstupem jsou konzolové skripty a Snakemake."""
    assert any(PACKAGE.rglob("*.py"))
    assert _entry_violations(PACKAGE) == []


def test_main_module_is_an_entry_violation(tmp_path: Path) -> None:
    """Soubor `__main__.py` je porušení i bez guardu, guard v jiném modulu také."""
    (tmp_path / "__main__.py").write_text("print('x')\n", encoding="utf-8")
    (tmp_path / "guarded.py").write_text(
        'if __name__ == "__main__":\n    pass\n', encoding="utf-8"
    )
    (tmp_path / "fine.py").write_text("X = 1\n", encoding="utf-8")

    assert _entry_violations(tmp_path) == ["__main__.py", "guarded.py"]


def test_workflow_scripts_stay_thin() -> None:
    """Skript Snakemake jen rozbalí vstupy pravidla a zavolá vstupní bod."""
    scripts = _workflow_scripts()

    assert scripts
    assert [
        script.name for script in scripts if _code_lines(script) > MAX_SCRIPT_LINES
    ] == []


def test_workflow_rules_have_no_inline_python() -> None:
    """Blok `run:` by skryl Python před import-linterem, ruffem i ty."""
    rules = _workflow_rules()

    assert [
        rule.name
        for rule in rules
        if RUN_DIRECTIVE.search(rule.read_text(encoding="utf-8"))
    ] == []


def _rule_import_violations(rules: list[Path]) -> list[str]:
    """Importy `mediparse` v pravidlech Snakemake mimo `mediparse.entrypoints`.

    Soubory `.smk` nejsou Python, takže import-linter je nečte; hlídá se první řádek
    importu a odsazený import uvnitř pravidla se počítá také.

    Returns:
        Popisy porušení; prázdný seznam znamená, že pravidlo platí.
    """
    return [
        f"{rule.name} importuje {module}"
        for rule in rules
        for module in RULE_IMPORT.findall(rule.read_text(encoding="utf-8"))
        if not _in_entrypoints(module)
    ]


def test_workflow_rules_import_only_entrypoints() -> None:
    """Pravidla Snakemake vstupují do balíčku jen přes entrypoints, jako skripty workflow."""
    assert _rule_import_violations(_workflow_rules()) == []


def test_rule_import_check_finds_violations(tmp_path: Path) -> None:
    """Kontrola najde import domény, holý import balíčku i odsazený import."""
    rule = tmp_path / "bad.smk"
    rule.write_text(
        "from mediparse.domain.x import y\n"
        "import mediparse\n"
        "rule a:\n"
        "    input:\n"
        "        z\n"
        "if True:\n"
        "    from mediparse.application.q import w\n"
        "from mediparse.entrypoints.ok import fine\n",
        encoding="utf-8",
    )

    assert _rule_import_violations([rule]) == [
        "bad.smk importuje mediparse.domain.x",
        "bad.smk importuje mediparse",
        "bad.smk importuje mediparse.application.q",
    ]


def test_import_linter_sees_workflow_scripts() -> None:
    """Skripty workflow musí import-linter vidět jako root package."""
    root_packages = _pyproject()["tool"]["importlinter"]["root_packages"]

    assert "workflow" in root_packages


def test_import_linter_sees_type_checking_imports() -> None:
    """Typ je také závislost: import pod ``TYPE_CHECKING`` nesmí obejít vrstvy."""
    config = _pyproject()["tool"]["importlinter"]

    assert config.get("exclude_type_checking_imports", False) is False


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


def _is_binary_mode(node: ast.expr | None) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and FILE_MODE.fullmatch(node.value) is not None
        and "b" in node.value
    )


def _called_name(call: ast.Call) -> str:
    """Jméno volané funkce nebo metody bez ohledu na to, nad čím se volá.

    Returns:
        ``open`` pro ``open()`` i ``gzip.open()``; prázdný řetězec u jiných volání.
    """
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _keywords(call: ast.Call) -> dict[str, ast.expr]:
    return {item.arg: item.value for item in call.keywords if item.arg}


def _is_true(node: ast.expr | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _is_text_subprocess(call: ast.Call) -> bool:
    keywords = _keywords(call)
    return _is_true(keywords.get("text")) or _is_true(
        keywords.get("universal_newlines")
    )


def _lacks_encoding(call: ast.Call, name: str) -> bool | None:
    """Zda textové I/O volání chybí kódování; ``None`` pro volání, které I/O není.

    Jména se berou bez vlastníka, aby je nezakrylo ``from subprocess import run``
    ani ``import subprocess as sp``.

    Returns:
        ``True`` pro volání bez ``encoding``, ``False`` pro volání s ním nebo
        v binárním režimu, ``None`` pro volání mimo pravidlo.
    """
    keywords = _keywords(call)
    if name in SUBPROCESS_CALLS:
        return "encoding" not in keywords if _is_text_subprocess(call) else None
    if name not in TEXT_IO_METHODS:
        return None
    if name == "open" and (
        _is_binary_mode(keywords.get("mode"))
        or any(_is_binary_mode(arg) for arg in call.args)
    ):
        return False
    encoding_position = ENCODING_POSITION.get(name)
    positional = encoding_position is not None and len(call.args) > encoding_position
    return "encoding" not in keywords and not positional


def _module_encoding_findings(module: Path, root: Path) -> tuple[list[str], int]:
    found: list[str] = []
    inspected = 0
    tree = ast.parse(module.read_text(encoding="utf-8"))
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        name = _called_name(call)
        lacks = _lacks_encoding(call, name)
        if lacks is None:
            continue
        inspected += 1
        if lacks:
            where = module.relative_to(root.parent).as_posix()
            found.append(f"{where}:{call.lineno} {name}")
    return found, inspected


def _encoding_violations(roots: Iterable[Path]) -> tuple[list[str], int]:
    """Textové I/O bez ``encoding`` v souborech ``*.py`` pod kořeny.

    Hlídá ``open``, ``read_text``, ``write_text`` a ``subprocess`` s ``text=True``;
    binární režim ``open`` zadaný literálem přeskakuje. Ruff PLW1514 tato volání nad cestami
    nehlásí, protože nezná typ objektu.

    Args:
        roots: Existující adresáře k projití.

    Returns:
        Nálezy ``soubor:řádek volání`` relativně k rodiči kořene a počet prošlých
        volání I/O (stráž proti pravidlu, které nic nehlídá).
    """
    found: list[str] = []
    inspected = 0
    for root in roots:
        for module in sorted(root.rglob("*.py")):
            module_found, module_inspected = _module_encoding_findings(module, root)
            found.extend(module_found)
            inspected += module_inspected
    return found, inspected


def test_text_io_names_its_encoding() -> None:
    """Textové I/O v repu uvádí ``encoding``; locale stroje by jinak rozhodla o češtině."""
    violations, inspected = _encoding_violations(ENCODING_ROOTS)

    assert all(root.is_dir() for root in ENCODING_ROOTS)
    assert inspected > 0
    assert violations == []


def test_encoding_check_finds_text_io_without_encoding(tmp_path: Path) -> None:
    """Čtení, ``open``, ``subprocess`` s ``text=True`` (i po ``from subprocess import``) a ``write_text('rb')`` bez kódování jsou nález."""
    (tmp_path / "bad.py").write_text(
        "import subprocess\n"
        "from subprocess import run\n"
        "p.read_text()\n"
        "open(p)\n"
        "subprocess.run(['git'], text=True)\n"
        "p.write_text('rb')\n"
        "run(['git'], text=True)\n",
        encoding="utf-8",
    )

    violations, inspected = _encoding_violations([tmp_path])

    assert violations == [
        f"{tmp_path.name}/bad.py:3 read_text",
        f"{tmp_path.name}/bad.py:4 open",
        f"{tmp_path.name}/bad.py:5 run",
        f"{tmp_path.name}/bad.py:6 write_text",
        f"{tmp_path.name}/bad.py:7 run",
    ]
    assert inspected == len(violations)


def test_encoding_check_skips_binary_and_explicit_encoding(tmp_path: Path) -> None:
    """Binární režim a ``encoding='utf-8'`` nález nedají, počítají se ale jako prošlá volání."""
    checked_calls = 5
    (tmp_path / "fine.py").write_text(
        "import gzip\n"
        "import subprocess\n"
        "gzip.open(p, 'rb')\n"
        "p.open(mode='wb')\n"
        "p.read_bytes()\n"
        "p.read_text(encoding='utf-8')\n"
        "p.write_text('x', encoding='utf-8')\n"
        "subprocess.run(['git'], text=True, encoding='utf-8')\n"
        "subprocess.run(['git'])\n",
        encoding="utf-8",
    )

    violations, inspected = _encoding_violations([tmp_path])

    assert violations == []
    assert inspected == checked_calls


def _eda_documents(root: Path) -> list[Path]:
    documents = sorted(root.rglob("*.qmd"))
    assert documents
    return documents


def _eda_files(root: Path) -> list[Path]:
    documents = _eda_documents(root)
    r_files = sorted(root.rglob("*.R"))
    assert r_files
    return [*documents, *r_files]


def _eda_imports(file: Path) -> list[str]:
    text = file.read_text(encoding="utf-8")
    return [*R_IMPORT.findall(text), *PYTHON_IMPORT.findall(text)]


def _eda_import_violations(files: list[Path]) -> list[str]:
    return [
        f"{file.name} importuje {module}"
        for file in files
        for module in _eda_imports(file)
        if (module == "mediparse" or module.startswith("mediparse."))
        and not _in_entrypoints(module)
    ]


def _eda_python_violations(documents: list[Path]) -> tuple[list[str], int]:
    violations = [
        f"{document.name}:{index}: Python v EDA"
        for document in documents
        for index, line in enumerate(
            document.read_text(encoding="utf-8").splitlines(), start=1
        )
        if EDA_PYTHON.search(line)
    ]
    return violations, len(documents)


@EDA_PRESENT
def test_eda_small_cell_threshold_matches_domain() -> None:
    """Python i R potlačují malé buňky podle stejného projektového prahu."""
    source = (EDA / "R" / "dua.R").read_text(encoding="utf-8")
    threshold = re.search(r"DUA_THRESHOLD <- (\d+)L", source)
    assert threshold is not None
    assert int(threshold.group(1)) == MIN_CELL


@EDA_PRESENT
def test_eda_imports_only_entrypoints() -> None:
    """EDA vstupuje do balíčku přes stejné hranice jako pravidla workflow."""
    assert _eda_import_violations(_eda_files(EDA)) == []


@EDA_PRESENT
def test_eda_documents_do_not_use_python() -> None:
    """Všechny sešity i přílohy používají R na knitru."""
    violations, inspected = _eda_python_violations(_eda_documents(EDA))
    assert inspected > 0
    assert violations == []


def test_eda_import_check_finds_violations(tmp_path: Path) -> None:
    """Reticulate v R smí vstupovat jen přes entrypointy."""
    document = tmp_path / "bad.qmd"
    document.write_text(
        '```{r}\nx <- import("mediparse.domain.x")\n```\n', encoding="utf-8"
    )
    r_file = tmp_path / "bad.R"
    r_file.write_text(
        'x <- reticulate::import("mediparse.infrastructure.x")\n'
        'y <- import("mediparse.entrypoints.ok")\n',
        encoding="utf-8",
    )
    assert _eda_import_violations(_eda_files(tmp_path)) == [
        "bad.qmd importuje mediparse.domain.x",
        "bad.R importuje mediparse.infrastructure.x",
    ]


def test_eda_python_check_finds_all_three_forms(tmp_path: Path) -> None:
    """Kontrola zachytí chunk, inline výraz i engine Pythonu."""
    document = tmp_path / "bad.qmd"
    document.write_text(
        "engine: jupyter\n```{python}\nx = 1\n```\n`{python} x`\n",
        encoding="utf-8",
    )
    assert _eda_python_violations([document]) == (
        [
            "bad.qmd:1: Python v EDA",
            "bad.qmd:2: Python v EDA",
            "bad.qmd:5: Python v EDA",
        ],
        1,
    )


def test_eda_python_check_accepts_r_document(tmp_path: Path) -> None:
    """Čistý QMD se počítá jako prošlý dokument bez nálezu."""
    document = tmp_path / "fine.qmd"
    document.write_text("```{r}\nx <- 1\n```\n`r x`\n", encoding="utf-8")
    assert _eda_python_violations([document]) == ([], 1)
    assert _eda_python_violations([]) == ([], 0)


def test_eda_python_check_refuses_uninspected_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kontrola nad repem nesmí zezelenat bez jediného dokumentu."""
    monkeypatch.setattr("tests.test_architecture.EDA", tmp_path)
    with pytest.raises(AssertionError):
        test_eda_documents_do_not_use_python()


def test_eda_import_check_requires_r_files(tmp_path: Path) -> None:
    """Dokud se kontrolují i R importy, jejich seznam nesmí být prázdný."""
    (tmp_path / "only.qmd").write_text("Text.\n", encoding="utf-8")
    with pytest.raises(AssertionError):
        _eda_files(tmp_path)
