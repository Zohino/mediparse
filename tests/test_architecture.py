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
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import mediparse
import mediparse.domain
import mediparse.synthetic.domain
from mediparse.domain.cohort import MIN_CELL, CohortCounts
from mediparse.infrastructure.duckdb_parquet import SPILL_DIRECTORY
from mediparse.infrastructure.eda_duckdb import DEFAULT_MEMORY
from tests.support import REPOSITORY

if TYPE_CHECKING:
    from collections.abc import Iterable

PACKAGE = Path(mediparse.__file__).parent
WORKFLOW = REPOSITORY / "workflow"
MAX_SCRIPT_LINES = 20
MAIN_GUARD = re.compile(r"__name__\s*==\s*[\"']__main__[\"']")
RULE_IMPORT = re.compile(r"^\s*(?:from|import)\s+(mediparse[\w.]*)", re.MULTILINE)
ENTRYPOINTS_MODULES = ("mediparse.entrypoints", "mediparse.synthetic.entrypoints")
RUN_DIRECTIVE = re.compile(r"^\s*run\s*:", re.MULTILINE)
APPLICATIONS = (
    PACKAGE / "application",
    PACKAGE / "synthetic" / "application",
)
ENCODING_ROOTS = tuple(
    REPOSITORY / name for name in ("src", "tests", "workflow", "translation")
)
TEXT_IO_METHODS = frozenset({"open", "read_text", "write_text"})
SUBPROCESS_CALLS = frozenset({"run", "check_output", "check_call", "call", "Popen"})
ENCODING_POSITION = {"read_text": 0, "write_text": 1}
FILE_MODE = re.compile(r"[rwxabt+]+")

EDA = REPOSITORY / "eda"
EDA_PYTHON = re.compile(
    r"^\s*```\{python(?:[ ,}])|`\{python\}"
    r"|^\s*(?:engine:\s*[\"']?jupyter\b|jupyter\s*:)"
    r"|^\s*```\{r\b[^}\n]*\bengine\s*=\s*[\"']python[\"']",
    re.MULTILINE,
)
EDA_PRESENT = pytest.mark.skipif(
    not EDA.is_dir(),
    reason="eda/ není v obrazu kontejneru; hlídá job Test v CI",
)

ANALYSIS = REPOSITORY / "analysis"
ANALYSIS_PRESENT = pytest.mark.skipif(
    not ANALYSIS.is_dir(),
    reason="analysis/ není v obrazu kontejneru; hlídá job Test v CI",
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
    return any(
        module == entrypoints or module.startswith(f"{entrypoints}.")
        for entrypoints in ENTRYPOINTS_MODULES
    )


def _code_lines(path: Path) -> int:
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return sum(1 for line in lines if line and not line.startswith("#"))


def test_console_scripts_enter_through_entrypoints() -> None:
    """Konzolový skript vede jen do entrypoints jádra nebo syntetiky."""
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
        "from mediparse.synthetic.domain.x import y\n"
        "from mediparse.entrypoints.ok import fine\n"
        "from mediparse.synthetic.entrypoints.ok import fine\n",
        encoding="utf-8",
    )

    assert _rule_import_violations([rule]) == [
        "bad.smk importuje mediparse.domain.x",
        "bad.smk importuje mediparse",
        "bad.smk importuje mediparse.application.q",
        "bad.smk importuje mediparse.synthetic.domain.x",
    ]


def _rule_script_path(rule: Path, body: str) -> Path | None:
    """Najde Python skript nebo shellový skript workflow, který vyžaduje otisk.

    Returns:
        Absolutní cesta skriptu, nebo None pro pravidlo bez sledovaného skriptu.
    """
    script = re.search(r"^    script:\s*[\"']([^\"']+)[\"']", body, re.MULTILINE)
    if script:
        return (rule.parent / script[1]).resolve()
    shell = re.search(r"^    shell:([\s\S]*)", body, re.MULTILINE)
    if shell is None:
        return None
    path = re.search(r"\bworkflow/scripts/[\w.-]+", shell[1])
    if path is None:
        return None
    if path[0] == "workflow/scripts/download.sh":
        # Změna downloaderu nesmí znovu stahovat chráněné a ověřené vstupy.
        return None
    return (REPOSITORY / path[0]).resolve()


def _rule_code_violations(rule: Path) -> list[str]:
    """Najde pravidla, jejichž params.code neotiskuje vlastní skript.

    Returns:
        Jména porušujících pravidel; use rule bez script dědí otisk rodiče.
    """
    source = rule.read_text(encoding="utf-8")
    blocks = re.findall(
        r"^(?:rule (\w+):|use rule (\w+)(?: as (\w+))? with:)\n"
        r"((?:[ \t]+[^\n]*(?:\n|$)|\n)*)",
        source,
        re.MULTILINE,
    )
    violations = []
    for name, parent, alias, body in blocks:
        expected = _rule_script_path(rule, body)
        if expected is None:
            continue
        params = re.search(
            r"^    params:[ \t]*\n((?:        [^\n]*(?:\n|$)|\n)*)",
            body,
            re.MULTILINE,
        )
        code = re.search(
            r"^        code\s*=\s*code_fingerprint\([\"']([^\"']+)[\"']\)",
            params[1] if params else "",
            re.MULTILINE,
        )
        if code is None or (REPOSITORY / code[1]).resolve() != expected:
            violations.append(f"{rule.name}: {name or alias or parent}")
    return violations


def test_workflow_scripts_have_matching_code_fingerprint() -> None:
    """Pravidla s Pythonem i shellovými skripty mají v params otisk svého skriptu."""
    assert [
        violation
        for rule in _workflow_rules()
        for violation in _rule_code_violations(rule)
    ] == []


def test_rule_code_check_finds_missing_and_mismatched_fingerprints(
    tmp_path: Path,
) -> None:
    """Kontrola rozliší otisky Pythonu, shellu i vlastního skriptu děděného pravidla."""
    rule = tmp_path / "bad.smk"
    script = tmp_path / "run.py"
    rule.write_text(
        "rule missing:\n"
        "    script:\n"
        '        "run.py"\n'
        "rule mismatched:\n"
        "    params:\n"
        '        code=code_fingerprint("other.py"),\n'
        "    script:\n"
        '        "run.py"\n'
        "rule commented:\n"
        "    params:\n"
        f'        # code=code_fingerprint("{script}"),\n'
        "        other=1,\n"
        "    script:\n"
        '        "run.py"\n'
        "rule matching:\n"
        "    params:\n"
        f'        code=code_fingerprint("{script}"),\n'
        "    script:\n"
        '        "run.py"\n'
        "rule shell_missing:\n"
        "    shell:\n"
        '        "workflow/scripts/inventory.sh"\n'
        "rule shell_mismatched:\n"
        "    params:\n"
        '        code=code_fingerprint("workflow/scripts/other.sh"),\n'
        "    shell:\n"
        '        "workflow/scripts/inventory.sh"\n'
        "rule shell_matching:\n"
        "    params:\n"
        '        code=code_fingerprint("workflow/scripts/inventory.sh"),\n'
        "    shell:\n"
        '        "workflow/scripts/inventory.sh"\n'
        "rule download:\n"
        "    output:\n"
        '        protected("verified.csv"),\n'
        "    shell:\n"
        '        "workflow/scripts/download.sh"\n'
        "use rule matching as inherited with:\n"
        "    output:\n"
        '        "out.txt",\n'
        "use rule matching as custom_missing with:\n"
        "    script:\n"
        '        "run.py"\n'
        "use rule matching as custom_mismatched with:\n"
        "    params:\n"
        '        code=code_fingerprint("other.py"),\n'
        "    script:\n"
        '        "run.py"\n'
        "use rule matching as custom_matching with:\n"
        "    params:\n"
        f'        code=code_fingerprint("{script}"),\n'
        "    script:\n"
        '        "run.py"\n',
        encoding="utf-8",
    )

    assert _rule_code_violations(rule) == [
        "bad.smk: missing",
        "bad.smk: mismatched",
        "bad.smk: commented",
        "bad.smk: shell_missing",
        "bad.smk: shell_mismatched",
        "bad.smk: custom_missing",
        "bad.smk: custom_mismatched",
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
    for info in (
        info
        for domain in (mediparse.domain, mediparse.synthetic.domain)
        for info in pkgutil.walk_packages(domain.__path__, prefix=f"{domain.__name__}.")
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


def _imported_module(module: Path, node: ast.ImportFrom) -> str:
    """Úplné jméno importovaného modulu; relativní import se přeloží podle ``node.level``.

    Returns:
        Tečkované jméno modulu.
    """
    if node.level == 0:
        return node.module or ""
    package = module.relative_to(PACKAGE.parent).parts[:-1]
    base = package[: len(package) - (node.level - 1)]
    return ".".join([*base, *(node.module.split(".") if node.module else [])])


def _use_case_source(imported: str) -> tuple[Path, str] | None:
    for application in APPLICATIONS:
        prefix = ".".join(application.relative_to(PACKAGE.parent).parts) + "."
        if not imported.startswith(prefix):
            continue
        source = imported.removeprefix(prefix)
        if source == "ports" or source.startswith("ports."):
            return None
        source_file = application / Path(*source.split(".")).with_suffix(".py")
        if source_file.is_file():
            return source_file, source
    return None


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
            source = _use_case_source(_imported_module(module, node))
            if source is None:
                continue
            source_file, source_name = source
            protocols = _protocol_names(source_file)
            violations.extend(
                f"{module.name} importuje port {alias.name} z {source_name}"
                for alias in node.names
                if alias.name in protocols
            )
    return violations


@pytest.mark.parametrize("application", APPLICATIONS)
def test_shared_port_lives_in_ports_package(application: Path) -> None:
    """Port, který používá víc use casů, leží v ``application.ports``, ne v modulu use casu."""
    assert any(application.glob("*.py"))
    assert _foreign_protocol_imports(application) == []


def test_use_case_may_import_function_of_another_use_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pravidlo se týká portů; funkci jiného use casu smí use case volat (S11s)."""
    application = tmp_path / "mediparse" / "application"
    application.mkdir(parents=True)
    monkeypatch.setattr("tests.test_architecture.PACKAGE", application.parent)
    monkeypatch.setattr("tests.test_architecture.APPLICATIONS", (application,))
    (application / "gate.py").write_text(
        "from typing import Protocol\n\n"
        "class Port(Protocol):\n    ...\n\n"
        "def audit_violations() -> None:\n    ...\n",
        encoding="utf-8",
    )
    (application / "user.py").write_text(
        "from mediparse.application.gate import audit_violations\n", encoding="utf-8"
    )
    (application / "bad.py").write_text(
        "if True:\n    from mediparse.application.gate import Port\n",
        encoding="utf-8",
    )

    assert _foreign_protocol_imports(application) == [
        "bad.py importuje port Port z gate"
    ]


@pytest.mark.parametrize(
    "source", ["mediparse.application.gate", "...application.gate"]
)
def test_shared_port_rule_covers_imports_from_another_application(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    """Port jiného kontextu patří do ports; funkci jeho use casu volat lze."""
    package = tmp_path / "mediparse"
    application = package / "application"
    synthetic_application = package / "synthetic" / "application"
    application.mkdir(parents=True)
    synthetic_application.mkdir(parents=True)
    (application / "gate.py").write_text(
        "from typing import Protocol\n\n"
        "class Port(Protocol):\n    ...\n\n"
        "def audit_violations() -> None:\n    ...\n",
        encoding="utf-8",
    )
    (synthetic_application / "bad.py").write_text(
        f"from {source} import Port\n", encoding="utf-8"
    )
    (synthetic_application / "user.py").write_text(
        f"from {source} import audit_violations\n", encoding="utf-8"
    )
    monkeypatch.setattr("tests.test_architecture.PACKAGE", package)
    monkeypatch.setattr(
        "tests.test_architecture.APPLICATIONS",
        (application, synthetic_application),
    )

    assert _foreign_protocol_imports(synthetic_application) == [
        "bad.py importuje port Port z gate"
    ]


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


def _render_sources(root: Path) -> list[Path]:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.suffix in {".R", ".qmd"}
        and "_freeze" not in path.relative_to(root).parts
    )
    assert files
    return files


def _reticulate_violations(files: Iterable[Path]) -> list[str]:
    return [
        f"{file.name}:{index}: reticulate"
        for file in files
        for index, line in enumerate(
            file.read_text(encoding="utf-8").splitlines(), start=1
        )
        if re.search(r"\breticulate\b", line)
    ]


def _eda_python_violations(documents: list[Path]) -> list[str]:
    return [
        f"{document.name}:{index}: Python v EDA"
        for document in documents
        for index, line in enumerate(
            document.read_text(encoding="utf-8").splitlines(), start=1
        )
        if EDA_PYTHON.search(line)
    ]


@ANALYSIS_PRESENT
def test_analysis_metric_threshold_matches_cohort() -> None:
    """R i kohorta vyžadují stejné počty obou tříd kvůli stabilitě metrik."""
    source = (ANALYSIS / "R" / "metrics.R").read_text(encoding="utf-8")
    threshold = re.search(r"^METRIC_MIN_NOTES <- (\d+)L$", source, re.MULTILINE)
    assert threshold is not None
    assert int(threshold.group(1)) == MIN_CELL


@EDA_PRESENT
@ANALYSIS_PRESENT
def test_eda_and_analysis_do_not_source_each_other() -> None:
    """EDA a analýza sdílejí R funkce bez zpětných a křížových závislostí."""
    thesis_files = sorted((REPOSITORY / "R/thesis").rglob("*.R"))
    assert thesis_files
    inspected = set()
    violations = []
    for files, forbidden in (
        (_render_sources(EDA), "analysis"),
        (_render_sources(ANALYSIS), "eda"),
        (thesis_files, "(?:eda|analysis)"),
    ):
        inspected.update(file.suffix for file in files)
        pattern = re.compile(rf"""\b(?:sys\.)?source\s*\(\s*["'][^"']*\b{forbidden}/""")
        violations.extend(
            str(file.relative_to(REPOSITORY))
            for file in files
            if pattern.search(file.read_text(encoding="utf-8"))
        )

    assert {".qmd", ".R"} <= inspected
    assert violations == []


@EDA_PRESENT
@ANALYSIS_PRESENT
def test_eda_and_analysis_do_not_use_reticulate() -> None:
    """R a Quarto čtou soubory bez živého Pythonu přes reticulate."""
    for root in (EDA, ANALYSIS, REPOSITORY / "R/thesis"):
        assert _reticulate_violations(_render_sources(root)) == []


@EDA_PRESENT
def test_eda_documents_do_not_use_python() -> None:
    """Všechny sešity i přílohy používají R na knitru."""
    assert _eda_python_violations(_eda_documents(EDA)) == []


@pytest.mark.parametrize("suffix", [".R", ".qmd"])
def test_reticulate_check_finds_violations_outside_freeze(
    tmp_path: Path, suffix: str
) -> None:
    """Kontrola zachytí R i Quarto, zmražené výstupy se přitom neprohledávají."""
    source = 'x <- reticulate::import("mediparse.entrypoints.x")\n'
    (tmp_path / f"bad{suffix}").write_text(source, encoding="utf-8")
    (tmp_path / "_freeze").mkdir()
    (tmp_path / "_freeze/cache.R").write_text(source, encoding="utf-8")
    assert _reticulate_violations(_render_sources(tmp_path)) == [
        f"bad{suffix}:1: reticulate"
    ]


def test_eda_python_check_finds_all_three_forms(tmp_path: Path) -> None:
    """Kontrola zachytí chunk, inline výraz i engine Pythonu."""
    document = tmp_path / "bad.qmd"
    document.write_text(
        "engine: jupyter\n```{python}\nx = 1\n```\n`{python} x`\n",
        encoding="utf-8",
    )
    assert _eda_python_violations([document]) == [
        "bad.qmd:1: Python v EDA",
        "bad.qmd:2: Python v EDA",
        "bad.qmd:5: Python v EDA",
    ]


@pytest.mark.parametrize(
    "source",
    [
        'engine: "jupyter"',
        "engine: 'jupyter'",
        "jupyter: python3",
        "```{python echo=FALSE}",
        '```{r engine="python"}',
        '```{r, engine = "python"}',
        "```{r engine='python'}",
    ],
)
def test_eda_python_check_detects_engine_variants(tmp_path: Path, source: str) -> None:
    """Nastavení enginu ani volby chunku nesmějí obejít zákaz Pythonu."""
    document = tmp_path / "bad.qmd"
    document.write_text(source + "\n", encoding="utf-8")
    violations = _eda_python_violations([document])
    assert violations == ["bad.qmd:1: Python v EDA"]


def test_eda_python_check_accepts_r_document(tmp_path: Path) -> None:
    """Čistý QMD na knitru nemá žádný nález."""
    document = tmp_path / "fine.qmd"
    document.write_text("```{r}\nx <- 1\n```\n`r x`\n", encoding="utf-8")
    assert _eda_python_violations([document]) == []


def test_eda_python_check_refuses_uninspected_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kontrola nad repem nesmí zezelenat bez jediného dokumentu."""
    monkeypatch.setattr("tests.test_architecture.EDA", tmp_path)
    with pytest.raises(AssertionError):
        test_eda_documents_do_not_use_python()


def test_reticulate_check_requires_sources(tmp_path: Path) -> None:
    """Kontrola nesmí zezelenat bez jediného souboru R nebo Quarto."""
    with pytest.raises(AssertionError):
        _render_sources(tmp_path)


@EDA_PRESENT
def test_eda_connection_constants_match_python() -> None:
    """R používá stejný paměťový limit a spill jako Python."""
    source = (EDA / "R/db.R").read_text(encoding="utf-8")
    memory = re.search(r'memory <- "([^"\n]+)"', source)
    spill = re.search(r'temp <- file.path\(dirname\(directory\), "([^"\n]+)"\)', source)
    assert memory is not None
    assert spill is not None
    assert memory.group(1) == DEFAULT_MEMORY
    assert spill.group(1) == SPILL_DIRECTORY


@EDA_PRESENT
def test_eda_stop_words_match_sklearn() -> None:
    """EDA drží přesně seřazený seznam anglických stop slov sklearn."""
    words = (EDA / "stop-words.txt").read_text(encoding="utf-8").splitlines()
    assert words == sorted(ENGLISH_STOP_WORDS)


DATA_ADAPTER = re.compile(
    r"(?:Parquet\w+|DuckDb\w+|SkopsModelFile|CsvParquetConversion)"
)
DATA_ACCESS_IMPORTS = frozenset({
    ("mediparse.infrastructure.data_access_guard", "check_inputs_access"),
    ("mediparse.infrastructure.eda_duckdb", "EdaDatabase"),
})
DATA_ACCESS_EXCEPTIONS = {
    "training_input.py": (
        "Skládá líný adaptér bez čtení dat; stráž volají model_training a model_inference."
    ),
}


def _infrastructure_imports(module: Path) -> set[tuple[str, str]]:
    tree = ast.parse(module.read_text(encoding="utf-8"))
    return {
        (_imported_module(module, node), alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and _imported_module(module, node).startswith("mediparse.infrastructure.")
        for alias in node.names
    }


def _unguarded_data_entrypoints(root: Path) -> tuple[list[str], set[str]]:
    violations = []
    inspected = set()
    for module in sorted(root.rglob("*.py")):
        imports = _infrastructure_imports(module)
        if not any(DATA_ADAPTER.fullmatch(name) for _, name in imports):
            continue
        name = module.relative_to(root).as_posix()
        inspected.add(name)
        if name not in DATA_ACCESS_EXCEPTIONS and imports.isdisjoint(
            DATA_ACCESS_IMPORTS
        ):
            violations.append(name)
    return violations, inspected


def test_data_reading_entrypoints_import_access_guard() -> None:
    """Entrypoint s datovým adaptérem zná přímou stráž nebo stráženou databázi EDA."""
    violations, inspected = _unguarded_data_entrypoints(PACKAGE / "entrypoints")

    assert "cohort.py" in inspected
    assert DATA_ACCESS_EXCEPTIONS.keys() <= inspected
    assert violations == []


@pytest.mark.parametrize(
    ("source", "adapter"),
    [
        ("note_table", "ParquetNoteTable"),
        ("note_texts_parquet", "ParquetNoteTexts"),
        ("duckdb_cohort", "DuckDbCohort"),
        ("sklearn_classifier", "SkopsModelFile"),
        ("csv_parquet", "CsvParquetConversion"),
    ],
)
def test_data_guard_check_detects_unprotected_adapter(
    tmp_path: Path, source: str, adapter: str
) -> None:
    """Stráž vyžadují různé datové adaptéry i s přejmenovaným importem."""
    (tmp_path / "unguarded.py").write_text(
        f"from mediparse.infrastructure.{source} import {adapter} as Adapter\n",
        encoding="utf-8",
    )

    violations, inspected = _unguarded_data_entrypoints(tmp_path)

    assert violations == ["unguarded.py"]
    assert inspected == {"unguarded.py"}


@pytest.mark.parametrize(
    "guard",
    [
        "data_access_guard import check_inputs_access as check",
        "eda_duckdb import EdaDatabase as Database",
    ],
)
def test_data_guard_check_accepts_both_guard_imports(
    tmp_path: Path, guard: str
) -> None:
    """Přímá stráž i databáze EDA plní kontrakt při importu pod aliasem."""
    (tmp_path / "guarded.py").write_text(
        "from mediparse.infrastructure.note_table import ParquetNoteTable\n"
        f"from mediparse.infrastructure.{guard}\n",
        encoding="utf-8",
    )

    violations, inspected = _unguarded_data_entrypoints(tmp_path)

    assert violations == []
    assert inspected == {"guarded.py"}
