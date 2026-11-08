"""Předání prostředí CI skutečným spouštěčem kontejnerového smoketestu."""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest

from tests.git_support import git_environment, init_repo, needs_git
from tests.support import REPOSITORY

if TYPE_CHECKING:
    from pathlib import Path


@needs_git
@pytest.mark.parametrize("ci", [None, "true"])
def test_smoketest_forwards_ci_without_setting_it(
    tmp_path: Path, ci: str | None
) -> None:
    """Spouštěč předá volbu CI kontejneru a zachová nastavení hostitele i jeho absenci."""
    root = tmp_path / "repo"
    init_repo(root, "synthetic repository\n")
    script = root / "smoketest.sh"
    shutil.copyfile(REPOSITORY / "smoketest.sh", script)
    calls = tmp_path / "engine-calls.jsonl"
    engine = tmp_path / "container-engine"
    engine.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['ENGINE_CALLS'], 'a', encoding='utf-8') as log:\n"
        "    log.write(json.dumps({'args': sys.argv[1:], 'ci': os.environ.get('CI')}) + '\\n')\n",
        encoding="utf-8",
    )
    engine.chmod(0o755)
    environ = git_environment()
    environ.pop("CI", None)
    if ci is not None:
        environ["CI"] = ci
    environ.update(CONTAINER_ENGINE=str(engine), ENGINE_CALLS=str(calls))

    done = subprocess.run(
        ["sh", str(script)],
        env=environ,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert done.returncode == 0, done.stderr
    build, run = [
        json.loads(line) for line in calls.read_text(encoding="utf-8").splitlines()
    ]
    assert build["args"][0] == "build"
    arguments = run["args"]
    assert arguments[0] == "run"
    assert "-e" in arguments
    assert arguments[arguments.index("-e") + 1] == "CI"
    assert run["ci"] == ci
