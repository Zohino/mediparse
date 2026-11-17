"""Spouštěč pytestu čeká na obě souběžné části a zachová jejich výstup."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from contextlib import suppress
from typing import TYPE_CHECKING

import pytest

from tests.support import REPOSITORY

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("download_code", "rest_code", "success"),
    [(0, 0, True), (1, 0, False), (0, 2, False), (1, 2, False)],
)
def test_parallel_pytest(
    tmp_path: Path,
    download_code: int,
    rest_code: int,
    *,
    success: bool,
) -> None:
    """Nula platí právě pro dvě úspěšné části; argumenty i bloky zůstanou celé."""
    fake_pytest = tmp_path / "pytest"
    fake_pytest.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent("""\
            import os
            import sys
            import time
            from pathlib import Path

            if len(sys.argv) != 3 or sys.argv[1] != "argument s mezerou":
                sys.exit(90)
            part, other = {
                "tests/test_download_script.py": ("download", "rest"),
                "--ignore=tests/test_download_script.py": ("rest", "download"),
            }[sys.argv[2]]
            root = Path(os.environ["RENDEZVOUS"])
            (root / f"{part}.nice").write_text(str(os.nice(0)))
            print(f"{part} start", flush=True)
            (root / part).touch()
            while not (root / other).exists():
                time.sleep(0.01)
            print(f"{part} end", file=sys.stderr, flush=True)
            sys.exit(int(os.environ[f"{part.upper()}_CODE"]))
            """),
        encoding="utf-8",
    )
    fake_pytest.chmod(0o755)

    result = subprocess.run(
        ["bash", str(REPOSITORY / "scripts/pytest-parallel.sh"), "argument s mezerou"],
        env={
            **os.environ,
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "RENDEZVOUS": str(tmp_path),
            "DOWNLOAD_CODE": str(download_code),
            "REST_CODE": str(rest_code),
        },
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
    )

    assert (result.returncode == 0) is success, result.stderr
    assert result.stdout == "rest start\nrest end\ndownload start\ndownload end\n"
    assert int((tmp_path / "rest.nice").read_text(encoding="utf-8")) > int(
        (tmp_path / "download.nice").read_text(encoding="utf-8")
    )


def test_termination_stops_background_pytest(tmp_path: Path) -> None:
    """SIGTERM ukončí čekací část a zachová výstup obou částí."""
    pid_file = tmp_path / "slow.pid"
    ready_file = tmp_path / "ready"
    fake_pytest = tmp_path / "pytest"
    fake_pytest.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == tests/test_download_script.py ]]; then\n'
        '  echo "slow partial output"\n'
        '  echo "$$" > "$SLOW_PID_FILE"\n'
        "  exec sleep 60\n"
        "fi\n"
        'until [[ -s "$SLOW_PID_FILE" ]]; do sleep 0.01; done\n'
        'echo "rest ready"\n'
        'touch "$READY_FILE"\n',
        encoding="utf-8",
    )
    fake_pytest.chmod(0o755)

    with subprocess.Popen(
        ["bash", str(REPOSITORY / "scripts/pytest-parallel.sh")],
        env={
            **os.environ,
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "SLOW_PID_FILE": str(pid_file),
            "READY_FILE": str(ready_file),
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    ) as process:
        try:
            deadline = time.monotonic() + 5
            while not ready_file.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert ready_file.exists(), "Zbytek testů nedoběhl souběžně s čekací částí."
            process.terminate()
            output, _ = process.communicate(timeout=5)

            assert process.returncode != 0
            with pytest.raises(ProcessLookupError):
                os.kill(int(pid_file.read_text(encoding="utf-8")), 0)
            assert output == "rest ready\nslow partial output\n"
        finally:
            if process.poll() is None:
                process.kill()
            with suppress(FileNotFoundError, ProcessLookupError):
                os.kill(int(pid_file.read_text(encoding="utf-8")), signal.SIGKILL)
            process.communicate(timeout=5)
