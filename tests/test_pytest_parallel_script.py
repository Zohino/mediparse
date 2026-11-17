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
            import signal
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
            (root / f"{part}.sigint").write_text(
                str(signal.getsignal(signal.SIGINT) == signal.SIG_IGN)
            )
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
    assert (tmp_path / "download.sigint").read_text(encoding="utf-8") == "False"


def test_termination_stops_background_pytest_and_child(tmp_path: Path) -> None:
    """SIGTERM ukončí čekací část i její dítě a zachová výstup obou částí."""
    pid_file = tmp_path / "slow.pid"
    child_pid_file = tmp_path / "child.pid"
    ready_file = tmp_path / "ready"
    fake_pytest = tmp_path / "pytest"
    fake_pytest.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent("""\
            import os
            import signal
            import subprocess
            import sys
            import time
            from pathlib import Path

            child_pid_file = Path(os.environ["CHILD_PID_FILE"])
            if sys.argv[1] == "tests/test_download_script.py":
                child = subprocess.Popen(["sleep", "60"])
                signal.signal(signal.SIGTERM, lambda *_: None)
                print("slow partial output", flush=True)
                Path(os.environ["SLOW_PID_FILE"]).write_text(str(os.getpid()))
                child_pid_file.write_text(str(child.pid))
                child.wait()
                sys.exit(0)
            while not child_pid_file.exists():
                time.sleep(0.01)
            print("rest ready", flush=True)
            Path(os.environ["READY_FILE"]).touch()
            """),
        encoding="utf-8",
    )
    fake_pytest.chmod(0o755)

    with subprocess.Popen(
        ["bash", str(REPOSITORY / "scripts/pytest-parallel.sh")],
        env={
            **os.environ,
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "SLOW_PID_FILE": str(pid_file),
            "CHILD_PID_FILE": str(child_pid_file),
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
            with pytest.raises(ProcessLookupError):
                os.kill(int(child_pid_file.read_text(encoding="utf-8")), 0)
            assert output == "rest ready\nslow partial output\n"
        finally:
            if process.poll() is None:
                process.kill()
            with suppress(FileNotFoundError, ProcessLookupError):
                os.kill(int(child_pid_file.read_text(encoding="utf-8")), signal.SIGKILL)
            with suppress(FileNotFoundError, ProcessLookupError):
                os.kill(int(pid_file.read_text(encoding="utf-8")), signal.SIGKILL)
            process.communicate(timeout=5)
