"""Spouštěč pytestu čeká na obě souběžné části a zachová jejich výstup."""

from __future__ import annotations

import os
import subprocess
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
        "#!/usr/bin/env bash\n"
        '[[ "$#" == 2 && "$1" == "argument s mezerou" ]] || exit 90\n'
        'case "$2" in\n'
        "  tests/test_download_script.py) part=download; other=rest; "
        "code=$DOWNLOAD_CODE ;;\n"
        "  --ignore=tests/test_download_script.py) part=rest; other=download; "
        "code=$REST_CODE ;;\n"
        "  *) exit 91 ;;\n"
        "esac\n"
        'echo "$part start"\n'
        'touch "$RENDEZVOUS/$part"\n'
        'until [[ -f "$RENDEZVOUS/$other" ]]; do sleep 0.01; done\n'
        'echo "$part end" >&2\n'
        'exit "$code"\n',
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
    assert result.stdout in {
        "download start\ndownload end\nrest start\nrest end\n",
        "rest start\nrest end\ndownload start\ndownload end\n",
    }
