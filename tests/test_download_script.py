"""Stažení souboru z PhysioNetu skriptem workflow/scripts/download.sh.

Skript běží proti lokálnímu HTTP serveru, který napodobuje chování PhysioNetu:
Basic auth z ``.netrc``, Range, spojení, které po pár bajtech přestane posílat
data, a odmítnutí 401, 403 a 404.
"""

from __future__ import annotations

import base64
import os
import subprocess
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

SCRIPT: Final = Path(__file__).parents[1] / "workflow" / "scripts" / "download.sh"
PAYLOAD: Final = bytes(range(256)) * 12_000
LOGIN: Final = "user"
PASSWORD: Final = "secret"
PARTIAL_BYTES: Final = 500_000
# aria2c navazuje kousek před koncem rozpracovaného souboru (z 500 000 B od 491 520).
RESUME_SLACK: Final = 16 * 1024
QUICK_RETRIES: Final = {
    "MEDIPARSE_DOWNLOAD_TIMEOUT": "2",
    "MEDIPARSE_DOWNLOAD_ATTEMPTS": "2",
    "MEDIPARSE_DOWNLOAD_RETRY_WAIT": "0",
}


class FakePhysioNet(ThreadingHTTPServer):
    """Server s jedním souborem; cesta požadavku volí chování."""

    daemon_threads = True

    def __init__(self) -> None:
        """Naslouchá na volném portu localhostu."""
        super().__init__(("127.0.0.1", 0), _Handler)
        self.ranges: list[str | None] = []
        self.stalls_left = 1
        self.release = threading.Event()
        self.lock = threading.Lock()

    def url(self, path: str) -> str:
        """Adresa souboru na tomto serveru.

        Returns:
            URL ``path`` na portu, který server dostal.
        """
        return f"http://127.0.0.1:{self.server_address[1]}/{path}"


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def physionet(self) -> FakePhysioNet:
        """Server, který požadavek obsluhuje."""
        return cast("FakePhysioNet", self.server)

    def log_message(self, format: str, *args: object) -> None:  # ruff: ignore[builtin-argument-shadowing]
        """Bez výpisu požadavků do stderr testu."""

    def do_GET(self) -> None:
        """Odpoví podle cesty: ``data``, ``stall``, ``dead``, ``missing``, ``forbidden``."""
        if self.path == "/missing":
            self._empty(HTTPStatus.NOT_FOUND)
            return
        if self.path == "/forbidden":
            self._empty(HTTPStatus.FORBIDDEN)
            return
        if not self._authorized():
            self._empty(
                HTTPStatus.UNAUTHORIZED, {"WWW-Authenticate": 'Basic realm="x"'}
            )
            return
        body = self._send_headers()
        if self.path == "/dead" or (self.path == "/stall" and self._take_stall()):
            self.wfile.write(body[:1000])
            self.wfile.flush()
            self.physionet.release.wait(timeout=60)
            return
        self.wfile.write(body)

    def _authorized(self) -> bool:
        token = base64.b64encode(f"{LOGIN}:{PASSWORD}".encode()).decode()
        return self.headers.get("Authorization") == f"Basic {token}"

    def _take_stall(self) -> bool:
        with self.physionet.lock:
            stall = self.physionet.stalls_left > 0
            self.physionet.stalls_left -= 1
        return stall

    def _send_headers(self) -> bytes:
        requested = self.headers.get("Range")
        with self.physionet.lock:
            self.physionet.ranges.append(requested)
        start, end = 0, len(PAYLOAD) - 1
        if requested:
            first, last = requested.removeprefix("bytes=").split("-")
            start, end = int(first), int(last) if last else end
        self.send_response(HTTPStatus.PARTIAL_CONTENT if requested else HTTPStatus.OK)
        if requested:
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(PAYLOAD)}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        return PAYLOAD[start : end + 1]

    def _empty(self, status: HTTPStatus, headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.send_header("Content-Length", "0")
        self.end_headers()


@pytest.fixture
def physionet() -> Iterator[FakePhysioNet]:
    """Běžící server; zaseknutá spojení po testu uvolní.

    Yields:
        Server obsluhovaný vláknem na pozadí.
    """
    server = FakePhysioNet()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.release.set()
    server.shutdown()
    server.server_close()


@pytest.fixture
def netrc(tmp_path: Path) -> Path:
    """``.netrc`` s platným heslem, čitelný jen vlastníkem.

    Returns:
        Cesta k souboru.
    """
    return _netrc(tmp_path / ".netrc", PASSWORD)


def _netrc(path: Path, password: str) -> Path:
    path.write_text(
        f"machine 127.0.0.1\nlogin {LOGIN}\npassword {password}\n", encoding="utf-8"
    )
    path.chmod(0o600)
    return path


def _download(netrc: Path, url: str, output: Path) -> subprocess.CompletedProcess[str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        ["bash", str(SCRIPT), str(netrc), url, str(output), str(output) + ".log"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, **QUICK_RETRIES},
        timeout=120,
    )


def test_downloads_file_and_keeps_attempt_log(
    physionet: FakePhysioNet, netrc: Path, tmp_path: Path
) -> None:
    """Úspěch přesune celý soubor na výstup a log pokusů vedle logu."""
    output = tmp_path / "out" / "table.csv"

    done = _download(netrc, physionet.url("data"), output)

    assert done.returncode == 0, done.stderr
    assert output.read_bytes() == PAYLOAD
    assert sorted(p.name for p in output.parent.iterdir()) == [
        "table.csv",
        "table.csv.log",
    ]
    assert "stahuji" in Path(str(output) + ".log").read_text(encoding="utf-8")


def test_splits_download_into_parallel_ranges(
    physionet: FakePhysioNet, netrc: Path, tmp_path: Path
) -> None:
    """PhysioNet škrtí rychlost na spojení, soubor se proto stahuje po kusech."""
    _download(netrc, physionet.url("data"), tmp_path / "table.csv")

    assert len([r for r in physionet.ranges if r]) > 1


def test_stalled_connection_is_replaced_and_file_completes(
    physionet: FakePhysioNet, netrc: Path, tmp_path: Path
) -> None:
    """Spojení, které přestane posílat data, se po timeoutu nahradí novým."""
    output = tmp_path / "table.csv"

    done = _download(netrc, physionet.url("stall"), output)

    assert done.returncode == 0, done.stderr
    assert output.read_bytes() == PAYLOAD


def test_dead_server_leaves_no_output_but_keeps_partial_state(
    physionet: FakePhysioNet, netrc: Path, tmp_path: Path
) -> None:
    """Po vyčerpání pokusů nevznikne useknutý výstup a rozpracovaný stav zůstane."""
    output = tmp_path / "table.csv"

    done = _download(netrc, physionet.url("dead"), output)

    assert done.returncode != 0
    assert not output.exists()
    assert (tmp_path / "table.csv.part").exists()
    assert "stažení" in done.stderr


def test_resumes_from_partial_file(
    physionet: FakePhysioNet, netrc: Path, tmp_path: Path
) -> None:
    """Rozpracovaný ``.part`` z dřívějšího běhu se nestahuje znovu od začátku.

    aria2c pošle nejdřív požadavek bez Range, ze kterého zjistí velikost, a po
    hlavičkách ho zavře; navazuje kousek před posledním staženým bajtem.
    """
    output = tmp_path / "table.csv"
    (tmp_path / "table.csv.part").write_bytes(PAYLOAD[:PARTIAL_BYTES])

    done = _download(netrc, physionet.url("data"), output)

    assert done.returncode == 0, done.stderr
    assert output.read_bytes() == PAYLOAD
    starts = [
        int(r.removeprefix("bytes=").split("-")[0]) for r in physionet.ranges if r
    ]
    assert min(starts) > PARTIAL_BYTES - RESUME_SLACK


@pytest.mark.parametrize(
    ("path", "message"),
    [("missing", "server odmítl"), ("forbidden", "server odmítl")],
)
def test_refused_download_fails_with_reason(
    physionet: FakePhysioNet, netrc: Path, tmp_path: Path, path: str, message: str
) -> None:
    """404 a 403 skončí hned s důvodem, bez opakování a bez výstupu."""
    output = tmp_path / "table.csv"

    done = _download(netrc, physionet.url(path), output)

    assert done.returncode != 0
    assert message in done.stderr
    assert not output.exists()


def test_wrong_password_fails_with_login_hint(
    physionet: FakePhysioNet, tmp_path: Path
) -> None:
    """Odmítnuté přihlášení odkáže na ``.netrc``."""
    netrc = _netrc(tmp_path / ".netrc", "wrong")

    done = _download(netrc, physionet.url("data"), tmp_path / "table.csv")

    assert done.returncode != 0
    assert "odmítl přihlášení" in done.stderr


def test_netrc_readable_by_others_is_refused(
    physionet: FakePhysioNet, netrc: Path, tmp_path: Path
) -> None:
    """Heslo čitelné pro ostatní se nepoužije."""
    netrc.chmod(0o644)

    done = _download(netrc, physionet.url("data"), tmp_path / "table.csv")

    assert done.returncode != 0
    assert "chmod 600" in done.stderr
    assert physionet.ranges == []
