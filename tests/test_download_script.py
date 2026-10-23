"""Stažení souboru z PhysioNetu skriptem workflow/scripts/download.sh.

Skript běží proti lokálnímu HTTP serveru, který napodobuje chování PhysioNetu:
Basic auth z ``.netrc`` jen pro User-Agent začínající ``Wget/``, Range, spojení,
které po pár bajtech přestane posílat data, a odmítnutí 401, 403 a 404.
"""

from __future__ import annotations

import base64
import contextlib
import os
import signal
import subprocess
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

SCRIPT: Final = Path(__file__).parents[1] / "workflow" / "scripts" / "download.sh"
PAYLOAD: Final = bytes(range(256)) * 12_000
LOGIN: Final = "user"
PASSWORD: Final = "secret"
PARTIAL_BYTES: Final = 500_000
# aria2c navazuje kousek před koncem rozpracovaného souboru (z 500 000 B od 491 520).
RESUME_SLACK: Final = 16 * 1024
SLOW_PIECE: Final = 10_000
SLOW_PAUSE: Final = 0.1
STUTTER_BYTES: Final = 100_000
STUTTER_PAUSE: Final = 3.0
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
        """Odpoví podle cesty.

        ``data`` celý soubor, ``stall`` jednou zaseknuté spojení, ``dead``
        každé spojení zaseknuté, ``truncated`` zavřené před koncem těla,
        ``half`` druhá polovina zaseknutá a první pomalu, ``stutter`` každé
        spojení se po kusu dat na chvíli zasekne, ``missing`` 404 a
        ``forbidden`` 403.
        """
        if self.path == "/missing":
            self._empty(HTTPStatus.NOT_FOUND)
            return
        if self.path == "/forbidden":
            self._empty(HTTPStatus.FORBIDDEN)
            return
        if not self.headers.get("User-Agent", "").startswith("Wget/"):
            # PhysioNet (ověřeno 2026-10-23) jiným klientům než Wget přihlášení
            # nenabídne a odpoví 403 i s platným heslem.
            self._empty(HTTPStatus.FORBIDDEN)
            return
        if not self._authorized():
            self._empty(
                HTTPStatus.UNAUTHORIZED, {"WWW-Authenticate": 'Basic realm="x"'}
            )
            return
        body = self._send_headers()
        respond = {
            "/truncated": self._truncated,
            "/stall": self._stall_once,
            "/dead": self._stall,
            "/half": self._half,
            "/stutter": self._stutter,
        }.get(self.path, self.wfile.write)
        respond(body)

    def _truncated(self, body: bytes) -> None:
        self.wfile.write(body[:1000])
        self.close_connection = True

    def _stall_once(self, body: bytes) -> None:
        if self._take_stall():
            self._stall(body)
        else:
            self.wfile.write(body)

    def _stall(self, body: bytes) -> None:
        self.wfile.write(body[:1000])
        self.wfile.flush()
        self.physionet.release.wait(timeout=60)

    def _half(self, body: bytes) -> None:
        if self._start() >= len(PAYLOAD) // 2:
            self.physionet.release.wait(timeout=60)
            return
        self._slowly(body)

    def _stutter(self, body: bytes) -> None:
        self._slowly(body[:STUTTER_BYTES])
        if len(body) > STUTTER_BYTES:
            self.physionet.release.wait(timeout=STUTTER_PAUSE)

    def _slowly(self, body: bytes) -> None:
        # aria2c zaseknuté nebo přebytečné spojení zavře uprostřed těla.
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            for offset in range(0, len(body), SLOW_PIECE):
                if self.physionet.release.is_set():
                    return
                self.wfile.write(body[offset : offset + SLOW_PIECE])
                self.wfile.flush()
                time.sleep(SLOW_PAUSE)

    def _start(self) -> int:
        requested = self.headers.get("Range")
        return int(requested.removeprefix("bytes=").split("-")[0]) if requested else 0

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


def _download(
    netrc: Path,
    url: str,
    output: Path,
    timeout: str = QUICK_RETRIES["MEDIPARSE_DOWNLOAD_TIMEOUT"],
    **env: str,
) -> subprocess.CompletedProcess[str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        _command(netrc, url, output),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=_environment(timeout, env),
        timeout=120,
    )


def _command(netrc: Path, url: str, output: Path) -> list[str]:
    return ["bash", str(SCRIPT), str(netrc), url, str(output), str(output) + ".log"]


def _environment(timeout: str, env: dict[str, str]) -> dict[str, str]:
    return {
        **os.environ,
        **QUICK_RETRIES,
        "MEDIPARSE_DOWNLOAD_TIMEOUT": timeout,
        **env,
    }


def _wait_for(condition: Callable[[], bool], seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "podmínka nenastala včas"
        time.sleep(0.2)


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


def test_every_connection_stalling_once_completes_in_one_attempt(
    physionet: FakePhysioNet, netrc: Path, tmp_path: Path
) -> None:
    """Zaseknutá spojení se obnoví a stažení nepřijde o souběh ani o pokus.

    Na PhysioNetu se zasekne časem každé spojení. Kontrola minimální rychlosti
    aria2c (``--lowest-speed-limit``) je ukončí dřív než timeout a zrušená
    spojení už neobnoví, takže souběh klesá až na nulu a pokus skončí. Timeout
    delší než ta kontrola (asi 10 s) to v testu odhalí.
    """
    physionet.stalls_left = 9
    output = tmp_path / "table.csv"

    done = _download(netrc, physionet.url("stall"), output, timeout="15")

    assert done.returncode == 0, done.stderr
    assert "pokus" not in done.stderr
    assert output.read_bytes() == PAYLOAD


def test_connections_stalling_repeatedly_keep_downloading_without_attempts(
    physionet: FakePhysioNet, netrc: Path, tmp_path: Path
) -> None:
    """Spojení se za celé stažení zasekne víckrát, než dovolí ``--max-tries``.

    aria2c počítá pokusy spojení za celý jeho život, ne po sobě jdoucí chyby,
    takže na dlouhém stažení z PhysioNetu spojení jedno po druhém natrvalo
    vypadnou. Skript proto aria2c po ``MEDIPARSE_DOWNLOAD_PERIOD`` sekundách
    spustí znovu s čerstvými spojeními a nepočítá to jako neúspěšný pokus.
    """
    output = tmp_path / "table.csv"

    done = _download(
        netrc,
        physionet.url("stutter"),
        output,
        timeout="1",
        MEDIPARSE_DOWNLOAD_PERIOD="8",
    )

    assert done.returncode == 0, done.stderr
    assert "spouštím ho znovu" in done.stderr
    assert "pokus" not in done.stderr
    assert output.read_bytes() == PAYLOAD


def test_interrupt_stops_download_and_keeps_partial_state(
    physionet: FakePhysioNet, netrc: Path, tmp_path: Path
) -> None:
    """Ctrl+C ukončí skript, i když aria2c po přerušení vrátí přechodný kód.

    Když už některé spojení vyčerpalo pokusy timeoutem, aria2c po SIGINT
    skončí kódem 2, ne 7. Terminál posílá SIGINT celé skupině procesů.
    """
    output = tmp_path / "table.csv"
    attempts = tmp_path / "table.csv.log.part"
    process = subprocess.Popen(
        _command(netrc, physionet.url("half"), output),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=_environment(QUICK_RETRIES["MEDIPARSE_DOWNLOAD_TIMEOUT"], {}),
        start_new_session=True,
    )
    try:
        _wait_for(
            lambda: (
                attempts.exists()
                and "Download aborted" in attempts.read_text(encoding="utf-8")
            ),
            seconds=60,
        )
        os.killpg(process.pid, signal.SIGINT)
        _, stderr = process.communicate(timeout=15)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()

    assert process.returncode != 0
    assert "pokus" not in stderr
    assert "přerušeno" in stderr
    assert not output.exists()
    assert (tmp_path / "table.csv.part.aria2").exists()


def test_restart_period_shorter_than_exhausted_tries_is_refused(
    physionet: FakePhysioNet, netrc: Path, tmp_path: Path
) -> None:
    """Restart dřív, než mrtvý server vyčerpá pokusy, by ho nikdy neodhalil."""
    done = _download(
        netrc,
        physionet.url("data"),
        tmp_path / "table.csv",
        timeout="10",
        MEDIPARSE_DOWNLOAD_PERIOD="30",
    )

    assert done.returncode != 0
    assert "MEDIPARSE_DOWNLOAD_PERIOD" in done.stderr
    assert physionet.ranges == []


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


def test_connection_closed_early_is_retried_and_never_moved(
    physionet: FakePhysioNet, netrc: Path, tmp_path: Path
) -> None:
    """Server, který zavře spojení před koncem těla, je přechodná chyba, ne konec."""
    output = tmp_path / "table.csv"

    done = _download(netrc, physionet.url("truncated"), output)

    assert done.returncode != 0
    assert "pokus 1/2" in done.stderr
    assert not output.exists()


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
