"""Tests for ``download_with_retry``."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import requests

from atomistic_cookbook_utils import download_with_retry


PAYLOAD = b"hello cookbook\n" * 1000


def _make_handler(fail_first_n: int, payload: bytes = PAYLOAD):
    state = {"calls": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # silence stderr noise
            pass

        def do_GET(self):
            state["calls"] += 1
            if state["calls"] <= fail_first_n:
                self.send_response(503)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return Handler, state


@pytest.fixture
def http_server():
    servers = []

    def _start(fail_first_n: int):
        handler, state = _make_handler(fail_first_n)
        server = HTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append((server, thread))
        host, port = server.server_address
        return f"http://{host}:{port}/file.bin", state

    yield _start

    for server, thread in servers:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_download_succeeds(http_server, tmp_path):
    url, state = http_server(fail_first_n=0)
    dest = tmp_path / "file.bin"

    result = download_with_retry(url, dest)

    assert result == dest
    assert dest.read_bytes() == PAYLOAD
    assert state["calls"] == 1


def test_download_retries_then_succeeds(http_server, tmp_path):
    url, state = http_server(fail_first_n=2)
    dest = tmp_path / "file.bin"

    download_with_retry(url, dest, retries=5, backoff_factor=0.0)

    assert dest.read_bytes() == PAYLOAD
    assert state["calls"] == 3  # two 503s then a 200


def test_download_skips_if_present(http_server, tmp_path):
    url, state = http_server(fail_first_n=0)
    dest = tmp_path / "file.bin"
    dest.write_bytes(b"pre-existing")

    download_with_retry(url, dest)

    assert dest.read_bytes() == b"pre-existing"
    assert state["calls"] == 0


def test_download_overwrites_when_requested(http_server, tmp_path):
    url, _ = http_server(fail_first_n=0)
    dest = tmp_path / "file.bin"
    dest.write_bytes(b"pre-existing")

    download_with_retry(url, dest, overwrite=True)

    assert dest.read_bytes() == PAYLOAD


def test_download_creates_parent_dirs(http_server, tmp_path):
    url, _ = http_server(fail_first_n=0)
    dest = tmp_path / "nested" / "deeper" / "file.bin"

    download_with_retry(url, dest)

    assert dest.read_bytes() == PAYLOAD


def test_download_exhausts_retries(http_server, tmp_path):
    url, _ = http_server(fail_first_n=10)
    dest = tmp_path / "file.bin"

    with pytest.raises(requests.exceptions.RetryError):
        download_with_retry(url, dest, retries=2, backoff_factor=0.0)

    assert not dest.exists()
    # The .part file should also have been cleaned up.
    assert not dest.with_suffix(dest.suffix + ".part").exists()
